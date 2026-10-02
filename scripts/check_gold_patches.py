"""Run a dataset's gold patches through the training reward path and report the tasks they do not solve.

Each row gets what a rollout gets: the agent yaml with the row's ``tools_kwargs`` merged over it
(and ``validation_overrides`` with ``--validate``), a sandbox started with the row's setup, the
privileged one included, and the row's reward spec. The gold patch is applied where the agent
would have edited, and ``compute_reward`` grades it exactly as it grades a rollout, sibling
container and feedback included. A task its own gold patch does not solve scores 0 whatever the
policy does, so it is a silent zero in every run that trains or validates on it.

    python scripts/check_gold_patches.py --agent-config rendered_agent.yaml --data val.parquet \\
        --out gold_check.jsonl [--validate] [--concurrency 16] [--limit N] [--instances id,id]

Every row is appended to ``--out`` as it finishes, so an interrupted run keeps what it did.
Statuses: ``solved``; ``unsolved`` (the eval ran and the gold patch failed: the task is broken on
this path); ``error`` (setup, patch or eval did not complete: infrastructure, rerun it).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import uuid
from pathlib import Path

import yaml

from uni_agent.agent_loop import row_config
from uni_agent.async_logging import add_file_handler, cleanup_handlers
from uni_agent.interaction import AgentEnv, AgentEnvConfig
from uni_agent.reward import load_reward_spec


def instance_id(row: dict) -> str:
    tools_kwargs = (row.get("extra_info") or {}).get("tools_kwargs") or {}
    return ((tools_kwargs.get("reward") or {}).get("metadata") or {}).get("instance_id") or ""


async def _start(config: dict, run_id: str) -> AgentEnv:
    """The rollout's own setup budget: ``setup_timeout`` per attempt, a fresh sandbox per retry."""
    timeout, retries = config.get("setup_timeout", 300), config.get("setup_retries", 2)
    for attempt in range(retries + 1):
        env = AgentEnv(run_id=run_id, env_config=AgentEnvConfig(**config["env"]))
        try:
            async with asyncio.timeout(timeout):
                await env.start()
        except Exception:
            await env.close()
            if attempt < retries:
                continue
            raise
        return env


async def check_row(row: dict, base_config: dict, validate: bool, log_dir: Path | None) -> dict:
    """One row through setup, the gold patch and the reward; never raises."""
    iid = instance_id(row)
    run_id = f"gold-{uuid.uuid4().hex[:12]}"
    record: dict = {"instance_id": iid, "run_id": run_id, "data_source": row.get("data_source")}
    if log_dir is not None:
        add_file_handler(log_dir / run_id / "run.log", run_id)
    t0 = time.perf_counter()
    env = None
    try:
        config = row_config(base_config, (row.get("extra_info") or {}).get("tools_kwargs"), validate=validate)
        if not config.get("reward"):
            raise ValueError("the row's config has no reward block")
        record["image"] = ((config.get("env") or {}).get("deployment") or {}).get("image")
        env = await _start(config, run_id)
        reward_spec = load_reward_spec({**config["reward"], "run_id": run_id, "env": env})
        await reward_spec.apply_gold_patch()
        _, result = await reward_spec.compute_reward(interaction_result={}, env_config=config["env"])
        result = result if isinstance(result, dict) else {}
        record.update({
            key: result.get(key)
            for key in ("resolved", "eval_completed", "patch_apply_failed", "empty_patch", "eval_error",
                        "eval_execution_time")
        })
        record["feedback"] = (result.get("reward_extra_info") or {}).get("feedback")
        if result.get("resolved"):
            record["status"] = "solved"
        elif result.get("eval_completed"):
            record["status"] = "unsolved"
        else:
            record["status"] = "error"
    except Exception as e:
        record.update(status="error", error=f"{type(e).__name__}: {e}")
    finally:
        if env is not None:
            await env.close()
        record["wall_s"] = round(time.perf_counter() - t0, 1)
        if log_dir is not None:
            await asyncio.to_thread(cleanup_handlers, run_id)
    return record


async def check_rows(rows: list[dict], base_config: dict, *, validate: bool, concurrency: int,
                     out: Path, log_dir: Path | None = None) -> list[dict]:
    semaphore = asyncio.Semaphore(concurrency)
    records: list[dict] = []

    async def one(row: dict) -> None:
        async with semaphore:
            record = await check_row(row, base_config, validate, log_dir)
        records.append(record)
        with out.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
        done = len(records)
        print(f"[{done}/{len(rows)}] {record['status']:>8}  {record['instance_id']}", flush=True)

    await asyncio.gather(*(one(row) for row in rows))
    return records


def summary(records: list[dict]) -> str:
    by_status: dict[str, list[str]] = {"solved": [], "unsolved": [], "error": []}
    for record in records:
        by_status[record["status"]].append(record["instance_id"])
    lines = [f"{status:>8}  {len(ids)}" for status, ids in by_status.items()]
    for status in ("unsolved", "error"):
        if by_status[status]:
            lines.append(f"{status}: " + " ".join(sorted(by_status[status])))
    return "\n".join(lines)


def load_agent_config(path: Path) -> dict:
    """A rendered agent yaml: the one-entry list verl's agent loop registry reads, or the entry itself."""
    loaded = yaml.safe_load(path.read_text())
    return loaded[0] if isinstance(loaded, list) else loaded


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--agent-config", type=Path, required=True, help="the agent yaml a run is launched with")
    parser.add_argument("--data", type=Path, required=True, help="a prepared parquet (extra_info.tools_kwargs per row)")
    parser.add_argument("--out", type=Path, required=True, help="JSONL, appended per row")
    parser.add_argument("--validate", action="store_true", help="apply validation_overrides, as a val rollout does")
    parser.add_argument("--concurrency", type=int, default=16, help="rows in flight (each may hold two sandboxes)")
    parser.add_argument("--limit", type=int, default=None, help="only the first N rows")
    parser.add_argument("--instances", default="", help="comma-separated instance ids to check")
    parser.add_argument("--log-dir", type=Path, default=None, help="per-row run.log, as a rollout writes")
    args = parser.parse_args(argv)

    import datasets

    # read the way the trainer's dataset reads it, so a row's config merges as it does in a run
    rows = datasets.load_dataset("parquet", data_files=str(args.data))["train"].to_list()
    if args.instances:
        wanted = {i.strip() for i in args.instances.split(",") if i.strip()}
        rows = [row for row in rows if instance_id(row) in wanted]
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        print("no rows selected", file=sys.stderr)
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    records = asyncio.run(check_rows(
        rows, load_agent_config(args.agent_config), validate=args.validate,
        concurrency=args.concurrency, out=args.out, log_dir=args.log_dir,
    ))
    print(summary(records))
    return 0


if __name__ == "__main__":
    sys.exit(main())
