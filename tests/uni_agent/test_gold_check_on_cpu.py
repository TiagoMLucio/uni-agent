"""The gold-patch check grades each row the way a rollout is graded.

Its whole value is being the same path: the row's ``tools_kwargs`` over the agent yaml, the row's
own sandbox setup, the gold patch where the agent edits, and ``compute_reward`` with the env
config the loop passes. A task its own gold patch does not solve is a silent zero in every run.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_gold_patches.py"


@pytest.fixture
def gold_check(monkeypatch):
    spec = importlib.util.spec_from_file_location("check_gold_patches", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(iid, **reward):
    return {"data_source": "swe_smith", "extra_info": {"tools_kwargs": {
        "env": {"deployment": {"image": f"img/{iid}"}},
        "reward": {"metadata": {"instance_id": iid, **reward}},
    }}}


BASE = {
    "env": {"deployment": {"type": "local", "image": "base"}, "privileged_setup_cmd": "flatten"},
    "reward": {"name": "swe_smith", "isolate": True, "feedback": {"enabled": True}},
    "setup_timeout": 5, "setup_retries": 1,
    "validation_overrides": {"reward": {"eval_timeout": 900}},
}


def _stub(module, monkeypatch, ledger, outcome, start_failures=0):
    class _Env:
        starts = 0

        def __init__(self, run_id, env_config):
            self.config = env_config

        async def start(self):
            _Env.starts += 1
            ledger.append(("start", self.config["deployment"]["image"], self.config["privileged_setup_cmd"]))
            if _Env.starts <= start_failures:
                raise RuntimeError("sandbox did not come up")

        async def close(self):
            ledger.append(("close",))

    class _Reward:
        def __init__(self, config):
            self.config = config

        async def apply_gold_patch(self):
            ledger.append(("gold", self.config["metadata"]["instance_id"]))

        async def compute_reward(self, **kwargs):
            ledger.append(("reward", kwargs["env_config"]["deployment"]["image"], self.config.get("eval_timeout")))
            result = outcome(self.config["metadata"]["instance_id"])
            return result.get("resolved", False), result

    monkeypatch.setattr(module, "AgentEnv", _Env)
    monkeypatch.setattr(module, "AgentEnvConfig", lambda **kw: kw)
    monkeypatch.setattr(module, "load_reward_spec", _Reward)


def _check(module, rows, tmp_path, validate=False):
    out = tmp_path / "gold.jsonl"
    records = asyncio.run(module.check_rows(rows, BASE, validate=validate, concurrency=2, out=out))
    return {r["instance_id"]: r for r in records}, [json.loads(line) for line in out.read_text().splitlines()]


def test_each_row_runs_the_rollout_path_and_is_classified(gold_check, monkeypatch, tmp_path):
    ledger = []
    outcomes = {
        "ok": {"resolved": True, "eval_completed": True},
        "broken": {"resolved": False, "eval_completed": True,
                   "reward_extra_info": {"feedback": "FAILED tests/test_a.py::test_x"}},
        "infra": {"resolved": False, "eval_completed": False, "eval_error": "sibling did not start"},
    }
    _stub(gold_check, monkeypatch, ledger, lambda iid: outcomes[iid])
    records, written = _check(gold_check, [_row(i) for i in outcomes], tmp_path)

    assert {i: r["status"] for i, r in records.items()} == {"ok": "solved", "broken": "unsolved", "infra": "error"}
    assert records["broken"]["feedback"] == "FAILED tests/test_a.py::test_x", "why the gold patch fails"
    assert len(written) == 3, "every row is on disk as it finishes"
    # the row's own image over the yaml's, the privileged setup kept, the gold patch before the grade
    for iid in outcomes:
        assert ("start", f"img/{iid}", "flatten") in ledger
        assert ledger.index(("gold", iid)) < ledger.index(("reward", f"img/{iid}", None))
    assert ledger.count(("close",)) == 3
    assert "unsolved: broken" in gold_check.summary(list(records.values()))


def test_validate_applies_the_validation_overrides(gold_check, monkeypatch, tmp_path):
    ledger = []
    _stub(gold_check, monkeypatch, ledger, lambda _iid: {"resolved": True, "eval_completed": True})
    _check(gold_check, [_row("v")], tmp_path, validate=True)
    assert ("reward", "img/v", 900) in ledger


def test_setup_is_retried_like_a_rollout_and_a_failure_is_an_error_row(gold_check, monkeypatch, tmp_path):
    ledger = []
    _stub(gold_check, monkeypatch, ledger, lambda _iid: {"resolved": True, "eval_completed": True},
          start_failures=1)
    records, _ = _check(gold_check, [_row("flaky")], tmp_path)
    assert records["flaky"]["status"] == "solved"
    assert [e[0] for e in ledger] == ["start", "close", "start", "gold", "reward", "close"]

    ledger.clear()
    _stub(gold_check, monkeypatch, ledger, lambda _iid: {}, start_failures=9)
    records, _ = _check(gold_check, [_row("dead")], tmp_path)
    assert records["dead"]["status"] == "error"
    assert "sandbox did not come up" in records["dead"]["error"]
    assert [e[0] for e in ledger] == ["start", "close", "start", "close"]


def test_the_cli_reads_a_rendered_config_and_a_parquet(gold_check, monkeypatch, tmp_path, capsys):
    import datasets
    import yaml

    ledger = []
    _stub(gold_check, monkeypatch, ledger, lambda iid: {"resolved": iid == "a", "eval_completed": True})
    data = tmp_path / "rows.parquet"
    datasets.Dataset.from_list([_row("a"), _row("b"), _row("c")]).to_parquet(str(data))
    config = tmp_path / "agent.yaml"
    config.write_text(yaml.safe_dump([BASE]))
    out = tmp_path / "out" / "gold.jsonl"

    assert gold_check.main(["--agent-config", str(config), "--data", str(data), "--out", str(out),
                            "--instances", "a,b"]) == 0
    assert sorted(json.loads(line)["instance_id"] for line in out.read_text().splitlines()) == ["a", "b"]
    assert "unsolved: b" in capsys.readouterr().out
