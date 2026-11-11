"""Reward spec for SWE-rebench V2.1, graded by each task's own Harbor verifier.

A V2.1 task ships ``tests/``: ``test.sh`` restores the graded test files, applies ``test.patch``,
runs the task's test command into ``/logs/verifier/test_output.log`` and calls ``grade.py``, which
writes ``report.json`` (every FAIL_TO_PASS and PASS_TO_PASS test, passed or not) and ``reward.txt``.
The row carries those files as text under ``metadata["tests"]``; the repo lives at ``/<repo name>``.

The agent's diff is graded in a fresh sibling sandbox from the task's image: its bytes as extracted, less the
files it may not change, applied with the same ladder as :mod:`uni_agent.reward.swe_smith`, then the verifier runs
unchanged except for pytest's traceback style, which becomes ``--tb=long`` for the feedback. ``grade.py`` parses
every line of the log, so the gold check ran with it too.
"""

import asyncio
import base64
import json
import re
import shlex
import time
import uuid
from pathlib import Path, PurePosixPath

from uni_agent.async_logging import get_logger
from uni_agent.interaction import AgentEnv
from uni_agent.reward.base import (
    PATCH_EXTRACT_OK,
    AbstractRewardSpec,
    empty_patch_flag,
    patch_extract_command,
)
from uni_agent.reward.registry import register_reward_spec
from uni_agent.reward.swe_bench import FeedbackConfig, _feedback_seed, clip_eval_report
from uni_agent.tracing import (
    TRACE_FEEDBACK_CHARS,
    TRACE_PATCH_CHARS,
    TRACE_TEST_OUTPUT_CHARS,
    rollout_trace_span,
    trace_clip,
)
from uni_agent.utils import auto_await

TESTS_DIR = "/tests"
VERIFIER_DIR = "/logs/verifier"
APPLY_PATCH_FAIL = ">>>>> Patch Apply Failed"
TEST_PATCH_APPLY_FAILED = "TEST_PATCH_APPLY_FAILED"
GRADER_PYTHON_UNAVAILABLE = "GRADER_PYTHON_UNAVAILABLE"
# --forward: a hunk that reads as already applied (the rung before applied part of the patch) is skipped and fails the
# apply, where --batch alone would reverse it and report success
GIT_APPLY_CMDS = ("git apply --verbose", "git apply --verbose --reject", "patch --batch --forward --fuzz=5 -p1 -i")
#: files that decide how pytest runs or reports, which no fix changes unless the reference fix does
RUNNER_FILES = {"conftest.py", "pytest.ini", "tox.ini", "setup.cfg", "pyproject.toml"}
_TEST_FILES_RE = re.compile(r"^TEST_FILES=\((.*)\)$", re.M)
_HEADER_RE = re.compile(r"^\S+ b/(.+)$")


def graded_test_files(test_sh: str) -> list[str]:
    """The files ``test.sh`` restores before applying ``test.patch``."""
    m = _TEST_FILES_RE.search(test_sh or "")
    return shlex.split(m.group(1)) if m else []


def long_tracebacks(test_sh: str) -> str:
    """``test.sh`` with every pytest ``--tb`` style in its test command set to ``long``."""
    return re.sub(r"--tb[= ]\S+", "--tb=long", test_sh)


def _blocks(patch: str) -> list[tuple[str, bool, str]]:
    """A diff's file blocks as (path, created, text); the text of all of them joined is the diff again."""
    parts = re.split(r"(?m)^(?=diff --git )", patch)
    out = [("", False, parts[0])] if parts[0] else []
    for block in parts[1:]:
        head, _, body = block.partition("\n")
        m = _HEADER_RE.match(head[len("diff --git "):])
        out.append((m.group(1) if m else "", "\nnew file mode " in "\n" + body, block))
    return out


def withheld_files(patch: str, gold_patch: str, graded_files: list[str]) -> list[str]:
    """Files of the prediction the eval leaves out, unless the reference fix touches the same path.

    The runner files decide what pytest runs and reports: a conftest hook or an ini option can pass
    every test, and no correct fix needs one the reference fix does not. A new test-like file is a
    scratch test pytest would import, and abort on, if the test command names its directory. The
    graded test files are ``test.sh``'s to restore.
    """
    gold = {path for path, _, _ in _blocks(gold_patch or "")}
    graded = set(graded_files or ())
    out = []
    for path, created, _ in _blocks(patch or ""):
        name = PurePosixPath(path).name
        test_like = created and name.endswith(".py") and (name.startswith("test_") or name.endswith("_test.py"))
        if path and path not in gold and path not in graded and (name in RUNNER_FILES or test_like):
            out.append(path)
    return out


def without_files(patch: str, paths: list[str]) -> str:
    """The diff less the blocks of ``paths``."""
    drop = set(paths)
    return "".join(text for path, _, text in _blocks(patch) if not (path and path in drop))


def make_eval_script_list(workdir: str, patch_b64_path: str, patch_path: str, empty: bool) -> list[str]:
    """Decode the prediction, apply it, run the verifier. No ``-e``: a failed apply is reported
    through :data:`APPLY_PATCH_FAIL` and the tests do not run."""
    if empty:
        apply_lines = ["echo 'empty prediction: nothing to apply'"]
    else:
        apply_lines = [f"base64 -d {patch_b64_path} > {patch_path}", "_applied=0"]
        for cmd in GIT_APPLY_CMDS:
            apply_lines.append(f'if [ "$_applied" -eq 0 ] && {cmd} {patch_path}; then _applied=1; fi')
        apply_lines.append(f"if [ \"$_applied\" -eq 0 ]; then echo '{APPLY_PATCH_FAIL}'; exit 0; fi")
    return [
        f"cd {shlex.quote(workdir)}",
        f"git config --global --add safe.directory {shlex.quote(workdir)}",
        "git update-index -q --refresh",
        *apply_lines,
        f"mkdir -p {VERIFIER_DIR}",
        f"bash {TESTS_DIR}/test.sh",
    ]


def eval_report_from(report: dict | None) -> dict:
    """``grade.py``'s report in the shape the feedback renderers read."""
    out = {"resolved": False, "found_eval_status": False, "test_status": None}
    if not report:
        return out
    groups = {}
    for key, name in (("fail_to_pass", "FAIL_TO_PASS"), ("pass_to_pass", "PASS_TO_PASS")):
        tests = report.get(key) or {}
        groups[name] = {"success": sorted(t for t, ok in tests.items() if ok),
                        "failure": sorted(t for t, ok in tests.items() if not ok)}
    out["test_status"] = groups
    out["found_eval_status"] = any((report.get("parsed_counts") or {}).values())
    out["resolved"] = bool(report.get("resolved"))
    return out


@register_reward_spec("swe_rebench_v21")
class SWERebenchV21RewardSpec(AbstractRewardSpec):
    def __init__(
        self,
        *,
        run_id: str,
        metadata: dict,
        env: AgentEnv,
        eval_timeout: int = 1800,
        feedback: dict | FeedbackConfig | None = None,
        env_config: dict | None = None,
        isolate: bool = True,
        agent_patch_diff_args: str = "",
    ):
        self.run_id = run_id
        self.metadata = metadata
        self.env = env
        self.logger = get_logger("reward_spec", run_id=run_id)
        self.eval_timeout = eval_timeout
        self.feedback = feedback if isinstance(feedback, FeedbackConfig) else FeedbackConfig(**(feedback or {}))
        self.env_config = env_config
        self.isolate = isolate
        self.agent_patch_diff_args = agent_patch_diff_args
        self.workdir = metadata.get("workdir") or "/" + metadata["repo"].split("/")[-1]

    @auto_await
    async def apply_gold_patch(self) -> None:
        """Apply the reference fix (``solution/gold.patch``) in the agent's sandbox as V2.1's ``solve.sh`` does: a
        hunk that no longer applies (a changelog, the docs) is rejected and the rest kept, without ``.rej`` files."""
        patch_path = f"/tmp/gold_{uuid.uuid4().hex}.diff"
        await self.env.write_file(Path(patch_path), self.metadata["patch"])
        # one command: a failed check="raise" closes the sandbox, so the fallback could never run after it
        await self.env.communicate(
            f"cd {shlex.quote(self.workdir)} && {{ git apply --verbose --whitespace=nowarn {patch_path} || "
            f"{{ git apply --verbose --reject --whitespace=nowarn {patch_path}; "
            "git ls-files -z --others --exclude-standard -- '*.rej' | xargs -0 -r rm -f --; }; }", check="ignore")

    @auto_await
    async def compute_reward(self, **kwargs) -> tuple[bool, dict]:
        result = {"eval_completed": False, "eval_execution_time": None, "eval_report": None, "resolved": False}
        instance_id = self.metadata["instance_id"]
        tests = dict(self.metadata["tests"])
        tests["test.sh"] = long_tracebacks(tests["test.sh"])
        graded = graded_test_files(tests["test.sh"])

        with rollout_trace_span("patch_extract") as patch_span:
            try:
                raw = await self._extract_patch()
                patch = raw.decode("utf-8", errors="backslashreplace")
            except Exception as e:
                raw, patch = b"", None
                result["eval_error"] = f"patch harvest failed: {type(e).__name__}: {e}"
            if patch_span is not None:
                patch_span.update(output=trace_clip(patch, TRACE_PATCH_CHARS), metadata={"chars": len(patch or "")})
        drop = withheld_files(patch or "", self.metadata.get("patch") or "", graded)
        result["dropped_agent_files"] = drop
        # latin-1 maps every byte to one character, so the cut keeps every other byte as extracted
        graded_patch = without_files(raw.decode("latin-1"), drop).encode("latin-1")

        output, stdout = "", ""
        eval_env, sibling, capture_path, diagnostics = self.env, None, None, None
        try:
            if result.get("eval_error"):
                raise RuntimeError(result["eval_error"])
            with rollout_trace_span("eval_env_setup", metadata={"isolate": self.isolate}) as env_span:
                if self.isolate:
                    sibling = await self._start_sibling_env(kwargs.get("env_config") or self.env_config)
                    eval_env = sibling
                for name, text in tests.items():
                    await eval_env.write_file(Path(TESTS_DIR) / name, text)
                patch_path = f"/tmp/swerebench_pred_{uuid.uuid4().hex}.diff"
                await eval_env.write_file(Path(f"{patch_path}.b64"), base64.b64encode(graded_patch).decode())
                lines = make_eval_script_list(self.workdir, f"{patch_path}.b64", patch_path,
                                              empty=not graded_patch.strip())
                if self.feedback.enabled and self.feedback.format == "diagnostic" and "pytest" in tests["test.sh"]:
                    from uni_agent.reward.feedback_capture import install_capture

                    try:
                        capture_path, observer_lines = await install_capture(
                            eval_env, {"student_patch": patch or "", "reference_patch": self.metadata.get("patch", ""),
                                       "reference_is_bug": False, "restored_test_files": graded,
                                       "dropped_student_files": drop}, self.feedback.value_chars, root=self.workdir,
                        )
                        lines = [*observer_lines, *lines]
                    except Exception as exc:
                        diagnostics = {"events": [], "complete": False,
                                       "errors": [f"Test observer could not be installed: {type(exc).__name__}: {exc}"]}
                script = Path(f"/tmp/swerebench_eval_{uuid.uuid4().hex}.sh")
                await eval_env.write_file(script, "\n".join(["#!/bin/bash", "set -uo pipefail", *lines]) + "\n")
                if env_span is not None:
                    env_span.update(output={"status": "ready", "sibling": sibling is not None})

            with rollout_trace_span("tests", input={"test_sh": trace_clip(tests["test.sh"], 2000)}) as tests_span:
                t0 = time.perf_counter()
                stdout = await eval_env.communicate(f"bash {script}", timeout=self.eval_timeout, check="ignore")
                result["eval_execution_time"] = time.perf_counter() - t0
                stdout = re.sub(r"\x1b\[[0-9;]*m|\r", "", stdout)
                result["verifier_stdout"] = stdout[-4000:]
                if APPLY_PATCH_FAIL in stdout:
                    result["patch_apply_failed"] = True
                report = None
                if not result.get("patch_apply_failed"):
                    output = await self._read_or_empty(eval_env, f"{VERIFIER_DIR}/test_output.log")
                    output = re.sub(r"\x1b\[[0-9;]*m|\r", "", output)
                    raw = await self._read_or_empty(eval_env, f"{VERIFIER_DIR}/report.json")
                    report = json.loads(raw) if raw.strip() else None
                for marker in (TEST_PATCH_APPLY_FAILED, GRADER_PYTHON_UNAVAILABLE):
                    if marker in stdout:
                        result["eval_error"] = marker
                eval_report = eval_report_from(report)
                reward = (await self._read_or_empty(eval_env, f"{VERIFIER_DIR}/reward.txt")).strip()
                # grade.py's verdict and test.sh's reward file must agree; a disagreement is not a pass
                eval_report["resolved"] = eval_report["resolved"] and reward == "1"
                result["eval_report"] = eval_report
                result["resolved"] = eval_report["resolved"]
                result["eval_completed"] = report is not None or bool(result.get("patch_apply_failed"))
                if tests_span is not None:
                    tests_span.update(
                        output={"eval_report": clip_eval_report(eval_report),
                                "stdout": trace_clip(output or stdout, TRACE_TEST_OUTPUT_CHARS)},
                        metadata={"resolved": result["resolved"], "eval_execution_time": result["eval_execution_time"],
                                  "patch_apply_failed": result.get("patch_apply_failed", False),
                                  "stdout_chars": len(output)},
                    )
        except Exception as e:
            self.logger.error(f"Failed to evaluate {instance_id}: {e}")
            result["eval_completed"] = False
            result["eval_error"] = f"{type(e).__name__}: {e}"
        finally:
            try:
                if capture_path is not None:
                    from uni_agent.reward.feedback_capture import read_capture

                    diagnostics = await read_capture(eval_env, capture_path)
            finally:
                if sibling is not None:
                    try:
                        await sibling.close()
                    except Exception as e:
                        self.logger.error(f"Failed to close sibling eval env: {e}")

        result.update(empty_patch_flag(patch))
        extra_info: dict = {}
        if self.feedback.enabled:
            with rollout_trace_span("feedback_render") as feedback_span:
                extra_info["feedback"], result["feedback_render_failed"] = await asyncio.to_thread(
                    self.feedback.render_or_raw,
                    self.logger,
                    result=result,
                    output=output,
                    patch=patch,
                    instance_id=instance_id,
                    seed=_feedback_seed(instance_id, kwargs.get("interaction_result")),
                    diagnostics=diagnostics,
                )
                if feedback_span is not None:
                    feedback_span.update(output=trace_clip(extra_info["feedback"], TRACE_FEEDBACK_CHARS))
        if self.agent_patch_diff_args:
            try:
                extra_info["agent_patch"] = await self._get_interaction_env_patch(self.agent_patch_diff_args)
            except Exception as e:
                self.logger.error(f"Failed to extract the agent patch for the reflector: {e}")
                extra_info["agent_patch"] = ""
        if extra_info:
            result["reward_extra_info"] = extra_info
        result["patch"] = patch
        return result["resolved"], result

    async def _read_or_empty(self, env, path: str) -> str:
        try:
            return await env.read_file(path, errors="backslashreplace")
        except Exception:
            return ""

    @auto_await
    async def _start_sibling_env(self, env_config: dict | None):
        """A fresh sandbox from the task's image, without the agent-side setup commands."""
        if not env_config:
            raise RuntimeError("isolate=True requires env_config; pass the agent-loop env config into compute_reward.")
        from uni_agent.interaction import AgentEnv, AgentEnvConfig

        sibling = AgentEnv(
            run_id=f"{self.run_id}-eval",
            env_config=AgentEnvConfig(**{**env_config, "post_setup_cmd": None, "privileged_setup_cmd": None}),
        )
        try:
            await sibling.start()
        except BaseException:
            await sibling.close()
            raise
        return sibling

    @auto_await
    async def _extract_patch(self, diff_args: str = "") -> bytes:
        """The agent's staged diff in its workdir as git wrote it, bytes and line endings alike (a side session, like
        swe_smith's)."""
        patch_file = f"/tmp/patch_{uuid.uuid4()}.diff"
        output = await self.env.communicate_isolated(patch_extract_command(patch_file, diff_args, self.workdir))
        if PATCH_EXTRACT_OK not in output:
            raise RuntimeError(f"the extraction never reached the diff: {output.strip()[-500:]}")
        return base64.b64decode(await self.env.read_file(f"{patch_file}.b64"))

    @auto_await
    async def _get_interaction_env_patch(self, diff_args: str = "") -> str:
        return (await self._extract_patch(diff_args)).decode("utf-8", errors="backslashreplace")
