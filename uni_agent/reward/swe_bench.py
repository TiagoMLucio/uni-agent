import json
import re
import time
import uuid
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field
from swebench.harness.constants import (
    END_TEST_OUTPUT,
    FAIL_ONLY_REPOS,
    MAP_REPO_VERSION_TO_SPECS,
    START_TEST_OUTPUT,
    EvalType,
    ResolvedStatus,
)
from swebench.harness.grading import get_eval_tests_report, get_resolution_status
from swebench.harness.log_parsers import MAP_REPO_TO_PARSER
from swebench.harness.test_spec.python import get_test_directives
from swebench.harness.utils import get_modified_files

from uni_agent.async_logging import get_logger
from uni_agent.interaction import AgentEnv
from uni_agent.reward.base import AbstractRewardSpec
from uni_agent.reward.registry import register_reward_spec
from uni_agent.utils import auto_await


# fix: https://github.com/SWE-bench/SWE-bench/issues/518
def _make_eval_script_list(instance, specs, env_name, repo_directory, base_commit, test_patch):
    """
    Same as swebench's make_eval_script_list_py, but when test_patch only adds new files,
    get_modified_files returns [] and swebench would run `git checkout base_commit` (no paths),
    which resets the whole repo (e.g. reverts tox.ini). We use no-op instead.
    """
    _HEREDOC_DELIMITER = "EOF_114329324912"
    base_commit = instance["base_commit"]
    test_files = get_modified_files(test_patch)
    if test_files:
        reset_tests_command = f"git checkout {base_commit} {' '.join(test_files)}"
    else:
        reset_tests_command = "echo 'skip reset'"

    apply_test_patch_command = f"git apply -v - <<'{_HEREDOC_DELIMITER}'\n{test_patch}\n{_HEREDOC_DELIMITER}"
    test_cmd = MAP_REPO_VERSION_TO_SPECS[instance["repo"]][instance["version"]]["test_cmd"]
    test_command = " ".join([test_cmd, *get_test_directives(instance)])

    eval_commands = [
        "source /opt/miniconda3/bin/activate",
        f"conda activate {env_name}",
        f"cd {repo_directory}",
    ]
    if "eval_commands" in specs:
        eval_commands += specs["eval_commands"]
    eval_commands += [
        f"git config --global --add safe.directory {repo_directory}",
        f"cd {repo_directory}",
        "git status",
        "git show",
        f"git -c core.fileMode=false diff {base_commit}",
        "source /opt/miniconda3/bin/activate",
        f"conda activate {env_name}",
    ]
    if "install" in specs:
        eval_commands.append(specs["install"])
    eval_commands += [
        reset_tests_command,
        apply_test_patch_command,
        f": '{START_TEST_OUTPUT}'",
        test_command,
        f": '{END_TEST_OUTPUT}'",
        reset_tests_command,
    ]
    return eval_commands


# --- Feedback templates (str.format) ------------------------------------------------
# All feedback wording lives here as config-overridable defaults, so it can be tweaked
# via config (reward.feedback_templates / feedback_item_templates / feedback_*_separator
# / feedback_join_template) without code changes. User-supplied templates are merged
# over these key-by-key in SWEBenchRewardSpec.__init__.

#: Parts rendered (in order) when ``FeedbackConfig.parts`` is not configured.
DEFAULT_FEEDBACK_PARTS = ["summary", "failing_tests", "regressions", "failure_mode"]

#: Every entry in ``FeedbackConfig.parts`` must be one of these.
SUPPORTED_FEEDBACK_PARTS = [
    "summary",
    "failing_tests",
    "newly_passing",
    "regressions",
    "failure_mode",
    "raw_output",
]

#: Source (swebench test category, outcome) for each list part.
_LIST_PART_SOURCE = {
    "failing_tests": ("FAIL_TO_PASS", "failure"),
    "newly_passing": ("FAIL_TO_PASS", "success"),
    "regressions": ("PASS_TO_PASS", "failure"),
}

#: Per-part templates. Available ``{vars}``:
#:   summary        -> resolved_status, f2p_passed, f2p_total, f2p_failed,
#:                     p2p_passed, p2p_total, p2p_failed
#:   failing_tests / newly_passing / regressions -> tests, count, shown, more
#:   failure_mode_* -> (no vars)
#:   raw_output     -> output
DEFAULT_FEEDBACK_TEMPLATES = {
    "summary": (
        "Evaluation summary: {resolved_status}. "
        "Target tests (FAIL_TO_PASS) passing: {f2p_passed}/{f2p_total}. "
        "Regressions (PASS_TO_PASS now failing): {p2p_failed}."
    ),
    "failing_tests": "Target tests still failing ({count}):\n{tests}",
    "newly_passing": "Target tests now passing ({count}):\n{tests}",
    "regressions": "Previously-passing tests your change broke ({count}):\n{tests}",
    "failure_mode_empty_patch": "No code changes were detected in the submission (empty patch).",
    "failure_mode_patch_apply_failed": "The submitted patch could not be applied to the repository.",
    "failure_mode_eval_incomplete": "The evaluation did not run to completion (environment error or timeout).",
    "failure_mode_unparseable": "The test results could not be parsed from the evaluation output.",
    "raw_output": "Test output:\n{output}",
}

#: Per-item templates for the list parts. Available ``{vars}``: test, i (1-based index).
DEFAULT_FEEDBACK_ITEM_TEMPLATES = {
    "failing_tests": "- {test}",
    "newly_passing": "- {test}",
    "regressions": "- {test}",
}

#: Wraps the joined parts. Available ``{vars}``: parts, instance_id, resolved_status.
DEFAULT_FEEDBACK_JOIN_TEMPLATE = "{parts}"
DEFAULT_FEEDBACK_SEPARATOR = "\n\n"  # between parts
DEFAULT_FEEDBACK_ITEM_SEPARATOR = "\n"  # between list items


def _test_status(result: dict) -> dict:
    return (result.get("eval_report") or {}).get("test_status") or {}


def _category(test_status: dict, category: str, outcome: str) -> list[str]:
    bucket = test_status.get(category, {}) if isinstance(test_status, dict) else {}
    return list(bucket.get(outcome, []) or [])


def _resolved_status(result: dict) -> str:
    return "resolved" if result.get("resolved") else "not resolved"


class FeedbackConfig(BaseModel):
    """Config + renderer for SWE-bench reward feedback.

    All wording is template-driven (``str.format``) and config-overridable, so feedback
    can be tweaked via config without code changes. :meth:`render` turns an eval
    ``result`` (plus the raw ``output`` and the agent ``patch``) into the feedback
    string stored under ``reward_extra_info['feedback']``. Each enabled entry in
    ``parts`` (a subset of :data:`SUPPORTED_FEEDBACK_PARTS`) is rendered, skipped when
    it has no content, joined with ``separator`` and wrapped by ``join_template``.
    ``templates`` / ``item_templates`` override the module defaults key-by-key.
    """

    enabled: bool = False
    parts: list[str] = Field(default_factory=lambda: list(DEFAULT_FEEDBACK_PARTS))
    templates: dict[str, str] = Field(default_factory=dict)
    item_templates: dict[str, str] = Field(default_factory=dict)
    join_template: str = DEFAULT_FEEDBACK_JOIN_TEMPLATE
    separator: str = DEFAULT_FEEDBACK_SEPARATOR
    item_separator: str = DEFAULT_FEEDBACK_ITEM_SEPARATOR
    max_chars: int = 4000
    max_tests: int = 25
    max_output_chars: int = 2000

    model_config = ConfigDict(extra="forbid")

    def _template(self, key: str) -> str:
        return self.templates.get(key) or DEFAULT_FEEDBACK_TEMPLATES.get(key, "")

    def _item_template(self, part: str) -> str:
        return self.item_templates.get(part) or DEFAULT_FEEDBACK_ITEM_TEMPLATES.get(part, "- {test}")

    def render(self, *, result: dict, output: str = "", patch: str | None = None, instance_id: str = "") -> str | None:
        """Render feedback from an eval result; ``None`` if no part produced content."""
        rendered: list[str] = []
        for part in self.parts:
            try:
                section = self._render_part(part, result=result, output=output, patch=patch)
            except Exception:
                section = None  # best-effort: a broken part must never break the rollout
            if section:
                rendered.append(section)
        if not rendered:
            return None
        feedback = self.join_template.format(
            parts=self.separator.join(rendered),
            instance_id=instance_id,
            resolved_status=_resolved_status(result),
        )
        if len(feedback) > self.max_chars:
            feedback = feedback[: self.max_chars] + "\n[... feedback truncated ...]"
        return feedback

    def _render_part(self, part: str, *, result: dict, output: str, patch: str | None) -> str | None:
        if part == "summary":
            return self._summary(result)
        if part in _LIST_PART_SOURCE:
            return self._test_list(part, result)
        if part == "failure_mode":
            return self._failure_mode(result, patch)
        if part == "raw_output":
            return self._raw_output(output)
        return None  # unknown part name: skip

    def _summary(self, result: dict) -> str | None:
        if not (result.get("eval_report") or {}).get("found_eval_status"):
            return None
        ts = _test_status(result)
        f2p_pass, f2p_fail = len(_category(ts, "FAIL_TO_PASS", "success")), len(_category(ts, "FAIL_TO_PASS", "failure"))
        p2p_pass, p2p_fail = len(_category(ts, "PASS_TO_PASS", "success")), len(_category(ts, "PASS_TO_PASS", "failure"))
        return self._template("summary").format(
            resolved_status=_resolved_status(result),
            f2p_passed=f2p_pass,
            f2p_total=f2p_pass + f2p_fail,
            f2p_failed=f2p_fail,
            p2p_passed=p2p_pass,
            p2p_total=p2p_pass + p2p_fail,
            p2p_failed=p2p_fail,
        )

    def _test_list(self, part: str, result: dict) -> str | None:
        category, outcome = _LIST_PART_SOURCE[part]
        tests = _category(_test_status(result), category, outcome)
        if not tests:
            return None
        shown = tests[: self.max_tests]
        item_template = self._item_template(part)
        items = [item_template.format(test=test, i=idx + 1) for idx, test in enumerate(shown)]
        return self._template(part).format(
            tests=self.item_separator.join(items),
            count=len(tests),
            shown=len(shown),
            more=max(0, len(tests) - len(shown)),
        )

    def _failure_mode(self, result: dict, patch: str | None) -> str | None:
        if patch is not None and not patch.strip():
            mode = "empty_patch"
        elif result.get("patch_apply_failed"):
            mode = "patch_apply_failed"
        elif not result.get("eval_completed"):
            mode = "eval_incomplete"
        elif not (result.get("eval_report") or {}).get("found_eval_status"):
            mode = "unparseable"
        else:
            return None
        return self._template(f"failure_mode_{mode}") or None

    def _raw_output(self, output: str) -> str | None:
        if not output or not output.strip():
            return None
        section = output
        if START_TEST_OUTPUT in output and END_TEST_OUTPUT in output:
            section = output.split(START_TEST_OUTPUT, 1)[1].split(END_TEST_OUTPUT, 1)[0]
        section = section.strip()
        if not section:
            return None
        if len(section) > self.max_output_chars:
            section = "[... output truncated ...]\n" + section[-self.max_output_chars :]
        return self._template("raw_output").format(output=section)


@register_reward_spec("swe_bench")
class SWEBenchRewardSpec(AbstractRewardSpec):
    def __init__(
        self,
        *,
        run_id: str,
        metadata: dict,
        env: AgentEnv,
        eval_timeout: int = 300,
        feedback: dict | FeedbackConfig | None = None,
        env_config: dict | None = None,
        isolate: bool = False,
    ):
        self.run_id = run_id
        self.metadata = metadata
        self.env = env
        self.logger = get_logger("reward_spec", run_id=run_id)
        self.eval_timeout = eval_timeout
        self.feedback = feedback if isinstance(feedback, FeedbackConfig) else FeedbackConfig(**(feedback or {}))
        self.env_config = env_config
        self.isolate = isolate

    @auto_await
    async def apply_gold_patch(self) -> str:
        gold_patch = self.metadata["patch"]
        await self._apply_patch(gold_patch)

    @auto_await
    async def compute_reward(self, **kwargs) -> tuple[bool, dict]:
        """Run the SWE-bench eval script and grade the result.

        By default the eval runs in the agent's own container (legacy behavior).
        With ``isolate=True`` it runs in a fresh sibling deployment built from the
        same image, so agent-corrupted state (broken conda/git/env vars, dead
        terminal) cannot affect scoring. When ``feedback.enabled`` is set, a textual
        ``reward_extra_info['feedback']`` describing the result is attached for
        downstream training code to consume.

        Returns ``(resolved, result)``.
        """
        result = {
            "eval_completed": False,
            "eval_execution_time": None,
            "eval_report": None,
            "resolved": False,
        }

        # 1. eval script (shared by the in-container and isolated paths)
        instance = self.metadata
        repo = instance["repo"]
        version = instance.get("version")
        specs = MAP_REPO_VERSION_TO_SPECS[repo][version]
        env_name = "testbed"
        repo_directory = f"/{env_name}"
        base_commit = instance["base_commit"]
        test_patch = instance["test_patch"]
        eval_script_list = _make_eval_script_list(
            instance=instance,
            specs=specs,
            env_name=env_name,
            repo_directory=repo_directory,
            base_commit=base_commit,
            test_patch=test_patch,
        )
        eval_script = "\n".join(["#!/bin/bash", "set -uxo pipefail"] + eval_script_list) + "\n"

        # Extract the agent's patch up-front when we need it: to apply it in a
        # sibling env (isolation) and/or to detect the empty-patch failure mode.
        patch: str | None = None
        if self.isolate or self.feedback.enabled:
            patch = await self._get_interaction_env_patch()

        output = ""
        eval_env = self.env
        sibling = None
        try:
            if self.isolate:
                env_config = kwargs.get("env_config") or self.env_config
                sibling = await self._start_sibling_env(env_config)
                eval_env = sibling
                try:
                    await self._apply_patch(patch or "", env=sibling)
                except Exception as e:
                    self.logger.error(f"Failed to apply patch in sibling eval env: {e}")
                    result["patch_apply_failed"] = True
                    raise

            # write eval script to the eval container
            eval_script_container = Path(f"/tmp/eval_script_{uuid.uuid4()}.sh")
            await eval_env.write_file(eval_script_container, eval_script)

            execution_t0 = time.perf_counter()

            cmd_str = f"bash {eval_script_container}"
            output = await eval_env.communicate(cmd_str, timeout=self.eval_timeout, check="ignore")

            execution_time = time.perf_counter() - execution_t0
            result["eval_completed"] = True
            result["eval_execution_time"] = execution_time

            # Remove ANSI escape codes and \r
            output = re.sub(r"\x1b\[[0-9;]*m|\r", "", output)

            eval_report = self._get_eval_report(output)
            result["eval_report"] = eval_report
            self.logger.info(f"Eval report: {eval_report}")
            result["resolved"] = eval_report["resolved"]
        except Exception as e:
            self.logger.error(f"Failed to evaluate: {e}")
        finally:
            if sibling is not None:
                try:
                    await sibling.close()
                except Exception as e:
                    self.logger.error(f"Failed to close sibling eval env: {e}")

        if self.feedback.enabled:
            feedback = self.feedback.render(
                result=result, output=output, patch=patch, instance_id=self.metadata.get("instance_id", "")
            )
            result["reward_extra_info"] = {"feedback": feedback}

        return result["resolved"], result

    @auto_await
    async def _start_sibling_env(self, env_config: dict | None):
        """Start a fresh deployment from the same env config for isolated eval."""
        if not env_config:
            raise RuntimeError(
                "isolate=True requires env_config; pass the agent-loop env config into compute_reward."
            )
        from uni_agent.interaction import AgentEnv, AgentEnvConfig

        sibling = AgentEnv(run_id=f"{self.run_id}-eval", env_config=AgentEnvConfig(**env_config))
        await sibling.start()
        self.logger.info("Started isolated sibling eval environment")
        return sibling

    @auto_await
    async def _get_interaction_env_patch(self) -> str:
        """Get the current staged diff in /testbed (interaction env state) as a patch string."""
        try:
            env_patch_file = Path(f"/tmp/patch_{uuid.uuid4()}.diff")
            await self.env.communicate(
                f"cd /testbed && git add -A && git diff --no-color --cached > {env_patch_file.as_posix()}",
                check="ignore",
            )
            patch_content = await self.env.read_file(env_patch_file)
            return patch_content
        except Exception as e:
            self.logger.error(f"Failed to get interaction environment patch: {e}")
            return ""

    @auto_await
    async def _apply_patch(self, patch: str, env=None) -> None:
        """Apply a patch string to ``env`` (default ``self.env``). Tries multiple
        apply strategies in order."""
        env = env or self.env
        if not patch or not patch.strip():
            self.logger.info("Empty patch, nothing to apply.")
            return
        patch_path = Path(f"/tmp/patch_{uuid.uuid4()}.diff")
        await env.write_file(patch_path, patch)
        commands = [
            f"cd /testbed && git apply --whitespace=fix {patch_path.as_posix()}",
            f"cd /testbed && git apply --reject --whitespace=nowarn {patch_path.as_posix()}",
            f"cd /testbed && patch --batch --fuzz=5 -p1 -i {patch_path.as_posix()}",
        ]
        last_error: Exception | None = None
        for cmd in commands:
            try:
                await env.communicate(cmd, check="raise")
                self.logger.info("Applied patch successfully!")
                return
            except RuntimeError as e:
                last_error = e
                continue
        raise RuntimeError("Failed to apply patch with any command") from last_error

    def _get_logs_eval(self, eval_output: str):
        instance = self.metadata
        repo = instance["repo"]
        log_parser = MAP_REPO_TO_PARSER[repo]
        if START_TEST_OUTPUT in eval_output and END_TEST_OUTPUT in eval_output:
            test_content = eval_output.split(START_TEST_OUTPUT)[1].split(END_TEST_OUTPUT)[0]
            status_map = log_parser(test_content, None)
            return status_map, True
        else:
            status_map = {}
            return status_map, False

    def _get_eval_report(self, eval_output: str):
        eval_report = {
            "resolved": False,
            "found_eval_status": False,
            "test_status": None,
        }

        # step 1: get logs eval
        status_map, found = self._get_logs_eval(eval_output)
        eval_report["found_eval_status"] = found
        if not found:
            return eval_report

        # step 2: get eval tests report
        eval_ref = {
            "instance_id": self.metadata["instance_id"],
            "FAIL_TO_PASS": json.loads(self.metadata.get("FAIL_TO_PASS", "[]")),
            "PASS_TO_PASS": json.loads(self.metadata.get("PASS_TO_PASS", "[]")),
        }
        repo = self.metadata["repo"]
        eval_type = EvalType.FAIL_ONLY if repo in FAIL_ONLY_REPOS else EvalType.PASS_AND_FAIL
        report = get_eval_tests_report(status_map, eval_ref, eval_type=eval_type)
        eval_report["test_status"] = report
        if get_resolution_status(report) == ResolvedStatus.FULL.value:
            eval_report["resolved"] = True
        return eval_report
