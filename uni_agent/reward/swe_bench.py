import json
import random
import re
import time
import uuid
import zlib
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
from uni_agent.tracing import rollout_trace_span
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
    "failure_mode_collection_abort": (
        "No test ran: pytest aborted during collection because a file failed to import -- {detail}. "
        "Pass/fail counts below are not meaningful; fix the import first."
    ),
    "raw_output": "Test output:\n{output}",
    # one rendered failure inside a traceback part. Available ``{vars}``: test, traceback
    "traceback": "______ {test} ______\n{traceback}",
}

#: Per-item templates for the *name* list parts. Available ``{vars}``: test, i (1-based index).
DEFAULT_FEEDBACK_ITEM_TEMPLATES = {
    "failing_tests": "- {test}",
    "newly_passing": "- {test}",
    "regressions": "- {test}",
}

#: Wraps the joined parts. Available ``{vars}``: parts, instance_id, resolved_status.
DEFAULT_FEEDBACK_JOIN_TEMPLATE = "{parts}"
DEFAULT_FEEDBACK_SEPARATOR = "\n\n"  # between parts
DEFAULT_FEEDBACK_ITEM_SEPARATOR = "\n"  # between list items


#: pytest aborts the whole session when a module fails to import at collection time, so no
#: test runs at all. Every listed test is then absent from the status map and grades as a
#: failure, which reads as "your change broke 1789 tests" when the truth is one bad file.
_COLLECTION_ABORT_RE = re.compile(r"Interrupted:\s*\d+\s*error[s]?\s+during collection")
_COLLECTION_ERROR_RE = re.compile(r"^ERROR\s+(\S+)\s+-\s+(.+)$", re.M)


def _collection_abort(output: str) -> str | None:
    """The collection errors that aborted the run, or ``None`` if it ran normally."""
    if not output or not _COLLECTION_ABORT_RE.search(output):
        return None
    hits = _COLLECTION_ERROR_RE.findall(output)
    if not hits:
        return "a module failed to import during collection"
    return "; ".join(f"{f} ({r.strip()})" for f, r in hits[:3])


#: A ``--tb=long`` run writes one block per failure under ``=== FAILURES ===`` (and per setup
#: error under ``=== ERRORS ===``), each headed by ``____ <name> ____``. That header carries the
#: test *name* only, so two same-named tests in different modules are indistinguishable; blocks
#: are therefore matched to node ids by run order, which is deterministic (no profile uses xdist).
_TB_HEADER_RE = re.compile(r"^_{4,} (.+?) _{4,}$", re.M)
_TB_STOP_RE = re.compile(r"^=+ .+ =+\s*$", re.M)
#: `--verbose` progress line. Ids may contain spaces (parametrized cases), so the group is lazy
#: and the match is anchored on the trailing percent column, which the short-summary lines
#: below (`FAILED <id> - <message>`) do not have.
_PROGRESS_RE = re.compile(
    r"^(\S.*?::.+?)[ \t]+(PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS)\b[^\n]*\[\s*\d+%\][ \t]*$", re.M
)
#: `-rA`/`-rfE` short-summary line, the only ordered source for profiles that omit --verbose
_SUMMARY_LINE_RE = re.compile(r"^(FAILED|ERROR) (.+?)(?: - .*)?$", re.M)


def _clip_middle(text: str, cap: int) -> str:
    """Keep both ends: the head carries the failing line, the tail the exception and location."""
    if len(text) <= cap:
        return text
    head = cap // 2
    return f"{text[:head]}\n[... {len(text) - cap} chars elided from the middle ...]\n{text[-(cap - head) :]}"


def _section_blocks(output: str, banner: str) -> list[str]:
    """The traceback bodies under ``=== <banner> ===``, in run order."""
    match = re.search(rf"^=+ {banner} =+\s*$", output, re.M)
    if not match:
        return []
    rest = output[match.end() :]
    stop = _TB_STOP_RE.search(rest)
    if stop:
        rest = rest[: stop.start()]
    parts = _TB_HEADER_RE.split(rest)
    return [parts[i + 1].strip() for i in range(1, len(parts) - 1, 2)]


def _ordered_ids(output: str, status: str) -> list[str]:
    ids = [nid for nid, outcome in _PROGRESS_RE.findall(output) if outcome == status]
    if ids:
        return ids
    return [nid for outcome, nid in _SUMMARY_LINE_RE.findall(output) if outcome == status]


def _extract_tracebacks(output: str) -> dict[str, str] | None:
    """``{node_id: traceback}`` in run order, or ``None`` when the pairing is not trustworthy.

    ``None`` means "fall back to the raw dump": a count mismatch is the only signal we have
    that the output does not have the shape this pairing assumes.
    """
    if not output:
        return None
    section = output
    if START_TEST_OUTPUT in output and END_TEST_OUTPUT in output:
        section = output.split(START_TEST_OUTPUT, 1)[1].split(END_TEST_OUTPUT, 1)[0]
    blocks: dict[str, str] = {}
    for banner, status in (("ERRORS", "ERROR"), ("FAILURES", "FAILED")):
        bodies = _section_blocks(section, banner)
        ids = _ordered_ids(section, status)
        if len(bodies) != len(ids):
            return None
        blocks.update(zip(ids, bodies, strict=True))
    return blocks


def _test_status(result: dict) -> dict:
    return (result.get("eval_report") or {}).get("test_status") or {}


def _category(test_status: dict, category: str, outcome: str) -> list[str]:
    bucket = test_status.get(category, {}) if isinstance(test_status, dict) else {}
    return list(bucket.get(outcome, []) or [])


#: parametrized suffix; `test_f[0]` and `test_f[1]` are the same test exercising one code path
_PARAM_SUFFIX_RE = re.compile(r"\[.*\]$")


def _feedback_seed(instance_id: str, interaction_result: dict | None) -> int:
    """Per (task, training step): a task re-sampled at a later step surfaces different failures,
    but any single (task, step) renders identically, so a run stays reproducible."""
    extra = ((interaction_result or {}).get("rollout_cache") or {}).get("extra_fields") or {}
    return zlib.crc32(f"{instance_id}:{extra.get('global_steps') or 0}".encode())


def _diverse_order(node_ids: list[str], seed: int) -> list[str]:
    """Round-robin over test functions, so every distinct function is offered a slot before any
    function gets a second one.

    Collection order groups a function's parametrized cases together and they almost always fail
    the same way, so taking the first N spends the budget on near-duplicates: across the corpus
    the median task has 6 failing tests over 4 distinct functions. Functions, and the cases
    within them, are shuffled from ``seed`` so a task seen again later shows different cases.
    """
    groups: dict[str, list[str]] = {}
    for node_id in node_ids:
        groups.setdefault(_PARAM_SUFFIX_RE.sub("", node_id), []).append(node_id)
    if not groups:
        return []
    rng = random.Random(seed)
    keys = list(groups)
    rng.shuffle(keys)
    for key in keys:
        rng.shuffle(groups[key])
    return [groups[key][i] for i in range(max(map(len, groups.values()))) for key in keys if i < len(groups[key])]


def _failing_ids(result: dict) -> set[str]:
    status = _test_status(result)
    return {t for cat, out in _LIST_PART_SOURCE.values() if out == "failure" for t in _category(status, cat, out)}


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
    #: cap on the *name* lists (``newly_passing``, and the traceback parts when extraction fails)
    max_names: int = 25
    #: total chars for every traceback shown, across all parts
    max_output_chars: int = 2000
    #: how ``max_output_chars`` is shared between the traceback parts. A part that cannot fill
    #: its share donates the rest, so the whole budget is used whenever there is content for it.
    #: Weight 0 (or absence) renders that part as a name list instead.
    traceback_ratio: dict[str, int] = Field(default_factory=lambda: {"failing_tests": 3, "regressions": 1})

    model_config = ConfigDict(extra="forbid")

    def _template(self, key: str) -> str:
        return self.templates.get(key) or DEFAULT_FEEDBACK_TEMPLATES.get(key, "")

    def _item_template(self, part: str) -> str:
        return self.item_templates.get(part) or DEFAULT_FEEDBACK_ITEM_TEMPLATES.get(part, "- {test}")

    def render(
        self, *, result: dict, output: str = "", patch: str | None = None, instance_id: str = "", seed: int = 0
    ) -> str | None:
        """Render feedback from an eval result; ``None`` if no part produced content."""
        rendered: list[str] = []
        abort = _collection_abort(output)
        # the budget is shared across parts, so the split is decided once, before rendering
        blocks = None if abort else _extract_tracebacks(output)
        failing = _failing_ids(result)
        # the report says tests failed but the output carries no block for any of them: the
        # pairing is not wrong, the output simply is not shaped like we assume -> raw dump
        if blocks is not None and failing and not (failing & set(blocks)):
            blocks = None
        shown = self._allocate_tracebacks(result, blocks, seed) if blocks else {}
        for part in self.parts:
            try:
                section = self._render_part(
                    part, result=result, output=output, patch=patch, abort=abort, shown=shown, blocks=blocks
                )
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

    def _render_part(
        self,
        part: str,
        *,
        result: dict,
        output: str,
        patch: str | None,
        abort: str | None = None,
        shown: dict[str, list[tuple[str, str]]] | None = None,
        blocks: dict[str, str] | None = None,
    ) -> str | None:
        shown = shown or {}
        # nothing ran, so every count and list below is an artefact of the abort
        if abort and (part == "summary" or part in _LIST_PART_SOURCE):
            return None
        if part == "summary":
            return self._summary(result)
        if part in _LIST_PART_SOURCE:
            if part in shown:
                return self._traceback_list(part, result, shown[part])
            # extraction worked but this part got no budget: the summary already carries its count
            if blocks is not None and self.traceback_ratio.get(part):
                return None
            return self._test_list(part, result)
        if part == "failure_mode":
            return self._failure_mode(result, patch, abort)
        if part == "raw_output":
            # fallback only: with per-test blocks available the model already has the tracebacks
            return None if blocks is not None else self._raw_output(output)
        return None  # unknown part name: skip

    def _allocate_tracebacks(
        self, result: dict, blocks: dict[str, str], seed: int = 0
    ) -> dict[str, list[tuple[str, str]]]:
        """Split ``max_output_chars`` across the traceback parts. Whole blocks only.

        Each part first fills its ``traceback_ratio`` share, then every char left over anywhere
        is offered to the parts in order, so an empty section never wastes budget. The single
        exception to "whole blocks only" is a traceback larger than the entire budget: it is
        clipped, because showing nothing at all would leave the rollout with no failure detail.
        """
        ratio = {p: w for p, w in self.traceback_ratio.items() if w > 0 and p in _LIST_PART_SOURCE}
        if not ratio:
            return {}
        status = _test_status(result)
        pools = {
            p: _diverse_order([n for n in blocks if n in set(_category(status, *_LIST_PART_SOURCE[p]))], seed)
            for p in ratio
        }
        budget, total_weight = self.max_output_chars, sum(ratio.values())
        taken: dict[str, list[str]] = {p: [] for p in ratio}
        used = 0
        for part, weight in ratio.items():
            share = budget * weight // total_weight
            for node_id in pools[part]:
                if len(blocks[node_id]) > share:
                    break
                share -= len(blocks[node_id])
                used += len(blocks[node_id])
                taken[part].append(node_id)
        for part in ratio:
            for node_id in pools[part][len(taken[part]) :]:
                if len(blocks[node_id]) > budget - used:
                    break
                used += len(blocks[node_id])
                taken[part].append(node_id)
        if not used:
            for part in ratio:
                if pools[part]:
                    node_id = pools[part][0]
                    return {part: [(node_id, _clip_middle(blocks[node_id], budget))]}
        return {p: [(n, blocks[n]) for n in ids] for p, ids in taken.items() if ids}

    def _traceback_list(self, part: str, result: dict, picked: list[tuple[str, str]]) -> str | None:
        template = self._template("traceback")
        items = [template.format(test=node_id, traceback=body) for node_id, body in picked]
        total = len(_category(_test_status(result), *_LIST_PART_SOURCE[part]))
        return self._template(part).format(
            tests=self.item_separator.join(items),
            count=total,
            shown=len(picked),
            more=max(0, total - len(picked)),
        )

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
        shown = tests[: self.max_names]
        item_template = self._item_template(part)
        items = [item_template.format(test=test, i=idx + 1) for idx, test in enumerate(shown)]
        return self._template(part).format(
            tests=self.item_separator.join(items),
            count=len(tests),
            shown=len(shown),
            more=max(0, len(tests) - len(shown)),
        )

    def _failure_mode(self, result: dict, patch: str | None, abort: str | None = None) -> str | None:
        if abort:
            return self._template("failure_mode_collection_abort").format(detail=abort) or None
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
        # keep both ends: the head carries the failing assertions and their tracebacks,
        # the tail the summary line, and a blind tail slice drops the first failures
        return self._template("raw_output").format(output=_clip_middle(section, self.max_output_chars))


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
        # extra git-diff flags for the reflector's copy of the patch ('-U10', '-W');
        # empty means do not capture one
        self.agent_patch_diff_args = agent_patch_diff_args

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
            with rollout_trace_span("patch_extract"):
                patch = await self._get_interaction_env_patch()

        output = ""
        eval_env = self.env
        sibling = None
        try:
            with rollout_trace_span("eval_env_setup", metadata={"isolate": self.isolate}) as env_span:
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
                if env_span is not None:
                    env_span.update(output={"status": "ready", "sibling": sibling is not None})

            with rollout_trace_span("tests") as tests_span:
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
                if tests_span is not None:
                    tests_span.update(output={"resolved": result["resolved"], "eval_execution_time": execution_time})
        except Exception as e:
            self.logger.error(f"Failed to evaluate: {e}")
        finally:
            if sibling is not None:
                try:
                    await sibling.close()
                except Exception as e:
                    self.logger.error(f"Failed to close sibling eval env: {e}")

        extra_info: dict = {}
        if self.feedback.enabled:
            instance_id = self.metadata.get("instance_id", "")
            with rollout_trace_span("feedback_render"):
                extra_info["feedback"] = self.feedback.render(
                    result=result,
                    output=output,
                    patch=patch,
                    instance_id=instance_id,
                    seed=_feedback_seed(instance_id, kwargs.get("interaction_result")),
                )
        if self.agent_patch_diff_args:
            # a wider re-render of the same prediction, for the reflector only
            with rollout_trace_span("patch_extract", metadata={"diff_args": self.agent_patch_diff_args}):
                extra_info["agent_patch"] = await self._get_interaction_env_patch(self.agent_patch_diff_args)
        if extra_info:
            result["reward_extra_info"] = extra_info

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
    async def _get_interaction_env_patch(self, diff_args: str = "") -> str:
        """Get the current staged diff in /testbed (interaction env state) as a patch string.

        ``diff_args`` are extra git-diff flags (``-U10``, ``-W``, ...) for the reflector's copy;
        the prediction that is graded is always taken with none, so the eval never depends on
        them. The attributes file is what makes ``-W`` find Python function boundaries: without
        it git falls back to a generic heuristic that expands every hunk to the whole file.
        """
        try:
            env_patch_file = Path(f"/tmp/patch_{uuid.uuid4()}.diff")
            attrs = "/tmp/.uniagent_gitattributes"
            # side session: the agent's own session may still be running whatever it
            # left attached, which would swallow this command until the timeout
            await self.env.communicate_isolated(
                f"cd /testbed && printf '*.py diff=python\\n' > {attrs} && git add -A && "
                f"git -c core.attributesFile={attrs} diff --no-color {diff_args} --cached "
                f"> {env_patch_file.as_posix()}",
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
