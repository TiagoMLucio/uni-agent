"""Unit tests for SWE-bench reward feedback (FeedbackConfig.render)."""

from __future__ import annotations

from swebench.harness.constants import END_TEST_OUTPUT, START_TEST_OUTPUT

from uni_agent.reward.swe_bench import FeedbackConfig


def _result(*, resolved=False, eval_completed=True, found=True, f2p_fail=None, f2p_pass=None, p2p_fail=None, p2p_pass=None, patch_apply_failed=False):
    if not eval_completed:
        return {"resolved": resolved, "eval_completed": False, "eval_report": None, "patch_apply_failed": patch_apply_failed}
    return {
        "resolved": resolved,
        "eval_completed": True,
        "patch_apply_failed": patch_apply_failed,
        "eval_report": {
            "found_eval_status": found,
            "test_status": {
                "FAIL_TO_PASS": {"failure": f2p_fail or [], "success": f2p_pass or []},
                "PASS_TO_PASS": {"failure": p2p_fail or [], "success": p2p_pass or []},
            },
        },
    }


# --- list parts -----------------------------------------------------------------------

def test_failing_tests_default_bullets():
    cfg = FeedbackConfig(parts=["failing_tests"])
    fb = cfg.render(result=_result(f2p_fail=["test_a", "test_b"]), patch="diff")
    assert "Target tests still failing (2):" in fb
    assert "- test_a" in fb and "- test_b" in fb


def test_failing_tests_custom_item_template():
    cfg = FeedbackConfig(parts=["failing_tests"], item_templates={"failing_tests": "  {i}. {test}"})
    fb = cfg.render(result=_result(f2p_fail=["test_a", "test_b"]), patch="diff")
    assert "  1. test_a" in fb and "  2. test_b" in fb


def test_regressions_and_newly_passing():
    cfg = FeedbackConfig(parts=["regressions", "newly_passing"])
    fb = cfg.render(result=_result(p2p_fail=["test_reg"], f2p_pass=["test_fixed"]), patch="diff")
    assert "test_reg" in fb and "test_fixed" in fb


def test_max_tests_cap_exposes_more():
    cfg = FeedbackConfig(
        parts=["failing_tests"],
        max_tests=2,
        templates={"failing_tests": "fails={count} shown={shown} more={more}\n{tests}"},
    )
    fb = cfg.render(result=_result(f2p_fail=[f"t{i}" for i in range(5)]), patch="diff")
    assert "fails=5 shown=2 more=3" in fb


# --- summary --------------------------------------------------------------------------

def test_summary_counts():
    cfg = FeedbackConfig(parts=["summary"])
    fb = cfg.render(result=_result(f2p_fail=["a", "b"], f2p_pass=["c"], p2p_fail=["x"]))
    assert "not resolved" in fb
    assert "1/3" in fb  # 1 passing of 3 FAIL_TO_PASS targets
    assert "Regressions (PASS_TO_PASS now failing): 1" in fb


# --- failure modes (separate template per mode) ---------------------------------------

def test_failure_mode_empty_patch_has_priority():
    cfg = FeedbackConfig(parts=["failure_mode"])
    fb = cfg.render(result=_result(eval_completed=False), patch="   ")
    assert "empty patch" in fb


def test_failure_mode_patch_apply_failed():
    cfg = FeedbackConfig(parts=["failure_mode"])
    fb = cfg.render(result=_result(eval_completed=True, patch_apply_failed=True), patch="diff")
    assert "could not be applied" in fb


def test_failure_mode_eval_incomplete():
    cfg = FeedbackConfig(parts=["failure_mode"])
    fb = cfg.render(result=_result(eval_completed=False), patch="diff")
    assert "did not run to completion" in fb


def test_failure_mode_unparseable():
    cfg = FeedbackConfig(parts=["failure_mode"])
    fb = cfg.render(result=_result(found=False), patch="diff")
    assert "could not be parsed" in fb


def test_no_failure_mode_when_clean():
    cfg = FeedbackConfig(parts=["failure_mode"])
    assert cfg.render(result=_result(resolved=True, f2p_pass=["a"]), patch="diff") is None


# --- raw output -----------------------------------------------------------------------

def test_raw_output_extracts_delimited_section():
    cfg = FeedbackConfig(parts=["raw_output"])
    output = f"noise before\n{START_TEST_OUTPUT}\nTRACEBACK HERE\n{END_TEST_OUTPUT}\nnoise after"
    fb = cfg.render(result=_result(f2p_fail=["a"]), output=output, patch="diff")
    assert "Test output:" in fb and "TRACEBACK HERE" in fb
    assert "noise before" not in fb and "noise after" not in fb


def test_raw_output_tail_truncation():
    cfg = FeedbackConfig(parts=["raw_output"], max_output_chars=20)
    fb = cfg.render(result=_result(f2p_fail=["a"]), output="X" * 100, patch="diff")
    assert "[... output truncated ...]" in fb
    assert fb.count("X") == 20


# --- composition ----------------------------------------------------------------------

def test_join_template_and_separator():
    cfg = FeedbackConfig(
        parts=["summary", "failing_tests"],
        separator="\n---\n",
        join_template="Feedback for {instance_id}:\n\n{parts}",
    )
    fb = cfg.render(result=_result(f2p_fail=["a"]), patch="diff", instance_id="repo__x-1")
    assert fb.startswith("Feedback for repo__x-1:")
    assert "\n---\n" in fb


def test_unknown_part_skipped_and_empty_returns_none():
    assert FeedbackConfig(parts=["bogus"]).render(result=_result(f2p_fail=["a"]), patch="diff") is None
    # nothing to report (resolved, no failing/regression parts enabled) -> None
    assert FeedbackConfig(parts=["failing_tests", "regressions"]).render(
        result=_result(resolved=True), patch="diff"
    ) is None


def test_max_chars_truncation():
    cfg = FeedbackConfig(parts=["failing_tests"], max_chars=40, max_tests=1000)
    fb = cfg.render(result=_result(f2p_fail=[f"some_long_test_name_{i}" for i in range(200)]), patch="diff")
    assert fb.endswith("[... feedback truncated ...]")


def test_templates_override_merges_over_defaults():
    cfg = FeedbackConfig(parts=["failing_tests"], templates={"failing_tests": "FAILS:\n{tests}"})
    fb = cfg.render(result=_result(f2p_fail=["a"]), patch="diff")
    assert fb == "FAILS:\n- a"
