"""The collection-abort failure mode must not fire on a run that executed tests.

pytest's and pylint's own suites print ``Interrupted: N errors during collection`` as the
expected output of the tests under test, so the marker alone proves nothing.
"""

from __future__ import annotations

from swebench.harness.constants import END_TEST_OUTPUT, START_TEST_OUTPUT

from uni_agent.reward.swe_bench import FeedbackConfig

ABORT_TEXT = (
    f"{START_TEST_OUTPUT}\n"
    "collected 3 items\n"
    "testing/test_capture.py ..\n"
    "    !!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!\n"
    f"{END_TEST_OUTPUT}\n"
)


def _result(f2p_pass, f2p_fail):
    return {
        "resolved": not f2p_fail,
        "eval_completed": True,
        "patch_apply_failed": False,
        "eval_report": {
            "found_eval_status": True,
            "test_status": {
                "FAIL_TO_PASS": {"failure": f2p_fail, "success": f2p_pass},
                "PASS_TO_PASS": {"failure": [], "success": []},
            },
        },
    }


def test_abort_marker_ignored_when_tests_ran():
    cfg = FeedbackConfig(parts=["summary", "failure_mode"])
    fb = cfg.render(result=_result(["test_a"], ["test_b"]), output=ABORT_TEXT, patch="diff")
    assert "aborted during collection" not in fb
    assert "passing: 1/2" in fb


def test_abort_marker_reported_when_nothing_ran():
    cfg = FeedbackConfig(parts=["summary", "failure_mode"])
    fb = cfg.render(result=_result([], ["test_a", "test_b"]), output=ABORT_TEXT, patch="diff")
    assert "aborted during collection" in fb
