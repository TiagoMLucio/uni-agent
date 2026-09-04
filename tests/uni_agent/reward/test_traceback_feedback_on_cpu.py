"""Per-test traceback feedback: extraction from real pytest output, then budget allocation."""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
from swebench.harness.constants import END_TEST_OUTPUT, START_TEST_OUTPUT

from uni_agent.reward.swe_bench import FeedbackConfig, _diverse_order, _extract_tracebacks, _feedback_seed

MOD = "def categorize(v):\n    return sorted(set(v))[:-1]\n"

# every shape the header is ambiguous about: duplicate names across modules, a class method,
# parametrized cases (including one with a space in the id) and a setup error
PKG_A = '''import pytest

def test_same_name():
    assert 0

class TestThing:
    def test_method(self):
        assert 0

@pytest.mark.parametrize("n", ["a b", "c"])
def test_param(n):
    assert n == "zzz"

@pytest.fixture
def broken():
    raise RuntimeError("fixture blew up")

def test_setup_error(broken):
    pass
'''

PKG_B = '''def test_same_name():
    assert 0

def test_ok():
    assert 1
'''


@pytest.fixture(scope="module")
def pytest_output() -> str:
    tmp = Path(tempfile.mkdtemp(prefix="tbfeedback-"))
    try:
        (tmp / "mod.py").write_text(MOD)
        (tmp / "pkg_a").mkdir()
        (tmp / "pkg_b").mkdir()
        (tmp / "pkg_a" / "__init__.py").touch()
        (tmp / "pkg_b" / "__init__.py").touch()
        (tmp / "pkg_a" / "test_dup.py").write_text(PKG_A)
        (tmp / "pkg_b" / "test_dup.py").write_text(PKG_B)
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "--tb=long", "--color=no", "--disable-warnings", "--verbose"],
            cwd=tmp,
            capture_output=True,
            text=True,
        )
        raw = (proc.stdout + proc.stderr).replace(str(tmp) + "/", "")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return f"{START_TEST_OUTPUT}\n{raw}\n{END_TEST_OUTPUT}"


def _result(f2p_fail=(), f2p_pass=(), p2p_fail=(), p2p_pass=()):
    return {
        "resolved": False,
        "eval_completed": True,
        "patch_apply_failed": False,
        "eval_report": {
            "found_eval_status": True,
            "test_status": {
                "FAIL_TO_PASS": {"failure": list(f2p_fail), "success": list(f2p_pass)},
                "PASS_TO_PASS": {"failure": list(p2p_fail), "success": list(p2p_pass)},
            },
        },
    }


# --- extraction -------------------------------------------------------------------------

def test_duplicate_names_across_modules_are_kept_apart(pytest_output):
    blocks = _extract_tracebacks(pytest_output)
    assert "pkg_a/test_dup.py::test_same_name" in blocks
    assert "pkg_b/test_dup.py::test_same_name" in blocks


def test_class_methods_and_parametrized_ids(pytest_output):
    blocks = _extract_tracebacks(pytest_output)
    assert "pkg_a/test_dup.py::TestThing::test_method" in blocks
    # the id contains a space, so a whitespace-split parse would truncate it
    assert "pkg_a/test_dup.py::test_param[a b]" in blocks


def test_setup_errors_come_from_the_errors_section(pytest_output):
    blocks = _extract_tracebacks(pytest_output)
    assert "fixture blew up" in blocks["pkg_a/test_dup.py::test_setup_error"]


def test_passing_tests_have_no_block(pytest_output):
    assert "pkg_b/test_dup.py::test_ok" not in _extract_tracebacks(pytest_output)


def test_unparseable_output_signals_fallback():
    assert _extract_tracebacks("not pytest output at all") == {}
    assert _extract_tracebacks("") is None


# --- allocation -------------------------------------------------------------------------

def _fb(pytest_output, f2p_fail, p2p_fail, **caps):
    cfg = FeedbackConfig(
        parts=["failing_tests", "regressions", "raw_output"],
        **{"max_output_chars": 4000, "max_chars": 100_000, **caps},
    )
    return cfg.render(result=_result(f2p_fail=f2p_fail, p2p_fail=p2p_fail), output=pytest_output, patch="diff")


def test_tracebacks_replace_the_name_list(pytest_output):
    fb = _fb(pytest_output, ["pkg_a/test_dup.py::test_same_name"], [])
    assert "______ pkg_a/test_dup.py::test_same_name ______" in fb
    assert "- pkg_a/test_dup.py::test_same_name" not in fb
    assert "Test output:" not in fb  # raw_output is suppressed while blocks are available


def test_only_graded_tests_appear(pytest_output):
    fb = _fb(pytest_output, ["pkg_a/test_dup.py::test_same_name"], [])
    assert "pkg_b/test_dup.py::test_same_name" not in fb


def test_unused_share_flows_to_the_other_part(pytest_output):
    f2p = ["pkg_a/test_dup.py::test_same_name"]
    p2p = ["pkg_a/test_dup.py::TestThing::test_method", "pkg_a/test_dup.py::test_param[a b]"]
    # ratio favours F2P 3:1, but F2P has only one failure, so P2P gets both of its own
    fb = _fb(pytest_output, f2p, p2p)
    for test in f2p + p2p:
        assert f"______ {test} ______" in fb


def test_whole_blocks_only(pytest_output):
    """A budget that fits one block but not two shows one block, uncut."""
    blocks = _extract_tracebacks(pytest_output)
    first = blocks["pkg_a/test_dup.py::test_same_name"]
    fb = _fb(
        pytest_output,
        ["pkg_a/test_dup.py::test_same_name", "pkg_a/test_dup.py::TestThing::test_method"],
        [],
        max_output_chars=len(first) + 10,
    )
    assert first in fb
    assert "elided from the middle" not in fb
    assert "TestThing::test_method ______" not in fb


def test_oversized_single_traceback_is_clipped_rather_than_dropped(pytest_output):
    only = _extract_tracebacks(pytest_output)["pkg_a/test_dup.py::test_same_name"]
    fb = _fb(pytest_output, ["pkg_a/test_dup.py::test_same_name"], [], max_output_chars=len(only) // 2)
    assert "______ pkg_a/test_dup.py::test_same_name ______" in fb
    assert "elided from the middle" in fb


# --- pick order ---------------------------------------------------------------------------

def test_round_robin_covers_every_function_before_repeating():
    ids = [f"t.py::test_a[{i}]" for i in range(10)] + ["t.py::test_b", "t.py::TestC::test_c"]
    order = _diverse_order(ids, seed=0)
    # one round covers all three functions, so the two singletons land in the first three picks
    assert {"t.py::test_b", "t.py::TestC::test_c"} <= set(order[:3])
    assert sorted(order) == sorted(ids)  # nothing lost or duplicated


def test_same_seed_is_reproducible_and_different_seeds_differ():
    ids = [f"t.py::test_a[{i}]" for i in range(10)] + [f"t.py::test_b[{i}]" for i in range(10)]
    assert _diverse_order(ids, seed=7) == _diverse_order(ids, seed=7)
    assert _diverse_order(ids, seed=7) != _diverse_order(ids, seed=8)


def test_seed_comes_from_instance_and_training_step():
    cache = lambda step: {"rollout_cache": {"extra_fields": {"global_steps": step}}}  # noqa: E731
    assert _feedback_seed("inst", cache(3)) == _feedback_seed("inst", cache(3))
    assert _feedback_seed("inst", cache(3)) != _feedback_seed("inst", cache(4))
    assert _feedback_seed("inst", cache(3)) != _feedback_seed("other", cache(3))
    assert _feedback_seed("inst", None) == _feedback_seed("inst", cache(0))  # missing step -> 0


def test_parametrized_cases_do_not_crowd_out_other_functions(pytest_output):
    """The real fixture: 2 parametrized cases plus 2 standalone tests in the target bucket."""
    f2p = [n for n in _extract_tracebacks(pytest_output) if n.startswith("pkg_a")]
    cfg = FeedbackConfig(parts=["failing_tests"], max_output_chars=1200, max_chars=100_000)
    fb = cfg.render(result=_result(f2p_fail=f2p), output=pytest_output, patch="diff", seed=0)
    functions = {n.split("[")[0] for n in f2p if f"______ {n} ______" in fb}
    assert len(functions) >= 3  # not three variants of test_param


def test_falls_back_to_raw_output_when_pairing_fails():
    output = f"{START_TEST_OUTPUT}\nsomething that is not a pytest report\n{END_TEST_OUTPUT}"
    cfg = FeedbackConfig(parts=["failing_tests", "raw_output"], max_output_chars=4000)
    fb = cfg.render(result=_result(f2p_fail=["a::b"]), output=output, patch="diff")
    assert "Test output:" in fb and "not a pytest report" in fb
    assert "- a::b" in fb  # and the name list comes back too
