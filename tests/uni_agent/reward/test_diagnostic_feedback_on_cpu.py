"""Information-preservation checks on real reports and known lossy edge cases."""

import difflib
import os
import re
import subprocess
import sys
import types
from pathlib import Path

import pytest

from uni_agent.reward.diagnostic_feedback import (
    END,
    OMISSIONS,
    START,
    _clean_address,
    _map_line,
    _matches_repr,
    _patch_hunks,
    changed_in_final,
    changed_lines,
    fit_message_feedback,
    fold_assertion,
    from_junit,
    load_capture,
    render_diagnostic,
    render_event,
    shrink_feedback,
    signature,
    source_context,
)


def result(fail=(), passed=(), regressions=(), complete=True):
    return {
        "eval_completed": complete,
        "resolved": False,
        "eval_report": {
            "found_eval_status": True,
            "test_status": {
                "FAIL_TO_PASS": {"failure": list(fail), "success": list(passed)},
                "PASS_TO_PASS": {"failure": list(regressions), "success": []},
            },
        },
    }


@pytest.fixture(scope="module")
def captured(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("diagnostic")
    (tmp / "test_cases.py").write_text("""import pytest
from decimal import Decimal

@pytest.fixture
def teardown():
    yield
    raise RuntimeError("teardown clue")

def emit():
    print("runtime-only diagnostic")

def test_two_failures(teardown):
    emit()
    assert False, "original assertion clue"

@pytest.mark.parametrize("value", [Decimal("0.000")], ids=["left::right [ spaces ]"])
def test_decimal(value):
    expected = "0.000"
    actual = "0"
    assert actual == expected

def test_group():
    shared = ValueError("cause one")
    shared.add_note("branch-specific diagnostic")
    raise ExceptionGroup("independent causes", [shared, TypeError("cause two"), shared])

def test_skipped():
    pytest.skip("not supported")

def test_generated():
    namespace = {}
    exec(compile("def run(payload):\\n    raise ValueError('generated clue')\\n", "<generated>", "exec"), namespace)
    namespace["run"](Decimal("0.000"))

def test_changing_first():
    import changing
    changing.run(1)

def test_changing_second():
    from pathlib import Path
    path = Path("changing.py").resolve()
    source = "def run(value):\\n    raise RuntimeError('second source clue')\\n"
    path.write_text(source)
    namespace = {}
    exec(compile(source, str(path), "exec"), namespace)
    namespace["run"](2)

def test_nested(tmp_path):
    inner = tmp_path / "test_inner.py"
    inner.write_text(
        "def test_inner(request):\\n"
        "    assert request.config.option.verbose == -1\\n"
        "    assert request.config.get_verbosity('assertions') == -1\\n"
    )
    assert pytest.main(["-q", "-p", "capture_plugin", str(inner)]) == 0
""")
    (tmp / "changing.py").write_text("def run(value):\n    raise ValueError('first source clue')\n")
    plugin = Path(__file__).resolve().parents[3] / "uni_agent/reward/pytest_feedback_capture.py"
    (tmp / "capture_plugin.py").write_text(plugin.read_text())
    report = tmp / "events.jsonl"
    env = {
        **os.environ,
        "PYTHONPATH": str(tmp),
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "UNI_AGENT_FEEDBACK_PATH": str(report),
        "UNI_AGENT_FEEDBACK_ROOT": str(tmp),
    }
    env.pop("PYTEST_ADDOPTS", None)
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "capture_plugin", "-p", "no:cacheprovider"],
        cwd=tmp,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert proc.returncode == 1, proc.stderr
    return load_capture(report.read_text())


def test_one_run_keeps_exact_identity_both_phases_and_captures(captured):
    events = [
        e for e in captured["events"] if e["nodeid"] == "test_cases.py::test_two_failures" and e["outcome"] == "failed"
    ]
    assert {e["phase"] for e in events} == {"call", "teardown"}
    assert "original assertion clue" in render_event(events[0], captured, False)
    assert "runtime-only diagnostic" in render_event(events[0], captured, False)
    decimal = next(e for e in captured["events"] if "test_decimal" in e["nodeid"] and e["outcome"] == "failed")
    assert decimal["nodeid"] == "test_cases.py::test_decimal[left::right [ spaces ]]"
    short = render_event(decimal, captured, False)
    assert "Decimal('0.000')" in short and "actual='0'" in short and "expected='0.000'" in short


def test_exception_group_branches_survive_short(captured):
    event = next(e for e in captured["events"] if "test_group" in e["nodeid"] and e["outcome"] == "failed")
    text = render_event(event, captured, False)
    assert "cause one" in text and "cause two" in text and "branch 2" in text
    assert text.count("branch-specific diagnostic") == 2
    assert "Cycle" not in text


def test_generated_code_keeps_native_arguments_and_labels_unavailable_source(captured):
    event = next(e for e in captured["events"] if "test_generated" in e["nodeid"] and e["outcome"] == "failed")
    text = render_event(event, captured, False)
    assert "payload=Decimal('0.000')" in text and "Statement/source unavailable" in text


def test_changed_source_snapshots_match_their_own_failure(captured):
    sources = captured["sources"]
    for name, clue in (("test_changing_first", "first source clue"), ("test_changing_second", "second source clue")):
        event = next(e for e in captured["events"] if name in e["nodeid"] and e["outcome"] == "failed")
        frame = next(f for f in event["exception"]["frames"] if f["path"] == "changing.py")
        assert clue in sources[frame["source"]]["text"]


def test_nested_pytest_runs_keep_their_configuration_and_do_not_mix_reports(captured):
    parent = [e for e in captured["events"] if e["nodeid"] == "test_cases.py::test_nested" and e["phase"] == "call"]
    assert parent[0]["outcome"] == "passed"
    assert not any("test_inner.py" in e["nodeid"] for e in captured["events"])


def test_old_pytest_keeps_the_progress_lines_the_official_parser_reads(tmp_path):
    (tmp_path / "shared_cases.py").write_text("def test_imported():\n    pass\n")
    (tmp_path / "test_cases.py").write_text("from shared_cases import test_imported\n")
    # pytest before 8: no assertion-only verbosity
    (tmp_path / "old_pytest.py").write_text(
        "import pytest\n\n"
        "@pytest.hookimpl(tryfirst=True)\n"
        "def pytest_configure(config):\n"
        "    config._parser._inidict.pop('verbosity_assertions', None)\n"
        "    config._inicache.pop('verbosity_assertions', None)\n"
    )
    plugin = Path(__file__).resolve().parents[3] / "uni_agent/reward/pytest_feedback_capture.py"
    (tmp_path / "capture_plugin.py").write_text(plugin.read_text())
    env = {
        **os.environ,
        "PYTHONPATH": str(tmp_path),
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "UNI_AGENT_FEEDBACK_PATH": str(tmp_path / "events.jsonl"),
        "UNI_AGENT_FEEDBACK_ROOT": str(tmp_path),
    }
    env.pop("PYTEST_ADDOPTS", None)
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-v", "--color=no", "-p", "old_pytest", "-p", "capture_plugin",
         "-p", "no:cacheprovider", "test_cases.py"],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "test_cases.py::test_imported PASSED" in proc.stdout


def test_old_pytest_lifts_truncation_for_the_evaluation_only(tmp_path, monkeypatch):
    from _pytest.assertion import truncate

    from uni_agent.reward import pytest_feedback_capture as plugin

    class Config:
        rootpath = tmp_path

        def __init__(self):
            self.option = types.SimpleNamespace(verbose=1, tbstyle="auto")

        def getini(self, name):
            raise ValueError(name)

    monkeypatch.setattr(truncate, "_should_truncate_item", lambda item: True, raising=False)
    for name in ("_ACTIVE_CONFIG", "_ROOT", "_FILE", "_VALUE_CHARS"):
        monkeypatch.setattr(plugin, name, getattr(plugin, name))
    monkeypatch.setattr(plugin, "_ACTIVE_CONFIG", None)
    monkeypatch.setenv("UNI_AGENT_FEEDBACK_ROOT", str(tmp_path))
    monkeypatch.delenv("UNI_AGENT_FEEDBACK_PATH", raising=False)
    monkeypatch.delenv("UNI_AGENT_FEEDBACK_CONTEXT", raising=False)
    config = Config()
    plugin.pytest_configure(config)
    assert config.option.verbose == 1 and config.option.tbstyle == "long"
    assert not truncate._should_truncate_item(types.SimpleNamespace(config=config))
    assert truncate._should_truncate_item(types.SimpleNamespace(config=Config()))


def _run_pytest(tmp_path, plugin, extra_env=()):
    (tmp_path / "test_cases.py").write_text("def test_ok():\n    pass\n")
    env = {
        **os.environ,
        "PYTHONPATH": str(tmp_path),
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "UNI_AGENT_FEEDBACK_PATH": str(tmp_path / "events.jsonl"),
        "UNI_AGENT_FEEDBACK_ROOT": str(tmp_path),
        **dict(extra_env),
    }
    env.pop("PYTEST_ADDOPTS", None)
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-rA", "--color=no", "-p", plugin, "-p", "no:cacheprovider", "test_cases.py"],
        cwd=tmp_path, env=env, text=True, capture_output=True, timeout=30,
    )


def test_the_plugin_parses_on_the_oldest_task_interpreters():
    import ast

    source = (Path(__file__).resolve().parents[3] / "uni_agent/reward/pytest_feedback_capture.py").read_text()
    # Python 3.6 rejects `from __future__ import annotations` (SWE-bench's sklearn environments)
    assert "from __future__ import annotations" not in source
    ast.parse(source, feature_version=(3, 6))


def test_a_plugin_the_interpreter_cannot_import_leaves_the_tests_running(tmp_path):
    from uni_agent.reward.feedback_capture import SHIM

    (tmp_path / "broken_impl.py").write_text("from __future__ import not_a_feature\n")
    (tmp_path / "capture_shim.py").write_text(SHIM.format(impl="broken_impl"))
    proc = _run_pytest(tmp_path, "capture_shim")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "PASSED test_cases.py::test_ok" in proc.stdout


def test_a_failing_configure_disables_capture_and_leaves_the_tests_running(tmp_path):
    from uni_agent.reward.feedback_capture import SHIM

    plugin = Path(__file__).resolve().parents[3] / "uni_agent/reward/pytest_feedback_capture.py"
    (tmp_path / "capture_impl.py").write_text(plugin.read_text())
    (tmp_path / "capture_shim.py").write_text(SHIM.format(impl="capture_impl"))
    proc = _run_pytest(tmp_path, "capture_shim", {"UNI_AGENT_FEEDBACK_VALUE_CHARS": "not-a-number"})
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "PASSED test_cases.py::test_ok" in proc.stdout
    assert "Feedback capture disabled" in (tmp_path / "events.jsonl").read_text()


def test_junit_preserves_duplicate_phase_records_complex_ids_and_captures():
    xml = """<testsuite><testcase classname="test_cases" name="test_f[a::b]">
    <failure message="call">original assertion</failure><system-out>runtime clue</system-out></testcase>
    <testcase classname="test_cases" name="test_f[a::b]">
    <error message="failed on teardown">cleanup error</error></testcase></testsuite>"""
    data = from_junit(xml, ["test_cases.py::test_f[a::b]"])
    assert len(data["events"]) == 2
    assert {e["phase"] for e in data["events"]} == {"call", "teardown"}
    assert "runtime clue" in render_event(data["events"][0], data, False)


def test_changed_neighbors_and_transitive_multiline_context():
    lines = ["def outer(value):"] + ["    # filler"] * 15
    lines += ["    mode = get_mode()"] + ["    # filler"] * 15
    changed_line = len(lines) + 1
    lines += ["    handler = handlers[mode]"] + ["    # filler"] * 20
    failure = len(lines) + 1
    lines += ["    return handler(", "        value", "    )"]
    text, _, name = source_context("\n".join(lines), failure, {changed_line}, 10)
    assert name == "outer"
    assert "mode = get_mode()" in text and "handler = handlers[mode]" in text
    assert "value" in text and f"{changed_line - 10}:" in text and f"{changed_line + 10}:" in text


def test_qualified_methods_and_module_changes():
    source = (
        "LIMIT = 3\nclass A:\n    def load(self):\n        return LIMIT\n"
        "class B:\n    def load(self):\n        return 4\n"
    )
    assert source_context(source, 4, {4})[2] == "A.load"
    assert source_context(source, 7, {7})[2] == "B.load"
    assert "LIMIT = 3" in source_context(source, 1, {1})[0]


def test_different_underlying_causes_are_not_grouped():
    def event(cause):
        return {
            "phase": "call",
            "exception": {
                "type": "Wrapper",
                "message": "load failed",
                "frames": [],
                "cause": {"type": cause, "message": "distinct", "frames": []},
            },
        }

    assert signature(event("IndexError")) != signature(event("ValueError"))


def test_incomplete_capture_and_missing_tests_are_not_reported_as_observed_regressions():
    data = load_capture(
        '{"event":{"nodeid":"a","phase":"call","outcome":"failed","text":"E AssertionError"}}\n{"event":'
    )
    text = render_diagnostic(result(["a"], regressions=["missing"], complete=False), data)
    assert "incomplete" in text and "missing (regression test)" in text and "Observed: not observed" in text
    assert "incomplete or invalid" in text
    assert "change broke" not in text


@pytest.mark.parametrize("budget", [100, 400, 1000, 8000])
def test_complete_budget_counts_and_explicit_omissions(budget):
    events = [
        {"nodeid": f"test_long_{i}", "phase": "call", "outcome": "failed", "text": "E AssertionError: " + "x" * 3000}
        for i in range(12)
    ]
    text = render_diagnostic(
        result([e["nodeid"] for e in events]), {"events": events, "complete": True}, max_chars=budget
    )
    assert len(text) <= budget
    assert "omitted" in text or "output limit too small" in text
    assert "budget" not in text


def test_prompt_shrink_preserves_test_identity():
    data = {
        "events": [{"nodeid": "a", "phase": "call", "outcome": "failed", "text": "E AssertionError"}],
        "complete": True,
    }
    text = render_diagnostic(result(["a"]), data)
    shrunk = shrink_feedback(text, 1000)
    assert "a (target test)" in shrunk and "FAILED" in shrunk


def test_reference_lines_are_mapped_to_student_tree():
    student = "diff --git a/x.py b/x.py\n@@ -1,1 +1,3 @@\n+a\n+b\n c\n"
    reference = "diff --git a/x.py b/x.py\n@@ -10,1 +10,1 @@\n-old\n+new\n"
    assert 12 in changed_in_final({"student_patch": student, "reference_patch": reference})["x.py"]


def test_branch_condition_dependencies_are_transitive():
    lines = ["def load(value):", "    import handlers", "    flag = get_flag()"]
    lines += ["    # irrelevant gap"] * 30
    lines += ["    if flag:", "        return handlers.transform(value)"]
    text, names, _ = source_context("\n".join(lines), len(lines), set())
    assert "flag = get_flag()" in text and "import handlers" in text
    assert "flag" in names


def test_module_change_visible_from_function_and_entire_neighbor_window():
    lines = ["LIMIT = 3"] + ["# context"] * 30 + ["def load():", "    return LIMIT"]
    text, _, name = source_context("\n".join(lines), len(lines), {1})
    assert name == "load" and "      1: LIMIT = 3" in text and "   11:" in text
    assert not re.search(r"^[>* ]*\*\s+\d+:", text, re.M)


def test_source_change_keeps_neighbors_when_failing_line_is_itself_changed():
    source = "\n".join(["# neighboring context"] * 20 + ["raise ValueError('changed clue')"] + ["# neighbor"] * 20)
    data = {
        "events": [
            {
                "nodeid": "a",
                "phase": "call",
                "outcome": "failed",
                "identity": "collector",
                "exception": {
                    "type": "ValueError",
                    "message": "changed clue",
                    "frames": [
                        {
                            "path": "a.py",
                            "line": 21,
                            "start": 1,
                            "end": 41,
                            "source": "a",
                            "repo": True,
                            "statement": "raise ValueError('changed clue')",
                        }
                    ],
                },
            }
        ],
        "sources": {"a": {"text": source}},
        "complete": True,
        "context": {"student_patch": "diff --git a/a.py b/a.py\n@@ -21 +21 @@\n-old\n+new\n"},
    }
    text = render_diagnostic(result(["a"]), data)
    assert ">    21:" in text and "   11:" in text and "   31:" in text
    # Enough room for this complete case; source neighbors survive a reduced prompt.
    shrunk = shrink_feedback(text + " " * 500, len(text) + 250)
    assert "   11:" in shrunk and "   31:" in shrunk


def test_only_unchanged_assertion_lines_and_duplicate_repr_are_folded():
    expected = "same context\n" * 80 + "expected clue\n"
    actual = "same context\n" * 80 + "actual clue\n"
    report = f"E       assert {actual!r} == {expected!r}\n"
    report += "\n".join("E         " + line for line in difflib.ndiff(expected.splitlines(), actual.splitlines()))
    rendered = fold_assertion(report)
    assert "- expected clue" in rendered and "+ actual clue" in rendered
    assert "Duplicate expanded string values omitted" in rendered
    assert "unchanged comparison lines omitted" in rendered
    unknown = "E       assert " + "x" * 1100
    assert fold_assertion(unknown) == unknown  # no known diff: retain conservative evidence


def test_unstructured_fallback_preserves_runtime_clue_without_assigning_it_to_a_test():
    text = render_diagnostic(result(["a"]), {"events": [], "complete": False}, output="runtime import clue")
    assert "runtime import clue" in text and "Observed: not observed" in text and "unattributed" in text


def test_shared_source_references_are_resolvable_after_output_reduction():
    events = [
        {
            "nodeid": f"a{i}",
            "phase": "call",
            "outcome": "failed",
            "identity": "collector",
            "exception": {
                "type": "ValueError",
                "message": "clue",
                "frames": [
                    {"path": "a.py", "line": 1, "values": {"data": "x" * 2000}, "statement": "raise ValueError(data)"}
                ],
            },
        }
        for i in range(8)
    ]
    text = render_diagnostic(result([e["nodeid"] for e in events]), {"events": events, "complete": True})
    assert "same value shown above at a.py:1" in text
    shrunk = shrink_feedback(text, len(text) - 800)
    if "same value shown above at a.py:1" in shrunk:
        assert "data=" + "x" * 2000 in shrunk
    assert "not shown" in shrunk and all(f"a{i}" in shrunk for i in range(8))


def test_spaced_unicode_and_renamed_diff_paths():
    spaced = "diff --git a/a file.py b/a file.py\n--- a/a file.py\n+++ b/a file.py\n@@ -3 +3 @@\n-old\n+new\n"
    assert changed_lines(spaced) == {"a file.py": {3}}
    quoted = 'diff --git "a/caf\\303\\251.py" "b/caf\\303\\251.py"\n'
    quoted += '--- "a/caf\\303\\251.py"\n+++ "b/caf\\303\\251.py"\n@@ -3 +3 @@\n-old\n+new\n'
    assert changed_lines(quoted) == {"café.py": {3}}
    literal = 'diff --git "a/🎈.py" "b/🎈.py"\n--- "a/🎈.py"\n+++ "b/🎈.py"\n@@ -3 +3 @@\n-old\n+new\n'
    assert changed_lines(literal) == {"🎈.py": {3}}
    renamed = "diff --git a/old.py b/new.py\nsimilarity index 100%\nrename from old.py\nrename to new.py\n"
    reference = "diff --git a/old.py b/old.py\n@@ -3 +3 @@\n-old\n+new\n"
    assert changed_in_final({"student_patch": renamed, "reference_patch": reference}) == {"new.py": {3}}


def test_junit_path_brackets_do_not_get_confused_with_parameters():
    data = from_junit(
        '<testsuite><testcase classname="tests.test[a]" name="test_f[x::y]">'
        "<failure>clue</failure></testcase></testsuite>",
        ["tests/test[a].py::test_f[x::y]"],
    )
    assert data["events"][0]["nodeid"] == "tests/test[a].py::test_f[x::y]"


def test_changed_class_attribute_keeps_its_class_header_outside_neighbor_window():
    lines = ["class A:"] + ["    # gap"] * 30 + ["    LIMIT = 3", "    def load(self):", "        return self.LIMIT"]
    text, _, name = source_context("\n".join(lines), len(lines), {32})
    assert name == "A.load" and "class A:" in text and "LIMIT = 3" in text


def test_discarded_student_test_changes_do_not_shift_reference_coordinates():
    student = "diff --git a/test_x.py b/test_x.py\n@@ -1,1 +1,3 @@\n+a\n+b\n c\n"
    reference = "diff --git a/test_x.py b/test_x.py\n@@ -10 +10 @@\n-old\n+new\n"
    changes = changed_in_final(
        {"student_patch": student, "reference_patch": reference, "restored_test_files": ["test_x.py"]}
    )
    assert changes == {"test_x.py": {10}}


def test_structurally_invalid_capture_records_keep_prior_evidence():
    data = load_capture('{"event":{"nodeid":"a","phase":"call","outcome":"failed"}}\n42\n{"sources":[]}')
    assert len(data["events"]) == 1 and len(data["errors"]) == 2


def test_grading_disagreements_keep_both_outcomes_and_all_failed_ids():
    events = [
        {"nodeid": "a", "phase": "call", "outcome": "passed"},
        {"nodeid": "b", "phase": "call", "outcome": "failed", "text": "E ValueError: runtime clue"},
    ]
    text = render_diagnostic(result(["a"], ["b"]), {"events": events, "complete": True})
    assert "a (target test)" in text and "Observed: passed; official grade: failure" in text
    assert "b (target test)" in text and "Observed: failed; official grade: success" in text
    assert "runtime clue" in text


def test_xfail_and_xpass_are_distinguished_from_observed_assertion_failures():
    events = [
        {"nodeid": "xf", "phase": "call", "outcome": "skipped", "xfail": "known issue"},
        {"nodeid": "xp", "phase": "call", "outcome": "passed", "xfail": "known issue"},
    ]
    text = render_diagnostic(result(["xf"], ["xp"]), {"events": events, "complete": True})
    assert "xf (target test)" in text and "Observed: xfail" in text
    assert "xp (target test)" in text and "Observed: xpass" in text
    assert "0 failed" not in text


def test_hex_input_messages_are_not_treated_as_memory_addresses():
    def event(message):
        return {"phase": "call", "exception": {"type": "ValueError", "message": message, "frames": []}}

    assert signature(event("invalid value 0xAA")) != signature(event("invalid value 0xBB"))
    assert signature(event("invalid <A object at 0xAA>")) == signature(event("invalid <A object at 0xBB>"))


def native_string_report(actual, expected):
    return f"E       assert {actual!r} == {expected!r}\n" + "\n".join(
        "E         " + line for line in difflib.ndiff(expected.splitlines(), actual.splitlines())
    )


def test_condensing_keeps_multiple_differing_regions_and_small_structural_tail():
    expected = "expected first\n" + "same context\n" * 80 + "expected last\n</root>\n"
    actual = "actual first\n" + "same context\n" * 80 + "actual last\n</root>\n"
    rendered = fold_assertion(native_string_report(actual, expected))
    for clue in ("- expected first", "+ actual first", "- expected last", "+ actual last", "</root>"):
        assert clue in rendered
    assert "unchanged comparison lines omitted" in rendered
    short = native_string_report("<root>\nactual\n</root>\n", "<root>\nexpected\n</root>\n")
    assert fold_assertion(short) == short


def test_incomplete_custom_and_newline_sensitive_comparisons_remain_intact():
    report = native_string_report("same\n" * 80 + "actual\n", "same\n" * 80 + "expected\n")
    incomplete = "\n".join(line for line in report.splitlines() if line != "E           same")
    assert fold_assertion(incomplete) == incomplete
    custom = "E       assert 'left' == 'right'\nE         - custom explanation\nE         + further clue"
    assert fold_assertion(custom) == custom
    newline = native_string_report("same\n" * 80 + "actual", "same\n" * 80 + "expected\n")
    assert fold_assertion(newline) == newline
    crlf = native_string_report("same\r\n" * 80 + "actual\r\n", "same\n" * 80 + "expected\n")
    assert fold_assertion(crlf) == crlf
    carriage_return = native_string_report("same\r" * 80 + "actual", "same\r" * 80 + "expected\r")
    assert fold_assertion(carriage_return) == carriage_return
    separate_terminators = native_string_report("a" * 200 + "\r", "\n" + "b" * 200)
    assert fold_assertion(separate_terminators) == separate_terminators


def test_each_test_has_one_block_with_both_body_and_teardown_errors(captured):
    node = "test_cases.py::test_two_failures"
    text = render_diagnostic(result([node]), captured)
    assert text.count(node) == 1
    assert "Target tests: 1 failed." in text
    assert "FAILED" in text and "ERROR at teardown" in text
    assert "original assertion clue" in text and "teardown clue" in text
    assert "0 skipped" not in text and "group=" not in text and "case=" not in text


def test_captured_output_preserves_nonconsecutive_order():
    event = {"identity": "collector", "text": "E ValueError: clue", "capture": "first\nsecond\nfirst\nfirst"}
    text = render_event(event, {}, False)
    assert "    first\n    second\n    first [repeated 2 consecutive times]" in text


def test_parameter_value_equal_to_a_previous_cases_input_is_not_silently_hidden():
    shared = {"values": {}, "frames": {}, "sources": set()}
    first = {"identity": "collector", "parameters": {"value": "'old input'"}, "text": "E ValueError: first"}
    render_event(first, {}, False, shared=shared)
    second = {
        "identity": "collector",
        "parameters": {"value": "'new input'"},
        "exception": {
            "type": "ValueError",
            "message": "second",
            "frames": [
                {"path": "a.py", "line": 1, "statement": "raise ValueError(value)", "values": {"value": "'old input'"}},
            ],
        },
    }
    text = render_event(second, {}, False, shared=shared)
    assert "value='new input'" in text and "value='old input'" in text


def test_ungraded_failures_are_filtered_but_collection_errors_explain_missing_tests():
    events = [
        {"nodeid": "target", "phase": "call", "outcome": "failed", "text": "E AssertionError: target clue"},
        {"nodeid": "ungraded", "phase": "call", "outcome": "failed", "text": "E AssertionError: unrelated clue"},
        {"nodeid": "broken.py", "phase": "collection", "outcome": "failed", "text": "E ImportError: blocker"},
    ]
    text = render_diagnostic(result(["target", "missing"]), {"events": events, "complete": False})
    assert "target clue" in text and "unrelated clue" not in text
    assert "ERROR collecting tests" in text and "blocker" in text
    assert "Target tests: 1 failed, 1 not observed." in text
    assert "Names of these 1 unobserved tests are omitted; no pass/fail outcome is inferred." in text
    assert "________________ missing " not in text


def test_bootstrap_cause_takes_priority_over_thousands_of_missing_records():
    nodes = [f"tests/test_domain.py::test_case[{i}]" for i in range(4000)]
    output = (
        "Switched to a new branch; irrelevant evaluator checkout advice\n"
        ">>>>> Start Test Output\n"
        "ImportError while loading conftest '/testbed/tests/conftest.py'.\n"
        'E     File "/testbed/package/decorators.py", line 1\n'
        "E       commit 22170507e839139c75b6b4fc38876d93d18f8fee\n"
        "E                            ^\n"
        "E   SyntaxError: invalid decimal literal\n"
        ">>>>> End Test Output\nirrelevant evaluator teardown\n"
    )
    text = render_diagnostic(result(nodes, complete=False), {"events": [], "complete": False}, output=output)
    assert "Target tests: 4000 not observed." in text
    assert "Names of these 4,000 unobserved tests are omitted" in text
    assert nodes[0] not in text
    assert "irrelevant evaluator" not in text
    for rendered in (text, shrink_feedback(text, 2000)):
        assert len(rendered) <= 2000
        assert "SyntaxError: invalid decimal literal" in rendered
        assert "commit 22170507e839139c75b6b4fc38876d93d18f8fee" in rendered
        assert 'File "/testbed/package/decorators.py", line 1' in rendered


def test_ambiguous_harness_boundaries_keep_raw_evaluator_evidence():
    output = (
        "outside clue\n>>>>> Start Test Output\n"
        "printed marker:\n>>>>> End Test Output\n"
        "E RuntimeError: later failure clue\n>>>>> End Test Output\n"
    )
    text = render_diagnostic(result(), {"events": [], "complete": False}, output=output)
    assert "outside clue" in text and "later failure clue" in text


def test_logged_traceback_alone_does_not_justify_grouping_missing_observations():
    nodes = [f"unobserved_{i}" for i in range(20)]
    output = (
        "Traceback (most recent call last):\n"
        '  File "tests/test_widget.py", line 12, in test_widget\n'
        "    do_work()\nValueError: ordinary_logged_test_error\n"
        "================== 1 failed, 5 passed in 0.3s ==================\n"
    )
    text = render_diagnostic(result(nodes), {"events": [], "complete": False}, output=output)
    assert all(node in text for node in nodes)
    assert "ordinary_logged_test_error" in text
    assert "Names of these" not in text
    assert "ERROR collecting tests" not in text


def test_real_error_precedes_names_when_identifiers_alone_exceed_limit():
    nodes = [f"test_domain.py::test_case[{i}]" for i in range(4000)]
    output = (
        "Traceback (most recent call last):\n"
        '  File "/usr/local/bin/pytest", line 8, in <module>\n'
        "    from pytest import console_main\n"
        "ModuleNotFoundError: No module named 'missing_dependency'\n"
    )
    text = render_diagnostic(result(nodes), {"events": [], "complete": False}, max_chars=2000, output=output)
    assert len(text) <= 2000
    assert "ModuleNotFoundError: No module named 'missing_dependency'" in text
    assert "Target tests: 4000 not observed." in text
    assert "additional test identifiers omitted" in text
    assert "4000 failed" not in text
    unlisted = int(re.search(r"(\d+) additional test identifiers omitted", text)[1])
    shrunk = shrink_feedback(text, len(text) - 200)
    assert len(shrunk) <= len(text) - 200
    assert int(re.search(r"(\d+) additional test identifiers omitted", shrunk)[1]) > unlisted
    assert "Test details not shown for these 4000 tests" in shrunk
    assert "ModuleNotFoundError: No module named 'missing_dependency'" in shrunk


def test_explicit_junit_collection_metadata_keeps_ungraded_blocking_cause():
    xml = (
        '<testsuite><testcase name="broken.py"><error message="collection failure">'
        "E ImportError: collection clue</error></testcase></testsuite>"
    )
    data = from_junit(xml, ["test_target.py::test_case"])
    assert data["events"][0]["phase"] == "collection"
    text = render_diagnostic(result(["test_target.py::test_case"]), data)
    assert "ERROR collecting tests" in text and "collection clue" in text
    generic = from_junit(xml.replace("collection failure", "ImportError"), [])
    assert generic["events"][0]["phase"] == "error"


def test_short_failure_keeps_branch_inputs_and_both_errors_before_huge_capture():
    node = "test_inputs.py::test_guard"
    event = {
        "nodeid": node,
        "phase": "call",
        "outcome": "failed",
        "parameters": {"payload": "'original input'"},
        "capture": "large irrelevant log " * 4000,
        "exception": {
            "type": "RuntimeError",
            "message": "guard clue",
            "frames": [
                {
                    "path": "guard.py",
                    "line": 3,
                    "source": "guard",
                    "function": "run",
                    "repo": True,
                    "statement": "raise RuntimeError('guard clue')",
                    "values": {"mode": "0", "unused": "'unrelated'"},
                }
            ],
        },
    }
    data = {
        "events": [
            event,
            {"nodeid": node, "phase": "teardown", "outcome": "failed", "text": "E ValueError: cleanup clue"},
        ],
        "sources": {"guard": {"text": "def run(mode):\n    if mode == 0:\n        raise RuntimeError('guard clue')\n"}},
        "complete": True,
    }
    text = render_diagnostic(result([node]), data, max_chars=2000)
    assert len(text) <= 2000
    assert "payload='original input'" in text and "mode=0" in text
    assert "guard clue" in text and "cleanup clue" in text and "ERROR at teardown" in text
    assert "captured output omitted" in text
    assert "large irrelevant log" not in text


def test_feedback_shrinking_removes_captures_before_failure_records():
    nodes = ["first", "second"]
    events = [
        {
            "nodeid": node,
            "phase": "call",
            "outcome": "failed",
            "text": f"E ValueError: {node} clue",
            "capture": "log " * 2000,
        }
        for node in nodes
    ]
    full = render_diagnostic(result(nodes), {"events": events, "complete": True})
    shrunk = shrink_feedback(full, 2000)
    assert len(shrunk) <= 2000
    assert "first clue" in shrunk and "second clue" in shrunk
    assert shrunk.count("[Captured output omitted to keep failure details.]") == 2
    assert "log log log" not in shrunk


def test_shrinking_preserves_omitted_status_and_all_stage_labels():
    events = [
        {"nodeid": "a", "phase": "call", "outcome": "failed", "text": "E AssertionError: " + "a" * 1200},
        {"nodeid": "b", "phase": "call", "outcome": "failed", "text": "E AssertionError: " + "b" * 1200},
        {"nodeid": "b", "phase": "teardown", "outcome": "failed", "text": "E RuntimeError: cleanup clue"},
    ]
    text = render_diagnostic(result(["a", "b", "missing"]), {"events": events, "complete": False})
    shrunk = shrink_feedback(text, 2000)
    assert len(shrunk) <= 2000
    assert "missing (target test): not observed" in shrunk
    assert "b (target test): failed; FAILED, ERROR at teardown" in shrunk
    assert "budget" not in shrunk


def test_identical_repeated_reports_fold_without_merging_distinct_failures():
    first = {"nodeid": "a", "phase": "call", "outcome": "failed", "text": "E ValueError: first clue"}
    different = {**first, "text": "E ValueError: distinct clue"}
    text = render_diagnostic(result(["a"]), {"events": [first, dict(first), different], "complete": True})
    assert "Target tests: 1 failed." in text
    assert "Identical failure report repeated 2 times" in text
    assert text.count("E ValueError: first clue") == 1 and "E ValueError: distinct clue" in text


@pytest.mark.parametrize(
    "patch,old,expected",
    [
        ("diff --git a/x.py b/x.py\n@@ -1,5 +1,6 @@\n line1\n+inserted\n line2\n line3\n line4\n line5\n", 4, 5),
        ("diff --git a/x.py b/x.py\n@@ -1,5 +1,4 @@\n line1\n-line2\n line3\n line4\n line5\n", 4, 3),
        (
            (
                "diff --git a/x.py b/x.py\n@@ -1,6 +1,9 @@\n line1\n-line2\n-line3\n"
                "+new1\n+new2\n+new3\n+new4\n+new5\n line4\n line5\n line6\n"
            ),
            5,
            8,
        ),
    ],
)
def test_reference_coordinate_mapping_inside_context(patch, old, expected):
    assert _map_line(old, _patch_hunks(patch)["x.py"]) == expected


@pytest.mark.parametrize("value", ["'<A object at 0xAA>'", "{'text': '<A object at 0xAA>'}", "['<A object at 0xAA>']"])
def test_literal_address_like_data_retained(value):
    assert _clean_address(value) == value


@pytest.mark.parametrize(
    "left,right",
    [
        ("same\n" * 80 + "actual", "same\n" * 80 + "expected\n"),
        ("same\r\n" * 80 + "actual\r\n", "same\n" * 80 + "expected\n"),
        ("same\v" * 80 + "actual\v", "same\n" * 80 + "expected\n"),
        ("same\x85" * 80 + "actual\x85", "same\n" * 80 + "expected\n"),
        ("same\u2028" * 80 + "actual\u2028", "same\u2029" * 80 + "expected\u2029"),
        ("same\n" * 80 + "actual\r\n", "same\n" * 80 + "expected\r\n"),
    ],
)
def test_mixed_and_normalized_line_endings_do_not_remove_literals(left, right):
    report = f"E       assert {left!r} == {right!r}\n" + "\n".join(
        "E         " + line for line in difflib.ndiff(right.splitlines(), left.splitlines())
    )
    assert fold_assertion(report) == report


@pytest.mark.parametrize("marker", [END, OMISSIONS])
def test_literal_formatter_markers_in_capture_do_not_drop_later_id(marker):
    nodes = ["first", "second"]
    result = {
        "eval_completed": True,
        "resolved": False,
        "eval_report": {"test_status": {"FAIL_TO_PASS": {"failure": nodes, "success": []}}},
    }
    events = [
        {
            "nodeid": "first",
            "outcome": "failed",
            "phase": "call",
            "text": "E ValueError: first clue",
            "capture": marker,
        },
        {"nodeid": "second", "outcome": "failed", "phase": "call", "text": "E ValueError: second clue"},
    ]
    full = render_diagnostic(result, {"events": events, "complete": True})
    shrunk = shrink_feedback(full, len(full) - 10)
    assert all(node in shrunk for node in nodes)


def test_literal_formatter_end_marker_in_parameter_id_survives_shrinking():
    nodes = ["test_edges.py::test_case[" + END + "]", "second"]
    result = {
        "eval_completed": True,
        "resolved": False,
        "eval_report": {"test_status": {"FAIL_TO_PASS": {"failure": nodes, "success": []}}},
    }
    events = [
        {"nodeid": node, "outcome": "failed", "phase": "call", "text": "E ValueError: clue " + ("x" * 300)}
        for node in nodes
    ]
    full = render_diagnostic(result, {"events": events, "complete": True})
    shrunk = shrink_feedback(full, len(full) - 10)
    assert all(node in shrunk for node in nodes)


def test_fit_message_feedback_ignores_end_marker_in_captured_output():
    nodes = ["first", "second"]
    result = {
        "eval_completed": True,
        "resolved": False,
        "eval_report": {"test_status": {"FAIL_TO_PASS": {"failure": nodes, "success": []}}},
    }
    events = [
        {"nodeid": "first", "outcome": "failed", "phase": "call", "text": "E ValueError: first clue", "capture": END},
        {"nodeid": "second", "outcome": "failed", "phase": "call", "text": "E ValueError: second clue " + ("x" * 1500)},
    ]
    full = render_diagnostic(result, {"events": events, "complete": True})
    text = fit_message_feedback([{"role": "user", "content": "prefix\n" + full + "\nsuffix"}], 1000)[0]["content"]
    assert len(text) <= 1000 + len("prefix\n\nsuffix")
    assert all(node in text for node in nodes)


def test_deep_valid_expression_does_not_crash_source_renderer():
    source = "def f(x):\n    result = " + " + ".join(["x"] * 1200) + "\n    return result\n"
    text, _, _ = source_context(source, 3, set())
    assert "return result" in text


@pytest.mark.parametrize(
    "before,after,expected",
    [
        (["a\n", "b\n", "c\n"], ["prefix\n", "a\n", "b\n", "c\n"], {1: 2, 2: 3, 3: 4}),
        (["a\n", "b\n", "c\n"], ["a\n", "middle\n", "b\n", "c\n"], {1: 1, 2: 3, 3: 4}),
        (["a\n", "b\n", "c\n"], ["a\n", "b\n", "c\n", "suffix\n"], {1: 1, 2: 2, 3: 3}),
        (["a\n", "b\n", "c\n"], ["b\n", "c\n"], {2: 1, 3: 2}),
        (["a\n", "b\n", "c\n"], ["a\n", "c\n"], {1: 1, 3: 2}),
        (["a\n", "b\n", "c\n"], ["a\n", "b\n"], {1: 1, 2: 2}),
    ],
)
def test_zero_context_pure_insertions_deletions(before, after, expected):
    patch = "diff --git a/x.py b/x.py\n" + "".join(difflib.unified_diff(before, after, "a/x.py", "b/x.py", n=0))
    hunks = _patch_hunks(patch)["x.py"]
    assert {old: _map_line(old, hunks) for old in expected} == expected


@pytest.mark.parametrize("context", [0, 1, 3, 10])
def test_multiple_hunks_keep_all_original_coordinates(context):
    before = [f"old_{i}\n" for i in range(200)]
    after = before[:]
    # Edit from bottom to top so the positions below are original coordinates.
    after[173:175] = []
    after[142:142] = ["new_42a\n", "new_42b\n", "new_42c\n"]
    after[92:94] = ["replacement\n"]
    after[22:22] = ["new_2\n"]
    patch = "diff --git a/x.py b/x.py\n" + "".join(difflib.unified_diff(before, after, "a/x.py", "b/x.py", n=context))
    hunks = _patch_hunks(patch)["x.py"]
    assert len(hunks) > 1
    for new, token in enumerate(after, 1):
        if token in before:
            assert _map_line(before.index(token) + 1, hunks) == new


@pytest.mark.parametrize("full", ["['a', 'b']", "('a', 'b')", "42", "None", "b'ab'"])
def test_shortened_string_repr_does_not_match_non_string_wrapper(full):
    assert not _matches_repr("'a' [representation shortened; 1 units omitted]", full)


def test_feedback_fitting_preserves_marker_text_in_other_prompt_components():
    nodes = ["first", "second"]
    events = [
        {"nodeid": node, "phase": "call", "outcome": "failed", "text": "E ValueError: " + "x" * 1500} for node in nodes
    ]
    feedback = render_diagnostic(result(nodes), {"events": events, "complete": True})
    prefix = "Task text:\n" + START + "\nordinary task description\n"
    suffix = "\nFull trajectory:\nUNIQUE_TURN_EVIDENCE\n" + END + "\nclosing instructions"
    content = prefix + feedback + suffix
    fitted = fit_message_feedback([{"role": "user", "content": content}], 800)[0]["content"]
    assert fitted.startswith(prefix) and fitted.endswith(suffix)
    assert len(fitted) <= len(prefix) + 800 + len(suffix)


def test_ambiguous_quoted_feedback_does_not_authorize_removing_prompt_components():
    feedback = render_diagnostic(result(["test_a"]), {"events": [], "complete": True})
    content = "Task quotes previous feedback:\n" + feedback + "\nCurrent feedback:\n" + feedback
    messages = [{"role": "user", "content": content}]
    assert fit_message_feedback(messages, 100) == messages


def test_one_full_record_per_failure_then_short_records_then_names():
    from uni_agent.reward.diagnostic_feedback import BLOCK_HEADER, SUMMARY_NOTE

    def event(node, kind, message):
        frame = {"path": "pkg/mod.py", "line": 7, "function": "pick", "statement": "return items[idx]",
                 "values": {"items": "[1, 2]", "idx": "3", "table": f"'{node}" + "x" * 1500 + "'"}}
        return {"nodeid": node, "phase": "call", "outcome": "failed",
                "exception": {"type": kind, "message": message, "frames": [frame]}}

    same = [f"test_same_{i}" for i in range(6)]
    events = [event(n, "IndexError", "list index out of range") for n in same]
    events.append(event("test_other", "KeyError", "'k'"))
    data = {"events": events, "complete": True}
    nodes = same + ["test_other"]

    def blocks(text):
        heads = list(BLOCK_HEADER.finditer(text))
        return {h[1].split(" (")[0]: text[h.start():(heads[k + 1].start() if k + 1 < len(heads) else len(text))]
                for k, h in enumerate(heads)}

    full = render_diagnostic(result(nodes), data, max_chars=10**9)
    assert SUMMARY_NOTE not in full and OMISSIONS not in full
    # one test per failure stays in full, the tests failing the same way turn short
    second = render_diagnostic(result(nodes), data, max_chars=len(full) - 1)
    shown = blocks(second)
    assert SUMMARY_NOTE not in shown["test_same_0"] and SUMMARY_NOTE not in shown["test_other"]
    assert all(SUMMARY_NOTE in shown[n] for n in same[1:]) and OMISSIONS not in second
    # then names, while the representatives keep their full records
    third = render_diagnostic(result(nodes), data, max_chars=len(second) - 200)
    shown = blocks(third)
    assert OMISSIONS in third and SUMMARY_NOTE not in shown["test_same_0"] and "test_other" in shown
    # a short record per failure when even one full record each does not fit
    reps_short = render_diagnostic(result(nodes), data, max_chars=len(full) // 6)
    shown = blocks(reps_short)
    assert {"test_same_0", "test_other"} <= set(shown) and all(SUMMARY_NOTE in b for b in shown.values())
