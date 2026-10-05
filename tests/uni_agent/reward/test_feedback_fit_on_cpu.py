"""Fitting failures under a tiny limit: the same text as recomposing every candidate, in linear renders.

``_limited_output`` below is the recomposing implementation, kept verbatim as the oracle.
"""

import random
import sys

import pytest

from uni_agent.reward import diagnostic_feedback as df
from uni_agent.reward.diagnostic_feedback import (
    DETAILS,
    END,
    OMISSIONS,
    START,
    _entry_description,
    _render_entry,
    render_diagnostic,
)


def _limited_output(summary, entries, max_chars, data, neighbors):
    """Exceptional tiny limits: omit complete records/IDs with exact counts."""
    base = [START, summary[1]] if len(summary) > 1 else [START]
    # Preserve status and exceptional evaluation notices before the identifier list.
    for line in summary[2:]:
        if len("\n".join([*base, line, END])) + 180 <= max_chars:
            base.append(line)

    def compose(selected, identifiers=()):
        shared = {"values": {}, "frames": {}, "sources": set()}
        blocks = [_render_entry(entries[i], data, "summary", neighbors, shared) for i in selected]
        missing = len(entries) - len(selected)
        footer = (
            "\n".join(
                [
                    OMISSIONS,
                    *identifiers,
                    f"Test details omitted for {missing} tests; "
                    f"{missing - len(identifiers)} additional test identifiers omitted.",
                ]
            )
            if missing
            else ""
        )
        return "\n\n".join(filter(None, ["\n".join(base), DETAILS if blocks else "", *blocks, footer, END]))

    # Even if the identifiers alone cannot fit, a real error takes precedence.
    # Recompose shared references only from records actually included.
    selected = []
    for i, entry in enumerate(entries):
        if entry["events"] and len(compose([*selected, i])) <= max_chars:
            selected.append(i)
    kept = []
    selected_set = set(selected)
    for i, entry in enumerate(entries):
        if i in selected_set:
            continue
        line = "- " + _entry_description(entry)
        if len(compose(selected, [*kept, line])) <= max_chars:
            kept.append(line)
    text = compose(selected, kept)
    if len(text) <= max_chars:
        return text
    minimal = "[Test feedback unavailable: output limit too small for evaluation status and omission counts.]"
    return minimal if len(minimal) <= max_chars else ""


CORE = '''import math

LIMIT = 3


def helper(payload, idx, table=None):
    """Pick one item."""
    table = dict(payload) if table is None else table
    if idx > LIMIT:
        raise ValueError(idx)
    value = table[idx]
    return value


def compute(record):
    total = 0
    for key, value in record.items():
        total += helper(value, key)
    assert total == record, "mismatch"
    return total
'''
TEST = '''from pkg.core import compute


def test_case(i):
    record = {"a": i}
    expected = {"total": 3 * i}
    assert compute(record) == expected
'''
CORE_LINES = [
    (10, "raise ValueError(idx)"),
    (11, "value = table[idx]"),
    (18, "total += helper(value, key)"),
    (19, 'assert total == record, "mismatch"'),
    (8, "table = dict(payload) if table is None else table"),
]
NAMES = ["payload", "idx", "table", "value", "self", "total", "record", "key", "unused"]
# Values at and just over the 180 characters past which a repeated value is referenced instead of repeated.
VALUES = [
    "3",
    "'a'",
    "None",
    "<pkg.Thing object at 0x7f00ab12>",
    "<function helper at 0x7fff0000>",
    "'" + "w" * 178 + "'",
    "'" + "x" * 179 + "'",
    "'" + "y" * 180 + "'",
    "[" + ", ".join(str(n) for n in range(90)) + "]",
    "{'k': '" + "z" * 400 + "'}",
]
MESSAGES = ["", "bad", "m" * 60, "list index out of range", "long message " * 40, "q" * 2500]
TEXTS = [
    "",
    "E   AssertionError: assert 'abc' == 'abd'\nE     \nE     - abd\nE     ?   ^\nE     + abc\nE     ?   ^",
    "E   ValueError: 3",
    "    def test_case(i):\n>       assert compute(record) == expected\nE       assert 0 == 1\n\n"
    "tests/test_core.py:7: AssertionError",
    "E   AssertionError: assert [1, 2] == [1, 3]\nE     Full output truncated (3 lines hidden), use '-vv' to show",
    "E   KeyError: " + "'k" + "e" * 600 + "'",
    "ImportError while loading conftest '/testbed/conftest.py'.\nE   ModuleNotFoundError: No module named 'x'",
]
LABELS = ["target test", "regression test", "evaluation error", "ungraded test"]
EXTRA = [
    "Target tests: 3 failed.",
    "Previously passing tests: 2,997 failed, 3 not observed.",
    "Diagnostic warning: " + "w" * 300,
    "Evaluation error: CommandTimeoutError: timeout after 180 seconds",
    "Submission: no code changes (empty patch).",
    "x" * 2000,
]


def frame(rng, test=False):
    if test:
        return {"path": "tests/test_core.py", "line": 7, "function": "test_case", "source": "test", "repo": False,
                "statement": "assert compute(record) == expected",
                "values": {name: rng.choice(VALUES) for name in ("record", "expected", "unused")}}
    line, statement = rng.choice(CORE_LINES)
    return {"path": "pkg/core.py", "line": line, "function": rng.choice(["helper", "compute", None]),
            "statement": statement, "source": rng.choice(["core", "core", "absent"]), "repo": rng.random() < 0.8,
            "values": {name: rng.choice(VALUES) for name in rng.sample(NAMES, rng.randint(0, len(NAMES)))}}


def exception(rng, depth=0):
    frames = [frame(rng, test=True)] + [frame(rng) for _ in range(rng.randint(0, 4))]
    exc = {"type": rng.choice(["ValueError", "builtins.KeyError", "AssertionError", "pkg.CustomError"]),
           "message": rng.choice(MESSAGES), "frames": frames}
    if depth < 2 and rng.random() < 0.2:
        exc[rng.choice(["cause", "context"])] = exception(rng, depth + 1)
    if rng.random() < 0.1:
        exc["notes"] = ["note " + rng.choice(MESSAGES)]
    if depth < 2 and rng.random() < 0.05:
        exc["branches"] = [exception(rng, depth + 1) for _ in range(2)]
    return exc


def event(rng, node):
    out = {"nodeid": node, "outcome": "failed", "text": rng.choice(TEXTS),
           "phase": rng.choice(["call", "call", "setup", "teardown", "collection", "diagnostic"]),
           "identity": rng.choice(["collector", "junit", "unparsed"])}
    if rng.random() < 0.8:
        out["exception"] = exception(rng)
    if rng.random() < 0.5:
        out["parameters"] = {"i": str(rng.randint(0, 2)), "payload": rng.choice(VALUES)}
    if rng.random() < 0.3:
        out["capture"] = "printed\nprinted\nother"
    if rng.random() < 0.2:
        out["_repeat_count"] = rng.randint(2, 4)
    return out


def entry(rng, k, with_events):
    node = f"tests/test_mod_{k % 7}.py::test_case[{k}]" + "p" * rng.choice([0, 0, 5, 120])
    out = {"node": node, "label": rng.choice(LABELS)}
    if rng.random() < with_events:
        out.update(status="failed", events=[event(rng, node) for _ in range(rng.choice([1, 1, 1, 2, 3]))])
    else:
        out.update(status=rng.choice(["not observed", "passed", "skipped", "xfail", "failed"]), events=[])
    if rng.random() < 0.8:
        out["grade"] = rng.choice(["failure", "success"])
    if rng.random() < 0.8:
        out["reason"] = rng.choice(["", "skipped: not supported", START + "\nreason after a marker"])
    return out


def case(seed, n, with_events=0.7):
    rng = random.Random(seed)
    summary = [START, "Evaluation completed. Task not resolved.", *rng.sample(EXTRA, rng.randint(0, len(EXTRA)))]
    summary = summary[: rng.choice([1, 2, len(summary), len(summary)])]
    changes = rng.choice([{}, {"pkg/core.py": {11}}, {"pkg/core.py": {9, 10}, "tests/test_core.py": {7}}])
    data = {"sources": {"core": {"text": CORE}, "test": {"text": TEST}}, "_changes": changes, "context": {}}
    return summary, [entry(rng, k, with_events) for k in range(n)], data, rng.choice([0, 1, 10])


def small_case(seed):
    rng = random.Random(seed)
    summary = [START, "Evaluation completed. Task not resolved.", "Target tests: 3 failed.", "w" * rng.randint(0, 40)]
    entries = []
    for k in range(4):
        node = f"tests/test_core.py::test_case[{k}]"
        if k == rng.randint(0, 3):
            entries.append({"node": node, "label": "regression test", "status": "not observed", "events": []})
            continue
        frame_ = {"path": "pkg/core.py", "line": 11, "statement": "value = table[idx]", "source": "core",
                  "values": {"table": rng.choice(VALUES[5:8]), "idx": rng.choice(VALUES[:3])}}
        exc = {"type": "KeyError", "message": "m" * rng.randint(0, 40), "frames": [frame_]}
        entries.append({"node": node, "label": "target test", "status": "failed", "grade": "failure",
                        "events": [{"nodeid": node, "phase": "call", "outcome": "failed", "exception": exc}]})
    return summary, entries, {"sources": {"core": {"text": CORE}}, "_changes": {}}, 10


def same(summary, entries, data, neighbors, budget):
    expected = _limited_output(summary, entries, budget, data, neighbors)
    assert df._limited_output(summary, entries, budget, data, neighbors) == expected, budget
    return expected


def walk(summary, entries, data, neighbors, budget, steps):
    """Each text again at its own length (fits by 0), one more (fits by 1) and one less (misses by 1)."""
    for _ in range(steps):
        text = same(summary, entries, data, neighbors, budget)
        for exact in (len(text) - 1, len(text), len(text) + 1):
            same(summary, entries, data, neighbors, max(0, exact))
        if not text:
            return
        budget = len(text) - 1


def test_every_limit_on_a_small_input():
    summary, entries, data, neighbors = small_case(0)
    full = same(summary, entries, data, neighbors, 10**9)
    for budget in range(len(full) + 2):
        same(summary, entries, data, neighbors, budget)


@pytest.mark.parametrize("seed, n", list(enumerate([1, 2, 3, 5, 8, 13])))
def test_texts_that_fit_exactly(seed, n):
    summary, entries, data, neighbors = case(seed, n)
    for budget in (10**9, 3_000):
        walk(summary, entries, data, neighbors, budget, steps=5)


@pytest.mark.parametrize("n", [10, 100, 1000])
def test_identifier_counts_losing_a_digit(n):
    summary, entries, data, neighbors = case(n, n, with_events=0.0)
    for budget in (500, 900):
        walk(summary, entries, data, neighbors, budget, steps=8)


@pytest.mark.parametrize(
    "n, with_events, budgets",
    [
        (0, 0.7, [0, 1, 50, 10**9]),
        (200, 0.7, [0, 99, 100, 180, 181, 500, 1_500]),
        (200, 0.2, [400, 1_500, 4_000]),
        (1000, 0.0, [0, 180, 4_000, 120_000]),
        (1000, 0.05, [0, 180, 1_500]),
        (3000, 0.0, [120_000, 10**9]),
        (3000, 0.05, [500, 1_500]),
    ],
)
def test_many_entries_across_limits(n, with_events, budgets):
    summary, entries, data, neighbors = case(n + int(10 * with_events), n, with_events)
    for budget in budgets:
        same(summary, entries, data, neighbors, budget)


def test_a_rejected_record_leaves_no_reference_for_later_records():
    value = "'" + "v" * 300 + "'"

    def failed(node, path, message):
        exc = {"type": "ValueError", "message": message,
               "frames": [{"path": path, "line": 1, "statement": "f(payload)", "values": {"payload": value}}]}
        return {"node": node, "label": "target test", "status": "failed", "grade": "failure",
                "events": [{"nodeid": node, "phase": "call", "outcome": "failed", "exception": exc}]}

    entries = [failed("big", "a.py", "q" * 5000), failed("first", "b.py", "small"), failed("second", "c.py", "small")]
    summary = [START, "Evaluation completed. Task not resolved."]
    # Up to where the first small record fits only by referring to the rejected one's value.
    for budget in range(1_500):
        same(summary, entries, {"sources": {}, "_changes": {}}, 10, budget)
    text = same(summary, entries, {"sources": {}, "_changes": {}}, 10, 3_000)
    assert "a.py" not in text and "payload=[same value shown above at b.py:1]" in text


def test_a_record_at_the_recursion_limit_renders_at_the_depth_compose_renders_it():
    # Before Python 3.12 compose's comprehension is a frame: a candidate rendered outside one meets the limit later.
    def outcome(implementation, links, budget):
        frame_ = {"path": "a.py", "line": 1, "statement": "f(payload)", "values": {"payload": "1"}}
        exc = {"type": "ValueError", "message": "leaf", "frames": [frame_]}
        for _ in range(links):
            exc = {"type": "ValueError", "message": "wrap", "frames": [], "cause": exc}
        event = {"nodeid": "deep", "phase": "call", "outcome": "failed", "exception": exc}
        entry_ = {"node": "deep", "label": "target test", "status": "failed", "grade": "failure", "events": [event]}
        summary = [START, "Evaluation completed. Task not resolved."]
        try:
            return implementation(summary, [entry_], budget, {"sources": {}, "_changes": {}}, 10)
        except RecursionError:
            return RecursionError

    # The longest cause chain the oracle renders from this test's stack, then a few links either side.
    rendered, failed = 0, sys.getrecursionlimit()
    while rendered + 1 < failed:
        middle = (rendered + failed) // 2
        if outcome(_limited_output, middle, 10**9) is RecursionError:
            failed = middle
        else:
            rendered = middle
    for links in range(rendered - 3, rendered + 4):
        for budget in (0, 10**9):
            expected = outcome(_limited_output, links, budget)
            assert outcome(df._limited_output, links, budget) == expected, (links, budget)


def test_records_render_once_as_candidates_and_once_in_the_text(monkeypatch):
    summary, entries, data, neighbors = case(7, 1000, with_events=0.9)
    calls = []

    def counted(*args):
        calls.append(args)
        return _render_entry(*args)

    monkeypatch.setattr(df, "_render_entry", counted)
    text = df._limited_output(summary, entries, 120_000, data, neighbors)
    shown = text.count("\n________________ ")
    assert shown and len(calls) == sum(bool(e["events"]) for e in entries) + shown


def test_render_diagnostic_beyond_the_identifier_list(monkeypatch):
    def failed(node, k):
        frame_ = {"path": "pkg/core.py", "line": 11, "function": "helper", "statement": "value = table[idx]",
                  "source": "core", "values": {"table": VALUES[k % len(VALUES)], "idx": str(k % 5)}}
        return {"nodeid": node, "phase": "call", "outcome": "failed", "parameters": {"i": str(k)},
                "exception": {"type": "KeyError", "message": str(k % 5), "frames": [frame_]}}

    nodes = [f"tests/test_mod_{k // 100}.py::test_compute_{k // 100}[{k % 100}]" for k in range(300)]
    data = {"events": [failed(node, k) for k, node in enumerate(nodes)], "sources": {"core": {"text": CORE}},
            "complete": True}
    result = {"eval_completed": True, "resolved": False, "eval_report": {"test_status": {
        "FAIL_TO_PASS": {"failure": nodes[:3], "success": []},
        "PASS_TO_PASS": {"failure": nodes[3:], "success": []}}}}
    for budget in (2_000, 4_000):
        text = render_diagnostic(result, data, max_chars=budget)
        assert "additional test identifiers omitted" in text
        with monkeypatch.context() as patched:
            patched.setattr(df, "_limited_output", _limited_output)
            assert render_diagnostic(result, data, max_chars=budget) == text
