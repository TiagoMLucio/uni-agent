"""Loss-aware test feedback. Pure stdlib; also usable to inspect saved evaluations.

Grading stays with the official harness. This module describes observations, never
interprets absent/skipped tests as observed failures, and accounts for every omission.
"""

from __future__ import annotations

import ast
import gzip
import hashlib
import json
import re
import shlex
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

START = "[Diagnostic test feedback begins]"
END = "[Diagnostic test feedback ends]"
DETAILS = "=================================== FAILURES / ERRORS ==================================="
OMISSIONS = "============================= omitted test output ============================="
CASE = "________________ "
SUMMARY_NOTE = (
    "[Additional source context and captured output omitted; "
    "frame values are limited to the shown statements and their dependencies.]"
)
BLOCK_HEADER = re.compile(
    r"^________________ (.+ \((?:target test|regression test|evaluation error|ungraded test)\)) ________________$", re.M
)
LOCATION = re.compile(r"^(.+?\.(?:py|pyx)):(\d+):\s*(.*)$")
CHAIN = ("During handling of the above exception", "The above exception was the direct cause")


def load_capture(text: str) -> dict:
    """Read incremental JSONL, retaining preceding events on a truncated last record."""
    data = {"events": [], "sources": {}, "complete": False, "errors": [], "context": {}}
    for line in (text or "").splitlines():
        try:
            row = json.loads(line)
        except (ValueError, TypeError):
            data["errors"].append("An incomplete or invalid capture record was ignored.")
            continue
        if not isinstance(row, dict) or any(
            row.get(key) is not None and not isinstance(row[key], dict)
            for key in ("sources", "event", "session", "context")
        ):
            data["errors"].append("A capture record with an unknown structure was ignored.")
            continue
        data["sources"].update(row.get("sources", {}))
        if row.get("event"):
            data["events"].append(row["event"])
        if "session" in row:
            data.update(row["session"])
        if "context" in row:
            data["context"] = row["context"]
        if row.get("capture_error"):
            data["errors"].append(row["capture_error"])
    return data


def junit_key(nodeid: str) -> tuple[str, str]:
    # Only split the address before the parameter suffix: parameters may contain ::.
    path, separator, rest = nodeid.partition("::")
    address, bracket, parameters = rest.partition("[")
    names = address.split("::") if separator else [path]
    module = re.sub(r"\.py$", "", path.replace("/", "."))
    return ".".join([module, *names[:-1]]), names[-1] + bracket + parameters


def _git_path(raw, strip_prefix=True):
    if raw.startswith('"'):
        try:
            raw = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            raw = raw.strip('"')
        try:
            # Git's quoted octal escapes are UTF-8 bytes, not Unicode code points.
            raw = raw.encode("latin1").decode("utf-8")
        except UnicodeError:
            pass  # core.quotepath=false can leave literal Unicode in quoted paths
    return raw[2:] if strip_prefix and raw.startswith(("a/", "b/")) else raw


def from_junit(xml: str, nodeids: list[str]) -> dict:
    data = {"events": [], "sources": {}, "complete": False, "errors": [], "context": {}}
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        data["errors"].append("JUnit report is missing, invalid, or incomplete; diagnostic coverage is unknown.")
        return data
    by_key = defaultdict(list)
    for node in nodeids:
        by_key[junit_key(node)].append(node)
    for case in root.iter("testcase"):
        matches = by_key[(case.get("classname", ""), case.get("name", ""))]
        if len(matches) > 1:
            data["errors"].append(f"Ambiguous JUnit identity {case.get('name')!r}; no guessed association.")
            continue
        node = matches[0] if matches else "::".join(filter(None, (case.get("classname"), case.get("name"))))
        capture = "\n".join(c.text or "" for c in case if c.tag in {"system-out", "system-err"})
        bad = [child for child in case if child.tag in {"failure", "error"}]
        if bad:
            for child in bad:
                message = child.get("message", "")
                phase = (
                    "call"
                    if child.tag == "failure"
                    else (
                        "collection"
                        if message.lower() == "collection failure"
                        else "teardown"
                        if "teardown" in message.lower()
                        else "setup"
                        if "setup" in message.lower()
                        else "error"
                    )
                )
                data["events"].append(
                    {
                        "nodeid": node,
                        "phase": phase,
                        "outcome": "failed",
                        "text": child.text or message,
                        "capture": capture,
                        "identity": "junit",
                    }
                )
        else:
            skip = case.find("skipped")
            data["events"].append(
                {
                    "nodeid": node,
                    "phase": "call",
                    "outcome": "skipped" if skip is not None else "passed",
                    "text": skip.get("message", "") if skip is not None else "",
                    "identity": "junit",
                }
            )
    data["complete"] = True
    data["errors"].append("JUnit fallback: phase/input values not present in the report are unavailable, not inferred.")
    return data


def _patch_hunks(patch: str) -> dict[str, list[tuple[int, int, int, int, list[str]]]]:
    out = defaultdict(list)
    path = None
    hunk = None
    for line in (patch or "").splitlines():
        if line.startswith("diff --git "):
            hunk = None
            path = None
            # Abbreviated fixture diffs may omit file headers. Real headers below
            # are authoritative and handle spaces and Git's quoted byte escapes.
            try:
                parts = shlex.split(line)
                if len(parts) == 4:
                    path = _git_path(parts[3])
            except ValueError:
                pass
        elif hunk is None and line.startswith("--- "):
            path = _git_path(line[4:])
        elif hunk is None and line.startswith("+++ "):
            if line[4:] != "/dev/null":
                path = _git_path(line[4:])
        elif line.startswith("@@ ") and path:
            m = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))?", line)
            if m:
                hunk = (int(m[1]), int(m[2] or 1), int(m[3]), int(m[4] or 1), [])
                out[path].append(hunk)
        elif hunk is not None and line[:1] in {" ", "+", "-"}:
            hunk[4].append(line)
    return dict(out)


def _flush_changed(output, removed, added, side, old_start, new_start):
    if removed or added:
        primary = added if side == "new" else removed
        anchor = new_start if side == "new" else old_start
        output.update(primary or [max(1, anchor)])
    removed.clear()
    added.clear()


def changed_lines(patch: str, side: str = "new") -> dict[str, set[int]]:
    out = defaultdict(set)
    for path, hunks in _patch_hunks(patch).items():
        for old, _, new, _, lines in hunks:
            removed, added = [], []
            old_start, new_start = old, new

            for line in lines:
                if line.startswith("+"):
                    added.append(new)
                    new += 1
                elif line.startswith("-"):
                    removed.append(old)
                    old += 1
                elif line.startswith(" "):
                    _flush_changed(out[path], removed, added, side, old_start, new_start)
                    new += 1
                    old += 1
                    old_start, new_start = old, new
            _flush_changed(out[path], removed, added, side, old_start, new_start)
    return dict(out)


def _map_line(line: int, hunks: list) -> int:
    offset = 0
    for old, count, new, new_count, body in hunks:
        if count == 0:
            # A zero-length old range names the line BEFORE an insertion.
            if line <= old:
                break
            offset = new + new_count - old - 1
            continue
        if line < old:
            break
        if line < old + count:
            old_cursor, new_cursor = old, new
            index = 0
            while index < len(body):
                if body[index].startswith(" "):
                    if line == old_cursor:
                        return max(1, new_cursor)
                    old_cursor += 1
                    new_cursor += 1
                    index += 1
                    continue
                removed = added = 0
                while index < len(body) and not body[index].startswith(" "):
                    removed += body[index].startswith("-")
                    added += body[index].startswith("+")
                    index += 1
                if old_cursor <= line < old_cursor + removed:
                    return max(1, new_cursor + min(line - old_cursor, max(0, added - 1)))
                old_cursor += removed
                new_cursor += added
            # Abbreviated fixture hunks may lack their complete body.
            return max(1, new + min(line - old, max(0, new_count - 1)))
        offset = new + new_count - old - count + (new_count == 0)
    return max(1, line + offset)


def changed_in_final(context: dict) -> dict[str, set[int]]:
    student = context.get("student_patch", "")
    out = changed_lines(student)
    gold = changed_lines(context.get("reference_patch", ""), "new" if context.get("reference_is_bug") else "old")
    hunks = _patch_hunks(student)
    ignored = set(context.get("restored_test_files", [])) | set(context.get("dropped_student_files", []))
    for path in ignored:
        out.pop(path, None)
        hunks.pop(path, None)
    renames = {}
    old_path = None
    for line in student.splitlines():
        if line.startswith("diff --git "):
            old_path = None
        elif line.startswith("rename from "):
            old_path = _git_path(line[12:], strip_prefix=False)
        elif line.startswith("rename to ") and old_path:
            renames[old_path] = _git_path(line[10:], strip_prefix=False)
    for path, lines in gold.items():
        path = renames.get(path, path)
        out.setdefault(path, set()).update(_map_line(line, hunks.get(path, [])) for line in lines)
    return out


def source_context(source: str, lineno: int, changed: set[int], neighbors: int = 10) -> tuple[str, set[str], str]:
    return _source_context(source, lineno, tuple(sorted(changed)), neighbors)


@lru_cache(maxsize=2048)
def _source_context(source: str, lineno: int, changed: tuple[int, ...], neighbors: int) -> tuple[str, set[str], str]:
    """AST spans, changed blocks with +/- neighbors, and conservative dependencies."""
    lines = source.splitlines()
    if not lines:
        return "[Source unavailable]", set(), "<unknown>"
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        keep = {
            i for n in set(changed) | {lineno} for i in range(max(1, n - neighbors), min(len(lines), n + neighbors) + 1)
        }
        return _source_lines(lines, keep, lineno, changed), set(), "<unparsed source>"
    functions, classes = [], []

    pending = [(tree, ())]
    while pending:
        node, parents = pending.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            name = (*parents, node.name)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions.append((node, ".".join(name)))
            else:
                classes.append(node)
            parents = name
        pending.extend((child, parents) for child in reversed(list(ast.iter_child_nodes(node))))
    candidates = [(n, name) for n, name in functions if n.lineno <= lineno <= n.end_lineno]
    scope, name = min(candidates, key=lambda it: it[0].end_lineno - it[0].lineno) if candidates else (tree, "<module>")
    lo, hi = getattr(scope, "lineno", 1), getattr(scope, "end_lineno", len(lines))
    nearby = {min(len(lines), max(1, n)) for n in changed}
    # Module constants, class attributes, and other changed functions can affect this
    # frame too. Do not restrict changed windows to the currently executing function.
    keep = {i for n in nearby for i in range(max(1, n - neighbors), min(len(lines), n + neighbors) + 1)}
    keep.update(range(max(lo, lineno - neighbors), min(hi, lineno + neighbors) + 1))
    for definition in [*(function for function, _ in functions), *classes]:
        start = min([definition.lineno] + [n.lineno for n in definition.decorator_list])
        if any(start <= n <= definition.end_lineno for n in nearby):
            first_line = definition.body[0].lineno if definition.body else definition.lineno + 1
            keep.update(range(start, first_line))
    body = getattr(scope, "body", [])
    if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
        first = body[0].lineno if body else lo
        keep.update(range(lo, first))  # entire multiline signature
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            doc = body[0]
            # Keep the first paragraph; fold examples unless they are changed or near the error.
            paragraph = next((i for i in range(doc.lineno, doc.end_lineno) if not lines[i - 1].strip()), doc.end_lineno)
            keep.update(range(doc.lineno, min(paragraph + 1, doc.lineno + neighbors)))
            first = doc.end_lineno + 1
        keep.update(range(first, min(hi + 1, first + neighbors)))
    nodes = []

    pending = [scope]
    while pending:
        node = pending.pop()
        nodes.append(node)
        for child in reversed(list(ast.iter_child_nodes(node))):
            if child is not scope and isinstance(
                child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
            ):
                continue
            pending.append(child)
    statements = [n for n in nodes if isinstance(n, ast.stmt) and hasattr(n, "lineno")]
    failed = [n for n in statements if n.lineno <= lineno <= n.end_lineno]
    statement = min(failed, key=lambda n: n.end_lineno - n.lineno) if failed else None
    names = {n.id for n in ast.walk(statement) if isinstance(n, ast.Name)} if statement else set()
    if statement:
        keep.update(range(statement.lineno, statement.end_lineno + 1))
    # Include transitive assignments and mutations to relevant variables, with their conditions.
    expanded = True
    while expanded:
        expanded = False
        for node in nodes:
            if isinstance(
                node, (ast.If, ast.For, ast.AsyncFor, ast.While, ast.With, ast.AsyncWith, ast.Try, ast.ExceptHandler)
            ):
                body = getattr(node, "body", [])
                if body and any(node.lineno <= n <= node.end_lineno for n in keep):
                    keep.update(range(node.lineno, body[0].lineno))
                    condition_names = set()
                    for child in (getattr(node, "test", None), getattr(node, "iter", None)):
                        if child:
                            condition_names |= {n.id for n in ast.walk(child) if isinstance(n, ast.Name)}
                    if not condition_names <= names:
                        names |= condition_names
                        expanded = True
        for node in statements:
            if node.lineno > lineno or not isinstance(
                node, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Expr, ast.Import, ast.ImportFrom)
            ):
                continue
            used = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
            targets = getattr(node, "targets", [getattr(node, "target", None)])
            assigned = {n.id for target in targets if target for n in ast.walk(target) if isinstance(n, ast.Name)}
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                assigned |= {alias.asname or alias.name.split(".")[0] for alias in node.names}
            mutation = isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
            if assigned & names or (mutation and used & names):
                keep.update(range(node.lineno, node.end_lineno + 1))
                if not used <= names:
                    names |= used
                    expanded = True
    return _source_lines(lines, keep, lineno, nearby), names, name


def _source_lines(lines, keep, failure, changed):
    out = []
    prior = None
    for n in sorted(keep):
        if not 1 <= n <= len(lines):
            continue
        if prior is not None and n > prior + 1:
            out.append(f"    [... {n - prior - 1} source lines folded ...]")
        marker = ">" if n == failure else " "
        out.append(f"{marker} {n:5}: {lines[n - 1]}")
        prior = n
    return "\n".join(out)


NDIFF = re.compile(r"^E\s{9}([ +?-]) (.*)$")
LINE_ENDING = re.compile(r"\r\n|[\n\r\v\f\x1c-\x1e\x85\u2028\u2029]")


@lru_cache(maxsize=128)
def _string_comparison(text: str):
    """Recognize a complete native string diff, checking both reconstructed sides.

    Custom explanations and incomplete diffs must not authorize removing values.
    The signs' orientation is checked against the literals, never assumed.
    """
    lines = text.splitlines()
    diffs = [(i, m) for i, line in enumerate(lines) if (m := NDIFF.match(line))]
    if not any(m[1] == "+" for _, m in diffs) or not any(m[1] == "-" for _, m in diffs):
        return None
    minus = [m[2] for _, m in diffs if m[1] in " -"]
    plus = [m[2] for _, m in diffs if m[1] in " +"]
    for i, line in enumerate(lines):
        match = re.match(r"^E\s+(?:AssertionError: )?(assert .+)$", line)
        if not match:
            continue
        try:
            statement = ast.parse(match[1]).body[0]
            comparison = statement.test
            if (
                not isinstance(statement, ast.Assert)
                or statement.msg is not None
                or not isinstance(comparison, ast.Compare)
                or len(comparison.ops) != 1
                or not isinstance(comparison.ops[0], ast.Eq)
            ):
                continue
            left = ast.literal_eval(comparison.left)
            right = ast.literal_eval(comparison.comparators[0])
        except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
            continue
        if not isinstance(left, str) or not isinstance(right, str):
            continue
        a, b = left.splitlines(), right.splitlines()
        if (minus == a and plus == b) or (minus == b and plus == a):
            # splitlines() hides terminator styles and a missing final terminator.
            # Mixed styles may differ even when both reconstructed sides match.
            endings = set(LINE_ENDING.findall(left)) | set(LINE_ENDING.findall(right))
            if len(endings) > 1 or left.endswith(tuple(endings)) != right.endswith(tuple(endings)):
                continue
            return i, (repr(left), repr(right)), tuple(j for j, _ in diffs)
    return None


def fold_assertion(text: str, context: int = 2) -> str:
    comparison = _string_comparison(text)
    if comparison is None:
        return text
    literal_index, _, diff_indices = comparison
    lines = text.splitlines()
    edits = [i for i in diff_indices if NDIFF.match(lines[i])[1] in "+-?"]
    keep = {j for i in edits for j in range(i - context, i + context + 1)}
    out, gap = [], []

    def flush():
        if gap:
            notice = f"    [... {len(gap)} unchanged comparison lines omitted ...]"
            # Tiny runs often cost less, and preserve more structure, than a notice.
            out.extend([notice] if len(gap) > 4 and len(notice) < sum(len(x) + 1 for x in gap) else gap)
            gap.clear()

    for i, line in enumerate(lines):
        m = NDIFF.match(line)
        if i == literal_index and len(line) > 240:
            flush()
            out.append("    [Duplicate expanded string values omitted; native comparison follows.]")
        elif m and m[1] == " " and i not in keep:
            gap.append(line)
        else:
            flush()
            out.append(line)
    flush()
    return "\n".join(out)


def _text_observations(text: str) -> str:
    """Short fallback: retain all E lines, complete statements, values and chain markers."""
    lines = fold_assertion(text).splitlines()
    keep = set()
    for i, line in enumerate(lines):
        if (
            line.startswith(("E ", ">"))
            or line.startswith(
                ("ImportError while loading conftest", "ERROR collecting", "Traceback (most recent call last)")
            )
            or re.match(r"^\s*(?:[\w.]+(?:Error|Exception|Exit)|ExceptionGroup):", line)
            or re.match(r'^\s*File ".+", line \d+', line)
            or any(x in line for x in CHAIN)
            or LOCATION.match(line)
            or "unchanged comparison lines omitted" in line
            or "Duplicate expanded string values omitted" in line
        ):
            keep.add(i)
        if re.match(r'^\s*File ".+", line \d+', line):
            j = i + 1
            while j < len(lines) and lines[j].startswith("    "):
                keep.add(j)
                j += 1
        if line.startswith(">"):
            # Continue the entire statement; includes expected values on continuation lines.
            j = i + 1
            while j < len(lines) and lines[j].startswith("    "):
                keep.add(j)
                j += 1
        if re.match(r"^[A-Za-z_]\w* = ", line):
            keep.add(i)
    if not keep or "ExceptionGroup" in text:
        return fold_assertion(text)  # conservative: specialized tracebacks must not disappear
    return "\n".join(lines[i] for i in sorted(keep))


def _exceptions(exc):
    if not exc:
        return
    if exc.get("cause"):
        yield from _exceptions(exc["cause"])
    elif exc.get("context"):
        yield from _exceptions(exc["context"])
    yield exc
    for branch in exc.get("branches", []):
        yield from _exceptions(branch)


def signature(event: dict) -> tuple:
    if event.get("exception"):
        parts = []
        for exc in _exceptions(event["exception"]):
            frames = exc.get("frames", [])
            loc = (frames[-1].get("path"), frames[-1].get("line")) if frames else ()
            message = re.sub(r"(?<= at )0x[0-9a-fA-F]+(?=>)", "<address>", exc.get("message", ""))
            # Assertion details are variations within a group; every case keeps its own diff.
            if exc.get("type", "").endswith("AssertionError"):
                message = "assertion mismatch"
            parts.append((exc.get("type"), loc, message, tuple(exc.get("notes", []))))
        return event.get("phase"), tuple(parts)
    text = event.get("text", "")
    locs = [m.groups() for line in text.splitlines() if (m := LOCATION.match(line)) and m[3]]
    messages = tuple(
        re.sub(r"(?<= at )0x[0-9a-fA-F]+(?=>)", "<address>", line.strip())
        for line in text.splitlines()
        if re.match(r"E\s+(?:[\w.]+(?:Error|Exception|Exit)|Failed)\b", line)
    )
    return event.get("phase"), tuple(locs), messages


def _values(event):
    rows = [f"    {k}={_clean_address(v)}" for k, v in event.get("parameters", {}).items()]
    if not rows and event.get("phase") in {"collection", "diagnostic"}:
        return ""
    if not rows and event.get("identity") != "collector":
        return "[Parameter values unavailable; reported frame values follow.]"
    return "Test inputs:\n" + "\n".join(rows) if rows else ""


def _clean_address(value):
    # Only opaque object reprs are noise. Quoted strings and nested containers
    # can contain address-like text that is actual test data.
    if re.fullmatch(r"<[^<>\n]+ at 0x[0-9a-fA-F]+>", value):
        return re.sub(r" at 0x[0-9a-fA-F]+(?=>)", "", value)
    return value


def _protect_markers(text):
    """Keep untrusted report text from becoming formatter control lines."""
    markers = {START, END, DETAILS, OMISSIONS}
    return "\n".join(
        "    " + line if line in markers or BLOCK_HEADER.fullmatch(line) else line for line in text.split("\n")
    )


def feedback_span(text):
    """Locate the outer feedback using standalone, unindented delimiters."""
    starts = list(
        re.finditer(
            r"^" + re.escape(START) + r"$\n(?=Evaluation [^\n]+\. Task (?:not )?resolved\.$)",
            text,
            re.M,
        )
    )
    # A task/trajectory can itself quote feedback. Ambiguous markers must never
    # authorize deleting other prompt components.
    if len(starts) != 1:
        return None
    start = starts[0]
    end = re.search(r"^" + re.escape(END) + r"$", text[start.end() :], re.M)
    if not end:
        return None
    return start.start(), start.end() + end.end()


def _matches_repr(value, full):
    if value == full:
        return True
    match = re.match(r"^(.*) \[representation shortened; (\d+) chars omitted\]$", value, re.S)
    if match:
        return full.startswith(match[1]) and len(full) == len(match[1]) + int(match[2])
    match = re.match(r"^(.*) \[representation shortened; (\d+) units omitted\]$", value, re.S)
    if match:
        try:
            prefix, whole = ast.literal_eval(match[1]), ast.literal_eval(full)
            return (
                isinstance(whole, (str, bytes))
                and isinstance(prefix, type(whole))
                and whole.startswith(prefix)
                and len(whole) == len(prefix) + int(match[2])
            )
        except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
            pass
    return False


@lru_cache(maxsize=128)
def _unpackings(source):
    try:
        tree = ast.parse(source)
    except (ValueError, SyntaxError, MemoryError, RecursionError):
        return []
    out = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], (ast.Tuple, ast.List))
            and all(isinstance(n, ast.Name) for n in node.targets[0].elts)
            and isinstance(node.value, ast.Name)
        ):
            out.append((node.lineno, node.value.id, tuple(n.id for n in node.targets[0].elts)))
    return out


def _comparison_bindings(statement, comparison):
    """Bind native expanded literals to plain names in the failing assertion."""
    if comparison is None:
        return {}
    try:
        node = ast.parse(statement.strip()).body[0]
        expr = node.test
        if not isinstance(node, ast.Assert) or not isinstance(expr, ast.Compare) or len(expr.ops) != 1:
            return {}
        operands = (expr.left, expr.comparators[0])
        return {
            operand.id: value
            for operand, value in zip(operands, comparison[1], strict=True)
            if isinstance(operand, ast.Name)
        }
    except (ValueError, SyntaxError, AttributeError, MemoryError, RecursionError):
        return {}


def _statement_excerpt(source, lineno, statement):
    lines, pieces = source.splitlines(), statement.splitlines()
    for start in range(max(0, lineno - len(pieces)), min(lineno, len(lines))):
        if lines[start : start + len(pieces)] == pieces:
            return _source_lines(lines, set(range(start + 1, start + len(pieces) + 1)), lineno, set())
    return "> " + statement.replace("\n", "\n  ")


def _reuse_source(code, source_id, shown):
    out, omitted = [], 0
    for line in code.splitlines():
        match = re.match(r"^[>* ]+\s+(\d+):", line)
        if match:
            key = (source_id, int(match[1]))
            if key in shown and not line.startswith(">"):
                omitted += 1
                continue
            shown.add(key)
        if omitted:
            out.append(f"    [... {omitted} source lines already shown above ...]")
            omitted = 0
        out.append(line)
    if omitted:
        out.append(f"    [... {omitted} source lines already shown above ...]")
    return "\n".join(out)


def render_event(
    event: dict, data: dict, detailed: bool, neighbors=10, *, shared=None, include_inputs=True, minimal=False
) -> str:
    out = [_values(event)] if include_inputs else []
    sources = data.get("sources", {})
    changes = data.get("_changes")
    if changes is None:
        changes = changed_in_final(data.get("context", {}))
    e_lines = [line for line in event.get("text", "").splitlines() if line.startswith("E ")]
    shared = {"values": {}, "frames": {}, "sources": set()} if shared is None else shared
    changed_sources = shared.setdefault("changed_sources", set())
    prior_values = shared["values"]
    parameter_values = set(event.get("parameters", {}).items())
    comparison = _string_comparison("\n".join(e_lines))

    def render_exception(exc, root=False):
        if exc.get("cause"):
            render_exception(exc["cause"])
            out.append("\nThe above exception was the direct cause of the following exception:\n")
        elif exc.get("context"):
            render_exception(exc["context"])
            out.append("\nDuring handling of the above exception, another exception occurred:\n")
        frames = exc.get("frames", [])
        repo = [i for i, f in enumerate(frames) if f.get("repo")]
        important = {0, len(frames) - 1, *(repo[-1:] if repo else [])}
        for i, frame in enumerate(frames):
            if minimal and i not in important:
                continue
            source_id = frame.get("source")
            source = sources.get(source_id, {}).get("text", "")
            path, line = frame.get("path", "?"), frame.get("line", 0)
            statement = frame.get("statement", "[Statement unavailable]")
            affected = changes.get(path, set())
            is_changed = any(frame.get("start", line) <= n <= frame.get("end", line) for n in affected)
            out.append(f"\n{path}:{line}: in {frame.get('function') or '<unknown>'}")
            vals = frame.get("values", {})
            bindings = _comparison_bindings(statement, comparison) if root else {}
            full_values = {
                name: bindings[name] if name in bindings and _matches_repr(value, bindings[name]) else value
                for name, value in vals.items()
            }
            wrappers = set()
            for start, wrapper, components in _unpackings(source):
                if start <= line and wrapper in vals and all(name in full_values for name in components):
                    reconstructed = "(" + ", ".join(full_values[name] for name in components)
                    reconstructed += ("," if len(components) == 1 else "") + ")"
                    if _matches_repr(vals[wrapper], reconstructed):
                        wrappers.add(wrapper)
            value_rows = []
            relevant_values = set(vals)
            if minimal:
                try:
                    relevant_values = {
                        node.id for node in ast.walk(ast.parse(statement.strip())) if isinstance(node, ast.Name)
                    }
                except (SyntaxError, ValueError, MemoryError, RecursionError):
                    pass  # unavailable syntax cannot justify guessing which inputs matter
                if source:
                    _, dependencies, _ = source_context(source, line, set(), neighbors=0)
                    relevant_values |= dependencies
            for name, value in vals.items():
                if name not in relevant_values:
                    continue
                if name in wrappers or (name == "self" and re.fullmatch(r"<[^<>]+ object at 0x[0-9a-fA-F]+>", value)):
                    continue
                if len(value) > 180 and name in bindings and _matches_repr(value, bindings[name]):
                    value_rows.append(f"    {name}=[shown in the assertion comparison below]")
                elif (name, value) in parameter_values:
                    continue
                elif (name, value) in prior_values:
                    value_rows.append(f"    {name}=[same value shown above at {prior_values[(name, value)]}]")
                else:
                    value_rows.append(f"    {name}={_clean_address(value)}")
                    if len(value) > 180:
                        prior_values[(name, value)] = f"{path}:{line}"
            if value_rows:
                out.append("Values at failure:\n" + "\n".join(value_rows))
            frame_key = (source_id, path, line, statement)
            need_changed = bool(affected) and source_id not in changed_sources
            if minimal:
                # Do not register source context we did not emit. A later expanded
                # record must still print it before referring to it.
                out.append(_statement_excerpt(source, line, statement))
            elif source and (need_changed or (detailed and (i in important or is_changed))):
                code, _, _ = source_context(source, line, affected, neighbors)
                out.append(_reuse_source(code, source_id, shared["sources"]))
                if affected and source:
                    changed_sources.add(source_id)
            elif frame_key in shared["frames"]:
                out.append(f"[Source and failing statement shown above at {path}:{line}.]")
            else:
                out.append(_statement_excerpt(source, line, statement))
            if not minimal:
                shared["frames"][frame_key] = f"{path}:{line}"
        kind = exc.get("type", "Exception").removeprefix("builtins.")
        if root and e_lines:
            out.append(fold_assertion("\n".join(e_lines)))
            # Some custom reports contain E lines without an exception message.
            if not kind.endswith("AssertionError") and exc.get("message", "") not in "\n".join(e_lines):
                out.append(f"{kind}: {exc.get('message', '')}")
        else:
            out.append(f"{kind}: {exc.get('message', '')}")
        if frames:
            out.append(f"{frames[-1].get('path', '?')}:{frames[-1].get('line', 0)}: {kind}")
        for note in exc.get("notes", []):
            out.append("Note: " + note)
        for i, branch in enumerate(exc.get("branches", []), 1):
            out.append(f"\nException group branch {i}:")
            render_exception(branch)

    if event.get("exception"):
        render_exception(event["exception"], root=True)
    else:
        out.append(fold_assertion(event.get("text", "")) if detailed else _text_observations(event.get("text", "")))
    capture = event.get("capture", "")
    if capture and not minimal:
        out.append("----------------------------- Captured output -----------------------------")
        lines = capture.splitlines()
        i = 0
        while i < len(lines):
            j = i + 1
            while j < len(lines) and lines[j] == lines[i]:
                j += 1
            out.append("    " + lines[i] + (f" [repeated {j - i} consecutive times]" if j - i > 1 else ""))
            i = j
    if any(x in event.get("text", "") for x in ("Full output truncated", "Use -v to get", "use '-vv'")):
        out.append("[Assertion output was shortened in the report; missing content is unavailable.]")
    if minimal:
        out.append(SUMMARY_NOTE)
    return _protect_markers("\n".join(filter(None, out)))


def _status(event):
    if event.get("xfail"):
        if event.get("outcome") == "skipped":
            return "xfail"
        if event.get("outcome") == "passed":
            return "xpass"
    return event.get("outcome", "unknown")


def _observed_status(observations):
    if any(e.get("outcome") == "failed" for e in observations):
        return "failed"
    for status in ("xfail", "skipped", "xpass"):
        if any(_status(e) == status for e in observations):
            return status
    if any(e.get("phase") == "call" and e.get("outcome") == "passed" for e in observations):
        return "passed"
    return "not observed"


def _phase_heading(event):
    phase = event.get("phase")
    return {
        "call": "FAILED",
        "setup": "ERROR at setup",
        "teardown": "ERROR at teardown",
        "collection": "ERROR collecting tests",
        "diagnostic": "Evaluation error",
    }.get(phase, "ERROR (execution stage unavailable)")


def _distinct_reports(events):
    reports, counts = {}, Counter()
    for event in events:
        key = json.dumps(event, sort_keys=True)
        reports.setdefault(key, event)
        counts[key] += 1
    return [{**event, "_repeat_count": counts[key]} for key, event in reports.items()]


def _entry_description(entry):
    detail = entry["status"]
    phases = list(dict.fromkeys(_phase_heading(e) for e in entry["events"]))
    if phases and phases != ["FAILED"]:
        detail += "; " + ", ".join(phases)
    if entry.get("grade") and (
        (entry["status"] == "passed" and entry["grade"] == "failure")
        or (entry["status"] == "failed" and entry["grade"] == "success")
    ):
        detail += "; official grade: " + entry["grade"]
    return f"{entry['node']}: {detail} ({entry['label']})"


def _omitted_output(entries):
    if not entries:
        return ""
    noun = "Failure output" if all(e["status"] == "failed" for e in entries) else "Test details"
    return "\n".join(
        [
            OMISSIONS,
            f"{noun} not shown for these {len(entries)} tests:",
            *("- " + _entry_description(e) for e in entries),
            "These details were omitted from this output; this does not change the test outcomes.",
        ]
    )


def _render_entry(entry, data, detailed, neighbors, shared):
    out = [f"{CASE}{entry['node']} ({entry['label']}) ________________"]
    status, grade = entry["status"], entry.get("grade")
    if (status == "passed" and grade == "failure") or (status == "failed" and grade == "success"):
        out.append(f"Observed: {status}; official grade: {grade}.")
    if not entry["events"]:
        out.append("Observed: " + status + ".")
        if entry.get("reason"):
            out.append(_protect_markers(entry["reason"]))
        if status == "not observed":
            out.append("No adequate execution record; this does not establish that the test passed or failed.")
        return "\n".join(out)
    last_parameters = None
    for event in entry["events"]:
        out.append(_phase_heading(event))
        if event.get("_repeat_count", 1) > 1:
            out.append(f"[Identical failure report repeated {event['_repeat_count']} times.]")
        parameters = event.get("parameters", {})
        out.append(
            render_event(
                event,
                data,
                detailed is True,
                neighbors,
                shared=shared,
                include_inputs=parameters != last_parameters,
                minimal=detailed == "summary",
            )
        )
        last_parameters = parameters
    return "\n".join(out)


def _has_execution_blocker(result, events):
    """Positive evidence for grouping missing observations, never inferred failure."""
    if result.get("eval_error") or result.get("patch_apply_failed"):
        return True
    for event in events:
        if event.get("outcome") != "failed":
            continue
        if event.get("phase") == "collection":
            return True
        if event.get("phase") == "diagnostic" and re.search(
            r"ImportError while loading conftest|^\s*ERROR collecting\b",
            event.get("text", ""),
            re.M,
        ):
            return True
    return False


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


def render_diagnostic(result: dict, data: dict, *, max_chars=80000, neighbors=10, patch=None, output="") -> str:
    context = dict(data.get("context", {}))
    if result.get("patch_apply_failed"):
        context["student_patch"] = ""
    data = {**data, "_changes": changed_in_final(context)}
    events = list(data.get("events", []))
    if not events and output.strip():
        raw_output = output
        markers = list(re.finditer(r"^>>>>> (Start|End) Test Output\r?$", output, re.M))
        if len(markers) == 2 and [m[1] for m in markers] == ["Start", "End"]:
            test_output = output[markers[0].end() : markers[1].start()].strip()
            if test_output:
                # Official harness boundaries separate pytest diagnostics from
                # unrelated checkout/patch-application chatter. Ambiguous or
                # missing boundaries cannot justify removing raw evidence.
                raw_output = test_output
        events.append(
            {
                "nodeid": "<unattributed evaluator output>",
                "phase": "diagnostic",
                "outcome": "failed",
                "text": raw_output,
                "identity": "unparsed",
            }
        )
    by_node = defaultdict(list)
    for event in events:
        by_node[event.get("nodeid", "<unknown>")].append(event)
    ts = (result.get("eval_report") or {}).get("test_status") or {}
    state = "completed" if result.get("eval_completed") and data.get("complete") else "incomplete"
    if result.get("patch_apply_failed"):
        state = "patch application failed; results do not evaluate the submitted patch"
    summary = [START, f"Evaluation {state}. Task {'resolved' if result.get('resolved') else 'not resolved'}."]
    if patch is not None and not patch.strip():
        summary.append("Submission: no code changes (empty patch).")
    if result.get("eval_error"):
        summary.append("Evaluation error: " + str(result["eval_error"]))
    discarded = set(context.get("restored_test_files", [])) | set(context.get("dropped_student_files", []))
    discarded &= set(_patch_hunks(context.get("student_patch", "")))
    if discarded:
        summary.append("Student file edits discarded by evaluation: " + ", ".join(sorted(discarded)))
    if not ts:
        summary.append("Official per-test grading unavailable; the following are ungraded observations.")
    if data.get("exitstatus") not in {None, 0, 1}:
        summary.append(
            f"Test session exit status: {data['exitstatus']}; {data.get('collected', 'unknown')} tests collected."
        )

    entries, graded = [], set()
    status_names = {"xfail": "expected failure (xfail)", "xpass": "unexpected pass (xpass)"}
    for title, label, category in [
        ("Target tests", "target test", "FAIL_TO_PASS"),
        ("Previously passing tests", "regression test", "PASS_TO_PASS"),
    ]:
        buckets = ts.get(category, {})
        nodes = list(dict.fromkeys([*buckets.get("failure", []), *buckets.get("success", [])]))
        counts = Counter()
        for node in nodes:
            if node in graded:
                continue
            graded.add(node)
            observations = by_node.get(node, [])
            status = _observed_status(observations)
            counts[status] += 1
            bad = _distinct_reports([e for e in observations if e.get("outcome") == "failed"])
            grade = "failure" if node in buckets.get("failure", []) else "success"
            if bad or status != "passed" or grade == "failure":
                reasons = list(dict.fromkeys(e.get("text", "") for e in observations if e.get("text")))
                entries.append(
                    {
                        "node": node,
                        "label": label,
                        "status": status,
                        "grade": grade,
                        "events": bad,
                        "reason": "\n".join(reasons) if not bad else "",
                    }
                )
        if counts:
            summary.append(
                title
                + ": "
                + ", ".join(
                    f"{counts[s]} {status_names.get(s, s)}"
                    for s in ("failed", "passed", "skipped", "xfail", "xpass", "not observed")
                    if counts[s]
                )
                + "."
            )
    for node, observations in by_node.items():
        if node in graded:
            continue
        relevant = [
            e
            for e in observations
            if e.get("outcome") == "failed" and (not ts or e.get("phase") in {"collection", "diagnostic"})
        ]
        if relevant:
            entries.append(
                {
                    "node": node,
                    "label": "evaluation error" if ts else "ungraded test",
                    "status": "failed",
                    "events": _distinct_reports(relevant),
                }
            )
    summary.extend("Diagnostic warning: " + str(e) for e in dict.fromkeys(data.get("errors", [])))
    if _has_execution_blocker(result, events):
        unobserved = [entry for entry in entries if entry["status"] == "not observed"]
        if unobserved:
            total = len(unobserved)
            summary.append(f"Names of these {total:,} unobserved tests are omitted; no pass/fail outcome is inferred.")
            entries = [entry for entry in entries if entry["status"] != "not observed"]
    summary[1:] = [_protect_markers(item) for item in summary[1:]]

    # Similar signatures order the records and pick which test is shown in full when not all fit.
    # Each test keeps its own inputs and errors; no public group/case catalogue or assertion of
    # one common defect.
    groups = defaultdict(list)
    for i, entry in enumerate(entries):
        key = (entry["label"], tuple(signature(e) for e in entry["events"]), entry["status"])
        groups[key].append(i)

    def priority(index):
        entry = entries[index]
        if any(e.get("phase") in {"collection", "diagnostic"} for e in entry["events"]):
            return 0
        return 1 if entry["events"] else 2

    representatives = sorted((group[0] for group in groups.values()), key=priority)
    rest = sorted((i for group in groups.values() for i in group[1:]), key=priority)
    order = sorted(representatives + rest, key=priority)

    def compose(selection):
        shared = {"values": {}, "frames": {}, "sources": set()}
        blocks = [_render_entry(entries[i], data, selection[i], neighbors, shared) for i in order if i in selection]
        omitted = _omitted_output([entries[i] for i in order if i not in selection])
        return "\n\n".join(filter(None, ["\n".join(summary), DETAILS if blocks else "", *blocks, omitted, END]))

    chosen = {}
    text = compose(chosen)
    if len(text) > max_chars:
        return _limited_output(summary, [entries[i] for i in order], max_chars, data, neighbors)

    def take(candidate):
        nonlocal chosen, text
        proposal = compose(candidate)
        if len(proposal) <= max_chars:
            chosen, text = candidate, proposal
            return True
        return False

    # Every test in full when it fits.
    if take({i: True for i in order}):
        return text
    # Else one record per distinct failure, full or else without source windows; else as many full
    # records as fit, startup and collection errors first, then the smallest. Every other test then
    # gets a short record while they fit, the rest are named; with no full record fitting, that
    # leaves as many short records as fit.
    if not (take({i: True for i in representatives}) or take({i: False for i in representatives})):
        def full_size(i):
            return len(_render_entry(entries[i], data, True, neighbors, {"values": {}, "frames": {}, "sources": set()}))

        for i in sorted(representatives, key=lambda i: (priority(i), full_size(i))):
            take({**chosen, i: True})
    for i in order:
        if i not in chosen:
            take({**chosen, i: "summary"})
    return text


def shrink_feedback(text: str, max_chars: int) -> str:
    """Remove complete trailing blocks; earlier source/value references remain valid."""
    if len(text) <= max_chars or START not in text:
        return text
    details_marker = re.search(r"^" + re.escape(DETAILS) + r"$", text, re.M)
    if not details_marker:
        return text
    # Captures never supply the shared source/value references. Removing them
    # first keeps more complete failure records without invalidating references.
    text = re.sub(
        r"^----------------------------- Captured output -----------------------------\n(?:^    .*\n?)*",
        "[Captured output omitted to keep failure details.]\n",
        text,
        flags=re.M,
    )
    if len(text) <= max_chars:
        return text
    summary, tail = text[: details_marker.start()], text[details_marker.end() :]
    omission_marker = re.search(r"^" + re.escape(OMISSIONS) + r"$", tail, re.M)
    end_marker = re.search(r"^" + re.escape(END) + r"$", tail, re.M)
    details = tail[: omission_marker.start() if omission_marker else end_marker.start() if end_marker else len(tail)]
    matches = list(BLOCK_HEADER.finditer(details))
    blocks = [
        details[m.start() : matches[i + 1].start() if i + 1 < len(matches) else len(details)].strip()
        for i, m in enumerate(matches)
    ]
    previous_omissions = []
    unlisted = 0
    if omission_marker:
        omitted = tail[omission_marker.end() : end_marker.start() if end_marker else len(tail)]
        previous_omissions = [line for line in omitted.splitlines() if line.startswith("- ")]
        unlisted_match = re.search(r"(\d+) additional test identifiers omitted\.", omitted)
        if unlisted_match:
            unlisted = int(unlisted_match[1])
    for count in range(len(blocks), -1, -1):
        missing = []
        for match, block in zip(matches[count:], blocks[count:], strict=True):
            observation = re.search(r"^Observed: (.+)\.$", block, re.M)
            phases = list(
                dict.fromkeys(
                    re.findall(
                        r"^(?:FAILED|ERROR at setup|ERROR at teardown|ERROR collecting tests|Evaluation error)$",
                        block,
                        re.M,
                    )
                )
            )
            status = (
                observation[1]
                if observation
                else ("failed" + ("; " + ", ".join(phases) if phases and phases != ["FAILED"] else ""))
            )
            missing.append("- " + match[1] + ": " + status + "; details omitted")
        missing += previous_omissions
        # The tiny-limit renderer already permits counted identifier omissions.
        # Remove more supporting names before discarding the error it preserved.
        identifier_counts = range(len(missing), -1, -1) if unlisted else (len(missing),)
        for shown in identifier_counts:
            omitted_identifiers = unlisted + len(missing) - shown
            footer = (
                "\n".join(
                    [
                        OMISSIONS,
                        f"Test details not shown for these {len(missing) + unlisted} tests:",
                        *missing[:shown],
                        *(
                            [f"{omitted_identifiers} additional test identifiers omitted."]
                            if omitted_identifiers
                            else []
                        ),
                        "These details were omitted from this output; this does not change the test outcomes.",
                    ]
                )
                if missing or unlisted
                else ""
            )
            reduced = "\n\n".join(
                filter(
                    None,
                    [summary.strip(), DETAILS if count else "", *blocks[:count], footer, END],
                )
            )
            if len(reduced) <= max_chars:
                return reduced
    # The identifiers cannot fit. Let the caller shrink other prompt components
    # rather than silently truncate IDs or leave references to deleted evidence.
    return text


def fit_message_feedback(messages: list[dict], max_feedback_chars: int) -> list[dict]:
    out = []
    for message in messages:
        content = message.get("content", "")
        span = feedback_span(content) if isinstance(content, str) else None
        if span:
            a, b = span
            content = content[:a] + shrink_feedback(content[a:b], max_feedback_chars) + content[b:]
        out.append({**message, "content": content})
    return out


def save_evidence(data: dict, directory: str | None, instance_id: str):
    if directory:
        dest = Path(directory)
        dest.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(instance_id.encode()).hexdigest()[:12]
        from uuid import uuid4

        with gzip.open(dest / f"{digest}-{uuid4().hex[:8]}.json.gz", "wt", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=True)
