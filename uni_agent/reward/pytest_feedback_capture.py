"""Standalone pytest plugin copied into the evaluation container.

No uni-agent imports or model dependencies. Incremental reports survive interrupted
sessions; this plugin observes tests without changing their outcomes.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import linecache
import os
import reprlib
from pathlib import Path
from weakref import WeakKeyDictionary

import pytest

_SOURCES = {}
_STAMPS = {}
_SOURCE_HISTORY = {}
_SENT = set()
_ROOT = None
_FILE = None
_VALUE_CHARS = 4096
_ACTIVE_CONFIG = None
_ENABLED = WeakKeyDictionary()


def _emit(**row):
    if _FILE:
        try:
            with open(_FILE, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=True) + "\n")
                handle.flush()
        except (OSError, ValueError, TypeError):
            pass  # observability must not alter a test's status


def _repr(value):
    try:
        if isinstance(value, (str, bytes)) and len(value) > _VALUE_CHARS:
            return (
                repr(value[:_VALUE_CHARS]) + f" [representation shortened; {len(value) - _VALUE_CHARS} units omitted]"
            )
        if isinstance(value, (list, tuple, dict, set, frozenset)) and len(value) > 100:
            return reprlib.repr(value) + f" [representation shortened; {len(value)} elements total]"
        text = repr(value)
        if len(text) > _VALUE_CHARS:
            return text[:_VALUE_CHARS] + f" [representation shortened; {len(text) - _VALUE_CHARS} chars omitted]"
        return text
    except BaseException as exc:
        return f"[Value representation unavailable: {type(exc).__name__}]"


def _source(path):
    try:
        stat = os.stat(path)
        stamp = (stat.st_mtime_ns, stat.st_size)
    except OSError:
        stamp = None
    if path not in _SOURCES or _STAMPS[path] != stamp:
        linecache.checkcache(path)
        text = "".join(linecache.getlines(path))
        try:
            tree = ast.parse(text)
        except (SyntaxError, ValueError):
            tree = None
        digest = hashlib.sha256((path + "\0" + text).encode()).hexdigest()
        try:
            name = str(Path(path).resolve().relative_to(_ROOT))
            repo = True
        except (ValueError, OSError):
            name, repo = path, False
        if path.startswith("<"):
            name, repo = path, False
        _SOURCES[path] = (digest, text, tree, name, repo)
        _STAMPS[path] = stamp
        _SOURCE_HISTORY[digest] = {"path": name, "text": text}
    return _SOURCES[path]


def _frame(frame, lineno):
    path = frame.f_code.co_filename
    digest, source, tree, name, repo = _source(path)
    scope = None
    statement = None
    names = set()
    # Code-object arguments are available even for generated or missing source.
    try:
        args = inspect.getargvalues(frame)
        names.update(args.args)
        names.update(x for x in (args.varargs, args.keywords) if x)
    except (ValueError, TypeError):
        pass
    if tree:
        functions = [
            n
            for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.lineno <= lineno <= n.end_lineno
        ]
        scope = min(functions, key=lambda n: n.end_lineno - n.lineno) if functions else tree
        candidates = [
            n
            for n in ast.walk(scope)
            if isinstance(n, ast.stmt) and hasattr(n, "lineno") and n.lineno <= lineno <= n.end_lineno
        ]
        statement = min(candidates, key=lambda n: n.end_lineno - n.lineno) if candidates else None
        if statement:
            names |= {n.id for n in ast.walk(statement) if isinstance(n, ast.Name)}
        # Include variables used by enclosing branch conditions.
        for node in ast.walk(scope):
            if hasattr(node, "lineno") and node.lineno <= lineno <= getattr(node, "end_lineno", node.lineno):
                for attr in ("test", "iter"):
                    test = getattr(node, attr, None)
                    if test:
                        names |= {n.id for n in ast.walk(test) if isinstance(n, ast.Name)}
    lines = source.splitlines()
    start = statement.lineno if statement else lineno
    end = statement.end_lineno if statement else lineno
    text = "\n".join(lines[start - 1 : end]) if source else "[Statement/source unavailable]"
    values = {}
    for name_used in sorted(names):
        if name_used.startswith("__"):
            continue
        if name_used in frame.f_locals:
            values[name_used] = _repr(frame.f_locals[name_used])
        elif name_used in frame.f_globals:
            value = frame.f_globals[name_used]
            if isinstance(value, (str, bytes, int, float, bool, list, tuple, dict, set, frozenset, type(None))):
                values[name_used] = _repr(value)
    if not repo and source:
        # Library argument reprs are rarely useful; the raising frame's own statement remains.
        values = {n: v for n, v in values.items() if n in {"message", "msg", "value", "error", "exc"}}
    return {
        "path": name,
        "line": lineno,
        "function": getattr(frame.f_code, "co_qualname", frame.f_code.co_name),
        "start": getattr(scope, "lineno", 1),
        "end": getattr(scope, "end_lineno", len(lines)),
        "statement": text,
        "values": values,
        "source": digest,
        "repo": repo,
    }


def _exception(exc, seen=None):
    seen = set() if seen is None else seen
    if id(exc) in seen:
        return {"type": type(exc).__name__, "message": "[Exception chain cycle]", "frames": []}
    seen.add(id(exc))
    frames = []
    tb = exc.__traceback__
    while tb:
        # Keep every application/library frame; remove pytest's invocation scaffolding.
        filename = tb.tb_frame.f_code.co_filename.replace("\\", "/")
        try:
            Path(filename).resolve().relative_to(_ROOT)
            repo_frame = True
        except (ValueError, OSError):
            repo_frame = False
        if repo_frame or ("/_pytest/" not in filename and "/pluggy/" not in filename):
            frames.append(_frame(tb.tb_frame, tb.tb_lineno))
        tb = tb.tb_next
    value = {
        "type": f"{type(exc).__module__}.{type(exc).__qualname__}",
        "message": str(exc),
        "frames": frames,
        "notes": list(getattr(exc, "__notes__", ())),
    }
    if exc.__cause__ is not None:
        value["cause"] = _exception(exc.__cause__, seen)
    elif exc.__context__ is not None and not exc.__suppress_context__:
        value["context"] = _exception(exc.__context__, seen)
    branches = getattr(exc, "exceptions", ())
    if branches:
        value["branches"] = [_exception(branch, seen) for branch in branches]
    seen.remove(id(exc))
    return value


def _new_sources():
    out = {}
    for digest, source in _SOURCE_HISTORY.items():
        if digest not in _SENT:
            out[digest] = source
            _SENT.add(digest)
    return out


def pytest_configure(config):
    global _ROOT, _FILE, _VALUE_CHARS, _ACTIVE_CONFIG
    config_root = Path(str(getattr(config, "rootpath", getattr(config, "rootdir", Path.cwd())))).resolve()
    root = Path(os.environ.get("UNI_AGENT_FEEDBACK_ROOT", str(config_root))).resolve()
    try:
        config_root.relative_to(root)
        in_tree = True
    except ValueError:
        in_tree = False
    # pytester and nested pytest.main calls are tests' own observations, not this
    # evaluation. Leave their settings and reports alone and keep the parent state.
    _ENABLED[config] = _ACTIVE_CONFIG is None and in_tree
    if not _ENABLED[config]:
        return
    _ACTIVE_CONFIG = config
    _ROOT = root
    _FILE = os.environ.get("UNI_AGENT_FEEDBACK_PATH")
    _VALUE_CHARS = int(os.environ.get("UNI_AGENT_FEEDBACK_VALUE_CHARS", "4096"))
    # Full assertion details without setting CI/BUILD_NUMBER, which tests may themselves inspect.
    if root not in Path(pytest.__file__).resolve().parents:
        try:
            config.getini("verbosity_assertions")
            config._inicache["verbosity_assertions"] = 2
        except ValueError:
            config.option.verbose = max(2, config.option.verbose)
        config.option.tbstyle = "long"
    else:
        _emit(
            capture_error="Pytest belongs to the evaluated repository; reporting settings were preserved. "
            "Some assertion details may be shortened."
        )
    context = {}
    path = os.environ.get("UNI_AGENT_FEEDBACK_CONTEXT")
    if path:
        try:
            context = json.loads(Path(path).read_text())
        except (OSError, ValueError):
            _emit(capture_error="Changed-source metadata unavailable.")
    _emit(session={"complete": False, "pytest_version": pytest.__version__}, context=context)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if not _ENABLED.get(item.config):
        return
    try:
        event = {
            "nodeid": item.nodeid,
            "phase": report.when,
            "outcome": report.outcome,
            "xfail": getattr(report, "wasxfail", None),
            "identity": "collector",
        }
        if report.failed:
            event["parameters"] = {
                k: _repr(v) for k, v in getattr(getattr(item, "callspec", None), "params", {}).items()
            }
            event["text"] = report.longreprtext
            event["capture"] = "\n".join(f"{label}:\n{text}" for label, text in report.sections if text.strip())
            if call.excinfo is not None:
                event["exception"] = _exception(call.excinfo.value)
        elif report.skipped:
            event["text"] = report.longreprtext
        _emit(event=event, sources=_new_sources())
    except Exception as exc:
        _emit(
            event=event,
            sources=_new_sources(),
            capture_error=f"Diagnostic extraction failed for {item.nodeid}: {type(exc).__name__}: {exc}",
        )


@pytest.hookimpl(hookwrapper=True)
def pytest_make_collect_report(collector):
    outcome = yield
    report = outcome.get_result()
    if not _ENABLED.get(collector.config):
        return
    if report.failed:
        _emit(
            event={
                "nodeid": report.nodeid,
                "phase": "collection",
                "outcome": "failed",
                "text": report.longreprtext,
                "identity": "collector",
            }
        )


def pytest_sessionfinish(session, exitstatus):
    global _ACTIVE_CONFIG
    if not _ENABLED.get(session.config):
        return
    _emit(session={"complete": True, "exitstatus": int(exitstatus), "collected": session.testscollected})
    _ACTIVE_CONFIG = None
