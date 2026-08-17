"""End-to-end tests for the str_replace_editor script's str_replace matching."""
import os
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).parents[2] / "uni_agent" / "tools" / "str_replace_editor" / "str_replace_editor"


def run_edit(path: Path, old: str, new: str = "REPLACED", explain: bool = True):
    env = {**os.environ, "EXPLAIN_EDIT_FAILURES": "true" if explain else "false"}
    return subprocess.run(
        [sys.executable, str(SCRIPT), "str_replace", "--path", str(path),
         "--old_str", old, "--new_str", new],
        capture_output=True, text=True, timeout=30, env=env,
    ).stdout


def run_create(path: Path, file_text: str, explain: bool = True):
    env = {**os.environ, "EXPLAIN_EDIT_FAILURES": "true" if explain else "false"}
    return subprocess.run(
        [sys.executable, str(SCRIPT), "create", "--path", str(path),
         "--file_text", file_text],
        capture_output=True, text=True, timeout=30, env=env,
    ).stdout


def write(tmp_path: Path, content: str) -> Path:
    f = tmp_path / "target.py"
    f.write_text(content)
    return f


def test_exact_match_edits(tmp_path):
    f = write(tmp_path, "a = 1\nb = 2\nc = 3\n")
    out = run_edit(f, "b = 2", "b = 20")
    assert "has been edited" in out
    assert f.read_text() == "a = 1\nb = 20\nc = 3\n"


def test_surrounding_newlines_are_forgiven(tmp_path):
    f = write(tmp_path, "a = 1\nb = 2\nc = 3\n")
    out = run_edit(f, "\nb = 2\n", "b = 20")
    assert "has been edited" in out
    assert f.read_text() == "a = 1\nb = 20\nc = 3\n"


def test_outer_strip_fallback(tmp_path):
    f = write(tmp_path, "a = 1\nb = 2\nc = 3\n")
    out = run_edit(f, "  b = 2  ", "b = 20")
    assert "has been edited" in out
    assert f.read_text() == "a = 1\nb = 20\nc = 3\n"


def test_no_match_error(tmp_path):
    f = write(tmp_path, "a = 1\n")
    out = run_edit(f, "z = 9")
    assert "did not appear verbatim" in out
    assert f.read_text() == "a = 1\n"


def test_multiple_occurrences_error_with_lines(tmp_path):
    f = write(tmp_path, "x = 0\ny = 1\nx = 0\n")
    out = run_edit(f, "x = 0")
    assert "Multiple occurrences" in out
    assert "[1, 3]" in out
    assert f.read_text() == "x = 0\ny = 1\nx = 0\n"


def test_same_string_guard(tmp_path):
    f = write(tmp_path, "a = 1\n")
    out = run_edit(f, "a = 1", "a = 1")
    assert "is the same as new_str" in out
    assert f.read_text() == "a = 1\n"


def test_single_line_indentation_forgiven_by_strip_fallback(tmp_path):
    f = write(tmp_path, "def f():\n\tx = 1\n\ty = 2\n")
    out = run_edit(f, "        x = 1", "x = 10")
    assert "has been edited" in out
    assert f.read_text() == "def f():\n\tx = 10\n\ty = 2\n"


def test_tabs_are_no_longer_normalized(tmp_path):
    f = write(tmp_path, "def f():\n\tx = 1\n\ty = 2\n")
    # a mid-block whitespace mismatch is not forgiven: spaces vs the file's tab
    out = run_edit(f, "def f():\n        x = 1", "def f():\n        x = 10")
    assert "did not appear verbatim" in out
    out = run_edit(f, "def f():\n\tx = 1", "def f():\n\tx = 10")
    assert "has been edited" in out
    # untouched tab lines keep their tabs (no expandtabs rewrite of the file)
    assert f.read_text() == "def f():\n\tx = 10\n\ty = 2\n"


def test_deletion_with_empty_new_str(tmp_path):
    f = write(tmp_path, "a = 1\nb = 2\n")
    out = subprocess.run(
        [sys.executable, str(SCRIPT), "str_replace", "--path", str(f),
         "--old_str", "b = 2\n", "--new_str", ""],
        capture_output=True, text=True, timeout=30,
    ).stdout
    assert "has been edited" in out
    # the surrounding-newline pre-strip means deleting a whole line leaves its newline
    assert f.read_text() == "a = 1\n\n"


def test_snippet_in_success_message(tmp_path):
    f = write(tmp_path, "\n".join(f"line_{i} = {i}" for i in range(30)) + "\n")
    out = run_edit(f, "line_15 = 15", "line_15 = 150")
    assert "a snippet of" in out
    assert "line_15 = 150" in out


# --- failure diagnosis (EXPLAIN_EDIT_FAILURES) ---

FUNC = (
    "import numpy as np\n"
    "\n"
    "\n"
    "def _cstack(left, right):\n"
    '    """Function corresponding to the & operation."""\n'
    "    noutp = _compute_n_outputs(left, right)\n"
    "    cright = np.zeros((noutp, right.shape[1]))\n"
    "    return np.hstack([cleft, cright])\n"
)


def indent(text: str, spaces: int) -> str:
    return "\n".join(" " * spaces + line if line.strip() else line for line in text.splitlines())


def test_extra_indent_diagnosed(tmp_path):
    # the astropy-12907 failure: a module-level function reproduced with method-level indent
    f = write(tmp_path, FUNC)
    old = indent(FUNC[FUNC.index("def _cstack"):].rstrip("\n"), 4)
    out = run_edit(f, old)
    assert "did not appear verbatim" in out
    assert "lines 4-8 match exactly except" in out
    assert "4 extra leading space(s)" in out
    assert "noutp = _compute_n_outputs" not in out  # no echo of the body
    assert f.read_text() == FUNC


def test_missing_indent_diagnosed(tmp_path):
    f = write(tmp_path, "class A:\n    def m(self):\n        x = 1\n        return x\n")
    out = run_edit(f, "def m(self):\n    x = 1\n    return x")
    assert "match exactly except" in out
    assert "4 missing leading space(s)" in out


def test_literal_backslash_n_diagnosed(tmp_path):
    # the pypika/safety failure class: JSON escaping collapsed real newlines into `\n`
    f = write(tmp_path, 'click.secho(f"The Safety policy file " \\\n            "was parsed")\n')
    out = run_edit(f, 'click.secho(f"The Safety policy file " \\\\n            "was parsed")')
    assert "did not appear verbatim" in out
    assert "`\\n`" in out
    assert "collapsed your newlines" in out


def test_trailing_whitespace_in_old_diagnosed(tmp_path):
    f = write(tmp_path, "alpha = 1\nbeta = 2\ngamma = 3\n")
    out = run_edit(f, "alpha = 1  \nbeta = 2\t\ngamma = 3")
    assert "match exactly except" in out
    assert "trailing whitespace" in out


def test_tab_vs_space_indent_diagnosed(tmp_path):
    f = write(tmp_path, "def f():\n\tx = 1\n\ty = 2\n\treturn x + y\n")
    out = run_edit(f, "def f():\n    x = 1\n    y = 2\n    return x + y")
    assert "did not appear verbatim" in out
    assert "line-edge whitespace differs" in out
    assert "tabs" in out


def test_stale_line_shows_closest_window_diff(tmp_path):
    f = write(tmp_path, "def compute_total(values):\n    total = sum(values)\n    return total * 2\n")
    out = run_edit(f, "def compute_total(values):\n    total = sum(vals)\n    return total * 2")
    assert "Closest match: lines 1-3" in out
    assert "file has `total = sum(values)`" in out
    assert "your old_str has `total = sum(vals)`" in out


def test_no_similar_region_stays_short(tmp_path):
    f = write(tmp_path, "a = 1\nb = 2\n")
    secret = "completely_unrelated_text_that_should_not_be_echoed"
    out = run_edit(f, f"{secret} = 9\nmore({secret})")
    assert "did not appear verbatim" in out
    assert "No similar region found" in out
    assert out.count(secret) <= 1  # only the compact `starting ...` header may quote line 1


def test_failure_header_describes_without_echo(tmp_path):
    f = write(tmp_path, "x = 1\n")
    body_marker = "unique_body_line_marker"
    old = "first_line = 0\n" + "\n".join(f"{body_marker}_{i} = {i}" for i in range(20))
    out = run_edit(f, old)
    assert "old_str (21 lines," in out
    assert "starting `first_line = 0`" in out
    assert body_marker not in out


def test_multiple_occurrences_without_echo(tmp_path):
    f = write(tmp_path, "x = 0\ny = 1\nx = 0\n")
    out = run_edit(f, "x = 0")
    assert "Multiple occurrences" in out
    assert "[1, 3]" in out
    assert "surrounding context" in out


def test_probe_never_claims_ambiguous_match(tmp_path):
    # dedenting matches two identical regions: the probe must not pick one and assert certainty
    region = "def dup():\n    return 1\n"
    f = write(tmp_path, region + "\n" + region)
    out = run_edit(f, indent(region.rstrip("\n"), 4))
    assert "match exactly except" not in out


def test_legacy_messages_with_flag_off(tmp_path):
    f = write(tmp_path, "a = 1\n")
    out = run_edit(f, "z = 9", explain=False)
    assert "old_str `z = 9` did not appear verbatim" in out
    out = run_edit(f, "a = 1", "a = 1", explain=False)
    assert "is the same as new_str `a = 1`" in out


# --- syntax warnings on written .py content ---

def test_create_broken_python_warns(tmp_path):
    out = run_create(tmp_path / "repro.py", "x = call(a,\\n    b)\n")
    assert "File created successfully" in out
    assert "does not parse as Python" in out


def test_create_valid_python_no_warning(tmp_path):
    out = run_create(tmp_path / "ok.py", "x = 1\n")
    assert "File created successfully" in out
    assert "does not parse" not in out


def test_edit_introducing_syntax_error_warns(tmp_path):
    f = write(tmp_path, "def f():\n    return 1\n")
    out = run_edit(f, "    return 1", "    return (1")
    assert "has been edited" in out
    assert "introduced a Python syntax error" in out


def test_edit_on_already_broken_file_does_not_warn(tmp_path):
    f = write(tmp_path, "def f(:\n    return 1\n")
    out = run_edit(f, "    return 1", "    return 2")
    assert "has been edited" in out
    assert "syntax error" not in out


def test_non_python_files_never_lint(tmp_path):
    f = tmp_path / "notes.txt"
    f.write_text("some ( unbalanced\n")
    out = run_edit(f, "some ( unbalanced", "still ( unbalanced")
    assert "has been edited" in out
    assert "syntax" not in out
