"""End-to-end tests for the str_replace_editor script's str_replace matching."""
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).parents[2] / "uni_agent" / "tools" / "str_replace_editor" / "str_replace_editor"


def run_edit(path: Path, old: str, new: str = "REPLACED"):
    return subprocess.run(
        [sys.executable, str(SCRIPT), "str_replace", "--path", str(path),
         "--old_str", old, "--new_str", new],
        capture_output=True, text=True, timeout=30,
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
