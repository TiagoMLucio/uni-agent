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


def test_consecutive_repeats_get_the_periodic_message(tmp_path):
    """When every occurrence's surroundings are identical (pasted duplicates whose
    repetition extends past the first and last match), the enumeration is noise; the
    message must say so and describe the spanning edit."""
    period = "a = 1\nb = 2\nc = 3\nd = 4\ne = 5\nf = 6\n"
    content = "c = 3\nd = 4\ne = 5\nf = 6\n" + period * 4 + "a = 1\nb = 2\n"
    f = write(tmp_path, content)
    out = run_edit(f, "b = 2\nc = 3", "b = 20\nc = 3")
    assert "Multiple occurrences" in out, "gallery marker stays"
    assert "surrounding lines are identical" in out
    assert "covers the whole repeated region" in out
    assert out.count("occurrence at line") == 1, "one copy shown, not the enumeration"


def test_distinct_surroundings_keep_the_enumeration(tmp_path):
    f = write(tmp_path, "x = 1\nb = 2\ny = 3\nz = 4\nq = 5\nw = 6\nb = 2\nr = 7\n")
    out = run_edit(f, "b = 2", "b = 20")
    assert "Each match with its surrounding lines" in out
    assert out.count("occurrence at line") == 2


def test_empty_old_str_gets_its_own_message(tmp_path):
    """An empty (or whitespace-only) old_str matches everywhere; it must get the insertion
    guidance, never the multiple-occurrence enumeration of the whole file."""
    f = write(tmp_path, "a = 1\nb = 2\nc = 3\n")
    for old in ("", "\n", "   "):
        out = run_edit(f, old, "x = 0")
        assert "old_str is empty" in out
        assert "Multiple occurrences" not in out
    assert f.read_text() == "a = 1\nb = 2\nc = 3\n", "file untouched"


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
    assert "byte-identical" in out and "leave the file unchanged" in out
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
    assert "extend your old_str with the neighbouring lines" in out
    assert "occurrence at line 1:" in out and "occurrence at line 3:" in out


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


# --- did-you-mean suggestion (STR_REPLACE_DID_YOU_MEAN=true) ------------------------------

def run_edit_dym(path: Path, old: str, dym: bool = True):
    env = {**os.environ, "EXPLAIN_EDIT_FAILURES": "true",
           "STR_REPLACE_DID_YOU_MEAN": "true" if dym else "false"}
    return subprocess.run(
        [sys.executable, str(SCRIPT), "str_replace", "--path", str(path),
         "--old_str", old, "--new_str", "REPLACED"],
        capture_output=True, text=True, timeout=30, env=env,
    ).stdout


def test_did_you_mean_prints_the_region_for_an_indent_miss(tmp_path):
    f = write(tmp_path, "def g():\n    a()\n    b()\n")
    out = run_edit_dym(f, "        a()\n        b()")
    assert "The matching region of the file is exactly this JSON value:" in out
    # escaped, not raw: the whitespace that caused the miss has to be visible
    assert '"old_str": "    a()\\n    b()"' in out
    assert "copied character for character" in out


def test_did_you_mean_makes_trailing_whitespace_visible(tmp_path):
    """The case the raw form could not show: printed raw, the region rendered
    identically to what the model sent, and it re-sent the same bytes."""
    f = write(tmp_path, "def g():\n    x = (0,)\n    return x\n")
    out = run_edit_dym(f, "    x = (0,) \n    return x")
    assert '"old_str": "    x = (0,)\\n    return x"' in out
    assert "(0,) \\n" not in out, "the trailing space must be gone from the suggestion"


def test_did_you_mean_stays_off_without_the_flag(tmp_path):
    f = write(tmp_path, "def g():\n    a()\n    b()\n")
    out = run_edit_dym(f, "        a()\n        b()", dym=False)
    assert "matching region" not in out and "Closest match" in out


def test_did_you_mean_prints_for_a_close_window(tmp_path):
    body = "def g():\n    x = 1\n    y = 2\n    z = 3\n    return x + y + z\n"
    f = write(tmp_path, body)
    out = run_edit_dym(f, "def g():\n    x = 1\n    y = 9\n    z = 3\n    return x + y + z")
    assert "The matching region of the file is exactly this JSON value:" in out
    assert "    y = 2" in out


def test_did_you_mean_skips_a_weak_window(tmp_path):
    f = write(tmp_path, "alpha\nbeta\ngamma\ndelta\n" * 3)
    out = run_edit_dym(f, "alpha\nBETA9\nGAMMA9\nDELTA9")
    assert "matching region" not in out


def test_multiple_occurrences_shows_each_match_in_context(tmp_path):
    """'Make it unique' is underspecified: the tool knows where the matches are, so it
    shows them and the retry becomes a choice."""
    body = ("class A:\n    def run(self):\n        pass\n\n"
            "class B:\n    def run(self):\n        pass\n")
    f = write(tmp_path, body)
    out = run_edit(f, "    def run(self):\n        pass", "    def run(self):\n        return 1")
    assert "Multiple occurrences" in out
    assert "occurrence at line 2:" in out and "occurrence at line 6:" in out
    assert "class A:" in out and "class B:" in out, "the disambiguating context must be shown"
    assert "extend your old_str with the neighbouring lines" in out


def test_no_op_edit_asks_for_the_intended_change(tmp_path):
    f = write(tmp_path, "def g():\n    x = 1\n    return x\n")
    out = run_edit(f, "    x = 1", "    x = 1")
    assert "byte-identical" in out and "leave the file unchanged" in out
    assert "state the change you intend" in out.lower(), "the message must instruct, not just report"


def test_create_on_existing_file_does_not_overwrite(tmp_path):
    """The refusal used to fall through to the write: the model was told 'Cannot overwrite'
    and the file was replaced anyway."""
    f = write(tmp_path, "def critical():\n    return 'ORIGINAL'\n")
    before = f.read_text()
    out = run_create(f, "print('CLOBBERED')\n")
    assert "Cannot overwrite" in out
    assert "created successfully" not in out, "a refused create must not also report success"
    assert f.read_text() == before, "the file must be untouched"


def test_view_of_a_missing_path_stops(tmp_path):
    out = run_edit(tmp_path / "nope.py", "x")
    assert "does not exist" in out
    assert "No replacement was performed" not in out, "the refusal must stop the command body"


def test_did_you_mean_region_spans_the_restored_lines_not_the_collapsed_count(tmp_path):
    """An escape-class old_str holds literal 2-char `\\n` sequences, so its own line count
    is collapsed; the matched region's extent must come from the restored candidate or the
    suggestion truncates to a fraction of the true region."""
    body = "def f():\n    a()\n    b()\n    c()\n    d()\n"
    f = write(tmp_path, body)
    out = run_edit_dym(f, "def f():\\n    a()\\n    b()\\n    c()\\n    d()")
    assert "collapsed your newlines" in out
    assert '"old_str": "def f():\\n    a()\\n    b()\\n    c()\\n    d()"' in out


DC_FILE = ("    def _end_dc_subject(self):\n        self._end_category()\n\n"
           "    def _end_dc_title(self):\n        pass\n\n"
           "    def _end_dcterms_created(self):\n        self._end_created()\n\n"
           "    def _end_dcterms_issued(self):\n        self._end_last_updated()\n\n"
           "    def _end_dcterms_modified(self):\n        self._end_updated()\n")


def test_anchored_window_reports_a_skipped_block_as_an_insert(tmp_path):
    """The model's old_str spliced two real regions, skipping the method between them.
    The same-length selector shifted the window and called the model's real anchor wrong;
    the anchored selector keeps both model edges and names the skipped block."""
    f = write(tmp_path, DC_FILE)
    out = run_edit_dym(f, "    def _end_dc_title(self):\n        pass\n\n"
                          "    def _end_dcterms_issued(self):\n        self._end_last_updated()")
    assert "your old_str skips" in out and "_end_dcterms_created" in out
    assert "your old_str has `    def _end_dc_title" not in out, "model's anchor called wrong"
    region = out.split("matching region", 1)[1]
    assert "_end_dcterms_created" in region, "suggestion must contain the missing block"


def test_anchored_window_falls_back_when_no_edge_anchors(tmp_path):
    f = write(tmp_path, "alpha\nbeta\ngamma\ndelta\n")
    out = run_edit_dym(f, "alphX\nbetY\ngammZ")
    assert "Closest match" in out or "No similar region" in out


def test_multiple_occurrences_elide_the_matched_middle(tmp_path):
    body = "def run(self):\n" + "".join(f"    line{i}()\n" for i in range(10))
    f = write(tmp_path, "class A:\n" + indent(body, 4) + "\nclass B:\n" + indent(body, 4))
    old = indent(body, 4).rstrip("\n")
    out = run_edit(f, old, "new")
    assert "Multiple occurrences" in out
    assert "elided" in out, "long matched body must be elided"
    assert out.count("occurrence at line") == 2


def test_noop_message_says_the_change_may_already_be_there(tmp_path):
    f = write(tmp_path, "def g():\n    return 2\n")
    out = run_edit(f, "def g():\n    return 2", "def g():\n    return 2")
    assert "ALREADY in the file" in out and "view or grep" in out


def test_did_you_mean_resend_is_conditional(tmp_path):
    f = write(tmp_path, "def g():\n    a()\n    b()\n")
    out = run_edit_dym(f, "        a()\n        b()")
    assert "If this is the region you meant to change" in out
    assert "view the file and locate the right region" in out


def _run_insert(path, line, text):
    import pathlib
    import subprocess
    import sys

    script = pathlib.Path(__file__).resolve().parents[2] / "uni_agent" / "tools" / "str_replace_editor" / "str_replace_editor"
    return subprocess.run([sys.executable, str(script), "insert", "--path", str(path), "--insert_line", str(line),
                           "--new_str", text], capture_output=True, text=True).stdout


def test_insert_goes_after_the_named_line_and_keeps_tabs(tmp_path):
    f = write(tmp_path, "all:\n\t@echo hi\n\t@echo bye\n")
    _run_insert(f, 1, "\t@echo new")
    assert f.read_text() == "all:\n\t@echo new\n\t@echo hi\n\t@echo bye\n"


def test_insert_at_zero_is_the_top(tmp_path):
    f = write(tmp_path, "a = 1\n")
    _run_insert(f, 0, "# top")
    assert f.read_text() == "# top\na = 1\n"


def test_strip_fallback_refuses_a_reindent_of_a_misread_line(tmp_path):
    f = write(tmp_path, "class A:\n       def g(self):\n            return 1\n")
    out = run_edit(f, "        def g(self):\n            return 1", "    def g(self):\n        return 2")
    assert "changes that line's indentation" in out
    assert f.read_text() == "class A:\n       def g(self):\n            return 1\n", "file untouched"


def test_strip_fallback_still_applies_a_pure_slip(tmp_path):
    f = write(tmp_path, "class A:\n       def g(self):\n            return 1\n")
    out = run_edit(f, "        def g(self):\n            return 1", "        def g(self):\n            return 2")
    assert "has been edited" in out
    assert f.read_text() == "class A:\n       def g(self):\n            return 2\n"
