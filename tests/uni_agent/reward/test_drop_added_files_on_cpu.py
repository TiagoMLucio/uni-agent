"""Files the agent adds must not reach pytest's collection.

pytest imports a ``test_*.py`` to collect it, so a scratch file with a module-level assert
aborts the whole session and no graded test runs. The eval drops the files the prediction
adds, but only when the reference fix needs no new file of its own.
"""

from __future__ import annotations

from uni_agent.reward.swe_smith import _files_to_drop, _make_eval_script_list, _prediction_adds_files

ADDS = """diff --git a/test_repro.py b/test_repro.py
new file mode 100644
index 0000000..e69de29
--- /dev/null
+++ b/test_repro.py
@@ -0,0 +1 @@
+assert detect("Hello") == "en"
diff --git a/pkg/core.py b/pkg/core.py
index 1111111..2222222 100644
--- a/pkg/core.py
+++ b/pkg/core.py
@@ -1,1 +1,1 @@
-if count < threshold:
+if count <= threshold:
"""

BUG_PATCH_PLAIN = "diff --git a/pkg/core.py b/pkg/core.py\n@@ -1 +1 @@\n-a\n+b\n"
BUG_PATCH_WITH_DELETE = (
    "diff --git a/pkg/new.py b/pkg/new.py\ndeleted file mode 100644\n--- a/pkg/new.py\n+++ /dev/null\n"
)


def test_only_added_paths_are_listed():
    assert _prediction_adds_files(ADDS) == ["test_repro.py"]


def test_added_file_is_dropped_when_the_gold_adds_none():
    assert _files_to_drop(ADDS, BUG_PATCH_PLAIN, []) == ["test_repro.py"]


def test_nothing_is_dropped_when_the_gold_itself_creates_a_file():
    # the bug patch deleting a file means the reversed gold fix creates it (pr_* mirrors)
    assert _files_to_drop(ADDS, BUG_PATCH_WITH_DELETE, []) == []


def test_empty_prediction_drops_nothing():
    assert _files_to_drop("", BUG_PATCH_PLAIN, []) == []


def test_paths_needing_shell_quoting_are_skipped():
    weird = ADDS.replace("test_repro.py", "test repro.py")
    assert _files_to_drop(weird, BUG_PATCH_PLAIN, []) == []


def test_modified_files_are_never_dropped():
    assert "pkg/core.py" not in _files_to_drop(ADDS, BUG_PATCH_PLAIN, [])


def test_script_removes_the_files_after_reverting_the_tests():
    script = _make_eval_script_list("inst", ADDS, "pytest -q", ["tests/test_x.py"], drop_files=["test_repro.py"])
    assert "rm -f -- test_repro.py" in script
    assert script.index("git checkout -- tests/test_x.py") < script.index("rm -f -- test_repro.py")
    assert script.index("rm -f -- test_repro.py") < script.index("pytest -q")


def test_script_is_unchanged_when_nothing_is_dropped():
    assert not [ln for ln in _make_eval_script_list("inst", ADDS, "pytest -q", []) if ln.startswith("rm -f")]


def test_a_recreated_graded_test_file_is_never_dropped():
    # the agent works with the graded tests deleted, so writing one back looks like an
    # addition; the eval has just restored the repo's copy and must keep it
    assert _files_to_drop(ADDS, BUG_PATCH_PLAIN, ["test_repro.py"]) == []


def test_output_altering_pytest_plugins_are_switched_off_before_the_tests():
    from uni_agent.reward.swe_smith import PYTEST_PLUGINS_OFF

    script = _make_eval_script_list("inst", ADDS, "pytest -q", ["tests/test_x.py"])
    export = next(ln for ln in script if ln.startswith("export PYTEST_ADDOPTS="))
    assert PYTEST_PLUGINS_OFF in export and "${PYTEST_ADDOPTS:-}" in export
    assert script.index(export) < script.index("pytest -q")
