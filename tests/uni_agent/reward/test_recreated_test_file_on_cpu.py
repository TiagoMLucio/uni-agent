"""A graded test file the agent wrote back must not cost it its source fix.

The agent works with the graded test files deleted. When the files it still has import one of
them, it writes a stub at that path; the eval checkout holds the repo's own copy, git refuses
to create a file that exists, and it refuses the whole prediction with it.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

from uni_agent.reward.swe_smith import GIT_APPLY_CMDS, _apply_cmds, _make_eval_script_list

PREDICTION = """diff --git a/pkg/core.py b/pkg/core.py
index 1111111..2222222 100644
--- a/pkg/core.py
+++ b/pkg/core.py
@@ -1,2 +1,2 @@
 def f(x):
-    return x < 1
+    return x <= 1
diff --git a/tests/test_x.py b/tests/test_x.py
new file mode 100644
index 0000000..3333333
--- /dev/null
+++ b/tests/test_x.py
@@ -0,0 +1 @@
+# stub written by the agent
"""
REPO_TEST = "from pkg.core import f\n\ndef test_f():\n    assert f(1)\n"


def test_the_attempt_without_the_graded_files_follows_the_first_rung():
    cmds = _apply_cmds(["tests/test_x.py", "tests/test_y.py"])
    assert cmds[0] == GIT_APPLY_CMDS[0]
    assert cmds[1] == "git apply --verbose --exclude=tests/test_x.py --exclude=tests/test_y.py"
    assert cmds[2:] == list(GIT_APPLY_CMDS[1:])


def test_the_ladder_is_the_librarys_when_no_file_is_graded():
    assert _apply_cmds([]) == list(GIT_APPLY_CMDS)
    # a test id that is no path (unittest ids) names no file to leave out
    assert _apply_cmds(["test_a (__main__.TestCase)"]) == list(GIT_APPLY_CMDS)


def test_the_script_tries_it_before_the_rungs_that_leave_half_a_patch():
    script = _make_eval_script_list("inst", PREDICTION, "pytest -q", ["tests/test_x.py"])
    tries = [ln for ln in script if "/tmp/swesmith_pred.diff; then _applied=1" in ln]
    assert [("--exclude=tests/test_x.py" in ln, "--reject" in ln, "patch --batch" in ln) for ln in tries] == [
        (False, False, False), (True, False, False), (False, True, False), (False, False, True)]
    assert script.index(tries[-1]) < script.index("git checkout -- tests/test_x.py")


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_the_source_fix_lands_and_the_repos_test_file_stays(tmp_path):
    def git(*args):
        return subprocess.run(["git", *args], cwd=tmp_path, capture_output=True, text=True)

    (tmp_path / "pkg").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "pkg" / "core.py").write_text("def f(x):\n    return x < 1\n")
    (tmp_path / "tests" / "test_x.py").write_text(REPO_TEST)
    git("init", "-q", ".")
    git("add", "-A")
    git("-c", "user.email=a@b", "-c", "user.name=a", "commit", "-qm", "base")
    patch = tmp_path.parent / f"{tmp_path.name}.diff"
    patch.write_text(PREDICTION)

    first, without_graded = _apply_cmds(["tests/test_x.py"])[:2]
    refused = subprocess.run(f"{first} {patch}", shell=True, cwd=tmp_path, capture_output=True, text=True)
    assert refused.returncode != 0 and "already exists" in refused.stderr
    assert git("status", "--short").stdout == ""
    applied = subprocess.run(f"{without_graded} {patch}", shell=True, cwd=tmp_path, capture_output=True, text=True)
    assert applied.returncode == 0, applied.stderr
    assert (tmp_path / "pkg" / "core.py").read_text() == "def f(x):\n    return x <= 1\n"
    assert (tmp_path / "tests" / "test_x.py").read_text() == REPO_TEST
