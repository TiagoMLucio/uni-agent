"""A patch extraction that fails must not be read back as an agent that changed nothing.

The command chained `cd && printf && git add -A && (unstage) ; git diff --cached > file`. The `;`
was there so the binary-unstaging step could fail harmlessly, but its scope is the whole prefix,
so a failed `cd` diffed whatever directory the session was in and a failed `git add` diffed an
unstaged index. Either way the redirect had already created the file, `read_file` returned "",
and the reward reported `empty_patch`: a silent zero indistinguishable from a real one.

Nothing raised and nothing could be checked: the side session runs with `check="ignore"`, so
swe-rex extracts no exit status. Hence the sentinel.

These run the command the reward actually builds through a real bash against a real repository,
with `repo_dir` pointed at a tmp checkout. The failing-`cd` case keeps the real `/testbed`.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

from uni_agent.reward.base import PATCH_EXTRACT_OK, empty_patch_flag, patch_extract_command

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "testbed"
    repo.mkdir()
    (repo / "src.py").write_text("def f():\n    return 1\n")
    run = lambda *a: subprocess.run(a, cwd=repo, check=True, capture_output=True)  # noqa: E731
    run("git", "init", "-q")
    run("git", "config", "user.email", "t@example.com")
    run("git", "config", "user.name", "t")
    run("git", "add", "-A")
    run("git", "commit", "-qm", "base")
    (repo / "src.py").write_text("def f():\n    return 2\n")
    return repo


def _shim(tmp_path: Path, failing_subcommand: str) -> dict[str, str]:
    """A PATH where one git subcommand fails and everything else is the real git."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    real = shutil.which("git")
    (bindir / "git").write_text(textwrap.dedent(f"""\
        #!/bin/bash
        for arg in "$@"; do
          if [ "$arg" = "{failing_subcommand}" ]; then
            echo "git {failing_subcommand}: simulated failure" >&2
            exit 1
          fi
        done
        exec {real} "$@"
    """))
    (bindir / "git").chmod(0o755)
    return {**os.environ, "PATH": f"{bindir}:{os.environ['PATH']}"}


def _run(command: str, env: dict[str, str] | None = None) -> str:
    """What `communicate_isolated` would hand back: the output, with no exit status."""
    done = subprocess.run(["bash", "-c", command], capture_output=True, text=True, env=env, cwd="/")
    return done.stdout + done.stderr


def test_the_happy_path_still_produces_the_diff(tmp_path):
    repo = _repo(tmp_path)
    out = tmp_path / "patch.diff"
    output = _run(patch_extract_command(str(out), repo_dir=str(repo)))
    assert PATCH_EXTRACT_OK in output
    assert "return 2" in out.read_text()


def test_an_untracked_file_still_reaches_the_patch(tmp_path):
    repo = _repo(tmp_path)
    (repo / "reproduce.py").write_text("print('x')\n")
    out = tmp_path / "patch.diff"
    output = _run(patch_extract_command(str(out), repo_dir=str(repo)))
    assert PATCH_EXTRACT_OK in output
    assert "reproduce.py" in out.read_text()


def test_a_failing_cd_does_not_pass_for_an_empty_patch(tmp_path):
    """The real `/testbed`, which does not exist here: the old command diffed the cwd instead."""
    out = tmp_path / "patch.diff"
    output = _run(patch_extract_command(str(out)))
    assert PATCH_EXTRACT_OK not in output
    assert not out.exists(), "the diff is inside the guarded group now, so its redirect never runs"


def test_the_old_shape_is_what_produced_the_silent_zero(tmp_path):
    """The command as it stood, kept as the regression: it reports nothing and writes "".

    Run from a directory that is a git repository, so the stray diff cannot even fail loudly.
    """
    out = tmp_path / "patch.diff"
    old_shape = (
        "cd /testbed && printf '*.py diff=python\\n' > /tmp/.uniagent_gitattributes && git add -A && "
        "(git diff --cached --numstat | awk -F'\\t' '$1==\"-\"{print $3}' "
        "| xargs -r -d '\\n' git reset -q --) ; "
        f"git diff --no-color --cached > {out}"
    )
    repo = _repo(tmp_path)
    done = subprocess.run(["bash", "-c", old_shape], capture_output=True, text=True, cwd=repo)
    assert out.read_text() == "", "an empty patch from a repository that has changes"
    assert done.returncode == 0, "and a clean exit, so even an exit status would not have caught it"


def test_a_failing_git_add_does_not_pass_for_an_empty_patch(tmp_path):
    repo = _repo(tmp_path)
    out = tmp_path / "patch.diff"
    output = _run(patch_extract_command(str(out), repo_dir=str(repo)), env=_shim(tmp_path, "add"))
    assert PATCH_EXTRACT_OK not in output


def test_a_failing_unstage_step_still_produces_the_diff(tmp_path):
    """It exists to drop binaries; losing it costs a hunk git cannot express, not the patch."""
    repo = _repo(tmp_path)
    (repo / "blob.bin").write_bytes(b"\x00\x01\x02binary")
    out = tmp_path / "patch.diff"
    output = _run(patch_extract_command(str(out), repo_dir=str(repo)), env=_shim(tmp_path, "reset"))
    assert PATCH_EXTRACT_OK in output
    assert "return 2" in out.read_text()


def test_a_repository_with_no_changes_extracts_an_empty_patch(tmp_path):
    """The real empty patch: the extraction succeeded and the agent changed nothing."""
    repo = _repo(tmp_path)
    subprocess.run(["git", "checkout", "--", "src.py"], cwd=repo, check=True, capture_output=True)
    out = tmp_path / "patch.diff"
    output = _run(patch_extract_command(str(out), repo_dir=str(repo)))
    assert PATCH_EXTRACT_OK in output, "an empty patch is an outcome, not a failure"
    assert out.read_text() == ""


def test_the_sentinel_is_the_last_thing_the_command_says():
    command = patch_extract_command("/tmp/p.diff")
    assert command.endswith(f"&& echo {PATCH_EXTRACT_OK}")
    assert " ; }" in command, "the unstage step keeps its own separator inside the guarded group"


def test_a_genuinely_empty_patch_is_still_reported_as_one():
    assert empty_patch_flag("") == {"empty_patch": True}
    assert empty_patch_flag("   \n") == {"empty_patch": True}


def test_a_real_patch_is_not_an_empty_one():
    assert empty_patch_flag("diff --git a/src.py b/src.py\n") == {"empty_patch": False}


def test_a_prediction_that_never_arrived_is_reported_as_nothing():
    """Absent, not False: the run says what went wrong under eval_error, and the health metrics
    read an absent key as never measured rather than as a healthy zero."""
    assert empty_patch_flag(None) == {}
