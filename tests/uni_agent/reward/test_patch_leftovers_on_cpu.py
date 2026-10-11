"""Whatever the agent leaves in the repository, its source edit reaches the grader.

A rollout's test script opened a SQLite database named ``:memory?:test_...``, which SQLite created
as a file of that name. Git reads a leading ``:`` as pathspec magic, so the file was never
unstaged, the diff carried "Binary files differ", and no apply command took the patch: a fix that
matched the reference one scored zero. The same family: a file named ``*`` unstaged every file and
left an empty patch, a nested repository without a commit made ``git add -A`` abort, and a name
git quotes (a newline, a tab, a non-ASCII byte) never matched what it named.

Each case runs the command the reward builds through a real bash against a real repository, then
applies the result to a clean clone with plain ``git apply``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from uni_agent.reward.base import PATCH_EXTRACT_OK, patch_extract_command

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")

BINARY = b"SQLite format 3\x00\x01"


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


def _repos(tmp_path: Path) -> tuple[Path, Path]:
    """The agent's checkout with its source edit, and a clean clone standing in for the eval sandbox."""
    repo = tmp_path / "testbed"
    repo.mkdir()
    (repo / "src.py").write_text("def f():\n    return 1\n")
    (repo / "data.bin").write_bytes(b"\x00\x01tracked")
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "t")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "base")
    clean = tmp_path / "eval"
    shutil.copytree(repo, clean)
    (repo / "src.py").write_text("def f():\n    return 2\n")
    return repo, clean


def _collect(repo: Path, tmp_path: Path) -> str:
    out = tmp_path / "patch.diff"
    done = subprocess.run(
        ["bash", "-c", patch_extract_command(str(out), repo_dir=str(repo))], capture_output=True, text=True, cwd="/"
    )
    assert PATCH_EXTRACT_OK in done.stdout + done.stderr, done.stdout + done.stderr
    return out.read_text(errors="backslashreplace")


def _applies(clean: Path, patch: str, tmp_path: Path) -> bool:
    sent = tmp_path / "sent.diff"
    sent.write_text(patch)
    return subprocess.run(["git", "apply", str(sent)], cwd=clean, capture_output=True).returncode == 0


def _nested_repo(repo: Path, commit: bool) -> None:
    sub = repo / "scratch"
    sub.mkdir()
    (sub / "notes.py").write_text("x = 1\n")
    _git(sub, "init", "-q")
    if commit:
        _git(sub, "config", "user.email", "t@example.com")
        _git(sub, "config", "user.name", "t")
        _git(sub, "add", "-A")
        _git(sub, "commit", "-qm", "x")


JUNK = {
    "leading colon": lambda r: (r / ":memory?:test_T1: table 'order' (SQL keyword)").write_bytes(BINARY),
    "leading dash": lambda r: (r / "-x.bin").write_bytes(BINARY),
    "glob star": lambda r: (r / "*").write_bytes(BINARY),
    "glob bracket": lambda r: (r / "a[b].bin").write_bytes(BINARY),
    "double quote": lambda r: (r / 'a"b.bin').write_bytes(BINARY),
    "newline": lambda r: (r / "a\nb.bin").write_bytes(BINARY),
    "tab": lambda r: (r / "a\tb.bin").write_bytes(BINARY),
    "backslash": lambda r: (r / "a\\b.bin").write_bytes(BINARY),
    "non-ascii": lambda r: (r / "café.bin").write_bytes(BINARY),
    "tracked binary rewritten": lambda r: (r / "data.bin").write_bytes(b"\x00\x02other"),
    "non-utf8 text": lambda r: (r / "dump.txt").write_bytes(b"caf\xe9 \xff\n"),
    "non-utf8 text, leading colon": lambda r: (r / ":dump.txt").write_bytes(b"caf\xe9\n"),
    "nested repo with a commit": lambda r: _nested_repo(r, commit=True),
    "nested repo without a commit": lambda r: _nested_repo(r, commit=False),
    "named pipe": lambda r: os.mkfifo(r / "pipe"),
}

KEPT = {
    "leading colon": ":notes.txt",
    "glob star": "*",
    "newline": "a\nb.txt",
    "non-ascii": "café.txt",
    "space": "my test.py",
}


@pytest.mark.parametrize("leftover", sorted(JUNK))
def test_what_a_patch_cannot_carry_is_dropped_and_the_edit_applies(tmp_path, leftover):
    repo, clean = _repos(tmp_path)
    JUNK[leftover](repo)

    patch = _collect(repo, tmp_path)

    assert patch.count("diff --git") == 1, patch
    assert "return 2" in patch and "Binary files" not in patch
    assert _applies(clean, patch, tmp_path)
    assert "return 2" in (clean / "src.py").read_text()


@pytest.mark.parametrize("name", sorted(KEPT))
def test_a_text_file_is_kept_whatever_its_name(tmp_path, name):
    repo, clean = _repos(tmp_path)
    (repo / KEPT[name]).write_text("print('x')\n")

    patch = _collect(repo, tmp_path)

    assert patch.count("diff --git") == 2, patch
    assert _applies(clean, patch, tmp_path)
    assert (clean / KEPT[name]).read_text() == "print('x')\n"
    assert "return 2" in (clean / "src.py").read_text()


def test_a_symlink_is_carried(tmp_path):
    repo, clean = _repos(tmp_path)
    os.symlink("src.py", repo / "alias.py")

    patch = _collect(repo, tmp_path)

    assert _applies(clean, patch, tmp_path)
    assert os.readlink(clean / "alias.py") == "src.py"


def test_the_command_is_one_line_without_a_tab():
    # it is typed into an interactive shell, where a tab completes and a newline submits
    command = patch_extract_command("/tmp/p.diff")

    assert "\t" not in command and "\n" not in command
