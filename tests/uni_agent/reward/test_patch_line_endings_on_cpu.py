"""An edit to a file with CRLF line endings reaches the grader as the agent made it.

The sandbox reads a file as text (``Path.read_text``), which turns every CRLF into LF. A diff read
that way no longer matches the file it was made from, so every apply command refused it in the
eval sandbox and the edit scored zero. The diff is now read through a base64 copy.

The collection command runs through a real bash against a real repository, the read goes through
a reader that behaves like the sandbox's, and the result is applied to a clean clone.
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess
from pathlib import Path

import pytest

from uni_agent.reward.base import PATCH_EXTRACT_OK, patch_extract_command, read_patch

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")

CRLF = b"a = 1\r\nb = 2\r\nc = 3\r\n"


class Sandbox:
    """Reads a file the way the sandbox runtime does."""

    async def read_file(self, path, encoding=None, errors=None):
        return Path(path).read_text(encoding=encoding, errors=errors)


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


def _repos(tmp_path: Path, name: str, content: bytes) -> tuple[Path, Path]:
    repo = tmp_path / "testbed"
    repo.mkdir()
    (repo / name).write_bytes(content)
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "t")
    _git(repo, "config", "core.autocrlf", "false")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "base")
    clean = tmp_path / "eval"
    shutil.copytree(repo, clean)
    return repo, clean


def _collect(repo: Path, tmp_path: Path) -> str:
    out = tmp_path / "patch.diff"
    done = subprocess.run(
        ["bash", "-c", patch_extract_command(str(out), repo_dir=str(repo))], capture_output=True, text=True, cwd="/"
    )
    assert PATCH_EXTRACT_OK in done.stdout + done.stderr, done.stdout + done.stderr
    return asyncio.run(read_patch(Sandbox(), str(out)))


def _apply(clean: Path, patch: str, tmp_path: Path) -> bool:
    sent = tmp_path / "sent.diff"
    sent.write_text(patch)
    return subprocess.run(["git", "apply", str(sent)], cwd=clean, capture_output=True).returncode == 0


def test_an_edit_to_a_crlf_file_applies_as_it_was_made(tmp_path):
    repo, clean = _repos(tmp_path, "win.py", CRLF)
    (repo / "win.py").write_bytes(CRLF.replace(b"b = 2", b"b = 22"))

    patch = _collect(repo, tmp_path)

    assert "b = 22\r\n" in patch
    assert _apply(clean, patch, tmp_path)
    assert (clean / "win.py").read_bytes() == (repo / "win.py").read_bytes()


def test_the_text_reader_alone_drops_the_line_endings(tmp_path):
    repo, _ = _repos(tmp_path, "win.py", CRLF)
    (repo / "win.py").write_bytes(CRLF.replace(b"b = 2", b"b = 22"))
    _collect(repo, tmp_path)

    as_text = asyncio.run(Sandbox().read_file(tmp_path / "patch.diff"))

    assert "\r" not in as_text


def test_bytes_that_are_not_utf8_are_escaped_not_refused(tmp_path):
    repo, _ = _repos(tmp_path, "notes.py", b"# caf\xe9\nx = 1\n")
    (repo / "notes.py").write_bytes(b"# caf\xe9\nx = 2\n")

    patch = _collect(repo, tmp_path)

    assert "+x = 2" in patch and "caf\\xe9" in patch
