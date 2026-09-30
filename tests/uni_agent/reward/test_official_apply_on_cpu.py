"""The prediction is applied the way swebench.harness.run_evaluation applies it.

psf__requests-1142's image ships an untracked, unignored ``build/`` (from ``pip install .``), so a
``git add -A`` prediction carries ``build/lib/...`` as new files that already exist in the eval
container. The old three-command ladder then failed (or, with ``patch``, reverse-applied the fix);
the official harness resets to a pristine tree (``git checkout -- . ; git clean -fd``) before every
retry, which removes ``build/`` and lets the patch apply.

The commands run through a real bash against a real repository standing in for ``/testbed``.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
from swebench.harness.constants import MAP_REPO_VERSION_TO_SPECS

from uni_agent.reward.base import PATCH_EXTRACT_OK
from uni_agent.reward.swe_bench import SWEBenchRewardSpec

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


def _image(tmp_path: Path, name: str) -> Path:
    """A checkout at the base commit with the image's untracked build/ next to it."""
    repo = tmp_path / name
    repo.mkdir()
    (repo / "models.py").write_text("def length(body):\n    return 0\n")
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "t")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "base")
    (repo / "build" / "lib").mkdir(parents=True)
    (repo / "build" / "lib" / "models.py").write_text("def length(body):\n    return 0\n")
    return repo


class Env:
    """Runs each command in a real bash with ``/testbed`` pointed at ``repo``; like the real env, a
    ``check="raise"`` failure closes it and every later command fails."""

    def __init__(self, repo: Path):
        self.repo = repo
        self.commands: list[str] = []
        self.closed = False

    async def write_file(self, path, content):
        Path(path).write_text(content)

    async def communicate(self, command, check="ignore", **kwargs):
        if self.closed:
            raise RuntimeError("env is closed")
        self.commands.append(command)
        done = subprocess.run(
            ["bash", "-c", command.replace("/testbed", str(self.repo))], capture_output=True, text=True
        )
        if check == "raise" and done.returncode != 0:
            self.closed = True
            raise RuntimeError(done.stdout + done.stderr)
        return done.stdout + done.stderr


def _spec(env) -> SWEBenchRewardSpec:
    repo = "psf/requests"
    return SWEBenchRewardSpec(
        run_id="r",
        metadata={
            "instance_id": "psf__requests-1142",
            "repo": repo,
            "version": next(iter(MAP_REPO_VERSION_TO_SPECS[repo])),
            "base_commit": "abc",
            "test_patch": "",
            "patch": "",
        },
        env=env,
    )


def _agent_patch(tmp_path: Path) -> str:
    """What ``git add -A && git diff --cached`` hands back after the agent's one-line fix."""
    agent = _image(tmp_path, "agent")
    (agent / "models.py").write_text("def length(body):\n    return len(body)\n")
    _git(agent, "add", "-A")
    return subprocess.run(["git", "diff", "--cached"], cwd=agent, check=True, capture_output=True, text=True).stdout


def test_a_patch_carrying_the_images_untracked_build_dir_applies(tmp_path):
    patch = _agent_patch(tmp_path)
    assert "build/lib/models.py" in patch, "the image's build/ rides along, as in the real prediction"

    eval_repo = _image(tmp_path, "eval")
    env = Env(eval_repo)
    _spec(env)._apply_patch(patch, env=env)

    assert "return len(body)" in (eval_repo / "models.py").read_text(), "the fix itself is in the tree"
    assert any("git clean -fd" in c for c in env.commands), "the retry started from a pristine tree"


def test_a_non_utf8_file_in_the_diff_does_not_lose_the_prediction(tmp_path):
    class Extract:
        async def communicate_isolated(self, command, **kwargs):
            return PATCH_EXTRACT_OK

        async def read_file(self, path, encoding=None, errors=None):
            raw = b"diff --git a/notes.txt b/notes.txt\n+caf\xe9\n"
            return raw.decode("utf-8", errors=errors or "strict")

    patch = _spec(Extract())._get_interaction_env_patch()

    assert "notes.txt" in patch
