"""A sandbox's processes can each stay under its per-process data cap (``ulimit -d``) and still take far more
together: a test runner forking one worker per CPU took 64 GiB of a node, Ray killed rollout workers for it
and the run hung. The guard kills the agent's command and all it started, never the sandbox's own shell."""

from __future__ import annotations

import asyncio
import logging
import subprocess
import sys
import time

import pytest

from uni_agent.deployment.local import deployment as deployment_module
from uni_agent.deployment.local.deployment import LocalDeployment

MIB = 1 << 20
# the agent's command: three workers holding 120 MiB each, under a 200 MiB cap per process
COMMAND = (
    "import subprocess, sys, time\n"
    "work = 'import time; held = b\"x\" * (120 << 20); time.sleep(60)'\n"
    "workers = [subprocess.Popen([sys.executable, '-c', work]) for _ in range(3)]\n"
    "time.sleep(60)\n"
)
# the sandbox's shell: runs the command when told, after the guard has seen the sandbox as set up, and
# reaps it as bash would
SHELL = (
    "import subprocess, sys\n"
    "print('ready', flush=True)\n"
    "sys.stdin.readline()\n"
    f"subprocess.Popen([sys.executable, '-c', {COMMAND!r}]).wait()\n"
    "sys.stdin.readline()\n"
)


def _sandbox(cap_bytes: int | None) -> LocalDeployment:
    deployment = LocalDeployment.__new__(LocalDeployment)
    deployment.logger = logging.getLogger("memory-guard-test")
    deployment._memory_guard = None
    deployment.memory_guard_interval = 0.1
    cap = f"ulimit -d {cap_bytes // 1024}; " if cap_bytes else ""
    deployment._server_process = subprocess.Popen(
        ["bash", "-c", f'{cap}exec "{sys.executable}" -c "$0"', SHELL],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    assert deployment._server_process.stdout.readline() == "ready\n"
    return deployment


async def _run_command(deployment: LocalDeployment, seconds: float) -> None:
    deployment.guard_memory()
    deployment._server_process.stdin.write("go\n")
    deployment._server_process.stdin.flush()
    await asyncio.sleep(1)
    deadline = time.monotonic() + seconds
    while deployment_module._parents(deployment._server_process.pid) and time.monotonic() < deadline:
        await asyncio.sleep(0.1)


def _teardown(deployment: LocalDeployment) -> None:
    if deployment._memory_guard is not None:
        deployment._memory_guard.cancel()
    deployment._kill_server_tree()
    deployment._server_process.wait(10)


def test_the_command_over_the_sandbox_cap_is_killed_and_the_shell_survives(caplog):
    deployment = _sandbox(200 * MIB)
    try:
        with caplog.at_level(logging.WARNING, logger="memory-guard-test"):
            asyncio.run(_run_command(deployment, seconds=10))
        assert deployment._server_process.poll() is None, "the sandbox's shell survives"
        assert deployment_module._parents(deployment._server_process.pid) == {}, "the command and its workers are gone"
        assert any("Memory guard" in r.message and "and the 3 processes it started" in r.message for r in caplog.records)
    finally:
        _teardown(deployment)


def test_a_sandbox_without_a_data_cap_is_not_guarded():
    deployment = _sandbox(None)
    try:
        asyncio.run(asyncio.sleep(0))
        deployment.guard_memory()
        assert deployment._memory_guard is None
    finally:
        _teardown(deployment)


@pytest.mark.parametrize("children_lists", [True, False], ids=["task children lists", "full /proc scan"])
def test_the_tree_is_the_same_either_way(monkeypatch, children_lists):
    root = subprocess.Popen(["bash", "-c", "sleep 30 & sleep 30 & wait"])
    try:
        time.sleep(0.3)
        if not children_lists:
            exists = deployment_module.os.path.exists
            monkeypatch.setattr(deployment_module.os.path, "exists", lambda p: False if p.endswith("/children") else exists(p))
        parents = deployment_module._parents(root.pid)
        assert sorted(parents.values()) == [root.pid, root.pid]
        assert sorted(parents) == sorted(deployment_module._process_tree(root.pid))
    finally:
        root.kill()
        root.wait(10)
