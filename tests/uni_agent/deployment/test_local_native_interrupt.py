"""A command that ignores SIGINT is suspended and killed; a job the agent backgrounded earlier is not."""

from __future__ import annotations

import asyncio

import pytest
from swerex.exceptions import CommandTimeoutError
from swerex.runtime.abstract import BashAction, BashInterruptAction, CreateBashSessionRequest

from uni_agent.deployment.local_native.runtime import BashSession

DEAF = "( trap '' INT; exec sleep 299 )"


async def _interrupt_then_jobs() -> tuple[str, str]:
    s = BashSession(CreateBashSessionRequest(startup_timeout=10), run_id="test")
    await s.start()

    async def run(command: str) -> str:
        return (await s.run(BashAction(command=command, timeout=5, check="ignore"))).output

    try:
        await run("sleep 300 > /dev/null 2>&1 &")
        with pytest.raises(CommandTimeoutError):
            await s.run(BashAction(command=DEAF, timeout=1, check="ignore"))
        interrupted = (await s.run(BashInterruptAction(timeout=1, n_retry=1))).output
        await run(":")  # bash reports the kill at the next prompt
        return interrupted, await run("jobs")
    finally:
        await run("kill -9 $(jobs -p) 2>/dev/null; true")
        await s.close()


@pytest.mark.timeout(60)
def test_the_fallback_kills_the_suspended_job_not_job_one():
    interrupted, jobs = asyncio.run(_interrupt_then_jobs())
    assert "Stopped" in interrupted, interrupted
    assert "sleep 300" in jobs and "Running" in jobs, jobs
    assert "sleep 299" not in jobs, jobs
