"""A transient setup failure must cost a retry, not the rollout.

One node-wide stall killed 7 in-flight rollouts at the same instant; each scored 0
for an infrastructure hiccup the agent never saw. Setup builds a fresh sandbox on
every attempt, so retrying is safe and the rollout survives.
"""

import asyncio

import pytest


class FlakyEnv:
    """Fails ``fail_times`` starts, then succeeds. Records lifecycle calls."""

    def __init__(self, ledger, fail_times, exc=TimeoutError):
        self.ledger, self.fail_times, self.exc = ledger, fail_times, exc
        ledger.append("init")

    async def start(self):
        self.ledger.append("start")
        if sum(1 for e in self.ledger if e == "start") <= self.fail_times:
            raise self.exc()

    async def install_tools(self, tools):
        self.ledger.append("install_tools")

    async def close(self):
        self.ledger.append("close")


async def _run_setup(ledger, fail_times, setup_retries, exc=TimeoutError):
    """The agent loop's setup block, reduced to the retry contract it must honor."""
    env = FlakyEnv(ledger, fail_times, exc)
    for attempt in range(setup_retries + 1):
        try:
            async with asyncio.timeout(30):
                await env.start()
                await env.install_tools([])
            return attempt + 1
        except Exception:
            if attempt == setup_retries:
                raise
            try:
                await env.close()
            except Exception:
                pass
            env = FlakyEnv(ledger, fail_times, exc)  # a fresh sandbox, as the loop does
    raise AssertionError("unreachable")


def test_transient_failure_is_retried_with_a_fresh_sandbox():
    ledger = []
    attempts = asyncio.run(_run_setup(ledger, fail_times=1, setup_retries=2))
    assert attempts == 2, "should have succeeded on the second attempt"
    # the broken sandbox is torn down and a new one built before retrying
    assert ledger == ["init", "start", "close", "init", "start", "install_tools"], ledger


def test_healthy_setup_does_not_retry():
    ledger = []
    assert asyncio.run(_run_setup(ledger, fail_times=0, setup_retries=2)) == 1
    assert ledger.count("start") == 1 and "close" not in ledger


def test_persistent_failure_still_raises_after_the_budget():
    ledger = []
    with pytest.raises(TimeoutError):
        asyncio.run(_run_setup(ledger, fail_times=99, setup_retries=2))
    assert ledger.count("start") == 3, "one initial attempt plus two retries"


def test_retry_covers_any_setup_exception_not_just_timeouts():
    ledger = []
    assert asyncio.run(_run_setup(ledger, fail_times=1, setup_retries=2, exc=ConnectionError)) == 2
