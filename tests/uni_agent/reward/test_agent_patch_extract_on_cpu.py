"""The reflector's wide patch is a courtesy copy: a sandbox that dies between the eval and that
second git diff must leave the graded row intact (feedback rendered, agent_patch empty) instead
of turning the rollout into agent_loop_failed. Driven through the SWE-bench spec; the SWE-smith
one carries the identical tail.
"""

from __future__ import annotations

from swebench.harness.constants import MAP_REPO_VERSION_TO_SPECS

from uni_agent.reward.base import PATCH_EXTRACT_OK
from uni_agent.reward.swe_bench import SWEBenchRewardSpec


class Env:
    """Serves the graded diff once, then the sandbox is gone."""

    def __init__(self):
        self.isolated_calls = 0

    async def communicate_isolated(self, command, **kwargs):
        self.isolated_calls += 1
        if self.isolated_calls > 1:
            raise RuntimeError("container exited")
        return PATCH_EXTRACT_OK

    async def read_file(self, path):
        return "diff --git a/x.py b/x.py\n+fix\n"

    async def write_file(self, path, content):
        pass

    async def communicate(self, command, **kwargs):
        return "no test output"


def test_wide_patch_failure_keeps_the_graded_row():
    repo = "astropy/astropy"
    version = next(iter(MAP_REPO_VERSION_TO_SPECS[repo]))
    spec = SWEBenchRewardSpec(
        run_id="r",
        metadata={
            "instance_id": "astropy__astropy-1",
            "repo": repo,
            "version": version,
            "base_commit": "abc",
            "test_patch": "",
            "patch": "",
        },
        env=(env := Env()),
        feedback={"enabled": True, "parts": ["summary", "failure_mode"]},
        agent_patch_diff_args="-U10",
    )

    # auto_await runs the coroutine when no loop is live
    resolved, result = spec.compute_reward()

    assert env.isolated_calls == 2, "the graded diff, then the wide one"
    assert resolved is False and result["eval_completed"] is True
    assert result["reward_extra_info"]["agent_patch"] == ""
    assert result["reward_extra_info"]["feedback"]
