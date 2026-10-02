"""Diagnostic installation and report failures must not change reward evaluation."""

import asyncio

import pytest

pytest.importorskip("swesmith")

from uni_agent.reward.swe_smith import SWESmithRewardSpec, registry


@pytest.mark.parametrize("failure", ["installation", "read"])
def test_observer_failure_preserves_evaluation_and_closes_sibling(monkeypatch, failure):
    class Profile:
        def get_test_cmd(self, instance, f2p_only=False):
            return "pytest tests/test_example.py", None

        def get_test_files(self, instance):
            return ["tests/test_example.py"], []

    class Env:
        closed = False
        evaluated = False

        async def write_file(self, path, content):
            if failure == "installation" and str(path).endswith(".py"):
                raise OSError("observer disk unavailable")

        async def read_file(self, path, **kwargs):
            raise OSError("observer report unavailable")

        async def communicate(self, command, **kwargs):
            self.evaluated = True
            return "evaluation ran normally"

        async def close(self):
            self.closed = True

    env = Env()
    spec = SWESmithRewardSpec(
        run_id="capture-test",
        metadata={"instance_id": "example", "patch": ""},
        env=env,
        feedback={"enabled": True, "format": "diagnostic", "max_chars": 8000},
        isolate=True,
        env_config={"deployment": {}},
    )
    monkeypatch.setattr(registry, "get_from_inst", lambda instance: Profile())

    async def patch(*args):
        return ""

    async def sibling(*args):
        return env

    monkeypatch.setattr(spec, "_get_interaction_env_patch", patch)
    monkeypatch.setattr(spec, "_start_sibling_env", sibling)
    monkeypatch.setattr(
        spec,
        "_get_eval_report",
        lambda output: {
            "resolved": True,
            "found_eval_status": True,
            "test_status": {
                "FAIL_TO_PASS": {"success": ["tests/test_example.py::test_a"], "failure": []},
                "PASS_TO_PASS": {"success": [], "failure": []},
            },
        },
    )

    async def run():
        return await spec.compute_reward()

    resolved, result = asyncio.run(run())
    assert resolved and result["eval_completed"] and env.evaluated and env.closed
    assert "observer" in result["reward_extra_info"]["feedback"].lower()
