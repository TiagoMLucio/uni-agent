"""An unknown top-level key in the agent yaml is an error, not a silent code default."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

pytest.importorskip("verl.experimental.agent_loop")

from uni_agent.agent_loop import AGENT_CONFIG_KEYS, UniAgentLoop  # noqa: E402

REPO = Path(__file__).parents[2]
EXAMPLE_CONFIGS = sorted(REPO.glob("examples/**/agent_config*.yaml"))


def _loop(config_path: Path) -> UniAgentLoop:
    loop = UniAgentLoop.__new__(UniAgentLoop)
    loop.server_manager = object()
    loop.tokenizer = object()
    loop.config = SimpleNamespace(
        actor_rollout_ref=SimpleNamespace(
            rollout=SimpleNamespace(
                agent=SimpleNamespace(agent_loop_config_path=str(config_path)),
                max_model_len=None,
                prompt_length=1024,
                response_length=1024,
            )
        )
    )
    return loop


def _write(tmp_path: Path, block: dict) -> Path:
    path = tmp_path / "agent_config.yaml"
    path.write_text(yaml.safe_dump([block]))
    return path


def _block(source: Path) -> dict:
    return yaml.safe_load(source.read_text())[0]


@pytest.mark.parametrize("config_path", EXAMPLE_CONFIGS, ids=lambda p: p.parent.name)
def test_shipped_configs_carry_only_known_keys(config_path):
    assert EXAMPLE_CONFIGS, "no example agent configs found"
    config = _loop(config_path)._init_config({})
    assert set(config) - {"model"} <= AGENT_CONFIG_KEYS


def test_a_typo_names_the_key_and_the_intended_one(tmp_path):
    block = _block(EXAMPLE_CONFIGS[0]) | {"contex_budget": 4096}
    with pytest.raises(ValueError) as e:
        _loop(_write(tmp_path, block))._init_config({})
    assert "contex_budget" in str(e.value)
    assert "did you mean context_budget?" in str(e.value)


def test_a_typo_inside_validation_overrides_fails_only_on_validation(tmp_path):
    block = _block(EXAMPLE_CONFIGS[0]) | {"validation_overrides": {"setup_timout": 90}}
    path = _write(tmp_path, block)
    _loop(path)._init_config({})
    with pytest.raises(ValueError, match="setup_timout"):
        _loop(path)._init_config({}, validate=True)
