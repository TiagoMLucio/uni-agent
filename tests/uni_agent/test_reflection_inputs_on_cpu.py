"""A reflector's privileged inputs must match the fields its stages name."""
import pytest

from uni_agent.reflection import build_reflection_config


def _block(**over):
    call = {"id": "write", "per": "trace", "parse": "hints",
            "system": "write at most {k}", "user": "{task} {gold} {agent_patch} {feedback} {turns}"}
    return {"name": "pipeline", "enabled": True, "calls": [call], **over}


def _user(text, **over):
    block = _block(**over)
    block["calls"][0]["user"] = text
    return block


def test_the_shipped_shape_builds():
    assert build_reflection_config(_block()).include_gold


@pytest.mark.parametrize("flag, text", [
    ("include_gold", "{task} {agent_patch} {feedback}"),
    ("include_agent_patch", "{task} {gold} {feedback}"),
    ("include_exec_feedback", "{task} {gold} {agent_patch}"),
])
def test_an_input_no_stage_names_is_refused(flag, text):
    with pytest.raises(ValueError, match=f"{flag} is on but no call names"):
        build_reflection_config(_user(text))


@pytest.mark.parametrize("flag, field", [
    ("include_gold", "gold"),
    ("include_agent_patch", "agent_patch"),
    ("include_exec_feedback", "feedback"),
])
def test_a_field_whose_input_is_off_is_refused(flag, field):
    with pytest.raises(ValueError, match=f"{flag} is off but a call names"):
        build_reflection_config(_block(**{flag: False}))


def test_delta_carries_both_patches():
    """{delta} is patch_delta(gold, agent_patch), so a stage naming it uses both inputs."""
    assert build_reflection_config(_user("{task} {delta} {feedback}")).include_gold
    with pytest.raises(ValueError, match="include_gold is off"):
        build_reflection_config(_user("{task} {delta} {feedback}", include_gold=False))
    with pytest.raises(ValueError, match="include_agent_patch is off"):
        build_reflection_config(_user("{task} {delta} {feedback}", include_agent_patch=False))


def test_a_disabled_reflector_is_not_checked():
    build_reflection_config(_block(enabled=False, include_gold=False))
