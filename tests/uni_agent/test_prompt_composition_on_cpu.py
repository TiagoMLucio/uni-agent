"""The opening messages come from the config, or from the row, and never from both.

Data prep used to bake the finished messages into the parquet, so changing a word meant
regenerating the data and the val parquet is frozen. With a `prompts:` block in the agent config
the loop composes them per rollout instead, and the prompt becomes an experiment axis.

The composed list has to be byte-identical to what the baked one would have been. Everything
downstream is built from the recorded token buffer, so a difference of one character is a
different prompt_ids, a different header, and a teacher probing for something that is not there.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("verl.experimental.agent_loop")

from uni_agent.agent_loop import compose_messages, opening_messages, reward_metrics  # noqa: E402
from uni_agent.interaction.interaction import AgentInteraction  # noqa: E402

SYSTEM = "You are a software engineer.\n\nYou work in a container.\n\nOnly your diff is graded."
TASK = (
    "<uploaded_files>\n{workdir}\n</uploaded_files>\n"
    "I have uploaded a {language} repository in {workdir}.\n\n"
    "<issue_description>\n{problem_statement}\n</issue_description>\n"
    "Fix it under {workdir}."
)
VALUES = {
    "family": "swe_repo_fix",
    "workdir": "/testbed",
    "language": "python",
    "problem_statement": "TypeError in astropy.wcs when slicing",
}


def test_the_two_opening_messages():
    system, user = compose_messages({"system": SYSTEM, "task": TASK}, VALUES)
    assert system == {"role": "system", "content": SYSTEM}, "the system text is used verbatim"
    assert user["role"] == "user"
    assert "TypeError in astropy.wcs when slicing" in user["content"]
    assert "{problem_statement}" not in user["content"]


def test_a_value_the_row_does_not_carry_is_loud():
    """A prompt shipping a literal placeholder would corrupt every rollout of the run in silence."""
    with pytest.raises(KeyError) as excinfo:
        compose_messages({"system": SYSTEM, "task": TASK}, {"workdir": "/testbed"})
    message = str(excinfo.value)
    assert "language" in message and "problem_statement" in message, "it names every missing one"
    assert "workdir" in message, "and what the row did carry"


def test_a_value_the_template_does_not_use_is_fine():
    """`family` selects the template upstream and is not in it; extras must not be an error."""
    assert compose_messages({"system": SYSTEM, "task": "{workdir}"}, VALUES)[1]["content"] == "/testbed"


def test_composition_matches_what_data_prep_would_have_baked():
    """Byte-equality with the parquet path, which is the whole safety argument for this change."""
    baked = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": TASK.format(**VALUES)},
    ]
    assert compose_messages({"system": SYSTEM, "task": TASK}, VALUES) == baked


PROMPTS_DIR = Path(__file__).parents[3] / "base" / "prompts"


@pytest.mark.skipif(not PROMPTS_DIR.is_dir(), reason="needs the orchestration checkout's prompts")
def test_composition_matches_the_real_prompt_files():
    """The same equality against the text a run actually ships, not a fixture that resembles it."""
    family = "swe_repo_fix"
    core = (PROMPTS_DIR / "system.txt").read_text().strip()
    block = (PROMPTS_DIR / "families" / f"{family}.system.txt").read_text().strip()
    task = (PROMPTS_DIR / "families" / f"{family}.task.txt").read_text().strip()
    values = {"family": family, "workdir": "/testbed", "language": "python",
              "problem_statement": "the issue text"}

    system = "\n\n".join(part for part in (core, block) if part)
    baked = [{"role": "system", "content": system}, {"role": "user", "content": task.format(**values)}]
    assert compose_messages({"system": system, "task": task}, values) == baked


BAKED = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": TASK.format(**VALUES)}]


def test_without_a_prompts_block_the_row_is_used_exactly_as_before():
    assert opening_messages(None, BAKED, VALUES) == BAKED


def test_the_fallback_copies_the_row_rather_than_aliasing_it():
    """The loop hands these to AgentInteraction, which appends to them for the whole rollout."""
    out = opening_messages(None, BAKED, VALUES)
    assert out is not BAKED


def test_a_prompts_block_composes_and_the_row_is_not_consulted():
    """Mutually exclusive: the baked messages here are wrong on purpose."""
    stale = [{"role": "system", "content": "an older system prompt"}]
    assert opening_messages({"system": SYSTEM, "task": TASK}, stale, VALUES) == BAKED


class _Skills:
    def build_manifest(self):
        return "<skills>\nreview: read it first\n</skills>"


def _interaction(messages):
    it = AgentInteraction.__new__(AgentInteraction)
    it.messages = messages
    it.skills_manager = _Skills()
    return it


def test_the_manifest_lands_in_the_composed_system_message():
    messages = compose_messages({"system": SYSTEM, "task": TASK}, VALUES)
    it = _interaction(messages)
    it.inject_skills_manifest()
    systems = [m for m in it.messages if m["role"] == "system"]
    assert len(systems) == 1, "composing one and appending to it must not make two"
    assert systems[0]["content"].startswith(SYSTEM)
    assert systems[0]["content"].endswith("</skills>")
    assert [m["role"] for m in it.messages] == ["system", "user"]


def test_without_a_system_message_the_manifest_still_makes_exactly_one():
    it = _interaction([{"role": "user", "content": "fix it"}])
    it.inject_skills_manifest()
    assert [m["role"] for m in it.messages] == ["system", "user"]


def test_the_reward_flags_are_absent_when_the_reward_did_not_report_them():
    """The regression for the agent-loop half of the extraction fix: defaulting empty_patch to
    False reads a prediction that was never extracted as an agent that changed nothing."""
    out = reward_metrics({"eval_completed": True}, applied_edits=5.0)
    assert "empty_patch" not in out and "work_lost" not in out
    assert out["eval_completed"] == 1.0 and out["patch_apply_failed"] == 0.0


def test_work_lost_is_an_empty_patch_that_had_edits():
    assert reward_metrics({"empty_patch": True}, applied_edits=5.0)["work_lost"] == 1.0
    assert reward_metrics({"empty_patch": True}, applied_edits=0.0)["work_lost"] == 0.0
    assert reward_metrics({"empty_patch": False}, applied_edits=5.0)["work_lost"] == 0.0


def test_an_empty_patch_that_was_measured_is_reported_even_at_zero():
    out = reward_metrics({"empty_patch": False}, applied_edits=0.0)
    assert out["empty_patch"] == 0.0 and out["work_lost"] == 0.0
