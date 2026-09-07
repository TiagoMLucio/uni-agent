"""The turn-hint SDPO teacher: reflector hints spliced into the student's own trajectory, one
teacher sub-row per hinted turn, scored only on that turn.

``hints`` pairs the hints with their turns and renders them under the chat template,
``splice`` builds the spliced teacher row and its mask, ``turn_hint_teacher`` is the
:class:`~verl.trainer.ppo.sdpo.SDPOTeacher` the trainer selects through
``self_distillation.teacher`` (verl's ``sdpo_teacher/turn_hints.yaml``).
"""

from uni_agent.sdpo.hints import HintedTurn, select_hinted_turns
from uni_agent.sdpo.splice import build_spliced_teacher_row
from uni_agent.sdpo.turn_hint_teacher import TurnHintTeacher

__all__ = ["HintedTurn", "TurnHintTeacher", "build_spliced_teacher_row", "select_hinted_turns"]
