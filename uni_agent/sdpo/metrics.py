"""Per-step metrics of a turn-hint batch: how far the hints reach and where they land.

Rows are condensation segments and ``traj_of_row`` names the trajectory each row belongs to.
"""

from collections import defaultdict
from statistics import median

from uni_agent.sdpo.hints import HintedTurn

__all__ = ["hint_metrics", "hint_position_metrics"]


def hint_metrics(hinted_per_row: list[list[HintedTurn]], extra_fields: list[dict], traj_of_row: list) -> dict:
    """Hint reach per trajectory, then where the hints land."""
    hinted_traces = {traj for traj, hinted in zip(traj_of_row, hinted_per_row, strict=True) if hinted}
    out = {
        "self_distillation/hinted_trace_fraction": len(hinted_traces) / len(set(traj_of_row)),
        "self_distillation/hinted_turns_per_trace": (
            sum(len(hinted) for hinted in hinted_per_row) / len(hinted_traces) if hinted_traces else 0.0
        ),
    }
    out.update(hint_position_metrics(hinted_per_row, extra_fields, traj_of_row))
    return out


def hint_position_metrics(hinted_per_row: list[list[HintedTurn]], extra_fields: list[dict], traj_of_row: list) -> dict:
    """Where in a trajectory the hints land: late hints supervise turns nothing can still fix.

    Step ranges are pooled per trajectory, since a condensed trace splits its turns across
    rows and a per-row range would call every segment-final hint a last-turn hint.
    """
    traj_steps = defaultdict(list)
    for traj, ef in zip(traj_of_row, extra_fields, strict=True):
        traj_steps[traj].extend(int(span[0]) for span in ef.get("turn_spans") or [])
    rel, turn = [], []
    for hinted, traj in zip(hinted_per_row, traj_of_row, strict=True):
        steps = sorted(traj_steps.get(traj) or [0])
        lo, hi = steps[0], steps[-1]
        span_len = max(hi - lo, 1)
        rel.extend((hint.step - lo) / span_len for hint in hinted)
        turn.extend(hint.step - lo + 1 for hint in hinted)
    if not rel:
        return {}
    return {
        "self_distillation/hint_position_mean": sum(rel) / len(rel),
        "self_distillation/hint_position_median": float(median(rel)),
        # the turn number itself (1 = first turn): how deep into the context the hint lands
        "self_distillation/hint_turn_mean": sum(turn) / len(turn),
        "self_distillation/hint_turn_median": float(median(turn)),
    }
