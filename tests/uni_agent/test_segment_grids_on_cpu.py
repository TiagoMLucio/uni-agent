"""A condensed rollout saves one grid per segment; an uncondensed one saves none.

Without this dump only the final buffer survives, and a turn from before a condensation has no
grid at all -- so anything scoring a hint offline silently has nothing to score it against.
"""

from __future__ import annotations

import pickle
import types

from uni_agent.agent_loop import SEGMENT_GRID_FIELDS, UniAgentLoop


def _cache(spans, n_prompt=6, n_resp=4):
    return {
        "prompt_ids": list(range(n_prompt + n_resp)),
        "response_mask": [1] * n_resp,
        "response_logprobs": [-0.5] * n_resp,
        "turn_spans": spans,
        "request_id": "r",
    }


def _result(segments):
    return {
        "trajectory": [],
        "rollout_cache": segments[-1]["rollout_cache"],
        "segments": segments,
        "execution_time": 1.0,
        "messages": [],
        "metrics": {},
    }


def _save(tmp_path, result):
    loop = types.SimpleNamespace(
        output_dir=tmp_path,
        identity={},
        env=types.SimpleNamespace(privileged_context=""),
        _save_interaction_result=None,
    )
    UniAgentLoop._save_interaction_result(loop, result)
    return tmp_path


def test_condensed_rollout_saves_every_segment(tmp_path):
    segments = [
        {"rollout_cache": _cache([[1, 0, 2], [2, 2, 4]]), "prompt_messages": []},
        {"rollout_cache": _cache([[3, 0, 4]]), "prompt_messages": []},
    ]
    out = _save(tmp_path, _result(segments))

    grids = pickle.loads((out / "segment_grids.pkl").read_bytes())
    assert [sorted(s for s, _, _ in g["turn_spans"]) for g in grids] == [[1, 2], [3]]
    assert all(set(g) == set(SEGMENT_GRID_FIELDS) for g in grids)
    # the float array is the bulk of the bytes and is recomputed by whatever needs it
    assert all("response_logprobs" not in g for g in grids)
    # every turn is reachable, which is the whole point
    assert {s for g in grids for s, _, _ in g["turn_spans"]} == {1, 2, 3}


def test_single_segment_writes_only_the_rollout_cache(tmp_path):
    segments = [{"rollout_cache": _cache([[1, 0, 4]]), "prompt_messages": []}]
    out = _save(tmp_path, _result(segments))

    assert not (out / "segment_grids.pkl").exists()
    assert pickle.loads((out / "rollout_cache.pkl").read_bytes())["turn_spans"] == [[1, 0, 4]]
