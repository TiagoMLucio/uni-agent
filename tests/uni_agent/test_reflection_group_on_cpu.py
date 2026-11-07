"""A task's attempts meeting so its failures' reflections read a digest of its successes: the folder they meet in,
the digest call, and the agent loop's group step."""

import asyncio
import json
import sys
import types

import pytest

from uni_agent.reflection import group as group_module
from uni_agent.reflection.api import MAX_DIGEST_SUCCESSES, NO_DIGEST, ApiReflector
from uni_agent.reflection.base import ReflectionFailed
from uni_agent.reflection.group import TaskGroup

from .test_reflection_api_on_cpu import TASK, TURNS, Policy, fake_openai


@pytest.fixture(autouse=True)
def fast_polls(monkeypatch):
    monkeypatch.setattr(group_module, "POLL_S", 0.01)


def notes_reply(*turns):
    return json.dumps({"notes": [{"group": "G1", "trigger": "", "turn": t, "quote": "q", "evidence": "e",
                                  "note": f"note {t}"} for t in turns], "no_note_reason": ""})


DIGEST_CALL = {"id": "digest", "per": "task", "parse": "digest", "system": "dsys", "user": "{task}|{gold}|{successes}"}
NOTES_CALL = {"id": "notes", "parse": "grouped_notes", "system": "sys", "user": "{task}|{turns}|{gold}|{digest}"}


def digest_config(**over):
    return ApiReflector.Config(**{"name": "openai", "model": "gpt-6-luna", "enabled": True, "max_selected_turns": 5,
                                  "include_agent_patch": False, "include_exec_feedback": False,
                                  "calls": [DIGEST_CALL, NOTES_CALL], **over})


def success(k, turns=TURNS):
    return {"turns": turns, "agent_patch": f"diff --git a/a.py b/a.py\n+fix{k}"}


# --- the folder the attempts meet in ---------------------------------------------------------------


def test_the_group_completes_once_every_attempt_has_written(tmp_path):
    async def go():
        members = [TaskGroup(tmp_path, 3, f"m{i}") for i in range(3)]
        await members[0].publish(False)
        await members[1].publish(True, success(1))
        early = await members[0].wait(timeout=0.0)
        await members[2].publish(True, success(2))
        return early, await members[0].wait(timeout=5.0), await members[0].attempts(["m1", "m2"])

    (early_names, early_written, early_done), (names, written, done), attempts = asyncio.run(go())
    assert (early_names, early_written, early_done) == (["m1"], 2, False), "the backstop returns what is there"
    assert (names, written, done) == (["m1", "m2"], 3, True)
    assert [a["agent_patch"][-4:] for a in attempts] == ["fix1", "fix2"]
    # a success's attempt is on disk before its outcome, so a reader never finds the outcome without it
    assert sorted(p.name for p in (tmp_path / "successes").iterdir()) == ["m1.json", "m2.json"]


def test_one_failure_writes_the_digest_and_the_others_read_it(tmp_path):
    calls = []

    async def write():
        calls.append(1)
        await asyncio.sleep(0.05)
        return "how they did it"

    async def go():
        return await asyncio.gather(*(TaskGroup(tmp_path, 4, f"f{i}").digest(write, timeout=5.0) for i in range(3)))

    results = asyncio.run(go())
    assert calls == [1], "written once per task"
    assert sorted(r[2] for r in results) == [False, False, True]
    assert {r[0] for r in results} == {"how they did it"} and all(r[1] == "" for r in results)
    assert json.loads((tmp_path / "digest.json").read_text())["digest"] == "how they did it"


def test_a_failed_digest_is_recorded_and_every_reader_goes_on_without_it(tmp_path):
    async def write():
        raise ReflectionFailed("digest: render over budget at every shrink level")

    async def go():
        wrote = await TaskGroup(tmp_path, 2, "a").digest(write, timeout=1.0)
        read = await TaskGroup(tmp_path, 2, "b").digest(write, timeout=1.0)
        return wrote, read

    (text, error, wrote), (text2, error2, wrote2) = asyncio.run(go())
    assert text is None and wrote and "over budget" in error
    assert text2 is None and not wrote2 and error2 == error


def test_a_reader_gives_up_when_the_writer_never_finishes(tmp_path):
    (tmp_path / "digest.lock").touch()
    text, error, wrote = asyncio.run(TaskGroup(tmp_path, 2, "b").digest(None, timeout=0.05))
    assert text is None and not wrote and "no digest within" in error


# --- the digest call -----------------------------------------------------------------------------


def test_a_digest_call_is_first_and_feeds_a_call_that_names_it():
    cfg = digest_config()
    assert cfg.digest_call.id == "digest" and [c.id for c in cfg.trace_calls] == ["notes"]
    with pytest.raises(ValueError, match="first call"):
        digest_config(calls=[NOTES_CALL, DIGEST_CALL])
    with pytest.raises(ValueError, match="parses its answer as digest"):
        digest_config(calls=[dict(DIGEST_CALL, parse="hints"), NOTES_CALL])
    with pytest.raises(ValueError, match="cannot be given"):
        digest_config(calls=[dict(DIGEST_CALL, user="{successes}|{turns}"), NOTES_CALL])
    with pytest.raises(ValueError, match="no call names"):
        digest_config(calls=[DIGEST_CALL, dict(NOTES_CALL, user="{task}|{turns}|{gold}")])
    with pytest.raises(ValueError, match="failed_only"):
        digest_config(failed_only=False)
    with pytest.raises(ValueError, match="cannot be given"):
        digest_config(calls=[NOTES_CALL])
    from uni_agent.reflection.pipeline import PipelineReflectionConfig

    with pytest.raises(ValueError, match="per=task"):
        PipelineReflectionConfig(name="pipeline", calls=[
            {"id": "d", "per": "task", "system": "s", "user": "{task}"},
            {"id": "b", "system": "s", "user": "{turns}", "parse": "hints"}])


def run_digest(monkeypatch, replies, successes, policy=None, **over):
    seen = []
    monkeypatch.setitem(sys.modules, "openai", fake_openai(replies, seen))
    reflector = ApiReflector(policy or Policy(), digest_config(**over))
    return asyncio.run(reflector.write_digest(task=TASK, gold="diff --git a/a.py b/a.py\n+x", successes=successes)), \
        seen, reflector


def test_the_digest_reads_at_most_three_successes_and_their_patches(monkeypatch):
    digest, seen, reflector = run_digest(monkeypatch, [json.dumps({"digest": "  all read a.py  "})],
                                         [success(k) for k in range(1, 6)])
    assert digest == "all read a.py"
    user = seen[0]["input"]
    assert user.startswith("rows are dropped|")
    assert [f"<successful_attempt S{k}>" in user for k in range(1, 6)] == [True] * MAX_DIGEST_SUCCESSES + [False] * 2
    assert "+fix3" in user and "+fix4" not in user and "### Turn 0" in user
    assert seen[0]["text"]["format"]["name"] == "success_digest"
    assert reflector.call_metrics()["reflect_digest_attempts"] == 3.0


def test_the_digest_drops_an_attempt_when_they_do_not_fit_together(monkeypatch):
    """Ten turns of 3000 characters each: under the ladder's floors three attempts still overflow, two fit."""
    long = [{"step": i, "response": "x" * 3000, "tools": [{"observation": "o"}]} for i in range(10)]
    digest, seen, reflector = run_digest(monkeypatch, [json.dumps({"digest": "d"})],
                                         [success(k, long) for k in range(1, 4)],
                                         policy=Policy(max_model_len=20000), max_output_tokens=1000)
    assert digest == "d" and len(seen) == 1
    assert "<successful_attempt S2>" in seen[0]["input"] and "<successful_attempt S3>" not in seen[0]["input"]
    assert reflector.call_metrics()["reflect_digest_attempts"] == 2.0


def test_an_unusable_digest_fails(monkeypatch):
    with pytest.raises(ReflectionFailed, match="no usable reply"):
        run_digest(monkeypatch, [json.dumps({"digest": " "})], [success(1)])


def test_the_notes_call_reads_the_digest_or_none(monkeypatch):
    for digest, shown in (("they ran the issue's example", "they ran the issue's example"), (None, NO_DIGEST)):
        seen = []
        monkeypatch.setitem(sys.modules, "openai", fake_openai([notes_reply(1)], seen))
        reflector = ApiReflector(Policy(), digest_config())
        hints = asyncio.run(reflector.reflect_trajectory(task=TASK, turns=TURNS, gold="g", feedback="f",
                                                         digest=digest))
        assert hints == {1: "note 1"}
        assert seen[0]["input"].endswith("|" + shown)


# --- the agent loop's group step (imports verl) -------------------------------------------------------


class StubReflector:
    """What the loop asks of the reflector, recorded."""

    def __init__(self, log):
        self.log = log

    async def write_digest(self, task, gold, successes):
        self.log.append(("digest", [s["agent_patch"] for s in successes]))
        await asyncio.sleep(0.02)
        return "the digest"

    async def reflect_trajectory(self, **kwargs):
        self.log.append(("notes", kwargs.get("digest")))
        return {0: "a note"}

    def call_metrics(self):
        return {"reflect_calls": 1.0}


def _step(idx):
    return types.SimpleNamespace(step_idx=idx, response="r", tool_results=[], exit_reason="turn_done", done=False)


def attempt(tmp_path, name, reward, log, size=3):
    """A fake loop and its graded result, as ``_maybe_reflect`` sees them in a group of ``size``."""
    from uni_agent.agent_loop import UniAgentLoop

    loop = types.SimpleNamespace(
        env=types.SimpleNamespace(privileged_context="gold"), chat_model=None, run_id=name, output_dir=tmp_path / name,
        identity={}, logger=types.SimpleNamespace(critical=print, warning=print),
        group=TaskGroup(tmp_path / "groups" / "uid", size, name), _group_published=False)
    for method in ("_publish_attempt", "_group_digest"):
        setattr(loop, method, types.MethodType(getattr(UniAgentLoop, method), loop))
    result = {
        "reward_score": reward, "resolved": bool(reward), "metrics": {},
        "reward_extra_info": {"feedback": "f", "agent_patch": f"patch-{name}"},
        "messages": [{"role": "user", "content": "t"}], "trajectory": [_step(0), _step(1)],
        "rollout_cache": {"turn_spans": [[0, 0, 2], [1, 2, 4]]},
    }
    return loop, result


def run_group(monkeypatch, tmp_path, rewards):
    from uni_agent import agent_loop
    from uni_agent.agent_loop import UniAgentLoop

    log = []
    monkeypatch.setattr(agent_loop, "load_reflector", lambda *a, **k: StubReflector(log))
    block = digest_config().model_dump()
    attempts = [attempt(tmp_path, f"a{i}", r, log, size=len(rewards)) for i, r in enumerate(rewards)]

    async def go():
        return await asyncio.gather(*(UniAgentLoop._maybe_reflect(loop, result, {"reflection": block}, validate=False)
                                      for loop, result in attempts))

    return asyncio.run(go()), [result for _, result in attempts], log


def test_failures_wait_for_their_task_and_read_one_digest(monkeypatch, tmp_path):
    hints, results, log = run_group(monkeypatch, tmp_path, [0.0, 1.0, 0.0])
    assert hints == [{0: "a note"}, {}, {0: "a note"}], "a success is never reflected"
    assert log.count(("digest", ["patch-a1"])) == 1, "one digest per task, from its success"
    assert [entry for entry in log if entry[0] == "notes"] == [("notes", "the digest")] * 2
    for failure in (results[0], results[2]):
        assert failure["metrics"]["agent/reflect_group_complete"] == 1.0
        assert failure["metrics"]["agent/reflect_group_successes"] == 1.0
        assert failure["metrics"]["agent/reflect_digest"] == 1.0
    assert "agent/reflect_digest" not in results[1]["metrics"]


def test_a_task_with_no_success_is_reflected_without_a_digest(monkeypatch, tmp_path):
    hints, results, log = run_group(monkeypatch, tmp_path, [0.0, 0.0])
    assert hints == [{0: "a note"}] * 2
    assert log == [("notes", None)] * 2
    assert all(r["metrics"]["agent/reflect_digest"] == 0.0 for r in results)
