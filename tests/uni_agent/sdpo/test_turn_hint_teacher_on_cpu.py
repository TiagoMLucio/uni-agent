"""The turn-hint teacher on a toy tokenizer: its fields agree with the splice called directly
and decode nothing; ``self_distillation.teacher`` names it by ``_target_`` (the packaged
``uni_agent/conf/sdpo_teacher/turn_hints.yaml``, reached from verl's config dir through
``hydra.searchpath``) and it owns its options; and the trainer's teacher-build step writes the
six fields and the batch metrics a turn-hint run logs."""

import inspect
from dataclasses import asdict
from importlib.resources import files
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from tensordict import TensorDict

import uni_agent.conf
import verl
from uni_agent.sdpo import TurnHintTeacher, select_hinted_turns
from uni_agent.sdpo.hints import assistant_header_ids, hint_user_turn_ids
from uni_agent.sdpo.splice import build_spliced_teacher_row, turn_token_mask
from uni_agent.sdpo.turn_hint_teacher import TurnHintBatch
from verl.trainer import main_ppo_sync
from verl.trainer.ppo.sdpo import TeacherInputs, make_teacher
from verl.trainer.ppo.sdpo.batch import trace_weights
from verl.trainer.ppo.sdpo.teacher_meta import DEGENERATE_META
from verl.workers.config.actor import SelfDistillationConfig

VERL_CONFIG_DIR = Path(verl.__file__).parent / "trainer" / "config"
TURN_HINTS_YAML = files(uni_agent.conf) / "sdpo_teacher" / "turn_hints.yaml"


class ToyTokenizer:
    """Character tokens (id = ord), a chat template that renders ``<role>content</>`` per turn
    and ``<assistant>`` as the generation header."""

    eos_token = "<eos>"
    pad_token_id = 0

    def __init__(self):
        self.decode_calls = 0

    @staticmethod
    def render(messages, add_generation_prompt):
        text = "".join(f"<{m['role']}>{m['content']}</>" for m in messages)
        return text + "<assistant>" if add_generation_prompt else text

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def decode(self, ids, skip_special_tokens=False):
        self.decode_calls += 1
        return "".join(chr(int(i)) for i in ids)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kwargs):
        assert not tokenize
        return self.render(messages, add_generation_prompt)


def ids(text):
    return torch.tensor([ord(c) for c in text], dtype=torch.int64)


HEADER = "<assistant>"
PROMPT = ids("<user>task</>" + HEADER)
# turn 0, an observation, turn 1 (reasoning then a tool call)
TURN0, OBS, TURN1 = "abc</>", "<user>obs</>" + HEADER, "def<tool_call>x</>"
RESPONSE = ids(TURN0 + OBS + TURN1)
SPANS = [[0, 0, len(TURN0)], [1, len(TURN0 + OBS), len(TURN0 + OBS + TURN1)]]
SEGMENT_PROMPT = [{"role": "user", "content": "condensed"}, {"role": "assistant", "content": "so far"}]


def turn_hints_options(**overrides) -> dict:
    """The teacher block as the config group file ships it."""
    options = yaml.safe_load(TURN_HINTS_YAML.read_text())
    options.update(overrides)
    return options


def _inputs(extra_fields, uids, seq_scores, feedback, responses=None, traj_of_row=None):
    n = len(extra_fields)
    responses = responses or [RESPONSE.clone() for _ in range(n)]
    mask = [torch.ones(r.shape[0], dtype=torch.int64) for r in responses]
    for m in mask:
        m[1] = 0  # a tool-observation token the student never wrote
    return TeacherInputs(
        prompts=[PROMPT.clone() for _ in range(n)],
        responses=responses,
        response_mask=mask,
        raw_prompts=[
            [{"role": "system", "content": "sys"}, {"role": "user", "content": f"task {i}"}] for i in range(n)
        ],
        uids=list(uids),
        seq_scores=list(seq_scores),
        feedback=list(feedback),
        extra_fields=extra_fields,
        traj_of_row=traj_of_row or [f"{uid}_{i}" for i, uid in enumerate(uids)],
    )


def test_turn_hint_teacher_matches_the_splice_and_decodes_nothing():
    tok = ToyTokenizer()
    cfg = SelfDistillationConfig(teacher=turn_hints_options())
    teacher = make_teacher(cfg, tok, max_prefix_len=4096)
    assert isinstance(teacher, TurnHintTeacher) and teacher.needs_prompts
    assert cfg.teacher["_target_"] == "uni_agent.sdpo.TurnHintTeacher"
    assert teacher.max_prefix_len == 4096
    extra = [
        {"turn_spans": SPANS, "turn_hints": [[0, "h0"], [1, "h1"]]},
        {"turn_spans": SPANS, "turn_hints": []},
        {"turn_spans": SPANS, "turn_hints": [[1, "h2"]], "segment_index": 1, "segment_prompt": SEGMENT_PROMPT},
    ]
    inputs = _inputs(extra, ["a", "a", "b"], [0.0, 0.0, 0.0], [None] * 3)

    out = teacher.build(inputs)

    assert tok.decode_calls == 0
    assert set(out.fields) == {"teacher_input_ids", "teacher_seq_meta", "self_distillation_mask", "loss_mask"}
    assert out.hinted_per_row == [select_hinted_turns(ef, RESPONSE.shape[0]) for ef in extra]
    assert [len(h) for h in out.hinted_per_row] == [2, 0, 1]

    header = torch.tensor(assistant_header_ids(tok), dtype=torch.int64)
    assert torch.equal(header, ids(HEADER))
    for row in (0, 2):
        hinted = out.hinted_per_row[row]
        hint_ids = [
            torch.tensor(hint_user_turn_ids(tok, teacher.turn_hint_template.format(hint=h.text)), dtype=torch.int64)
            for h in hinted
        ]
        seq, meta, fallbacks, spans = build_spliced_teacher_row(PROMPT, RESPONSE, hinted, hint_ids, 4096, header)
        assert fallbacks == 0
        assert torch.equal(out.fields["teacher_input_ids"][row], seq)
        assert out.fields["teacher_seq_meta"][row].tolist() == meta
        mask = turn_token_mask(RESPONSE.shape[0], spans)
        assert torch.equal(out.fields["self_distillation_mask"][row], mask)
        assert torch.equal(out.fields["loss_mask"][row], inputs.response_mask[row] * mask.to(torch.int64))
    assert out.fields["loss_mask"][0][1] == 0, "observation tokens stay out of the loss"

    assert out.fields["teacher_seq_meta"][1].tolist() == DEGENERATE_META
    assert torch.equal(out.fields["teacher_input_ids"][1], torch.cat([PROMPT[-1:], RESPONSE[:1]]))
    assert out.fields["self_distillation_mask"][1].sum() == 0 and out.fields["loss_mask"][1].sum() == 0

    assert out.metrics == {
        "self_distillation/hinted_sample_fraction": 2 / 3,
        "self_distillation/hinted_turns_per_sample": 1.5,
        "self_distillation/hint_injection_fallbacks": 0,
        "self_distillation/teacher_prefix_clips": 0,
    }


def test_turn_hint_teacher_counts_one_fallback_per_hint_without_its_header():
    tok = ToyTokenizer()
    teacher = make_teacher(SelfDistillationConfig(teacher=turn_hints_options()), tok, max_prefix_len=4096)
    mid_turn = [[0, 1, len(TURN0)], SPANS[1]]
    extra = [
        # a span that starts mid-turn is not preceded by the assistant header: one fallback
        {"turn_spans": mid_turn, "turn_hints": [[0, "h0"], [1, "h1"]]},
        {"turn_spans": mid_turn, "turn_hints": [[0, "h2"]]},
        {"turn_spans": SPANS, "turn_hints": [[0, "h3"], [1, "h4"]]},
    ]
    out = teacher.build(_inputs(extra, ["a", "b", "c"], [0.0] * 3, [None] * 3))
    assert out.metrics["self_distillation/hint_injection_fallbacks"] == 2
    assert [len(h) for h in out.hinted_per_row] == [2, 1, 2]
    assert torch.equal(out.fields["self_distillation_mask"][1][1 : len(TURN0)], torch.ones(len(TURN0) - 1))


def test_turn_hint_teacher_counts_a_prefix_the_budget_clipped():
    """A prompt longer than ``max_prefix_len`` is left-truncated, so the teacher scores the turn
    from a shorter history than the student wrote it from. Counted, and only on hinted rows."""
    tok = ToyTokenizer()
    teacher = make_teacher(SelfDistillationConfig(teacher=turn_hints_options()), tok, max_prefix_len=4096)
    extra = [{"turn_spans": SPANS, "turn_hints": [[0, "h0"]]}, {"turn_spans": SPANS, "turn_hints": []}]
    inputs = _inputs(extra, ["a", "b"], [0.0, 0.0], [None, None])
    whole = teacher.build(inputs)
    assert whole.metrics["self_distillation/teacher_prefix_clips"] == 0

    teacher.max_prefix_len = PROMPT.shape[0] - 1
    clipped = teacher.build(inputs)
    assert clipped.metrics["self_distillation/teacher_prefix_clips"] == 1
    # one token short, and it is the oldest one that went
    assert torch.equal(clipped.fields["teacher_input_ids"][0], whole.fields["teacher_input_ids"][0][1:])


def test_turn_hint_teacher_trajectory_metrics():
    """``trajectory_metrics`` reads hint reach and hint placement off the batch, pooled per
    trajectory; the row weights are each row's share of its trajectory's supervision."""
    tok = ToyTokenizer()
    teacher = make_teacher(SelfDistillationConfig(teacher=turn_hints_options()), tok, max_prefix_len=4096)
    # a three-turn response for the two-hint row, so its hint positions are not only 0 and 1
    response3 = ids(TURN0 + OBS + TURN1 + OBS + TURN1)
    start1, start2 = len(TURN0 + OBS), len(TURN0 + OBS + TURN1 + OBS)
    spans3 = [SPANS[0], [1, start1, start1 + len(TURN1)], [2, start2, start2 + len(TURN1)]]
    extra = [
        {"turn_spans": spans3, "turn_hints": [[0, "h0"], [1, "h1"]]},
        {"turn_spans": SPANS, "turn_hints": [], "segment_index": 0},
        {"turn_spans": SPANS, "turn_hints": [[1, "h2"]], "segment_index": 1, "segment_prompt": SEGMENT_PROMPT},
        {"turn_spans": SPANS, "turn_hints": [[0, "h3"]]},
        {"turn_spans": SPANS, "turn_hints": []},
    ]
    responses = [response3] + [RESPONSE.clone() for _ in range(4)]
    traj_of_row = ["t0", "t1", "t1", "t2", "t3"]
    inputs = _inputs(extra, ["a", "b", "b", "c", "d"], [0.0] * 5, [None] * 5, responses, traj_of_row)

    out = teacher.build(inputs)

    assert isinstance(out, TurnHintBatch)
    assert [len(hinted) for hinted in out.hinted_per_row] == [2, 0, 1, 1, 0]

    supervised_per_row = [float(m.sum()) for m in out.fields["loss_mask"].unbind()]
    assert supervised_per_row == [len(TURN0) - 1 + len(TURN1), 0.0, len(TURN1), len(TURN0) - 1, 0.0]
    weights = trace_weights(supervised_per_row, traj_of_row)
    # every supervised row is its trajectory's only supervised segment
    assert weights == pytest.approx([1.0, 0.0, 1.0, 1.0, 0.0])

    metrics = teacher.trajectory_metrics(out, inputs, supervised_per_row, weights)
    # t0 spans steps 0..2 with hints at 0 and 1 (relative 0 and 0.5), t1 pools its two
    # segments' steps 0..1 and is hinted at 1, t2 at 0 of 0..1, t3 is unhinted
    assert metrics == pytest.approx({
        "self_distillation/hinted_trace_fraction": 3 / 4,
        "self_distillation/hinted_turns_per_trace": 4 / 3,
        "self_distillation/hint_position_mean": (0.0 + 0.5 + 1.0 + 0.0) / 4,
        "self_distillation/hint_position_median": 0.5,
        "self_distillation/hint_position_first_half": 3 / 4,
        "self_distillation/hint_in_last_two_turns": 3 / 4,
        "self_distillation/hint_gap_mean": 1.0,
        "self_distillation/hint_adjacent_fraction": 1.0,
    })


def test_turn_hint_options_are_the_yaml_keys_and_validated_at_construction():
    tok = ToyTokenizer()
    options = turn_hints_options()
    params = inspect.signature(TurnHintTeacher.__init__).parameters
    trainer_provided = {"self", "tokenizer", "max_prefix_len", "apply_chat_template_kwargs", "success_reward_threshold"}
    assert set(options) - {"_target_"} == set(params) - trainer_provided
    assert set(options) == {"_target_", "turn_hint_template", "chat_template_kwargs"}
    assert options["chat_template_kwargs"] == {}

    teacher = make_teacher(
        SelfDistillationConfig(teacher=turn_hints_options(chat_template_kwargs={"enable_thinking": False})),
        tok,
        max_prefix_len=4096,
        apply_chat_template_kwargs={"a": 1},
    )
    assert teacher.template_kwargs == {"enable_thinking": False} and teacher.apply_chat_template_kwargs == {"a": 1}
    assert teacher.success_reward_threshold == 1.0

    def build(**overrides):
        return make_teacher(SelfDistillationConfig(teacher=turn_hints_options(**overrides)), tok, max_prefix_len=4096)

    with pytest.raises(TypeError, match="max_reprompt_len"):
        build(max_reprompt_len=8)
    with pytest.raises(TypeError, match="max_hinted_turns"):
        build(max_hinted_turns=3)  # the reflector's max_selected_turns is the one cap
    with pytest.raises(ValueError, match="turn_hint_template"):
        build(turn_hint_template="no placeholder")
    with pytest.raises(ValueError, match="turn_hint_template"):
        build(turn_hint_template="{hint} and {other}")


def test_launcher_config_builds_the_turn_hint_teacher():
    """The launcher's ``--config-name sdpo`` from verl's config dir, with ``uni_agent.conf`` on
    the hydra searchpath, its group override and the sbatch's ``chat_template_kwargs`` key,
    composes exactly the yaml's block, which the teacher accepts."""
    with initialize_config_dir(config_dir=str(VERL_CONFIG_DIR), version_base=None):
        cfg = compose(config_name="sdpo", overrides=[
            "hydra.searchpath=[pkg://uni_agent.conf]",
            "sdpo_teacher@actor_rollout_ref.actor.self_distillation.teacher=turn_hints",
            "+actor_rollout_ref.actor.self_distillation.teacher.chat_template_kwargs.enable_thinking=False",
        ])
    sd = cfg.actor_rollout_ref.actor.self_distillation
    assert OmegaConf.to_container(sd.teacher, resolve=True) == turn_hints_options(
        chat_template_kwargs={"enable_thinking": False}
    )
    teacher = make_teacher(sd, ToyTokenizer(), max_prefix_len=4096)
    assert isinstance(teacher, TurnHintTeacher)
    assert teacher.template_kwargs == {"enable_thinking": False}
    assert teacher.turn_hint_template == turn_hints_options()["turn_hint_template"]


class TQStub:
    def __init__(self, data):
        self.data = data
        self.select_fields = None
        self.put = None

    def kv_batch_get(self, keys, partition_id, select_fields):
        self.select_fields = list(select_fields)
        return {k: self.data[k] for k in select_fields}

    def kv_batch_put(self, keys, partition_id, fields):
        assert isinstance(fields, TensorDict)
        self.put = (list(keys), fields)


TIMINGS = dict(
    loop_wall=10.0, generate_sequences=4.0, tool_calls=2.0, env_setup=1.0, reward_eval=0.5, reflect=0.25,
    num_preempted=1, eval_completed=1, capped_turns=0,
)


def test_trainer_turn_hints_batch_fields_and_metrics(monkeypatch):
    """One trainer build over a batch with a two-hint row, a condensed trajectory whose only
    hint sits on its second segment, an unhinted successful sibling, a row with no extra_fields
    and a row with a first-turn hint."""
    # key, uid, reward, feedback, extra_fields
    rows = [
        ("u1_0_0", "u1", 0.0, "fb0", dict(turn_spans=SPANS, turn_hints=[[0, "h0"], [1, "h1"]],
                                          segment_index=0, num_segments=1, traj_exit_reason="finished",
                                          timings=TIMINGS)),
        ("u1_1_0", "u1", 1.0, None, dict(turn_spans=SPANS, turn_hints=[], segment_index=0, num_segments=2,
                                         traj_exit_reason="submitted")),
        ("u1_1_1", "u1", 1.0, None, dict(turn_spans=SPANS, turn_hints=[[1, "h2"]], segment_index=1,
                                         num_segments=2, segment_prompt=SEGMENT_PROMPT)),
        ("u2_0_0", "u2", 0.0, "   ", None),
        ("u2_1_0", "u2", 0.0, "fb4", dict(turn_spans=SPANS, turn_hints=[[0, "h3"]], segment_index=0,
                                          num_segments=1, traj_exit_reason="finished")),
    ]
    keys = [r[0] for r in rows]
    n = len(rows)
    inputs = _inputs([{} for _ in rows], [r[1] for r in rows], [r[2] for r in rows], [None] * n)
    rm_scores = []
    for r in rows:
        score = torch.zeros(RESPONSE.shape[0], dtype=torch.float32)
        score[-1] = r[2]
        rm_scores.append(score)
    extra_fields = [None if ef is None else dict(ef, reward_extra_info={"feedback": fb}) for _, _, _, fb, ef in rows]
    data = {
        "responses": torch.nested.nested_tensor(inputs.responses, layout=torch.jagged),
        "response_mask": torch.nested.nested_tensor(inputs.response_mask, layout=torch.jagged),
        "prompts": torch.nested.nested_tensor(inputs.prompts, layout=torch.jagged),
        "rm_scores": torch.nested.nested_tensor(rm_scores, layout=torch.jagged),
        "uid": inputs.uids,
        "raw_prompt": inputs.raw_prompts,
        "extra_fields": extra_fields,
    }
    stub = TQStub(data)
    monkeypatch.setattr(main_ppo_sync, "tq", stub)

    sd = OmegaConf.create(asdict(SelfDistillationConfig(
        success_reward_threshold=0.5,
        include_environment_feedback=True,
        teacher=turn_hints_options(),
    )))
    tok = ToyTokenizer()
    trainer = object.__new__(main_ppo_sync.PPOTrainer)
    trainer.config = OmegaConf.create(
        {"actor_rollout_ref": {"actor": {"policy_loss": {"loss_mode": "sdpo"}, "self_distillation": sd}}}
    )
    trainer.tokenizer = tok
    trainer.sdpo_teacher = make_teacher(sd, tok, max_prefix_len=4096)
    metrics = {}
    trainer._maybe_build_self_distillation_batch(SimpleNamespace(keys=keys, partition_id="train"), metrics)

    assert tok.decode_calls == 0
    assert stub.select_fields == [
        "responses", "rm_scores", "raw_prompt", "uid", "extra_fields", "response_mask", "prompts"
    ]
    put_keys, fields = stub.put
    assert put_keys == keys
    assert set(fields.keys()) == {
        "teacher_input_ids", "teacher_seq_meta", "self_distillation_mask", "loss_mask", "trace_weight", "traj_id",
        "row_id"
    }
    # supervised tokens: turn 0 minus the observation token at index 1, plus the whole of turn 1
    supervised = [len(TURN0) - 1 + len(TURN1), 0, len(TURN1), 0, len(TURN0) - 1]
    assert [int(m.sum()) for m in fields["loss_mask"].unbind()] == supervised
    assert fields["traj_id"].squeeze(-1).tolist() == [0, 1, 1, 2, 3]
    # every supervised row is its trajectory's only supervised segment
    assert fields["trace_weight"].squeeze(-1).tolist() == pytest.approx([1.0, 0.0, 1.0, 0.0, 1.0])
    assert fields["teacher_seq_meta"][1].tolist() == DEGENERATE_META
    assert fields["teacher_seq_meta"][3].tolist() == DEGENERATE_META

    tokens = RESPONSE.shape[0] - 1
    turns = len(SPANS)
    expected = {
        "self_distillation/rows_per_step": 5.0,
        "self_distillation/traces_per_step": 4.0,
        "self_distillation/segments_per_trace_max": 2.0,
        "self_distillation/supervised_segments_per_trace_max": 1.0,
        "self_distillation/unsupervised_row_fraction": 2 / 5,
        "self_distillation/unsupervised_row_tokens": 2.0 * tokens,
        "self_distillation/supervised_row_tokens": 3.0 * tokens,
        "self_distillation/reprompt_sample_fraction": 3 / 5,
        "rollout/generated_tokens": 4.0 * (len(TURN0) + len(TURN1)),
        "rollout/generated_tokens_per_trace": 1.0 * (len(TURN0) + len(TURN1)),
        "self_distillation/hinted_sample_fraction": 3 / 5,
        "self_distillation/hinted_turns_per_sample": 4 / 3,
        "self_distillation/hint_injection_fallbacks": 0,
        "self_distillation/teacher_prefix_clips": 0,
        "rollout/condensed_trace_fraction": 1 / 4,
        "rollout/segments_per_trace": 5 / 4,
        "rollout/solve_rate_1seg": 0.0,
        "rollout/trace_fraction_1seg": 3 / 4,
        "rollout/solve_rate_2seg": 1.0,
        "rollout/trace_fraction_2seg": 1 / 4,
        "rollout/exit_finished_fraction": 2 / 4,
        "rollout/solve_rate_exit_finished": 0.0,
        "rollout/exit_submitted_fraction": 1 / 4,
        "rollout/solve_rate_exit_submitted": 1.0,
        "rollout/turns_in_segment_0": float(turns),
        "rollout/turns_in_segment_1": float(turns),
        "self_distillation/hinted_trace_fraction": 3 / 4,
        "self_distillation/hinted_turns_per_trace": 4 / 3,
        # hints at steps 0 and 1 of u1_0, 1 of u1_1, 0 of u2_1, every trajectory spanning steps 0..1
        "self_distillation/hint_position_mean": 0.5,
        "self_distillation/hint_position_median": 1.0,
        "self_distillation/hint_position_first_half": 0.5,
        "self_distillation/hint_in_last_two_turns": 1.0,
        "self_distillation/hint_gap_mean": 1.0,
        "self_distillation/hint_adjacent_fraction": 1.0,
    }
    # the one trajectory with timings sets every mean, max and quantile
    total = TIMINGS["loop_wall"] + TIMINGS["env_setup"] + TIMINGS["reward_eval"] + TIMINGS["reflect"]
    unattributed = TIMINGS["loop_wall"] - TIMINGS["generate_sequences"] - TIMINGS["tool_calls"]
    for key in ("generate_sequences", "tool_calls", "condense", "parse_action", "tokenize_observations",
                "loop_wall", "env_setup", "reward_eval", "reflect"):
        expected[f"traj_time/{key}_mean"] = float(TIMINGS.get(key, 0.0))
        expected[f"traj_time/slowest_{key}"] = float(TIMINGS.get(key, 0.0))
    expected.update({
        "rollout/preempted_reported_fraction": 1.0,
        "rollout/preempted_mean": 1.0,
        "rollout/preempted_max": 1.0,
        "rollout/preempted_trace_fraction": 1.0,
        "traj_time/unattributed_mean": unattributed,
        "traj_time/total_mean": total,
        "traj_time/total_max": total,
        "traj_time/total_p50": total,
        "traj_time/total_p90": total,
        "traj_time/unattributed_share": unattributed / total,
        "reward_health/eval_completed_fraction": 1.0,
        "reward_health/capped_turns_mean": 0.0,
        "reward_health/capped_rollouts_fraction": 0.0,
    })
    assert metrics == pytest.approx(expected)
    assert isinstance(metrics["self_distillation/hint_injection_fallbacks"], int)
