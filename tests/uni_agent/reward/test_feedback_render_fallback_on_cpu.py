"""A renderer bug costs the feedback's form, not the rollout: the raw test output stands in, and the
failure is logged with the task and reported to the metrics."""

from uni_agent.reward import diagnostic_feedback
from uni_agent.reward.swe_bench import RENDER_FAILED, FeedbackConfig


class Log:
    def __init__(self):
        self.errors = []

    def exception(self, message):
        self.errors.append(message)


def test_a_renderer_error_falls_back_to_the_raw_test_output(monkeypatch):
    def broken(*args, **kwargs):
        raise IndexError("list index out of range")

    monkeypatch.setattr(diagnostic_feedback, "render_diagnostic", broken)
    log = Log()
    output = "collected 4 items\n" + "x" * 5_000 + "\n1 failed, 3 passed"
    text, failed = FeedbackConfig(enabled=True, format="diagnostic", max_chars=500).render_or_raw(
        log, result={"resolved": False}, output=output, instance_id="task-1")
    assert failed and text.startswith(RENDER_FAILED + "\n")
    assert len(text) <= 500 and "collected 4 items" in text and text.endswith("1 failed, 3 passed")
    assert len(log.errors) == 1 and "task-1" in log.errors[0]


def test_a_working_renderer_is_not_flagged():
    log = Log()
    text, failed = FeedbackConfig(enabled=True, format="diagnostic", max_chars=5_000).render_or_raw(
        log, result={"resolved": False}, output="1 failed")
    assert not failed and not (text or "").startswith(RENDER_FAILED) and log.errors == []
