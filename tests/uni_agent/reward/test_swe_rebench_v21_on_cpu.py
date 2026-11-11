"""SWE-rebench V2.1: the Harbor verifier's files decide the reward, graded in a sibling sandbox."""

import asyncio
import base64
import json
import shutil
import subprocess

import pytest
from uni_agent.reward import swe_rebench_v21 as v21

TEST_SH = """#!/bin/bash
set -uo pipefail

WORKDIR=/werkzeug
TEST_FILES=(tests/test_debug.py 'tests/data/a b.py')
IFS= read -r -d '' HARBOR_TEST_CMD <<'HARBOR_TEST_CMD_EOF'
pytest --no-header -rA --tb=line --color=no -p no:cacheprovider tests/test_debug.py
HARBOR_TEST_CMD_EOF
"""


def added(path, line="x"):
    return f"diff --git a/{path} b/{path}\nnew file mode 100644\n--- /dev/null\n+++ b/{path}\n@@ -0,0 +1 @@\n+{line}\n"


def edited(path, old="a", new="b"):
    return f"diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n@@ -1 +1 @@\n-{old}\n+{new}\n"


EDIT_FILE = edited("src/a.py")


def test_graded_files_and_long_tracebacks():
    assert v21.graded_test_files(TEST_SH) == ["tests/test_debug.py", "tests/data/a b.py"]
    out = v21.long_tracebacks(TEST_SH)
    assert "--tb=long" in out and "--tb=line" not in out


def test_runner_files_and_scratch_tests_are_withheld_unless_the_reference_touches_them():
    patch = EDIT_FILE + edited("tests/conftest.py") + added("conftest.py") + edited("setup.cfg") + \
        added("test_repro.py") + added("pkg/new_module.py") + added("reproduce_issue.py")
    assert v21.withheld_files(patch, EDIT_FILE, []) == ["tests/conftest.py", "conftest.py", "setup.cfg",
                                                        "test_repro.py"]
    # the reference fix changes setup.cfg and adds the test file: a fix may too
    assert v21.withheld_files(patch, EDIT_FILE + edited("setup.cfg") + added("test_repro.py"), []) == [
        "tests/conftest.py", "conftest.py"]
    assert v21.withheld_files(patch, EDIT_FILE, ["test_repro.py", "conftest.py"]) == ["tests/conftest.py", "setup.cfg"]


def test_without_files_keeps_every_other_byte():
    raw = (EDIT_FILE + edited("conftest.py")).encode() + b"diff --git a/b.py b/b.py\n--- a/b.py\n+++ b/b.py\n" \
        b"@@ -1 +1 @@\n-\xe9\r\n+\xe8\r\n"
    kept = v21.without_files(raw.decode("latin-1"), ["conftest.py"]).encode("latin-1")
    assert b"conftest" not in kept and kept.startswith(EDIT_FILE.encode()) and kept.endswith(b"-\xe9\r\n+\xe8\r\n")


def test_eval_report_from_grade_report():
    report = {"resolved": False, "parsed_counts": {"PASSED": 2, "FAILED": 1},
              "fail_to_pass": {"t::f": False}, "pass_to_pass": {"t::a": True, "t::b": True}}
    r = v21.eval_report_from(report)
    assert r["found_eval_status"] and not r["resolved"]
    assert r["test_status"]["FAIL_TO_PASS"] == {"success": [], "failure": ["t::f"]}
    assert r["test_status"]["PASS_TO_PASS"]["success"] == ["t::a", "t::b"]
    assert v21.eval_report_from(None)["found_eval_status"] is False


class FakeEnv:
    def __init__(self, files=None, stdout="test_cmd exit code: 0\nRESOLVED"):
        self.files = dict(files or {})
        self.written = {}
        self.commands = []
        self.stdout = stdout
        self.closed = False

    async def write_file(self, path, content):
        self.written[str(path)] = content

    async def read_file(self, path, **kwargs):
        if str(path) not in self.files:
            raise FileNotFoundError(path)
        return self.files[str(path)]

    async def communicate(self, command, **kwargs):
        # a failed check="raise" closes the sandbox, so no step may rely on raising
        assert kwargs.get("check") != "raise"
        self.commands.append(command)
        return self.stdout

    async def close(self):
        self.closed = True


def run_spec(sibling, raw_patch, monkeypatch, **kwargs):
    agent = FakeEnv()
    spec = v21.SWERebenchV21RewardSpec(
        run_id="t", env=agent, env_config={"deployment": {}},
        metadata={"instance_id": "pallets__werkzeug-2248", "repo": "pallets/werkzeug", "patch": EDIT_FILE,
                  "tests": {"test.sh": TEST_SH, "config.json": "{}", "grade.py": "", "test.patch": ""}},
        **kwargs,
    )

    async def extract(*args):
        return raw_patch

    async def start(*args):
        return sibling

    monkeypatch.setattr(spec, "_extract_patch", extract)
    monkeypatch.setattr(spec, "_start_sibling_env", start)

    async def run():
        return await spec.compute_reward()

    resolved, result = asyncio.run(run())
    return agent, resolved, result


def verifier_files(resolved, reward):
    report = {"resolved": resolved, "parsed_counts": {"PASSED": 1},
              "fail_to_pass": {"t::f": resolved}, "pass_to_pass": {}}
    return {"/logs/verifier/report.json": json.dumps(report), "/logs/verifier/reward.txt": reward,
            "/logs/verifier/test_output.log": "PASSED t::f"}


def sent_patch(env):
    b64 = next(v for k, v in env.written.items() if k.endswith(".diff.b64"))
    return base64.b64decode(b64)


def test_grading_runs_in_the_sibling_on_the_exact_bytes_less_the_withheld_files(monkeypatch):
    sibling = FakeEnv(verifier_files(True, "1\n"))
    raw = EDIT_FILE.encode() + edited("conftest.py").encode() + b"diff --git a/c.py b/c.py\n--- a/c.py\n" \
        b"+++ b/c.py\n@@ -1 +1 @@\n-\xe9\r\n+\xe8\r\n"
    agent, resolved, result = run_spec(sibling, raw, monkeypatch)
    assert resolved and result["eval_completed"] and sibling.closed and not agent.closed
    assert not agent.written and not agent.commands
    assert sent_patch(sibling) == EDIT_FILE.encode() + b"diff --git a/c.py b/c.py\n--- a/c.py\n+++ b/c.py\n" \
        b"@@ -1 +1 @@\n-\xe9\r\n+\xe8\r\n"
    assert result["dropped_agent_files"] == ["conftest.py"]
    assert "--tb=long" in sibling.written["/tests/test.sh"]
    script = next(v for k, v in sibling.written.items() if k.startswith("/tmp/swerebench_eval_"))
    assert "cd /werkzeug" in script and "base64 -d" in script and "bash /tests/test.sh" in script


def test_resolved_needs_the_reward_file_too(monkeypatch):
    _, resolved, _ = run_spec(FakeEnv(verifier_files(True, "0\n")), EDIT_FILE.encode(), monkeypatch)
    assert not resolved


def test_a_prediction_left_empty_by_withholding_is_not_applied(monkeypatch):
    sibling = FakeEnv(verifier_files(False, "0\n"))
    _, resolved, _ = run_spec(sibling, edited("pytest.ini").encode(), monkeypatch)
    script = next(v for k, v in sibling.written.items() if k.startswith("/tmp/swerebench_eval_"))
    assert not resolved and "empty prediction" in script and "base64 -d" not in script


def test_failed_apply_is_reported_and_not_graded(monkeypatch):
    sibling = FakeEnv(verifier_files(True, "1\n"), stdout=v21.APPLY_PATCH_FAIL)
    _, resolved, result = run_spec(sibling, EDIT_FILE.encode(), monkeypatch)
    assert not resolved and result["patch_apply_failed"] and result["eval_completed"] and sibling.closed


def test_missing_report_is_not_completed(monkeypatch):
    sibling = FakeEnv({}, stdout="TEST_PATCH_APPLY_FAILED")
    _, resolved, result = run_spec(sibling, EDIT_FILE.encode(), monkeypatch)
    assert not resolved and not result["eval_completed"]
    assert result["eval_error"] == "TEST_PATCH_APPLY_FAILED"


@pytest.mark.parametrize("fmt", ["legacy", "diagnostic"])
def test_feedback_renders_for_a_failure(monkeypatch, fmt):
    sibling = FakeEnv(verifier_files(False, "0\n"))
    _, resolved, result = run_spec(sibling, EDIT_FILE.encode(), monkeypatch,
                                   feedback={"enabled": True, "format": fmt, "max_chars": 8000})
    assert not resolved and result["feedback_render_failed"] is False
    assert result["reward_extra_info"]["feedback"]
    if fmt == "diagnostic":
        assert any("UNI_AGENT_FEEDBACK_ROOT=/werkzeug" in v for v in sibling.written.values())


def test_gold_apply_keeps_what_applies_and_drops_reject_files():
    env = FakeEnv()
    spec = v21.SWERebenchV21RewardSpec(run_id="t", env=env, metadata={
        "instance_id": "x", "repo": "o/proj", "patch": EDIT_FILE, "tests": {"test.sh": TEST_SH}})

    async def run():
        await spec.apply_gold_patch()

    asyncio.run(run())
    assert "--reject" in env.commands[-1] and "'*.rej'" in env.commands[-1] and "cd /proj" in env.commands[-1]


def test_the_last_rung_never_reverses_a_hunk():
    assert all("--batch" not in c or "--forward" in c for c in v21.GIT_APPLY_CMDS)


@pytest.mark.skipif(not shutil.which("git") or not shutil.which("base64"), reason="needs git and base64")
def test_the_apply_lines_on_a_real_repo(tmp_path):
    """The generated apply lines, run by bash in a git repo whose tree differs from HEAD: the prediction's bytes
    land as sent (a CRLF line included), and a patch that does not apply is reported, never reversed."""
    repo = tmp_path / "repo"
    repo.mkdir()
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-C", str(repo)]
    (repo / "a.py").write_bytes(b"x = 1\r\n")
    (repo / "b.py").write_text("y = 1\n")
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run([*git, "add", "-A"], check=True)
    subprocess.run([*git, "commit", "-qm", "base"], check=True)
    (repo / "=4.0.3").write_text("junk from the image build\n")
    patch = b"diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-x = 1\r\n+x = 2\r\n"

    def run(patch_bytes):
        b64 = tmp_path / "p.diff.b64"
        b64.write_text(base64.b64encode(patch_bytes).decode())
        lines = v21.make_eval_script_list(str(repo), str(b64), str(tmp_path / "p.diff"), empty=False)[:-2]
        return subprocess.run(["bash", "-c", "\n".join(["set -uo pipefail", *lines, "echo APPLIED"])],
                              capture_output=True, text=True)

    out = run(patch)
    assert "APPLIED" in out.stdout and (repo / "a.py").read_bytes() == b"x = 2\r\n"
    out = run(patch)  # applied already: every rung refuses, and the edit stays
    assert v21.APPLY_PATCH_FAIL in out.stdout and (repo / "a.py").read_bytes() == b"x = 2\r\n"
