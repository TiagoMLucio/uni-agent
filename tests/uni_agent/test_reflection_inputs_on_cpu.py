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


def _block_for(path, lines, kind="modified"):
    head = {"created": "new file mode 100644\n--- /dev/null\n+++ b/" + path,
            "deleted": "deleted file mode 100644\n--- a/" + path + "\n+++ /dev/null",
            "modified": "--- a/" + path + "\n+++ b/" + path}[kind]
    sign = "-" if kind == "deleted" else "+"
    return f"diff --git a/{path} b/{path}\n{head}\n@@ -1 +1,{lines} @@\n" + "".join(f"{sign}line {i}\n" for i in range(lines))


def test_patch_view_shows_source_changes_and_names_the_scratch():
    from uni_agent.reflection.facts import patch_view

    patch = "".join([
        _block_for("pkg/core.py", 3),
        _block_for("reproduce_issue.py", 40, "created"),
        _block_for("tests/test_fix.py", 25, "created"),
        *(_block_for(f".eggs/dep/mod{i}.py", 10, "created") for i in range(7)),
        _block_for("FIX_SUMMARY.md", 12, "created"),
        _block_for("pkg/compat.py", 8, "created"),
        _block_for("pkg/helper.py", 5, "created"),
        _block_for("pkg/old.py", 30, "deleted"),
    ])
    view = patch_view(patch, reference=_block_for("pkg/helper.py", 5, "created"))
    shown, _, rest = view.partition("Files the attempt created, not shown:")
    assert "diff --git a/pkg/core.py" in shown, "a change to an existing file is shown whole"
    assert "diff --git a/pkg/compat.py" in shown, "a new module of the package is shown"
    assert "diff --git a/pkg/helper.py" in shown, "a file the reference creates too is shown"
    assert "- reproduce_issue.py (40 lines)" in rest and "- tests/test_fix.py (25 lines)" in rest
    assert "- .eggs/ (7 files, 70 lines)" in rest and "- FIX_SUMMARY.md (12 lines)" in rest
    assert "Files the attempt deleted:\n- pkg/old.py (30 lines)" in rest
    assert "line 39" not in view, "a script's body is not shown"


def test_patch_view_of_scratch_only_says_no_existing_file_changed():
    from uni_agent.reflection.facts import patch_view

    view = patch_view(_block_for("reproduce_issue.py", 4, "created"))
    assert view.startswith("(no change to an existing file)")
