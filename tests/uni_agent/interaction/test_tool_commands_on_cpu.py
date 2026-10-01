"""A ``tools[].commands`` list in the agent config is the editor the model is offered and may use;
unset, the schema is the module's, byte for byte, and every command still runs."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest
from pydantic import ValidationError

from uni_agent.deployment import HostDeploymentConfig
from uni_agent.interaction.behaviour import behaviour_metrics
from uni_agent.interaction.env import AgentEnv, AgentEnvConfig
from uni_agent.interaction.interaction import AgentInteraction, ToolResult, get_logger
from uni_agent.interaction.tool_parser import FunctionCallFormatError
from uni_agent.interaction.tools_manager import ToolsManager, ToolsManagerConfig
from uni_agent.tools import StrReplaceEditorTool, ToolConfig
from uni_agent.tools.str_replace_editor import DESCRIPTION, StrReplaceEditorArguments

TRAINING = ["view", "create", "str_replace"]


def _manager(**editor) -> ToolsManager:
    tools = [ToolConfig(name="str_replace_editor", **editor), ToolConfig(name="execute_bash")]
    return ToolsManager(ToolsManagerConfig(tools=tools))


def _editor(manager: ToolsManager) -> dict:
    return next(s["function"] for s in manager.tools_schemas if s["function"]["name"] == "str_replace_editor")


def _call(function: str, command: str) -> str:
    return (
        f"<tool_call>\n<function={function}>\n<parameter=command>{command}</parameter>\n"
        "<parameter=path>/testbed/a.py</parameter>\n</function>\n</tool_call>"
    )


def test_unset_commands_leave_the_schema_byte_identical():
    module = StrReplaceEditorTool().build_tool_schema(DESCRIPTION, StrReplaceEditorArguments)
    assert json.dumps(_editor(_manager())) == json.dumps(module["function"])
    properties = module["function"]["parameters"]["properties"]
    assert properties["command"]["enum"] == list(StrReplaceEditorTool.commands)
    assert "insert_line" in properties and "`undo_edit`" in DESCRIPTION


def test_enabled_commands_are_all_the_schema_offers():
    function = _editor(_manager(commands=TRAINING))
    properties = function["parameters"]["properties"]
    assert properties["command"]["enum"] == TRAINING
    assert properties["command"]["description"] == (
        "The commands to run. Allowed options are: `view`, `create`, `str_replace`."
    )
    assert list(properties) == ["command", "path", "file_text", "old_str", "new_str", "view_range"]
    assert properties["new_str"]["description"].startswith("Optional parameter of `str_replace` command")
    kept = [line for line in DESCRIPTION.splitlines() if "undo_edit" not in line]
    assert function["description"].splitlines() == kept
    assert "insert" not in json.dumps(function) and "undo_edit" not in json.dumps(function)


def test_a_configured_description_replaces_the_text_not_the_narrowing():
    function = _editor(_manager(commands=TRAINING, description="Edit files."))
    assert function["description"] == "Edit files."
    assert function["parameters"]["properties"]["command"]["enum"] == TRAINING
    assert "insert_line" not in function["parameters"]["properties"]


def test_a_command_the_tool_does_not_have_is_refused_at_config():
    with pytest.raises(ValidationError, match=r"has no commands \['delete'\]"):
        ToolConfig(name="str_replace_editor", commands=["view", "delete"])
    with pytest.raises(ValidationError, match=r"tool 'execute_bash' has no commands"):
        ToolConfig(name="execute_bash", commands=["ls"])
    with pytest.raises(ValidationError):
        ToolConfig(name="str_replace_editor", commands=[])
    with pytest.raises(ValidationError, match="lists a command twice"):
        ToolConfig(name="str_replace_editor", commands=["view", "view"])


@pytest.mark.parametrize("command", ["insert", "undo_edit"])
def test_a_disabled_command_is_an_invalid_action(command):
    with pytest.raises(FunctionCallFormatError) as e:
        asyncio.run(_manager(commands=TRAINING).parse_action(_call("str_replace_editor", command)))
    assert str(e.value) == (
        f"Invalid action: command '{command}' is not enabled for function 'str_replace_editor'.\n"
        "Allowed commands for function 'str_replace_editor': ['view', 'create', 'str_replace']."
    )


def test_a_call_without_a_command_is_refused_too():
    without = _call("str_replace_editor", "view").replace("<parameter=command>view</parameter>\n", "")
    for call in (without, _call("str_replace_editor", "null")):
        with pytest.raises(FunctionCallFormatError, match="command 'None' is not enabled"):
            asyncio.run(_manager(commands=TRAINING).parse_action(call))
    _, calls = asyncio.run(_manager().parse_action(without))
    assert "command" not in calls[0].function.arguments


def test_a_structured_call_is_refused_the_same_way():
    arguments = {"command": "insert", "path": "/testbed/a.py", "insert_line": 1, "new_str": "x"}
    calls = [{"id": "c1", "function": {"name": "str_replace_editor", "arguments": json.dumps(arguments)}}]
    with pytest.raises(FunctionCallFormatError, match="command 'insert' is not enabled"):
        asyncio.run(_manager(commands=TRAINING).parse_structured_action("", calls))


def test_enabled_unset_and_other_tools_commands_still_parse():
    _, calls = asyncio.run(_manager(commands=TRAINING).parse_action(_call("str_replace_editor", "view")))
    assert calls[0].function.arguments["command"] == "view"
    _, calls = asyncio.run(_manager().parse_action(_call("str_replace_editor", "undo_edit")))
    assert calls[0].function.arguments["command"] == "undo_edit"
    # execute_bash's `command` is a shell line, not one of the editor's
    bash = _call("execute_bash", "insert").replace("<parameter=path>/testbed/a.py</parameter>\n", "")
    _, calls = asyncio.run(_manager(commands=TRAINING).parse_action(bash))
    assert calls[0].function.arguments == {"command": "insert"}


class _Model:
    def __init__(self, output: str):
        self.output = output

    async def query(self, messages, rollout_cache):
        rollout_cache["response_mask"] = rollout_cache.get("response_mask", []) + [1] * 5
        return self.output, None, rollout_cache, {"prompt_tokens": 1, "completion_tokens": 5}

    async def append_messages_to_rollout_cache(self, messages, rollout_cache):
        return rollout_cache


def test_a_disabled_command_never_reaches_the_sandbox_and_counts_as_a_format_error():
    executed: list = []
    it = AgentInteraction.__new__(AgentInteraction)
    it.logger = get_logger("interaction", "test")
    it.messages = [{"role": "user", "content": "fix it"}]
    it.rollout_cache = {"metrics": {}, "response_mask": [], "prompt_ids": []}
    it.condense_max_retries = 0
    it.condenser = None
    it.chat_mode = False
    it.observation_role = "tool"
    it.trajectory = []
    it.model = _Model("Undoing.\n\n" + _call("str_replace_editor", "undo_edit"))
    it.tools_manager = _manager(commands=TRAINING)

    async def _execute(tool_call):
        executed.append(tool_call.function.arguments)
        return ToolResult(tool_call_id=tool_call.id, name=tool_call.function.name, observation="ok", status="ok")

    it._execute_tool_call = _execute
    out = asyncio.run(it.step(1))
    assert out.exit_reason == "format_error"
    assert executed == []
    assert "command 'undo_edit' is not enabled" in it.messages[-1]["content"]
    assert behaviour_metrics([out])["format_errors"] == 1


def _script(path, *argv: str, enabled: list[str] | None) -> str:
    env = {k: v for k, v in os.environ.items() if k != StrReplaceEditorTool.commands_env}
    if enabled is not None:
        env[StrReplaceEditorTool.commands_env] = ",".join(enabled)
    script = StrReplaceEditorTool().local_path
    return subprocess.run(
        [sys.executable, str(script), *argv, "--path", str(path)], capture_output=True, text=True, timeout=30, env=env
    ).stdout


def test_the_script_run_from_bash_refuses_a_disabled_command_in_the_tool_call_wording(tmp_path):
    target = tmp_path / "a.py"
    target.write_text("y = 1\n")
    with pytest.raises(FunctionCallFormatError) as e:
        asyncio.run(_manager(commands=TRAINING).parse_action(_call("str_replace_editor", "insert")))
    for command in ("insert", "undo_edit"):
        out = _script(target, command, "--insert_line", "0", "--new_str", "x = 0", enabled=TRAINING)
        assert out == str(e.value).replace("'insert'", f"'{command}'") + "\n"
    assert target.read_text() == "y = 1\n"
    assert "cat -n" in _script(target, "view", enabled=TRAINING)
    assert "has been edited" in _script(target, "insert", "--insert_line", "0", "--new_str", "x = 0", enabled=None)
    assert target.read_text() == "x = 0\ny = 1\n"


class _Runtime:
    def __init__(self):
        self.sent: list[str] = []

    async def run_in_session(self, action):
        self.sent.append(action.command)
        return types.SimpleNamespace(output="", exit_code=0)

    async def execute(self, command):
        return None

    async def upload(self, request):
        return None


def _installed(**editor) -> list[str]:
    env = AgentEnv.__new__(AgentEnv)  # __init__ needs a real deployment
    env.tool_install_dir = Path("/usr/local/bin")
    noop = lambda *a, **k: None  # noqa: E731
    env.logger = types.SimpleNamespace(info=noop, error=noop, debug=noop)
    env.deployment = types.SimpleNamespace(runtime=_Runtime())
    env.install_tools(_manager(**editor).tools)
    return env.deployment.runtime.sent


def test_the_session_holds_the_enabled_commands_only_when_configured():
    configured, unset = _installed(commands=TRAINING), _installed()
    export = "export STR_REPLACE_COMMANDS=view,create,str_replace"
    assert export in configured
    assert [command for command in configured if command != export] == unset
    assert not any("STR_REPLACE_COMMANDS" in command for command in unset)


def test_the_enabled_commands_cannot_come_from_env_variables():
    with pytest.raises(ValidationError, match=r"\['STR_REPLACE_COMMANDS'\] come from the tools config's `commands`"):
        AgentEnvConfig(deployment=HostDeploymentConfig(), env_variables={"STR_REPLACE_COMMANDS": "view"})
    AgentEnvConfig(deployment=HostDeploymentConfig(), env_variables={"STR_REPLACE_USE_FILEMAP": "true"})
