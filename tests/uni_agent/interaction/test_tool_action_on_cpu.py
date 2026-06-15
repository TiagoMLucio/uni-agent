"""Unit tests for ToolsManager.get_tool_action (tool_call -> EnvAction)."""

from __future__ import annotations

from uni_agent.interaction.env import EnvAction
from uni_agent.interaction.tool_schemas import OpenAIFunctionCallSchema, OpenAIFunctionToolCall
from uni_agent.interaction.tools_manager import ToolsManager


def _call(name: str, arguments: dict) -> OpenAIFunctionToolCall:
    return OpenAIFunctionToolCall(
        id="call-1",
        type="function",
        function=OpenAIFunctionCallSchema(name=name, arguments=arguments),
    )


def _manager() -> ToolsManager:
    # get_tool_action only uses get_tool_bash_command + _coerce_timeout, neither of
    # which touches instance state, so a bare instance is enough (no env/registry).
    return ToolsManager.__new__(ToolsManager)


def test_execute_bash_plain_command():
    action = _manager().get_tool_action(_call("execute_bash", {"command": "ls -la"}))
    assert action == EnvAction(command="ls -la", is_input=False, timeout=None)


def test_execute_bash_is_input_and_timeout():
    action = _manager().get_tool_action(
        _call("execute_bash", {"command": "print(1)", "is_input": True, "timeout": 120})
    )
    assert action.command == "print(1)"
    assert action.is_input is True
    assert action.timeout == 120


def test_execute_bash_timeout_coerced_from_string():
    action = _manager().get_tool_action(_call("execute_bash", {"command": "x", "timeout": "90"}))
    assert action.timeout == 90


def test_execute_bash_invalid_timeout_is_none():
    action = _manager().get_tool_action(_call("execute_bash", {"command": "x", "timeout": "abc"}))
    assert action.timeout is None


def test_submit_is_never_input():
    action = _manager().get_tool_action(_call("submit", {}))
    assert action == EnvAction(command="echo '<<<Finished>>>'", is_input=False, timeout=None)


def test_other_tool_uses_bash_command_and_no_interactive_flags():
    mgr = _manager()
    call = _call("str_replace_editor", {"command": "view", "path": "/x"})
    action = mgr.get_tool_action(call)
    assert action.command == mgr.get_tool_bash_command(call)
    assert action.is_input is False
    assert action.timeout is None
