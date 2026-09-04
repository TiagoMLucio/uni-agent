"""Unit tests for the tool-call parsers: a call cut before its closer is never executed."""

from __future__ import annotations

import pytest

from uni_agent.interaction.tool_parser import (
    FunctionCallFormatError,
    HermesToolParser,
    SweStarXMLToolParser,
    XMLToolParser,
)
from uni_agent.interaction.tool_schemas import (
    OpenAIFunctionParametersSchema,
    OpenAIFunctionPropertySchema,
    OpenAIFunctionSchema,
    OpenAIFunctionToolSchema,
)

TOOLS = [
    OpenAIFunctionToolSchema(
        type="function",
        function=OpenAIFunctionSchema(
            name="str_replace_editor",
            description="editor",
            parameters=OpenAIFunctionParametersSchema(
                type="object",
                properties={
                    "command": OpenAIFunctionPropertySchema(type="string"),
                    "path": OpenAIFunctionPropertySchema(type="string"),
                    "old_str": OpenAIFunctionPropertySchema(type="string"),
                    "new_str": OpenAIFunctionPropertySchema(type="string"),
                },
                required=["command", "path"],
            ),
        ),
    )
]

COMPLETE = (
    "Let me fix it.\n\n<tool_call>\n<function=str_replace_editor>\n"
    "<parameter=command>\nstr_replace\n</parameter>\n"
    "<parameter=path>\n/testbed/a.py\n</parameter>\n"
    "<parameter=old_str>\nx = 1\n</parameter>\n"
    "<parameter=new_str>\nx = 2\n</parameter>\n"
    "</function>\n</tool_call>"
)


def test_xml_complete_call_parses():
    content, calls = XMLToolParser().extract_tool_calls(COMPLETE, TOOLS)
    assert content == "Let me fix it.\n\n"
    assert len(calls) == 1
    assert calls[0].function.name == "str_replace_editor"
    assert calls[0].function.arguments == {
        "command": "str_replace",
        "path": "/testbed/a.py",
        "old_str": "x = 1",
        "new_str": "x = 2",
    }


def test_xml_no_marker_returns_no_calls():
    content, calls = XMLToolParser().extract_tool_calls("just prose", TOOLS)
    assert content == "just prose"
    assert calls == []


def test_xml_call_cut_inside_a_value_raises():
    cut = COMPLETE[: COMPLETE.index("x = 2") + 2]
    with pytest.raises(FunctionCallFormatError, match="Unclosed tool call"):
        XMLToolParser().extract_tool_calls(cut, TOOLS)


def test_xml_call_cut_inside_old_str_raises():
    # without the closer this used to run str_replace with new_str missing, i.e. a deletion
    cut = COMPLETE[: COMPLETE.index("x = 1") + 3]
    with pytest.raises(FunctionCallFormatError, match="Unclosed tool call"):
        XMLToolParser().extract_tool_calls(cut, TOOLS)


def test_xml_call_cut_in_closing_tags_raises():
    cut = COMPLETE[: COMPLETE.index("</function")]
    with pytest.raises(FunctionCallFormatError, match="Unclosed tool call"):
        XMLToolParser().extract_tool_calls(cut, TOOLS)


def test_xml_closed_then_cut_executes_nothing():
    two = COMPLETE + "\n\n" + COMPLETE[: COMPLETE.index("x = 2")]
    with pytest.raises(FunctionCallFormatError, match="Unclosed tool call"):
        XMLToolParser().extract_tool_calls(two, TOOLS)


def test_xml_two_closed_calls_both_parse():
    _, calls = XMLToolParser().extract_tool_calls(COMPLETE + "\n" + COMPLETE, TOOLS)
    assert [c.function.name for c in calls] == ["str_replace_editor", "str_replace_editor"]


def test_swestar_unclosed_function_raises():
    bare = COMPLETE[COMPLETE.index("<function=") : COMPLETE.index("</function>")]
    with pytest.raises(FunctionCallFormatError, match="Unclosed tool call"):
        SweStarXMLToolParser().extract_tool_calls(bare, TOOLS)


def test_swestar_closed_function_parses():
    bare = COMPLETE[COMPLETE.index("<function=") : COMPLETE.index("</tool_call>")]
    _, calls = SweStarXMLToolParser().extract_tool_calls(bare, TOOLS)
    assert calls[0].function.arguments["new_str"] == "x = 2"


def test_hermes_unclosed_raises():
    cut = '<tool_call>\n{"name": "str_replace_editor", "arguments": {"command": "vi'
    with pytest.raises(FunctionCallFormatError, match="Unclosed tool call"):
        HermesToolParser().extract_tool_calls(cut, TOOLS)


def test_closed_block_with_unclosed_function_raises():
    from uni_agent.interaction.tool_parser import FunctionCallFormatError, XMLToolParser

    text = (
        "<tool_call>\n<function=str_replace_editor>\n<parameter=command>str_replace</parameter>\n"
        "<parameter=path>/testbed/a.py</parameter>\n<parameter=old_str>def f():\n    return 1\n</tool_call>"
    )
    with pytest.raises(FunctionCallFormatError, match="missing </function>"):
        XMLToolParser()._get_function_calls(text)
