# ruff: noqa: E501
"""Str-replace editor tool definition."""

import re
from pathlib import Path

from pydantic import BaseModel, Field

from uni_agent.tools.base import AbstractTool
from uni_agent.tools.registry import register_tool

DESCRIPTION = """
Custom editing tool for viewing, creating and editing files
* State is persistent across command calls and discussions with the user
* If `path` is a file, `view` displays the result of applying `cat -n`. If `path` is a directory, `view` lists non-hidden files and directories up to 2 levels deep
* The `create` command cannot be used if the specified `path` already exists as a file
* If a `command` generates a long output, it will be truncated and marked with `<response clipped>`
* The `undo_edit` command will revert the last edit made to the file at `path`

Notes for using the `str_replace` command:
* The `old_str` parameter should match EXACTLY one or more consecutive lines from the original file. Be mindful of whitespaces!
* If the `old_str` parameter is not unique in the file, the replacement will not be performed. Make sure to include enough context in `old_str` to make it unique
* The `new_str` parameter should contain the edited lines that should replace the `old_str`
* If `old_str` is not found, the error names the closest matching lines and exactly how your string differs from the file. Correct your next attempt with that description; do not resend the same string
""".strip()


class StrReplaceEditorArguments(BaseModel):
    command: str = Field(
        description="The commands to run. Allowed options are: `view`, `create`, `str_replace`, `insert`, `undo_edit`.",
        json_schema_extra={"enum": ["view", "create", "str_replace", "insert", "undo_edit"]},
    )
    path: str = Field(description="Absolute path to file or directory, e.g. `/testbed/file.py` or `/testbed`.")
    file_text: str = Field(
        default=None, description="Required parameter of `create` command, with the content of the file to be created."
    )
    old_str: str = Field(
        default=None,
        description="Required parameter of `str_replace` command containing the string in `path` to replace.",
    )
    new_str: str = Field(
        default=None,
        description="Optional parameter of `str_replace` command containing the new string (if not given, no string will be added). Required parameter of `insert` command containing the string to insert.",
    )
    insert_line: int = Field(
        default=None,
        description="Required parameter of `insert` command. The `new_str` will be inserted AFTER the line `insert_line` of `path`.",
    )
    view_range: list[int] = Field(
        default=None,
        description="Optional parameter of `view` command when `path` points to a file. If none is given, the full file is shown. If provided, the file will be shown in the indicated line number range, e.g. [11, 12] will show lines 11 and 12. Indexing at 1 to start. Setting `[start_line, -1]` shows all lines from `start_line` to the end of the file.",
    )


#: the arguments each command reads besides `command` and `path`
COMMAND_ARGUMENTS = {
    "view": ("view_range",),
    "create": ("file_text",),
    "str_replace": ("old_str", "new_str"),
    "insert": ("insert_line", "new_str"),
    "undo_edit": (),
}


def _without_mentions(parts: list[str], names: list[str]) -> list[str]:
    return [part for part in parts if not any(f"`{name}`" in part for name in names)]


@register_tool("str_replace_editor")
class StrReplaceEditorTool(AbstractTool):
    commands = tuple(COMMAND_ARGUMENTS)
    commands_env = "STR_REPLACE_COMMANDS"

    @property
    def name(self) -> str:
        return "str_replace_editor"

    @property
    def local_path(self) -> Path:
        return Path(__file__).parent / "str_replace_editor"

    def get_tool_schema(self) -> dict:
        return self.build_tool_schema(
            description=DESCRIPTION,
            arguments_model=StrReplaceEditorArguments,
        )

    def narrow_schema(self, schema: dict, commands: list[str]) -> dict:
        """The arguments no enabled command reads are dropped, and so is every description line,
        or argument sentence, naming a command or argument gone."""
        function = schema["function"]
        properties = function["parameters"]["properties"]
        kept = {"command", "path"}.union(*(COMMAND_ARGUMENTS[command] for command in commands))
        gone = [name for name in [*COMMAND_ARGUMENTS, *properties] if name not in commands and name not in kept]
        function["description"] = "\n".join(_without_mentions(function["description"].split("\n"), gone)).strip()
        for name in [name for name in properties if name not in kept]:
            del properties[name]
        for prop in properties.values():
            prop["description"] = " ".join(_without_mentions(re.split(r"(?<=\.) (?=[A-Z])", prop["description"]), gone))
        properties["command"]["enum"] = list(commands)
        properties["command"]["description"] = (
            f"The commands to run. Allowed options are: {', '.join(f'`{command}`' for command in commands)}."
        )
        return schema

    def get_install_command(self) -> str:
        return "python -m pip install 'tree-sitter==0.21.3' || true && python -m pip install 'tree-sitter-languages' || true"
