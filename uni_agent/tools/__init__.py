# ruff: noqa
"""
Scaffold tools.
"""

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .finish import FinishTool
from .registry import get_tool, AbstractTool
from .execute_bash import ExecuteBashTool
from .lark_cli import LarkCliTool
from .search_arxiv import SearchArxivTool
from .search import SearchWikiTool
from .str_replace_editor import StrReplaceEditorTool
from .submit import SubmitTool
from .think import ThinkTool


class ToolConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    #: replaces the tool module's DESCRIPTION in the schema the model reads; None keeps it
    description: str | None = None
    #: the only values of the tool's `command` argument its schema offers and a call may use; None keeps all
    commands: list[str] | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def _known_commands(self) -> "ToolConfig":
        if self.commands is not None:
            known = get_tool(self.name).commands
            if unknown := [command for command in self.commands if command not in known]:
                raise ValueError(f"tool {self.name!r} has no commands {unknown}; it has {list(known)}")
            if len(set(self.commands)) != len(self.commands):
                raise ValueError(f"tool {self.name!r} lists a command twice: {self.commands}")
        return self

    def get_tool(self) -> AbstractTool:
        """Return a tool instance (for env.install_tools / init_for_interaction)."""
        tool = get_tool(self.name)
        tool.enabled_commands = self.commands
        return tool


__all__ = [
    "ToolConfig",
    "ExecuteBashTool",
    "FinishTool",
    "LarkCliTool",
    "SearchArxivTool",
    "SearchWikiTool",
    "StrReplaceEditorTool",
    "SubmitTool",
    "ThinkTool",
]
