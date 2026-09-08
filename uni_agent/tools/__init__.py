# ruff: noqa
"""
Scaffold tools.
"""

from pydantic import BaseModel, ConfigDict

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

    def get_tool(self) -> AbstractTool:
        """Return a tool instance (for env.install_tools / init_for_interaction)."""
        return get_tool(self.name)


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
