"""Think tool definition."""

from pathlib import Path

from pydantic import BaseModel, Field

from uni_agent.tools.base import AbstractTool
from uni_agent.tools.registry import register_tool

DESCRIPTION = """
Use the tool to think about something. It will not obtain new information or change the
database, but just append the thought to the log. Use it when complex reasoning or some
cache memory is needed.
""".strip()


class ThinkArguments(BaseModel):
    thought: str = Field(description="A thought to think about.")


@register_tool("think")
class ThinkTool(AbstractTool):
    @property
    def name(self) -> str:
        return "think"

    @property
    def local_path(self) -> Path:
        return Path(__file__).parent / "think"

    def get_tool_schema(self) -> dict:
        return self.build_tool_schema(
            description=DESCRIPTION,
            arguments_model=ThinkArguments,
        )

    def get_install_command(self) -> str | None:
        return None
