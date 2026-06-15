"""Execute bash command tool."""

from pathlib import Path

from pydantic import BaseModel, Field

from uni_agent.tools.base import AbstractTool
from uni_agent.tools.registry import register_tool

DESCRIPTION = """
Execute a bash command in the terminal.

For interactive programs (e.g. python, gdb, a prompt waiting for input), set
is_input=true and put the text to send (or "C-c" to interrupt) in `command`.
""".strip()


class ExecuteBashArguments(BaseModel):
    command: str = Field(
        description=(
            "The bash command to execute. When is_input=true, this is instead the input/keystrokes "
            'to send to the currently running interactive program (use "C-c" to interrupt it).'
        )
    )
    is_input: bool = Field(
        default=False,
        description=(
            "If true, `command` is sent as input to the already-running interactive program instead "
            'of starting a new command. Use "C-c" to interrupt a running process.'
        ),
    )
    timeout: int | None = Field(
        default=None,
        description="Optional timeout in seconds for this command (defaults to the harness action timeout).",
    )


@register_tool("execute_bash")
class ExecuteBashTool(AbstractTool):
    @property
    def name(self) -> str:
        return "execute_bash"

    @property
    def local_path(self) -> Path:
        return Path(__file__).parent / "execute_bash"

    def get_tool_schema(self) -> dict:
        return self.build_tool_schema(
            description=DESCRIPTION,
            arguments_model=ExecuteBashArguments,
        )

    def get_install_command(self) -> str:
        return None
