"""Execute bash command tool."""

from pathlib import Path

from pydantic import BaseModel, Field

from uni_agent.tools.base import AbstractTool
from uni_agent.tools.registry import register_tool

DESCRIPTION = """
Execute a bash command in the terminal.

A command that exceeds its timeout is NOT stopped: it stays attached to the session,
and no new command can run until it finishes or is cancelled. When that happens, call
this tool again with is_input=true and `command` set to one of:
  - the text to send to the program
  - "C-c" to cancel it
  - "C-d" to send EOF (a REPL ignores C-c but exits on EOF)
  - "" (empty) to wait and collect more output
So prefer forms that cannot block on a prompt: `patch --batch`, always give `grep` a
path to search, and avoid REPLs (`python` with no script, `scrapy shell`) unless you
intend to drive them with is_input.

For searching, `grep -rl 'pattern' /testbed` is 8-30x faster than
`find ... -exec grep {} \\;`, which spawns one process per file. Narrowing it to the
repository's own language helps further, e.g. `--include='*.py'` in a Python repo.

Redirect long-running work instead of waiting on it:
`<command> > /tmp/out.log 2>&1 &` then `tail -n 50 /tmp/out.log`.
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
