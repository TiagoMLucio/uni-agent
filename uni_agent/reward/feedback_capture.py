"""Install/read the standalone test observer in an evaluation environment."""

import json
import shlex
import uuid
from pathlib import Path

from uni_agent.reward.diagnostic_feedback import load_capture

SHIM = "try:\n    from {impl} import *\nexcept Exception:\n    pass\n"


async def install_capture(env, context: dict, value_chars: int = 4096):
    suffix = uuid.uuid4().hex
    module = f"uniagent_feedback_{suffix}"
    report = Path(f"/tmp/{module}.jsonl")
    metadata = Path(f"/tmp/{module}.context.json")
    plugin = Path(__file__).with_name("pytest_feedback_capture.py").read_text()
    await env.write_file(Path(f"/tmp/{module}_impl.py"), plugin)
    # pytest loads this shim: a plugin the task's interpreter cannot import must leave the tests running
    await env.write_file(Path(f"/tmp/{module}.py"), SHIM.format(impl=f"{module}_impl"))
    await env.write_file(metadata, json.dumps(context))
    lines = [
        'export PYTHONPATH="/tmp:${PYTHONPATH:-}"',
        f"export UNI_AGENT_FEEDBACK_PATH={shlex.quote(str(report))}",
        f"export UNI_AGENT_FEEDBACK_CONTEXT={shlex.quote(str(metadata))}",
        "export UNI_AGENT_FEEDBACK_ROOT=/testbed",
        f"export UNI_AGENT_FEEDBACK_VALUE_CHARS={value_chars}",
        f'export PYTEST_ADDOPTS="${{PYTEST_ADDOPTS:-}} -p {module}"',
    ]
    return report, lines


async def read_capture(env, path):
    if path is None:
        return None
    try:
        return load_capture(await env.read_file(path, errors="backslashreplace"))
    except Exception as exc:
        return {
            "events": [],
            "sources": {},
            "complete": False,
            "errors": [f"Structured test report unavailable: {type(exc).__name__}: {exc}"],
        }
