"""A ``tools[].description`` in the agent config is what the model reads; unset, the module's applies."""

from uni_agent.interaction.tools_manager import ToolsManager, ToolsManagerConfig
from uni_agent.tools import ToolConfig
from uni_agent.tools.execute_bash import DESCRIPTION as BASH_DESCRIPTION
from uni_agent.tools.submit import DESCRIPTION as SUBMIT_DESCRIPTION


def _schemas(*tools: ToolConfig) -> dict[str, dict]:
    manager = ToolsManager(ToolsManagerConfig(tools=list(tools)))
    return {schema["function"]["name"]: schema["function"] for schema in manager.tools_schemas}


def test_a_configured_description_replaces_the_module_default():
    custom = "Run a shell command; prefer grep -rl over find -exec."
    schemas = _schemas(ToolConfig(name="execute_bash", description=custom), ToolConfig(name="submit"))
    assert schemas["execute_bash"]["description"] == custom
    assert schemas["submit"]["description"] == SUBMIT_DESCRIPTION
    # the argument schema is the module's either way
    assert set(schemas["execute_bash"]["parameters"]["properties"]) == {"command", "is_input", "timeout"}


def test_no_description_keeps_the_module_default():
    schemas = _schemas(ToolConfig(name="execute_bash"))
    assert schemas["execute_bash"]["description"] == BASH_DESCRIPTION
