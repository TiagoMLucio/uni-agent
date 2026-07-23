"""The HTTP request timeout must outlast the in-session command timeout it carries."""

import aiohttp
import pytest
from aiohttp import web

from swerex.runtime.abstract import BashAction

from uni_agent.deployment.remote_runtime import RemoteRuntime


@pytest.mark.asyncio
async def test_request_timeout_covers_command_timeout():
    seen = {}

    async def handler(request):
        seen["timeout_header"] = True
        body = await request.json()
        seen["command_timeout"] = body.get("timeout")
        return web.json_response(
            {"output": "ok", "exit_code": 0, "failure_reason": "", "expect_string": ""}
        )

    app = web.Application()
    app.router.add_post("/run_in_session", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]

    captured = {}
    orig_post = aiohttp.ClientSession.post

    def spy_post(self, url, **kwargs):
        captured["total"] = kwargs.get("timeout").total if kwargs.get("timeout") else None
        return orig_post(self, url, **kwargs)

    try:
        aiohttp.ClientSession.post = spy_post
        rt = RemoteRuntime(run_id="t", auth_token="x", host="http://127.0.0.1", port=port, timeout=60)
        await rt.run_in_session(BashAction(command="true", timeout=300))
    finally:
        aiohttp.ClientSession.post = orig_post
        await runner.cleanup()

    assert seen["command_timeout"] == 300
    assert captured["total"] >= 330, f"HTTP timeout {captured['total']} must outlast the 300s command"
