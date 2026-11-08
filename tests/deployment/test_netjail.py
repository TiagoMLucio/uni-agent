"""The network jail on this machine's kernel: connect() reaches the ephemeral range and nothing else, run directly
and through the shell text a sandbox's start command carries. Skipped where Landlock has no network rules."""

import ctypes
import os
import subprocess
import sys

import pytest

from uni_agent.deployment import netjail

ABI = ctypes.CDLL(None, use_errno=True).syscall(444, None, 0, 1) if sys.platform == "linux" else -1
pytestmark = pytest.mark.skipif(ABI < 4, reason="no Landlock network rules on this kernel")

PROBE = """
import socket, threading, http.server
for port in (443, 80, 5000):
    try:
        socket.create_connection(("127.0.0.1", port), timeout=2)
        print("reached", port)
    except PermissionError:
        print("refused", port)
    except OSError:
        print("reached", port)
srv = http.server.HTTPServer(("127.0.0.1", 0), http.server.BaseHTTPRequestHandler)
threading.Thread(target=srv.serve_forever, daemon=True).start()
socket.create_connection(("127.0.0.1", srv.server_address[1]), timeout=2)
print("local server", srv.server_address[1])
"""


def _expect_jailed(out: str) -> None:
    lines = out.split("\n")
    assert {"refused 443", "refused 80", "refused 5000"} <= set(lines)
    low, high = netjail.ephemeral_range()
    port = int(next(line for line in lines if line.startswith("local server")).split()[-1])
    assert low <= port <= high


def test_only_the_ephemeral_range_is_reachable():
    out = subprocess.run([sys.executable, netjail.__file__, "--", sys.executable, "-c", PROBE],
                         capture_output=True, text=True, timeout=60, check=True).stdout
    _expect_jailed(out)


def test_the_start_command_prefix_runs_the_server_command_jailed():
    command = f'JPY="{sys.executable}"; exec {netjail.jail_prefix()}"{sys.executable}" -c "$PROBE"'
    out = subprocess.run(["/bin/sh", "-c", command], env={**os.environ, "PROBE": PROBE},
                         capture_output=True, text=True, timeout=60, check=True).stdout
    _expect_jailed(out)
