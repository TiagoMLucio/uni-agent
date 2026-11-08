"""Outgoing TCP for a sandbox's processes, restricted by Landlock to the kernel's ephemeral port range.

A sandbox that shares the host's network can still reach the internet, so a policy can fetch a task's upstream
source and read the fix from it. Run in front of the sandbox server, this keeps ``connect()`` to the ephemeral
range (where a test suite's own local servers listen) and refuses every other port: the web, git and ssh, and
services on fixed ports such as an image registry. The kernel enforces it, no privilege is needed, every
descendant inherits it, and none can lift it. Listening is untouched.

Inside a sandbox it runs as ``"$JPY" -c <this source> -- <command...>`` (see ``jail_prefix``), under whatever
interpreter the image has, so this file keeps to Python 3.5 syntax.
"""
import base64
import ctypes
import os
import struct
import sys

_CREATE_RULESET, _ADD_RULE, _RESTRICT_SELF = 444, 445, 446  # the same on x86_64 and aarch64
_ACCESS_NET_CONNECT_TCP = 1 << 1
_RULE_NET_PORT = 2
_PR_SET_NO_NEW_PRIVS = 38
#: the first Landlock ABI with network rules
_NET_ABI = 4
_PORT_RANGE = "/proc/sys/net/ipv4/ip_local_port_range"


def ephemeral_range():
    with open(_PORT_RANGE) as f:
        low, high = f.read().split()
    return int(low), int(high)


def _fail(message):
    sys.stderr.write("netjail: {}\n".format(message))
    sys.exit(1)


def restrict_connect(ports):
    """Restrict this process and its future descendants to ``connect()`` only to ``ports``."""
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    abi = libc.syscall(_CREATE_RULESET, None, 0, 1)
    if abi < _NET_ABI:
        _fail("this kernel's Landlock (ABI {}, errno {}) cannot restrict the network".format(abi, ctypes.get_errno()))
    attr = ctypes.create_string_buffer(struct.pack("QQ", 0, _ACCESS_NET_CONNECT_TCP))
    fd = libc.syscall(_CREATE_RULESET, attr, ctypes.sizeof(attr), 0)
    if fd < 0:
        _fail("landlock_create_ruleset failed, errno {}".format(ctypes.get_errno()))
    for port in ports:
        rule = ctypes.create_string_buffer(struct.pack("QQ", _ACCESS_NET_CONNECT_TCP, port))
        if libc.syscall(_ADD_RULE, fd, _RULE_NET_PORT, rule, 0) != 0:
            _fail("landlock_add_rule for port {} failed, errno {}".format(port, ctypes.get_errno()))
    if libc.prctl(_PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0 or libc.syscall(_RESTRICT_SELF, fd, 0) != 0:
        _fail("landlock_restrict_self failed, errno {}".format(ctypes.get_errno()))
    os.close(fd)


def jail_prefix():
    """Shell text that runs the command after it under the jail, with the interpreter in ``$JPY``."""
    with open(os.path.abspath(__file__), "rb") as f:
        source = base64.b64encode(f.read()).decode("ascii")
    return "\"$JPY\" -c \"import base64;exec(base64.b64decode('{}'))\" -- ".format(source)


def main():
    command = sys.argv[1:]
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        _fail("no command to run")
    low, high = ephemeral_range()
    restrict_connect(range(low, high + 1))
    os.execvp(command[0], command)


if __name__ == "__main__":
    main()
