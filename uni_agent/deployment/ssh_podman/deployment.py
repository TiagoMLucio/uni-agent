import asyncio
import os
import re
import socket
import subprocess
import uuid
from pathlib import Path
from typing import Any, Self

import docker
from docker.errors import NotFound
from swerex.deployment.abstract import AbstractDeployment
from swerex.deployment.hooks.abstract import CombinedDeploymentHook, DeploymentHook
from swerex.exceptions import DeploymentNotStartedError
from swerex.runtime.abstract import Command, CreateBashSessionRequest, IsAliveResponse, UploadRequest

from uni_agent.async_logging import get_logger
from uni_agent.deployment.config import SshPodmanDeploymentConfig
from uni_agent.deployment.remote_runtime import RemoteRuntime as LocalRuntime
from uni_agent.deployment.remote_runtime import RemoteRuntimeConfig as LocalRuntimeConfig


def _sanitize(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", value).strip("-").lower() or "uni-agent"


def _free_local_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class _SshMaster:
    """One persistent OpenSSH ControlMaster per worker process, shared by every sandbox.

    It multiplexes, over a single SSH connection (kernel-level forwarding, no Python
    threads/GIL and no per-sandbox connection):
      * a forward of the remote podman API socket -> a shared docker-py client (control plane);
      * one dynamic ``ssh -O forward`` per sandbox -> that container's swerex port (data plane).

    The single shared connection sidesteps sshd MaxSessions/MaxStartups (``-O forward``
    channels are ``direct-tcpip``); validated to 64 concurrent sandboxes on one event loop.
    """

    _instance: "_SshMaster | None" = None
    _lock = asyncio.Lock()

    def __init__(self, config: SshPodmanDeploymentConfig):
        self._config = config
        self._ctl = f"/tmp/uniagent-ssh-{os.getpid()}.ctl"
        self.docker: docker.DockerClient | None = None
        self._started = False
        # serialize control-socket ops: concurrent `ssh -O forward` clients race on the master
        self._op_lock = asyncio.Lock()

    @classmethod
    async def get(cls, config: SshPodmanDeploymentConfig) -> "_SshMaster":
        if cls._instance is None:
            async with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(config)
        await cls._instance._ensure()
        return cls._instance

    def _ssh_opts(self) -> list[str]:
        opts = [
            "-p",
            str(self._config.ssh_port or 22),
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            "ServerAliveInterval=30",
        ]
        if self._config.ssh_key:
            opts += ["-i", self._config.ssh_key]
        return opts

    def _start_blocking(self) -> None:
        podman_local = _free_local_port()
        cmd = (
            ["ssh", "-M", "-S", self._ctl, "-fN", "-o", "ControlPersist=yes", "-o", "ExitOnForwardFailure=yes"]
            + self._ssh_opts()
            + ["-L", f"127.0.0.1:{podman_local}:127.0.0.1:{self._config.podman_port}", self._config.ssh_host]
        )
        subprocess.run(cmd, check=True, timeout=self._config.startup_timeout, capture_output=True)
        self.docker = docker.DockerClient(
            base_url=f"tcp://127.0.0.1:{podman_local}", timeout=int(self._config.startup_timeout)
        )
        self._started = True

    async def _ensure(self) -> None:
        if self._started:
            return
        async with _SshMaster._lock:
            if not self._started:
                await asyncio.to_thread(self._start_blocking)

    async def forward(self, remote_port: int) -> int:
        """Add a dynamic local->container forward over the master; return the local port."""
        async with self._op_lock:
            last: Exception | None = None
            for _ in range(3):  # retry covers the rare free-port reuse between pick and bind
                local = _free_local_port()
                cmd = (
                    ["ssh", "-S", self._ctl, "-O", "forward"]
                    + self._ssh_opts()
                    + ["-L", f"127.0.0.1:{local}:127.0.0.1:{remote_port}", self._config.ssh_host]
                )
                try:
                    await asyncio.to_thread(subprocess.run, cmd, check=True, timeout=30, capture_output=True)
                    return local
                except subprocess.CalledProcessError as exc:
                    last = exc
            raise RuntimeError(f"ssh -O forward failed for remote port {remote_port}") from last

    async def cancel(self, local_port: int, remote_port: int) -> None:
        cmd = (
            ["ssh", "-S", self._ctl, "-O", "cancel"]
            + self._ssh_opts()
            + ["-L", f"127.0.0.1:{local_port}:127.0.0.1:{remote_port}", self._config.ssh_host]
        )
        try:
            async with self._op_lock:
                await asyncio.to_thread(
                    subprocess.run, cmd, timeout=15, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
        except Exception:
            pass


class SshPodmanDeployment(AbstractDeployment):
    """Per-task rootless-podman sandbox on a remote host, over a shared SSH ControlMaster.

    Control plane: the remote podman API, reached via docker-py over the master's single
    forwarded socket. Data plane: the container's swerex port is exposed by a dynamic
    ``ssh -O forward`` on the same master, so swerex is reached directly (low latency, no
    GIL-bound in-process forwarder). The master + docker client are shared per worker; this
    deployment only owns its container + its one forward.
    """

    def __init__(self, run_id: str, **kwargs: Any):
        self.run_id = run_id
        self._config = SshPodmanDeploymentConfig(**kwargs)
        self._runtime: LocalRuntime | None = None
        self.logger = get_logger("deployment", run_id)
        self._hooks = CombinedDeploymentHook()
        self._master: _SshMaster | None = None
        self._container_name: str | None = None
        self._local_port: int | None = None
        self._remote_port: int | None = None
        self._stopped = False

    def add_hook(self, hook: DeploymentHook):
        self._hooks.add_hook(hook)

    @classmethod
    def from_config(cls, config: SshPodmanDeploymentConfig, run_id: str | None = None) -> Self:
        return cls(run_id=run_id or str(uuid.uuid4()), **config.model_dump())

    def _run_container(self, name: str, command: str) -> int:
        """Create + start the sandbox via the shared podman API; return its published port."""
        d = self._master.docker
        assert d is not None
        try:  # clear a stale container left by a previously-crashed run
            d.containers.get(name).remove(force=True)
        except NotFound:
            pass
        container = d.containers.run(
            self._config.image,
            # `timeout` makes pid 1 self-exit after max_lifetime so `auto_remove` reaps the
            # container even if the owning process is SIGKILLed and never calls stop().
            entrypoint=["timeout", "-k", "30", str(self._config.max_lifetime), self._config.shell],
            command=["-lc", command],
            name=name,
            detach=True,
            auto_remove=True,
            ports={f"{self._config.runtime_port}/tcp": ("127.0.0.1", None)},
        )
        container.reload()
        return int(container.ports[f"{self._config.runtime_port}/tcp"][0]["HostPort"])

    def _logs(self, name: str) -> str:
        try:
            return self._master.docker.containers.get(name).logs(tail=50).decode("utf-8", "replace")
        except Exception as exc:
            return f"<failed to fetch logs: {exc}>"

    async def is_alive(self, *, timeout: float | None = None) -> IsAliveResponse:
        if self._runtime is None:
            raise DeploymentNotStartedError("Runtime not started")
        return await self._runtime.is_alive(timeout=timeout)

    async def _wait_until_alive(self, timeout: float) -> IsAliveResponse:
        # swerex's _wait_until_alive polls with a blocking time.sleep, freezing the shared
        # event loop under concurrent startups; re-implemented with `await asyncio.sleep`
        loop = asyncio.get_event_loop()
        end = loop.time() + timeout
        last: IsAliveResponse | None = None
        while loop.time() < end:
            last = await self.is_alive(timeout=5.0)
            if last:
                return last
            await asyncio.sleep(0.5)
        # no stop() here: the caller's retry handler fetches container logs before tearing down
        self.logger.error("Remote podman runtime did not start within timeout.")
        raise TimeoutError(
            f"Remote podman runtime did not start within {timeout}s; last={getattr(last, 'message', last)}"
        )

    async def start(self, max_retries: int = 5) -> None:
        token = str(uuid.uuid4())
        name = f"uni-agent-{_sanitize(self.run_id)}"
        command = self._config.command.format(token=token, port=self._config.runtime_port)
        last_error: Exception | None = None
        for attempt in range(max_retries):
            self._stopped = False
            self.logger.info(f"Starting podman sandbox on {self._config.ssh_host}, image={self._config.image}.")
            self._hooks.on_custom_step("Creating remote podman sandbox")
            try:
                self._master = await _SshMaster.get(self._config)
                self._container_name = name
                self._remote_port = await asyncio.to_thread(self._run_container, name, command)
                self._local_port = await self._master.forward(self._remote_port)
                self._runtime = LocalRuntime.from_config(
                    LocalRuntimeConfig(
                        auth_token=token,
                        host="http://127.0.0.1",
                        port=self._local_port,
                        timeout=self._config.timeout,
                    ),
                    run_id=self.run_id,
                )
                await self._wait_until_alive(timeout=self._config.startup_timeout)
                await self.runtime.create_session(
                    CreateBashSessionRequest(startup_source=["/root/.bashrc"], startup_timeout=60)
                )
                return
            except Exception as exc:
                last_error = exc
                logs = self._logs(name) if (self._master and self._master.docker) else "<no client>"
                self.logger.error(f"Failed to start podman sandbox: {exc}\nContainer logs:\n{logs}")
                await self.stop()
                if attempt < max_retries - 1:
                    await asyncio.sleep(min(30, 2**attempt))
        raise RuntimeError(f"Failed to create podman sandbox after {max_retries} retries") from last_error

    async def stop(self):
        if self._stopped:
            return
        if self._runtime is not None:
            try:
                await self._runtime.close()
            except Exception as exc:
                self.logger.error(f"Failed to close remote runtime: {exc}")
            self._runtime = None
        # tear down only THIS sandbox's forward + container; the master/docker are shared.
        if self._local_port is not None and self._remote_port is not None and self._master is not None:
            await self._master.cancel(self._local_port, self._remote_port)
        self._local_port = None
        self._remote_port = None
        if self._container_name is not None and self._master is not None and self._master.docker is not None:
            # rootless podman netns teardown fails transiently under concurrent ops; container TTL is the backstop
            for attempt, delay in enumerate((0, 3, 9), start=1):
                if delay:
                    await asyncio.sleep(delay)
                try:
                    await asyncio.to_thread(self._master.docker.containers.get(self._container_name).remove, force=True)
                    break
                except Exception as exc:
                    log = self.logger.error if attempt == 3 else self.logger.warning
                    log(f"Failed to remove remote sandbox {self._container_name} (attempt {attempt}/3): {exc}")
            self._container_name = None
        self._stopped = True

    @property
    def runtime(self) -> LocalRuntime:
        if self._runtime is None:
            raise DeploymentNotStartedError()
        return self._runtime

    @property
    def tool_install_dir(self) -> Path:
        return Path("/usr/local/bin")

    async def copy_to_container(self, src: Path, tgt: Path):
        await self.runtime.execute(Command(command=["mkdir", "-p", str(tgt.parent)]))
        await self.runtime.upload(UploadRequest(source_path=str(src), target_path=str(tgt)))

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()

    def __del__(self):
        if getattr(self, "_container_name", None) and not getattr(self, "_stopped", False):
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self.stop())
                else:
                    loop.run_until_complete(self.stop())
            except Exception:
                pass
        self._stopped = True
