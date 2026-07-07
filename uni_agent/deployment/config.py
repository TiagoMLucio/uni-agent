from pathlib import PurePath
from typing import Annotated, Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field


class HostDeploymentConfig(BaseModel):
    """Configuration for host-local execution (no container)."""

    type: Literal["host"] = "host"
    """Discriminator for (de)serialization. Do not change."""
    timeout: float = 60.0
    """Default timeout for runtime operations."""
    startup_timeout: float = 120.0
    """Timeout for the initial bash session handshake.

    During parameter-sync weight reloads, fork()/exec() and even the asyncio event loop can be
    starved for tens of seconds.
    """

    model_config = ConfigDict(extra="forbid")

    def get_deployment(self, run_id: str):
        from .host.deployment import HostDeployment

        return HostDeployment.from_config(self, run_id)


class LocalNativeDeploymentConfig(BaseModel):
    """Configuration for in-process pexpect-based host execution.

    Like ``HostDeploymentConfig`` this runs commands directly on the host (no
    container), but drives bash via ``pexpect`` / PTY rather than
    ``asyncio.create_subprocess_exec``. Compatible with the framework's
    sync-style ``auto_await`` API. See
    ``uni_agent/deployment/local_native/runtime.py`` for details.
    """

    type: Literal["local_native"] = "local_native"
    """Discriminator for (de)serialization. Do not change."""
    timeout: float = 60.0
    """Default timeout for runtime operations."""
    startup_timeout: float = 120.0
    """Timeout for the initial bash session handshake."""

    model_config = ConfigDict(extra="forbid")

    def get_deployment(self, run_id: str):
        from .local_native.deployment import LocalNativeDeployment

        return LocalNativeDeployment.from_config(self, run_id)


class LocalDeploymentConfig(BaseModel):
    """Configuration for a local sandbox."""

    image: str = "python:3.12"
    """Container image used for the sandbox."""
    command: str = (
        "python3 -m pip install -q swe-rex && "
        "python3 -m swerex.server --host 0.0.0.0 --port {port} --auth-token {token}"
    )
    """Command to run inside the sandbox."""
    timeout: float = 60.0
    """Timeout for runtime operations."""
    startup_timeout: float = 180.0
    """Timeout waiting for runtime to start."""
    container_runtime: str = "apptainer"
    """Container runtime executable. If omitted by the user, local deployment discovers one at startup."""
    container_name: str | None = None
    """Optional container name override."""
    host: str | None = None
    """Override the runtime host. Defaults to localhost outside containers and container IP inside containers."""
    published_port: int | None = None
    """Host port mapped to the sandbox runtime port. If unset, a free local port is chosen."""
    runtime_port: int = 8000
    """Port exposed by the swerex server inside the sandbox."""
    network: str | None = None
    """Optional Docker network to attach the sandbox to."""
    shell: str = "/bin/bash"
    """Shell executable used as the container entrypoint."""
    extra_run_args: list[str] = Field(default_factory=list)
    """Extra args appended to the container runtime startup command."""

    type: Literal["local"] = "local"
    """Discriminator for (de)serialization/CLI. Do not change."""
    model_config = ConfigDict(extra="forbid")

    def get_deployment(self, run_id: str):
        from .local.deployment import LocalDeployment

        return LocalDeployment.from_config(self, run_id)


class LocalAttachDeploymentConfig(BaseModel):
    """Configuration for attaching to a user-managed swerex server.

    Unlike ``LocalDeploymentConfig`` (which ``docker run``s a fresh sandbox),
    this deployment **does not** start, stop, or otherwise manage a container.
    The user is responsible for launching a container ahead of time, running
    ``swerex.server`` inside it, and exposing it on a reachable host/port.
    ``start()`` attaches over HTTP; ``stop()`` is a no-op.
    """

    host: str = "http://127.0.0.1"
    """Host of the user-managed swerex server (e.g. ``http://127.0.0.1``)."""
    port: int = 8000
    """Port the swerex server is listening on (the host-side published port)."""
    auth_token: str
    """Auth token passed to ``swerex.server --auth-token`` by the user."""
    timeout: float = 60.0
    """Timeout for runtime operations."""
    startup_timeout: float = 30.0
    """Timeout for the initial ``is_alive`` probe inside ``start()``."""
    proxy: str | None = None
    """Optional proxy for the runtime HTTP client."""

    type: Literal["local_attach"] = "local_attach"
    """Discriminator for (de)serialization/CLI. Do not change."""
    model_config = ConfigDict(extra="forbid")

    def get_deployment(self, run_id: str):
        from .local_attach.deployment import LocalAttachDeployment

        return LocalAttachDeployment.from_config(self, run_id)


class ModalDeploymentConfig(BaseModel):
    """Configuration for Modal deployment."""

    image: str | PurePath = "python:3.11"
    """Image to use for the deployment."""
    startup_timeout: float = 180.0
    """Timeout waiting for runtime to start."""
    runtime_timeout: float = 60.0
    """Timeout for runtime operations."""
    deployment_timeout: float = 3600.0
    """Timeout for the Modal sandbox."""
    modal_sandbox_kwargs: dict[str, Any] = Field(default_factory=dict)
    """Additional keyword arguments passed to `modal.Sandbox.create`."""
    proxy: str | None = None
    """Proxy to use for runtime HTTP requests."""
    type: Literal["modal"] = "modal"
    """Discriminator for (de)serialization/CLI. Do not change."""
    install_pipx: bool = True
    """Whether to install pipx in the Modal image."""

    model_config = ConfigDict(extra="forbid")

    def get_deployment(self, run_id: str):
        from .modal.deployment import ModalDeployment

        return ModalDeployment.from_config(self, run_id)


class VefaasDeploymentConfig(BaseModel):
    """Configuration for veFaaS deployment."""

    image: str | None = None
    """Docker image to use for the sandbox."""
    command: str = "python3 -m swerex.server --auth-token {token}"
    """Command to run in the sandbox with authentication token."""
    timeout: float = 60.0
    """Timeout for runtime operations."""
    startup_timeout: float = 120.0
    """Timeout waiting for runtime to start."""
    function_id: str | None = None
    """veFaaS function ID."""
    function_route: str | None = None
    """veFaaS function route."""
    proxy: str | None = None
    """Proxy to use for the connection."""

    type: Literal["vefaas"] = "vefaas"
    """Discriminator for (de)serialization/CLI. Do not change."""
    model_config = ConfigDict(extra="forbid")

    def get_deployment(self, run_id: str):
        from .vefaas.deployment import VefaasDeployment

        return VefaasDeployment.from_config(self, run_id)


class SshPodmanDeploymentConfig(BaseModel):
    """Run each sandbox in rootless podman on a remote host, over a shared SSH ControlMaster.

    Lets a separate CPU box serve as the sandbox/eval pool while training runs elsewhere;
    mechanism details on :mod:`uni_agent.deployment.ssh_podman.deployment`.
    """

    image: str = "python:3.12"
    """Container image for the sandbox (per-task image is merged in from the data)."""
    command: str = (
        "swerex-remote --host 0.0.0.0 --port {port} --auth-token {token} || "
        "( python3 -m pip install -q swe-rex && "
        "python3 -m swerex.server --host 0.0.0.0 --port {port} --auth-token {token} )"
    )
    """Command run inside the sandbox to start the swerex server. Uses the swerex
    already present in the image, falling back to a pip install for plain images."""
    ssh_host: str
    """SSH target running rootless podman, e.g. ``user@host``."""
    ssh_key: str | None = None
    """Path to the SSH private key used to reach the host."""
    ssh_port: int | None = None
    """SSH port, if not 22."""
    podman_port: int = 8085
    """Loopback TCP port the remote ``podman system service`` listens on."""
    runtime_port: int = 8000
    """Port the swerex server binds inside the sandbox."""
    shell: str = "/bin/bash"
    """Shell used as the container entrypoint."""
    timeout: float = 60.0
    """Timeout for runtime operations."""
    startup_timeout: float = 300.0
    """Timeout waiting for the runtime to come alive (includes image pull)."""
    max_lifetime: int = 7200
    """Hard cap (s) on a sandbox's lifetime, enforced container-side via ``timeout``.

    In-process teardown (``stop``) can't run if the owning process is SIGKILLed, which would
    otherwise leak the container forever. This makes the container self-terminate so
    ``auto_remove`` reaps it. Set well above the longest real rollout."""

    type: Literal["ssh_podman"] = "ssh_podman"
    """Discriminator for (de)serialization/CLI. Do not change."""
    model_config = ConfigDict(extra="forbid")

    def get_deployment(self, run_id: str):
        from .ssh_podman.deployment import SshPodmanDeployment

        return SshPodmanDeployment.from_config(self, run_id)


DeployConfig: TypeAlias = Annotated[
    VefaasDeploymentConfig
    | LocalDeploymentConfig
    | LocalAttachDeploymentConfig
    | HostDeploymentConfig
    | LocalNativeDeploymentConfig
    | ModalDeploymentConfig
    | SshPodmanDeploymentConfig,
    Field(discriminator="type"),
]
