# Installation

Uni-Agent can run directly on top of the standard `verl` training environment. In practice, this means you can start from an existing `verl` setup or an official `verl` Docker image, and then install a small set of additional dependencies required by Uni-Agent.

This is the recommended setup for both large-scale inference and agent RL training, because Uni-Agent reuses `verl` for the training/runtime stack rather than replacing it.

---

## Recommended Base Environment

Start from one of the following:

- an existing `verl` training environment that is already working
- an official `verl` Docker image that matches your rollout backend, such as vLLM or SGLang

If you are not sure which image to use, check the `verl` Docker documentation first and choose the image that matches your backend and CUDA stack.

After that, clone Uni-Agent and install the extra dependencies described below.

```bash
git clone https://github.com/yyDing1/uni-agent.git
cd uni-agent
pip install -e .
```

---

## Required Extra Dependencies

On top of the base `verl` environment, Uni-Agent typically needs the following Python packages:

```bash
pip install --no-cache-dir swe-rex loguru pydantic pydantic_settings
pip install --no-cache-dir --upgrade aiohttp
```

These packages are used for:

- `swe-rex`: persistent sandbox runtime used by Uni-Agent environments
- `loguru`: structured logging used by Uni-Agent
- `pydantic` and `pydantic_settings`: config models and settings management
- `aiohttp`: upgraded for compatibility with the runtime stack

---

## Optional Dependencies By Task

Some dependencies are only needed for specific workloads.

### VEFAAS / Volcengine remote deployment

If you use the VEFAAS deployment backend, install the Volcengine Python SDK:

```bash
pip install --no-cache-dir volcengine-python-sdk
```

This is only needed for remote deployment on VEFAAS. It is not required for purely local runs.

### SWE-Bench style tasks

If you run SWE-Bench-based interaction, verification, or reward evaluation, install:

```bash
pip install --no-cache-dir swebench
```

This is needed by Uni-Agent reward/evaluation code for SWE-Bench tasks.

### R2E-Gym tasks

If you use R2E-Gym datasets or rewards, install `R2E-Gym` from source:

```bash
git clone https://github.com/R2E-Gym/R2E-Gym.git /home/R2E-Gym
cd /home/R2E-Gym
pip install --no-cache-dir --no-deps -e .
```

In some containerized setups, Git may complain about repository ownership. If that happens, mark the repo as safe:

```bash
git config --system --add safe.directory /home/R2E-Gym
```

### CuPy

Some `verl`-side distributed utilities depend on CuPy. If your chosen base image does not already include it, install:

```bash
pip install --no-cache-dir cupy-cuda12x==13.6.0
```

If your `verl` image already ships with a compatible CuPy build, you do not need to install it again.

---

## Minimal Installation Matrix

Use the table below as a quick checklist:

| Scenario | Extra packages |
|----------|----------------|
| Any Uni-Agent run | `swe-rex`, `loguru`, `pydantic`, `pydantic_settings`, upgraded `aiohttp` |
| VEFAAS deployment | `volcengine-python-sdk` |
| SWE-Bench tasks | `swebench` |
| R2E-Gym tasks | editable install of `R2E-Gym` |
| Some `verl` distributed backends | `cupy-cuda12x==13.6.0` if missing |

---

## Example: Derived Docker Image

Below is the logical diff of a Uni-Agent-ready image on top of a `verl` base image:

```dockerfile
FROM <your-verl-base-image>

RUN pip install --no-cache-dir swe-rex loguru pydantic pydantic_settings
RUN pip install --no-cache-dir --upgrade aiohttp

# Optional: VEFAAS
RUN pip install --no-cache-dir volcengine-python-sdk

# Optional: SWE-Bench
RUN pip install --no-cache-dir swebench

# Optional: CuPy, only if missing from the base image
RUN pip install --no-cache-dir cupy-cuda12x==13.6.0

# Optional: R2E-Gym
RUN git clone https://github.com/R2E-Gym/R2E-Gym.git /home/R2E-Gym
WORKDIR /home/R2E-Gym
RUN pip install --no-cache-dir --no-deps -e .
RUN git config --system --add safe.directory /home/R2E-Gym
```

---

## What To Read Next

Once the environment is ready:

- go to [`Launch an Agent Environment`](agent_env.html) for sandbox setup
- go to [`Run parallel agent interaction`](agent_interaction.html) for large-scale inference and verification
- go to [`Train an agent with reinforcement learning`](agent_train.html) for RL training
