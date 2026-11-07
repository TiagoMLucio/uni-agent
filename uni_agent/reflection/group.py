"""Where one task's attempts in a training step meet, so a failure's reflection can read its successful siblings.

The attempts of a task run on any rollout worker of any node, so they meet in a folder on the shared filesystem: each
writes its outcome there once it is graded (a success also its turn table and patch), a failure waits until every
attempt has written, and the first failure to claim it writes the task's digest, which the others read.
"""

import asyncio
import json
import os
import time
from collections.abc import Awaitable, Callable
from pathlib import Path

#: how often a waiting attempt looks at the folder again
POLL_S = 10.0


class TaskGroup:
    """The ``size`` attempts of one task, met in ``folder``; ``member`` is this attempt's name there."""

    def __init__(self, folder: Path, size: int, member: str):
        self.folder = Path(folder)
        self.size = size
        self.member = member

    async def publish(self, success: bool, attempt: dict | None = None) -> None:
        """This attempt's outcome, and for a success what the digest reads of it. The attempt is written before the
        outcome, so a reader that sees a success's outcome finds its attempt."""
        if success:
            await asyncio.to_thread(_write, self.folder / "successes" / f"{self.member}.json", attempt or {})
        await asyncio.to_thread(_write, self.folder / "members" / f"{self.member}.json", {"success": bool(success)})

    async def wait(self, timeout: float) -> tuple[list[str], int, bool]:
        """Once every attempt has written its outcome, or ``timeout`` has passed: the successes by name, the outcomes
        written, and whether that is every attempt."""
        deadline = time.monotonic() + timeout
        while True:
            names = await asyncio.to_thread(_names, self.folder / "members")
            if len(names) >= self.size or time.monotonic() >= deadline:
                break
            await asyncio.sleep(min(POLL_S, max(deadline - time.monotonic(), 0.0)))
        outcomes = await asyncio.to_thread(lambda: [_read(self.folder / "members" / f"{n}.json") for n in names])
        successes = [n for n, row in zip(names, outcomes, strict=True) if row and row.get("success")]
        return successes, len(names), len(names) >= self.size

    async def attempts(self, names: list[str]) -> list[dict]:
        return await asyncio.to_thread(lambda: [_read(self.folder / "successes" / f"{n}.json") or {} for n in names])

    async def digest(self, write: Callable[[], Awaitable[str]], timeout: float) -> tuple[str | None, str, bool]:
        """The task's digest, written by the first attempt to claim it and read by the others: (the digest or None,
        the writer's error when it failed, whether this attempt wrote it). A reader gives up after ``timeout``."""
        path = self.folder / "digest.json"
        if await asyncio.to_thread(_claim, self.folder / "digest.lock"):
            try:
                text, error = await write(), ""
            except Exception as exc:
                text, error = None, repr(exc)
            await asyncio.to_thread(_write, path, {"digest": text, "error": error, "writer": self.member})
            return text, error, True
        deadline = time.monotonic() + timeout
        while True:
            row = await asyncio.to_thread(_read, path)
            if row is not None:
                return row.get("digest"), row.get("error") or "", False
            if time.monotonic() >= deadline:
                return None, f"no digest within {timeout:.0f} s", False
            await asyncio.sleep(min(POLL_S, max(deadline - time.monotonic(), 0.0)))


def _write(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(row))
    os.replace(tmp, path)


def _read(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return None


def _names(folder: Path) -> list[str]:
    try:
        entries = os.listdir(folder)
    except FileNotFoundError:
        return []
    return sorted(e[: -len(".json")] for e in entries if e.endswith(".json") and not e.startswith("."))


def _claim(path: Path) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.close(os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
    except FileExistsError:
        return False
    return True
