"""Facts about a trajectory that a model reads unreliably, computed instead of asked for.

Placement did not move across four rounds of telling a reflector to look earlier, because what
it misses is mechanical: whether anything had been run before the first edit, whether an edit was
repeated verbatim, whether a submit followed any passing check at all.
"""

import re

from uni_agent.interaction.behaviour import EDIT_CMDS, edit_command, edit_path, is_source_path


def call_facts(call: dict) -> tuple:
    action = call.get("action") or ""
    return edit_command(action), edit_path(action), action, call.get("observation") or ""


def turn_candidates(turns: list[dict]) -> str:
    """Turns worth considering, found by reading the trajectory rather than by asking for them.

    Placement has not moved in four rounds of telling the writer to look earlier, and what it
    keeps missing is mechanical: whether anything had been run before the first edit, whether an
    edit was repeated verbatim, whether a submit followed any passing check at all.
    """
    notes, ran, edits, first_edit = [], None, {}, None
    for turn in turns:
        step = turn.get("step")
        for call in turn.get("tools") or []:
            cmd, path, action, obs = call_facts(call)
            src = is_source_path(path)
            if call.get("name") == "execute_bash" and "python" in action and ran is None:
                ran = step
                notes.append(f"turn {step}: the first time anything is actually run")
            if cmd in EDIT_CMDS and src:
                if first_edit is None:
                    first_edit = step
                    notes.append(f"turn {step}: first edit to source ({path})"
                                 + ("" if ran else ", and nothing has been run yet"))
                key = action[:300]
                if key in edits:
                    notes.append(f"turn {step}: repeats verbatim the edit made at turn {edits[key]}")
                edits[key] = step
            if "No replacement was performed" in obs:
                notes.append(f"turn {step}: the edit matched nothing in the file")
            elif "Traceback" in obs:
                notes.append(f"turn {step}: what it ran raised an exception")
            if call.get("name") == "submit":
                notes.append(f"turn {step}: submits"
                             + ("" if ran else ", having never run anything"))
    return "\n".join(notes[:35]) or "(nothing mechanical stands out)"


#: build and install output, and dependencies copied into the tree: never the agent's own code
_ARTIFACT_DIR = re.compile(r"(^|/)(\.eggs|build|dist|\.tox|\.?venv|site-packages|_vendor|[^/]+\.egg-info)/")
_COPY_NAME = re.compile(
    r"(\.(orig|bak|backup|old|rej|save)$)|([._-](backup|orig|old|copy|restored|bak|original|fixed|new|temp|tmp|buggy|broken)\.\w+$)",
    re.I,
)
_TEST_DIR = re.compile(r"(^|/)(tests?|testing)/")


def _file_blocks(diff: str):
    """(path, block, created, deleted) for every file a unified diff touches."""
    for block in re.split(r"(?m)^(?=diff --git )", diff or ""):
        if not block.startswith("diff --git "):
            continue
        header = re.match(r'diff --git "?a/(.+?)"? "?b/', block)
        path = header.group(1) if header else block.split("\n", 1)[0][len("diff --git "):]
        yield (path, block.rstrip("\n"), bool(re.search(r"(?m)^new file mode", block)),
               bool(re.search(r"(?m)^deleted file mode", block)))


def _changed_lines(block: str) -> int:
    return sum(1 for line in block.split("\n") if line[:1] in "+-" and not line.startswith(("+++", "---")))


def created_source(path: str) -> bool:
    """A file the agent created that reads as part of the package rather than a script, a test,
    a note, a copy of another file or build output."""
    return ("/" in path and path.endswith((".py", ".pyx", ".pyi")) and is_source_path(path)
            and not _TEST_DIR.search(path) and not _ARTIFACT_DIR.search(path) and not _COPY_NAME.search(path))


def patch_view(agent_patch: str, reference: str = "") -> str:
    """The attempt's patch as the reflector reads it: every change to an existing file whole, and of
    the files it created only those that are package source or that the reference creates too.

    Over 6,361 SWE-smith attempts, 99.5% of the files agents created were written by a command the
    trajectory already shows (scripts, tests, notes, backups), none was imported by a passing fix,
    and middle-cutting the raw patch hid every source change in 20% of reflections. The others are
    listed by name and size; a directory holding many collapses to one line.
    """
    created_also = {path for path, _, created, _ in _file_blocks(reference) if created}
    shown, listed, deleted = [], [], []
    for path, block, created, gone in _file_blocks(agent_patch):
        if gone:
            deleted.append(f"{path} ({_changed_lines(block)} lines)")
        elif created and path not in created_also and not created_source(path):
            listed.append((path, _changed_lines(block)))
        else:
            shown.append(block)
    parts = ["\n".join(shown) if shown else "(no change to an existing file)"]
    if listed:
        by_dir: dict[str, list[tuple[str, int]]] = {}
        for path, lines in listed:
            by_dir.setdefault(path.split("/", 1)[0] + "/" if "/" in path else "", []).append((path, lines))
        rows = []
        for top, files in by_dir.items():
            if top and len(files) > 5:
                rows.append(f"- {top} ({len(files)} files, {sum(n for _, n in files)} lines)")
            else:
                rows += [f"- {path} ({lines} lines)" for path, lines in files]
        parts.append("Files the attempt created, not shown:\n" + "\n".join(rows))
    if deleted:
        parts.append("Files the attempt deleted:\n" + "\n".join(f"- {d}" for d in deleted))
    return "\n\n".join(parts)


def patch_delta(gold: str, agent_patch: str) -> str:
    """The two patches reduced to the changes they disagree on, stated as replacements.

    A reflector handed the two diffs has to work out which side is the destination, and a 4B
    reading a traceback instead gets it backwards whenever the fix DELETES the line that raises:
    on one task five of six arms wrote "add `realms = []` before the line", while the fix removes
    the conditional entirely. Only the arms reading a delta got it right. Signs are therefore
    never shown here: each file says what the working code stops containing and what it contains
    instead, and says outright when a file the fix needs was never touched.
    """
    def by_file(diff: str) -> dict[str, tuple[list[str], list[str]]]:
        out: dict[str, tuple[list[str], list[str]]] = {}
        current = None
        for line in (diff or "").splitlines():
            header = re.match(r"^diff --git a/(\S+)", line)
            if header:
                current = header.group(1)
                out.setdefault(current, ([], []))
            elif current is None or not line or line.startswith(("+++", "---")):
                continue
            elif line[0] == "+":
                out[current][1].append(line[1:])
            elif line[0] == "-":
                out[current][0].append(line[1:])
        return out

    gold_by, agent_by = by_file(gold), by_file(agent_patch)
    blocks = []
    for path, (gone, arrived) in gold_by.items():
        was, now = agent_by.get(path, ([], []))
        removed = [x for x in gone if x not in was and x.strip()]
        added = [x for x in arrived if x not in now and x.strip()]
        if not removed and not added:
            continue
        part = [path]
        if path not in agent_by:
            part.append("  this file is not changed at all, and it has to be")
        if removed:
            part.append("  the working code no longer contains:")
            part += [f"      {x.strip()}" for x in removed]
        if added:
            part.append("  the working code contains instead:")
            part += [f"      {x.strip()}" for x in added]
        blocks.append("\n".join(part))
    extra = [f"  {path}" for path, (was, now) in agent_by.items()
             if path not in gold_by and any(x.strip() for x in was + now)]
    tail = ("\n\nChanged where the working fix changes nothing:\n" + "\n".join(extra)
            if extra else "")
    body = "\n\n".join(blocks) or "  nothing: the two agree on every line"
    return "What the working code has that this one does not:\n\n" + body + tail
