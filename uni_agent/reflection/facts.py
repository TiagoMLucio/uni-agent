"""Facts about a trajectory that a model reads unreliably, computed instead of asked for.

Placement did not move across four rounds of telling a reflector to look earlier, because what
it misses is mechanical: whether anything had been run before the first edit, whether an edit was
repeated verbatim, whether a submit followed any passing check at all.
"""

import re

ARG_PATH_RE = re.compile(r"--path\s+(\S+)")
CMD_RE = re.compile(r"str_replace_editor\s+(\w+)")
EDIT_CMDS = {"str_replace", "insert", "create"}


def call_facts(call: dict) -> tuple:
    action = call.get("action") or ""
    cmd = (CMD_RE.match(action) or [None, None])[1] if CMD_RE.match(action) else None
    path = (ARG_PATH_RE.search(action) or [None, None])[1] if ARG_PATH_RE.search(action) else None
    return cmd, path, action, call.get("observation") or ""


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
            # on the basename, not the path: every task lives under /testbed/, so matching
            # "test" anywhere killed this branch and with it the two placement signals below
            name = path.rsplit("/", 1)[-1] if path else ""
            src = bool(name) and "reproduce" not in name and "test" not in name
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
