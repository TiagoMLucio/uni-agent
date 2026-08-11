"""Facts about a trajectory that a model reads unreliably, computed instead of asked for.

Placement did not move across four rounds of telling a reflector to look earlier, because what
it misses is mechanical: whether anything had been run before the first edit, whether an edit was
repeated verbatim, whether a submit followed any passing check at all.
"""

import re

ARG_PATH_RE = re.compile(r"--path\s+(\S+)")
CMD_RE = re.compile(r"str_replace_editor\s+(\w+)")
EDIT_CMDS = {"str_replace", "insert", "create"}


def _call_facts(call: dict) -> tuple:
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
            cmd, path, action, obs = _call_facts(call)
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
