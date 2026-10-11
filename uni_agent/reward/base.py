"""Abstract base for reward specs."""

import base64
from abc import ABC, abstractmethod

#: Echoed only when the whole extraction chain ran. The side session issues its command with
#: ``check="ignore"``, so swe-rex extracts no exit status, and the redirect creates the output
#: file whether or not the diff was ever reached: without this a failed extraction is read back
#: as an empty patch, which is the agent changing nothing.
PATCH_EXTRACT_OK = "__uniagent_patch_extract_ok__"


def patch_extract_command(patch_file: str, diff_args: str = "", repo_dir: str = "/testbed") -> str:
    """The staged working-tree diff, written to ``patch_file``, with the sentinel after it.

    One definition for both specs: this text lived in two files and so did the defect in it. The
    unstaging step may fail, everything up to the diff may not, so the diff sits inside a group
    the ``&&`` chain guards; a bare ``;`` covers the whole prefix, and then a failed ``cd`` diffs
    whatever directory the session is in and a failed ``git add`` diffs an unstaged index.

    A text diff cannot carry a binary (git emits only "Binary files differ", which no apply
    command accepts), so binaries the agent left behind are unstaged first. Git calls a file
    binary only on a NUL byte, so created files that are not UTF-8 go too: a profiler dump that
    passed as text made the patch unreadable and zeroed a correct fix. ``diff_args`` are
    extra git-diff flags for the reflector's copy; the graded prediction is always taken with
    none. The attributes file is what makes ``-W`` find Python function boundaries: without it
    git falls back to a heuristic that expands every hunk to the whole file.

    What the agent leaves behind is not ours to predict, so nothing here trusts a file's name or
    kind. Tracked changes are staged first, then only the untracked regular files and symlinks,
    which is all a text patch carries: ``git add -A`` aborts on a nested repository without a
    commit, and that lost the whole prediction. Paths travel NUL-separated and are taken
    literally: a name starting with ``:`` is pathspec magic to git, ``*`` is a glob that unstaged
    every file, and a quoted name (a newline, a tab, a non-ASCII byte) never matched what it named.
    No literal tab or newline may appear in this text, since it is typed into a shell.

    The diff is also left base64-encoded beside it, for :func:`read_patch`.
    """
    attrs = "/tmp/.uniagent_gitattributes"
    git = "git --literal-pathspecs"
    each = "xargs -0 -r sh -c"
    # an untracked file that cannot be staged is dropped; it must not take the tracked edits with it
    stage = (
        "git add -u && { git ls-files -z --others --exclude-standard "
        f"| {each} 'for f; do if [ -L \"$f\" ] || [ -f \"$f\" ]; then printf \"%s\\0\" \"$f\"; fi; done' _ "
        f"| xargs -0 -r {git} add --ignore-errors -- ; true ; }}"
    )
    # one numstat record per argument, "-<tab>-<tab><path>" for a binary
    unstage_binaries = (
        "git diff --cached --no-renames --numstat -z "
        f"| {each} 't=$(printf \"\\t\"); for r; do case \"$r\" in \"-$t-$t\"*) "
        f"{git} reset -q -- \"${{r#-$t-$t}}\";; esac; done' _"
    )
    unstage_non_utf8 = (
        "git diff --cached --no-renames --name-only --diff-filter=A -z | { command -v iconv >/dev/null && "
        f"{each} 'for f; do [ -L \"$f\" ] || iconv -f UTF-8 -t UTF-8 < \"$f\" >/dev/null 2>&1 || "
        f"{git} reset -q -- \"$f\"; done' _ ; }}"
    )
    return (
        f"cd {repo_dir} && printf '*.py diff=python\\n' > {attrs} && {stage} && "
        f"{{ {unstage_binaries} ; {unstage_non_utf8} ; "
        f"git -c core.attributesFile={attrs} diff --no-color --ignore-submodules=all {diff_args} --cached "
        f"> {patch_file} ; }} && base64 {patch_file} > {patch_file}.b64 && echo {PATCH_EXTRACT_OK}"
    )


async def read_patch(env, patch_file: str) -> str:
    """The diff :func:`patch_extract_command` wrote, with its line endings as git wrote them.

    The sandbox reads a file as text, which turns every CRLF into LF: the patch of a file with CRLF
    line endings then matches nothing in the eval sandbox, and the edit is lost. Hence the base64
    copy. Bytes that are not UTF-8 are escaped rather than refused.
    """
    raw = base64.b64decode(await env.read_file(f"{patch_file}.b64"))
    return raw.decode("utf-8", errors="backslashreplace")


def empty_patch_flag(patch: str | None) -> dict[str, bool]:
    """``{"empty_patch": ...}`` when a prediction was extracted, and nothing at all when it was not.

    Absent is not False. An empty patch is an outcome, the agent changed nothing; a prediction
    that never arrived is a failure on our side, and reporting it as the former is how a lost
    patch became a zero indistinguishable from a wrong fix.
    """
    return {} if patch is None else {"empty_patch": not patch.strip()}


class AbstractRewardSpec(ABC):
    """Reward spec: computes reward from interaction result and optional env eval."""

    @abstractmethod
    async def compute_reward(self, interaction_result: dict, **kwargs) -> tuple:
        """
        Compute reward (and optionally run eval in env) from the interaction result.

        Returns:
            A 2-tuple whose first element is the reward score (or eval report) and
            whose second element is auxiliary info; the concrete element types
            depend on the reward spec.

        Reward-extra-info convention:
            If the second element is a dict and contains a ``"reward_extra_info"``
            key, ``UniAgentLoop`` surfaces it on the trajectory's
            ``extra_fields["reward_extra_info"]``, where downstream training code can
            consume it (for example, textual ``feedback`` describing the attempt).
            Specs that emit such info should populate ``result["reward_extra_info"]``
            with the relevant keys (e.g. ``{"feedback": <str | None>}``; ``None`` when
            there is nothing to report, such as on success).
        """
        ...
