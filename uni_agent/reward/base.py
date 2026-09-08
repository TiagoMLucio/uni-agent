"""Abstract base for reward specs."""

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
    command accepts), so binaries the agent left behind are unstaged first. ``diff_args`` are
    extra git-diff flags for the reflector's copy; the graded prediction is always taken with
    none. The attributes file is what makes ``-W`` find Python function boundaries: without it
    git falls back to a heuristic that expands every hunk to the whole file.
    """
    attrs = "/tmp/.uniagent_gitattributes"
    return (
        f"cd {repo_dir} && printf '*.py diff=python\\n' > {attrs} && git add -A && "
        "{ git diff --cached --numstat | awk -F'\\t' '$1==\"-\"{print $3}' "
        "| xargs -r -d '\\n' git reset -q -- ; "
        f"git -c core.attributesFile={attrs} diff --no-color {diff_args} --cached "
        f"> {patch_file} ; }} && echo {PATCH_EXTRACT_OK}"
    )


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
