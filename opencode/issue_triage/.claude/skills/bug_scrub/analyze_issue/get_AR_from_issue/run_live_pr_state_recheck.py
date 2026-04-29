"""Step 2.5 helper: live PR-state re-check + replacement-PR search.

`pr_analysis.state` recorded at Phase-4b run time is a point-in-time snapshot.
By the time the report is rendered, that state may be stale — a CLOSED PR
may have been re-opened or replaced, an OPEN PR may have been merged.
Emitting verdicts from stale state is the documented failure mode that
produced wrong "closed unmerged; reassess" verbiage in past reports.

This module exposes two pure-ish functions:

    refresh_pr_state(pr)
        Re-query `gh pr view` for the live state/mergedAt of a single PR.
        Returns a NEW dict (does not mutate input).

    find_replacement_prs(issue_number, repo, dead_prs)
        When all known verified PRs are CLOSED-unmerged, re-run the
        cheap-to-run discovery vectors (C: title-keyword search; D:
        file-path search; E: Fix-Approach scan) and return any newer PR
        that touches the same files or symptoms. Caller is responsible
        for re-verifying each replacement via `verify_pr_linkage`.

Both functions assume the `gh` CLI is authenticated.
"""
import json
import re
import subprocess
from pathlib import Path
from typing import Iterable

REPO = Path(__file__).resolve().parents[7]


def _gh_json(args: list[str]) -> dict | list | None:
    """Run `gh` and parse stdout as JSON. Return None on failure (e.g.
    private/unverifiable repo, network error, deleted PR)."""
    try:
        out = subprocess.run(
            ["gh", *args],
            capture_output=True, text=True, timeout=30, check=True,
        ).stdout.strip()
        return json.loads(out) if out else None
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
            json.JSONDecodeError):
        return None


def refresh_pr_state(pr: dict) -> dict:
    """Return a new pr-dict with `state` / `mergedAt` / `closedAt` updated
    from live GitHub. If the live query fails (private repo, deleted PR,
    network error), return the input unchanged with `live_recheck_failed:
    True` recorded so callers can treat the snapshot as untrusted.

    Input dict shape (subset):
        {"pr_number": int, "repo": "<owner>/<name>", "state": str, ...}
    """
    pr_number = pr.get("pr_number") or pr.get("number")
    repo = pr.get("repo")
    if not pr_number or not repo:
        return {**pr, "live_recheck_failed": True}

    live = _gh_json([
        "pr", "view", str(pr_number),
        "--repo", repo,
        "--json", "state,mergedAt,closedAt,updatedAt",
    ])
    if live is None:
        return {**pr, "live_recheck_failed": True}

    refreshed = {**pr,
                 "state": live.get("state", pr.get("state")),
                 "mergedAt": live.get("mergedAt"),
                 "closedAt": live.get("closedAt"),
                 "updatedAt": live.get("updatedAt"),
                 "live_recheck_failed": False}
    # Normalise the boolean some downstream code expects.
    refreshed["merged"] = bool(live.get("mergedAt"))
    return refreshed


def is_dead_pr(pr: dict) -> bool:
    """A PR is 'dead' for verdict purposes when it is CLOSED but never
    merged. OPEN and MERGED PRs are not dead."""
    return pr.get("state") == "CLOSED" and not pr.get("mergedAt")


# ---------------------------------------------------------------------------
# Replacement-PR search
# ---------------------------------------------------------------------------

# PR-reference patterns shared with Vector B / Vector E. Stripped of the
# excluded-source rule here because Fix-Approach text doesn't carry
# `### Versions` blocks.
_PR_REF_RE = re.compile(
    r"(?:(?P<repo>[\w.-]+/[\w.-]+))?#(?P<num>\d+)|"
    r"https?://github\.com/(?P<url_repo>[\w.-]+/[\w.-]+)/pull/(?P<url_num>\d+)"
)


def extract_pr_refs(text: str, default_repo: str) -> list[tuple[str, int]]:
    """Extract `(repo, pr_number)` tuples from free-text. Used by Vector E
    on `Fix Approach` and reused by the Phase 5 reconciliation script."""
    if not text:
        return []
    refs: list[tuple[str, int]] = []
    for m in _PR_REF_RE.finditer(text):
        num = int(m.group("num") or m.group("url_num"))
        repo = m.group("repo") or m.group("url_repo") or default_repo
        refs.append((repo, num))
    # Dedupe preserving order.
    seen: set[tuple[str, int]] = set()
    return [r for r in refs if not (r in seen or seen.add(r))]


def find_replacement_prs(issue_number: int,
                         repo: str,
                         dead_prs: Iterable[dict],
                         fix_approach_text: str = "") -> list[dict]:
    """Search for PRs that may supersede a closed-unmerged PR.

    Strategy (cheap → expensive):
      1. Vector E rerun: parse `fix_approach_text` for PR references. This
         catches the common case where Phase 3 already named the
         replacement.
      2. For each dead PR, list other PRs that touch any of the same files
         AND were created/merged AFTER the dead PR's `closedAt`.

    Returns candidate dicts in the same shape as Vector A output:
        {"pr_number", "repo", "title", "state", "discovery_vector"}

    Caller MUST run these through `verify_pr_linkage` before treating any
    as a fix.
    """
    candidates: list[dict] = []

    # --- 1. Vector E rerun on Fix Approach text -----------------------------
    for ref_repo, ref_num in extract_pr_refs(fix_approach_text, repo):
        meta = _gh_json([
            "pr", "view", str(ref_num),
            "--repo", ref_repo,
            "--json", "number,title,state,mergedAt",
        ])
        if meta is None:
            continue
        candidates.append({
            "pr_number": meta["number"],
            "repo": ref_repo,
            "title": meta.get("title", ""),
            "state": meta.get("state", "UNKNOWN"),
            "discovery_vector": "E_fix_approach",
        })

    # --- 2. File-overlap search per dead PR --------------------------------
    for dead in dead_prs:
        if not is_dead_pr(dead):
            continue
        files = dead.get("files") or []
        closed_at = dead.get("closedAt")
        for path in files[:5]:  # cap to keep searches bounded
            results = _gh_json([
                "pr", "list",
                "--repo", dead.get("repo", repo),
                "--state", "all",
                "--search", path,
                "--json", "number,title,state,createdAt,mergedAt",
            ]) or []
            for r in results:
                if r["number"] == dead.get("pr_number"):
                    continue
                if closed_at and (r.get("createdAt") or "") < closed_at:
                    continue
                candidates.append({
                    "pr_number": r["number"],
                    "repo": dead.get("repo", repo),
                    "title": r.get("title", ""),
                    "state": r.get("state", "UNKNOWN"),
                    "discovery_vector": "D_replacement_search",
                })

    # Dedupe by (repo, pr_number).
    seen: set[tuple[str, int]] = set()
    unique: list[dict] = []
    for c in candidates:
        key = (c["repo"], c["pr_number"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(c)
    return unique


__all__ = ["refresh_pr_state", "is_dead_pr",
           "extract_pr_refs", "find_replacement_prs"]
