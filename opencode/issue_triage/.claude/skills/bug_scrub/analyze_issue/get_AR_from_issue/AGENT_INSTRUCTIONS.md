# Phase 4b Agent Instructions (canonical)

> **Canonical location.** This file is the source of truth for the
> Phase 4b explore-agent prompt. The runtime tree
> `agent_space/phase4b/AGENT_INSTRUCTIONS.md` is a copy; orchestration
> scripts that lay out per-issue waves should `cp` this file into
> place. Edit this file, then sync.

You are executing Phase 4b `get_AR_from_issue` of the bug_scrub workflow for ONE GitHub issue. Output ONE JSON file. No code changes.

## Inputs
- Per-issue context: read entry with matching `issue_number` from the wave's `batch.json` (fields: title, reporter, assignee, labels, module, category, priority, dependency, root_cause, fix_approach, cases[]).
- gh CLI is authenticated as `daisyden`. Use `gh api`, `gh pr view`, `gh pr list`, `gh issue view`, `gh pr checks`.
- Repo: intel/torch-xpu-ops (issue lives here). Optionally also search upstream pytorch/pytorch when relevant.

## Steps (execute in order)

### STEP 0 — NOT-TARGET CHECK
```
gh issue view <N> --repo intel/torch-xpu-ops --json body,comments,labels,author,assignees
```
Determine via deep reasoning (not pattern matching) whether an authoritative owner (OWNER/COLLABORATOR/MEMBER/assignee) declared the issue out-of-scope, won't-fix, expected behavior, or duplicate.
Verdict: `"label_not_target_and_close"` (full short-circuit), `"label_not_target_partial"` (continue Part 1/2 for remaining cases), or `null` (proceed).

### PART 1 — PR DISCOVERY (run all 6 vectors)
- **V0 GraphQL** (auto-VERIFY):
  ```
  gh api graphql -f query='{ repository(owner:"intel", name:"torch-xpu-ops") { issue(number: <N>) { closedByPullRequestsReferences(first: 20, includeClosedPrs: true) { nodes { number title state author { login } repository { nameWithOwner } createdAt mergedAt } } } } }'
  ```
- **VA timeline**:
  ```
  gh api repos/intel/torch-xpu-ops/issues/<N>/timeline --paginate --jq '.[] | select(.event=="cross-referenced") | {number: .source.issue.number, title: .source.issue.title, state: .source.issue.state, pr: (.source.issue.pull_request != null), repo: .source.issue.repository_url, url: .source.issue.html_url}'
  ```
- **VB body refs**: scan issue body for `#N`, full PR URLs, `owner/repo#N`. **STRIP** `### Versions` section, fenced code blocks, and `<!-- -->` HTML comments first.
- **VC title-keyword**: `gh pr list --repo intel/torch-xpu-ops --state all --search "<phrase>" --json number,title,state,author,createdAt`. Bound by issue creation date ±90 days. Skip generic tokens (XPU, bug, error, [Bug]).
- **VD file-path**: if issue cites specific files, `gh pr list ... --search "<filename>"` and inspect via `gh pr view <pr> --json files --jq '.files[].path'`.
- **VE Fix Approach scan**: scan the `fix_approach` text from batch.json for `#N` / PR URLs (same excluded-source rules as VB).
- Optional: also search upstream `pytorch/pytorch` if the root cause lives there.

UNION + DEDUPE candidates by (repo, pr_number).

### VERIFICATION (3-tier, mandatory per candidate)
- **VERIFIED — github_linked**: appears in V0 result.
- **VERIFIED — explicit_reference**: PR body or commit message has clean `#<N>` / `Fixes #<N>` / `Closes #<N>` / `<owner>/<repo>#<N>` / full issue URL, NOT inside an excluded section.
- **VERIFIED — content_match**: explore-style reasoning shows file overlap + symptom match + plausible timing. Justify in `match_reasoning`.
- **REJECTED**: fails all of the above.
- **UNVERIFIABLE_PRIVATE**: inner-source / private repo.

### STEP 2.5 — LIVE PR-STATE RE-CHECK (mandatory before emitting verdict)
For every VERIFIED PR:
```
gh pr view <pr> --repo <owner/name> --json state,mergedAt,closedAt,updatedAt,reviewDecision
```
State precedence:
- **MERGED** → action: `"Verify fix from merged PR <ref> and close"`
- **OPEN** → action: `"Track PR <ref> to merge"`
- **CLOSED unmerged** → re-run VC/VD/VE for replacement; if found use that. If still none → `"RETRIAGE_PRS"`.

### STEP 3 — check_pr_status 4 GATES (only for OPEN VERIFIED PRs)
- **Gate 1 Resolving**: unresolved review threads (`gh api repos/<repo>/pulls/<n>/reviews`, comments).
- **Gate 2 Review**: approvals vs required (use `reviewDecision`).
- **Gate 3 CI**: `gh pr checks <n> --repo <repo>`.
- **Gate 4 Merge**: ready to merge?
Identify the blocking gate and the specific owner action.

### PART 2 — COMMENT AR
For each issue comment, classify by author association (OWNER/COLLABORATOR/MEMBER/CONTRIBUTOR/NONE) and request type (`blocking` | `informational` | `answered`). Extract unresolved blocking requests with the owner who should act.

## Output JSON (write ONE file, no other side effects)

Path: `/home/daisydeng/pytorch/agent_space/phase4b/wave<W>/result_<N>.json`

```json
{
  "issue_number": <N>,
  "validation_status": "OK" | "ERROR" | "PRIVATE_ONLY",
  "not_target_verdict": "label_not_target_and_close" | "label_not_target_partial" | null,
  "not_target_reasoning": "...",
  "pr_candidates": [
    {
      "pr_number": 0,
      "repo": "owner/name",
      "vector": "0|A|B|C|D|E",
      "verdict": "VERIFIED|REJECTED|UNVERIFIABLE_PRIVATE",
      "verdict_source": "github_linked|explicit_reference|content_match",
      "live_state": "MERGED|OPEN|CLOSED",
      "live_merged_at": null,
      "review_decision": null,
      "blocking_gate": null,
      "match_reasoning": "...",
      "files_overlap": []
    }
  ],
  "comment_ar": [
    {
      "comment_idx": 0,
      "author": "login",
      "association": "OWNER|COLLABORATOR|MEMBER|CONTRIBUTOR|NONE",
      "request_type": "blocking|informational|answered",
      "text": "...",
      "owner_should_act": "login"
    }
  ],
  "action_TBD": ["..."],
  "action_reason": ["..."],
  "owner_transferred": ["earliest binding owner login"],
  "summary": "1-2 sentence narrative"
}
```

## Canonical action_TBD phrases (use when applicable)
- `"Verify fix from merged PR <ref> and close"`
- `"Track PR <ref> to merge"`
- `"RETRIAGE_PRS"`
- `"label not_target and close"`
- `"Resolve unresolved review comments on PR <ref>"`
- `"Address CI failures on PR <ref>"`
- `"Address comment AR from <owner>: <topic>"`
- `"No action — investigate further"`

Free-form is allowed if no canonical fits.

## DERIVATION RULE — action_TBD from pr_analysis (mandatory)

You MUST emit at least one entry in `action_TBD` whenever there is at least one
VERIFIED PR candidate. The verb is derived deterministically from the
`live_state` of the highest-priority VERIFIED PR per this precedence
(MERGED > OPEN > CLOSED-unmerged):

| Highest-priority live_state of VERIFIED PR(s) | action_Type    | Verb to emit                                                             |
|-----------------------------------------------|----------------|--------------------------------------------------------------------------|
| MERGED (live `state==MERGED` or `mergedAt!=null`) | VERIFY_AND_CLOSE | `"Verify fix from merged PR <ref> and close"`                            |
| OPEN                                          | TRACK_PR       | `"Track PR <ref> to merge"` (+ Step-3 gate-specific verbs if applicable) |
| CLOSED unmerged AND no replacement found      | RETRIAGE_PRS   | `"PR <ref> closed unmerged; reassess fix path"`                          |
| CLOSED unmerged AND replacement found via VC/VD/VE re-search | (recurse on the replacement PR's live_state) | (recurse) |

Additional rules:
- If the issue is OPEN and zero VERIFIED PR candidates exist, emit
  `"No action — investigate further"` (NEED_ACTION) — do NOT leave
  `action_TBD` empty.
- If `not_target_verdict == "label_not_target_and_close"`, emit
  `"label not_target and close"` and you MAY skip PR-derived verbs.
- For OPEN VERIFIED PRs that fail one of the 4 gates, ALSO emit the
  matching gate verb (`"Resolve unresolved review comments on PR <ref>"`,
  `"Address CI failures on PR <ref>"`, etc.) in addition to `Track PR ... to merge`.
- For each verb emitted, write a corresponding 1-sentence justification in
  `action_reason` (same array length is preferred but not required;
  downstream tooling unions across both arrays).

**Never** return `validation_status:"OK"` with empty `action_TBD` while
`pr_candidates` contains a VERIFIED entry — that combination produced
the wrong verdicts that the obsolete `run_pass_backfill.py` was patching.

## Critical rules
- Use ONLY actual gh CLI / GraphQL output. Never invent PR numbers.
- Empty arrays are valid.
- Output ONLY the JSON file — do not modify any other files.
- `<ref>` in action_TBD should be human-readable like `intel/torch-xpu-ops#3475` or `pytorch/pytorch#175657`.
