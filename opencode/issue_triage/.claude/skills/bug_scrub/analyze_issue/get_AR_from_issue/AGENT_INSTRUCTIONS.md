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
- **VERIFIED — github_linked**: appears in V0 `closedByPullRequestsReferences` result. (GitHub itself asserts the PR closes this issue.)
- **VERIFIED — explicit_reference**: PR body or commit message contains a *fixing-verb* reference to this issue: `Fixes #<N>`, `Closes #<N>`, `Resolves #<N>`, `Fix for #<N>`, `closes intel/torch-xpu-ops#<N>`, or full issue URL with one of those verbs in the same sentence — NOT inside an excluded section.
  - A bare `#<N>` mention, "see #<N>", "related to #<N>", "surfaced by #<N>", "exposed by #<N>", "tracked in #<N>", "follow-up to #<N>", "depends on #<N>" does **NOT** qualify. Mark these `verdict: "REJECTED"`, `rejection_reason: "reference_only"`.
  - When ambiguous, read the PR diff intent: a PR whose diff modifies the kernel/op/file named in the issue's `fix_approach` is a fix candidate; a PR that only adds/enables tests for an already-failing case is **not** a fix.
- **VERIFIED — content_match**: explore-style reasoning shows file overlap + symptom match + plausible timing. Justify in `match_reasoning`. Same fix-vs-surface caveat applies: enabling tests is not fixing.
- **REJECTED**: fails all of the above. Always set `rejection_reason` (`"reference_only"`, `"no_overlap"`, `"wrong_symptom"`, `"timing_mismatch"`, etc.).
- **UNVERIFIABLE_PRIVATE**: inner-source / private repo.

### RELATIONSHIP CLASSIFICATION (mandatory for every VERIFIED candidate)

For each VERIFIED PR, classify the issue↔PR relationship:

| relationship   | Meaning                                                                                | Flows into action_TBD?                  |
|----------------|----------------------------------------------------------------------------------------|------------------------------------------|
| `fixes`        | PR's intent is to make this issue's failure stop reproducing                           | YES — drives the DERIVATION RULE table   |
| `supersedes`   | PR replaces a previous fixing PR for this issue                                        | YES — recurse on this PR                 |
| `surfaces`     | PR exposes/discovers this issue (e.g. enables tests that were already failing)         | NO — log only                            |
| `related`      | Same module/area but neither fixes nor surfaces                                        | NO — log only                            |
| `unknown`      | Insufficient evidence                                                                  | NO — treat as `related`                  |

Only `fixes` and `supersedes` candidates contribute action_TBD verbs (`Track PR …`, `Verify fix from merged PR …`). `surfaces` and `related` are recorded for transparency but produce **no** PR-tracking verbs and **no** owner_transferred attribution to the PR author.

If after classification there are zero `fixes`/`supersedes` candidates, treat the issue as having zero VERIFIED PRs for action_TBD purposes — fall back to `"No action — investigate further"` (per the action_TBD derivation rule).

**Worked example (do NOT repeat this mistake)**: Issue intel/torch-xpu-ops#3530 body says "tracks an XPU numerical-accuracy gap surfaced (but not introduced) by PR #3475". PR #3475 enables 22 files of CUDA test coverage; it does not modify the `index_add_` kernel referenced in #3530's `fix_approach`. Correct classification: `relationship = "surfaces"`, do NOT emit `Track PR …#3475`. Correct verb: `"No action — investigate further"`. Correct `owner_transferred`: blank (no Assignee, and Reporter is forbidden per the owner_transferred rule).

### STEP 2.5 — LIVE PR-STATE RE-CHECK (mandatory before emitting verdict)
For every VERIFIED PR (regardless of relationship — we still record live state):
```
gh pr view <pr> --repo <owner/name> --json state,mergedAt,closedAt,updatedAt,reviewDecision
```
State precedence (only applied when `relationship in {"fixes","supersedes"}`):
- **MERGED** → action: `"Verify fix from merged PR <ref> and close"`
- **OPEN** → action: `"Track PR <ref> to merge"`
- **CLOSED unmerged** → re-run VC/VD/VE for replacement; if found use that. If still none → `"RETRIAGE_PRS"`.

For `relationship in {"surfaces","related","unknown"}`: live_state is recorded for transparency but produces NO action_TBD verb.

### STEP 3 — check_pr_status 4 GATES (only for OPEN VERIFIED `fixes`/`supersedes` PRs)
- **Gate 1 Resolving**: unresolved review threads (`gh api repos/<repo>/pulls/<n>/reviews`, comments). Capture the `createdAt` of the most recent unresolved-thread comment.
- **Gate 2 Review**: approvals vs required (use `reviewDecision`).
- **Gate 3 CI**: `gh pr checks <n> --repo <repo>`. Capture `completedAt` of the latest failing/required check.
- **Gate 4 Merge**: ready to merge?
Identify the blocking gate and the specific owner action.

### STEP 3.5 — STALENESS COMPUTATION (mandatory)
Compute "now" from bash: `date -u +%Y-%m-%dT%H:%M:%SZ`. Then for every gating signal and every comment AR, compute `age_days = (now - signal_timestamp).days` and `stale = age_days > 7`.

### PART 2 — COMMENT AR
For each issue comment from `gh issue view <N> --json comments`, capture `createdAt`, classify by author association (OWNER/COLLABORATOR/MEMBER/CONTRIBUTOR/NONE) and request type (`blocking` | `informational` | `answered`). For each blocking unresolved request, set `created_at`, compute `age_days` and `stale = age_days > 7`, and record `owner_should_act`.

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
      "rejection_reason": "reference_only|no_overlap|wrong_symptom|timing_mismatch|null",
      "relationship": "fixes|supersedes|surfaces|related|unknown",
      "live_state": "MERGED|OPEN|CLOSED",
      "live_merged_at": null,
      "review_decision": null,
      "blocking_gate": null,
      "blocking_signal_at": null,
      "blocking_signal_age_days": null,
      "blocking_signal_stale": false,
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
      "created_at": "2026-04-01T12:00:00Z",
      "age_days": 0,
      "stale": false,
      "text": "...",
      "owner_should_act": "login"
    }
  ],
  "action_TBD": ["..."],
  "action_reason": ["..."],
  "owner_transferred": ["earliest binding owner login (Assignee, never Reporter)"],
  "summary": "1-2 sentence narrative"
}
```

## Canonical action_TBD phrases (use when applicable)
- `"Verify fix from merged PR <ref> and close"`
- `"Track PR <ref> to merge"`
- `"RETRIAGE_PRS"`
- `"label not_target and close"`
- `"Resolve unresolved review comments on PR <ref>"`
- `"Resolve unresolved review comments on PR <ref> (>1 week)"` — when latest unresolved-thread comment > 7 days old
- `"Address CI failures on PR <ref>"`
- `"Address CI failures on PR <ref> (>1 week)"` — when latest failing required check `completedAt` > 7 days old
- `"Address comment AR from <owner>: <topic>"`
- `"Address comment AR from <owner> (>1 week): <topic>"` — when the comment_ar entry has `stale: true` (note the space before `(>1 week)`)
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
- If the issue is OPEN and zero VERIFIED PR candidates with `relationship in {"fixes","supersedes"}` exist, emit
  `"No action — investigate further"` (NEED_ACTION) — do NOT leave
  `action_TBD` empty. PR candidates classified as `surfaces`/`related` do NOT count toward this check.
- If `not_target_verdict == "label_not_target_and_close"`, emit
  `"label not_target and close"` and you MAY skip PR-derived verbs.
- For OPEN VERIFIED PRs (relationship `fixes`/`supersedes`) that fail one of the 4 gates, ALSO emit the
  matching gate verb (`"Resolve unresolved review comments on PR <ref>"`,
  `"Address CI failures on PR <ref>"`, etc.) in addition to `Track PR ... to merge`.
- **Staleness suffix**: when emitting a gate verb, check the `blocking_signal_stale` flag for that gate. If true (signal > 7 days old), append ` (>1 week)` per the canonical phrases. The `Track PR <ref> to merge` verb itself does NOT get the stale suffix — only gate verbs do.
- **Comment AR staleness**: when emitting `Address comment AR from <owner>: <topic>`, check the originating `comment_ar[].stale` flag. If true, emit the `(>1 week)` form instead. Note the required space before `(>1 week)`.
- For each verb emitted, write a corresponding 1-sentence justification in
  `action_reason` (same array length is preferred but not required;
  downstream tooling unions across both arrays).

**Never** return `validation_status:"OK"` with empty `action_TBD` while
`pr_candidates` contains a VERIFIED entry — that combination produced
the wrong verdicts that the obsolete `run_pass_backfill.py` was patching.

## DERIVATION RULE — owner_transferred (mandatory)

`owner_transferred` is the engineer who is on the hook for the next action.
It is NOT a record of who reported the issue.

Rules:
- Source of truth, in order: (1) issue `Assignee` if set; (2) the explicit
  `owner_should_act` from a binding comment AR; (3) blank.
- **NEVER** use the issue `Reporter` as `owner_transferred`. The reporter
  filed the bug — that does not make them responsible for fixing or
  investigating it.
- For rows whose `action_TBD` contains `"No action — investigate further"`
  (alone OR combined with other verbs like `"Address comment AR from …"`,
  `"Close the fixed issue"`, etc.): if no Assignee exists, leave
  `owner_transferred` **blank**. Do not fall back to the reporter. A blank
  cell is intentional — it lets the Phase 5 row classifier surface the
  issue under `NEEDS_OWNER` so an owner can be assigned manually.
- This rule applies regardless of how many tokens the `action_TBD` cell
  contains. Multi-token rows (e.g. `"Close the fixed issue | No action —
  investigate further"`) are subject to the same blank-vs-Assignee logic.

## Critical rules
- Use ONLY actual gh CLI / GraphQL output. Never invent PR numbers.
- Empty arrays are valid.
- Output ONLY the JSON file — do not modify any other files.
- `<ref>` in action_TBD should be human-readable like `intel/torch-xpu-ops#3475` or `pytorch/pytorch#175657`.
