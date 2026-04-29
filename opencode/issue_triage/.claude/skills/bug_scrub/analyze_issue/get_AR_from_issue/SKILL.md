# Get AR from Issue Skill

## Base Path Reference

Relative paths from this file location (`bug_scrub/analyze_issue/get_AR_from_issue/`):
```
../../..                     → issue_triage root
../../../ci_results/         → CI artifacts directory
../../../result/             → Excel results directory
```

## Overview
This skill extracts Action Required (AR) information from GitHub issues by analyzing:
1. Related PRs that fix the issue
2. Issue comment threads for unresolved requests

Uses deep investigation rather than surface pattern matching.

## When to Use
- Extract PR information linked to an issue
- Analyze PR status and action required
- Identify unresolved requests from issue comments
- Determine ownership for action items

## Preconditions

### Required Access
- GitHub CLI authenticated session (recommended)
- WebFetch capability as fallback

### Source Paths (relative from bug_scrub/)
```
Issue data: GitHub API or WebFetch
PR data: GitHub API or WebFetch  
Comment history: GitHub API or WebFetch
CI results: ../../../ci_results/
Excel file: ../../../result/torch_xpu_ops_issues.xlsx
```

---

## Tools

### 1. Bash (gh CLI)

Primary tool for GitHub API access:

```bash
# Find related PRs via timeline
gh api repos/intel/torch-xpu-ops/issues/{issue_number}/timeline --paginate --jq \
  '.items[] | select(.event == "cross-referenced") | .source.issue | {number: .number, title: .title}'

# Fetch PR details
gh pr view {pr_number} --repo intel/torch-xpu-ops --json number,title,body,state,author

# Fetch PR reviews  
gh api repos/intel/torch-xpu-ops/pulls/{pr_number}/reviews --jq '.[] | {user: .user.login, state: .state}'

# Fetch issue comments
gh api repos/intel/torch-xpu-ops/issues/{issue_number}/comments --jq '.[] | {user: .user.login, body: .body}'

# Fetch PR check runs
gh api repos/intel/torch-xpu-ops/pulls/{pr_number}/check-runs --jq '.check_runs[] | {name: .name, conclusion: .conclusion}'
```

### 2. WebFetch

Fallback when gh CLI unavailable:
- `{PR_url}` - Main PR page
- `{PR_url}/checks` - CI status
- `{issue_url}` - Main issue page with comments

### 3. Explore Agent (Deep Analysis)

Use for deep investigation of PR and code:

```python
task(description="pr_issue_deep_relationship",
     prompt=f"""
INVESTIGATION: PR #{{pr_number}} - Issue #{{issue_number}} Relationship

Deep analysis of:
1. PR title and body - verify issue linkage
2. Code changes - understand fix scope
3. Files modified - assess implementation
4. PR status - get check_pr_status gates
5. Issue comments - extract unresolved requests

REQUIRED VECTOR:
- Always verify findings with actual data
- Cross-reference PR body with issue reference patterns
- Use explore to deep-dive issue requirements
     """,
     subagent_type="explore")

task(description="issue_comments_deep_analysis",
     prompt=f"""
DEEP ANALYSIS: Issue #{{issue_number}} Comments for AR Extraction

Analyze:
1. Comment author associations (OWNER > COLLABORATOR > MEMBER)
2. Request type classification (blocking vs informational)
3. Pending unresolved requests
4. Escalation decisions vs technical requirements

Use deep analysis, not pattern matching.
     """,
     subagent_type="explore")
```

### 4. Read Tool
For accessing local skill files:

```python
# Access triage skills
read(filePath="~/ai_for_validation/opencode/issue_triage/.claude/skills/triage_skills/SKILL.md")
```

---

## Part 1: AR from Related PR

### Step 1: Find Related PR from Issue

PR discovery uses **multiple complementary vectors** because no single
vector catches every fix PR. In particular, Copilot- or bot-authored PRs
frequently solve an issue without ever referencing the issue number, so
the timeline alone misses them.

#### Excluded sources — do NOT extract PR numbers from these

The following content is environment metadata, not fix references.
Strip these before any text scan:

- `### Versions` section of GitHub bug-report templates (and any
  `Versions:` / `## Versions` heading variant). Contents typically
  include: framework repo URLs, commit SHAs, dependency versions,
  driver versions, OS info — none of these are fix PRs even when they
  resemble PR-style numbers.
- Stack-trace lines, log excerpts inside fenced code blocks, and `pip`
  / `conda` `freeze`-style output.
- Anything inside `<!-- ... -->` HTML comments.

A PR number that appears **only** inside an excluded section is not a
candidate; the fact that it appears elsewhere in the body is required
to elevate it.

#### Vector 0: GitHub linked PRs (highest authority — auto-verifies)

GitHub's "Development" sidebar (and the Copilot agent UI) records
explicit issue↔PR links that do **not** require any "Fixes #N" text in
the PR body. These links are GitHub-managed metadata and represent the
strongest possible intent signal — a PR returned by this vector is
auto-VERIFIED in Step 2 with `verdict_source: "github_linked"`.

This data is available **only via GraphQL**; REST and `gh issue view`
do not surface it. Always query both directions for completeness.

```bash
# From the issue side: PRs that GitHub will close on merge
gh api graphql -f query='{
  repository(owner:"intel", name:"torch-xpu-ops") {
    issue(number: {issue_number}) {
      closedByPullRequestsReferences(first: 20, includeClosedPrs: true) {
        nodes {
          number title state author { login }
          repository { nameWithOwner }
          createdAt mergedAt
        }
      }
    }
  }
}'
```

For each candidate from any subsequent vector (A–D below), it is also
worth checking the reverse direction to confirm the same link exists:

```bash
# From the PR side: issues this PR will close on merge
gh api graphql -f query='{
  repository(owner:"<repo_owner>", name:"<repo_name>") {
    pullRequest(number: {pr_number}) {
      closingIssuesReferences(first: 10) {
        nodes { number title repository { nameWithOwner } }
      }
    }
  }
}'
```

A PR appearing on either list is treated as VERIFIED in Step 2 even
when its body has no `#<issue>` text and its files don't overlap.

#### Vector A: Timeline cross-references

```bash
gh api repos/intel/torch-xpu-ops/issues/{issue_number}/timeline --paginate --jq \
  '.[] | select(.event == "cross-referenced") |
   {number: .source.issue.number,
    title:  .source.issue.title,
    state:  .source.issue.state,
    pull_request: (.source.issue.pull_request != null),
    repo_url: .source.issue.repository_url,
    url:    .source.issue.html_url}'
```

This catches PRs whose body or commit message says `#<issue>` /
`Fixes #<issue>` / `intel/torch-xpu-ops#<issue>`.

#### Vector B: Issue body explicit references (after excluded-source strip)

After stripping excluded sources, scan the remaining issue body for:
- `#<digits>` mentions
- Full PR URLs: `https?://github.com/[^/]+/[^/]+/pull/\d+`
- Cross-repo refs: `<owner>/<repo>#\d+`

Each surviving number/URL is a candidate. Note the surrounding text so
the verification step can judge intent (e.g., "ToDo: PR #N" is a
strong fix signal; "broken since PR #N" is a regression source, not
the fix).

#### Vector C: Title-keyword search via `gh pr list`

Required when Vectors A and B return zero candidates, **and**
recommended as a sanity check even when they do return candidates
(catches Copilot-authored PRs).

1. Extract distinctive technical keywords from the issue title:
   prefer multi-word phrases, error names, build flags, op names,
   file fragments. Discard generic tokens like `[Bug]`, `bug`, `XPU`,
   `error`, `failed`.
2. Run `gh pr list` over both states with each keyword phrase:

   ```bash
   gh pr list --repo intel/torch-xpu-ops --state all \
              --search "<keyword phrase>" \
              --json number,title,state,author,createdAt
   ```

3. Bound the search by issue creation date — only PRs created within a
   reasonable window around the issue (typically issue creation date
   ± 90 days) are plausible fixes.

#### Vector E: Fix-Approach text scan (post-Phase-3, when present)

After Phase 3.3 (`triage_skills`) has populated the `Fix Approach` column,
scan that text for PR references using the same patterns as Vector B —
`#<digits>`, full PR URLs, and `<owner>/<repo>#<digits>`. The Fix Approach
is LLM-generated and frequently names a concrete PR ("Land
intel/torch-xpu-ops PR #N: ..."), but because it is written *after* the
issue body it is missed by Vectors A–B which scan only the original issue.

Apply the same excluded-source rules as Vector B (strip code blocks and
HTML comments before scanning). Each surviving number/URL is a candidate
that must still pass Step 2 verification — Fix Approach text is
LLM-generated and CANNOT be trusted as a fix signal on its own.

Vector E is the safety net for two real failure modes:
1. The issue body never named the fix PR (PR was opened later).
2. PR discovery via Vectors A–D returned candidates that all REJECTED
   in Step 2, so Part 1 was about to emit "no verified PR".

When Vector E produces a VERIFIED candidate, it overrides the
"no verified PR" path. When it produces nothing, downstream behaviour
is unchanged.

#### Vector D: File-path search (when issue cites specific files)

If the issue body or stack trace identifies specific source files
(e.g., `src/BuildOnWindows.cmake`, `src/ATen/native/xpu/sycl/Atomics.h`):

```bash
gh pr list --repo intel/torch-xpu-ops --state all \
           --search "<filename>" \
           --json number,title,state
```

Then for each candidate, inspect changed files:

```bash
gh pr view <pr_number> --repo intel/torch-xpu-ops --json files \
  --jq '.files[].path'
```

A PR that touches the same file(s) named in the issue is a strong
candidate even without an issue-number reference.

#### Aggregation

Run Vector 0 first; if it returns any candidate, that candidate is
already auto-VERIFIED and Vectors A–E become a sanity check rather
than the primary discovery path. Always run Vectors A–E anyway to
catch additional related PRs (e.g., follow-up fixes, replacement PRs
opened after a prior PR was closed unmerged). Union all candidates
from Vectors 0 + A–E, dedupe by `(repo, pr_number)`, and pass each
through Step 2 verification. Inner-source / private repo PRs (e.g.,
`intel-innersource/...`) cannot be verified through the public API —
flag them as `unverifiable_private` in the output and treat them as
informational only.

**Replacement-PR rule**: if Vector A surfaces a CLOSED-unmerged PR
and Vectors C / D / E surface a later OPEN or MERGED PR touching the
same files or symptoms, the later PR supersedes the closed one for
verdict purposes. Both are recorded in `pr_analysis`, but the live
verdict is taken from the latest non-rejected one. This prevents the
report from parking on a stale dead PR after a replacement lands.

### Step 2: PR Verification (content-match aware)

**MANDATORY**: every candidate from Step 1 must be verified before
being treated as a fix PR. Verification produces one of three verdicts
per candidate: **VERIFIED**, **REJECTED**, or **UNVERIFIABLE_PRIVATE**.

A candidate is **VERIFIED** if any of the following holds (in priority
order):

0. **GitHub-linked** (Vector 0): the PR appears in the issue's
   `closedByPullRequestsReferences` list, or the issue appears in the
   PR's `closingIssuesReferences` list. This is GitHub-managed
   metadata and auto-verifies regardless of body text or file overlap.
   Record `verdict_source: "github_linked"`.

1. **Explicit reference**: PR body or commit message contains a clean
   reference to the issue (`#<N>`, `Fixes #<N>`, `Closes #<N>`,
   `<owner>/<repo>#<N>`, or full issue URL). The reference must NOT
   appear only inside an excluded section (per Step 1) of the PR body.
   Record `verdict_source: "explicit_reference"`.

2. **Content match (no explicit reference required)**: an explore-agent
   review of the PR concludes that the PR addresses the same problem
   the issue describes. Required evidence:
   - **Same surface**: PR-modified files overlap with files named in
     the issue body or stack trace, OR PR-modified files are the
     ones implicated by the issue's error path.
   - **Same symptom**: PR title/body describes the same error /
     behaviour / build flag / op / dtype as the issue.
   - **Plausible timing**: PR created on or after the issue (or in a
     plausible window before, if the PR predates the issue and the
     issue is a "regression caused by …" report).

   The agent must explicitly justify the match in
   `verification_details.match_reasoning`. Loose topical similarity is
   **not** sufficient — require concrete file or symptom overlap.
   Record `verdict_source: "content_match"`.

A candidate is **REJECTED** if:
- It fails both criteria above.
- The PR's body explicitly references a different issue and is scoped
  there.
- Title/files are clearly unrelated to the issue's domain.

A candidate is **UNVERIFIABLE_PRIVATE** if it lives in a private /
inner-source repo accessible only to the issue's owners — record it
as evidence requiring human follow-up, do not treat as a verified fix.

```bash
# Fetch full PR data for verification
gh pr view {pr_number} --repo {repo} \
  --json number,title,body,state,author,createdAt,mergedAt,files,commits

# For content-match, inspect changed files
gh pr view {pr_number} --repo {repo} --json files --jq '.files[].path'
```

**Verification output schema** (per candidate):

```json
{
  "pr_number": <int>,
  "repo": "<owner/name>",
  "verdict": "VERIFIED" | "REJECTED" | "UNVERIFIABLE_PRIVATE",
  "verification_details": {
    "explicit_reference": true | false,
    "reference_excerpt": "<quoted snippet, if any>",
    "content_match": true | false,
    "match_reasoning": "<agent's justification: which files / symptoms overlap>",
    "files_overlap": [ "<path>", ... ]
  }
}
```

If **all** candidates are REJECTED or UNVERIFIABLE_PRIVATE, Part 1
yields no fix PR and downstream AR comes from Part 2 (comments) only.

### Step 2.5: Live PR-State Re-Check (mandatory before emitting verdict)

`pr_analysis.state` recorded at Phase-4b run time is a **point-in-time
snapshot**. By the time Phase 5 renders `bug_scrub.md` (often hours or
days later), a PR's state may have changed:

- A CLOSED-unmerged PR may have been re-opened.
- An OPEN PR may have been merged (most common case).
- A new replacement PR may have superseded a CLOSED one.

A skill that emits "PR #N closed unmerged; reassess fix path" based on
a stale snapshot will mislead the triager. To prevent this, every
verified PR must be re-queried for its current state immediately
before the verdict verb is emitted (in the Phase 4b agent itself, and
in the Phase 5 reconciliation script):

```bash
gh pr view <pr_number> --repo <owner/name> \
  --json state,mergedAt,closedAt,updatedAt
```

Then apply the **state-precedence rule**:

| Live state | Snapshot state | Resulting verdict |
|---|---|---|
| MERGED | any | VERIFY_AND_CLOSE — "Verify fix from merged PR <ref> and close" |
| OPEN | any | TRACK_PR — "Track PR <ref> to merge" |
| CLOSED (unmerged) | any | RETRIAGE_PRS, **but** first re-run Vectors C/D/E to look for a replacement PR; emit RETRIAGE_PRS only if no replacement found |

The CLOSED→RETRIAGE_PRS branch is the failure case that produced wrong
verdicts in past reports — a closed PR is rarely the end of the story
when the issue itself is still open. The replacement-PR re-search is
**mandatory** for this branch, not optional.

The helper script
[`run_live_pr_state_recheck.py`](./run_live_pr_state_recheck.py)
implements this rule and is invoked by the Phase 4b agent during
Step 2.5 and by the Phase 5 reconciliation step.

### DERIVATION RULE: action_TBD from pr_analysis (mandatory)

After Step 2.5 the agent has, for each VERIFIED PR, a fresh
`live_state ∈ {MERGED, OPEN, CLOSED}`. The verb to emit in
`action_TBD` is derived deterministically per the precedence
**MERGED > OPEN > CLOSED-unmerged** of the highest-priority VERIFIED PR:

| Highest-priority live_state | action_Type      | Verb                                                          |
|-----------------------------|------------------|---------------------------------------------------------------|
| MERGED                      | VERIFY_AND_CLOSE | `"Verify fix from merged PR <ref> and close"`                 |
| OPEN                        | TRACK_PR         | `"Track PR <ref> to merge"` + any Step-3 gate-specific verbs  |
| CLOSED unmerged (no replacement) | RETRIAGE_PRS | `"PR <ref> closed unmerged; reassess fix path"`               |

If all VERIFIED PRs are CLOSED-unmerged, the agent **must** re-run
Vectors C/D/E to look for a replacement; if found, recurse the rule on
the replacement's live_state. If still no replacement, emit
RETRIAGE_PRS.

If `pr_candidates` contains zero VERIFIED entries on an OPEN issue,
emit `"No action — investigate further"` (NEED_ACTION); never leave
`action_TBD` blank when `validation_status == "OK"`.

> **History (v1.6):** This rule was previously implemented as a
> post-pass backfill script (`run_pass_backfill.py`) that ran after
> the wave-merge step to repair rows where the agent had returned
> `validation_status:"PASS"` with empty `action_TBD`. Inlining it into
> the agent prompt eliminates the post-pass repair step and produces
> deeper, per-issue justifications written directly into
> `action_reason` from the agent's own analysis context.

### Step 3: Deep PR Analysis Using check_pr_status

Once PR is verified, invoke deep analysis via explore agent:

```python
task(description="pr_deep_analysis_for_issue",
     prompt=f"""
DEEP PR ANALYSIS FOR ISSUE #{{issue_number}}

CONTEXT:
This PR is linked to fix issue #{{issue_number}}

REQUIRED ANALYSIS:

1. PR OVERVIEW
   - PR number, title, author, state
   - Decision: merge status, readiness
   - Key changes summary

2. PR STATUS DEEP DIVE
   Using check_pr_status skill logic:

   a) GATE 1: Waiting for Resolving
      - Check unresolved comments/threads
      - Identify blocking vs informational
      - Author response tracking

   b) GATE 2: Waiting for Review  
      - Count approvals vs required
      - Check review decision
      - Pending reviewer requests

   c) GATE 3: Waiting for CI
      - CI check status
      - Failure classification
      - Artifacts analysis (if available)

   d) GATE 4: Waiting for Merge
      - If all gates passed

3. ACTION REQUIRED EXTRACTION
   For each action item identified:
   - Owner assignment
   - Content description
   - Priority level
   - Blocking status

4. VERIFICATION STATUS
   If PR is MERGED:
   - Issue should be VERIFIED as fixed
   - Note any verification failures
   - Flag if issue remains unfixed

OUTPUT FORMAT:
For each finding, provide:
- Evidence from PR data
- Deep context analysis
- Recommendation

AVOID: Simple pattern matching
USE: Deep contextual investigation
""",
     subagent_type="explore")
```

### Step 3: PR-Issue Deep Investigation

```python
task(description="pr_issue_deep_relationship",
     prompt=f"""
INVESTIGATION: PR #{{pr_number}} - Issue #{{issue_number}} Relationship

CONTEXT:
PR#{{pr_number}} claims to fix issue #{{issue_number}}

INVESTIGATION SCOPE:

1. REQUIREMENTS TRACEABILITY
   - Map PR changes to issue requirements
   - Identify which test cases are addressed
   - Verify fix covers all failing scenarios

2. CODE IMPLEMENTATION ANALYSIS
   For each file modified:
   - Read key implementation sections
   - Identify how the fix addresses root cause
   - Verify no new regressions introduced

3. TEST COVERAGE VERIFICATION
   - Check if PR adds new tests
   - Verify tests cover issue scenario
   - Check broken test fix

4. MERGE IMPACT ASSESSMENT
   - If PR merged: Issue should be verified
   - If PR open: Assess readiness timeline

SOURCE CODE LOCATIONS:
- ~/pytorch/third_party/torch-xpu-ops/src/ATen/native/
- ~/pytorch/third_party/torch-xpu-ops/test/xpu/

EXPECTED DELIVERABLES:
- Issue fix coverage assessment
- Code implementation quality
- Action required for closure
""",
     subagent_type="explore")
```

### Step 4: AR Extraction Template

```markdown
## AR from Related PR

### PR Information
| Field | Value |
|-------|-------|
| PR Number | {pr_number} |
| Title | {title} |
| Author | @{author} |
| State | {merged/open/closed} |
| Decision | {review_decision} |

### PR Verification Status ✓/✗
| Check | Status | Evidence |
|-------|--------|----------|
| Title matches issue theme | ✓/✗ | {title} |
| Body contains "Fix #{issue}" | ✓/✗ | ... |
| State is OPEN/MERGED | ✓/✗ | {state} |

⚠️ IMPORTANT: If any verification check fails, STOP and report "No valid PR identified". Do NOT continue with incorrect data.

### PR Status Analysis

#### Gate 1: Waiting for Resolving
- Status: PASS/FAIL
- Blocking comments: {count}
- Evidence: {details}

#### Gate 2: Waiting for Review
- Status: PASS/FAIL
- Approvals: {count}/{required}
- Decision: {decision}

#### Gate 3: Waiting for CI
- Status: PASS/FAIL
- Failed checks: {list}
- Artifacts: {if available}

#### Gate 4: Waiting for Merge
- Status: PASS/FAIL
- Ready: {yes/no}

### Action Required from PR

| Owner | Content | Priority | Blocking |
|-------|---------|----------|----------|
| {owner} | {action_content} | {P0-P3} | {yes/no} |

### Verification Status
- If MERGED: Issue should be verified
- Current status: VERIFIED/PENDING/NOT_FIXED
```

---

## Part 2: AR from Issue Comments

### Step 1: Fetch Issue Comments

```python
# Get all comments for the issue
gh issue view {issue_number} --repo intel/torch-xpu-ops --json comments
```

Comment structure:
```json
{
  "comments": [
    {
      "author": {"login": "username"},
      "authorAssociation": "OWNER|COLLABORATOR|MEMBER|CONTRIBUTOR|NONE",
      "body": "comment text",
      "createdAt": "ISO-8601",
      "updatedAt": "ISO-8601"
    }
  ]
}
```

### Step 2: Deep Comment Analysis

```python
task(description="issue_comments_deep_analysis",
     prompt=f"""
DEEP ANALYSIS: Issue #{{issue_number}} Comments for AR Extraction

CONTEXT:
Analyze issue #{{issue_number}} comment history to identify unresolved requests.

ANALYSIS FRAMEWORK:

1. COMMENT CLASSIFICATION
   Classify each comment by type and blocking level:

   a) MAINTAINER/CORE REVIEWER COMMENTS
      - High priority requests
      - Blocking actions required
      - Explicit requirements
   
   b) CONTIBUTOR/MEMBER COMMENTS  
      - Technical questions
      - Suggestions requiring consideration
      - Non-blocking
   
   c) BOT/AUTOMATED COMMENTS
      - CI results (informational)
      - Labels added/removed
      - Non-actionable by author

2. ACTION ITEM EXTRACTION
   
   For each actionable comment, identify:
   - Requester (author)
   - Request type (clarification, code change, test addition)
   - Priority (P0-P3)
   - Specific ask content
   - Deadline if mentioned

3. UNRESOLVED REQUEST TRACKING
   Track conversation threads to identify:
   - Open threads awaiting response
   - Follow-up requests without resolution
   - Stale requests (no author response)
   - Confirmed/acknowledged items

4. RESPONSE PATTERN DETECTION
   
   Analyze author response patterns:
   - Questions addressed
   - Requests accepted
   - Requests declined
   - Action items completed

SOURCE CODE LOCATIONS:
- ~/pytorch/aten/src/ATen/native/
- ~/pytorch/third_party/torch-xpu-ops/src/ATen/native/

ACTION:
For each unresolved request, provide:
- Owner (who requested)
- Content (what is needed)
- Priority (P0-P3)
- Blocking status

DO NOT USE simple pattern matching.
USE deep contextual investigation of comment intent.
""",
     subagent_type="explore")
```

### Step 3: Comment Thread Analysis Template

```python
def analyze_comment_threads(comments: list) -> dict:
    """
    Deep analysis of issue comment threads.
    
    Returns:
    {
        "unresolved_requests": [
            {
                "owner": "username",
                "content": "specific request",
                "priority": "P0-P3",
                "blocking": bool,
                "thread_context": "related discussion"
            }
        ],
        "resolved_requests": [...],
        "pending_responses": [...]
    }
    """
```

### Deep Comment Analysis Framework

#### 1. Request Type Classification

| Type | Indicators | Blocking |
|------|-------------|----------|
| Bug Confirmation | "Could you verify", "reproduce issue" | HIGH |
| Info Request | "Can you provide", "please clarify" | MEDIUM |
| Test Request | "Add test for", "test coverage needed" | MEDIUM |
| Documentation | "Document", "please add comment" | LOW |
| Suggestion | "Consider", "might want to" | NONE |

#### 2. Owner Classification

| Association | Weight | Priority |
|-------------|--------|----------|
| OWNER | 1.0 | P0-P1 |
| COLLABORATOR | 0.9 | P1-P2 |
| MEMBER | 0.8 | P2 |
| CONTRIBUTOR | 0.5 | informational |
| NONE | 0.3 | informational |

#### 3. Blocking Level Assessment

```python
BLOCKING_KEYWORDS = {
    "HIGH": ["critical", "must fix", "blocks", "cannot proceed", "required"],
    "MEDIUM": ["please check", "please address", "should verify"],
    "LOW": ["consider", "optional", "nice to have"]
}
```

### Step 4: AR from Comments Output Format

```markdown
## AR from Issue Comments

### Comment Analysis Summary
- Total comments: {count}
- From maintainers/core: {count}
- From contributors: {count}
- Bot/automated: {count}

### Unresolved Requests

| Owner | Comment Date | Request | Priority | Blocking |
|-------|--------------|----------|----------|----------|
| @{user} | {date} | {content} | {P0-P3} | {yes/no} |

### Pending Author Responses
| Requester | Days Pending | Request Summary |
|-----------|---------------|----------------|
| @{user} | {N} days | {truncated content} |

### Resolved/Dismissed Requests
| Owner | Resolution | Date |
|-------|------------|------|
```

---

## Part 3: Not-Target Check

### Purpose
Some issues are filed against features/test cases that the XPU team has explicitly decided NOT to support (won't-fix / out-of-scope / not a target feature). When an authoritative owner communicates such a decision in an issue comment, the triage action is to **label the issue `not_target`** (and close it if the decision covers every test case in the issue).

This check runs **before** the PR-based AR extraction (Part 1) and comment-based AR extraction (Part 2): if the issue is determined to be entirely not-target, downstream AR extraction is short-circuited.

### Detection Philosophy — READ THIS FIRST

**Not-target detection is a reasoning task, not a matching task.** Do NOT use regex, substring search, or a fixed phrase list to classify comments. Whether a comment expresses a binding not-target decision depends on:

- **Who said it** (authority of the commenter),
- **What they mean** (deny-the-feature vs. defer-the-fix vs. workaround-via-skip-list),
- **What scope it covers** (all cases in the issue, a specific subset, or just one case).

All three judgements require full-thread context. Accordingly, this check is implemented as a **single explore-agent invocation** that reads the entire issue (body + every comment, with `authorAssociation`) and returns a structured verdict. The local Python wrapper only gathers inputs and parses the agent's JSON output — it performs no classification itself.

### Step 1: Identify Authoritative Owners

Gather the set of logins whose comments can bind a not-target decision. This is the **only** mechanical step; the explore agent does everything else.

| Role | How to obtain |
|---|---|
| Issue assignees | `gh issue view <id> --repo intel/torch-xpu-ops --json assignees --jq '.assignees[].login'` |
| Accounts whose comments carry `authorAssociation = OWNER` | from comment metadata (returned by the fetch in Step 2) |
| Accounts whose comments carry `authorAssociation = COLLABORATOR` **and** appear in the triage-owner allowlist | allowlist is repo-specific; pass it into the agent |

Comments from `CONTRIBUTOR` / `MEMBER` / `NONE` authors are **never** binding not-target decisions; if they contain request-like content, they are handled by Part 2 (comment AR) instead.

### Step 2: Fetch Full Issue + Comment Thread

```bash
gh issue view {issue_number} --repo intel/torch-xpu-ops \
  --json number,title,body,assignees,comments \
  --jq '{number, title, body,
         assignees: [.assignees[].login],
         comments: [.comments[] | {author: .author.login,
                                   assoc: .authorAssociation,
                                   created: .createdAt,
                                   url: .url,
                                   body: .body}]}'
```

Pass the full result to the explore agent verbatim. Do **not** pre-filter comments by substring; the agent must see the entire thread to reason about scope, superseding statements, and reversals.

### Step 3: Delegate Classification to the Explore Agent (primary mechanism)

```python
task(description="not_target_check",
     subagent_type="explore",
     prompt=f"""
NOT-TARGET CHECK — Issue #{{issue_number}} in {{repo}}

You are classifying whether this issue has been declared not-target
(won't-fix / out-of-scope / not a supported feature on XPU) by an
authoritative owner. This is a *reasoning* task. Do not rely on keyword
matching; read every comment in context and judge intent.

INPUTS
- Issue title, body, assignees, and full comment thread (with
  authorAssociation, author login, timestamp, and URL per comment):
{{issue_json}}
- Authoritative owner set (only these authors can issue binding
  not-target decisions):
    - issue assignees
    - comment authors with authorAssociation in {{OWNER, COLLABORATOR (if
      in allowlist)}}
  Allowlist: {{owner_allowlist}}

PROCEDURE
1. Enumerate the test cases / sub-features the issue is about. Sources:
   issue title, issue body (tables, bullet lists, code blocks listing
   failing tests), and any comments from the reporter that add cases.
   Output the enumerated list as `all_cases`.
   - If the issue is a single-bug report with one test name, `all_cases`
     has one entry.
   - If cases are grouped (e.g., "all complex128 reductions"), preserve
     that grouping semantically — do not expand it into a fabricated
     list.

2. For every comment authored by a member of the authoritative owner
   set, decide: does this comment express a binding decision that the
   feature / test / sub-feature will not be supported on XPU?

   A binding not-target decision means the author is denying the
   feature itself on XPU (permanently, as a scope decision). Judge by
   meaning, not phrasing. Examples of shapes this can take — use them
   as illustrations, NOT as a match list:
     • "This op / dtype / path is not a target for XPU."
     • "We will not support this on XPU."
     • "Out of scope for XPU."
     • "Deprecated upstream; we are not picking it up on XPU."

   The following are explicitly NOT not-target — classify them as
   deferral / workaround / suggestion instead:
     • "Skip it in the skip list" (workaround; issue still open for
       eventual fix).
     • "Won't fix for this release / 2.x / until Triton supports it"
       (priority / deferral, feature is still in scope long-term).
     • "Skip for now" / "disable temporarily" (workaround).
     • A non-owner echoing a won't-fix sentiment.
     • An owner asking a question or speculating ("might not be
       targetable?") — only firm statements count.

3. For each binding not-target comment, determine which case(s) from
   `all_cases` it covers. Three shapes:
     • EXPLICIT: comment names the case(s).
     • BLANKET: comment refers to a whole class that subsumes listed
       cases (e.g., "complex128 reductions are not target" covers every
       complex128 reduction in `all_cases`).
     • SCOPE-LIMITED: comment covers only a subset (e.g., only Windows).
   Record the mapping per comment.

4. Compute:
     • covered_cases = union of cases covered across all binding
       not-target comments.
     • remaining_cases = all_cases − covered_cases.
     • If `covered_cases` is empty → no not-target decision exists.
     • If `remaining_cases` is empty → all cases are not-target (close).
     • Otherwise → partial not-target (label, do not close).

5. Pick `owner_transferred`: the login of the EARLIEST comment (by
   createdAt) that issued a binding not-target decision.

OUTPUT — return exactly this JSON, no prose:
{{
  "action_TBD": "label not_target"
                | "label not_target and close"
                | null,
  "owner_transferred": "<login>" | null,
  "all_cases": [ "<case>", ... ],
  "covered_cases": [ "<case>", ... ],
  "remaining_cases": [ "<case>", ... ],
  "evidence": [
    {{
      "comment_url": "...",
      "author": "<login>",
      "author_assoc": "OWNER|COLLABORATOR|ASSIGNEE",
      "created": "<iso8601>",
      "quoted_phrase": "<short excerpt for the triager's eyes, not a
                        matcher input>",
      "scope_shape": "EXPLICIT" | "BLANKET" | "SCOPE-LIMITED",
      "cases_covered": [ "<case>", ... ],
      "reasoning": "<1-3 sentences explaining why this is binding
                    not-target rather than deferral/workaround>"
    }}
  ]
}}

If no binding not-target decision exists, return `action_TBD: null`
with empty `covered_cases` / `evidence` / `remaining_cases`, and
`all_cases` still populated.
""")
```

### Step 4: Interpret the Agent's Verdict

The wrapper does three things only: map the agent's JSON onto `action_TBD`, pick the short-circuit branch, and attach evidence. No further classification is performed locally.

| Agent verdict | `action_TBD` | Downstream behaviour |
|---|---|---|
| `action_TBD: null` | *(skip Part 3 output; proceed to Part 1 / Part 2 normally)* | — |
| `action_TBD: "label not_target"` (partial) | `label not_target` | Run Part 1 and Part 2 restricted to `remaining_cases`. Tag AR items for covered cases as `superseded-by-not-target`. |
| `action_TBD: "label not_target and close"` (all covered) | `label not_target and close` | **Skip** Part 1 and Part 2. Emit a single combined-AR entry recording the decision. |

### Step 5: Output Template

```markdown
## Not-Target Check

### Enumerated Cases
1. `test_case_1`
2. `test_case_2`
3. ...

### Owner Determination
- Assignees: @{assignee_list}
- Owner allowlist: [...]

### Not-Target Evidence
| Comment Date | Author (assoc) | Scope | Quoted Phrase | Cases Covered | Reasoning |
|---|---|---|---|---|---|
| {date} | @{author} ({OWNER}) | BLANKET | "...not a target for XPU..." | [case_1, case_2] | Firm feature-denial, not a deferral |

### Decision
- `remaining_cases`: [case_3]        ← non-empty → partial
- `action_TBD`: **label not_target**
- `owner_transferred`: @{earliest_owner}

*(or, if remaining_cases is empty:)*
- `action_TBD`: **label not_target and close**
- `owner_transferred`: @{earliest_owner}
```

### Integration with AR Extraction

When `action_TBD` is set:
- Include it at the top of the combined AR output so triagers see the short-circuit decision immediately.
- If `... and close`, the combined AR list collapses to a single entry: `{source: "not_target_check", owner: <owner_transferred>, content: "Label not_target and close", priority: "P3", blocking: false}`.
- If just `label not_target`, add the entry AND continue with Part 1/Part 2 for `remaining_cases`.

---

## Complete AR Extraction Workflow

```python
def get_AR_from_issue(issue_number: int, repo: str = "intel/torch-xpu-ops") -> dict:
    """
    Extract complete AR from issue including not-target check, PR, and comments.
    
    Steps:
    0. NOT-TARGET CHECK: owner-issued won't-fix / out-of-scope decision
    1. Find related PRs via timeline
    2. VERIFY each PR before use
    3. Analyze validated PR status using check_pr_status logic
    4. If PR merged, mark issue verified
    5. Extract unresolved requests from comments
    6. Combine into structured AR output
    """
    
    ar_result = {
        "issue_number": issue_number,
        "action_TBD": None,              # set by not-target check if triggered
        "owner_transferred": None,       # set by not-target check if triggered
        "not_target_evidence": [],
        "remaining_cases": None,
        "pr_ar": {},
        "comment_ar": {},
        "combined_ar": [],
        "validation_status": "PENDING"
    }
    
    # Step 0: Not-Target check (see Part 3)
    not_target = not_target_check(issue_number, repo)
    if not_target["action_TBD"] is not None:
        ar_result["action_TBD"]         = not_target["action_TBD"]
        ar_result["owner_transferred"]  = not_target["owner_transferred"]
        ar_result["not_target_evidence"] = not_target["evidence"]
        ar_result["remaining_cases"]    = not_target["remaining_cases"]
        
        # Short-circuit if all cases are not-target
        if not_target["action_TBD"] == "label not_target and close":
            ar_result["combined_ar"].append({
                "source": "not_target_check",
                "owner": not_target["owner_transferred"],
                "content": "Label not_target and close — owner decision covers all cases",
                "priority": "P3",
                "blocking": False
            })
            ar_result["validation_status"] = "NOT_TARGET_ALL"
            return ar_result
        
        # Partial: continue Part 1/2 but only for remaining_cases
        ar_result["combined_ar"].append({
            "source": "not_target_check",
            "owner": not_target["owner_transferred"],
            "content": f"Label not_target (partial: {len(not_target['evidence'])} "
                       f"case(s) covered); continue AR for remaining_cases",
            "priority": "P3",
            "blocking": False
        })
    
    # Step 1: Find related PR via Vectors 0/A–E (graphql links + timeline +
    # body refs + title-keyword search + file-path search + Fix Approach
    # text scan). See Part 1 Step 1.
    potential_prs = find_related_prs_multivector(issue_number, repo)

    # Step 2: VERIFY each potential PR (content-match aware)
    verified_prs = []
    unverifiable_prs = []
    for pr_candidate in potential_prs:
        verified = verify_pr_linkage(
            pr_candidate["number"], issue_number,
            repo=pr_candidate.get("repo", repo),
        )
        if verified["verdict"] == "VERIFIED":
            verified_prs.append(verified["pr_data"])
        elif verified["verdict"] == "UNVERIFIABLE_PRIVATE":
            unverifiable_prs.append(pr_candidate)
        else:  # REJECTED
            log_warning(f"Rejected PR candidate #{pr_candidate['number']}: "
                        f"{verified['verification_details']}")

    # Step 2.5: Live PR-state re-check (mandatory before emitting verdict).
    # Updates state/mergedAt on each verified PR so a now-merged PR is not
    # reported as CLOSED, and a now-closed PR triggers replacement-PR search.
    verified_prs = [refresh_pr_state(pr) for pr in verified_prs]
    if any(pr["state"] == "CLOSED" and not pr.get("mergedAt") for pr in verified_prs):
        # Re-run Vectors C/D/E to find a replacement PR; if found, append
        # to verified_prs after re-verification.
        replacements = find_replacement_prs(issue_number, repo, verified_prs)
        verified_prs.extend(replacements)

    ar_result["validation_status"] = "PASS" if verified_prs else "NO_VALID_PR"
    if unverifiable_prs:
        ar_result["unverifiable_private_prs"] = unverifiable_prs

    # Exit if no valid PRs found
    if not verified_prs:
        msg = "No valid related PR identified - Issue needs owner assignment"
        if unverifiable_prs:
            msg += (f"; {len(unverifiable_prs)} inner-source candidate(s) "
                    "flagged for human verification")
        ar_result["combined_ar"].append({
            "source": "system",
            "owner": "Triage",
            "content": msg,
            "priority": "P1",
            "blocking": True
        })
        return ar_result
    
    # Step 3: Analyze each verified PR using check_pr_status
    for pr in verified_prs:
        pr_analysis = deep_pr_status_analysis(pr)
        ar_result["pr_ar"][pr['number']] = pr_analysis
        
        # If PR merged, issue should be verified
        if pr_analysis['state'] == 'MERGED':
            ar_result['verification_needed'] = 'VERIFIED'
        
        # Extract AR from PR
        if pr_analysis.get('action_required'):
            ar_result["combined_ar"].extend(pr_analysis['action_required'])
    
    # Step 4: Analyze issue comments
    comments = fetch_issue_comments(issue_number, repo)
    comment_analysis = deep_comment_analysis(comments)
    ar_result["comment_ar"] = comment_analysis
    
    # Extract unresolved requests
    for req in comment_analysis["unresolved_requests"]:
        ar_result["combined_ar"].append({
            "source": "issue_comment",
            "owner": req["owner"],
            "content": req["content"],
            "priority": req["priority"],
            "blocking": req["blocking"]
        })
    
    return ar_result


def not_target_check(issue_number: int, repo: str) -> dict:
    """
    Thin wrapper over the explore-agent invocation in Part 3 Step 3.

    This function performs NO classification of its own:
      1. Collect the authoritative-owner inputs (assignees + allowlist).
      2. Fetch the full issue + comment thread via `gh issue view`.
      3. Invoke the explore agent with the Part 3 Step 3 prompt.
      4. Parse and return the agent's JSON verdict verbatim.

    Do not add substring/regex prefilters here — the agent must see the
    entire thread to reason about scope, supersession, and intent.

    Returns:
    {
        "action_TBD": "label not_target" | "label not_target and close" | None,
        "owner_transferred": str | None,     # login of earliest binding owner
        "all_cases": [str, ...],
        "covered_cases": [str, ...],
        "remaining_cases": [str, ...],       # empty => close
        "evidence": [
            {
                "comment_url", "author", "author_assoc", "created",
                "quoted_phrase", "scope_shape", "cases_covered", "reasoning"
            }, ...
        ]
    }

    Returns action_TBD=None when no binding not-target decision exists.
    """
    ...

def verify_pr_linkage(pr_number: int, issue_number: int,
                      repo: str = "intel/torch-xpu-ops",
                      github_linked_set: set = frozenset()) -> dict:
    """
    Verify a discovered PR per Part 1 Step 2 (3-tier).

    Acceptance paths (priority order):
      0. github_linked  — PR is in the issue's
                          closedByPullRequestsReferences (or vice
                          versa). Auto-VERIFIED. Pass the precomputed
                          set via github_linked_set for O(1) lookup.
      1. explicit       — PR body / commit messages reference the
                          issue (excluded sections stripped).
      2. content_match  — explore agent confirms files / symptoms /
                          timing overlap.
    """
    # Inner-source / private repo? Cannot fetch via public API.
    if repo.startswith("intel-innersource/") or repo.startswith("frameworks.ai."):
        return {
            "verdict": "UNVERIFIABLE_PRIVATE",
            "verdict_source": None,
            "pr_data": None,
            "verification_details": {
                "reason": "Inner-source PR; cannot verify via public API. "
                          "Treat as informational; flag for human follow-up.",
            },
        }

    # Path 0: GitHub-managed link (highest authority).
    if (repo, pr_number) in github_linked_set:
        pr_data = gh_api(f"repos/{repo}/pulls/{pr_number}")
        return {
            "verdict": "VERIFIED",
            "verdict_source": "github_linked",
            "pr_data": pr_data,
            "verification_details": {
                "github_linked": True,
                "explicit_reference": False,
                "content_match": False,
                "match_reasoning": "PR appears in issue's "
                    "closedByPullRequestsReferences (GraphQL); "
                    "GitHub-managed intent link is sufficient.",
            },
        }

    pr_data = gh_api(f"repos/{repo}/pulls/{pr_number}")
    body = strip_excluded_sections(pr_data.get("body") or "")
    commit_msgs = "\n".join(c["commit"]["message"]
                            for c in gh_api(f"repos/{repo}/pulls/{pr_number}/commits"))

    # Path 1: explicit reference.
    explicit = (
        f"#{issue_number}" in body
        or f"/issues/{issue_number}" in body
        or f"#{issue_number}" in commit_msgs
    )
    if explicit:
        return {
            "verdict": "VERIFIED",
            "verdict_source": "explicit_reference",
            "pr_data": pr_data,
            "verification_details": {
                "github_linked": False,
                "explicit_reference": True,
                "reference_excerpt": body[:200],
                "content_match": False,
            },
        }

    # Path 2: content match (delegated to explore agent).
    agent = explore_agent_content_match(pr_number, issue_number, repo)
    if agent["content_match"]:
        return {
            "verdict": "VERIFIED",
            "verdict_source": "content_match",
            "pr_data": pr_data,
            "verification_details": {
                "github_linked": False,
                "explicit_reference": False,
                "content_match": True,
                "match_reasoning": agent["reasoning"],
                "files_overlap": agent["files_overlap"],
            },
        }

    return {
        "verdict": "REJECTED",
        "verdict_source": None,
        "pr_data": None,
        "verification_details": {
            "github_linked": False,
            "explicit_reference": False,
            "content_match": False,
            "match_reasoning": agent["reasoning"],
        },
    }


def fetch_github_linked_prs(issue_number: int, repo: str) -> list[dict]:
    """Vector 0 helper. Returns the list of PRs GitHub considers linked
    to this issue via the Development sidebar / Copilot link / 'Fixes
    #N' parser. Available only via GraphQL; not exposed in REST."""
    owner, name = repo.split("/", 1)
    q = (
        f'{{ repository(owner:"{owner}", name:"{name}") {{ '
        f'  issue(number: {issue_number}) {{ '
        f'    closedByPullRequestsReferences(first:20, includeClosedPrs:true) {{ '
        f'      nodes {{ number title state author {{ login }} '
        f'              repository {{ nameWithOwner }} '
        f'              createdAt mergedAt }} }} }} }} }}'
    )
    nodes = gh_graphql(q)["data"]["repository"]["issue"][
        "closedByPullRequestsReferences"]["nodes"]
    return [{"number": n["number"], "repo": n["repository"]["nameWithOwner"],
             "title": n["title"], "state": n["state"]}
            for n in nodes]


def strip_excluded_sections(text: str) -> str:
    """Remove `### Versions` (and variants), HTML comments, and fenced
    code blocks before scanning for issue/PR references. Used by both
    Step 1 issue-body scanning and Step 2 PR-body scanning."""
    ...
```

### Combined Output Template

```markdown
## Action Required Summary - Issue #{number}

### PR Verification Status
| Check | Status | Evidence |
|-------|--------|----------|
| VERIFICATION PASSED | ✓ | PR #{pr_number} verified |
| VERIFICATION FAILED | ✗ | No valid PR found |

### AR from Related PRs

{PH PR AR TABLE}

### AR from Issue Comments

{AR FROM COMMENTS TABLE}

### Combined Action Items

| # | Source | Owner | Content | Priority | Blocking |
|---|--------|-------|---------|----------|----------|
| 1 | PR | @{pr_author} | {action} | P1 | Yes |
| 2 | Comment | @{commenter} | {request} | P2 | No |

### Validation Status
- Status: {PASS/NO_VALID_PR}
- Verified PRs: {count}

### Issue Resolution Path
- If PR MERGED: Issue verified -> Close
- If PR OPEN + Review APPROVED: Merge -> Close
- If PR OPEN + Waiting Review: Awaiting approval
- If NO VALID PR: Issue needs attention
```

---

## Deep Analysis Guidelines

### Conflict Resolution for Multiple PRs

When multiple PRs address the same issue:
1. Identify primary vs supporting PRs
2. Use check_pr_status analysis for each
3. Consolidate action items by priority
4. Flag conflicting requirements

### Priority Escalation Rules

| Condition | Escalation |
|-----------|------------|
| PR CI Failing + Comment blocking | P0 |
| PR Waiting + Comment request | P1 |
| Multiple unresolved requests | +1 level |
| Stale issues (>30 days pending) | +1 level |

### Owner Assignment Logic

```python
def assign_owner(ar_item: dict) -> str:
    """
    Determine primary owner for AR item.
    Priority: PR Author > Comment Author > Maintainer > Team
    """
    if ar_item["source"] == "pr" and ar_item.get("pr_author"):
        return f"@{ar_item['pr_author']} (PR author)"
    elif ar_item["source"] == "comment":
        return f"@{ar_item['commenter']} (requester)"
    elif ar_item.get("blocking"):
        return "Maintainer"
    else:
        return "Team"
```

---

## Usage Examples

### Example 1: Issue with Merged PR

```python
# Get AR from issue 2442
result = get_AR_from_issue(2442, "intel/torch-xpu-ops")

# Output:
# - PR #3404 is MERGED
# - Issue verification: PASSED
# - No additional AR from comments
# - Combined AR: Issue FIXED
```

### Example 2: Issue with Open PR

```python
# Get AR from issue 3290
result = get_AR_from_issue(3290, "intel/torch-xpu-ops")

# Output:
# - No PR found
# - AR from comments: dtype mismatch investigation
# - Owner: Team
# - Combined AR: Investigate precision issue
```

### Example 3: Issue with Multiple PRs

```python
# Get AR from complex issue
result = get_AR_from_issue(XXXX, "intel/torch-xpu-ops")

# Output:
# - PR #A1 (MERGED) - partial fix
# - PR #A2 (OPEN) - additional changes needed
# - AR locations identified
# - Combined AR consolidated
```

---

## Bug Fix Applied

### Issue Fixed
Used the skill to extract AR from issue #2442 and incorrectly identified PR relationship.

### Root Cause
Used unreliable `select(.body | test("2442"))` pattern matching which returned wrong PR numbers from memory/cache.

### Fix Applied
1. Added **Step 2: CRITICAL Verification of PR Data** - must verify ANY discovered PR
2. Added `verify_pr_linkage()` function to validate PR-issue connection
3. Updated workflow to require verification before proceeding
4. Added validation status tracking in output

---

## Scripts (in this folder)

Helper scripts co-located with this skill. All use `Path(__file__).resolve().parents[7]` to locate the repo root, so they are safe to run from any CWD.

| Script | Purpose |
|---|---|
| [`AGENT_INSTRUCTIONS.md`](./AGENT_INSTRUCTIONS.md) | Canonical Phase 4b explore-agent prompt. Copied into `agent_space/phase4b/` at runtime by wave-builder scripts. Encodes the 6-vector PR discovery, 3-tier verification, Step 2.5 live re-check, 4-gate analysis, and the **DERIVATION RULE** that emits `action_TBD`/`action_reason`/`action_Type` from `pr_analysis`. |
| [`run_phase4b_merge.py`](./run_phase4b_merge.py) | Merge Phase 4b per-issue AR JSON results (from `agent_space/phase4b/wave*/`) into the Issues sheet of `result/torch_xpu_ops_issues.xlsx`. Populates `action_TBD`, `action_reason`, `owner_transferred`. |
| [`run_live_pr_state_recheck.py`](./run_live_pr_state_recheck.py) | Step 2.5 helper. `refresh_pr_state(pr)` re-queries `gh pr view` for the live `state` / `mergedAt` of a single PR. `find_replacement_prs(issue_number, repo, dead_prs)` re-runs Vectors C/D/E when the only verified candidates are CLOSED-unmerged. Used by Phase 4b agents directly and by the Phase 5 reconciliation script. |

> **Note (v1.6 — 2026-04-29):** The previous helper `run_pass_backfill.py`
> was retired. Its rule (VERIFIED + MERGED → VERIFY_AND_CLOSE,
> VERIFIED + OPEN → TRACK_PR, VERIFIED + CLOSED-unmerged → RETRIAGE_PRS)
> was inlined into the Phase 4b agent prompt as the **DERIVATION RULE**
> (see `agent_space/phase4b/AGENT_INSTRUCTIONS.md`). Agents now emit the
> correct `action_TBD` verb during deep analysis, eliminating the need
> for a post-pass backfill pass.

Typical run:
```bash
python3 opencode/issue_triage/.claude/skills/bug_scrub/analyze_issue/get_AR_from_issue/run_phase4b_merge.py
```

---

## Skill Metadata

- **Version**: 1.6.0
- **Created**: 2026-04-20
- **Updated**: 2026-04-29 v1.6 (retired `run_pass_backfill.py`; its classification rule is now the agent-side **DERIVATION RULE** in `AGENT_INSTRUCTIONS.md` — verdicts emitted during deep analysis instead of post-pass backfill)
- **Updated**: 2026-04-27 v1.5 (Vector E: scan `Fix Approach` text for PR references; Step 2.5 live PR-state re-check with replacement-PR search to prevent stale-snapshot verdicts; new helper `run_live_pr_state_recheck.py`)
- **Updated**: 2026-04-21 v1.4 (Part 1 Vector 0: GitHub linked PRs via GraphQL `closedByPullRequestsReferences` — auto-verifies; highest authority)
- **Updated**: 2026-04-21 v1.3 (Part 1 PR discovery rewritten: exclude `### Versions` section, add title-keyword + file-path search vectors, content-match verification path)
- **Updated**: 2026-04-21 v1.2 (added Part 3 not-target check — explore-agent-driven, no pattern matching)
- **Related Skills**: check_pr_status
- **Repository**: intel/torch-xpu-ops
- **Requires**: Deep investigation via explore agent
- **Bug Fix**: Verified PR data before analysis