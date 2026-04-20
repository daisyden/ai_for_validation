# Get AR from Issue Skill

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
- Access to check_pr_status skill at: `~/ai_for_validation/opencode/issue_triage/.claude/skills/check_pr_status/SKILL.md`

### Source Paths
```
Issue data: GitHub API or WebFetch
PR data: GitHub API or WebFetch  
Comment history: GitHub API or WebFetch
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
# Access check_pr_status skill
read(filePath="~/ai_for_validation/opencode/issue_triage/.claude/skills/check_pr_status/SKILL.md")

# Access triage skills
read(filePath="~/ai_for_validation/opencode/issue_triage/.claude/skills/triage_skills/SKILL.md")
```

---

## Part 1: AR from Related PR

### Step 1: Find Related PR from Issue

**IMPORTANT**: Always use `gh pr view` with known PR numbers. Avoid unreliable search patterns.

**Method 1: Via Timeline API (RECOMMENDED)**
```bash
# Get timeline events referencing cross-referenced PRs
# This finds PRs that explicitly mention this issue in the timeline
gh api repos/intel/torch-xpu-ops/issues/{issue_number}/timeline --paginate --jq \
  '.items[] | select(.event == "cross-referenced") | .source.issue | {number: .number, title: .title, state: .state, url: .html_url}'
```

**Method 2: Via Timeline - Event Type Check**
```bash
# Alternative: Get all cross-reference types
gh api repos/intel/torch-xpu-ops/issues/{issue_number}/timeline --paginate --jq \
  '.items[] | select(.event == "cross-referenced") | {source_type: .source_type, source_issue: .source.issue}'
```

**Method 3: Via PR Body Check (Direct PR Number Reference)**
```bash
# Safely check PR body for issue reference
# NOTE: Body can be null, handle gracefully with (.... // "")
gh api repos/intel/torch-xpu-ops/pulls/{pr_number} --jq \
  'select((.body // "") | contains("{issue_number}")) | {number: .number, title: .title, state: .state}'
```

**Method 4: Via Body Extraction and Regex**
```bash
# Extract issue body for manual pattern matching
gh issue view {issue_number} --repo intel/torch-xpu-ops --json body
# Then grep for patterns like "Fixes #N", "Closes #N", PR URLs
```

### Step 2: CRITICAL Verification of PR Data

**MANDATORY**: After finding a PR number, ALWAYS verify the actual PR content using `gh pr view` with the discovered number. Do NOT assume PR is related based on search results alone.

```bash
# VERIFY: Fetch PR with discovered number to confirm
gh pr view {pr_number} --repo intel/torch-xpu-ops --json number,title,body,state,author

# CRITICAL CHECKS:
# 1. Title context - does it match issue theme?
# 2. Body contains - explicit "Fix #issue_number" or related reference
# 3. Author alignment - expected author for this fix
```

**Verification Checklist**:
- [ ] PR title mentions related operator/feature from issue
- [ ] PR body contains the issue number (exact match)
- [ ] PR state is OPEN or MERGED (not closed without merge)
- [ ] Author is expected for this fix area

**If verification FAILS**:
- No valid PR found
- Document "No related PR identified"
- Do NOT proceed with false PR data

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

## Complete AR Extraction Workflow

```python
def get_AR_from_issue(issue_number: int, repo: str = "intel/torch-xpu-ops") -> dict:
    """
    Extract complete AR from issue including PR and comments.
    
    Steps:
    1. Find related PRs via timeline
    2. VERIFY each PR before use   <-- CRITICAL ADDITION
    3. Analyze validated PR status using check_pr_status logic
    4. If PR merged, mark issue verified
    5. Extract unresolved requests from comments
    6. Combine into structured AR output
    """
    
    ar_result = {
        "issue_number": issue_number,
        "pr_ar": {},
        "comment_ar": {},
        "combined_ar": [],
        "validation_status": "PENDING"
    }
    
    # Step 1: Find related PR
    potential_prs = find_related_prs_via_timeline(issue_number, repo)
    
    # Step 2: VERIFY each potential PR  <-- MANDATORY
    verified_prs = []
    for pr_candidate in potential_prs:
        # ALWAYS fetch and verify actual PR content
        verified = verify_pr_linkage(pr_candidate["number"], issue_number)
        if verified["is_valid"]:
            verified_prs.append(verified["pr_data"])
        else:
            # Log but skip invalid candidates
            log_warning(f"Skipping invalid PR candidate #{pr_candidate['number']}")
    
    ar_result["validation_status"] = "PASS" if verified_prs else "NO_VALID_PR"
    
    # Exit if no valid PRs found
    if not verified_prs:
        ar_result["combined_ar"].append({
            "source": "system",
            "owner": "Triage",
            "content": "No valid related PR identified - Issue needs owner assignment",
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

def verify_pr_linkage(pr_number: int, issue_number: int) -> dict:
    """
    MUST BE CALLED: Verify discovered PR actually links to this issue.
    
    Returns:
    {
        "is_valid": bool,
        "pr_data": {...} or None,
        "verification_details": {...}
    }
    """
    # Step 1: Fetch actual PR data
    pr_data = gh_api(f"repos/intel/torch-xpu-ops/pulls/{pr_number}")
    
    # Step 2: Verify title context
    title_valid = True  # Add domain-specific checks
    
    # Step 3: Verify body contains explicit issue reference
    body = pr_data.get("body", "") or ""
    issue_ref_in_body = f"#{issue_number}" in body or f"/issues/{issue_number}" in body
    
    # Step 4: Make determination
    is_valid = title_valid and issue_ref_in_body
    
    return {
        "is_valid": is_valid,
        "pr_data": pr_data if is_valid else None,
        "verification_details": {
            "title_valid": title_valid,
            "issue_ref_in_body": issue_ref_in_body,
            "body_excerpt": body[:200] if body else ""
        }
    }
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

## Skill Metadata

- **Version**: 1.1.0
- **Created**: 2026-04-20
- **Updated**: 2026-04-20 v1.1 (added PR verification)
- **Related Skills**: check_pr_status
- **Repository**: intel/torch-xpu-ops
- **Requires**: Deep investigation via explore agent
- **Bug Fix**: Verified PR data before analysis