# check_pr_status

Analyzes GitHub PR status using deep contextual analysis to determine triage state and provide detailed next steps. Performs comprehensive examination of PR review status, discussion threads, CI checks, and requirements to categorize readiness for merge.

## Usage

```
User: check_pr_status
Input: Provide a GitHub PR URL
Example: https://github.com/intel/torch-xpu-ops/pull/3090
```

## Preconditions

### Supported Repositories

- `intel/torch-xpu-ops` - Intel XPU operations for PyTorch
- `pytorch/pytorch` - PyTorch main repository
- Other GitHub public repositories with similar review policies

### Input Requirements

- Valid HTTPS GitHub PR URL (e.g., `https://github.com/org/repo/pull/123`)
- Public read access to the repository

### Access Requirements

- **gh CLI authenticated session**: Recommended but optional. When authenticated, enables structured API access.
- **Fallback**: Uses WebFetch when gh CLI has bad credentials or is unavailable.
- **GitHub rate limits**: Must respect GitHub API rate limits when using gh api. Use WebFetch as fallback for heavy requests.

## Tools

### Primary Tools

#### 1. gh api (GitHub CLI)

GitHub REST API via gh CLI for structured data access:

```bash
# Fetch PR metadata
gh api repos/{owner}/{repo}/pulls/{pr_number}

# Fetch reviews
gh api repos/{owner}/{repo}/pulls/{pr_number}/reviews

# Fetch comments
gh api repos/{owner}/{repo}/pulls/{pr_number}/comments
```

**Limitation**: Requires authenticated gh session. May return "Bad credentials" error.

#### 2. WebFetch

Web content fetcher for GitHub PR pages when gh CLI unavailable:

| URL Pattern | Content Retrieved |
|-------------|-------------------|
| `{PR_url}` | Main PR page: title, state, author, reviewers, conversation |
| `{PR_url}/checks` | CI checks status, job details, annotations, artifacts |
| `{PR_url}/comments` | Discussion thread comments |
| `actions/runs/{run_id}` | Detailed CI run information |
| `actions/runs/{run_id}/job/{job_id}` | Specific job details and logs |

**Parameters**:
- `format`: markdown (default)
- `timeout`: max 120 seconds

#### 3. bash (for command execution)

Used for gh CLI commands and file operations.

## Triage Decision Logic

The triage state is determined by evaluating four gates in strict order. Only ONE state applies at any time:

### Gate 1: Waiting for Resolving

**Condition**: Unresolved active discussion threads or open comments requiring author response.

**Blocking Level**: HIGH
- Comments typically block merge until resolved by maintainer action or explicit clarification.

**Indicators**:
- Open discussion threads from reviewers/maintainers
- Comments marked as "Outdated" but not resolved
- Author-tagged follow-up requests pending response
- Comments with explicit "Check" tags or "@author" mentions

**Analysis Required**:
1. Extract ALL comment threads from PR
2. Identify thread state (open/resolved/hidden)
3. Determine comment source (maintainer/human vs automated/bot)
4. Evaluate if comment is blocking opinion vs informational suggestion

**Deep Analysis Criteria**:
```python
# Pseudocode for evaluation
unresolved_comments = filter_by_state(comments, state="open")
blocking_comments = []

for comment in unresolved_comments:
    # Copilot AI and review bots are typically informational
    if is_automated_review(comment):
        continue  # Informational only
    
    # Human maintainer comments need response
    if is_maintainer_reviewer(comment):
        if suggests_code_change(comment) OR requests_clarification(comment):
            blocking_comments.append(comment)
        else:
            informational_comments.append(comment)
    
    # Check for explicit resolution requirements
    if has_followup_request(comment):
        blocking_comments.append(comment)
```

### Gate 2: Waiting for Review

**Condition**: Comments resolved/closed BUT insufficient approving reviews.

**Blocking Level**: MEDIUM

**Thresholds by Repository**:

| Repository | Minimum Approvals Required |
|-----------|---------------------------|
| `intel/torch-xpu-ops` | 1 (verified from requirements) |
| `pytorch/pytorch` | 1 (standard GitHub default) |
| *Custom* | Check repo settings |

**Indicators**:
- Zero approvals - needs review
- Approvals < required minimum
- Pending review requests visible
- Approval withdrawn or dismissed

**Analysis Required**:
1. Extract ALL review objects
2. Count APPROVED reviews only (dismissed/changes_requested don't count)
3. Compare against minimum threshold
4. Note requested reviewers pending action

**Deep Analysis Criteria**:
```python
# Evaluate approval status
approvals = filter_by_state(reviews, state="APPROVED")
required_minimum = get_repo_minimum_approvals(repo)

if len(approvals) >= required_minimum:
    review_gate = "PASSED"
else:
    review_gate = "FAILED"
```

### Gate 3: Waiting for CI

**Condition**: Unpassed CI checks identified.

**Blocking Level**: HIGH

**Requirements**: All required CI checks MUST pass before merge.

**Analysis Depth**:

1. **Status Collection**:
   - Query CI checks page for all jobs
   - Identify job states: queued, in_progress, completed, skipped, cancelled

2. **Annotation Review** (CRITICAL):
   - Parse annotations for each job
   - Identify ERROR annotations (cause blocking)
   - Separate from WARNING annotations (informational)

3. **Failure Classification**:

| Failure Type | Indicator | Blocking |
|-------------|-----------|----------|
| Test Failure | "Process completed with exit code 1" or test assertion failures | YES |
| Build Failure | Compilation/linking errors | YES |
| Runtime Error | Segfault, OOM, timeout | YES |
| Infrastructure Warning | Node.js deprecation notices | NO |
| Skipped (by label) | Jobs skipped via disable_* labels | NO |

4. **Artifact Investigation**:
   When failure artifacts available:
   - Download failure lists (XPU-UT-Failure-List-*)
   - Analyze specific test names that failed
   - Determine if failures are:
     - Related to PR changes
     - Pre-existing flakiness
     - Infrastructure/environment issues

**Indicators from Checks Page**:
```markdown
Status: Failure
Annotations: 2 errors and 9 warnings
[linux-ut (op_ut) / summary]
  Process completed with exit code 1.
Show more Show less
```

### Gate 4: Waiting for Merge

**Condition**: ALL gates passed BUT PR not merged.

**Prerequisites for This State**:
- ✅ 2+ approvals (or 1+ per repo requirement)
- ✅ All unresolved comments addressed/closed
- ✅ All required CI checks passing
- ✅ No blocking labels or draft state

**Next Action**: PR ready for merge - squash/merge button should be enabled.

---

## Deep Analysis Framework

### Comment Classification

Not all comments are merge-blocking. Evaluate context:

| Comment Type | Example | Blocking |
|-------------|---------|----------|
| Maintainer Request | "@author Please address this review comment" | YES |
| Technical Suggestion | Copilot AI code review with improvement suggestions | NO |
| Documentation Request | "Please add comments explaining X vs Y" | OPTIONAL |
| INFO/NIT | Minor style suggestions | NO |
| Question | "Can you clarify why X is different from Y?" | LATER |
| Bug Report | "This breaks Z scenario" | YES |

**Key Distinction**: Copilot AI and automated review systems provide suggestions but DO NOT constitute blocking reviews unless maintainer explicitly adopts them as requirements.

### CI Annotation Deep Dive

#### Parse Annotations Structure:

```markdown
# Success
Status: ✅ Passed/Jobs completed

# Failure with detail
Status: Failure
Annotations: 2 errors and 9 warnings
[linux-ut (op_ut) / summary]
  Process completed with exit code 1.
Show more Show less

# Infrastructure warning (non-blocking)
[conditions-filter-win]
Node.js 20 actions are deprecated...
```

#### Classification Logic:

```python
def classify_annotation(annotation):
    if "exit code 1" in annotation:
        return CI_RESULT.FAILED
    elif "exit code 0" in annotation:
        return CI_RESULT.PASSED
    elif "deprecated" in annotation.lower():
        return CI_RESULT.INFRASTRUCTURE_WARNING
    elif "failed" in annotation.lower() and "waiting" not in annotation.lower():
        return CI_RESULT.FAILED
    else:
        return CI_RESULT.UNCERTAIN
```

### Label Interpretation

Labels indicate PR state and CI scoping:

| Label | Meaning | Action Required |
|-------|---------|-----------------|
| `disable_e2e` | End-to-end tests skipped intentionally | NO |
| `disable_build` | Build job skipped, use nightly wheel | NO |
| `disable_distributed` | Distributed tests skipped | NO |
| `WIP` | Work in Progress | YES - Remove before merge |
| `breaking-change` | Breaking API change | Requires approval from specific owners |
| `needs-rebase` | Branch out of date | YES - Rebase required |

---

## Output Format

### Structured Response Sections

1. **Current TBD**: Primary triage state determination with single source truth
2. **Detailed Status Table**: Gate-by-gate status comparison

| Criteria | Status | Notes |
|----------|--------|-------|

3. **Deep Analysis**: Per-gate detailed reasoning
4. **Next Steps**: Prioritized action items to advance PR toward merge
5. **Evidence Links**: URLs to relevant GitHub pages for verification

### Example Output Structure:

```markdown
## PR #{number} Triage Status

### Current TBD: {state}

### Detailed Analysis

#### 1. Reviews: {status}
- Approved by: {list}

#### 2. Comments: {status}
- Open threads: {count}
- Blocking: {count}

#### 3. CI Status: {status}
- Failed jobs: {list}
- Profile: {classification}

### Next Steps Required

1. **[Critical]**: {action}
2. **[Optional]**: {action}

### Summary

| Gate | Status | Blocker |
|------|--------|---------|
| Waiting for resolving | YES | No |
| Waiting for review | NO | - |
| Waiting for CI | YES | Yes |
| Waiting for merge | NO | - |
```

---

## Interpretation Notes

### Critical Rules

1. **Order Matters**: Check gates in exact order: resolving → review → CI → merge
2. **First Blocking Gate Wins**: Once any gate blocks, that becomes TBD
3. **Copilot AI Exception**: Automated review suggestions do NOT block merge
4. **Label Scoping**: Jobs skipped via disable_* labels are not failures
5. **Infrastructure Warnings**: Node.js deprecation notices are never blocking

### Maintainer Policy Considerations

Different repositories have different policies:
- Some require 2+ approvals regardless of default
- Some block merge on ANY unresolved comment
- Some treat Copilot/bot comments as informational only

Always verify against specific repo settings.

### Artifact Analysis Value

Failure artifacts (test lists) are goldmines for root cause:
- Download `XPU-UT-Failure-List-*.txt` files
- Parse specific test names that failed
- Compare failures against files changed
- Determine if fix needed in PR or pre-existing issue

---

## Workflow Steps

```
1. VALIDATE INPUT → Parse PR URL, extract owner/repo/number
         ↓
2. FETCH PR METADATA → gh api or WebFetch main page
         ↓
3. FETCH REVIEWS → gh api or parse review section
         ↓
4. FETCH COMMENTS → gh api or WebFetch comments page
         ↓
5. FETCH CI STATUS → WebFetch checks page
         ↓
6. FETCH CI DETAILS → WebFetch action run pages for annotations
         ↓
7. ANALYZE → Apply triage decision logic
         ↓
8. OUTPUT → Generate structured response with next steps
```

---

## Action Required Analysis - Deep Comment Thread Analysis

This section provides deep analysis of comment threads to extract action items and determine unresolved requests.

### 1. Comment Thread Structure Analysis

#### Thread Architecture

GitHub PR comments form hierarchical conversation structures:

```
PR Conversation
├── Thread N (top-level comment)
│   ├── Reply 1 (inline response to thread)
│   ├── Reply 2
│   └── Reply K
├── Thread N+1
│   └── ...
└── Thread M
    └── ...
```

#### Thread Metadata Extraction

For each thread, extract:

```python
thread_analysis = {
    "thread_id": "discussion_$uuid",
    "original_comment": {
        "author": "username",
        "author_type": "maintainer|contributor|bot|automated",
        "timestamp": "ISO-8601",
        "line_ref": "line numbers or null for PR-level",
        "content_summary": "first 200 chars",
        "request_type": "clarification|code_change|approval|deprecation Drama",
    },
    "reply_chain": [
        {
            "author": "username",
            "timestamp": "ISO-8601",
            "is_author_response": True/False,
            "content_summary": "first 200 chars"
        }
    ],
    "thread_state": "open|resolved|hiden|outdated",
    "is_action_required": True/False,
    "blocking_level": "HIGH|MEDIUM|LOW|NONE",
    "action_request": "specific ask if present",
    "awaiting_response_from": "username or null"
}
```

### 2. Reply History Deep Analysis

#### Tracing Conversation Flow

Analyze reply chains to determine conversation health:

```python
def analyze_reply_chain(thread):
    """
    Deep analysis of reply chain to identify:
    1. Open requests awaiting author response
    2. Follow-up questions from reviewers
    3. Action item confirmation status
    """
    chain_analysis = {
        "total_replies": len(thread.replies),
        "author_responded": False,
        "pending_asks": [],
        "resolved_items": []
    }
    
    for idx, reply in enumerate(thread.replies):
        # First message is original comment
        if idx == 0:
            original_ask = extract_action_request(reply)
            if original_ask:
                chain_analysis["pending_asks"].append(original_ask)
        
        # Check if author has responded to original ask
        if is_author_response(reply):
            chain_analysis["author_responded"] = True
            # Mark previously pending items as potentially resolved
            mark_for_resolution_check(original_ask)
        
        # Identify new asks in follow-up messages
        if has_new_request(reply):
            new_ask = extract_action_request(reply)
            chain_analysis["pending_asks"].append(new_ask)
    
    return chain_analysis
```

#### Response Pattern Recognition

| Pattern | Description | Implication |
|---------|-------------|-------------|
| Author acknowledged with no change | `@author Thanks, good idea` | Request may be closed or pending |
| Author responded with code change | New commit pushed referencing comment | Action item completed |
| Author responded asking for clarification | `Can you explain what you mean?` | Conversation in progress |
| No author response after N days | Thread open with replies | Action item pending |
| Maintainer rider: `@author Please check` | Explicit follow-up request | High priority pending |

### 3. Action Item Classification Framework

#### Classification Hierarchy

```
Action Item
├── BLOCKING (requires immediate response)
│   ├── Code Change Request (from maintainer)
│   ├── Bug Report (breaks functionality)
│   ├── Security Concern
│   └── Blocker Label Applied
├── REQUIRES RESPONSE (author must acknowledge)
│   ├── Clarification Request
│   ├── Design Question
│   └── Alternative Approach Suggestion
├── OPTIONAL (author discretion)
│   ├── Documentation Improvement
│   ├── Code Style Suggestion
│   └── NIT/Minor Improvements
└── INFORMATIONAL (awareness only)
    ├── Automated Review Suggestions
    ├── Bot Comments (CI результатты)
    └── Performance Notes
```

#### Action Item Extraction Algorithm

```python
def classify_comment_action(comment):
    """
    Deep classification of comment into action types.
    Returns structured action item with blocking level.
    """
    result = {
        "has_action": False,
        "action_type": "NONE",
        "blocking_level": "NONE",
        "urgency": "LOW",
        "requires_code_change": False,
        "requires_documentation": False,
        "suggestion_content": None
    }
    
    # Check source classification first
    if is_automated_bot_comment(comment):
        # Copilot AI, GitHub Actions bot, etc. - typically informational
        if suggests_technical_improvement(comment):
            result["action_type"] = "OPTIONAL_SUGGESTION"
            result["blocking_level"] = "NONE"
            result["suggestion_content"] = comment.body
        return result
    
    # Human review - analyze content
    body = comment.body.lower()
    
    # Identify blocking patterns
    blocking_patterns = [
        "this breaks", "blocks merge", "cannot merge",
        "must fix", "critical", "security issue",
        "changes required"
    ]
    
    for pattern in blocking_patterns:
        if pattern in body:
            result["has_action"] = True
            result["action_type"] = "BLOCKING_CODE_CHANGE"
            result["blocking_level"] = "HIGH"
            result["requires_code_change"] = True
            return result
    
    # Identify response-required patterns
    response_patterns = [
        "please check", "please address", "can you clarify",
        "@author", "waiting for", "needs discussion"
    ]
    
    for pattern in response_patterns:
        if pattern in body:
            result["has_action"] = True
            result["action_type"] = "HUMAN_REVIEW_FEEDBACK"
            result["blocking_level"] = "MEDIUM"
            return result
    
    # Identify documentation requests
    doc_patterns = [
        "please add comment", "document", "could we add some comments here"
    ]
    
    for pattern in doc_patterns:
        if pattern in body:
            result["has_action"] = True
            result["action_type"] = "DOCUMENTATION_REQUEST"
            result["blocking_level"] = "OPTIONAL"
            result["requires_documentation"] = True
            return result
    
    # Suggestion patterns
    suggestion_patterns = [
        "consider", "might want to", "could be improved",
        "may want to", "nit:", "optional"
    ]
    
    for pattern in suggestion_patterns:
        if pattern in body:
            result["has_action"] = True
            result["action_type"] = "CODE_SUGGESTION"
            result["blocking_level"] = "LOW"
            result["suggestion_content"] = comment.body
            return result
    
    return result
```

### 4. Author Response Tracking

#### Response Timeline Analysis

Construct author response timeline to identify delays:

```python
def build_response_timeline(pr):
    """
    Build chronological timeline of:
    1. Author's commits
    2. Reviewer comments requiring action
    3. Author responses
    4. Time-to-response metrics
    """
    timeline = []
    
    # Add all comments with timestamps
    for comment in pr.comments:
        timeline.append({
            "type": "comment",
            "author": comment.user.login,
            "timestamp": comment.created_at,
            "needs_response": classify_needs_response(comment)
        })
    
    # Add commits
    for commit in pr.commits:
        timeline.append({
            "type": "commit",
            "sha": commit.sha,
            "timestamp": commit.commit.author.date,
            "responds_to": find_related_comments(commit)
        })
    
    # Sort by timestamp
    timeline.sort(key=lambda x: x["timestamp"])
    
    return timeline
```

#### Pending Response Detection

```python
def detect_pending_responses(pr, timeline):
    """
    Identify comments where author
expects response but hasn't received one.
    """
    pending = []
    cut_off_date = datetime.now() - timedelta(days=7)
    
    for comment in pr.comments:
        # Check if comment is from reviewer (not author)
        if comment.user.login == pr.author.login:
            continue
        
        # Check if comment asks for response
        if not asks_for_response(comment):
            continue
        
        # Check if author responded
        if author_responded_to(comment, pr):
            continue
        
        # Check if comment is recent or stale
        comment_date = parse_date(comment.created_at)
        
        pending_item = {
            "comment_id": comment.id,
            "author": comment.user.login,
            "timestamp": comment.created_at,
            "days_pending": (datetime.now() - comment_date).days,
            "request_summary": truncate(comment.body, 200),
            "urgency": "HIGH" if comment_date < cut_off_date else "MEDIUM"
        }
        
        pending.append(pending_item)
    
    return pending
```

### 5. Thread State Evaluation

#### State Machine Model

```
Thread State Transitions:

[OPEN] ──────────────────────────────────────────────────────┐
   │                                                           │
   ├─ Author responds with fix ──→ [RESOLVED]                  │
   │                                                           │
   ├─ Author acknowledges but no change ──→ [ACKNOWLEDGED] ──→ wait for more
   │                                                           │
   ├─ Maintainer marks resolved ──→ [RESOLVED]                 │
   │                                                           │
   ├─ Thread goes stale (7+ days) ──→ [STALE_OPEN] ──────────→ needs attention
   │                                                           │
   └─ Code changes on relevant lines ──→ [ZOMBIE_THREAD] ─────→ may need cleanup
                                                               │
[OUTDATED] ────────────────────────────────────────────────────┘
   │
   └─ New code pushed addressing old comment ──→ [RESOLVED]

[HIDDEN] ── Author or maintainer hides ──→ [CULLED]
```

#### State Evaluation Criteria

```python
def evaluate_thread_state(thread):
    """
    Deep evaluation of thread state considering:
    1. Comment age
    2. Author response presence
    3. Related code changes
    4. Re-review status
    """
    evaluation = {
        "state": "OPEN",
        "needs_attention": False,
        "blocking_merge": False,
        "reason": None
    }
    
    # Check if outdated
    if thread.position and is_line_outdated(thread.position):
        if not related_code_changed(thread):
            evaluation["state"] = "OUTDATED"
            evaluation["reason"] = "Comment on deleted/changed line"
        else:
            evaluation["state"] = "RESOLVED"
            return evaluation
    
    # Check if author responded
    if has_author_response(thread):
        evaluation["state"] = "ACKNOWLEDGED"
    
    # Check for maintainer resolution
    if thread.position and has_maintainer_resolution(thread):
        evaluation["state"] = "RESOLVED"
    
    # Determine blocking status
    if evaluation["state"] == "OPEN":
        if requires_immediate_action(thread):
            evaluation["blocking_merge"] = True
            evaluation["needs_attention"] = True
    
    return evaluation
```

### 6. Action Item Prioritization Matrix

#### Priority Classification

| Priority | Criteria | SLA |
|----------|----------|-----|
| P0 - Critical | Blocker, breaks CI, security issue | 1 day |
| P1 - High | Code change required by maintainer | 3 days |
| P2 - Medium | Documentation request from maintainer | 7 days |
| P3 - Low | Optional suggestions, NITs | Next iteration |

#### Triage Action Matrix

| Status | Human Comment | Bot Comment | Action |
|--------|--------------|-------------|--------|
| Approval + comment with request | Maintainer review | Copilot AI | Need to address request |
| Approval + informational comment | LGTM, good job | Nitpick | No action required |
| Request changes with blocking comment | Bug report | - | Fix required |
| Comment with no action keyword | General feedback | Performance note | Awareness only |

### 7. Response Template Generation

Generate appropriate response templates based on comment type:

```python
def generate_response_template(comment):
    """
    Generate response template for author to use
    when addressing comments.
    """
    action_type = classify_action_type(comment)
    
    if action_type == "DOCUMENTATION_REQUEST":
        return """**@{author}** - Thanks for the suggestion.

I'll add clarifying comments explaining the XPU-specific behavior:
- Reference to XPU native packing contract
- Distinction from CUDA implementation
- Documentation for future upstreaming

[class Add this as follow-up commit]
"""
    
    elif action_type == "CODE_REFACTOR_SUGGESTION":
        return """**@{author}** - Thanks for the technical insight.

Regarding moving assertions outside `@torch.compile`:
- This is noted as an optimization opportunity
- Current implementation maintains test validity
- Will consider for future refinement

[Either implement suggestion or provide justification]
"""
    
    elif action_type == "CLARIFICATION_QUEST":
        return """**@{author}** - Sure, here's clarification on {topic}:

[Provide detailed explanation]

Does this address your question?
"""
    
    elif action_type == "BLOCKING_ISSUE":
        return """**@{author}** - Thanks for catching this.

I'll address this {issue_type} immediately:
- Root cause analysis: {cause}
- Fix approach: {approach}
- ETA for resolution: {timeframe}

[class Push fix in next commit]
"""
    
    return None  # No template for informational comments
```

### 8. Deep Analysis Output Schema

#### Complete Response Structure

```markdown
## Action Items Analysis - PR #{number}

### Executive Summary
- Total Open Threads: {N}
- Blocking Items: {M}
- Awaiting Author Response: {K}

### Blockers Requiring Immediate Action

#### 1. [Blocker Title]
- **Source**: {reviewer_name} | {timestamp}
- **Type**: Blocking Code Change | Documentation Request | etc.
- **Location**: Lines {start}-{end} in {file}
- **Request**: {specific ask}
- **Suggested Resolution**: {approach}
- **Priority**: P0/P1/P2/P3

### Non-Blocking Suggestions

#### 1. [Suggestion Title]
- **Source**: Copilot AI | Stonepia | etc.
- **Type**: Optional Improvement | Documentation
- **Blocking Level**: None
- **Recommendation**: {action_or_ignore}

### Reply Chain Analysis

#### Thread: discussion_{id}
```
├── {Author1} [{date}]: {original_request}
│   └── {Author2} [{date}]: {follow_up}
│       └── [NO AUTHOR RESPONSE - PENDING]
```

#### Pending Author Responses
| Comment ID | Reviewer | Days Pending | Request Summary |
|------------|----------|---------------|------------------|
| {id1} | {name} | {N} days | {truncated request} |

### Recommended Response Actions

1. **Immediate (Within 24h)**:
   - [Action 1]
   - [Action 2]

2. **This Week**:
   - [Optional improvements]
   - [Documentation additions]

### Updated Triage Status

| Gate | Status | Blockers | Priority |
|------|--------|----------|----------|
| Waiting for resolving | YES | {count} | P{level} |
| Waiting for review | NO | - | - |
| Waiting for CI | YES/NO | {details} | - |
| Waiting for merge | YES/NO | - | - |
```

---

### 9. Special Comment Type Handling

#### Handle Follow-up Tag Patterns

GitHub uses @mention patterns for follow-ups:

```python
def parse_followup_patterns(body):
    """
    Identify follow-up request patterns in comments:
    - @author Please check
    - @author Please address
    - @author Can you clarify
    """
    patterns = [
        r'@(\w+)\s+please\s+check',
        r'@(\w+)\s+please\s+address',
        r'@(\w+)\s+can\s+you\s+clarify',
        r'@(\w+)\s+look\s+at',
    ]
    
    followups = []
    for pattern in patterns:
        matches = re.finditer(pattern, body, re.IGNORECASE)
        for match in matches:
            followups.append({
                "mentioned_user": match.group(1),
                "pattern_type": extract_pattern_type(pattern),
                "position": match.span()
            })
    
    return followups
```

#### Handle Rope-Attached Comments

Comments can be:
- **Direct**: On specific lines/files
- **Dangling**: On deleted/renamed lines (marked Outdated)
- **PR-level**: General PR discussion

Each requires different analysis approach:

```python
def classify_comment_location(comment):
    if comment.path and comment.line:
        return {
            "type": "LINE_COMMENT",
            "file": comment.path,
            "line": comment.line,
            "state": "ACTIVE" if not is_outdated(comment) else "OUTDATED"
        }
    elif comment.path and not comment.line:
        return {
            "type": "FILE_COMMENT",
            "file": comment.path,
            "state": "ACTIVE"
        }
    else:
        return {
            "type": "PR_COMMENT",
            "state": "ACTIVE"
        }
```

---

## Author Notes

This skill uses deep analysis rather than surface pattern matching. It interprets comment context, distinguishes automated vs human reviews, and provides nuanced triage recommendations considering repository-specific policies.

Key philosophical approach:
- Context matters more than keywords
- Source of comment determines blocking status
- CI infrastructure warnings are distinct from test failures
- Human judgment remains primary for edge cases
- Thread continuity and reply history provide deeper signal than isolated comments

Advanced analysis principles:
1. **Pull-th comma matters**: Review comment implies conversation; discrete comments may link via threading
2. **Stale threads signal neglect**: Unresolved old threads often indicate forgotten items
3. **Bot suggestions are noise floor**: Automated reviews rarely constitute merge blockers
4. **Maintainer endorsement weight**: Human review with approval carries more authority than suggestion-only

---

## Constraints

- **Rate Limits**: Respect GitHub API limits; use WebFetch as fallback
- **Authentication**: gh CLI must be authenticated for API; fallback to WebFetch
- **Timeout**: WebFetch max 120 seconds
- **Merge Blockers**: Only actual blocking items considered blockers
- **Privacy**: Never exposeprivate conversation details in summaries