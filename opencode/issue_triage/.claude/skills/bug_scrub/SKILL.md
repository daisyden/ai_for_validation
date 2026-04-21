# Bug Scrub Workflow

## Overview
Comprehensive workflow for triaging torch-xpu-ops issues through 4 phases,
collecting AR (Action Required) data for each issue with deep analysis.

## Workflow Phases

```
Phase 1: Prepare Data
    ↓
Phase 2: Analyze CI Result (match-ut → match-e2e → case-duplication → check case)
    ↓
Phase 3: Analyze Issue (dup detection → triage)
    ↓
Phase 4: Collect AR (close_or_skip → get_AR_from_issue [with check_pr_status] → case_existence_check)
```

**Relative Path Note**: In SKILL.md files, relative paths from bug_scrub/ to issue_triage root:
- `../../ci_results/` → CI artifacts
- `../../result/` → Excel results
- `../../data/` → JSON data
- `../../` prefix not shown in paths below

---

## Phase 1: Prepare Data

### 1.1 Issue Basic Info Extraction
- **Skill**: `prepare_data/issue-basic-info-extraction/`
- **Steps**: Fetch open issues from GitHub API → Parse to Excel
- **Output**: `result/torch_xpu_ops_issues.xlsx` (Issues, Test Cases, E2E Test Cases sheets)

### 1.2 Download CI Result
- **Skill**: `prepare_data/download_ci_result/`
- **Steps**: Download artifacts from torch-xpu-ops + stock pytorch CI
- **Output**: CI artifacts in `ci_results/`

### 1.3 Create Not Applicable Sheet
- **Skill**: `prepare_data/create-not-applicable-sheet/`
- **Steps**: Filter wontfix/not_target labeled issues → Deep analysis of each via Explore Agent
- **Output**: "Not Applicable" sheet in Excel (Operation/API, Category, Technical Details)
- **Note**: Uses Explore Agent deep analysis instead of script/pattern matching to identify torch operations/not targeted reasons

### 1.4 PyTorch XPU Backend Analysis
- **Skill**: `prepare_data/pytorch_xpu_backend_analysis/`
- **Steps**: Deep analysis of operator implementation
- **Output**: Analysis documentation (~51KB)

---

## Phase 2: Analyze CI Result

### 2.1 Match UT CI Matching
- **Skill**: `analyze_ci_result/match-ut-ci-matching/`
- **Output**: Match UT test cases to CI results

### 2.2 Match E2E CI Matching
- **Skill**: `analyze_ci_result/match-e2e-ci-matching/`
- **Output**: Match E2E benchmark tests to CI results

### 2.3 Case Duplicate Detection
- **Skill**: `analyze_ci_result/case-duplication-detection/`
- **Output**: `duplicate_group_id` column

### 2.4 Check XPU Case Existence
- **Skill**: `analyze_ci_result/check_xpu_case_existence/`
- **Trigger**: Rows where BOTH XPU Status AND Stock Status are blank
- **Scope**: FIRST blank case per issue only

| Output Column | Type | Description |
|--------------|------|-------------|
| `xpu_case_existence` | Boolean | True = case found, False = not found |
| `case_existence_explanation` | Text | Explanation of where/how/not found |

**Execution Logic**:
```
For each Issue:
    Find Test Cases
    For each Test Case row:
        IF XPU_Status blank AND Stock_Status blank:
            Run deep analysis on this test case
            Set xpu_case_existence, case_existence_explanation
            SKIP remaining test cases for this Issue  ← FIRST ONE ONLY
            BREAK
```

---

## Phase 3: Analyze Issue

### 3.1 Duplicated Issue Detection
- **Skill**: `analyze_issue/duplicated-issue-detection/`
- **Output**: Issue duplicate groups

### 3.3 Triage Skills
- **Skill**: `analyze_issue/triage_skills/`
- **Trigger**: Each issue in Issues sheet
- **Scope**: Full deep triage for EACH issue (NO batch script - one-by-one)

| Output Column | Type | Description |
|--------------|------|-------------|
| `Category` | Text | Issue type (bug/feature/perf/API/ci/distributed) |
| `Priority` | Text | P0/P1/P2/P3 |
| `Dependency` | Text | Components (torch-xpu-ops, PyTorch core, CI, upstream) |
| `Root Cause` | Text | Technical root cause category |
| `Fix Approach` | Text | Recommended fix strategy |

**Execution Logic**:
```
For each Issue in ['Issues' sheet]:
    Run deep triage analysis:
        Analyze title, body, error logs, AR
        Classify Category using predefined patterns
        Assign Priority based on severity
        Identify Dependency components
        Determine Root Cause type
        Propose Fix Approach
    
    Set Category = "<analysis result>"
    Set Priority = "<P0/P1/P2/P3>"
    Set Dependency = "<component>"
    Set Root Cause = "<root cause type>"
    Set Fix Approach = "<recommended approach>"
```

---

## Phase 4: Collect AR

### Phase 4 Execution Order
```
4a. close_or_skip   → 4b. get_AR_from_issue (+ check_pr_status) → 4c. case_existence_check
```

---

### 4a close_or_skip

**RULE 1: Close Issue (Fixed)**
| Condition | Output |
|-----------|--------|
| All test cases of issue are fixed + double-verify cases in body |
| `action_TBD = "Close the fixed issue"` |
| `owner_transferred = reporter` |
| `action_reason = "Fixed and passed in CI"` |

**RULE 2: Skip Issue (Not Target)**
| Condition | Output |
|-----------|--------|
| Issue labeled with `not target` OR `wontfix` |
| `action_TBD = "Skip issue"` |
| `owner_transferred = reporter` |
| `action_reason = "not target feature"` |

**Decision Priority**:
```
1. Apply RULE 1 if all cases fixed
2. Apply RULE 2 if labeled not target/wontfix
3. ELSE → Proceed to 4b
```

---

### 4b get_AR_from_issue (includes check_pr_status)

- **Location**: `analyze_issue/get_AR_from_issue/`
- **Skill**: `analyze_issue/get_AR_from_issue/SKILL.md`
- **Execution**: After 4a, only if issue not closed/skipped
- **Trigger**: Each issue in Issues sheet
- **Integration**: Internally calls check_pr_status logic for PR analysis

| Output Column | Description |
|--------------|-------------|
| `action_TBD` | AR list from get_AR_from_issue (includes PR AR via check_pr_status) |
| `action_reason` | AR reasons from PR status analysis |
| `owner_transferred` | Owner list from PR status |

**Tools Used by get_AR_from_issue**:
1. `gh api` - GitHub API access for PR/comment data
2. `WebFetch` - Fallback for PR/issue pages
3. `Explore Agent` - Deep PR and comment analysis
4. `check_pr_status` logic - Integrated PR gate analysis

**Source Paths** (relative from bug_scrub/):
```
CI results: ../../ci_results/
Excel file: ../../result/torch_xpu_ops_issues.xlsx
```

---

### 4c case_existence_check

- **Location**: `collect_AR/case_existence_check/`
- **Cross-Reference**: Phase 2.4 xpu_case_existence data

**Execution Logic**:
```
For each Issue:
    For each Case with XPU+Stock blank:
        IF xpu_case_existence == False:
            Append 'check_case_existence' to action_TBD
            Append case_existence_comments to action_reason
```

---

## Phase 4 Column Summary

| Phase | Column | Description |
|-------|--------|-------------|
| 2.4 | `xpu_case_existence` | True/False if case found |
| | `case_existence_explanation` | Explanation text |
| 3.3 | `Category` | Issue category |
| | `Priority` | P0/P1/P2/P3 |
| | `Dependency` | Components |
| | `Root Cause` | Root cause type |
| | `Fix Approach` | Fix strategy |
| 4a | `action_TBD` | Close/Skip decision |
| | `action_reason` | Close/Skip reason |
| | `owner_transferred` | Reporter |
| 4b | `action_TBD` | + PR action items |
| | `action_reason` | + PR reasons |
| | `owner_transferred` | + owner info |
| 4c | `action_TBD` | + check_case_existence |
| | `action_reason` | + case_existence_comments |

---

## Documents Created During Workflow

| Document | Phase | Relative Path |
|----------|-------|---------------|
| `torch_xpu_ops_issues.xlsx` | 1.1 | `result/` |
| CI artifacts | 1.2 | `ci_results/` |
| Not Applicable sheet | 1.3 | In Excel |
| XPU Backend Analysis | 1.4 | `result/pytorch_xpu_backend_analysis.md` |
| AR documentation | 4 | Various per skill |

---

## Path Reference (Relative from bug_scrub/)

| Destination | Relative Path |
|-------------|---------------|
| CI results | `../../ci_results/` |
| Excel results | `../../result/` |
| JSON data | `../../data/` |
| Skills root | `../../.claude/skills/` |

---

## Version
v3.2 - April 21, 2026 - All paths updated to relative paths, directory renamed (case-duplication-detection)