# Bug Scrub Workflow

## Overview
Comprehensive workflow for triaging torch-xpu-ops issues through 5 phases,
collecting AR (Action Required) data for each issue with deep analysis.

**Incremental by default**: when re-running on an existing Excel, each phase
skips rows that already have completed analysis columns. See
[Incremental Mode](#incremental-mode-skip-already-completed-work) for the
full skip-rule table.

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

## Incremental Mode (Skip Already-Completed Work)

When re-running the pipeline on an existing `torch_xpu_ops_issues.xlsx`,
each phase MUST check for prior results and **skip rows that are already
populated**. This avoids duplicated work and preserves manually-curated
values.

### Skip Rules by Phase

| Phase | Column(s) to Check | Skip Condition |
|-------|-------------------|----------------|
| 2.3 case-duplication-detection | `duplicate_group_id` | If the cell is non-blank, skip duplicate detection for this row. Do NOT force a full case-duplication rerun in incremental mode. |
| 2.4 check_xpu_case_existence | `xpu_case_existence` | If the cell is non-blank (True or False already set), skip this case entirely. Do NOT re-run the deep analysis. |
| 3.3 triage_skills | `Category`, `Priority`, `Root Cause`, `Fix Approach` | If **all four** columns are non-blank for an issue, skip triage for that issue. If any of the four is blank, re-run triage for that issue and fill only the missing columns (preserve existing non-blank values). `Dependency` is optional and must not be used as a completion gate because not all issues have one. |
| 4a–4c (all Phase 4) | — | **NEVER skip.** Phase 4 always re-runs for every issue because PR status, CI results, and comment activity change frequently. Stale AR verdicts are worse than re-computation cost. |

### How to Detect "Already Done"

```python
import openpyxl

wb = openpyxl.load_workbook("result/torch_xpu_ops_issues.xlsx")

# Phase 2.3: check Test Cases sheet for duplicate detection
tc_sheet = wb["Test Cases"]
for row in tc_sheet.iter_rows(min_row=2):
    duplicate_group_id = row[col_index("duplicate_group_id")].value
    if duplicate_group_id is not None:
        # SKIP 2.3 - duplicate detection already checked
        continue

# Phase 2.4: check Test Cases sheet for case existence
for row in tc_sheet.iter_rows(min_row=2):
    xpu_case_existence = row[col_index("xpu_case_existence")].value
    if xpu_case_existence is not None:
        # SKIP 2.4 - case existence already checked
        continue

# Phase 3.3: check Issues sheet
issues_sheet = wb["Issues"]
for row in issues_sheet.iter_rows(min_row=2):
    category = row[col_index("Category")].value
    priority = row[col_index("Priority")].value
    root_cause = row[col_index("Root Cause")].value
    fix_approach = row[col_index("Fix Approach")].value
    if all(v is not None and str(v).strip() for v in
           [category, priority, root_cause, fix_approach]):
        # SKIP - already triaged
        continue
```

### Execution Logic with Incremental Checks

Each phase's "For each Issue/Case" loop MUST prepend the skip check:

```
For each Issue/Case:
    IF already_done(row, phase):    ← NEW: incremental check
        LOG "Skipping Issue #{id} for Phase X.Y - already completed"
        CONTINUE
    ... original logic ...
```

### Notes

- Phase 1 (Prepare Data) always runs fully — it fetches fresh data from
  GitHub and CI. New issues are appended; existing issues are updated with
  fresh metadata (labels, status, comments) but analysis columns are
  preserved.
- Phase 2.1–2.2 (match-ut, match-e2e) always re-run because CI results may
  have changed. Phase 2.3 case-duplication can skip rows with an existing
  `duplicate_group_id` in incremental mode.
- Phase 5 (Generate Report) always re-runs — it is purely presentational.
- When in doubt, **preserve existing values**. Never overwrite a non-blank
  analysis column unless the user explicitly requests a full re-run.

---

## Phase 1: Prepare Data

### 1.1 Issue Basic Info Extraction
- **Skill**: `prepare_data/issue-basic-info-extraction/`
- **Steps**:
  1. Fetch open issues from GitHub REST API → `data/torch_xpu_ops_issues.json` (skip with `SKIP_PHASE_1_1=1`)
  2. Fetch 5 PyTorchXPU project fields per issue via a single `gh api graphql` request
  3. **Script-first** extraction: `parse_test_cases_from_body` (matches `Cases:` blocks etc.) and `parse_e2e_info` (matches structured benchmark commands)
  4. **LLM fallback**: only consulted for issues the script extractors did not match (prose-only references, embedded scripts, URL-bearing reproducers). Cached by body hash; never overrides a successful script match.
  5. Build sheets via routing decision tree (UNITTEST → E2E → Others)
  6. Post-pass: align `Issues.Test Module` to actual placement
- **Routing**: Each issue is placed in **exactly one** of Test Cases / E2E Test Cases / Others / Not applicable. The `Issues.Test Module` column is rewritten in the final post-pass to equal the sheet the issue is in. See the skill's SKILL.md for the full decision tree, schema, and invariants.
- **Output**: `result/torch_xpu_ops_issues.xlsx` with five sheets:
  - **Issues** (19 columns): basic info + Type/Module/Test Module/Dependency/Priority + `PyTorchXPU Status` / `PyTorchXPU Estimate` / `PyTorchXPU Depending` / `PyTorchXPU Short Comments` (cols 16–19)
  - **Test Cases**
  - **E2E Test Cases**
  - **Others**: issues whose body has a reproducer/error but no enumerable test (columns: ID, Title, Labels, reproduce step, Error Message, Traceback). URLs in reproducer are preserved verbatim.
  - **Not applicable** (owned by Phase 1.3, not by this step)
- **Priority initialization**: If `PyTorchXPU Priority` is non-blank and matches `P0`/`P1`/`P2`/`P3`, set Excel `Priority` to that value. The other four PyTorchXPU fields are written verbatim (sanitized for Excel illegal chars, truncated to 32767 chars).

### 1.2 Download CI Result
- **Skill**: `prepare_data/download_ci_result/`
- **Steps**: Download artifacts from torch-xpu-ops + stock pytorch CI
- **Output**: CI artifacts in `ci_results/`

### 1.3 Create Not Applicable Sheet
- **Skill**: `prepare_data/create-not-applicable-sheet/`
- **Default mode (carry-forward)**: Copy the `Not applicable` sheet verbatim from the latest trusted backup `result/torch_xpu_ops_issues_bk_*.xlsx`. No re-analysis.
- **Deep-analysis mode**: Used for **new** wontfix/not_target issues only. Spawn one Explore Agent per issue to determine root-cause category (CUDA-specific / hardware / deprecated / etc.) and the operation/API involved. Append the row to the sheet.
- **Output**: `Not applicable` sheet in Excel with columns Issue ID, Title, Operation/API, Category, Technical Details, Workaround, Labels.
- **Note**: This skill never touches Issues / Test Cases / E2E Test Cases / Others.

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
        Determine Root Cause type
        Propose Fix Approach
        Identify Dependency components from the confirmed root cause and fix approach
        Assign Priority based on severity unless Priority is already populated
        from the GitHub Projects `PyTorchXPU Priority` field
    
    Set Category = "<analysis result>"
    If existing Priority is non-blank:
        Preserve Priority = "<existing labeled priority>"
    Else:
        Set Priority = "<P0/P1/P2/P3>"
    Set Root Cause = "<root cause type>"
    Set Fix Approach = "<recommended approach>"
    Set Dependency = "<component>"
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

## Phase 5: Generate Report

- **Skill**: `collect_AR/generate_report/`
- **Trigger**: After Phase 4c, once the Issues sheet is stable.

Classifies each row's free-text `action_TBD` into a 17-category
`action_Type` column, then renders the human-readable
`result/bug_scrub.md` grouped by action_Type with per-section
back-to-index anchors and a Duplicates column.

Phase 5 is **purely presentational** — it does not call `gh` or rewrite
verdict columns. PR-state correctness is owned by Phase 4b (Vector E +
Step 2.5 live re-check in `get_AR_from_issue/`). If a row reaches Phase
5 with the wrong verb, fix it in Phase 4b and re-run; do not patch in
Phase 5.

**Execution Order**:
```
run_action_type.py                # populate action_Type
        ↓
gen_bug_scrub_md.py               # render result/bug_scrub.md
```

| Output | Description |
|---|---|
| `action_Type` column | 17-leaf taxonomy (`+`-joined in priority order) |
| `result/bug_scrub.md` | Section-per-category report |

See `collect_AR/generate_report/SKILL.md` for the full taxonomy and
script details.

---

## Phase 5b: Generate HTML Report (optional, on demand)

- **Skill**: `collect_AR/generate_html_report/`
- **Trigger**: After Phase 4c, when an interactive triage console is wanted.

Wraps Phase 5 — re-runs `gen_bug_scrub_md.py` internally so the HTML
always reflects the current workbook, then converts the markdown to a
single self-contained `result/bug_scrub.html` with:

- Per-row "Done" checkbox in §3 Action required and §4 QA tables;
  checked-state persists in browser `localStorage` (per-browser, not
  shared, not embedded in the file).
- Sticky filter bar with five dropdowns (Assignee, Owner Transferred,
  Priority, Category, Dependency), free-text search, and a "Hide Done"
  toggle. Filters apply across all sections and are AND-combined.
- "Export Done IDs" button — copies the comma-separated list of
  done-checked issue IDs to clipboard.

Phase 5b is purely presentational and never touches the workbook.
`bug_scrub.html` is regenerated on demand and not committed by default —
`bug_scrub.md` remains the canonical, diffable artifact.

**Execution Order**:
```
gen_bug_scrub_html.py
    ├── (calls) gen_bug_scrub_md.py     # refresh result/bug_scrub.md
    └── parse markdown → render HTML    # emit result/bug_scrub.html
```

| Output | Description |
|---|---|
| `result/bug_scrub.html` | Self-contained interactive report (CSS/JS inlined) |

See `collect_AR/generate_html_report/SKILL.md` for filter mapping,
markdown subset supported, and customization points.

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
v4.3 - May 17, 2026 - Phase 1.1 reworked to a **script-first, LLM-fallback** pipeline: script extractors (`parse_test_cases_from_body`, `parse_e2e_info`) run first and are authoritative when they match; LLM extraction (parallel sub-agents emitting structured JSON, cached by body hash) is consulted only for issues the scripts could not handle. Added filesystem-verified test paths and a post-pass that aligns `Issues.Test Module` to the actual sheet placement. Routing decision tree (UNITTEST → E2E → Others) now guarantees exactly-one-sheet placement and zero overlap. URLs in reproducers are preserved verbatim. Phase 1.3 split into carry-forward (default) and deep-analysis (new wontfix issues only) modes.
v4.2 - May 14, 2026 - Phase 1.1 now extracts all 5 PyTorchXPU project fields (Priority, Status, Estimate, Depending, Short Comments) via a single GraphQL request per issue and writes the four non-Priority fields to Issues cols 16-19. Added "Others" sheet listing issues with no parseable UT or E2E test case (columns: ID, Title, Labels, reproduce step, Error Message, Traceback).
v4.1 - May 13, 2026 - Refined Incremental Mode: Phase 2.3 case-duplication can skip rows with existing duplicate_group_id, and Phase 3.3 completion no longer requires Dependency because not all issues have one.
v4.0 - May 11, 2026 - Added Incremental Mode: skip rules for Phases 2.4 and 3.3 to avoid re-processing rows that already have completed analysis columns. Phase 4 always re-runs. Preserves existing non-blank values.
v3.5 - April 27, 2026 - Added Phase 5b (`collect_AR/generate_html_report/`): on-demand interactive HTML report with per-row Done checkboxes (§3/§4, persisted in browser localStorage), sticky filter bar (Assignee / Owner Transferred / Priority / Category / Dependency + free-text + Hide Done), and "Export Done IDs" — fully self-contained, regenerated on demand from the markdown report. Phase 5 markdown remains canonical.
v3.4 - April 27, 2026 - Phase 4b: added Vector E (scan `Fix Approach` text for PR references) and Step 2.5 (mandatory live `gh pr view` re-check + replacement-PR search via Vectors C/D/E for CLOSED-only verified sets) to fix stale-snapshot and missed-PR mis-verdicts. Phase 5 remains purely presentational.
v3.3 - April 22, 2026 - Reorganized helper scripts into skill-colocated folders (`analyze_issue/get_AR_from_issue/`, `analyze_issue/triage_skills/`, `collect_AR/generate_report/`) with `__file__`-anchored paths. Added Phase 5 (generate_report) section.
v3.2 - April 21, 2026 - All paths updated to relative paths, directory renamed (case-duplication-detection)
