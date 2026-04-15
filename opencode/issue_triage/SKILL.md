# Issue Triage Pipeline

## Overview

This pipeline processes PyTorch XPU issues through 4 steps to collect, extract, and analyze issue data and test results.

## Pipeline Steps

### Step 1: Issue Collection
**Location**: `issue_analysis/issue_basic_info_extraction/`

Collects GitHub issues from intel/torch-xpu-ops repository and extracts:
- Issue basic info (ID, Title, Status, Assignee, Labels, etc.)
- Test cases from issue body

**Note**: PR extraction is handled separately by Step 3 (issue_analysis/pr-extraction/).

**Scripts**:
- `generate_excel.py` - Main script to collect issues and test cases

**When to use**:
- When starting fresh to collect issues from GitHub
- When need to update issue list from GitHub API

**Usage**:
```bash
# All issues (default)
cd ~/ai_for_validation/opencode/issue_triage/issue_analysis/issue_basic_info_extraction
python3 generate_excel.py

# Specific issues only
python3 generate_excel.py --issues "3246,3243"
```

**Output**: `$RESULT_DIR/torch_xpu_ops_issues.xlsx` (default: `~/ai_for_validation/opencode/issue_triage/result/`)

---

### Step 2: Torch Ops Extraction
**Location**: `test_result_analysis/torch-ops-extraction/`

Extracts torch operation information from issue test cases to classify issues by operation type.

**Scripts**:
- `extract_torch_ops.py` - Analyzes test cases and extracts torch operations

**When to use**:
- After Step 1 to classify issues by operation
- When need to add operation classification to issues

**Usage**:
```bash
cd ~/ai_for_validation/opencode/issue_triage/test_result_analysis/torch-ops-extraction
python3 extract_torch_ops.py $RESULT_DIR/torch_xpu_ops_issues.xlsx
```

**Input**: Excel file from Step 1 (`$RESULT_DIR/torch_xpu_ops_issues.xlsx`)

---

### Step 3: PR Extraction
**Location**: `issue_analysis/pr-extraction/`

Extracts PR references from GitHub issue comments with fix-related keywords and fetches PR status.

**Scripts**:
- `pr_extraction.py` - Main script to extract PRs from comments

**When to use**:
- After Step 1 or Step 2 to link issues to fixing PRs
- When need to know if issues are fixed by merged PRs

**Usage**:
```bash
cd ~/ai_for_validation/opencode/issue_triage/issue_analysis/pr-extraction
python3 pr_extraction.py $RESULT_DIR/torch_xpu_ops_issues.xlsx
```

**Output**: Updates PR, PR Owner, PR Status columns in Issues sheet

---

### Step 4: Test Results Update
**Location**: `update_test_results/`

Gets test results from CI artifacts (torch-xpu-ops nightly and stock PyTorch XPU CI) and analyzes case existence.

**Scripts**:
- `update_test_results.py` - Main script to update test results
- `generate_report.py` - Generate markdown report with issue analysis

**When to use**:
- When need to know test pass/fail status for issue test cases
- When need to explain why tests are not found
- When need root cause analysis for issues with blank action_TBD

**Root Cause Analysis**:
- `analyze_root_cause()` function determines root cause based on:
  - Issue title, summary, error message, traceback
  - Test file, test class, test case information from Test Cases and E2E Test Cases sheets
  - Test module classification (ut/e2e)
- Categories: Memory, Dtype/Precision, Inductor/Compilation, DNNL, Flash Attention, Distributed, Skip, API Mismatch, etc.

**Usage**:
```bash
cd ~/ai_for_validation/opencode/issue_triage/update_test_results
python3 update_test_results.py
python3 generate_report.py
```

**Output**:
- Updates status columns (torch-xpu-ops nightly, stock CI, case existence)
- Adds Root Cause column (col 25) to Issues sheet for issues with blank action_TBD
- Generates `issue_report.md` with By Root Cause statistics

---

## Excel File Structure

Input/Output: `$RESULT_DIR/torch_xpu_ops_issues.xlsx` (default: `~/ai_for_validation/opencode/issue_triage/result/`)

### Sheets:
1. **Issues** - Main issue data with PR information
2. **Test Cases** - Test case details with CI results

### Test Cases Sheet Columns:
- A: Issue ID
- B: Test Reproducer
- C: Test Type
- D: Test File
- E: Origin Test File
- F: Test Class
- G: Test Case
- H: Error Message
- I: Traceback
- J: torch-ops
- K: status in torch-xpu-ops nightly
- L: comments in torch-xpu-ops nightly
- M: Commit
- N: Run_id
- O: XML
- P: status in stock CI
- Q: comments in stock CI
- R: cuda_case_exist
- S: xpu_case_exist
- T: case_existence_comments

---

## Key Concepts

### CI Sources:
1. **torch-xpu-ops Nightly**: Tests run from torch-xpu-ops with XPUPatchForImport
2. **Stock PyTorch XPU CI**: Tests run directly from pytorch/test

### Test File Patterns:
- **Path format**: `torch-xpu-ops/test/xpu/nn/test_convolution_xpu.py`
- **Dot notation**: `test.dynamo.test_ctx_manager.CtxManagerTests`

### XPUPatchForImport:
Some tests use this pattern to import from pytorch/test with XPU patches instead of having separate XPU files.

### Common "Not Found" Reasons:
- Test possibly removed/renamed in CUDA
- Uses XPUPatchForImport (runs via patch)
- XPU file missing (not yet created)
- Inductor tests not in torch-xpu-ops (use stock CI)
- Needs _xpu suffix (parameterization)

---

## Run Full Pipeline

```bash
# Set result directory (optional, default: ~/ai_for_validation/opencode/issue_triage/result)
export RESULT_DIR=~/ai_for_validation/opencode/issue_triage/result

# Step 1: Collect issues
cd ~/ai_for_validation/opencode/issue_triage/issue_analysis/issue_basic_info_extraction
python3 generate_excel.py

# Step 2: Extract torch ops
cd ~/ai_for_validation/opencode/issue_triage/test_result_analysis/torch-ops-extraction
python3 extract_torch_ops.py $RESULT_DIR/torch_xpu_ops_issues.xlsx

# Step 3: Extract PRs
cd ~/ai_for_validation/opencode/issue_triage/issue_analysis/pr-extraction
python3 pr_extraction.py $RESULT_DIR/torch_xpu_ops_issues.xlsx

# Step 4: Update test results (includes priority, category, root cause analysis)
cd ~/ai_for_validation/opencode/issue_triage/update_test_results
python3 update_test_results.py

# Step 5: Generate Issue Report (Action Required section)
cd ~/ai_for_validation/opencode/issue_triage/issue_analysis
python3 generate_issue_report.py
```

---

## Issue Report Structure

The `generate_issue_report.py` script creates `issue_report.md` with a structured Table of Contents:

### Section 1: Summary
Overview of issues by category

### Section 2: Action Required
Organized by assignee responsibility:

**2.1 Developer AR (Need Investigation by Action Reason)**
- Issues needing developer investigation
- Grouped by type of Action Reason extracted from `Action Reason` column
- Examples:
  - "Missing XPU kernel implementation"
  - "Test failure investigation needed"
  - "Precision/dtype fix needed"

**2.2 Reporter AR (Other Action TBD)**
- Actions for reporters/community to take
- Categories: Awaiting response, needs skiplist, E2E accuracy issue, etc.

### Sections 3-8
3. Issues by Category
4. Last Week Issues
5. Stale Issues
6. Dependency Issues
7. Duplicated Issues
8. Statistics (Action TBD, Category, Test Module distributions)

### Anchors
All sections use `span id='anchor-name'` for cross-referencing:
- `#2-action-required` - Action Required section
- `#action-required-developer` - Developer AR subsection
- `#action-required-reporter` - Reporter AR subsection
- Section-specific anchors (e.g., `#2.1-1-missing-xpu-kernel-implementation`)