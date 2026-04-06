# Issue Triage Pipeline

## Overview

This pipeline processes PyTorch XPU issues through 4 steps to collect, extract, and analyze issue data and test results.

## Pipeline Steps

### Step 1: Issue Collection
**Location**: `torch-xpu-ops-issue-collection/`

Collects GitHub issues from intel/torch-xpu-ops repository and extracts:
- Issue basic info (ID, Title, Status, Assignee, Labels, etc.)
- Test cases from issue body
- PR references that fix the issue

**PR Extraction Logic**:
- Only extracts PRs from fix context (e.g., "PR: https://...", "PR #1234", "fixed in PR")
- For intel/torch-xpu-ops PRs: Skips if "Closed with unmerged commits"
- For pytorch/pytorch PRs: Only includes if has "Merged" label

**Scripts**:
- `generate_excel.py` - Main script to collect issues, test cases, and PRs

**When to use**: 
- When starting fresh to collect issues from GitHub
- When need to update issue list from GitHub API

**Usage**:
```bash
cd ~/ai_for_validation/opencode/issue_triage/torch-xpu-ops-issue-collection
python3 generate_excel.py
```

**Output**: `~/issue_traige/data/torch_xpu_ops_issues.xlsx`

---

### Step 2: Torch Ops Extraction
**Location**: `torch-ops-extraction/`

Extracts torch operation information from issue test cases to classify issues by operation type.

**Scripts**:
- `extract_torch_ops.py` - Analyzes test cases and extracts torch operations

**When to use**:
- After Step 1 to classify issues by operation
- When need to add operation classification to issues

**Usage**:
```bash
cd ~/ai_for_validation/opencode/issue_triage/torch-ops-extraction
python3 extract_torch_ops.py ~/issue_traige/data/torch_xpu_ops_issues.xlsx
```

**Input**: Excel file from Step 1

---

### Step 3: PR Extraction
**Location**: `pr-extraction/`

Extracts PR references from GitHub issue comments with fix-related keywords and fetches PR status.

**Scripts**:
- `pr_extraction.py` - Main script to extract PRs from comments

**When to use**:
- After Step 1 or Step 2 to link issues to fixing PRs
- When need to know if issues are fixed by merged PRs

**Usage**:
```bash
cd ~/ai_for_validation/opencode/issue_triage/pr-extraction
python3 pr_extraction.py ~/issue_traige/data/torch_xpu_ops_issues.xlsx
```

**Output**: Updates PR, PR Owner, PR Status columns in Issues sheet

---

### Step 4: Test Results Update
**Location**: `update_test_results/`

Gets test results from CI artifacts (torch-xpu-ops nightly and stock PyTorch XPU CI) and analyzes case existence.

**Scripts**:
- `update_test_results.py` - Main script to update test results

**When to use**:
- When need to know test pass/fail status for issue test cases
- When need to explain why tests are not found

**Usage**:
```bash
cd ~/ai_for_validation/opencode/issue_triage/update_test_results
python3 update_test_results.py
```

**Output**: Updates status columns (torch-xpu-ops nightly, stock CI, case existence)

---

## Excel File Structure

Input/Output: `~/issue_traige/data/torch_xpu_ops_issues.xlsx`

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
# Step 1: Collect issues
cd ~/ai_for_validation/opencode/issue_triage/torch-xpu-ops-issue-collection
python3 generate_excel.py

# Step 2: Extract torch ops
cd ~/ai_for_validation/opencode/issue_triage/torch-ops-extraction
python3 extract_torch_ops.py ~/issue_traige/data/torch_xpu_ops_issues.xlsx

# Step 3: Extract PRs
cd ~/ai_for_validation/opencode/issue_triage/pr-extraction
python3 pr_extraction.py ~/issue_traige/data/torch_xpu_ops_issues.xlsx

# Step 4: Update test results
cd ~/ai_for_validation/opencode/issue_triage/update_test_results
python3 update_test_results.py
```