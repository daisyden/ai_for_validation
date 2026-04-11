# Issue Triage Pipeline Summary

## Overview

The issue_triage pipeline processes torch-xpu-ops GitHub issues, extracts test cases, PRs, CI results, and generates analysis reports.

## Pipeline Flow

```
1. generate_excel.py
   └─> torch_xpu_ops_issues.xlsx (Issues, Test Cases, E2E Test Cases)

2. update_test_results.py
   └─> Adds CI results, case existence analysis, owner_transfer, action_TBD
   └─> generate_report.py
       └─> issue_report.md
```

## Input Data

- **GitHub Issues**: Fetched from intel/torch-xpu-ops repository
- **CI Results**: 
  - torch-xpu-ops nightly: `/home/daisydeng/issue_traige/ci_results/torch-xpu-ops/`
  - Stock PyTorch XPU CI: `/home/daisydeng/issue_traige/ci_results/stock/`
- **E2E Reports**: Inductor_E2E_Test_Report.xlsx in CI artifacts
- **PyTorch Test Files**: `~/pytorch/test/` and `~/pytorch/third_party/torch-xpu-ops/test/xpu/`

## Output Files

| File | Description |
|------|-------------|
| `torch_xpu_ops_issues.xlsx` | Main Excel with 3 sheets |
| `issue_report.md` | Markdown summary report |

## Excel Sheets

### 1. Issues Sheet (21 columns)
- Basic: Issue ID, Title, Status, Assignee, Reporter, Labels, Created/Updated Time, Milestone
- Classification: Type, Module, Test Module, Dependency
- PR Info: PR, PR Owner, PR Status, PR Description
- Analysis: owner_transfer, action_TBD, duplicated_issue

### 2. Test Cases Sheet (21 columns)
- Basic: Issue ID, Test Reproducer, Test Type, Test File, Origin Test File, Test Class, Test Case
- Error: Error Message, Traceback, torch-ops
- CI Results: status/comments in torch-xpu-ops nightly & stock CI, Commit, Run_id, XML
- Analysis: cuda_case_exist, xpu_case_exist, case_existence_comments, duplicated_issue

### 3. E2E Test Cases Sheet (13 columns)
- Basic: Issue ID, Test Reproducer, Benchmark, Model, Phase, Dtype, AMP
- Test Info: Backend, Test Type, Cudagraph
- Error: Error Message, Traceback
- CI: torch-xpu-ops nightly status - accuracy

## Key Features

### 1. Issue Classification
- **Type**: functionality bug, performance issue, feature request, etc.
- **Module**: distributed, inductor, dynamo, aten_ops, AO, low_precision, etc.
- **Test Module**: ut, e2e, build
- **Dependency**: oneAPI, oneDNN, oneMKL, Triton, driver, transformers, AO, oneCCL

### 2. PR Extraction
- Only extracts PRs that fix issues (not all mentioned PRs)
- Filters out closed unmerged PRs for intel/torch-xpu-ops
- For pytorch/pytorch, only includes merged PRs

### 3. CI Test Results
- Matches test cases with XML results from:
  - torch-xpu-ops nightly CI
  - Stock PyTorch XPU CI
- Extracts pass/fail/skipped status

### 4. Case Existence Analysis
- Uses check-cuda-test-existence and check-xpu-test-existence skills
- Analyzes why tests don't exist:
  - Skip decorators (@onlyCUDA, @skipCUDAIfNoHipdnn, etc.)
  - Parameterization from base tests
  - Test removal/renaming

### 5. Duplicated Issue Detection
- Same Test Class + Test Case in different issues
- Similar Traceback (excluding "Tensor-likes are not close!")

### 6. Owner Transfer & Action TBD

| Condition | owner_transfer | action_TBD |
|-----------|----------------|------------|
| All tests passed | reporter | Close fixed issue |
| All E2E tests passed | reporter | Close fixed issue |
| PR closed, no failed tests | reporter | Verify the issue |
| PR closed, has failed tests | assignee | Revisit the PR as case failed |
| Labels: wontfix/not target | assignee | add to skiplist |
| Labels: not target + ut_upstream | assignee | Needs Upstream Skip PR |
| Error: "Torch not compiled..." | daisyden | Enable test |
| Random label, no test failure | - | - |
| ut_upstream label | assignee | Needs PyTorch Repo Changes |
| Flash attention/SDPA related | assignee | Flash Attention / Transformer Related |
| Sparse ops related | assignee | Sparse Operations Related |
| Inductor/compile related | assignee | Inductor / Compilation Related |
| Dtype/precision issues | assignee | Dtype / Precision Related |

### 7. E2E Accuracy Status
- Matches E2E test cases with accuracy results from Inductor_E2E_Test_Report.xlsx
- Supports: huggingface, timm_models, torchbench
- Extracts per-model status: pass, fail_to_run, pass_due_to_skip

### 8. Report Generation

**Statistics:**
- By Test Module
- By Module
- By Dependency
- By Action TBD
- Other Stats (Not Assigned, Duplicated, Others)

**Issue Lists:**
- Action Required (split by action_TBD type)
- Issues without Assignee
- Duplicated Issues
- Issues with Dependency
- Others

## Scripts

| Script | Location | Description |
|--------|----------|-------------|
| generate_excel.py | issue_basic_info_extraction/ | Scrapes GitHub issues, generates Excel |
| update_test_results.py | update_test_results/ | Adds CI results, case analysis |
| generate_report.py | update_test_results/ | Generates markdown report |
| check-xpu-test-existence | check-xpu-test-existence/ | Skill for XPU test existence |
| check-cuda-test-existence | check-cuda-test-existence/ | Skill for CUDA test existence |

## Usage

```bash
# Step 1: Generate Excel
cd ~/ai_for_validation/opencode/issue_triage/issue_basic_info_extraction
python3 generate_excel.py

# Step 2: Update with CI results and generate report
cd ~/ai_for_validation/opencode/issue_triage/update_test_results
python3 update_test_results.py
```

## Notes

- LLM analysis is limited to MAX_LLM_CASES=3 unique issues (faster processing)
- Model used: opencode/gpt-5-nano for faster LLM analysis
- Dependency detection requires explicit context (e.g., "caused by", "depends on")
- Module classification checks labels first, then keywords
