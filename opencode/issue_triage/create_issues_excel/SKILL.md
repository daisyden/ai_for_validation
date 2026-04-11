# Create torch_xpu_ops_issues.xlsx

## Overview
This skill creates the torch_xpu_ops_issues.xlsx Excel file by collecting open issues from intel/torch-xpu-ops GitHub repository and extracting test case information.

## When to Use
- When need to generate a fresh Excel file with all open issues from torch-xpu-ops
- When issues need to be re-collected from GitHub

## Workflow Steps

### Step 1: Fetch Issues from GitHub
The script automatically fetches issues if JSON files don't exist:
- API: `https://api.github.com/repos/intel/torch-xpu-ops/issues?state=open&per_page=100`
- Filters out pull requests
- Saves to: `/home/daisydeng/issue_traige/data/torch_xpu_ops_issues.json`

### Step 2: Fetch Comments
For each issue, fetch associated comments:
- API: `https://api.github.com/repos/intel/torch-xpu-ops/issues/{issue_num}/comments`
- Saves to: `/home/daisydeng/issue_traige/data/torch_xpu_ops_comments.json`

### Step 3: Parse Issue Data
Extract fields:
- **Basic Info**: Issue ID, Title, Status, Assignee, Reporter, Labels, Created/Updated Time, Milestone
- **Classification**: Type (bug/feature/performance), Module (distributed/inductor/autograd/etc), Test Module (ut/e2e/build)
- **PR Extraction**: Only PRs that fix the issue
  - Extract from issue body and comments
  - Skip intel/torch-xpu-ops PRs if "Closed with unmerged commits"
  - Skip pytorch/pytorch PRs unless they have "Merged" label

### Step 4: Parse Test Cases
From issue body in format:
```
Cases:
op_ut,third_party.torch-xpu-ops.test.xpu.nn.test_convolution_xpu.TestConvolutionNNDeviceTypeXPU,test_conv2d_hipdnn_backend_selection_xpu
```

Extract:
- Test Type: op_ut, op_extend, e2e, benchmark, ut
- Test File: torch-xpu-ops/test/xpu/...
- Origin Test File: test/...
- Test Class: TestXXXDeviceTypeXPU
- Test Case: test_xxx_xpu

### Step 5: Create Excel File
Three sheets:
1. **Issues**: 418 rows with columns: Issue ID, Title, Status, Assignee, Reporter, Labels, Created Time, Updated Time, Milestone, Summary, Type, Module, Test Module, Dependency, PR, PR Owner, PR Status, PR Description

2. **Test Cases**: ~2017 rows with columns: Issue ID, Test Reproducer, Test Type, Test File, Origin Test File, Test Class, Test Case, Error Message, Traceback, torch-ops, dependency

3. **E2E Test Cases**: ~119 rows for benchmark tests (huggingface/timm/torchbench)

## Usage

```bash
cd ~/ai_for_validation/opencode/issue_triage/issue_basic_info_extraction
python3 generate_excel.py
```

## Output
- `$RESULT_DIR/torch_xpu_ops_issues.xlsx` (default: `~/ai_for_validation/opencode/issue_triage/result/`)

## Prerequisites
- GitHub token with repo access (set GITHUB_TOKEN env var)
- Python with: openpyxl, requests, json

## Related Skills
- update_test_results: Add CI test results and case existence analysis
- check-xpu-test-existence: Check if XPU test exists in torch-xpu-ops
- check-cuda-test-existence: Check if CUDA test exists in PyTorch