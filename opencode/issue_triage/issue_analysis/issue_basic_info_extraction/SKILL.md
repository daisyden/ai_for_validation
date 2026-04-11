# Issue Basic Info Extraction

## Overview
This skill creates the torch_xpu_ops_issues.xlsx Excel file by collecting open issues from intel/torch-xpu-ops GitHub repository and extracting test case information.

## When to Use
- When need to generate a fresh Excel file with all open issues from torch-xpu-ops
- When issues need to be re-collected from GitHub

## What It Does

### Step 1: Fetch Issues from GitHub
The script automatically fetches issues if JSON files don't exist:
- API: `https://api.github.com/repos/intel/torch-xpu-ops/issues?state=open&per_page=100`
- Filters out pull requests
- Saves to: `/home/daisydeng/issue_traige/data/torch_xpu_ops_issues.json`

### Step 2: Parse Issue Data
Extract fields:
- **Basic Info**: Issue ID, Title, Status, Assignee, Reporter, Labels, Created/Updated Time, Milestone
- **Classification**: Type (bug/feature/performance), Module (distributed/inductor/autograd/etc), Test Module (ut/e2e/build)

### Step 3: Parse Test Cases
Parse test cases from issue body in these formats:
- **Format 1**: `op_ut,third_party.torch-xpu-ops.test.xpu.test_nn_xpu.TestNNDeviceTypeXPU,test_case_name`
  - Skip cases wrapped with `~~` (fixed issues)
  - Known test types: op_ut, op_extend, op_extended, e2e, benchmark, ut, test_xpu
- **Format 2**: `python benchmarks/dynamo/huggingface.py ...` (e2e tests)
- **Format 3**: pytest code blocks with test paths

Extract:
- Test Type
- Test File: `torch-xpu-ops/test/xpu/test_xxx_xpu.py`
- Origin Test File: mapped to pytorch test path (`test/test_xxx.py`)
- Test Class: extracted from path (e.g., `TestNNDeviceTypeXPU`)
- Test Case: the actual test name

**Fields left blank for test_result_analysis to fill:**
- Error Message, Traceback, torch-ops, dependency

### Step 4: Parse E2E Test Cases
E2E test cases are benchmark tests from huggingface, timm, or torchbench. Model lists from:
- https://github.com/intel/torch-xpu-ops/tree/main/.ci/benchmarks

**Parse E2E Info from Issue Body:**
Extract:
1. **Benchmark**: huggingface, timm, or torchbench (identified from model name)
2. **Model**: Model name from the benchmark suite
3. **Phase**: training or inference
4. **Dtype**: bfloat16, float16, float32, int8
5. **AMP**: auto mixed precision setting
6. **Backend**: inductor or eager
7. **Test Type**: accuracy or performance
8. **Cudagraph**: yes or no
9. **Reproducer**: command to reproduce

### Step 5: Create Excel File
Three sheets:

1. **Issues**: Columns: Issue ID, Title, Status, Assignee, Reporter, Labels, Created Time, Updated Time, Milestone, Summary, Type, Module, Test Module, Dependency

2. **Test Cases**: Columns:
   - Issue ID, Test Reproducer, Test Type, Test File, Origin Test File, Test Class, Test Case
   - Error Message, Traceback, torch-ops, dependency (left blank for test_result_analysis/)

3. **E2E Test Cases**: Columns:
   - Issue ID, Test Reproducer, Benchmark, Model, Phase, Dtype, AMP, Backend, Test Type, Cudagraph
   - Error Message, Traceback (left blank for test_result_analysis/)

## Usage

```bash
cd ~/ai_for_validation/opencode/issue_triage/issue_analysis/issue_basic_info_extraction
python3 generate_excel.py
```

## Output
- `$RESULT_DIR/torch_xpu_ops_issues.xlsx` (default: `~/ai_for_validation/opencode/issue_triage/result/`)

## Prerequisites
- GitHub token with repo access (set GITHUB_TOKEN env var)
- Python with: openpyxl, requests, json

## Related Skills
- issue_analysis/pr-extraction: Extract PR references from issue comments (separate script)
- test_result_analysis/Test_Cases: Add CI results, Error Message, Traceback, torch-ops, dependency
- test_result_analysis/E2E_Test_Cases: Add E2E Error Message, Traceback
- test_result_analysis/check-xpu-test-existence: Check if XPU test exists
- test_result_analysis/check-cuda-test-existence: Check if CUDA test exists