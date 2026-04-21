# Issue Basic Info Extraction

## Base Path Reference

Relative paths from this file location (`bug_scrub/prepare_data/issue-basic-info-extraction/`):
```
../../../                    → issue_triage root
../../../data/              → JSON data directory
../../../result/            → Excel results directory
```

## Overview
Creates torch_xpu_ops_issues.xlsx by collecting open issues from intel/torch-xpu-ops GitHub repository and extracting test case information.

## When to Use
- Generate a fresh Excel file with all open issues from torch-xpu-ops
- Re-collect issues from GitHub

## Workflow

### Step 1: Fetch Issues from GitHub
- API: `https://api.github.com/repos/intel/torch-xpu-ops/issues?state=open&per_page=100`
- Filters out pull requests
- Saves to: `../../../data/torch_xpu_ops_issues.json`

### Step 2: Parse Issue Data
Extract fields:
- **Basic Info**: Issue ID, Title, Status, Assignee, Reporter, Labels, Created/Updated Time, Milestone
- **Classification**: Type (bug/feature/performance), Module (distributed/inductor/autograd/etc), Test Module (ut/e2e/build)

### Step 3: Parse Test Cases
Parse from issue body in formats:
- **Format 1**: `op_ut,third_party.torch-xpu-ops.test.xpu.test_nn_xpu.TestNNDeviceTypeXPU,test_case_name`
- **Format 2**: `python benchmarks/dynamo/huggingface.py ...` (e2e tests)
- **Format 3**: pytest code blocks with test paths

### Step 4: Parse E2E Test Cases
Extract: Benchmark (huggingface/timm/torchbench), Model, Phase (training/inference), Dtype, AMP, Backend, Test Type, Cudagraph

### Step 5: Create Excel File
Three sheets:
1. **Issues**: Issue ID, Title, Status, Assignee, Labels, Type, Module, Test Module, Dependency
2. **Test Cases**: Issue ID, Test Reproducer, Test Type, Test File, Origin Test File, Test Class, Test Case
3. **E2E Test Cases**: Issue ID, Test Reproducer, Benchmark, Model, Phase, Dtype, AMP, Backend, Test Type, Cudagraph

## Usage
```bash
cd ../../../../ai_for_validation/opencode/issue_triage/issue_analysis/issue_basic_info_extraction
python3 generate_excel.py
```

## Output
- `../../../result/torch_xpu_ops_issues.xlsx` (Issues, Test Cases, E2E Test Cases sheets)

## Prerequisites
- GitHub token with repo access (set GITHUB_TOKEN env var)
- Python with: openpyxl, requests, json