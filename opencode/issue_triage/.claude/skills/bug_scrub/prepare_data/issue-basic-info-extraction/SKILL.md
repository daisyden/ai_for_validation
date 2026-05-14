# Issue Basic Info Extraction

## Base Path Reference

Relative paths from this file location (`bug_scrub/prepare_data/issue-basic-info-extraction/`):
```
../../../                    → issue_triage root
../../../data/              → JSON data directory
../../../result/            → Excel results directory
..                          → WORKDIR (SCRIPT_DIR here)
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
- **PyTorchXPU Project Fields**: Fetch all 5 fields from the `PyTorchXPU` GitHub Project in a single GraphQL request per issue:
  - `PyTorchXPU Priority` — normalized to `P0`/`P1`/`P2`/`P3` and written to the Excel `Priority` column
  - `PyTorchXPU Status` — raw text written to Excel column 16
  - `PyTorchXPU Estimate` — raw value (string/number) written to column 17
  - `PyTorchXPU Depending` — raw text written to column 18
  - `PyTorchXPU Short Comments` — raw text written to column 19 (sanitized for Excel illegal chars, truncated to 32767)
  - Fields are matched by literal prefixed name (`"PyTorchXPU Status"`) OR by `(project_title == "PyTorchXPU", field_name == "Status")` to handle either naming convention in the project.

### Step 3: Parse Test Cases
Parse from issue body in formats:
- **Format 1**: `op_ut,third_party.torch-xpu-ops.test.xpu.test_nn_xpu.TestNNDeviceTypeXPU,test_case_name`
- **Format 2**: `python benchmarks/dynamo/huggingface.py ...` (e2e tests)
- **Format 3**: pytest code blocks with test paths

### Step 4: Parse E2E Test Cases
Extract: Benchmark (huggingface/timm/torchbench), Model, Phase (training/inference), Dtype, AMP, Backend, Test Type, Cudagraph

### Step 5: Create Excel File
Four sheets:
1. **Issues**: Issue ID, Title, Status, Assignee, Labels, Type, Module, Test Module, Dependency, Priority, PyTorchXPU Status, PyTorchXPU Estimate, PyTorchXPU Depending, PyTorchXPU Short Comments
2. **Test Cases**: Issue ID, Test Reproducer, Test Type, Test File, Origin Test File, Test Class, Test Case
3. **E2E Test Cases**: Issue ID, Test Reproducer, Benchmark, Model, Phase, Dtype, AMP, Backend, Test Type, Cudagraph
4. **Others**: ID, Title, Labels, reproduce step, Error Message, Traceback — issues where NO unit test case AND NO E2E test case could be parsed from the issue body. Reproduce step reuses the existing E2E reproducer extractor; Error Message and Traceback are extracted from the issue body via regex.

**Note**: After this step, use `create-not-applicable-sheet` skill (Step 1.3) to add "Not applicable" sheet for `wontfix`/`not_target` issues.

## Usage
```bash
cd ..
python3 generate_excel.py
```

## Output
- `../../../result/torch_xpu_ops_issues.xlsx` (Issues, Test Cases, E2E Test Cases sheets)

## Prerequisites
- GitHub token with repo + Projects access (set GITHUB_TOKEN env var) to fetch `PyTorchXPU Priority`; without it, issue data is still generated but project priority stays blank.
- Python with: openpyxl, requests, json

## Script Location
`generate_excel.py` - Main script resides in the same folder as this SKILL.md

## Run Examples

### Fresh extraction (clears and re-fetches from GitHub)
```bash
cd ..
rm -f ../../../data/torch_xpu_ops_issues.json  # Optional: clear cache
python3 generate_excel.py
```

### Extract specific issues
```bash
cd ..
python3 generate_excel.py --issues "3306,3305,3300"
```

## Expected Output
```
Fetching issues from GitHub...
Fetched 45 issues...
...
Fetched 375 issues...
Processed 49 issues...
...
Total issues: 375
Total test case rows: 1931
Total e2e case rows: 72

Saved to ../../../result/torch_xpu_ops_issues.xlsx
```

## Next Step
After running this skill, proceed to Phase 1.3: Execute `create-not-applicable-sheet` skill to add "Not applicable" sheet with wontfix/not_target issues.
