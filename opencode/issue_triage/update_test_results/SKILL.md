# Update Test Results

## Overview
This skill updates the torch_xpu_ops_issues.xlsx with CI test results from XML files and performs case existence analysis using LLM.

## When to Use
- After create_issues_excel skill generates the initial Excel file
- When need to add CI test status and case existence info

## Workflow Steps

### Step 1: Load XML Files
- **torch-xpu-ops nightly**: `/home/daisydeng/issue_traige/ci_results/torch-xpu-ops/`
  - Pattern: `Inductor-XPU-UT-Data-*-op_ut-*-1/op_ut_with_*.xml`
- **Stock PyTorch CI**: `/home/daisydeng/issue_traige/ci_results/stock/`
  - Pattern: `test-reports-*.zip/test-reports/python-pytest/`

### Step 2: Update CI Status Columns (11-17)
For each test case in Excel:
- **Column 11**: status in torch-xpu-ops nightly (passed/failed/skipped/not found)
- **Column 12**: comments in torch-xpu-ops nightly
- **Column 13**: Commit
- **Column 14**: Run_id
- **Column 15**: XML filename
- **Column 16**: status in stock CI (passed/failed/skipped/not found)
- **Column 17**: comments in stock CI

### Step 3: Case Existence Analysis (Columns 18-20)
Only for first N unique issues with "not found" status (MAX_LLM_CASES=3):
- **Column 18**: cuda_case_exist (Yes/No)
- **Column 19**: xpu_case_exist (Yes/No)
- **Column 20**: case_existence_comments

LLM uses:
- check-cuda-test-existence skill
- check-xpu-test-existence skill

Analysis includes:
- Base test name
- CUDA/XPU file paths
- Decorators (@onlyCUDA, @onlyXPU, @skipCUDAIfNoHipdnn, etc.)
- Why XPU test doesn't exist (skipped, parametrized, removed, etc.)

### Step 4: Blue Highlighting
Mark LLM-analyzed rows with blue background (light blue: ADD8E6)

## Usage

```bash
cd ~/ai_for_validation/opencode/issue_triage/update_test_results
python3 update_test_results.py
```

## Output
- Updated `$RESULT_DIR/torch_xpu_ops_issues.xlsx` (default: `~/ai_for_validation/opencode/issue_triage/result/`)

## Key Parameters
- MAX_LLM_CASES = 3 (unique issues with double "not found")
- LLM timeout = 240 seconds
- Model = opencode/gpt-5-nano (faster free model)

## Related Skills
- create_issues_excel: Generate initial Excel file
- check-xpu-test-existence: Check XPU test existence
- check-cuda-test-existence: Check CUDA test existence