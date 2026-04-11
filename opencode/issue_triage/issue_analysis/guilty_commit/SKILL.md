# Skill: Guilty Commit Detection for PyTorch/XPU Issues

This skill helps find the guilty commit that caused test failures in intel/torch-xpu-ops repo by analyzing pytorch/pytorch and intel/torch-xpu-ops git history.

## Input Sources

1. **Issue ID** - from intel/torch-xpu-ops or pytorch/pytorch GitHub
2. **Excel file** - from `~/ai_for_validation/opencode/issue_triage/update_test_results`
3. **Manual input** - test file, test name, test cases, error message, traceback

## Test Type Detection

The script automatically detects test type based on test file and test name:
- **eager**: test_ops_xpu.py, test_core.py, etc. (default)
- **inductor**: test_inductor.py, compile tests
- **dynamo**: test_dynamo.py
- **functorch**: test_functorch.py, vmap, vjp tests

## Filtering Logic

### For Eager Mode Tests (e.g., test_ops_xpu.py)
- **Include**: Core operator changes in `aten/src/ATen/`, `torch/testing/`, `test/*_ops.py`
- **Exclude**: Purely inductor/dynamo changes (`torch/_inductor/`, `torch/_dynamo/decompositions.py`, `test/dynamo/`, `test/inductor/`)

### For Inductor/Dynamo Tests
- Include all relevant commits (no filtering)

## Workflow

### Step 1: Extract Information
- Parse issue content to extract: test file, test name, test cases, error message, traceback
- If Excel file provided, extract the failed test cases
- If manual input, parse the provided information

### Step 2: Determine Commit Range
Option A: Use last known good commit vs current commit
Option B: Use 3 days before issue submit time to submit time

### Step 3: Search in pytorch/pytorch Repo
For the determined commit range, check:
1. Test case definitions and decorators
2. Related torch operation (e.g., addcmul, conv2d, etc.)
3. Updates on op_db (OpInfo)
4. **Filter** based on test type (exclude inductor/dynamo for eager tests)

### Step 4: Search in intel/torch-xpu-ops Repo
If no relevant commit found in pytorch/pytorch, check:
1. torch-xpu-ops git log for the period
2. Related test file changes
3. Operator implementation changes

## Usage

```bash
python ~/ai_for_validation/opencode/issue_triage/issue_analysis/guilty_commit/find_guilty_commit.py \
  --issue-id "2640" \
  --repo "torch-xpu-ops" \
  --output-dir ~/ai_for_validation/opencode/issue_triage/issue_analysis/guilty_commit/output
```

Or with manual input:
```bash
python ~/ai_for_validation/opencode/issue_triage/issue_analysis/guilty_commit/find_guilty_commit.py \
  --test-file "test_ops_xpu.py" \
  --test-name "test_compare_cpu_addcmul" \
  --error-message "AssertionError: Tensor-likes are not close" \
  --days-before 30
```

## Output

- Generated report: `guilty_commit_report.md` with:
  - Test information and detected test type
  - Filter explanation based on test type
  - List of relevant pytorch/pytorch commits
  - List of relevant intel/torch-xpu-ops commits
  - Detailed analysis of filtered commits with file paths and categorization