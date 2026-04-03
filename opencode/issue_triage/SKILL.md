# Issue Triage Skill ## Description This skill helps with triaging PyTorch XPU issues by: 1. Getting test results from torch-xpu-ops nightly CI
2. Getting test results from stock PyTorch XPU CI
3. Analyzing case existence (checking if tests exist in pytorch/test and torch-xpu-ops/test/xpu)
4. Extracting PR information from GitHub issue comments

## When to Use
- When asked to analyze test case results from torch-xpu-ops nightly CI
- When asked to cross-reference with stock PyTorch XPU CI results
- When asked to explain why test cases are not found (removed, renamed, parametrized, etc.)
- When asked to extract PR information from GitHub issue comments

## Sub-skills

### 1. Update Test Results (`update_test_results.py`)
See separate documentation in that folder for details.

### 2. PR Extraction (`pr-extraction/`)
See separate documentation in pr-extraction/SKILL.md for details.

## Files and Locations

### Input Files
- `~/issue_traige/data/torch_xpu_ops_issues.xlsx` - Excel file with test cases
- `~/issue_traige/ci_results/torch-xpu-ops/` - torch-xpu-ops nightly CI artifacts
- `~/issue_traige/ci_results/stock/` - stock PyTorch XPU CI artifacts
- `~/pytorch/test/` - PyTorch test files
- `~/pytorch/third_party/torch-xpu-ops/test/xpu/` - torch-xpu-ops test files

### Output
- Updated `~/issue_traige/data/torch_xpu_ops_issues.xlsx` with columns:
  - K: status in torch-xpu-ops nightly
  - L: comments in torch-xpu-ops nightly
  - M: Commit
  - N: Run_id
  - O: XML file name
  - P: status in stock CI
  - Q: comments in stock CI
  - R: cuda_case_exist (Yes/No)
  - S: xpu_case_exist (Yes/No)
  - T: case_existence_comments

## Usage

### Run the script
```bash
python3 ~/ai_for_validation/opencode/issue_triage/update_test_results.py
```

### Key Concepts

1. **torch-xpu-ops Nightly CI**: Tests run from torch-xpu-ops with XPUPatchForImport pattern
   - XML files in: `Inductor-XPU-UT-Data-<commit>-op_ut-<run_id>-1/`
   - File pattern: `op_ut_with_all.*.xml` or `op_ut_with_skip.*.xml`

2. **Stock PyTorch XPU CI**: Tests run directly from pytorch/test
   - XML files in: `test-reports-*.zip/test-reports/python-pytest/`

3. **Case Existence Analysis**:
   - CUDA file: `~/pytorch/test/<module>.py`
   - XPU file: `~/pytorch/third_party/torch-xpu-ops/test/xpu/<module>_xpu.py`
   - XPUPatchForImport: Some tests use this pattern to import from pytorch/test with XPU patches
   - Parametrized tests: Some tests are parameterized with device type (cuda, xpu, cpu)

4. **Common "Not Found" Reasons**:
   - Test possibly removed/renamed in CUDA
   - Uses XPUPatchForImport (runs via patch)
   - XPU file missing (not yet created in torch-xpu-ops)
   - Inductor tests not in torch-xpu-ops (use stock CI)
   - Needs _xpu suffix (parameterization)
