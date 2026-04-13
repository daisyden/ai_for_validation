# PASS 3: LLM Verification for Test Existence

## Overview
Verify whether test cases exist in CI for both CUDA and XPU backends using LLM analysis.

## Workflow
1. Load test case data from torch_xpu_ops_issues.xlsx
2. For each case, query LLM with test file path and test case name
3. Analyze LLM response for:
   - Test file existence in PyTorch repo
   - Presence of test class and method
   - XPU/CUDA compatibility (decorators)
   - Skip conditions
4. Populate 'Exist in CI?' and 'XPU Enabled' columns

## Usage
```bash
cd /home/daisydeng/ai_for_validation/opencode/issue_triage/test_result_analysis/Test_Cases
python run_processor_steps.py --steps 3
```

## LLM Model
- Qwen3-32B (default)
- Takes ~20s per test case
- ~26 min for 68 test cases

## Key Checks
- `@onlyCUDA` - Not available on XPU
- `@onlyXPU` - XPU specific
- `@skipCUDAIfNoHipdnn` - ROCm only
- `@requires_xccl` - Requires Intel MPI
- `XPUPatchForImport` usage patterns

## Input
- Test Cases sheet with Test Reproducer info
- CI data from test_cases_all.xlsx

## Output
- Column 9 "Exist in CI?"
- Column 10 "XPU Enabled"
- Column 11 "Comment"