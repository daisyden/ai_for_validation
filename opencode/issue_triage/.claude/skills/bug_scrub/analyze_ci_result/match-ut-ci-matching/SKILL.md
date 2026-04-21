# PASS 1: CI Result Matching

## Base Path Reference

Relative paths from this file location (`bug_scrub/analyze_ci_result/match-ut-ci-matching/`):
```
../../../                    → issue_triage root
../../../ci_results/        → CI artifacts directory
../../../result/            → Excel results directory
```

## Overview
Collect and match CI test results for stock PyTorch and torch-xpu-ops into test_cases_all.xlsx.

## Workflow
1. Create test_cases_all.xlsx with 'stock' and 'torch-xpu-ops' sheets
2. Collect stock CI test cases from PyTorch repository
3. Collect torch-xpu-ops CI test cases from third_party/torch-xpu-ops
4. Match test cases between stock and xpu CI results

## Usage
```bash
cd ../../../
python3 run_processor_steps.py --steps 1
```

## Input
- Stock CI test files from PyTorch repo (`test/`)
- XPU CI test files from torch-xpu-ops (`third_party/torch-xpu-ops/test/xpu/`)

## Output
- `test_cases_all.xlsx` with stock and torch-xpu-ops sheets
- 'Test Cases' sheet in torch_xpu_ops_issues.xlsx updated with CI matching info

## Related Files
- pass1_ci_matcher.py
- pass1_ci_matching.py