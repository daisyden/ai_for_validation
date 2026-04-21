# Match E2E Test Status from Inductor Reports

## Base Path Reference

Relative paths from this file location (`bug_scrub/analyze_ci_result/match-e2e-ci-matching/`):
```
../../../                    → issue_triage root
../../../ci_results/        → CI artifacts directory
../../../result/            → Excel results directory
```

## Overview
This skill matches E2E test cases in torch_xpu_ops_issues.xlsx to accuracy status from Inductor_E2E_Test_Report.xlsx files.

## Input Parameters
- **Excel File**: `../../../result/torch_xpu_ops_issues.xlsx`
- **Base Dir**: `../../ci_results/torch-xpu-ops/`
- E2E report folders within base dir contain Inductor_E2E_Test_Report.xlsx

## Usage
```bash
python run_match_e2e_status.py --excel ../../../result/torch_xpu_ops_issues.xlsx --base-dir ../../ci_results/
```

## Workflow
1. Load E2E Test Cases sheet from torch_xpu_ops_issues.xlsx
2. Parse Inductor_E2E_Test_Report.xlsx files from report directories
3. Extract benchmark (huggingface/timm/torchbench), dtype, phase (inf/tra), AMP mode
4. Build status map with model name variants for fuzzy matching
5. Match each E2E test case to accuracy status with fallback logic

## Key Features
- **Model name fuzzy matching**: Handles variants like `AllenaiLongformerBase` ↔ `allenailongformerbase`
- **Dtype fallback**: bfloat16 → float32 → float16
- **AMP fallback**: AMP enabled → AMP disabled
- **Phase handling**: inference (inf), training (tra)

## Reports Format
Inductor_E2E_Test_Report.xlsx sheets named as:
- `<benchmark>_<dtype>_<inf|tra>_acc` (e.g., `huggingface_float32_inf_acc`)
- `<benchmark>_<dtype>_<inf|tra>_amp_acc` (e.g., `huggingface_amp_bf16_tra_acc`)

## Output
Column "XPU Accuracy Status" populated with values:
- `pass`, `pass_due_to_skip`, `fail`, `accuracy_mismatch`, etc.
- `Status not found` if no matching entry exists
- `E2E report not found` if reports cannot be loaded

## Known Limitations
- Corrupt timm report files will result in unmatched entries
- Model naming mismatches may occur (e.g., `hf_Roberta_base` vs `RobertaForCausalLM`)
- Unknown benchmarks cannot be matched