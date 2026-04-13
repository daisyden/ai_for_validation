# Match E2E Test Status from Inductor Reports

## Overview
This skill matches E2E test cases in torch_xpu_ops_issues.xlsx to accuracy status from Inductor_E2E_Test_Report.xlsx files.

## Input Parameters
- **Excel File**: Path to torch_xpu_ops_issues.xlsx (default: `/home/daisydeng/ai_for_validation/opencode/issue_triage/result/torch_xpu_ops_issues.xlsx`)
- **Base Dir**: Directory containing E2E report folders with Inductor_E2E_Test_Report.xlsx (default: `/home/daisydeng/issue_traige/ci_results/torch-xpu-ops/`)

## Usage
```bash
python run_match_e2e_status.py --excel EXCEL_FILE --base-dir BASE_DIR
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
Column 13 "XPU Accuracy Status" populated with values:
- `pass`, `pass_due_to_skip`, `fail`, `accuracy_mismatch`, etc.
- `Status not found` if no matching entry exists
- `E2E report not found` if reports cannot be loaded

## Known Limitations
- Corrupt timm report files will result in unmatched entries
- Model naming mismatches may occur (e.g., `hf_Roberta_base` vs `RobertaForCausalLM`)
- Unknown benchmarks cannot be matched