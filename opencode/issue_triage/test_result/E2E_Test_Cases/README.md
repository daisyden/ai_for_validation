# E2E Test Cases Processor Module

## Overview

This module contains all logic for processing the 'E2E_Test_Cases' sheet in `torch_xpu_ops_issues.xlsx`.

## Usage

```python
import openpyxl
from test_result.E2E_Test_Cases.e2e_cases_processor import process_e2e_cases

wb = openpyxl.load_workbook('torch_xpu_ops_issues.xlsx')
process_e2e_cases(wb)
wb.save('torch_xpu_ops_issues.xlsx')
```

## Functions

| Function | Description |
|----------|-------------|
| `process_e2e_cases(wb)` | Main entry point - adds accuracy status column |
| `build_e2e_status_mapping(base_dir)` | Build status mapping from E2E reports |
| `parse_e2e_sheet_name(sheet_name, benchmark)` | Parse sheet name to extract dtype/amp/phase |
| `match_e2e_status(e2e_model_status, benchmark, dtype, amp, phase, model)` | Match test case to status |
| `normalize_key_value(value)` | Normalize key values for matching |

## Columns Processed

| Col | Header |
|-----|--------|
| 13 | torch-xpu-ops nightly status - accuracy |

## Input Data

E2E reports located at: `/home/daisydeng/issue_traige/ci_results/torch-xpu-ops/*E2E*/Inductor_E2E_Test_Report.xlsx`

**Supported Benchmarks:**
- `huggingface` - HuggingFace models
- `timm_models` - TIMM models
- `torchbench` - TorchBench models

**Sheet Naming Convention:**
- `{benchmark}_{dtype}_{phase}_acc` (e.g., `huggingface_float32_inf_acc`)
- `_amp_` in name indicates AMP enabled
- `inf` = inference, `tra` = training

## Processing Steps

1. Find all `Inductor_E2E_Test_Report.xlsx` files in E2E folders
2. Parse accuracy sheets (ending with `_acc`)
3. Extract: benchmark, dtype, amp, phase, model, status
4. Build mapping: `(benchmark, dtype, amp, phase, model)` -> status
5. For each row in E2E Test Cases sheet, match and write status

## Example

```python
# Input E2E report sheet: huggingface_float32_inf_acc
# Model: AlbertForMaskedLM -> Column B
# Status: pass/fail_to_run -> Column D

# In E2E Test Cases sheet:
# Row: benchmark=huggingface, model=albertbase, dtype=float32, phase=inference, amp=False
# Result: Col13 = "pass" (matching the model status from E2E report)
```