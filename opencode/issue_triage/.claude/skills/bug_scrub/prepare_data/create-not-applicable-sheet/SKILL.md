# Create Not Appliable Sheet

## Base Path Reference

Relative paths from this file location (`bug_scrub/prepare_data/create-not-applicable-sheet/`):
```
../../../                    → issue_triage root
../../../result/            → Excel results directory
```

## Overview
This skill creates a "Not Appliable" sheet in `torch_xpu_ops_issues.xlsx` that lists all issues tagged with `wontfix` or `not_target` labels, extracting the torch ops or APIs mentioned in each issue.

## Workflow
1. Load issues from `../../../result/torch_xpu_ops_issues.xlsx` (Issues sheet)
2. Filter issues with `wontfix` or `not_target` in labels
3. Extract torch ops/APIs from issue titles using regex patterns:
   - `aten::xxx` - ATen operator names
   - `torch.xxx` - PyTorch API references
   - `operator 'aten::xxx'` - quoted ATen operators
   - Specific keywords (TypedStorage, SDPA, etc.)
4. Create/update "Not Appliable" sheet with extracted data

## Usage
```bash
cd ../../../
python3 create_not_applicable_sheet.py
```

## Input
- `../../../result/torch_xpu_ops_issues.xlsx` (Issues sheet)

## Output
- Adds/updates "Not Appliable" sheet with columns:
  - Issue ID
  - Title
  - Torch Ops/API
  - Labels
  - Reason

## Example Output
| Issue ID | Torch Ops/API | Reason |
|----------|---------------|--------|
| 3133 | scaled_dot_product_attention | wontfix/not_target |
| 2508 | TypedStorage/TypedTensors | wontfix |
| 2472 | aten::_cudnn_rnn | wontfix |