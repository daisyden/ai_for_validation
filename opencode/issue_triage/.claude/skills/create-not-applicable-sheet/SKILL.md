# Create Not applicable Sheet

## Overview
This skill creates a "Not applicable" sheet in `torch_xpu_ops_issues.xlsx` that lists all issues tagged with `wontfix` or `not_target` labels, extracting the torch ops or APIs mentioned in each issue.

## Workflow
1. Load issues from `torch_xpu_ops_issues.xlsx` (Issues sheet)
2. Filter issues with `wontfix` or `not_target` in labels
3. Extract torch ops/APIs from issue titles using regex patterns:
   - `aten::xxx` - ATen operator names
   - `torch.xxx` - PyTorch API references
   - `operator 'aten::xxx'` - quoted ATen operators
   - Specific keywords (TypedStorage, SDPA, etc.)
4. Create/update "Not applicable" sheet with extracted data

## Usage
```bash
cd /home/daisydeng/ai_for_validation/opencode/issue_triage
python3 create_not_applicable_sheet.py
```

## Input
- `/home/daisydeng/ai_for_validation/opencode/issue_triage/result/torch_xpu_ops_issues.xlsx` (Issues sheet)

## Output
- Adds/updates "Not applicable" sheet with columns:
  - Issue ID
  - Title
  - Torch Ops/API
  - Labels
  - Reason

## Script: create_not_applicable_sheet.py
```python
import openpyxl
import re

def create_not_applicable_sheet(
    excel_path: str = '/home/daisydeng/ai_for_validation/opencode/issue_triage/result/torch_xpu_ops_issues.xlsx',
    output_path: str = None
):
    """Create Not applicable sheet from issues with wontfix/not_target labels"""
    wb = openpyxl.load_workbook(excel_path)
    ws = wb['Issues']
    
    not_appliable = []
    
    for row in ws.iter_rows(min_row=2, values_only=True):
        issue_id = row[0]
        title = row[1] if row[1] else ''
        labels = row[5] if row[5] else ''
        
        # Check if issue is wontfix or not_target
        if 'wontfix' in labels.lower() or 'not_target' in labels.lower():
            torch_ops = extract_torch_ops(title)
            not_appliable.append({
                'issue_id': issue_id,
                'title': title,
                'torch_ops': torch_ops,
                'labels': labels,
                'reason': 'wontfix/not_target'
            })
    
    # Create/update Not applicable sheet
    if 'Not applicable' in wb.sheetnames:
        del wb['Not applicable']
    
    ws_na = wb.create_sheet('Not applicable')
    ws_na.append(['Issue ID', 'Title', 'Torch Ops/API', 'Labels', 'Reason'])
    
    for item in not_appliable:
        ws_na.append([
            item['issue_id'],
            item['title'],
            item['torch_ops'],
            item['labels'],
            item['reason']
        ])
    
    save_path = output_path or excel_path
    wb.save(save_path)
    
    return len(not_appliable)

def extract_torch_ops(title: str) -> str:
    """Extract torch ops/API references from issue title"""
    ops = []
    
    # ATen operator patterns
    aten_matches = re.findall(r"aten::[\w_.]+", title)
    ops.extend(aten_matches)
    
    # Quoted ATen operators
    quoted_aten = re.findall(r"'aten::[\w_]+'", title)
    ops.extend([m.strip("'") for m in quoted_aten])
    
    # PyTorch API patterns
    torch_matches = re.findall(r'torch\.[\w.]+(?:\.[\w.]+)*', title)
    for m in torch_matches:
        if 'torch' in m and len(m) < 30 and not m.startswith('torch._'):
            ops.append(m)
    
    # Specific known deprecated/unsupported features
    if 'TypedStorage' in title or 'TypedTensors' in title:
        ops.append('TypedStorage/TypedTensors')
    if 'cuda_monkeypatch' in title:
        ops.append('cuda_monkeypatch (CUDA specific)')
    if 'scaled_dot_product_attention' in title.lower():
        ops.append('scaled_dot_product_attention')
    
    # Remove duplicates while preserving order
    unique_ops = list(dict.fromkeys(ops))
    
    return ', '.join(unique_ops) if unique_ops else title[:60].strip()
```

## Example Output
| Issue ID | Torch Ops/API | Reason |
|----------|---------------|--------|
| 3133 | scaled_dot_product_attention | wontfix/not_target |
| 2508 | TypedStorage/TypedTensors | wontfix |
| 2472 | aten::_cudnn_rnn | wontfix |