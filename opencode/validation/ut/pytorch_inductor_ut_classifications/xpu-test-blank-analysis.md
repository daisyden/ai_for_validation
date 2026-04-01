# XPU Test Blank Status Analysis Skill

## Description
Analyzes PyTorch Inductor UT status Excel files to identify test cases where status_xpu is blank and Reason is blank, determines if CUDA/XPU test cases exist in the test files, and categorizes them appropriately.

## Usage
This skill is triggered when user wants to:
- Analyze test cases with blank status_xpu and Reason in Excel files
- Check if CUDA/XPU parameterized test cases exist in pytorch/test
- Determine why tests are not running on XPU (removed vs skipped vs needs investigation)
- Update Excel with cuda_case_exists, xpu_case_exists, Comments, Reason, and DetailReason

## Steps

### 1. Git Pull Repositories
```bash
cd ~/pytorch && git pull
cd ~/pytorch/third_party/torch-xpu-ops && git pull
```

### 2. Read Excel File
- File: `/home/daisydeng/classify_inductor/Inductor_ut_status_ww14_26.xlsx`
- Sheet: 'Cuda pass xpu skip'
- Find rows where `status_xpu` is NaN/empty AND `Reason` is NaN/empty

### 3. Analyze Test Cases

For each blank row, analyze:
1. **testfile_cuda**: The test file path (e.g., `test/inductor/test_compile_subprocess.py`)
2. **name_cuda**: The CUDA test name (e.g., `test_low_memory_max_pool_dilation_1_dim_2_cuda`)
3. **classname_cuda**: The test class name

#### Determine test existence:
```python
import subprocess
import re

def check_test_exists(testfile, name_cuda):
    # Extract base test name
    name_base = re.sub(r'_cuda(_.*)?$', '', name_cuda)
    
    testfile_path = f"/home/daisydeng/pytorch/{testfile}"
    
    # Check if test file exists
    file_check = subprocess.run(['ls', testfile_path], capture_output=True)
    if file_check.returncode != 0:
        return False, False
    
    # Check for CUDA test
    result = subprocess.run(
        ['grep', '-E', f'def {name_base}\\(', testfile_path],
        capture_output=True, text=True
    )
    cuda_exists = bool(result.stdout.strip())
    
    # Check for XPU support in test file
    result_xpu = subprocess.run(
        ['grep', '-iE', 'xpu|skipIfXpu', testfile_path],
        capture_output=True, text=True
    )
    xpu_exists = bool(result_xpu.stdout.strip())
    
    return cuda_exists, xpu_exists
```

### 4. Categorize Based on Results

#### Case 1: Both cuda_case_exists=False and xpu_case_exists=False
- **Comment**: "Test removed - parametrization changed in slow_tests.json update"
- **Reason**: "Not Applicable"
- **DetailReason**: "Community change"
- This happens when:
  - Test was removed from test files
  - Test parametrization changed (e.g., new parameters added like `use_block_ptr`)
  - Test moved to slow_tests.json

#### Case 2: cuda_case_exists=True, xpu_case_exists=False
- Check for skip decorators in test file
- If `@skipIfXpu` decorator found:
  - **Comment**: "Test has @skipIfXpu decorator"
  - **Reason**: "To be enabled"
  - **DetailReason**: "Daisy"
- Otherwise:
  - **Comment**: "Human investigation - test exists but needs verification"
  - **Reason**: "To be enabled"  
  - **DetailReason**: "Human investigation"

#### Case 3: xpu_case_exists=True
- **Comment**: "Human investigation - test exists and has XPU support"
- **Reason**: "To be enabled"
- **DetailReason**: "Human investigation"

### 5. Human Investigation Re-analysis (Post-processing)

**IMPORTANT**: After initial classification, Human Investigation cases need further verification:

1. Check if the test name in Excel matches the actual parametrization in test files
2. If test file has different parametrization (e.g., `test_low_memory_max_pool(dilation, dim, use_block_ptr)`), explain the expected test case name in DetailReason
3. Mark Reason as "Not applicable" with detailed explanation

```python
def analyze_human_investigation(row):
    name_xpu = row['name_xpu']
    testfile = row['testfile_cuda']
    
    # Check for parametrization changes
    if 'low_memory_max_pool' in name_xpu:
        # Extract dilation and dim from test name
        match_dilation = name_xpu.split('_dilation_')[1].split('_')[0] if '_dilation_' in name_xpu else ""
        match_dim = name_xpu.split('_dim_')[1].split('_')[0] if '_dim_' in name_xpu else ""
        
        # Check if test exists in file with new parametrization
        result = subprocess.run(
            ['grep', '-E', 'def test_low_memory_max_pool', f'/home/daisydeng/pytorch/{testfile}'],
            capture_output=True, text=True
        )
        
        if 'dilation' in result.stdout and 'use_block_ptr' in result.stdout:
            # New parametrization exists
            expected = f"test_low_memory_max_pool(dilation={match_dilation}, dim={match_dim}, use_block_ptr=True/False)"
            return "Not applicable", expected
    
    # Check other test patterns similarly...
    
    return "Not applicable", "Test not found - test removed or parametrization changed"
```

### 6. Update Excel

```python
import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import PatternFill

excel_path = '/home/daisydeng/classify_inductor/Inductor_ut_status_ww14_26.xlsx'
wb = load_workbook(excel_path)
ws = wb['Cuda pass xpu skip']

# Add new columns if not exist
header_row = [cell.value for cell in ws[1]]
current_cols = len(header_row)

ws.cell(1, current_cols + 1, 'cuda_case_exists')
ws.cell(1, current_cols + 2, 'xpu_case_exists')
ws.cell(1, current_cols + 3, 'Comments')

# Get column indices
cuda_col = current_cols + 1
xpu_col = current_cols + 2
comment_col = current_cols + 3
reason_col = header_row.index('Reason') + 1
detail_col = header_row.index('DetailReason') + 1

# Blue fill for updated cells
blue_fill = PatternFill(start_color='B4C7E7', end_color='B4C7E7', fill_type='solid')

# Update rows
for r in results:
    idx = r['idx']
    excel_row = idx + 2
    
    ws.cell(excel_row, cuda_col, r['cuda_exists'])
    ws.cell(excel_row, cuda_col).fill = blue_fill
    
    ws.cell(excel_row, xpu_col, r['xpu_exists'])
    ws.cell(excel_row, xpu_col).fill = blue_fill
    
    ws.cell(excel_row, comment_col, r['comment'])
    ws.cell(excel_row, comment_col).fill = blue_fill
    
    ws.cell(excel_row, reason_col, r['reason'])
    ws.cell(excel_row, reason_col).fill = blue_fill
    
    ws.cell(excel_row, detail_col, r['detail_reason'])
    ws.cell(excel_row, detail_col).fill = blue_fill

wb.save(excel_path)
```

### 7. Git Log Investigation (for removed tests)

To find the commit that removed/changed tests:
```bash
cd ~/pytorch

# Search for specific test name in git history
git log --all --oneline -S "test_low_memory_max_pool_dilation" | head -10

# Check slow_tests.json updates
git log --oneline --all --since="2026-02-01" -- test/slow_tests.json | head -5

# Check commit that changed parametrization
git show <commit_id> -p -- test/inductor/test_torchinductor.py
```

## Common Patterns

### Tests removed due to parametrization changes
- New `@parametrize` decorators added (e.g., `use_block_ptr`)
- Test names change from `test_xxx_dim_2` to `test_xxx(dilation=X, dim=Y, use_block_ptr=Z)`
- These are tracked in `slow_tests.json` which gets updated periodically
- **DetailReason should show**: Expected test name with new parametrization

### Tests with XPU skip decorators
- `@skipIfXpu(msg="...")` - explicitly skips XPU
- `@skipIfNotXpu` - skips if not XPU
- Check test files for these decorators
- **Reason**: "To be enabled", **DetailReason**: "Daisy"

### Human investigation cases
- Test file has XPU support but specific test case doesn't run
- May need to add XPU device to parameterized test
- May need to remove XPU skip decorator
- **IMPORTANT**: Verify actual test name and parametrization, then mark as "Not applicable" with explanation

## Requirements
- pandas
- openpyxl
- access to ~/pytorch/test directory

## Output
- Updated Excel file with columns:
  - `cuda_case_exists`: Boolean
  - `xpu_case_exists`: Boolean
  - `Comments`: Explanation string
  - `Reason`: "Not Applicable" or "To be enabled"
  - `DetailReason`: "Community change", "Daisy", or explanation with expected test name
- All updated cells marked with blue color (B4C7E7)
- Human Investigation cases verified and marked "Not applicable" with parametrization explanation