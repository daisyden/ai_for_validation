# XPU Test Failure Analysis Skill

## Description
Analyzes PyTorch Inductor UT status Excel files to categorize XPU failed/skipped test cases, determines failure reasons (Feature gap vs Failure xpu broken), adds GitHub issue links, and handles closed/fixed issues.

## Usage
This skill is triggered when user wants to:
- Analyze XPU test failures in Excel status files
- Categorize failures as "Feature gap" or "Failure (xpu broken)"
- Add GitHub issue URLs to DetailReason column
- Fix truncated issue URLs to match message_xpu
- Mark closed/fixed issues as "To be enabled"

## Steps

### 1. Update Repositories
```bash
cd ~/pytorch && git pull
cd ~/pytorch/third_party/torch-xpu-ops && git pull
```

### 2. Read Excel File
- File: `~/Inductor_ut_status_ww14_26.xlsx` (or specified file)
- Sheet: 'Cuda pass xpu skip'
- Key columns: `status_xpu`, `message_xpu`, `Reason`, `DetailReason`

### 3. Categorize Failures

For each row where `status_xpu` is "failed" or "skipped", categorize based on the following rules in order:

#### 3.1 First: Handle Empty Reasons (Auto-categorize)

**Rule 1: "To be enabled" for requires CUDA/cudagraph**
- If `Reason` is empty AND `status_xpu` is "skipped" AND `message_xpu` contains "requires cuda" or "cudagraph"
- Set Reason = "To be enabled"
- Set DetailReason = "Daisy"

**Rule 2: "Not applicable" for CPU/x86 only**
- If `Reason` is empty AND `message_xpu` indicates case can only run on CPU or x86 (contains "cpu" or "x86")
- Set Reason = "Not applicable"
- Set DetailReason = summary of message_xpu (first line or first 100 chars)

**Rule 3: "Not applicable" for CPU test names**
- If `Reason` is empty AND test name (classname_xpu.name_xpu) contains "cpu" (case insensitive)
- Set Reason = "Not applicable"
- Set DetailReason = "CPU only"

#### 3.2 Then: Categorize Based on Message Patterns

**Feature gap patterns:**
- "skipIfXpu: Test for x86 backend" → Feature gap
- "Mixed-device test requires GPU" → Feature gap
- "requires multiple xpu devices" → Feature gap
- "requires CUDA" → Feature gap
- "mkldnn tensors unsupported" → Feature gap
- "_fused_rms_norm is not implemented on XPU" → Feature gap
- "cudagraph not supported" → Feature gap
- "NotImplementedError" → Feature gap (for dynamic shapes cpp wrapper)
- "autotune_at_compile_time doesn't work" → Feature gap
- "Profile not enabled on XPU CI" → Feature gap
- "tl.inline_asm_elementwise not yet supported" → Feature gap

**Failure (xpu broken) patterns:**
- "FlashAttentionForward headdim limitation" → Failure (xpu broken)
- "compile error" / "CppCompileError" → Failure (xpu broken)
- "RuntimeError" with issue reference → Failure (xpu broken)
- "AssertionError" with issue reference → Failure (xpu broken)
- "TypeError" with issue reference → Failure (xpu broken)
- "dtype is needed to compute eps1" → Failure (xpu broken)

### 4. Add GitHub Issue URLs

#### 4.1 Existing Issue Mapping (from skill doc)
```python
issue_map = {
    1981: "https://github.com/intel/torch-xpu-ops/issues/1981",  # DTensor compile RuntimeError
    2330: "https://github.com/intel/torch-xpu-ops/issues/2330",  # inline_asm_elementwise not supported
    2331: "https://github.com/intel/torch-xpu-ops/issues/2331",  # _get_exceeding_shared_memory_checker
    2334: "https://github.com/intel/torch-xpu-ops/issues/2334",  # Profile not enabled on XPU CI (CLOSED - FIXED)
    2554: "https://github.com/intel/torch-xpu-ops/issues/2554",  # triton dependency
    2609: "https://github.com/intel/torch-xpu-ops/issues/2609",  # CppCompileError for custom ops
    2697: "https://github.com/intel/torch-xpu-ops/issues/2697",  # RuntimeError
    2698: "https://github.com/intel/torch-xpu-ops/issues/2698",  # FlashAttentionForward headdim limitation
    2802: "https://github.com/intel/torch-xpu-ops/issues/2802",  # Explicit attn_mask is_causal conflict
    2853: "https://github.com/intel/torch-xpu-ops/issues/2853",  # _flash_attention_forward
    2888: "https://github.com/intel/torch-xpu-ops/issues/2888",  # float8 conversion
    2891: "https://github.com/intel/torch-xpu-ops/issues/2891",  # RuntimeError
    2958: "https://github.com/intel/torch-xpu-ops/issues/2958",  # AssertionError
    3004: "https://github.com/intel/torch-xpu-ops/issues/3004",  # TypeError
    3007: "https://github.com/intel/torch-xpu-ops/issues/3007",  # AssertionError
    3187: "https://github.com/intel/torch-xpu-ops/issues/3187",  # gpu_cpp_wrapper NotImplementedError
    6401: "https://github.com/intel/intel-xpu-backend-for-triton/issues/6401",  # Triton XPU backend
    140805: "https://github.com/pytorch/pytorch/issues/140805",  # AOTIModelContainerRunnerCpu
    143239: "https://github.com/pytorch/pytorch/issues/143239",  # scatter not implemented on XPU
    170049: "https://github.com/pytorch/pytorch/issues/170049",  # Blocked by PyTorch issue
    170636: "https://github.com/pytorch/pytorch/issues/170636",  # multiprocessing tensor reduction
    176191: "https://github.com/pytorch/pytorch/issues/176191",  # autotune_at_compile_time not working
    176204: "https://github.com/pytorch/pytorch/issues/176204",  # Test disabled
    176968: "https://github.com/pytorch/pytorch/issues/176968",  # Test disabled
    176969: "https://github.com/pytorch/pytorch/issues/176969",  # Test disabled
    176971: "https://github.com/pytorch/pytorch/issues/176971",  # Test disabled
    176972: "https://github.com/pytorch/pytorch/issues/176972",  # Test disabled
    177003: "https://github.com/pytorch/pytorch/issues/177003",  # Test disabled
    177004: "https://github.com/pytorch/pytorch/issues/177004",  # Test disabled
    177005: "https://github.com/pytorch/pytorch/issues/177005",  # Test disabled
    177054: "https://github.com/pytorch/pytorch/issues/177054",  # Test disabled
    177055: "https://github.com/pytorch/pytorch/issues/177055",  # Test disabled
    177056: "https://github.com/pytorch/pytorch/issues/177056",  # Test disabled
    177483: "https://github.com/pytorch/pytorch/issues/177483",  # Test disabled
    178575: "https://github.com/pytorch/pytorch/issues/178575",  # Test disabled
}
```

#### 4.2 Auto-match Issues Based on Message Patterns
```python
def find_issue(row):
    msg = str(row['message_xpu']) if pd.notna(row['message_xpu']) else ""
    classname = str(row['classname_xpu']) if pd.notna(row['classname_xpu']) else ""
    
    # DynamicShapesGpuWrapperGpuTests - NotImplementedError
    if "DynamicShapesGpuWrapperGpuTests" in classname and "NotImplementedError" in msg:
        return "3187"
    
    # TestGpuWrapper - NotImplementedError (same gpu_cpp_wrapper issue)
    if ("TestGpuWrapper" in classname or "GpuWrapperGpuTests" in classname) and "NotImplementedError" in msg:
        return "3187"
    
    # autotune_at_compile_time
    if "autotune_at_compile_time" in msg:
        return "176191"
    
    # Profile not enabled on XPU CI
    if "Profile not enabled" in msg:
        return "2334"
    
    # float8 conversion
    if "float8" in msg:
        return "2888"
    
    return None
```

### 5. Fix Truncated Issue URLs

**IMPORTANT**: Issue URLs in DetailReason must match the full issue numbers from message_xpu:

```python
def fix_issue_urls(df, wb, ws):
    """Fix truncated issue URLs by extracting from message_xpu"""
    detail_col = header_row.index('DetailReason') + 1
    
    for row_idx in range(2, ws.max_row + 1):
        df_idx = row_idx - 2
        if df_idx >= len(df):
            continue
        
        msg = str(df.loc[df_idx, 'message_xpu']) if pd.notna(df.loc[df_idx, 'message_xpu']) else ""
        current_detail = ws.cell(row_idx, detail_col).value
        
        # Extract issue number from message_xpu
        msg_issues = re.findall(r'github\.com/(?:pytorch/pytorch|intel/torch-xpu-ops)/issues/(\d+)', msg)
        
        if msg_issues and current_detail and 'github.com' in str(current_detail):
            correct_issue = msg_issues[0]
            if correct_issue in issue_map:
                correct_url = issue_map[correct_issue]
                
                # Replace truncated URL (e.g., #176 -> #176204)
                short_pattern = correct_issue[:3]  # First 3 digits
                if short_pattern in str(current_detail):
                    new_detail = current_detail.replace(short_pattern, correct_issue)
                    ws.cell(row_idx, detail_col, new_detail)
```

### 6. Check for Closed/Fixed Issues

For each issue URL in DetailReason, check if the issue is closed and fixed:

```python
def check_closed_issues():
    """Check GitHub issue status and mark fixed ones"""
    
    # Known closed/fixed issues:
    # - #2334 (Profile not enabled on XPU CI): CLOSED - XPU now supports profiling
    
    # For closed issues:
    # - Change Reason to "To be enabled"
    # - Append " - issue fixed" to DetailReason
    # - Apply green highlighting
    
    closed_issues = {
        "2334": "Profile not enabled on XPU CI - now fixed, XPU supports profiling"
    }
    
    for idx in feature_gap_rows:
        if issue_num in closed_issues:
            ws.cell(excel_row, reason_col, "To be enabled")
            ws.cell(excel_row, detail_col, current_detail + " - issue fixed")
            # Apply green fill
```

### 7. Update Excel with Formatting

- Add "Reason" column with "Feature gap" or "Failure (xpu broken)"
- Add "DetailReason" column with detailed explanation + issue URL
- **Blue fill with white text**: Has issue URL (open issue)
- **Yellow fill with black text**: No issue URL found (feature gap without tracked issue)
- **Green fill with black text**: Issue is fixed (closed issue) - Reason = "To be enabled"
- **No highlight**: Reason is "To be enabled", "Runner limitation", or "Not applicable"

```python
from openpyxl import load_workbook
from openpyxl.styles import Font, PatternFill

blue_fill = PatternFill(start_color='0000FF', end_color='0000FF', fill_type='solid')
white_font = Font(color='FFFFFF')
yellow_fill = PatternFill(start_color='FFFF00', end_color='FFFF00', fill_type='solid')
black_font = Font(color='000000')
green_fill = PatternFill(start_color='00FF00', end_color='00FF00', fill_type='solid')

# Check if has URL in DetailReason
if 'github.com' in detail:
    # Check if issue is closed/fixed
    if 'issue fixed' in detail:
        # Green highlighting
        ws.cell(row=row_num, column=detail_col).fill = green_fill
    else:
        # Blue highlighting
        ws.cell(row=row_num, column=detail_col).fill = blue_fill
        ws.cell(row=row_num, column=detail_col).font = white_font
else:
    # Check if it's a special case that doesn't need URL
    skip_patterns = ['To be enabled', 'Runner limitation', 'Not applicable']
    should_skip = any(pattern.lower() in reason.lower() for pattern in skip_patterns)
    
    if should_skip:
        # No highlighting
        ws.cell(row=row_num, column=detail_col).fill = PatternFill()
    else:
        # Yellow highlighting - no issue URL found
        ws.cell(row=row_num, column=detail_col).fill = yellow_fill
        ws.cell(row=row_num, column=detail_col).font = black_font
```

## Requirements
- pandas
- openpyxl
- PyGithub (optional, for fetching issues)
- Access to GitHub (for checking issue status)

## Output
- Updated Excel file with categorized failures
- Blue formatted Reason and DetailReason columns (has open issue URL)
- Yellow formatted Reason and DetailReason columns (no issue URL - feature gap)
- Green formatted Reason and DetailReason columns (issue is fixed - "To be enabled")
- No highlighting for "To be enabled", "Runner limitation", "Not applicable"
- GitHub issue URLs appended to DetailReason where applicable
- Truncated issue URLs fixed to match full issue numbers from message_xpu
- Closed/fixed issues marked with " - issue fixed" suffix