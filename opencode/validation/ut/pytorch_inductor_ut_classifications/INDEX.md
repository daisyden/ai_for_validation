# Inductor UT Test Analysis Skills

## Index

| # | Skill | File | Description |
|---|-------|------|-------------|
| 1 | Blank Status Analysis | `xpu-test-blank-analysis.md` | Analyze test cases with blank status_xpu and Reason |
| 2 | Failure Analysis | `xpu-test-analysis.md` | Categorize failed/skipped tests and add issue URLs |

## Quick Usage

### Step 1: Git Pull Repositories
```bash
cd ~/pytorch && git pull
cd ~/pytorch/third_party/torch-xpu-ops && git pull
```

### Step 2: Run Analysis
```bash
# Run the analysis scripts (see individual skill files for full details)
python3 <analysis_script>.py
```

### Step 3: Verify Results
- Check "Reason" column for: Feature gap, Failure (xpu broken), Not applicable, To be enabled
- Check "DetailReason" for GitHub issue URLs
- Blue = has issue URL, Yellow = no URL, Green = fixed

## Workflow Summary

### Blank Status Analysis (Skill 1)
1. Find rows where `status_xpu` is empty AND `Reason` is empty
2. For each row, check if test exists in pytorch/test files:
   - `cuda_case_exists`: Does test function exist in testfile_cuda?
   - `xpu_case_exists`: Does testfile contain XPU support keywords?
3. Categorize:
   - **Test removed** (cuda=False, xpu=False): Reason="Not applicable", DetailReason="Community change"
   - **@skipIfXpu** (cuda=True, xpu=False, has skip decorator): Reason="To be enabled", DetailReason="Daisy"
   - **Human investigation** (xpu=True): Reason="To be enabled", DetailReason="Human investigation"
4. **Post-processing**: Verify Human Investigation cases - if test parametrization changed (e.g., `test_low_memory_max_pool(dilation, dim, use_block_ptr)`), explain expected name in DetailReason and mark "Not applicable"

### Failure Analysis (Skill 2)
1. Find rows where `status_xpu` is "failed" or "skipped" with empty Reason
2. Categorize based on message patterns:
   - Feature gap: skipIfXpu, NotImplementedError, autotune_at_compile_time, etc.
   - Failure (xpu broken): RuntimeError, AssertionError, TypeError, etc.
   - To be enabled: requires cuda/cudagraph
   - Not applicable: CPU only, CPU in name
3. Add GitHub issue URLs to DetailReason:
   - Match issue based on error message and test class
   - Ensure URL matches full issue number from message_xpu
4. Check for fixed issues:
   - If issue is closed/fixed, change Reason to "To be enabled"
   - Append " - issue fixed" to DetailReason

## Excel File
- **Path**: `/home/daisydeng/classify_inductor/Inductor_ut_status_ww14_26.xlsx`
- **Sheet**: `Cuda pass xpu skip`

## Columns Updated

| Column | Description |
|--------|-------------|
| `Reason` | Feature gap, Failure (xpu broken), Not applicable, To be enabled |
| `DetailReason` | Explanation + GitHub issue URL |
| `cuda_case_exists` | Boolean (blank analysis) |
| `xpu_case_exists` | Boolean (blank analysis) |
| `Comments` | Explanation (blank analysis) |

## Color Coding

| Color | Meaning |
|-------|---------|
| Blue | Has GitHub issue URL (open issue) |
| Green | Issue is fixed - should be re-tested |
| Yellow | No issue URL found - needs investigation |
| No highlight | Not applicable, To be enabled, Runner limitation |

## Requirements
- pandas
- openpyxl
- Access to ~/pytorch/test directory
- GitHub access (for checking issue status)