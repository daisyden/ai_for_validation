# PR Extraction

## Overview
Extracts PR information from GitHub issue comments and updates the Excel file with PR details (URL, owner, state, description).

## Workflow
1. Load `torch_xpu_ops_issues.xlsx` with Issues sheet
2. Extract PR references from issue comments using fix-related keywords (fix, closes, resolved, PR #, etc.)
3. Fetch PR information via GitHub API (pytorch/pytorch and intel/torch-xpu-ops repos)
4. Add PR columns at first blank column (preserves existing headers)
5. Populate: PR, PR Owner, PR Status, PR Description

## Usage
```bash
cd /home/daisydeng/ai_for_validation/opencode/issue_triage/issue_analysis/pr-extraction
python3 pr_extraction.py <EXCEL_FILE> [--issues COMMA_SEPARATED_IDS]
```

## Examples
```bash
# Run with default Excel file
python3 pr_extraction.py $RESULT_DIR/torch_xpu_ops_issues.xlsx

# Process specific issues only
python3 pr_extraction.py $RESULT_DIR/torch_xpu_ops_issues.xlsx --issues 3286,3284,3258
```

## Output
PR columns added at first blank column:
- **PR**: PR URLs (semicolon-separated)
- **PR Owner**: Owner usernames (semicolon-separated)
- **PR Status**: open/closed/merged
- **PR Description**: PR titles (semicolon-separated)

## PR Detection Criteria
- Extracts PR URLs from issue comments only (not issue body)
- Matches fix-related keywords: fix, fixes, fixed, close, closes, closed, resolve, resolved, PR #, PR:, Pull Request
- Supports PRs from: `pytorch/pytorch` and `intel/torch-xpu-ops`

## Status Logic
- **pytorch/pytorch**: Checks "Merged" label for merged status
- **intel/torch-xpu-ops**: Checks `merged` or `merged_at` field

## Related Info
- Input: Issues sheet with issue_id in column A
- API: GitHub REST API v3
- Requires: `GITHUB_TOKEN` env var for higher rate limits (optional)
- Note: Adds columns dynamically at first blank column - preserves existing headers