# Action TBD Analysis

## Overview
Analyzes PyTorch XPU issues to determine appropriate actions (Action TBD, Owner Transfer, Action Reason) using rule-based ActionAnalyzer. Adds columns to Issues sheet without modifying existing data.

## Workflow
1. Load `torch_xpu_ops_issues.xlsx` with Issues and Test Cases sheets
2. Dynamically detect column indices by header names (not hardcoded positions)
3. For each issue, analyze:
   - Labels (not_target, wontfix, upstream, etc.)
   - Test statuses (XPU, Stock from Test Cases sheet)
   - PR status
   - Issue content (title, summary, error info)
4. Apply action rules from ActionAnalyzer in priority order
5. Add columns: Action TBD, Owner Transfer, Action Reason

## Usage to Generate Actions
```bash
cd ~/ai_for_validation/opencode/issue_triage/issue_analysis/action_TBD
python3 run_action.py [--excel EXCEL_FILE] [--limit N] [--force]
```

## Issue Report Generation
Generate comprehensive Markdown report at `result/issue_report.md`:
```bash
cd ~/ai_for_validation/opencode/issue_triage/issue_analysis
python3 generate_issue_report.py [EXCEL_FILE] [OUTPUT_FILE]
```

## Action Required Section Structure

Issues are categorized into two main sections based on who needs to take action:

### 2.1 Developer AR (Reporter/Community action needed)

| # | Action | Owner | Current Count |
|---|--------|-------|-------------|
| 2.1.1 | Need Investigation | assignee | 89 |
| 2.1.2 | Awaiting response | reporter | 53 |
| 2.1.3 | Awaiting response from reporter | reporter | 143 |
| 2.1.4 | E2E accuracy issue | assignee | 11 |

### 2.2 Reporter AR (Other actions pending)

| # | Action | Owner | Current Count |
|---|--------|-------|-------------|
| 2.2.5 | Needs Upstream Skip PR | assignee | 76 |
| 2.2.6 | add to skiplist | varies | 5 |
| 2.2.7 | Close fixed issue | reporter | 4 |
| 2.2.8 | Verify the issue | reporter | 3 |

**Total Issues: 384**

## Priority Details

### Priority 1: add to skiplist
- Triggers: Issue has `not_target` or `wontfix` label
- Owner: assignee or default

### Priority 2: Close fixed issue
- Triggers: BOTH XPU and Stock all passed AND NO failures
- Owner: reporter

### Priority 3: Verify the issue
- Triggers: Any PR reported AND no failures
- Owner: reporter

### Priority 4: Need Investigation
- Fallback when no other priority matches
- Owner: assignee

### Priority 5: Needs Upstream Skip PR
- Triggers: Issue has upstream label
- Owner: assignee

### Priority 7: E2E accuracy issue
- Triggers: E2E module AND accuracy keyword in title/summary
- Owner: assignee

### Priority 8: Awaiting response
- Triggers: Bug/Perf type AND no failures
- Owner: reporter

### Priority 9: Awaiting response from reporter
- Triggers: Maintainer asked for info in issue title/summary
- Owner: reporter

## Report Sections
1. Summary (Dev AR + Reporter AR counts)
2. Action Required (Detail tables for each action)
3. Issues by Category
4. Last Week Issues
5. Stale Issues (>2 weeks)
6. Dependency Issues
7. Duplicated Issues
8. Statistics

## Related Files
- Analyzer: `issue_analysis/action_TBD/action_analyzer.py`
- Runner: `issue_analysis/action_TBD/run_action.py`
- Report Generator: `issue_analysis/generate_issue_report.py`
- Input: Issues sheet and Test Cases sheet (both require headers)
- Output: Action columns at first blank column (preserves existing data)
- Report: `result/issue_report.md`