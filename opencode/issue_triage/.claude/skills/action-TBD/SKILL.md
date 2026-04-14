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

## Usage
```bash
cd /home/daisydeng/ai_for_validation/opencode/issue_triage/issue_analysis/action_TBD
python3 run_action.py [--excel EXCEL_FILE] [--limit N] [--force]
```

## Examples
```bash
# Run full action analysis on all issues
python3 run_action.py

# Test with first 10 issues
python3 run_action.py --limit 10

# Force overwrite existing values
python3 run_action.py --force
```

## Action Types

| Action | Condition |
|--------|-----------|
| **add to skiplist** | not_target/wontfix issues or tests that cannot be enabled |
| **Close fixed issue** | All test cases passed on XPU/stock or all E2E accuracy passed |
| **Enable test** | Test cases can be enabled on XPU (takes precedence over skiplist) |
| **Verify the issue** | PR closed but no failed tests |
| **Revisit the PR as case failed** | PR closed but tests still failing |
| **Needs Upstream Skip PR (not_target + ut_upstream)** | Upstream issue with not_target label |
| **Needs Skip PR (wontfix / not_target)** | wontfix or not_target issues |
| **Awaiting response from reporter** | Info already requested from reporter |
| **Need reproduce steps** | LLM suggests missing repro steps |
| **LLM Suggestion: ...** | Other LLM-suggested actions |
| **Need more information - ...** | Missing key details (accuracy data, perf numbers, etc.) |
| **Need Investigation** | Fallback action - no specific action found OR E2E accuracy issue pending |

## Action Priority Order

1. **add to skiplist** - not_target/wontfix or cannot enable
2. **Close fixed issue** - all tests passed
3. **Enable test** - can_enable_true takes precedence over can_enable_false
4. **Verify the issue** - PR closed, no failures
5. **Revisit the PR as case failed** - PR closed, still failing
6. **Needs Upstream Skip PR** - upstream + not_target (consolidated in action_needs_skip_pr)
7. **Needs Skip PR** - wontfix/not_target
8. **Awaiting response from reporter** - info requested
9. **Need reproduce steps** - LLM suggests repro needed
10. **LLM Suggestion** - other LLM recommendations
11. **Need more information** - missing key details

## Key Updates (Recent Changes)

### 1. Dynamic Column Detection by Header Names
```python
# OLD: Hardcoded column positions
xpu_status = ws_test.cell(tr, 12).value  # Fixed column index

# NEW: Detect by header name
xpu_status_col = get_column_by_header(ws_test, 'XPU Status')
xpu_status = ws_test.cell(tr, xpu_status_col).value
```

### 2. action_enable_test() Logic
```python
# Now properly checks both flags with precedence
def action_enable_test(can_enable_true, can_enable_false, reporter):
    if not can_enable_true and not can_enable_false:
        return (None, None, None)
    
    if can_enable_true:
        return (reporter, 'Enable test', 'Test cases can be enabled on XPU')
    else:
        return (reporter, 'add to skiplist', 'Test cases cannot be enabled on XPU')
```

### 3. Removed Duplicate Function
- `action_needs_upstream_skip_pr()` was identical to logic inside `action_needs_skip_pr()`
- Consolidated into single `action_needs_skip_pr()` function
- The upstream skip PR logic is now handled within `action_needs_skip_pr()` as first condition

## Output
- **Action TBD column**: Action type (e.g., "Close fixed issue", "add to skiplist")
- **Owner Transfer column**: Suggested owner to transfer
- **Action Reason column**: Detailed explanation of the action

## Key Features
- **Rule-based analysis**: No LLM calls for main logic
- **LLM fallback**: Uses Qwen3-32B for reproduce steps detection
- **Dynamic column detection**: Finds columns by header names, not fixed positions
- **Safe column addition**: Uses existing columns or adds at first blank
- **Force mode**: Overwrite existing action values with `--force`
- **Skip existing**: By default skips issues that already have action

## Related Info
- Analyzer: `issue_analysis/action_TBD/action_analyzer.py`
- Runner: `issue_analysis/action_TBD/run_action.py`
- Input: Issues sheet and Test Cases sheet (both require headers)
- Output: Action columns at first blank column (preserves existing data)