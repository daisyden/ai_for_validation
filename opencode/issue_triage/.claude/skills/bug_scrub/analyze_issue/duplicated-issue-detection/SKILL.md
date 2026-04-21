# Duplicated Issue Detection

## Overview
Detects duplicate issues in the Test_Cases sheet based on Test Class + Test Case matching, and adds a `duplicated_issue` column to the Issues sheet.

## Workflow
1. Load `torch_xpu_ops_issues.xlsx` with Issues and Test Cases sheets
2. Build index of (Test Class, Test Case) → (row, issue_id) mappings
3. Find test cases with identical Test Class + Test Case across multiple issues
4. Add `duplicated_issue` column at first blank column in Issues sheet
5. Populate with comma-separated list of duplicate issue IDs

## Usage
```bash
cd /home/daisydeng/ai_for_validation/opencode/issue_triage/issue_analysis/duplicated
python3 duplicated_issue_detector.py [--excel EXCEL_FILE]
```

## Example
```bash
# Run with default paths
python3 duplicated_issue_detector.py

# With custom Excel file
python3 duplicated_issue_detector.py --excel /path/to/torch_xpu_ops_issues.xlsx
```

## Output
- Adds `duplicated_issue` column at first blank column (column 15) in Issues sheet
- Populates with duplicate issue IDs, e.g., "2715,3286"
- Example: Issue 2715 has duplicate Issue 3286

## Duplicate Criteria
- **Test Class + Test Case Match**: Same test class AND same test case name
- Different errors/tracebacks are grouped together if they test the same case

## Related Info
- Based on `duplicated_analyzer.py` from issue_analysis/duplicated/
- Runs on Test Cases sheet (Col 1=issue_id, Col 6=test_class, Col 7=test_case)
- Updates Issues sheet column

## Known Limitations
- Only checks exact Test Class + Test Case matches
- Does NOT use traceback similarity (unlike PASS 5 in test_result_analysis)