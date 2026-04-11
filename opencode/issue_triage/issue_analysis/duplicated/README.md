# Duplicated Issue Detection Module

Detects duplicated issues in the Test_Cases sheet for updating the Issues sheet.

## Overview

This module provides functionality to detect and manage duplicate issues based on:
1. **Test Class + Test Case matching** - Same test class and case across multiple issues
2. **Traceback similarity** - Similar error tracebacks indicating same root cause

## Usage

```python
from issue_analysis.duplicated import DuplicatedIssueAnalyzer, find_duplicated_issues

# Using the analyzer class
analyzer = DuplicatedIssueAnalyzer(skip_tensor_likes=True)

# Detect duplicates in Test_Cases sheet
duplicates = analyzer.analyze_duplicates(ws_test)
# Returns: {row_number: [issue_id1, issue_id2, ...], ...}

# Get issue ID to duplicate issue IDs mapping
issue_dup_map = analyzer.get_issue_duplicates_map(ws_test)
# Returns: {issue_id: {dup_id1, dup_id2, ...}, ...}

# Update Issues sheet with duplicate info
analyzer.update_issues_sheet_duplicates(ws_issues, ws_test, duplicates)

# Using the function directly
duplicates = find_duplicated_issues(ws_test)
```

## Functions

### `find_duplicated_issues(ws)` 
Main function to detect duplicated issues.

**Args:**
- `ws`: openpyxl worksheet for Test_Cases sheet (expected columns: 1=issue_id, 6=test_class, 7=test_case, 9=traceback)

**Returns:**
- `Dict[int, List[str]]`: Mapping of row number to list of duplicate issue IDs

### `find_duplicates_by_test_class_case(ws, skip_tensor_likes=True)`
Detect duplicates based only on Test Class + Test Case matching.

### `find_duplicates_by_traceback(ws, skip_tensor_likes=True, min_length=50)`
Detect duplicates based on traceback similarity.

### `get_duplicate_issues_by_issue_id(ws_test, ws_issues=None)`
Aggregate duplicates by issue ID.

## Integration with Issues Sheet

```python
# In update_test_results.py flow
ws_test = wb['Test Cases']
ws_issues = wb['Issues']

# Detect duplicates
dups = find_duplicated_issues(ws_test)

# Update Issues sheet column 22
for row in range(2, ws_issues.max_row + 1):
    issue_id = ws_issues.cell(row, 1).value
    if issue_id and issue_id in issue_row_map:
        dup_issues = dup_map.get(issue_row_map[issue_id], [])
        if dup_issues:
            ws_issues.cell(row, 22, ','.join(dup_issues))
```