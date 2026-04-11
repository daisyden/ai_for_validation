"""
Duplicated Issue Detection Module

Detects duplicated issues in the Test_Cases sheet based on:
1. Test Class + Test Case matching
2. Similar Traceback patterns

Used for updating the Issues sheet with duplicated_issue column.
"""

from collections import defaultdict
from typing import Dict, List, Set, Any


def find_duplicated_issues(ws) -> Dict[int, List[str]]:
    """
    Find duplicated issues based on Test Class + Test Case or similar Traceback.

    Build two indexes for duplicate detection:
    1. (Test Class, Test Case) -> [(row, issue_id)]
    2. Traceback -> [(row, issue_id)]

    Args:
        ws: openpyxl worksheet for Test_Cases sheet
            Expected columns:
            - Col 1: issue_id
            - Col 6: test_class
            - Col 7: test_case
            - Col 9: traceback

    Returns:
        Dict mapping row number -> list of duplicated issue IDs
    """
    class_case_index = defaultdict(list)
    for row in range(2, ws.max_row + 1):
        test_class = ws.cell(row, 6).value
        test_case = ws.cell(row, 7).value
        issue_id = ws.cell(row, 1).value
        if test_class and test_case:
            key = (test_class, test_case)
            class_case_index[key].append((row, issue_id))

    traceback_index = defaultdict(list)
    for row in range(2, ws.max_row + 1):
        traceback = ws.cell(row, 9).value or ''
        issue_id = ws.cell(row, 1).value

        if 'AssertionError: Tensor-likes are not close!' in traceback:
            continue

        norm = traceback.strip()
        if norm:
            traceback_index[norm].append((row, issue_id))

    class_case_duplicates = {}
    for key, rows in class_case_index.items():
        if len(rows) > 1:
            issue_ids = [rid for _, rid in rows]
            for row, rid in rows:
                other_issues = [i for i in issue_ids if i != rid]
                if other_issues:
                    class_case_duplicates[row] = other_issues

    traceback_duplicates = {}
    for key, rows in traceback_index.items():
        if len(rows) > 1:
            issue_ids = [rid for _, rid in rows]
            for row, rid in rows:
                other_issues = [i for i in issue_ids if i != rid]
                if other_issues:
                    if row in traceback_duplicates:
                        traceback_duplicates[row].extend(other_issues)
                    else:
                        traceback_duplicates[row] = other_issues

    merged_duplicates = {}
    for row in range(2, ws.max_row + 1):
        dup_set = set()
        if row in class_case_duplicates:
            dup_set.update(class_case_duplicates[row])
        if row in traceback_duplicates:
            dup_set.update(traceback_duplicates[row])
        if dup_set:
            merged_duplicates[row] = sorted(list(dup_set))

    return merged_duplicates


def find_duplicates_by_test_class_case(
    ws, skip_tensor_likes: bool = True
) -> Dict[int, List[str]]:
    """
    Find duplicates based only on Test Class + Test Case matching.

    Args:
        ws: openpyxl worksheet for Test_Cases sheet
        skip_tensor_likes: Skip issues with "Tensor-likes are not close!" traceback

    Returns:
        Dict mapping row number -> list of duplicated issue IDs
    """
    class_case_index = defaultdict(list)
    for row in range(2, ws.max_row + 1):
        test_class = ws.cell(row, 6).value
        test_case = ws.cell(row, 7).value
        issue_id = ws.cell(row, 1).value
        if test_class and test_case:
            key = (test_class, test_case)
            class_case_index[key].append((row, issue_id))

    duplicates = {}
    for key, rows in class_case_index.items():
        if len(rows) > 1:
            issue_ids = [rid for _, rid in rows]
            for row, rid in rows:
                other_issues = [i for i in issue_ids if i != rid]
                if other_issues:
                    duplicates[row] = sorted(other_issues)

    return duplicates


def find_duplicates_by_traceback(
    ws, skip_tensor_likes: bool = True, min_length: int = 50
) -> Dict[int, List[str]]:
    """
    Find duplicates based on traceback similarity.

    Args:
        ws: openpyxl worksheet for Test_Cases sheet
        skip_tensor_likes: Skip issues with "Tensor-likes are not close!" traceback
        min_length: Minimum traceback length to consider

    Returns:
        Dict mapping row number -> list of duplicated issue IDs
    """
    traceback_index = defaultdict(list)
    for row in range(2, ws.max_row + 1):
        traceback = ws.cell(row, 9).value or ''
        issue_id = ws.cell(row, 1).value

        if skip_tensor_likes and 'AssertionError: Tensor-likes are not close!' in traceback:
            continue

        norm = traceback.strip()
        if norm and len(norm) >= min_length:
            traceback_index[norm].append((row, issue_id))

    duplicates = {}
    for key, rows in traceback_index.items():
        if len(rows) > 1:
            issue_ids = [rid for _, rid in rows]
            for row, rid in rows:
                other_issues = [i for i in issue_ids if i != rid]
                if other_issues:
                    if row in duplicates:
                        duplicates[row].extend(other_issues)
                    else:
                        duplicates[row] = sorted(other_issues)

    return duplicates


def get_duplicate_issues_by_issue_id(
    ws_test, ws_issues=None
) -> Dict[str, Set[str]]:
    """
    Get set of duplicate issue IDs for each issue ID.
    Aggregates duplicates from all test cases belonging to the same issue.

    Args:
        ws_test: Test_Cases worksheet
        ws_issues: Issues worksheet (optional)

    Returns:
        Dict mapping issue_id -> set of duplicate issue IDs
    """
    issue_duplicates = {}

    for row in range(2, ws_test.max_row + 1):
        issue_id = ws_test.cell(row, 1).value
        dup_issue = ws_test.cell(row, 22).value

        if issue_id not in issue_duplicates:
            issue_duplicates[issue_id] = set()

        if dup_issue:
            for dup_id in str(dup_issue).split(','):
                dup_id = dup_id.strip()
                if dup_id and dup_id != str(issue_id):
                    issue_duplicates[issue_id].add(dup_id)

    if ws_issues:
        for row in range(2, ws_issues.max_row + 1):
            issue_id = ws_issues.cell(row, 1).value
            dup_issue = ws_issues.cell(row, 22).value

            if issue_id not in issue_duplicates:
                issue_duplicates[issue_id] = set()

            if dup_issue:
                for dup_id in str(dup_issue).split(','):
                    dup_id = dup_id.strip()
                    if dup_id and dup_id != str(issue_id):
                        issue_duplicates[issue_id].add(dup_id)

    return {k: sorted(v) for k, v in issue_duplicates.items() if v}


class DuplicatedIssueAnalyzer:
    """
    High-level interface for duplicated issue detection and management.
    """

    def __init__(self, skip_tensor_likes: bool = True):
        """
        Initialize the duplicate analyzer.

        Args:
            skip_tensor_likes: Skip issues with Tensor-likes assertion errors
        """
        self.skip_tensor_likes = skip_tensor_likes

    def analyze_duplicates(self, ws_test) -> Dict[int, List[str]]:
        """
        Analyze Test_Cases sheet for duplicated issues.

        Args:
            ws_test: Test_Cases worksheet

        Returns:
            Dict mapping row -> list of duplicate issue IDs
        """
        return find_duplicated_issues(ws_test)

    def get_issue_duplicates_map(self, ws_test) -> Dict[str, Set[str]]:
        """
        Get mapping of issue ID -> set of duplicate issue IDs.

        Args:
            ws_test: Test_Cases worksheet

        Returns:
            Dict mapping issue_id -> set of duplicate issue IDs
        """
        return get_duplicate_issues_by_issue_id(ws_test)

    def update_issues_sheet_duplicates(
        self, ws_issues, ws_test, dup_map: Dict[int, List[str]]
    ) -> int:
        """
        Update Issues sheet with duplicated_issue column.

        Args:
            ws_issues: Issues worksheet
            ws_test: Test_Cases worksheet
            dup_map: Dict from find_duplicated_issues

        Returns:
            Number of issues updated
        """
        count = 0

        if 22 not in [cell.column for cell in ws_issues[1]]:
            ws_issues.cell(1, 22, 'duplicated_issue')

        issue_row_map = {}
        for row in range(2, ws_issues.max_row + 1):
            issue_id = ws_issues.cell(row, 1).value
            if issue_id:
                issue_row_map[issue_id] = row

        for row in range(2, ws_test.max_row + 1):
            issue_id = ws_test.cell(row, 1).value
            if issue_id and issue_id in issue_row_map:
                dup_issues = dup_map.get(row, [])
                if dup_issues:
                    issue_row = issue_row_map[issue_id]
                    ws_issues.cell(issue_row, 22, ','.join(dup_issues))
                    count += 1

        return count

    def merge_with_existing_duplicates(
        self, existing_dups: Dict[str, Set[str]], new_dups: Dict[str, Set[str]]
    ) -> Dict[str, Set[str]]:
        """
        Merge existing duplicate info with newly detected duplicates.

        Args:
            existing_dups: Existing duplicate mapping
            new_dups: New duplicate mapping

        Returns:
            Merged duplicate mapping
        """
        merged = {}
        all_issue_ids = set(existing_dups.keys()) | set(new_dups.keys())

        for issue_id in all_issue_ids:
            dup_set = set()
            if issue_id in existing_dups:
                dup_set.update(existing_dups[issue_id])
            if issue_id in new_dups:
                dup_set.update(new_dups[issue_id])
            if dup_set:
                merged[issue_id] = dup_set

        return merged