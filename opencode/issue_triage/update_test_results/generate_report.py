#!/usr/bin/env python3
"""
Generate markdown report from torch_xpu_ops_issues.xlsx
"""

import openpyxl
from datetime import datetime


def generate_report():
    wb = openpyxl.load_workbook('/home/daisydeng/issue_traige/data/torch_xpu_ops_issues.xlsx')
    ws_issues = wb['Issues']
    
    # Collect issue data
    issues = []
    for row in range(2, ws_issues.max_row + 1):
        issue = {
            'id': ws_issues.cell(row, 1).value,
            'title': ws_issues.cell(row, 2).value,
            'status': ws_issues.cell(row, 3).value,
            'assignee': ws_issues.cell(row, 4).value,
            'reporter': ws_issues.cell(row, 5).value,
            'labels': ws_issues.cell(row, 6).value,
            'type': ws_issues.cell(row, 11).value,
            'module': ws_issues.cell(row, 12).value,
            'test_module': ws_issues.cell(row, 13).value,
            'dependency': ws_issues.cell(row, 14).value,
            'owner_transfer': ws_issues.cell(row, 19).value,
            'action_TBD': ws_issues.cell(row, 20).value,
            'duplicated_issue': ws_issues.cell(row, 21).value,
        }
        issues.append(issue)
    
    # Categorize issues
    # Categorize issues
    # Note: Priority - duplicated issues first (they won't appear in action_required)
    action_required = []  # Issues with action_TBD
    no_assignee = []  # Issues without assignee
    duplicated = []  # Issues with duplicates (ALL of them)
    with_dependency = []  # Issues with dependency (non-None and non-"None" string)
    others = []  # Everything else
    
    for issue in issues:
        dep = issue['dependency']
        is_valid_dep = dep and dep not in ['None', None, '']
        
        if issue['duplicated_issue']:
            # Always add to duplicated - include ALL issues with duplicates
            duplicated.append(issue)
        elif issue['action_TBD']:
            action_required.append(issue)
        elif not issue['assignee'] or issue['assignee'] == 'None':
            no_assignee.append(issue)
        elif is_valid_dep:
            with_dependency.append(issue)
        else:
            others.append(issue)
    
    # Sort action_required by action_TBD
    action_required.sort(key=lambda x: (x['action_TBD'] or '', x['id']))
    
    # Sort duplicated by ID
    duplicated.sort(key=lambda x: x['id'])
    
    # Sort with_dependency by ID
    with_dependency.sort(key=lambda x: x['id'])
    
    # Build markdown
    md = f"""# Torch XPU Ops Issue Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary

| Category | Count |
|----------|-------|
| Action Required | {len(action_required)} |
| No Assignee | {len(no_assignee)} |
| Duplicated Issues | {len(duplicated)} |
| With Dependency | {len(with_dependency)} |
| Others | {len(others)} |
| **Total** | {len(issues)} |

---

## 1. Action Required

Issues that need action based on test results analysis.

### 1.1 Issues with action_TBD

| ID | Title | Owner | Owner Transferred | TBD | Module | Test Module |
|---|-------|-------|-------------------|-----|--------|-------------|
"""
    
    for issue in action_required:
        title = (issue['title'] or '')[:50]
        owner = issue['assignee'] or ''
        owner_transfer = issue['owner_transfer'] or ''
        action = issue['action_TBD'] or ''
        module = issue['module'] or ''
        test_module = issue['test_module'] or ''
        md += f"| {issue['id']} | {title} | {owner} | {owner_transfer} | {action} | {module} | {test_module} |\n"
    
    md += """
### 1.2 Issues without Assignee

| ID | Title | Owner | Owner Transferred | TBD | Module | Test Module |
|---|-------|-------|-------------------|-----|--------|-------------|
"""
    
    for issue in no_assignee[:50]:  # Limit to 50
        title = (issue['title'] or '')[:50]
        owner = issue['assignee'] or ''
        owner_transfer = 'chuanqi'
        action = 'assign owner'
        module = issue['module'] or ''
        test_module = issue['test_module'] or ''
        md += f"| {issue['id']} | {title} | {owner} | {owner_transfer} | {action} | {module} | {test_module} |\n"
    
    if len(no_assignee) > 50:
        md += f"\n*... and {len(no_assignee) - 50} more issues*\n"
    
    md += """
---

## 2. Duplicated Issues

Issues that share test cases with other issues.

| ID | Title | Owner | Reporter | Duplicated With | Module | Test Module |
|---|-------|-------|----------|-----------------|--------|-------------|
"""
    
    for issue in duplicated:
        title = (issue['title'] or '')[:40]
        owner = issue['assignee'] or ''
        reporter = issue['reporter'] or ''
        dup = issue['duplicated_issue'] or ''
        module = issue['module'] or ''
        test_module = issue['test_module'] or ''
        md += f"| {issue['id']} | {title} | {owner} | {reporter} | {dup} | {module} | {test_module} |\n"
    
    md += """
---

## 3. Issues with Dependency

Issues that have dependencies on other components (non-empty).

| ID | Title | Owner | Type | Module | Test Module | Dependency | Labels |
|---|-------|------|------|--------|-------------|------------|--------|
"""
    
    # Filter and sort
    with_dependency = [i for i in with_dependency if i['dependency']]
    with_dependency.sort(key=lambda x: x['id'])
    for issue in with_dependency:
        title = (issue['title'] or '')[:35]
        owner = issue['assignee'] or ''
        issue_type = issue['type'] or ''
        module = issue['module'] or ''
        test_module = issue['test_module'] or ''
        dep = issue['dependency'] or ''
        labels = (issue['labels'] or '')[:25]
        md += f"| {issue['id']} | {title} | {owner} | {issue_type} | {module} | {test_module} | {dep} | {labels} |\n"
    
    md += """
---

## 4. Others

Issues that don't fall into the categories above.

| ID | Title | Owner | Reporter | Labels | Module | Test Module |
|---|-------|-------|----------|--------|--------|-------------|
"""
    
    for issue in others[:50]:  # Limit to 50
        title = (issue['title'] or '')[:40]
        owner = issue['assignee'] or ''
        reporter = issue['reporter'] or ''
        labels = (issue['labels'] or '')[:25]
        module = issue['module'] or ''
        test_module = issue['test_module'] or ''
        md += f"| {issue['id']} | {title} | {owner} | {reporter} | {labels} | {module} | {test_module} |\n"
    
    if len(others) > 50:
        md += f"\n*... and {len(others) - 50} more issues*\n"
    
    # Save to file
    output_path = '/home/daisydeng/ai_for_validation/opencode/issue_triage/torch-xpu-ops-issue-collection/issue_report.md'
    with open(output_path, 'w') as f:
        f.write(md)
    
    print(f"Report saved to: {output_path}")
    print(f"Total issues: {len(issues)}")
    print(f"  Action Required: {len(action_required)}")
    print(f"  No Assignee: {len(no_assignee)}")
    print(f"  Duplicated: {len(duplicated)}")
    print(f"  With Dependency: {len(with_dependency)}")
    print(f"  Others: {len(others)}")


if __name__ == '__main__':
    generate_report()
