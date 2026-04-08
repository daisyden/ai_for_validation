#!/usr/bin/env python3
"""
Generate markdown report from torch_xpu_ops_issues.xlsx
"""

import openpyxl
from datetime import datetime, timedelta
import os

ROOT_DIR = "/home/daisydeng"
RESULT_DIR = "/home/daisydeng/ai_for_validation/opencode/issue_triage/result"


def generate_report():
    wb = openpyxl.load_workbook(os.path.join(RESULT_DIR, 'torch_xpu_ops_issues.xlsx'))
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
            'created_time': ws_issues.cell(row, 7).value,
            'type': ws_issues.cell(row, 11).value,
            'module': ws_issues.cell(row, 12).value,
            'test_module': ws_issues.cell(row, 13).value,
            'dependency': ws_issues.cell(row, 14).value,
            'pr': ws_issues.cell(row, 15).value,
            'owner_transfer': ws_issues.cell(row, 19).value,
            'action_TBD': ws_issues.cell(row, 20).value,
            'duplicated_issue': ws_issues.cell(row, 21).value,
            'priority': ws_issues.cell(row, 22).value,
            'priority_reason': ws_issues.cell(row, 23).value,
            'category': ws_issues.cell(row, 24).value,
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
    
    # Group action_required by action_TBD
    action_groups = {}
    for issue in action_required:
        action = issue['action_TBD'] or 'Unknown'
        if action not in action_groups:
            action_groups[action] = []
        action_groups[action].append(issue)
    
    # Build statistics
    test_module_stats = {}
    module_stats = {}
    dep_stats = {}
    action_tbd_stats = {}
    category_stats = {}
    no_assignee_count = len(no_assignee)
    duplicated_count = len(duplicated)
    others_count = len(others)
    
    for issue in issues:
        # Test Module stats
        tm = issue['test_module'] or 'unknown'
        test_module_stats[tm] = test_module_stats.get(tm, 0) + 1
        
        # Module stats
        m = issue['module'] or 'unknown'
        module_stats[m] = module_stats.get(m, 0) + 1
        
        # With Dependency stats
        dep = issue['dependency']
        if dep and dep != 'None':
            dep_stats[dep] = dep_stats.get(dep, 0) + 1
        
        # Action_TBD stats
        action = issue['action_TBD']
        if action:
            action_tbd_stats[action] = action_tbd_stats.get(action, 0) + 1
        
        # Category stats
        cat = issue['category'] or 'unknown'
        category_stats[cat] = category_stats.get(cat, 0) + 1
    
    # Build markdown
    md = f"""# Torch XPU Ops Issue Report

**Repository:** [intel/torch-xpu-ops](https://github.com/intel/torch-xpu-ops)

**Issues Extracted:** April 5, 2026 (from GitHub issue data: `torch_xpu_ops_issues.json`)

**CI Data Sources:**
- Torch-XPU-OPS Nightly CI: XML files from `Inductor-XPU-UT-Data-*` + `Inductor-wheel-nightly-LTS2-XPU-E2E-Data-*` (commit: `a2d516a58c64f18b76880f3a77efbc02885d65af`)
- Stock PyTorch XPU CI: XML files from `test-default-*-linux.idc.xpu_*.zip` (run IDs: 69741866812, 69741866834, 69741866862, 69741866911, etc.)

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary

| Category | Count |
|----------|-------|
| Action Required | {len(action_required)} |
| No Assignee | {no_assignee_count} |
| Duplicated Issues | {duplicated_count} |
| With Dependency | {len(with_dependency)} |
| Others | {others_count} |
| **Total** | {len(issues)} |

---

## Statistics

### By Test Module

| Test Module | Count |
|-------------|-------|
"""
    
    for tm, count in sorted(test_module_stats.items(), key=lambda x: -x[1]):
        md += f"| {tm} | {count} |\n"
    
    md += """
### By Module

| Module | Count |
|--------|-------|
"""
    
    for m, count in sorted(module_stats.items(), key=lambda x: -x[1]):
        md += f"| {m} | {count} |\n"
    
    md += """
### By Dependency

| Dependency | Count |
|------------|-------|
"""
    
    for dep, count in sorted(dep_stats.items(), key=lambda x: -x[1]):
        md += f"| {dep} | {count} |\n"
    
    md += """
### By Action TBD

| Action TBD | Count |
|------------|-------|
"""
    
    for action, count in sorted(action_tbd_stats.items(), key=lambda x: -x[1]):
        md += f"| {action} | {count} |\n"
    
    md += """
### By Category

| Category | Count |
|----------|-------|
"""
    for cat, count in sorted(category_stats.items(), key=lambda x: -x[1]):
        md += f"| {cat} | {count} |\n"
    
    # Priority stats
    priority_stats = {}
    for issue in issues:
        p = issue['priority'] or 'unknown'
        priority_stats[p] = priority_stats.get(p, 0) + 1
    
    md += """
### By Priority

| Priority | Count |
|----------|-------|
"""
    for p, count in sorted(priority_stats.items()):
        md += f"| {p} | {count} |\n"
    
    md += f"""
### Other Stats

| Category | Count |
|----------|-------|
| Not Assigned | {no_assignee_count} |
| Duplicated Issues | {duplicated_count} |
| Others | {others_count} |

---

## 1. Action Required

Issues that need action based on test results analysis.

"""
    
    # Split action_required by action_TBD type
    for action, issues_list in sorted(action_groups.items()):
        md += f"""
### 1.{list(action_groups.keys()).index(action)+1} {action}

| ID | Title | Owner | Owner Transferred | TBD | Priority | Reason | Category | PR | Module | Test Module |
|---|-------|-------|-------------------|-----|---------|--------|----------|-----|--------|-------------|
"""
        for issue in issues_list:
            title = (issue['title'] or '')[:35]
            owner = issue['assignee'] or ''
            owner_transfer = issue['owner_transfer'] or ''
            module = issue['module'] or ''
            test_module = issue['test_module'] or ''
            pr = issue['pr'] or ''
            pr_link = f"[PR]({pr})" if pr else ''
            priority = issue['priority'] or ''
            reason = issue['priority_reason'] or ''
            category = issue['category'] or ''
            md += f"| [{issue['id']}](https://github.com/intel/torch-xpu-ops/issues/{issue['id']}) | {title} | {owner} | {owner_transfer} | {action} | {priority} | {reason} | {category} | {pr_link} | {module} | {test_module} |\n"
    
    md += """
### Issues without Assignee

| ID | Title | Owner | Owner Transferred | TBD | Priority | Reason | Category | PR | Module | Test Module |
|---|-------|-------|-------------------|-----|---------|--------|----------|-----|--------|-------------|
"""
    
    for issue in no_assignee[:50]:  # Limit to 50
        title = (issue['title'] or '')[:35]
        owner = issue['assignee'] or ''
        owner_transfer = 'chuanqi'
        action = 'assign owner'
        module = issue['module'] or ''
        test_module = issue['test_module'] or ''
        pr = issue['pr'] or ''
        pr_link = f"[PR]({pr})" if pr else ''
        priority = issue['priority'] or ''
        reason = issue['priority_reason'] or ''
        category = issue['category'] or ''
        md += f"| [{issue['id']}](https://github.com/intel/torch-xpu-ops/issues/{issue['id']}) | {title} | {owner} | {owner_transfer} | {action} | {priority} | {reason} | {category} | {pr_link} | {module} | {test_module} |\n"
    
    if len(no_assignee) > 50:
        md += f"\n*... and {len(no_assignee) - 50} more issues*\n"

    md += """
---

## 2. Duplicated Issues

Issues that share test cases with other issues.

| ID | Title | Owner | Reporter | Duplicated With | Priority | Reason | Category | PR | Module | Test Module |
|---|-------|-------|----------|-----------------|---------|--------|----------|-----|--------|-------------|
"""
    
    for issue in duplicated:
        title = (issue['title'] or '')[:30]
        owner = issue['assignee'] or ''
        reporter = issue['reporter'] or ''
        dup = issue['duplicated_issue'] or ''
        module = issue['module'] or ''
        test_module = issue['test_module'] or ''
        pr = issue['pr'] or ''
        pr_link = f"[PR]({pr})" if pr else ''
        priority = issue['priority'] or ''
        reason = issue['priority_reason'] or ''
        category = issue['category'] or ''
        md += f"| [{issue['id']}](https://github.com/intel/torch-xpu-ops/issues/{issue['id']}) | {title} | {owner} | {reporter} | {dup} | {priority} | {reason} | {category} | {pr_link} | {module} | {test_module} |\n"
    
    md += """
---

## 3. Issues with Dependency

Issues that have dependencies on other components (non-empty).

| ID | Title | Owner | Type | Priority | Reason | Category | Dependency | PR | Labels |
|---|-------|------|------|---------|--------|----------|------------|-----|--------|
"""
    
    # Filter and sort
    with_dependency = [i for i in with_dependency if i['dependency']]
    with_dependency.sort(key=lambda x: x['id'])
    for issue in with_dependency:
        title = (issue['title'] or '')[:25]
        owner = issue['assignee'] or ''
        issue_type = issue['type'] or ''
        dep = issue['dependency'] or ''
        labels = issue['labels'] or ''
        pr = issue['pr'] or ''
        pr_link = f"[PR]({pr})" if pr else ''
        priority = issue['priority'] or ''
        reason = issue['priority_reason'] or ''
        category = issue['category'] or ''
        md += f"| [{issue['id']}](https://github.com/intel/torch-xpu-ops/issues/{issue['id']}) | {title} | {owner} | {issue_type} | {priority} | {reason} | {category} | {dep} | {pr_link} | {labels} |\n"
    
    md += """
---

## 4. Others

Issues that don't fall into the categories above.

| ID | Title | Owner | Priority | Reason | Category | Labels | PR | Module | Test Module |
|---|-------|-------|---------|--------|----------|--------|-----|--------|-------------|
"""
    
    for issue in others:
        title = (issue['title'] or '')[:30]
        owner = issue['assignee'] or ''
        labels = issue['labels'] or ''
        module = issue['module'] or ''
        test_module = issue['test_module'] or ''
        pr = issue['pr'] or ''
        pr_link = f"[PR]({pr})" if pr else ''
        priority = issue['priority'] or ''
        reason = issue['priority_reason'] or ''
        category = issue['category'] or ''
        md += f"| [{issue['id']}](https://github.com/intel/torch-xpu-ops/issues/{issue['id']}) | {title} | {owner} | {priority} | {reason} | {category} | {labels} | {pr_link} | {module} | {test_module} |\n"
    
    # New issues in last 10 days
    ten_days_ago = datetime.now() - timedelta(days=10)
    recent_issues = []
    for issue in issues:
        created_str = issue.get('created_time')
        if created_str:
            try:
                created_date = datetime.strptime(created_str.replace('Z', '+00:00'), '%Y-%m-%dT%H:%M:%S%z')
                # Convert to naive datetime for comparison
                created_date = created_date.replace(tzinfo=None)
                if created_date >= ten_days_ago:
                    recent_issues.append(issue)
            except:
                pass
    
    recent_issues.sort(key=lambda x: x['id'], reverse=True)
    
    md += f"""
---

## 5. Recent Issues (Last 10 Days)

Issues created in the last 10 days (as of {datetime.now().strftime('%Y-%m-%d')}).

| ID | Title | Status | Owner | Priority | Reason | Category | Labels | Module | Test Module |
|---|-------|--------|-------|---------|--------|----------|--------|--------|-------------|
"""
    
    for issue in recent_issues:
        title = (issue['title'] or '')[:40]
        status = issue['status'] or ''
        owner = issue['assignee'] or ''
        labels = issue['labels'] or ''
        module = issue['module'] or ''
        test_module = issue['test_module'] or ''
        priority = issue['priority'] or ''
        reason = issue['priority_reason'] or ''
        category = issue['category'] or ''
        md += f"| [{issue['id']}](https://github.com/intel/torch-xpu-ops/issues/{issue['id']}) | {title} | {status} | {owner} | {priority} | {reason} | {category} | {labels} | {module} | {test_module} |\n"
    
    # Save to file
    output_path = os.path.join(RESULT_DIR, 'issue_report.md')
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
