#!/usr/bin/env python3
"""
Generate markdown report from torch_xpu_ops_issues.xlsx
Reads Excel columns by header names, not hardcoded positions.
"""

import openpyxl
from datetime import datetime, timedelta
import os
import re

ROOT_DIR = "/home/daisydeng"
RESULT_DIR = os.environ.get("RESULT_DIR", "/home/daisydeng/ai_for_validation/opencode/issue_triage/result")


def get_all_columns_by_header(ws):
    """Get dict of all columns by header name."""
    col_map = {}
    for col in range(1, ws.max_column + 1):
        h = ws.cell(1, col).value
        if h:
            col_map[h] = col
    return col_map


def clean_category_name(cat):
    """Remove number prefix like '9 - ', '3. ' from category names."""
    if not cat:
        return 'unknown'
    cat = re.sub(r'^\d+\s*-\s*', '', cat)
    cat = re.sub(r'^\d+\.\s*', '', cat)
    return cat.strip()


def generate_report():
    wb = openpyxl.load_workbook(os.path.join(RESULT_DIR, 'torch_xpu_ops_issues.xlsx'))
    ws_issues = wb['Issues']
    
    # Get columns by header names
    issue_cols = get_all_columns_by_header(ws_issues)
    
    print(f"Available columns: {list(issue_cols.keys())}")
    
    # Collect issue data using header names
    issues = []
    for row in range(2, ws_issues.max_row + 1):
        def get_val(header, default=None):
            col = issue_cols.get(header)
            return ws_issues.cell(row, col).value if col else default
        
        issue = {
            'id': get_val('Issue ID'),
            'title': get_val('Title'),
            'summary': get_val('Summary'),
            'status': get_val('Status'),
            'assignee': get_val('Assignee'),
            'reporter': get_val('Reporter'),
            'labels': get_val('Labels'),
            'created_time': get_val('Created Time'),
            'updated_time': get_val('Updated Time'),
            'type': get_val('Type'),
            'module': get_val('Module'),
            'test_module': get_val('Test Module'),
            'dependency': get_val('Dependency'),
            'pr': get_val('PR'),
            'pr_owner': get_val('PR Owner'),
            'pr_status': get_val('PR Status'),
            'pr_desc': get_val('PR Description'),
            'category': clean_category_name(get_val('Category')),
            'category_reason': get_val('Category Reason'),
            'priority': get_val('Priority'),
            'priority_reason': get_val('Priority Reason'),
            'root_cause': get_val('Root Cause'),
            'root_cause_reason': get_val('Root Cause Reason'),
            'owner_transfer': get_val('Owner Transfer'),
            'action_TBD': get_val('Action TBD'),
            'action_reason': get_val('Action Reason'),
        }
        issues.append(issue)
    
    # Group issues by action_TBD
    action_groups = {}
    for issue in issues:
        action = str(issue.get('action_TBD') or 'Unknown')
        if action not in action_groups:
            action_groups[action] = []
        action_groups[action].append(issue)
    
    # Separate Need Investigation by category
    need_inv_issues = action_groups.get('Need Investigation', [])
    need_inv_by_category = {}
    for issue in need_inv_issues:
        cat = clean_category_name(issue.get('category') or 'unknown')
        if cat not in need_inv_by_category:
            need_inv_by_category[cat] = []
        need_inv_by_category[cat].append(issue)
    
    # Other actions (non-Need Investigation)
    other_action_groups = {}
    for action, issues_list in action_groups.items():
        if action != 'Need Investigation':
            other_action_groups[action] = issues_list
    
    # Build statistics
    test_module_stats = {}
    module_stats = {}
    dep_stats = {}
    priority_stats = {}
    category_stats = {}
    
    for issue in issues:
        tm = issue.get('test_module') or 'unknown'
        test_module_stats[tm] = test_module_stats.get(tm, 0) + 1
        
        m = issue.get('module') or 'unknown'
        module_stats[m] = module_stats.get(m, 0) + 1
        
        dep = issue.get('dependency')
        if dep and dep != 'None':
            dep_stats[dep] = dep_stats.get(dep, 0) + 1
        
        cat = clean_category_name(issue.get('category') or 'unknown')
        category_stats[cat] = category_stats.get(cat, 0) + 1
        
        p = issue.get('priority') or 'unknown'
        priority_stats[p] = priority_stats.get(p, 0) + 1
    
    # Build markdown
    md = ""
    md += f"# Torch XPU Ops Issue Report\n\n"
    md += "**Repository:** [intel/torch-xpu-ops](https://github.com/intel/torch-xpu-ops)\n\n"
    md += "**CI Data Sources:**\n"
    md += "- Torch-XPU-OPS Nightly CI: XML files from `Inductor-XPU-UT-Data-*` + `Inductor-wheel-nightly-LTS2-XPU-E2E-Data-*`\n"
    md += "- Stock PyTorch XPU CI: XML files from `test-default-*-linux.idc.xpu_*.zip`\n\n"
    md += f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    md += f"**Total Issues:** {len(issues)}\n\n"
    md += "---\n\n"
    
    # Index
    md += "## Index\n\n"
    md += "1. [Summary](#1-summary)\n"
    md += "2. [Need Investigation by Category](#2-need-investigation-by-category)\n"
    md += "3. [Other Actions by Type](#3-other-actions-by-type)\n"
    md += "4. [Statistics](#4-statistics)\n\n"
    md += "---\n\n"
    
    # 1. Summary
    md += "## 1. Summary\n\n"
    md += "| Action Type | Count |\n"
    md += "|-------------|-------|\n"
    total = len(issues)
    need_inv_count = len(need_inv_issues)
    for action, issues_list in sorted(action_groups.items(), key=lambda x: -len(x[1])):
        md += f"| {action} | {len(issues_list)} |\n"
    md += f"| **Total** | **{total}** |\n\n"
    
    # 2. Need Investigation by Category
    md += "## 2. Need Investigation by Category\n\n"
    md += f"Total Need Investigation issues: {need_inv_count}\n\n"
    
    for cat in sorted(need_inv_by_category.keys(), key=lambda x: -len(need_inv_by_category[x])):
        cat_issues = need_inv_by_category[cat]
        cat_issues.sort(key=lambda x: x.get('id', 0))
        md += f"### {cat} ({len(cat_issues)} issues)\n\n"
        md += "| ID | Title | Summary | Assignee | Owner Transfer | PR Status | Test Module | Action Reason |\n"
        md += "|---|-------|---------|----------|----------------|-----------|-------------|---------------|\n"
        
        for issue in cat_issues:
            issue_id = issue.get('id') or ''
            title = issue.get('title') or ''
            summary = issue.get('summary') or ''
            assignee = issue.get('assignee') or ''
            owner_transfer = issue.get('owner_transfer') or ''
            pr_status = issue.get('pr_status') or ''
            test_module = issue.get('test_module') or ''
            action_reason = issue.get('action_reason') or ''
            
            md += f"| [{issue_id}](https://github.com/intel/torch-xpu-ops/issues/{issue_id}) | {title} | {summary} | {assignee} | {owner_transfer} | {pr_status} | {test_module} | {action_reason} |\n"
        md += "\n"
    
    # 3. Other Actions by Type
    md += "## 3. Other Actions by Type\n\n"
    
    action_order = ['Close fixed issue', 'Verify the issue', 'Enable test', 'add to skiplist', 'Revisit the PR as case failed']
    ordered_actions = []
    for a in action_order:
        if a in other_action_groups:
            ordered_actions.append(a)
    for a in sorted(other_action_groups.keys()):
        if a not in ordered_actions:
            ordered_actions.append(a)
    
    for action in ordered_actions:
        issues_list = other_action_groups.get(action, [])
        if not issues_list:
            continue
        
        issues_list.sort(key=lambda x: x.get('id', 0))
        
        md += f"### {action} ({len(issues_list)} issues)\n\n"
        md += "| ID | Title | Summary | Assignee | Owner Transfer | PR Status | Test Module | Action Reason |\n"
        md += "|---|-------|---------|----------|----------------|-----------|-------------|---------------|\n"
        
        for issue in issues_list:
            issue_id = issue.get('id') or ''
            title = issue.get('title') or ''
            summary = issue.get('summary') or ''
            assignee = issue.get('assignee') or ''
            owner_transfer = issue.get('owner_transfer') or ''
            pr_status = issue.get('pr_status') or ''
            test_module = issue.get('test_module') or ''
            action_reason = issue.get('action_reason') or ''
            
            md += f"| [{issue_id}](https://github.com/intel/torch-xpu-ops/issues/{issue_id}) | {title} | {summary} | {assignee} | {owner_transfer} | {pr_status} | {test_module} | {action_reason} |\n"
        md += "\n"
    
    # 4. Statistics
    md += "## 4. Statistics\n\n"
    
    md += "### By Test Module\n\n"
    md += "| Test Module | Count |\n"
    md += "|-------------|-------|\n"
    for tm, count in sorted(test_module_stats.items(), key=lambda x: -x[1]):
        md += f"| {tm} | {count} |\n"
    md += "\n"
    
    md += "### By Module\n\n"
    md += "| Module | Count |\n"
    md += "|--------|-------|\n"
    for m, count in sorted(module_stats.items(), key=lambda x: -x[1]):
        md += f"| {m} | {count} |\n"
    md += "\n"
    
    md += "### By Category (Need Investigation Only)\n\n"
    md += "| Category | Count |\n"
    md += "|----------|-------|\n"
    for cat, count in sorted(category_stats.items(), key=lambda x: -x[1]):
        md += f"| {cat} | {count} |\n"
    md += "\n"
    
    md += "### By Dependency\n\n"
    md += "| Dependency | Count |\n"
    md += "|------------|-------|\n"
    for dep, count in sorted(dep_stats.items(), key=lambda x: -x[1]):
        md += f"| {dep} | {count} |\n"
    md += "\n"
    
    md += "### By Priority\n\n"
    md += "| Priority | Count |\n"
    md += "|----------|-------|\n"
    for p, count in sorted(priority_stats.items()):
        md += f"| {p} | {count} |\n"
    md += "\n"
    
    # Save to file
    output_path = os.path.join(RESULT_DIR, 'issue_report.md')
    with open(output_path, 'w') as f:
        f.write(md)
    
    print(f"\nReport saved to: {output_path}")
    print(f"Total issues: {total}")
    print(f"  Need Investigation: {need_inv_count}")
    
    # Print action summary
    first = True
    for action, issues_list in sorted(action_groups.items(), key=lambda x: -len(x[1])):
        if first:
            first = False
        else:
            print(f"  {action}: {len(issues_list)}")
    
    # Print Need Investigation by category
    print("\nNeed Investigation by Category:")
    for cat in sorted(need_inv_by_category.keys(), key=lambda x: -len(need_inv_by_category[x])):
        print(f"  {cat}: {len(need_inv_by_category[cat])}")


if __name__ == '__main__':
    generate_report()