#!/usr/bin/env python3
"""
Generate markdown report from torch_xpu_ops_issues.xlsx
Reads Excel columns by header names, not hardcoded positions.
Filters out 'enhancement' labeled issues.
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


def slugify(text):
    """Create URL-friendly anchor from text."""
    text = str(text).lower().replace('/', '-').replace('&', 'and').replace(' ', '-')
    text = re.sub(r'[^a-z0-9\s\-]', '', text)
    text = re.sub(r'\s+', '-', text)
    return text


def extract_action_type(action_reason):
    """Extract action type prefix from action reason (before first colon)."""
    if not action_reason:
        return 'Unknown Action'
    reason = str(action_reason).strip()
    if ':' in reason:
        return reason.split(':')[0].strip()
    return 'Unknown Action'


def generate_report():
    wb = openpyxl.load_workbook(os.path.join(RESULT_DIR, 'torch_xpu_ops_issues.xlsx'))
    ws_issues = wb['Issues']
    
    issue_cols = get_all_columns_by_header(ws_issues)
    print(f"Available columns: {list(issue_cols.keys())}")
    
    issues = []
    enhancement_count = 0
    for row in range(2, ws_issues.max_row + 1):
        def get_val(header, default=None):
            col = issue_cols.get(header)
            return ws_issues.cell(row, col).value if col else default
        
        labels = get_val('Labels') or ''
        if 'enhancement' in str(labels).lower():
            enhancement_count += 1
            continue
        
        issue = {
            'id': get_val('Issue ID'),
            'title': get_val('Title'),
            'summary': get_val('Summary'),
            'status': get_val('Status'),
            'assignee': get_val('Assignee'),
            'reporter': get_val('Reporter'),
            'labels': labels,
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
            'action_type': extract_action_type(get_val('Action Reason')),
            'duplicated_issue': get_val('duplicated_issue'),
        }
        issues.append(issue)
    
    action_groups = {}
    for issue in issues:
        action = str(issue.get('action_TBD') or 'Unknown')
        if action not in action_groups:
            action_groups[action] = []
        action_groups[action].append(issue)
    
    need_inv_issues = action_groups.get('Need Investigation', [])
    need_inv_by_category = {}
    need_inv_by_action_type = {}
    for issue in need_inv_issues:
        cat = clean_category_name(issue.get('category') or 'unknown')
        if cat not in need_inv_by_category:
            need_inv_by_category[cat] = []
        need_inv_by_category[cat].append(issue)
        
        # For 'unknown' category, also group by action_type
        if cat == 'unknown':
            action_type = issue.get('action_type', 'Unknown Action')
            if action_type not in need_inv_by_action_type:
                need_inv_by_action_type[action_type] = []
            need_inv_by_action_type[action_type].append(issue)
    
    other_action_groups = {}
    for action, issues_list in action_groups.items():
        if action != 'Need Investigation':
            other_action_groups[action] = issues_list
    
    with_dependency = [i for i in issues if i.get('dependency') and str(i.get('dependency', '')).strip() not in ['None', '', None]]
    duplicated = [i for i in issues if i.get('duplicated_issue') is not None and str(i.get('duplicated_issue', '')).strip()]
    
    dep_stats = {}
    for issue in issues:
        dep = issue.get('dependency')
        if dep and dep != 'None':
            dep_stats[dep] = dep_stats.get(dep, 0) + 1
    
    md = ""
    md += f"# Torch XPU Ops Issue Report\n\n"
    md += "**Repository:** [intel/torch-xpu-ops](https://github.com/intel/torch-xpu-ops)\n\n"
    md += "**CI Data Sources:**\n"
    md += "- Torch-XPU-OPS Nightly CI: XML files from `Inductor-XPU-UT-Data-*` + `Inductor-wheel-nightly-LTS2-XPU-E2E-Data-*`\n"
    md += "- Stock PyTorch XPU CI: XML files from `test-default-*-linux.idc.xpu_*.zip`\n\n"
    md += f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    md += f"**Total Issues:** {len(issues)} (excluded {enhancement_count} enhancement issues)\n\n"
    md += "---\n\n"
    
    total = len(issues)
    need_inv_count = len(need_inv_issues)
    dup_count = len(duplicated)
    dep_count = len(with_dependency)
    other_count = total - need_inv_count
    
    md += "## <span id='toc'>Index</span>\n\n"
    md += f"- [1. Summary (#1-summary)](#1-summary) - {total} issues |\n"
    md += f"- [2. Need Investigation by Category (#2-need-investigation-by-category)](#2-need-investigation-by-category) - {need_inv_count} issues |\n"
    for cat in sorted(need_inv_by_category.keys(), key=lambda x: -len(need_inv_by_category[x])):
        anchor = slugify(cat)
        md += f"   - [{cat}](#{anchor}) - {len(need_inv_by_category[cat])} issues |\n"
        if cat == 'unknown':
            for action_type in sorted(need_inv_by_action_type.keys(), key=lambda x: -len(need_inv_by_action_type[x])):
                type_anchor = f"{anchor}-{slugify(action_type)}"
                md += f"      - [{action_type}](#{type_anchor}) - {len(need_inv_by_action_type[action_type])} issues |\n"
    md += f"- [3. Other Actions by Type (#3-other-actions-by-type)](#3-other-actions-by-type) - {other_count} issues |\n"
    for action in other_action_groups:
        anchor = slugify(action)
        md += f"   - [{action}](#{anchor}) - {len(other_action_groups[action])} issues |\n"
    md += f"- [4. Duplicated Issues (#4-duplicated-issues)](#4-duplicated-issues) - {dup_count} issues |\n"
    md += f"- [5. Issues with Dependency (#5-issues-with-dependency)](#5-issues-with-dependency) - {dep_count} issues |\n"
    md += f"- [6. Statistics (#6-statistics)](#6-statistics) - Dependency stats |\n\n"
    
    md += "---\n\n"
    
    md += "## <span id='1-summary'>1. Summary</span>\n\n"
    md += f"**Total: {total} issues** (excluded {enhancement_count} enhancement issues)\n\n"
    md += "| # | Action Type | Count | Link |\n"
    md += "|--:|-------------|-------|------|\n"
    idx = 1
    for action, issues_list in sorted(action_groups.items(), key=lambda x: -len(x[1])):
        anchor = slugify(action)
        md += f"| {idx} | [{action}](#{anchor}) | {len(issues_list)} | [View Issues](#{anchor}) |\n"
        idx += 1
    md += f"| | **Total** | **{total}** | |\n\n"
    
    md += "## <span id='2-need-investigation-by-category'>2. Need Investigation by Category</span>\n\n"
    md += f"**Total: {need_inv_count} issues** - Issues requiring further investigation\n\n"
    
    thread_idx = 1
    for cat in sorted(need_inv_by_category.keys(), key=lambda x: -len(need_inv_by_category[x])):
        cat_issues = need_inv_by_category[cat]
        cat_issues.sort(key=lambda x: x.get('id', 0))
        anchor = slugify(cat)
        
        # For 'unknown' category, group by action_type
        if cat == 'unknown':
            md += f"### <span id='{anchor}'>{cat}</span> ({len(cat_issues)} issues)\n\n"
            md += f"**Grouped by Action Type:**\n\n"
            
            for action_type, type_issues in sorted(need_inv_by_action_type.items(), key=lambda x: -len(x[1])):
                type_anchor = f"{anchor}-{slugify(action_type)}"
                md += f"#### <span id='{type_anchor}'>{action_type}</span> ({len(type_issues)} issues)\n"
                md += "| # | ID | Title | Action Reason | Summary | Assignee | Owner Transfer | PR Status | Test Module |\n"
                md += "|--:|----|-------|---------------|---------|----------|----------------|-----------|-------------|\n"
                
                sub_idx = 1
                for issue in type_issues:
                    issue_id = issue.get('id') or ''
                    title = issue.get('title') or ''
                    action_reason = issue.get('action_reason') or ''
                    summary = issue.get('summary') or ''
                    assignee = issue.get('assignee') or ''
                    owner_transfer = issue.get('owner_transfer') or ''
                    pr_status = issue.get('pr_status') or ''
                    test_module = issue.get('test_module') or ''
                    
                    md += f"| {sub_idx} | {issue_id} | {title} | {action_reason} | {summary} | {assignee} | {owner_transfer} | {pr_status} | {test_module} |\n"
                    sub_idx += 1
                md += f"| | | **Subtotal: {len(type_issues)} issues** | | | | | |\n\n"
            md += f"| | | **Category Total: {len(cat_issues)} issues** | | | | | |\n\n"
        
        # For other categories, keep flat table
        else:
            md += f"### <span id='{anchor}'>{cat}</span> ({len(cat_issues)} issues)\n\n"
            md += "| # | ID | Title | Action Reason | Summary | Assignee | Owner Transfer | PR Status | Test Module |\n"
            md += "|--:|----|-------|---------------|---------|----------|----------------|-----------|-------------|\n"
            
            for issue in cat_issues:
                issue_id = issue.get('id') or ''
                title = issue.get('title') or ''
                action_reason = issue.get('action_reason') or ''
                summary = issue.get('summary') or ''
                assignee = issue.get('assignee') or ''
                owner_transfer = issue.get('owner_transfer') or ''
                pr_status = issue.get('pr_status') or ''
                test_module = issue.get('test_module') or ''
                
                md += f"| {thread_idx} | {issue_id} | {title} | {action_reason} | {summary} | {assignee} | {owner_transfer} | {pr_status} | {test_module} |\n"
                thread_idx += 1
            md += f"| | | **Subtotal: {len(cat_issues)} issues** | | | | | |\n\n"
    
    md += "[Back to Index](#toc) |\n\n"
    
    md += "## <span id='3-other-actions-by-type'>3. Other Actions by Type</span>\n\n"
    md += f"**Total: {other_count} issues** - Actions other than Need Investigation\n\n"
    
    action_order = ['Close fixed issue', 'Verify the issue', 'Enable test', 'add to skiplist', 'Revisit the PR as case failed']
    ordered_actions = []
    for a in action_order:
        if a in other_action_groups:
            ordered_actions.append(a)
    for a in sorted(other_action_groups.keys()):
        if a not in ordered_actions:
            ordered_actions.append(a)
    
    thread_idx = 1
    for action in ordered_actions:
        issues_list = other_action_groups.get(action, [])
        if not issues_list:
            continue
        
        issues_list.sort(key=lambda x: x.get('id', 0))
        anchor = slugify(action)
        md += f"### <span id='{anchor}'>{action}</span> ({len(issues_list)} issues)\n\n"
        md += "| # | ID | Title | Action Reason | Summary | Assignee | Owner Transfer | PR Status | Test Module |\n"
        md += "|--:|----|-------|---------------|---------|----------|----------------|-----------|-------------|\n"
        
        for issue in issues_list:
            issue_id = issue.get('id') or ''
            title = issue.get('title') or ''
            action_reason = issue.get('action_reason') or ''
            summary = issue.get('summary') or ''
            assignee = issue.get('assignee') or ''
            owner_transfer = issue.get('owner_transfer') or ''
            pr_status = issue.get('pr_status') or ''
            test_module = issue.get('test_module') or ''
            
            md += f"| {thread_idx} | {issue_id} | {title} | {action_reason} | {summary} | {assignee} | {owner_transfer} | {pr_status} | {test_module} |\n"
            thread_idx += 1
        md += f"| | | **Subtotal: {len(issues_list)} issues** | | | | | |\n\n"
    
    md += "[Back to Index](#toc) |\n\n"
    
    md += "## <span id='4-duplicated-issues'>4. Duplicated Issues</span>\n\n"
    md += f"**Total: {dup_count} issues** - Issues sharing test cases with other issues\n\n"
    md += "| # | ID | Title | Summary | Assignee | Priority | Root Cause | Dependency | Duplicated With | Test Module |\n"
    md += "|--:|----|-------|---------|----------|---------|-----------|-----------|----------------|-------------|\n"
    
    duplicated.sort(key=lambda x: x.get('id', 0))
    thread_idx = 1
    for issue in duplicated:
        issue_id = issue.get('id') or ''
        title = issue.get('title') or ''
        summary = issue.get('summary') or ''
        assignee = issue.get('assignee') or ''
        priority = issue.get('priority') or ''
        root_cause = issue.get('root_cause') or ''
        dependency = issue.get('dependency') or ''
        dup = issue.get('duplicated_issue') or ''
        test_module = issue.get('test_module') or ''
        
        md += f"| {thread_idx} | {issue_id} | {title} | {summary} | {assignee} | {priority} | {root_cause} | {dependency} | {dup} | {test_module} |\n"
        thread_idx += 1
    md += f"| | | **Subtotal: {thread_idx-1} issues** | | | | | | |\n\n"
    
    md += "[Back to Index](#toc) |\n\n"
    
    md += "## <span id='5-issues-with-dependency'>5. Issues with Dependency</span>\n\n"
    md += f"**Total: {dep_count} issues** - Issues with external dependencies\n\n"
    md += "| # | ID | Title | Summary | Assignee | Priority | Root Cause | Category | Dependency | PR Status | Test Module |\n"
    md += "|--:|----|-------|---------|----------|---------|-----------|----------|------------|-----------|-------------|\n"
    
    with_dependency.sort(key=lambda x: x.get('id', 0))
    thread_idx = 1
    for issue in with_dependency:
        issue_id = issue.get('id') or ''
        title = issue.get('title') or ''
        summary = issue.get('summary') or ''
        assignee = issue.get('assignee') or ''
        priority = issue.get('priority') or ''
        root_cause = issue.get('root_cause') or ''
        category = issue.get('category') or ''
        dependency = issue.get('dependency') or ''
        pr_status = issue.get('pr_status') or ''
        test_module = issue.get('test_module') or ''
        
        md += f"| {thread_idx} | {issue_id} | {title} | {summary} | {assignee} | {priority} | {root_cause} | {category} | {dependency} | {pr_status} | {test_module} |\n"
        thread_idx += 1
    md += f"| | | **Subtotal: {thread_idx-1} issues** | | | | | | | |\n\n"
    
    md += "[Back to Index](#toc) |\n\n"
    
    md += "## <span id='6-statistics'>6. Statistics</span>\n\n"
    
    md += "### <span id='stats-dependency'>By Dependency</span>\n\n"
    md += "| Dependency | Count |\n"
    md += "|------------|-------|\n"
    for dep, count in sorted(dep_stats.items(), key=lambda x: -x[1]):
        md += f"| {dep} | {count} |\n"
    md += "\n"
    
    md += "[Back to Index](#toc) |\n\n"
    
    md += "---\n"
    md += f"*Report generated with {total} issues (excluded {enhancement_count} enhancement-labeled issues)*\n"
    
    output_path = os.path.join(RESULT_DIR, 'issue_report.md')
    with open(output_path, 'w') as f:
        f.write(md)
    
    print(f"\nReport saved to: {output_path}")
    print(f"Total issues: {total} (excluded {enhancement_count} enhancement)")
    print(f"  Need Investigation: {need_inv_count}")
    print(f"  Other Actions: {other_count}")
    print(f"  Duplicated: {dup_count}")
    print(f"  With Dependency: {dep_count}")


if __name__ == '__main__':
    generate_report()