#!/usr/bin/env python3
"""
Generate UT-specific custom markdown report with only selected issues
"""

import openpyxl
from datetime import datetime
import os

ROOT_DIR = "/home/daisydeng"
RESULT_DIR = "/home/daisydeng/ai_for_validation/opencode/issue_triage/result"

# List of issue IDs to include (as provided by user)
INCLUDED_ISSUES = {
    1893, 1962, 1972, 2015, 2024, 2164, 2169, 2214, 2229, 2244, 2245, 2253, 2255, 2270,
    2283, 2285, 2287, 2295, 2301, 2309, 2329, 2376, 2436, 2442, 2482, 2512, 2531, 2532,
    2536, 2541, 2554, 2578, 2609, 2611, 2613, 2615, 2620, 2694, 2697, 2698, 2715, 2720,
    2783, 2800, 2802, 2806, 2810, 2853, 2888, 2891, 2958, 2997, 2999, 3004, 3006, 3007,
    3033, 3126, 3127, 3128, 3129, 3131, 3132, 3133, 3136, 3137, 3140, 3141, 3142, 3143,
    3163, 3166, 3170, 3177, 3187, 3238
}


def generate_report():
    wb = openpyxl.load_workbook(os.path.join(RESULT_DIR, 'torch_xpu_ops_issues.xlsx'))
    ws_issues = wb['Issues']
    
    # Collect issue data - only included IDs
    issues = []
    for row in range(2, ws_issues.max_row + 1):
        issue_id = ws_issues.cell(row, 1).value
        if issue_id not in INCLUDED_ISSUES:
            continue
            
        issue = {
            'id': issue_id,
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
            'root_cause': ws_issues.cell(row, 25).value,
        }
        issues.append(issue)
    
    # Sort by ID
    issues.sort(key=lambda x: x['id'] if x['id'] else 0)
    
    # Build statistics
    module_stats = {}
    action_tbd_stats = {}
    category_stats = {}
    priority_stats = {}
    category_groups = {}
    
    for issue in issues:
        cat = issue['category'] or 'unknown'
        if cat not in category_groups:
            category_groups[cat] = []
        category_groups[cat].append(issue)
    
    for issue in issues:
        m = issue['module'] or 'unknown'
        module_stats[m] = module_stats.get(m, 0) + 1
        
        action = issue['action_TBD']
        if action:
            action_tbd_stats[action] = action_tbd_stats.get(action, 0) + 1
        
        cat = issue['category'] or 'unknown'
        category_stats[cat] = category_stats.get(cat, 0) + 1
        
        p = issue['priority'] or 'unknown'
        priority_stats[p] = priority_stats.get(p, 0) + 1
    
    # Build markdown
    md = ""
    md += "# Torch XPU Ops UT Issue Report (Custom Filtered)\n\n"
    md += "**Repository:** [intel/torch-xpu-ops](https://github.com/intel/torch-xpu-ops)\n\n"
    md += "**Report Type:** UT (Unit Test) Issues - Custom Filtered List\n\n"
    md += f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    md += "---\n\n"
    
    # Summary
    md += "## 1. Summary {#1-summary}\n\n"
    md += f"| Total Issues | {len(issues)} |\n"
    md += "|-------------|-------------|\n\n"
    
    md += "| Count by Action TBD | |\n"
    md += "|---------------------|---|\n"
    for action, count in sorted(action_tbd_stats.items(), key=lambda x: -x[1]):
        md += f"| {action} | {count} |\n"
    md += "\n"
    
    md += "| Count by Category | |\n"
    md += "|-------------------|---|\n"
    for cat, count in sorted(category_stats.items(), key=lambda x: -x[1]):
        md += f"| {cat} | {count} |\n"
    md += "\n"
    
    md += "| Count by Priority | |\n"
    md += "|-------------------|---|\n"
    for p, count in sorted(priority_stats.items()):
        md += f"| {p} | {count} |\n"
    md += "\n"
    
    # Detailed Issue List
    md += "---\n\n"
    md += "## 2. Detailed Issue List\n\n"
    md += "| ID | Title | Owner | Priority | Category | Root Cause | Labels | Module |\n"
    md += "|---|-------|-------|---------|----------|-----------|--------|--------|\n"
    
    for issue in issues:
        issue_id = issue['id']
        title = (issue['title'] or '')[:40]
        owner = issue['assignee'] or ''
        priority = issue['priority'] or ''
        category = issue['category'] or ''
        root_cause = issue['root_cause'] or ''
        labels = issue['labels'] or ''
        labels_short = (labels[:20] + '..') if labels and len(labels) > 20 else labels or ''
        module = issue['module'] or ''
        md += f"| [{issue_id}](https://github.com/intel/torch-xpu-ops/issues/{issue_id}) | {title} | {owner} | {priority} | {category} | {root_cause} | {labels_short} | {module} |\n"
    
    # By Category
    md += "\n---\n\n"
    md += "## 3. By Category {#3-by-category}\n\n"
    
    for cat in sorted(category_groups.keys()):
        cat_issues = [i for i in issues if (i['category'] or 'unknown') == cat]
        if not cat_issues:
            continue
        cat_issues.sort(key=lambda x: x['id'] if x['id'] else 0)
        cat_count = len(cat_issues)
        cat_anchor = "#" + cat.lower().replace(" ", "-").replace("/", "-")
        md += "### " + cat + " (" + cat_anchor + ") - " + str(cat_count) + " issues\n\n"
        
        for issue in cat_issues:
            issue_id = issue['id']
            title = (issue['title'] or '')[:40]
            owner = issue['assignee'] or ''
            priority = issue['priority'] or ''
            root_cause = issue['root_cause'] or ''
            action_tbd = issue['action_TBD'] or ''
            labels = issue['labels'] or ''
            module = issue['module'] or ''
            status = issue['status'] or ''
            md += f"- [{issue_id}](https://github.com/intel/torch-xpu-ops/issues/{issue_id}): {title} | Owner: {owner} | Priority: {priority} | Action: {action_tbd}\n"
        
        md += "\n"
    
    # Save to file
    output_path = os.path.join(RESULT_DIR, 'issue_report_ut.md')
    with open(output_path, 'w') as f:
        f.write(md)
    
    print(f"UT Custom Report saved to: {output_path}")
    print(f"Total issues included: {len(issues)}")


if __name__ == '__main__':
    generate_report()