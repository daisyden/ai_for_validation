#!/usr/bin/env python3
"""
Generate issue_report.md

Creates a comprehensive Markdown report with:
1. Summary
2. Action Required (by Action TBD)
3. Issues by Category
4. Last Week Issues
5. Stale Issues
6. Dependency Issues
7. Duplicated Issues
8. Statistics

Each table includes: Priority, Priority Reason, Action Reason columns
"""

import os
import sys
from datetime import datetime, timedelta
import openpyxl

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
RESULT_DIR = os.environ.get('RESULT_DIR', os.path.join(ROOT_DIR, 'result'))

# Priority mapping from SKILL.md
PRIORITY_MAP = {
    'add to skiplist': 1,
    'Close fixed issue': 2,
    'Verify the issue': 3,
    'No Test Status in CI': 4,
    'Needs Upstream Skip PR': 5,
    'Needs Skip PR': 6,
    'E2E accuracy issue': 7,
    'Awaiting response': 8,
    'Awaiting response from reporter': 9,
}

PRIORITY_REASON_MAP = {
    1: 'Issues marked as not_target/wontfix or cannot be enabled',
    2: 'All test cases passed on both XPU and stock',
    3: 'PR exists but no failures',
    4: 'No test status available - needs testing',
    5: 'Issue is upstream - needs skip PR upstream',
    6: 'Issue marked as wontfix/not_target - needs skip PR',
    7: 'E2E accuracy issue pending - needs upstream investigation',
    8: 'Bug/Perf issue pending reporter response',
    9: 'Maintainer requested info from reporter',
    'N': 'Fallback - no specific action identified',
}


def get_priority(issue_row: dict) -> int:
    """Get priority number based on Action TBD."""
    action_tbd = issue_row.get('Action TBD', '')
    if action_tbd in PRIORITY_MAP:
        return PRIORITY_MAP[action_tbd]
    return 99  # Will be mapped to 'N' later


def get_priority_reason(priority: int) -> str:
    """Get priority reason based on priority number."""
    return PRIORITY_REASON_MAP.get(priority, PRIORITY_REASON_MAP['N'])


def format_date(date_val) -> str:
    """Format date value to string."""
    if date_val is None:
        return ''
    if isinstance(date_val, datetime):
        return date_val.strftime('%Y-%m-%d')
    if isinstance(date_val, str):
        return date_val[:10] if len(str(date_val)) >= 10 else str(date_val)
    return str(date_val)[:10] if len(str(date_val)) >= 10 else str(date_val)


def truncate(text: str, max_len: int = 80) -> str:
    """Return full text without truncation for cell wrapping."""
    if text is None:
        return ''
    return str(text)


def load_issues(excel_file: str) -> list:
    """Load issues from Excel."""
    print(f"Loading: {excel_file}")
    wb = openpyxl.load_workbook(excel_file)
    ws = wb['Issues']
    
    # Get headers
    headers = [ws.cell(1, col).value for col in range(1, ws.max_column + 1)]
    
    # Build issue list
    issues = []
    for row in range(2, ws.max_row + 1):
        issue = {}
        for col, header in enumerate(headers, 1):
            issue[header] = ws.cell(row, col).value
        issues.append(issue)
    
    print(f"Loaded {len(issues)} issues")
    return issues


def build_action_tbd_order() -> list:
    """Define action TBD order."""
    return [
        'No Test Status in CI',
        'Needs Upstream Skip PR',
        'Awaiting response',
        'Awaiting response from reporter',
        'E2E accuracy issue',
        'Need Investigation',
        'add to skiplist',
        'Close fixed issue',
        'Verify the issue',
    ]


def build_categories() -> list:
    """Define all categories."""
    return [
        'unknown',
        'Torch Operations',
        'Dtype/Precision',
        'Others',
        'Inductor/Compilation',
        'TorchAO',
        'Flash Attention/Transformer',
        'Sparse',
        'Feature Not Supported',
        'Distributed',
        'Torch Runtime',
        'Skip/No Test Exists',
        'Build/Compilation',
        'Performance',
        'PT2E',
        'Accuracy',
        'Profiler',
    ]


def generate_table_header() -> str:
    """Generate standard table header."""
    return '| # | ID | Title | Priority | Priority Reason | Action Reason | Owner Transfer | Category | Test Module |'


def generate_table_header2() -> str:
    """Generate standard table header with Action TBD column."""
    return '| # | ID | Title | Priority | Priority Reason | Action Reason | Owner Transfer | Action TBD | Category | Test Module |'


def generate_table_separator() -> str:
    """Generate table separator."""
    return '|--:|----|-------|----------|-----------------|---------------|----------------|-----------|------------|'


def generate_table_separator2() -> str:
    """Generate table separator with Action TBD column."""
    return '|--:|----|-------|----------|-----------------|---------------|----------------|-----------|------------|------------|'


def generate_table_row(idx: int, issue: dict, show_action: bool = False) -> str:
    """Generate a table row."""
    priority = get_priority(issue)
    priority_display = priority if priority < 99 else 'N'
    priority_reason = get_priority_reason(priority)
    
    row = [
        idx,
        issue.get('Issue ID', ''),
        truncate(issue.get('Title', ''), 60),
        priority_display,
        truncate(priority_reason, 40),
        truncate(issue.get('Action Reason', ''), 40),
        issue.get('Owner Transfer', ''),
    ]
    
    if show_action:
        row.extend([
            issue.get('Action TBD', ''),
        ])
    
    row.extend([
        issue.get('Category', ''),
        issue.get('Test Module', ''),
    ])
    
    return '| ' + ' | '.join(str(x) for x in row) + ' |'


def generate_summary_section(issues: list) -> str:
    """Generate section 1: Summary."""
    lines = []
    lines.append('## <span id=\'1-summary\'>1. Summary</span>\n')
    
    total = len(issues)
    lines.append(f'**Total Issues: {total}**\n')
    
    # Action TBD summary
    action_counts = {}
    for issue in issues:
        action = issue.get('Action TBD', '') or 'Need Investigation'
        action_counts[action] = action_counts.get(action, 0) + 1
    
    lines.append('| # | Action TBD | Count | Link |')
    lines.append('|--:|------------|-------:|------|')
    
    ordered_actions = build_action_tbd_order()
    for idx, action in enumerate(ordered_actions, 1):
        count = action_counts.get(action, 0)
        anchor = action.lower().replace(' ', '-').replace('/', '-')
        lines.append(f'| {idx} | [{action}](#{anchor}) | {count} | [View Issues](#{anchor}) |')
    
    lines.append('')
    return '\n'.join(lines)


def generate_action_required_section(issues: list) -> str:
    """Generate section 2: Action Required."""
    lines = []
    lines.append('## <span id=\'2-action-required\'>2. Action Required</span>\n')
    
    # Group by Action TBD
    by_action = {}
    for issue in issues:
        action = issue.get('Action TBD', '') or 'Need Investigation'
        if action not in by_action:
            by_action[action] = []
        by_action[action].append(issue)
    
    # Sort by ID descending
    for action in by_action:
        by_action[action].sort(key=lambda x: x.get('Issue ID', 0), reverse=True)
    
    # Build tables
    ordered_actions = build_action_tbd_order()
    for idx, action in enumerate(ordered_actions, 1):
        if action not in by_action or not by_action[action]:
            continue
        
        anchor = action.lower().replace(' ', '-').replace('/', '-')
        issue_list = by_action[action]
        count = len(issue_list)
        
        lines.append(f'### <span id=\'{anchor}\'>{idx}. {action}</span> ({count} issues)\n')
        lines.append(generate_table_header2())
        lines.append(generate_table_separator2())
        
        for row_idx, issue in enumerate(issue_list, 1):
            lines.append(generate_table_row(row_idx, issue, show_action=True))
        
        lines.append('')
    
    return '\n'.join(lines)


def generate_category_section(issues: list) -> str:
    """Generate section 3: Issues by Category."""
    lines = []
    lines.append('## <span id=\'3-issues-by-category\'>3. Issues by Category</span>\n')
    
    # Group by Category
    by_category = {}
    for issue in issues:
        cat = issue.get('Category', 'unknown') or 'unknown'
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(issue)
    
    # Build category list
    categories = build_categories()
    
    # Add any categories not in predefined list
    for cat in by_category:
        if cat not in categories:
            categories.append(cat)
    
    # Generate tables for each category
    for cat in categories:
        if cat not in by_category or not by_category[cat]:
            continue
        
        issue_list = by_category[cat]
        issue_list.sort(key=lambda x: x.get('Issue ID', 0), reverse=True)
        count = len(issue_list)
        
        anchor = cat.lower().replace(' ', '-').replace('/', '-')
        lines.append(f'### <span id=\'{anchor}\'>{cat}</span> ({count} issues)\n')
        lines.append(generate_table_header())
        lines.append(generate_table_separator())
        
        for row_idx, issue in enumerate(issue_list, 1):
            lines.append(generate_table_row(row_idx, issue))
        
        lines.append('')
    
    return '\n'.join(lines)


def generate_last_week_section(issues: list) -> str:
    """Generate section 4: Last Week Issues."""
    lines = []
    lines.append('## <span id=\'4-last-week-issues\'>4. Last Week Issues</span>\n')
    
    # Calculate date 7 days ago
    seven_days_ago = datetime.now() - timedelta(days=7)
    
    # Filter issues from last week
    last_week = []
    for issue in issues:
        created = issue.get('Created Time')
        if created:
            if isinstance(created, datetime):
                if created >= seven_days_ago:
                    last_week.append(issue)
            elif isinstance(created, str) and len(created) >= 10:
                try:
                    created_dt = datetime.strptime(created[:10], '%Y-%m-%d')
                    if created_dt >= seven_days_ago:
                        last_week.append(issue)
                except ValueError:
                    pass
    
    # Sort by ID descending
    last_week.sort(key=lambda x: x.get('Issue ID', 0), reverse=True)
    count = len(last_week)
    
    lines.append(f'**Issues reported in last 7 days: {count}**\n')
    
    if count > 0:
        lines.append('| # | ID | Title | Priority | Action Reason | Category | Created Time |')
        lines.append('|--:|----|-------|----------|---------------|----------|--------------|')
        
        for idx, issue in enumerate(last_week, 1):
            priority = get_priority(issue)
            priority_display = priority if priority < 99 else 'N'
            created = format_date(issue.get('Created Time'))
            
            lines.append(f"| {idx} | {issue.get('Issue ID', '')} | {truncate(issue.get('Title', ''), 50)} | {priority_display} | {truncate(issue.get('Action Reason', ''), 35)} | {issue.get('Category', '')} | {created} |")
    else:
        lines.append('No issues reported in the last 7 days.\n')
    
    lines.append('')
    return '\n'.join(lines)


def generate_stale_section(issues: list) -> str:
    """Generate section 5: Stale Issues."""
    lines = []
    lines.append('## <span id=\'5-stale-issues\'>5. Stale Issues - No Update 2+ Weeks</span>\n')
    
    # Calculate date 14 days ago
    fourteen_days_ago = datetime.now() - timedelta(days=14)
    
    # Filter stale, open issues
    stale = []
    for issue in issues:
        # Skip closed issues
        status = str(issue.get('Status', '')).lower()
        if 'closed' in status:
            continue
        
        updated = issue.get('Updated Time')
        if updated:
            if isinstance(updated, datetime):
                if updated < fourteen_days_ago:
                    stale.append(issue)
            elif isinstance(updated, str) and len(updated) >= 10:
                try:
                    updated_dt = datetime.strptime(updated[:10], '%Y-%m-%d')
                    if updated_dt < fourteen_days_ago:
                        stale.append(issue)
                except ValueError:
                    pass
    
    # Sort by Updated Time ascending (oldest first)
    stale.sort(key=lambda x: x.get('Updated Time', datetime.min) if x.get('Updated Time') else datetime.min)
    count = len(stale)
    
    lines.append(f'**Issues without update for 2+ weeks (excluding closed): {count}**\n')
    
    if count > 0:
        lines.append('| # | ID | Title | Priority | Action Reason | Category | Updated Time | Days Since Update |')
        lines.append('|--:|----|-------|----------|---------------|----------|---------------|-------------------|')
        
        for idx, issue in enumerate(stale, 1):
            priority = get_priority(issue)
            priority_display = priority if priority < 99 else 'N'
            updated = format_date(issue.get('Updated Time'))
            
            # Calculate days since update
            try:
                if isinstance(issue.get('Updated Time'), datetime):
                    days = (datetime.now() - issue.get('Updated Time')).days
                elif isinstance(issue.get('Updated Time'), str):
                    days = (datetime.now() - datetime.strptime(issue.get('Updated Time')[:10], '%Y-%m-%d')).days
                else:
                    days = 'N/A'
            except:
                days = 'N/A'
            
            lines.append(f"| {idx} | {issue.get('Issue ID', '')} | {truncate(issue.get('Title', ''), 45)} | {priority_display} | {truncate(issue.get('Action Reason', ''), 30)} | {issue.get('Category', '')} | {updated} | {days} |")
    else:
        lines.append('No stale issues found.\n')
    
    lines.append('')
    return '\n'.join(lines)


def generate_dependency_section(issues: list) -> str:
    """Generate section 6: Dependency Issues."""
    lines = []
    lines.append('## <span id=\'6-dependency-issues\'>6. Dependency Issues</span>\n')
    
    # Filter issues with dependencies
    deps = []
    for issue in issues:
        dep = issue.get('Dependency', '')
        if dep and str(dep).strip():
            deps.append(issue)
    
    # Sort by ID descending
    deps.sort(key=lambda x: x.get('Issue ID', 0), reverse=True)
    count = len(deps)
    
    lines.append(f'**Issues with dependencies: {count}**\n')
    
    if count > 0:
        lines.append('| # | ID | Title | Priority | Dependency | Category |')
        lines.append('|--:|----|-------|----------|------------|----------|')
        
        for idx, issue in enumerate(deps, 1):
            priority = get_priority(issue)
            priority_display = priority if priority < 99 else 'N'
            
            lines.append(f"| {idx} | {issue.get('Issue ID', '')} | {truncate(issue.get('Title', ''), 50)} | {priority_display} | {truncate(issue.get('Dependency', ''), 40)} | {issue.get('Category', '')} |")
    else:
        lines.append('No issues with dependencies.\n')
    
    lines.append('')
    return '\n'.join(lines)


def generate_duplicated_section(issues: list) -> str:
    """Generate section 7: Duplicated Issues."""
    lines = []
    lines.append('## <span id=\'7-duplicated-issues\'>7. Duplicated Issues</span>\n')
    
    # Filter duplicated issues
    dupes = []
    for issue in issues:
        dupe = issue.get('duplicated_issue', '')
        if dupe and str(dupe).strip():
            dupes.append(issue)
    
    # Sort by ID descending
    dupes.sort(key=lambda x: x.get('Issue ID', 0), reverse=True)
    count = len(dupes)
    
    lines.append(f'**Issues marked as duplicated: {count}**\n')
    
    if count > 0:
        lines.append('| # | ID | Title | Priority | Duplicated Issue | Category |')
        lines.append('|--:|----|-------|----------|-----------------|----------|')
        
        for idx, issue in enumerate(dupes, 1):
            priority = get_priority(issue)
            priority_display = priority if priority < 99 else 'N'
            
            lines.append(f"| {idx} | {issue.get('Issue ID', '')} | {truncate(issue.get('Title', ''), 45)} | {priority_display} | {truncate(issue.get('duplicated_issue', ''), 40)} | {issue.get('Category', '')} |")
    else:
        lines.append('No duplicated issues.\n')
    
    lines.append('')
    return '\n'.join(lines)


def generate_statistics_section(issues: list) -> str:
    """Generate section 8: Statistics."""
    lines = []
    lines.append('## <span id=\'8-statistics\'>8. Statistics</span>\n')
    
    # Action TBD counts
    lines.append('### Action TBD Distribution\n')
    action_counts = {}
    for issue in issues:
        action = issue.get('Action TBD', '') or 'Need Investigation'
        action_counts[action] = action_counts.get(action, 0) + 1
    
    lines.append('| Action TBD | Count |')
    lines.append('|------------|------:|')
    for action in build_action_tbd_order():
        if action in action_counts:
            lines.append(f'| {action} | {action_counts[action]} |')
    
    # Category counts
    lines.append('\n### Category Distribution\n')
    cat_counts = {}
    for issue in issues:
        cat = issue.get('Category', 'unknown') or 'unknown'
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
    
    lines.append('| Category | Count |')
    lines.append('|----------|------:|')
    for cat in sorted(cat_counts.keys()):
        lines.append(f'| {cat} | {cat_counts[cat]} |')
    
    # Test Module counts
    lines.append('\n### Test Module Distribution\n')
    module_counts = {}
    for issue in issues:
        module = issue.get('Test Module', 'unknown') or 'unknown'
        module_counts[module] = module_counts.get(module, 0) + 1
    
    lines.append('| Test Module | Count |')
    lines.append('|-------------|------:|')
    for module in sorted(module_counts.keys()):
        lines.append(f'| {module} | {module_counts[module]} |')
    
    lines.append('')
    return '\n'.join(lines)


def generate_index(issues: list) -> str:
    """Generate index/TOC."""
    lines = []
    lines.append('## Index\n')
    
    # Get counts
    action_counts = {}
    for issue in issues:
        action = issue.get('Action TBD', '') or 'Need Investigation'
        action_counts[action] = action_counts.get(action, 0) + 1
    
    total_action = sum(action_counts.values())
    
    lines.append(f'- [1. Summary](#1-summary) - {total_action} issues |')
    
    # Action Required
    idx = 1
    lines.append('- [2. Action Required](#2-action-required)')
    for action in build_action_tbd_order():
        count = action_counts.get(action, 0)
        anchor = action.lower().replace(' ', '-').replace('/', '-')
        lines.append(f'  - [{idx}. {action}](#{anchor}) - {count} issues |')
        idx += 1
    
    # Category count
    cat_counts = {}
    for issue in issues:
        cat = issue.get('Category', 'unknown') or 'unknown'
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
    total_cat = sum(cat_counts.values())
    lines.append(f'- [3. Issues by Category](#3-issues-by-category) - {total_cat} issues |')
    
    # Last week
    seven_days_ago = datetime.now() - timedelta(days=7)
    last_week_count = 0
    for issue in issues:
        created = issue.get('Created Time')
        if created and isinstance(created, datetime):
            if created >= seven_days_ago:
                last_week_count += 1
        elif isinstance(created, str) and len(created) >= 10:
            try:
                if datetime.strptime(created[:10], '%Y-%m-%d') >= seven_days_ago:
                    last_week_count += 1
            except ValueError:
                pass
    lines.append(f'- [4. Last Week Issues](#4-last-week-issues) - {last_week_count} issues |')
    
    # Stale
    stale_count = 0
    fourteen_days_ago = datetime.now() - timedelta(days=14)
    for issue in issues:
        status = str(issue.get('Status', '')).lower()
        if 'closed' in status:
            continue
        updated = issue.get('Updated Time')
        if updated and isinstance(updated, datetime):
            if updated < fourteen_days_ago:
                stale_count += 1
        elif isinstance(updated, str) and len(updated) >= 10:
            try:
                if datetime.strptime(updated[:10], '%Y-%m-%d') < fourteen_days_ago:
                    stale_count += 1
            except ValueError:
                pass
    lines.append(f'- [5. Stale Issues - No Update 2+ Weeks](#5-stale-issues) - {stale_count} issues |')
    
    # Dependency
    dep_count = sum(1 for issue in issues if issue.get('Dependency', '') and str(issue.get('Dependency', '')).strip())
    lines.append(f'- [6. Dependency Issues](#6-dependency-issues) - {dep_count} issues |')
    
    # Duplicated
    dupe_count = sum(1 for issue in issues if issue.get('duplicated_issue', '') and str(issue.get('duplicated_issue', '')).strip())
    lines.append(f'- [7. Duplicated Issues](#7-duplicated-issues) - {dupe_count} issues |')
    
    lines.append('- [8. Statistics](#8-statistics)\n')
    
    return '\n'.join(lines)


def generate_report(excel_file: str, output_file: str):
    """Generate the complete issue report."""
    print(f"\nGenerating issue report...")
    print(f"Excel: {excel_file}")
    print(f"Output: {output_file}")
    
    # Load issues
    issues = load_issues(excel_file)
    
    # Generate sections
    print("Generating sections...")
    
    lines = []
    lines.append('# Torch XPU Ops Issue Report\n')
    lines.append(f'**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    lines.append(f'**Total Issues:** {len(issues)}\n')
    lines.append('---\n')
    
    lines.append(generate_index(issues))
    lines.append('\n---\n')
    lines.append(generate_summary_section(issues))
    lines.append('\n---\n')
    lines.append(generate_action_required_section(issues))
    lines.append('\n---\n')
    lines.append(generate_category_section(issues))
    lines.append('\n---\n')
    lines.append(generate_last_week_section(issues))
    lines.append(generate_stale_section(issues))
    lines.append(generate_dependency_section(issues))
    lines.append(generate_duplicated_section(issues))
    lines.append(generate_statistics_section(issues))
    
    # Write to file
    content = '\n'.join(lines)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"\nReport saved to: {output_file}")
    print(f"Report size: {len(content)} characters")


def main():
    """Main entry point."""
    excel_file = os.environ.get('ISSUE_EXCEL', os.path.join(RESULT_DIR, 'torch_xpu_ops_issues.xlsx'))
    excel_file = sys.argv[1] if len(sys.argv) > 1 else excel_file
    
    output_file = os.environ.get('ISSUE_REPORT', os.path.join(RESULT_DIR, 'issue_report.md'))
    output_file = sys.argv[2] if len(sys.argv) > 2 else output_file
    
    if not os.path.exists(excel_file):
        print(f"ERROR: Excel file not found: {excel_file}")
        return 1
    
    generate_report(excel_file, output_file)
    return 0


if __name__ == '__main__':
    sys.exit(main())