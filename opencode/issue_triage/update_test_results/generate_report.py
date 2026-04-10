#!/usr/bin/env python3
"""
Generate markdown report from torch_xpu_ops_issues.xlsx
"""

import openpyxl
from datetime import datetime, timedelta
import os

ROOT_DIR = "/home/daisydeng"
RESULT_DIR = os.environ.get("RESULT_DIR", "/home/daisydeng/ai_for_validation/opencode/issue_triage/result")


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
            'pr_owner': ws_issues.cell(row, 16).value,
            'pr_status': ws_issues.cell(row, 17).value,
            'pr_desc': ws_issues.cell(row, 18).value,
            'owner_transfer': ws_issues.cell(row, 19).value,
            'action_TBD': ws_issues.cell(row, 20).value,
            'action_TBD_reason': ws_issues.cell(row, 21).value,
            'duplicated_issue': ws_issues.cell(row, 22).value,
            'priority': ws_issues.cell(row, 23).value,
            'priority_reason': ws_issues.cell(row, 24).value,
            'category': ws_issues.cell(row, 25).value,
            'category_reason': ws_issues.cell(row, 26).value,
            'root_cause': ws_issues.cell(row, 27).value,
        }
        issues.append(issue)
    
    # Categorize issues
    action_required = []
    no_assignee = []
    duplicated = []
    with_dependency = []
    others = []
    
    for issue in issues:
        dep = issue['dependency']
        is_valid_dep = dep and dep not in ['None', None, '']
        
        if issue['duplicated_issue']:
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
    
    # Group by category for Section 7
    category_groups = {}
    for issue in issues:
        cat = issue['category'] or 'unknown'
        if cat not in category_groups:
            category_groups[cat] = []
        category_groups[cat].append(issue)
    
    # Build statistics
    test_module_stats = {}
    module_stats = {}
    dep_stats = {}
    action_tbd_stats = {}
    category_stats = {}
    no_assignee_count = len(no_assignee)
    duplicated_count = len(duplicated)
    others_count = len(others)
    with_dependency_count = len(with_dependency)
    action_required_count = len(action_required)
    
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
    
    # Priority stats
    priority_stats = {}
    for issue in issues:
        p = issue['priority'] or 'unknown'
        priority_stats[p] = priority_stats.get(p, 0) + 1
    
    # Recent issues (last 7 days)
    seven_days_ago = datetime.now() - timedelta(days=7)
    recent_issues = []
    for issue in issues:
        created_str = issue.get('created_time')
        if created_str:
            try:
                created_date = datetime.strptime(created_str.replace('Z', '+00:00'), '%Y-%m-%dT%H:%M:%S%z')
                created_date = created_date.replace(tzinfo=None)
                if created_date >= seven_days_ago:
                    recent_issues.append(issue)
            except:
                pass
    recent_issues.sort(key=lambda x: x['id'], reverse=True)
    
    # Build markdown
    md = ""
    md += f"# Torch XPU Ops Issue Report\n\n"
    md += "**Repository:** [intel/torch-xpu-ops](https://github.com/intel/torch-xpu-ops)\n\n"
    md += "**Issues Extracted:** April 5, 2026 (from GitHub issue data: `torch_xpu_ops_issues.json`)\n\n"
    md += "**CI Data Sources:**\n"
    md += "- Torch-XPU-OPS Nightly CI: XML files from `Inductor-XPU-UT-Data-*` + `Inductor-wheel-nightly-LTS2-XPU-E2E-Data-*` (commit: `a2d516a58c64f18b76880f3a77efbc02885d65af`)\n"
    md += "- Stock PyTorch XPU CI: XML files from `test-default-*-linux.idc.xpu_*.zip` (run IDs: 69741866812, 69741866834, 69741866862, 69741866911, etc.)\n\n"
    md += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    md += "---\n\n"
    
    # Index
    md += "## Index\n\n"
    md += "1. [Summary](#user-content-1-summary)\n"
    md += "2. [Statistics](#user-content-2-statistics)\n"
    md += "   - [By Test Module](#user-content-by-test-module)\n"
    md += "   - [By Module](#user-content-by-module)\n"
    md += "   - [By Dependency](#user-content-by-dependency)\n"
    md += "   - [By Action TBD](#user-content-by-action-tbd)\n"
    md += "   - [By Category](#user-content-by-category-stats)\n"
    md += "   - [By Priority](#user-content-by-priority)\n"
    md += "3. [New Submitted Issues (Past Week)](#user-content-3-new-submitted-issues-past-week)\n"
    md += "4. [Action Required](#user-content-4-action-required)\n"
    md += "   - [Reporter Actions](#user-content-reporter-actions)\n"
    md += "     - [Information Required](#user-content-information-required)\n"
    md += "     - [Close Fixed Issue](#user-content-close-fixed-issue)\n"
    md += "     - [Enable Test](#user-content-enable-test)\n"
    md += "     - [Add to Skiplist](#user-content-add-to-skiplist)\n"
    md += "     - [Verify the Issue](#user-content-verify-the-issue)\n"
    md += "     - [Need Reproduce Steps](#user-content-need-reproduce-steps)\n"
    md += "   - [Engineer Actions](#user-content-engineer-actions)\n"
    md += "     - [Needs PyTorch Repo Changes (upstream)](#user-content-needs-pytorch-repo-changes-upstream)\n"
    md += "     - [Revisit the PR as Case Failed](#user-content-revisit-the-pr-as-case-failed)\n"
    md += "5. [By Category](#user-content-5-by-category)\n"
    md += "6. [Duplicated Issues](#user-content-6-duplicated-issues)\n"
    md += "7. [Issues with Dependency](#user-content-7-issues-with-dependency)\n\n"
    md += "---\n\n"
    
    # 1. Summary
    md += "## 1. Summary\n\n"
    md += "| Category | Count |\n"
    md += "|----------|-------|\n"
    md += f"| Action Required | {action_required_count} |\n"
    md += f"| No Assignee | {no_assignee_count} |\n"
    md += f"| Duplicated Issues | {duplicated_count} |\n"
    md += f"| With Dependency | {with_dependency_count} |\n"
    md += f"| Others | {others_count} |\n"
    md += f"| **Total** | {len(issues)} |\n\n"
    md += "---\n\n"
    
    # 2. Statistics
    md += "## 2. Statistics\n\n"
    
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
    
    md += "### By Dependency\n\n"
    md += "| Dependency | Count |\n"
    md += "|------------|-------|\n"
    for dep, count in sorted(dep_stats.items(), key=lambda x: -x[1]):
        md += f"| {dep} | {count} |\n"
    md += "\n"
    
    md += "### By Action TBD\n\n"
    md += "| Action TBD | Count |\n"
    md += "|------------|-------|\n"
    for action, count in sorted(action_tbd_stats.items(), key=lambda x: -x[1]):
        md += f"| {action} | {count} |\n"
    md += "\n"
    
    md += "### By Category (Statistics)\n\n"
    md += "| Category | Count |\n"
    md += "|----------|-------|\n"
    for cat, count in sorted(category_stats.items(), key=lambda x: -x[1]):
        md += f"| {cat} | {count} |\n"
    md += "\n"
    
    md += "### By Priority\n\n"
    md += "| Priority | Count |\n"
    md += "|----------|-------|\n"
    for p, count in sorted(priority_stats.items()):
        md += f"| {p} | {count} |\n"
    md += "\n"
    
    # 3. New submitted issues in past week
    md += f"---\n\n"
    md += f"## 3. New Submitted Issues (Past Week)\n\n"
    md += f"Issues created in the past 7 days (as of {datetime.now().strftime('%Y-%m-%d')}).\n\n"
    md += "| ID | Title | Status | Owner | Priority | Reason | Category | Root Cause | Dependency | PR | PR Owner | PR Status | Labels | Module | Test Module |\n"
    md += "|---|-------|--------|-------|---------|--------|----------|-----------|-----------|-------|----------|----------|--------|--------|-------------|\n"
    
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
        root_cause = issue['root_cause'] or ''
        dependency = issue['dependency'] or ''
        pr = issue['pr'] or ''
        pr_owner = issue['pr_owner'] or ''
        pr_status = issue['pr_status'] or ''
        md += f"| [{issue['id']}](https://github.com/intel/torch-xpu-ops/issues/{issue['id']}) | {title} | {status} | {owner} | {priority} | {reason} | {category} | {root_cause} | {dependency} | {pr} | {pr_owner} | {pr_status} | {labels} | {module} | {test_module} |\n"
    
    # 4. Action Required
    reporter_actions = {
        'information_required': [],
        'close_fixed': [],
        'enable_test': [],
        'add_skiplist': [],
        'verify_issue': [],
        'need_reproduce': [],
    }
    
    engineer_actions = {
        'upstream': [],
        'revisit_pr': [],
    }
    
    info_keywords = ['reproduce', 'information', 'missing', 'awaiting']
    
    for action, issues_list in action_groups.items():
        action_lower = action.lower()
        
        if 'upstream' in action_lower or 'pytorch' in action_lower:
            for iss in issues_list:
                engineer_actions['upstream'].append(iss)
        elif 'revisit' in action_lower:
            for iss in issues_list:
                engineer_actions['revisit_pr'].append(iss)
        elif 'information' in action_lower or any(kw in action_lower for kw in info_keywords):
            for iss in issues_list:
                reporter_actions['information_required'].append(iss)
        elif 'close' in action_lower or 'fix' in action_lower:
            for iss in issues_list:
                reporter_actions['close_fixed'].append(iss)
        elif 'enable' in action_lower:
            for iss in issues_list:
                reporter_actions['enable_test'].append(iss)
        elif 'skiplist' in action_lower or 'not target' in action_lower:
            for iss in issues_list:
                reporter_actions['add_skiplist'].append(iss)
        elif 'verify' in action_lower:
            for iss in issues_list:
                reporter_actions['verify_issue'].append(iss)
        elif 'reproduce' in action_lower:
            for iss in issues_list:
                reporter_actions['need_reproduce'].append(iss)
    
    md += "\n---\n\n"
    md += "## 4. Action Required\n\n"
    
    # Reporter Actions
    md += "### Reporter Actions\n\n"
    
    md += "#### Information Required\n\n"
    md += "| ID | Title | Owner | Owner Transferred | Priority | Category | Category Reason | Root Cause | Dependency | PR | PR Owner | PR Status | Test Module |\n"
    md += "|---|-------|-------|-------------------|---------|----------|----------------|-----------|-----------|-------|----------|----------|-------------|\n"

    for issue in reporter_actions['information_required']:
        title = (issue['title'] or '')[:35]
        owner = issue['assignee'] or ''
        owner_transfer = issue['owner_transfer'] or ''
        priority = issue['priority'] or ''
        category = issue['category'] or ''
        category_reason = issue['category_reason'] or ''
        root_cause = issue['root_cause'] or ''
        dependency = issue['dependency'] or ''
        pr = issue['pr'] or ''
        pr_owner = issue['pr_owner'] or ''
        pr_status = issue['pr_status'] or ''
        test_module = issue['test_module'] or ''
        md += f"| [{issue['id']}](https://github.com/intel/torch-xpu-ops/issues/{issue['id']}) | {title} | {owner} | {owner_transfer} | {priority} | {category} | {category_reason} | {root_cause} | {dependency} | {pr} | {pr_owner} | {pr_status} | {test_module} |\n"

    md += "\n#### Close Fixed Issue\n\n"
    md += "| ID | Title | Owner | Owner Transferred | Priority | Category | Category Reason | Root Cause | Dependency | PR | PR Owner | PR Status | Test Module |\n"
    md += "|---|-------|-------|-------------------|---------|----------|----------------|-----------|-----------|-------|----------|----------|-------------|\n"

    for issue in reporter_actions['close_fixed']:
        title = (issue['title'] or '')[:35]
        owner = issue['assignee'] or ''
        owner_transfer = issue['owner_transfer'] or ''
        priority = issue['priority'] or ''
        category = issue['category'] or ''
        category_reason = issue['category_reason'] or ''
        root_cause = issue['root_cause'] or ''
        dependency = issue['dependency'] or ''
        pr = issue['pr'] or ''
        pr_owner = issue['pr_owner'] or ''
        pr_status = issue['pr_status'] or ''
        test_module = issue['test_module'] or ''
        md += f"| [{issue['id']}](https://github.com/intel/torch-xpu-ops/issues/{issue['id']}) | {title} | {owner} | {owner_transfer} | {priority} | {category} | {category_reason} | {root_cause} | {dependency} | {pr} | {pr_owner} | {pr_status} | {test_module} |\n"

    md += "\n#### Enable Test\n\n"
    md += "| ID | Title | Owner | Owner Transferred | Priority | Category | Category Reason | Root Cause | Dependency | PR | PR Owner | PR Status | Test Module |\n"
    md += "|---|-------|-------|-------------------|---------|----------|----------------|-----------|-----------|-------|----------|----------|-------------|\n"

    for issue in reporter_actions['enable_test']:
        title = (issue['title'] or '')[:35]
        owner = issue['assignee'] or ''
        owner_transfer = issue['owner_transfer'] or ''
        priority = issue['priority'] or ''
        category = issue['category'] or ''
        category_reason = issue['category_reason'] or ''
        root_cause = issue['root_cause'] or ''
        dependency = issue['dependency'] or ''
        pr = issue['pr'] or ''
        pr_owner = issue['pr_owner'] or ''
        pr_status = issue['pr_status'] or ''
        test_module = issue['test_module'] or ''
        md += f"| [{issue['id']}](https://github.com/intel/torch-xpu-ops/issues/{issue['id']}) | {title} | {owner} | {owner_transfer} | {priority} | {category} | {category_reason} | {root_cause} | {dependency} | {pr} | {pr_owner} | {pr_status} | {test_module} |\n"

    md += "\n#### Add to Skiplist\n\n"
    md += "| ID | Title | Owner | Owner Transferred | Priority | Category | Category Reason | Root Cause | Dependency | PR | PR Owner | PR Status | Test Module |\n"
    md += "|---|-------|-------|-------------------|---------|----------|----------------|-----------|-----------|-------|----------|----------|-------------|\n"

    for issue in reporter_actions['add_skiplist']:
        title = (issue['title'] or '')[:35]
        owner = issue['assignee'] or ''
        owner_transfer = issue['owner_transfer'] or ''
        priority = issue['priority'] or ''
        category = issue['category'] or ''
        category_reason = issue['category_reason'] or ''
        root_cause = issue['root_cause'] or ''
        dependency = issue['dependency'] or ''
        pr = issue['pr'] or ''
        pr_owner = issue['pr_owner'] or ''
        pr_status = issue['pr_status'] or ''
        test_module = issue['test_module'] or ''
        md += f"| [{issue['id']}](https://github.com/intel/torch-xpu-ops/issues/{issue['id']}) | {title} | {owner} | {owner_transfer} | {priority} | {category} | {category_reason} | {root_cause} | {dependency} | {pr} | {pr_owner} | {pr_status} | {test_module} |\n"

    md += "\n#### Verify the Issue\n\n"
    md += "| ID | Title | Owner | Owner Transferred | Priority | Category | Category Reason | Root Cause | Dependency | PR | PR Owner | PR Status | Test Module |\n"
    md += "|---|-------|-------|-------------------|---------|----------|----------------|-----------|-----------|-------|----------|----------|-------------|\n"

    for issue in reporter_actions['verify_issue']:
        title = (issue['title'] or '')[:35]
        owner = issue['assignee'] or ''
        owner_transfer = issue['owner_transfer'] or ''
        priority = issue['priority'] or ''
        category = issue['category'] or ''
        category_reason = issue['category_reason'] or ''
        root_cause = issue['root_cause'] or ''
        dependency = issue['dependency'] or ''
        pr = issue['pr'] or ''
        pr_owner = issue['pr_owner'] or ''
        pr_status = issue['pr_status'] or ''
        test_module = issue['test_module'] or ''
        md += f"| [{issue['id']}](https://github.com/intel/torch-xpu-ops/issues/{issue['id']}) | {title} | {owner} | {owner_transfer} | {priority} | {category} | {category_reason} | {root_cause} | {dependency} | {pr} | {pr_owner} | {pr_status} | {test_module} |\n"

    md += "\n#### Need Reproduce Steps\n\n"
    md += "| ID | Title | Owner | Owner Transferred | Priority | Category | Category Reason | Root Cause | Dependency | PR | PR Owner | PR Status | Test Module |\n"
    md += "|---|-------|-------|-------------------|---------|----------|----------------|-----------|-----------|-------|----------|----------|-------------|\n"

    for issue in reporter_actions['need_reproduce']:
        title = (issue['title'] or '')[:35]
        owner = issue['assignee'] or ''
        owner_transfer = issue['owner_transfer'] or ''
        priority = issue['priority'] or ''
        category = issue['category'] or ''
        category_reason = issue['category_reason'] or ''
        root_cause = issue['root_cause'] or ''
        dependency = issue['dependency'] or ''
        pr = issue['pr'] or ''
        pr_owner = issue['pr_owner'] or ''
        pr_status = issue['pr_status'] or ''
        test_module = issue['test_module'] or ''
        md += f"| [{issue['id']}](https://github.com/intel/torch-xpu-ops/issues/{issue['id']}) | {title} | {owner} | {owner_transfer} | {priority} | {category} | {category_reason} | {root_cause} | {dependency} | {pr} | {pr_owner} | {pr_status} | {test_module} |\n"

    # Engineer Actions
    md += "\n### Engineer Actions\n\n"

    md += "#### Needs PyTorch Repo Changes (upstream)\n\n"
    md += "| ID | Title | Owner | Owner Transferred | Priority | Category | Category Reason | Root Cause | Dependency | PR | PR Owner | PR Status | Test Module |\n"
    md += "|---|-------|-------|-------------------|---------|----------|----------------|-----------|-----------|-------|----------|----------|-------------|\n"

    for issue in engineer_actions['upstream']:
        title = (issue['title'] or '')[:35]
        owner = issue['assignee'] or ''
        owner_transfer = issue['owner_transfer'] or ''
        priority = issue['priority'] or ''
        category = issue['category'] or ''
        category_reason = issue['category_reason'] or ''
        root_cause = issue['root_cause'] or ''
        dependency = issue['dependency'] or ''
        pr = issue['pr'] or ''
        pr_owner = issue['pr_owner'] or ''
        pr_status = issue['pr_status'] or ''
        test_module = issue['test_module'] or ''
        md += f"| [{issue['id']}](https://github.com/intel/torch-xpu-ops/issues/{issue['id']}) | {title} | {owner} | {owner_transfer} | {priority} | {category} | {category_reason} | {root_cause} | {dependency} | {pr} | {pr_owner} | {pr_status} | {test_module} |\n"
    
    md += "\n#### Revisit the PR as Case Failed\n\n"
    md += "| ID | Title | Owner | Owner Transferred | Priority | Category | Category Reason | Root Cause | Dependency | PR | PR Owner | PR Status | Test Module |\n"
    md += "|---|-------|-------|-------------------|---------|----------|----------------|-----------|-----------|-------|----------|----------|-------------|\n"

    for issue in engineer_actions['revisit_pr']:
        title = (issue['title'] or '')[:35]
        owner = issue['assignee'] or ''
        owner_transfer = issue['owner_transfer'] or ''
        priority = issue['priority'] or ''
        category = issue['category'] or ''
        category_reason = issue['category_reason'] or ''
        root_cause = issue['root_cause'] or ''
        dependency = issue['dependency'] or ''
        pr = issue['pr'] or ''
        pr_owner = issue['pr_owner'] or ''
        pr_status = issue['pr_status'] or ''
        test_module = issue['test_module'] or ''
        md += f"| [{issue['id']}](https://github.com/intel/torch-xpu-ops/issues/{issue['id']}) | {title} | {owner} | {owner_transfer} | {priority} | {category} | {category_reason} | {root_cause} | {dependency} | {pr} | {pr_owner} | {pr_status} | {test_module} |\n"
    
    # 5. By Category
    md += "\n---\n\n"
    md += "## 5. By Category\n\n"

    for cat in sorted(category_groups.keys()):
        issues_list = category_groups[cat]
        issues_list.sort(key=lambda x: x['id'])
        cat_count = len(issues_list)
        md += "#### " + cat + "\n\n"
        md += "| ID | Title | Status | Owner | Priority | Category Reason | Root Cause | Dependency | PR | PR Owner | PR Status | Labels | Module | Test Module |\n"
        md += "|---|-------|--------|-------|---------|-----------------|-----------|-----------|-------|----------|----------|--------|--------|-------------|\n"

        category_reason = issue['category_reason'] or ''
        root_cause = issue['root_cause'] or ''
        dependency = issue['dependency'] or ''
        pr = issue['pr'] or ''
        pr_owner = issue['pr_owner'] or ''
        pr_status = issue['pr_status'] or ''
        for issue in issues_list:
            title = (issue['title'] or '')[:30]
            status = issue['status'] or ''
            owner = issue['assignee'] or ''
            priority = issue['priority'] or ''
            category_reason = issue['category_reason'] or ''
            root_cause = issue['root_cause'] or ''
            dependency = issue['dependency'] or ''
            pr = issue['pr'] or ''
            pr_owner = issue['pr_owner'] or ''
            pr_status = issue['pr_status'] or ''
            labels = issue['labels'] or ''
            module = issue['module'] or ''
            test_module = issue['test_module'] or ''
            md += f"| [{issue['id']}](https://github.com/intel/torch-xpu-ops/issues/{issue['id']}) | {title} | {status} | {owner} | {priority} | {category_reason} | {root_cause} | {dependency} | {pr} | {pr_owner} | {pr_status} | {labels} | {module} | {test_module} |\n"
        md += "\n"

    # 6. Duplicated Issues
    md += "\n---\n\n"
    md += "## 6. Duplicated Issues\n\n"
    md += "Issues that share test cases with other issues.\n\n"
    md += "| ID | Title | Owner | Reporter | Duplicated With | Priority | Category Reason | Root Cause | Dependency | PR | PR Owner | PR Status | Labels | Module | Test Module |\n"
    md += "|---|-------|-------|----------|-----------------|---------|-----------------|-----------|-----------|-------|----------|----------|--------|--------|-------------|\n"

    for issue in duplicated:
        title = (issue['title'] or '')[:30]
        owner = issue['assignee'] or ''
        reporter = issue['reporter'] or ''
        dup = issue['duplicated_issue'] or ''
        priority = issue['priority'] or ''
        category_reason = issue['category_reason'] or ''
        root_cause = issue['root_cause'] or ''
        dependency = issue['dependency'] or ''
        pr = issue['pr'] or ''
        pr_owner = issue['pr_owner'] or ''
        pr_status = issue['pr_status'] or ''
        labels = issue['labels'] or ''
        module = issue['module'] or ''
        test_module = issue['test_module'] or ''
        md += f"| [{issue['id']}](https://github.com/intel/torch-xpu-ops/issues/{issue['id']}) | {title} | {owner} | {reporter} | {dup} | {priority} | {category_reason} | {root_cause} | {dependency} | {pr} | {pr_owner} | {pr_status} | {labels} | {module} | {test_module} |\n"

    # 7. Issues with Dependency
    md += "\n---\n\n"
    md += "## 7. Issues with Dependency\n\n"
    md += "Issues that have dependencies on other components.\n\n"
    md += "| ID | Title | Owner | Priority | Category Reason | Root Cause | Dependency | Category | PR | PR Owner | PR Status | Labels | Test Module |\n"
    md += "|---|-------|---------|---------|-----------------|-----------|------------|----------|-------|----------|----------|--------|-------------|\n"

    with_dependency = [i for i in with_dependency if i['dependency']]
    with_dependency.sort(key=lambda x: x['id'])
    for issue in with_dependency:
        title = (issue['title'] or '')[:25]
        owner = issue['assignee'] or ''
        priority = issue['priority'] or ''
        category_reason = issue['category_reason'] or ''
        root_cause = issue['root_cause'] or ''
        dep = issue['dependency'] or ''
        category = issue['category'] or ''
        pr = issue['pr'] or ''
        pr_owner = issue['pr_owner'] or ''
        pr_status = issue['pr_status'] or ''
        labels = issue['labels'] or ''
        test_module = issue['test_module'] or ''
        md += f"| [{issue['id']}](https://github.com/intel/torch-xpu-ops/issues/{issue['id']}) | {title} | {owner} | {priority} | {category_reason} | {root_cause} | {dep} | {category} | {pr} | {pr_owner} | {pr_status} | {labels} | {test_module} |\n"
    
    # Save to file
    output_path = os.path.join(RESULT_DIR, 'issue_report.md')
    with open(output_path, 'w') as f:
        f.write(md)
    
    print(f"Report saved to: {output_path}")
    print(f"Total issues: {len(issues)}")
    print(f"  Action Required: {action_required_count}")
    print(f"  No Assignee: {no_assignee_count}")
    print(f"  Duplicated: {duplicated_count}")
    print(f"  With Dependency: {with_dependency_count}")
    print(f"  Others: {others_count}")


if __name__ == '__main__':
    generate_report()