"""
Action TBD Analysis Module

Manages action_TBD and owner_transfer for Issues sheet updates.
Provides separate functions for different action types.

Action Types:
- add to skiplist
- Close fixed issue
- Enable test
- Verify the issue
- Revisit the PR as case failed
- Needs Skip PR (wontfix / not_target + upstream)
- Awaiting response from reporter
- Need reproduce steps
- and more...
"""

from typing import Optional, Tuple, Dict, Set, Any, List


def action_add_to_skiplist(
    is_not_target: bool,
    can_enable_false: bool,
    reporter: str = None
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Action: Add to skiplist (for not_target/wontfix issues or tests that cannot be enabled).

    Returns:
        Tuple of (owner_transfer, action_tbd, reason) or (None, None, None) if not applicable
    """
    if is_not_target or can_enable_false:
        owner = reporter
        action = 'add to skiplist'
        reason = 'Issue marked as not_target/wontfix - should be skipped for XPU enablement'
        if can_enable_false and not is_not_target:
            reason = 'Test cases cannot be enabled on XPU - marked for skiplist'
        return (owner, action, reason)
    return (None, None, None)


def action_close_fixed_issue(
    xpu_all_passed: bool,
    stock_all_passed: bool,
    e2e_all_passed: bool,
    reporter: str,
    is_random: bool = False,
    xpu_statuses: set = None,
    stock_statuses: set = None,
    has_failed: bool = False,
    is_model_issue: bool = False
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Action: Close fixed issue (ALL test cases passed with no failures).

    Returns:
        Tuple of (owner_transfer, action_tbd, reason) or (None, None, None) if not applicable
    """
    if is_random:
        return (None, None, None)

    # Issue is closed fixed ONLY if:
    # 1. BOTH XPU and Stock have passed status (not just one)
    # 2. NO failures in ANY status set
    # This ensures the issue is truly resolved on both platforms
    
    if xpu_all_passed and stock_all_passed and not has_failed:
        owner = reporter
        action = 'Close fixed issue'
        reason = 'All test cases passed on both XPU and stock - issue is resolved'
        return (owner, action, reason)

    return (None, None, None)


def action_enable_test(
    can_enable_true: bool,
    can_enable_false: bool,
    reporter: str = None
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Action: Enable test (when test cases can be enabled on XPU).

    Note: If both can_enable_true and can_enable_false are True, 
    'Enable test' takes precedence (tests can be enabled).

    Returns:
        Tuple of (owner_transfer, action_tbd, reason) or (None, None, None)
    """
    if not can_enable_true and not can_enable_false:
        return (None, None, None)

    owner = reporter
    if can_enable_true:
        action = 'Enable test'
        reason = 'Test cases can be enabled on XPU'
    else:
        action = 'add to skiplist'
        reason = 'Test cases cannot be enabled on XPU - marked for skiplist'
    return (owner, action, reason)


def action_enable_test(
    has_cuda_enabled_error: bool,
    cuda_case_exists: bool,
    xpu_case_missing: bool,
    both_status_blank: bool,
    reporter: str = None
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    return (None, None, None)


def action_awaiting_reporter_response(
    has_requested_info: bool,
    is_bug_or_perf: bool,
    reporter: str = None
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Action: Awaiting response from reporter when info was requested.
    
    Conditions:
    1. Maintainer has requested more information from reporter (has_requested_info=True)
    2. Issue is a bug or performance issue
    
    Then action = 'Awaiting response from reporter', owner = reporter

    Returns:
        Tuple of (owner_transfer, action_tbd, reason) or (None, None, None)
    """
    if has_requested_info and is_bug_or_perf:
        return (
            reporter,
            'Awaiting response from reporter',
            'Maintainer has requested more information from reporter - waiting for response'
        )
    
    return (None, None, None)


def action_verify_issue(
    pr_closed: bool,
    has_failed: bool,
    reporter: str = None,
    assignee: str = None
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Action: Verify the issue (PR closed but no failed tests).

    Returns:
        Tuple of (owner_transfer, action_tbd, reason) or (None, None, None) if not applicable
    """
    if pr_closed and not has_failed:
        owner = reporter
        action = 'Verify the issue'
        reason = 'PR closed but no failed tests - verify if issue still reproduces'
        return (owner, action, reason)

    return (None, None, None)


def action_revisit_pr_failed(
    pr_closed: bool,
    has_failed: bool,
    assignee: str = None
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Action: Revisit the PR as case failed (PR closed but tests still failing).

    Returns:
        Tuple of (owner_transfer, action_tbd, reason) or (None, None, None) if not applicable
    """
    if pr_closed and has_failed:
        owner = assignee
        action = 'Revisit the PR as case failed'
        reason = 'PR closed but tests still failing - revisit PR for fix'
        return (owner, action, reason)

    return (None, None, None)


def action_needs_skip_pr(
    is_wontfix: bool,
    is_upstream: bool,
    is_not_target: bool,
    assignee: str = None,
    reporter: str = None
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Action: Needs Skip PR (wontfix / not_target + upstream).
    Combines upstream skip PR logic - removed duplicate function.

    Returns:
        Tuple of (owner_transfer, action_tbd, reason) or (None, None, None) if not applicable
    """
    is_not_target_upstream = is_not_target and is_upstream
    owner = assignee if assignee else reporter

    if is_not_target_upstream:
        action = 'action_needs_skip_pr'
        reason = 'Issue has not_target label and is upstream - needs skip PR upstream'
        return (owner, action, reason)

    if is_wontfix:
        action = 'action_needs_skip_pr'
        reason = 'Issue marked as wontfix or not_target - needs skip PR'
        return (owner, action, reason)

    return (None, None, None)


def action_awaiting_response_from_reporter(
    has_already_requested: bool,
    is_bug_or_perf: bool,
    reporter: str = None
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Action: Awaiting response from reporter (info already requested).

    Returns:
        Tuple of (owner_transfer, action_tbd, reason) or (None, None, None) if not applicable
    """
    if has_already_requested and is_bug_or_perf:
        owner = reporter
        action = 'Awaiting response from reporter'
        reason = 'Information requested from reporter - pending response'
        return (owner, action, reason)

    return (None, None, None)


def action_need_reproduce_steps(
    llm_info_action: str,
    reporter: str = None
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Action: Need reproduce steps (LLM suggests reporter needs to provide repro steps).

    Returns:
        Tuple of (owner_transfer, action_tbd, reason) or (None, None, None) if not applicable
    """
    if llm_info_action and 'reproduce' in llm_info_action.lower():
        owner = reporter
        action = 'Need reproduce steps'
        reason = f'LLM suggests: {llm_info_action} - reporter needs to provide reproduce steps'
        return (owner, action, reason)

    return (None, None, None)


def action_llm_suggestion(
    llm_info_action: str,
    reporter: str = None
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Action: LLM Suggestion (other LLM suggestions for info request).

    Returns:
        Tuple of (owner_transfer, action_tbd, reason) or (None, None, None) if not applicable
    """
    if llm_info_action and llm_info_action not in ['Ready to analyze']:
        owner = reporter
        action = f'LLM Suggestion: {llm_info_action}'
        reason = f'LLM analysis suggests: {llm_info_action}'
        return (owner, action, reason)

    return (None, None, None)


def action_bug_ready_for_upstream(
    has_test_info: bool,
    is_bug_or_perf: bool,
    has_reproduce_steps: bool,
    reporter: str = None
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Action: Bug/Perf issue ready for upstream fix analysis.

    Returns:
        Tuple of (owner_transfer, action_tbd, reason) or (None, None, None) if not applicable
    """
    if is_bug_or_perf and has_test_info:
        owner = None
        action = ''
        reason = 'Test info provided - ready for upstream fix analysis'
        return (owner, action, reason)

    if is_bug_or_perf and not has_test_info and not has_reproduce_steps:
        owner = None
        action = ''
        reason = 'Bug/Perf issue - needs upstream fix investigation'
        return (owner, action, reason)

    return (None, None, None)


def action_need_more_information(
    is_feature_request: bool,
    is_bug_or_perf: bool,
    is_e2e_issue: bool,
    title_lower: str,
    summary_lower: str,
    reporter: str = None
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Action: Need more information (reporter needs to provide specific details).

    Returns:
        Tuple of (owner_transfer, action_tbd, reason) or (None, None, None) if not applicable
    """
    if is_feature_request or is_e2e_issue:
        return (None, None, None)

    if not is_bug_or_perf:
        info_needed = []
        if 'accuracy' in title_lower or 'accuracy' in summary_lower:
            info_needed.append('accuracy comparison data')
        if 'performance' in title_lower or 'performance' in summary_lower:
            info_needed.append('performance numbers/baseline')
        if 'regression' in title_lower or 'regression' in summary_lower:
            info_needed.append('previous good version info')

        if info_needed:
            owner = reporter
            action = f'Need more information - {", ".join(info_needed)}'
            reason = f'Reporter needs to provide: {", ".join(info_needed)}'
            return (owner, action, reason)

    return (None, None, None)


def action_no_status_pending(
    is_public: bool,
    all_statuses_empty: bool,
    e2e_all_passed: bool,
    reporter: str,
    assignee: str = None
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    DISABLED: No Test Status in CI action removed.
    Issues with no test status will fall through to Need Investigation fallback.

    Returns:
        Tuple of (None, None, None) - action disabled
    """
    return (None, None, None)


def action_e2e_issue(
    is_e2e_issue: bool,
    has_e2e_status: bool,
    e2e_all_passed: bool,
    is_accuracy_issue: bool,
    reporter: str,
    assignee: str = None
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Action: E2E accuracy issue pending - needs upstream investigation.
    Only triggers for accuracy-related E2E issues.

    Returns:
        Tuple of (owner_transfer, action_tbd, reason) or (None, None, None) if not applicable
    """
    if not is_e2e_issue or not is_accuracy_issue:
        return (None, None, None)

    if has_e2e_status and e2e_all_passed:
        owner = reporter
        action = 'Close fixed issue'
        reason = 'All E2E accuracy test cases passed - issue is resolved'
        return (owner, action, reason)
    else:
        owner = assignee
        action = 'Need Investigation'
        reason = 'E2E accuracy issue pending - needs upstream investigation'
        return (owner, action, reason)


def analyze_action_all(
    issue_id: str,
    labels: str,
    title_raw: str,
    summary_raw: str,
    reporter: str,
    assignee: str,
    test_module: str,
    xpu_statuses: Set[str],
    stock_statuses: Set[str],
    e2e_statuses: Set[str],
    issue_can_enable: Dict[str, Dict[str, List[Any]]],
    issue_duplicated_map: Dict[str, Set[str]],
    pr_status: str,
    test_cases_info: List[Dict[str, Any]],
    llm_info_action: str = None,
    version_info: str = None,
    error_msg: str = None,
    test_case_cuda_exists: bool = False,
    test_case_xpu_exists: bool = True,
    has_cuda_enabled_error: bool = False
) -> Tuple[str, str, str]:
    """
    Comprehensive action analysis for an issue.

    Checks all action types in priority order and returns the first matching action.

    Returns:
        Tuple of (owner_transfer, action_tbd, reason)
    """
    is_random = 'random' in str(labels).lower()
    labels_str = str(labels).lower() if labels else ''
    title_lower = title_raw.lower()
    summary_lower = summary_raw.lower() if summary_raw else ''

    is_not_target = ('not target' in labels_str or 'wont' in labels_str or "won't" in labels_str)
    is_wontfix = 'wont ' in labels_str or ' wont ' in labels_str or 'wontfix' in labels_str or 'not target' in labels_str.replace('nottarget', '')
    is_upstream = 'ut_upstream' in labels_str or 'inductor' in labels_str

    is_e2e_issue = test_module == 'e2e' or 'e2e' in (test_module or '').lower()
    is_accuracy_issue = 'accuracy' in title_lower or 'accuracy' in summary_lower

    xpu_all_passed = xpu_statuses == {'passed'}
    stock_all_passed = stock_statuses == {'passed'}
    e2e_all_passed = all(s == 'pass' for s in e2e_statuses) if e2e_statuses else False
    has_failed = ('failed' in xpu_statuses) or ('failed' in stock_statuses)
    pr_closed = pr_status in ['closed', 'merged']

    is_public = version_info and is_public_branch(version_info)
    has_xpu_status = bool(xpu_statuses)
    has_stock_status = bool(stock_statuses)
    has_e2e_status = bool(e2e_statuses)
    all_statuses_empty = not has_xpu_status and not has_stock_status and not has_e2e_status

    is_feature_request = (
        ('feature' in title_lower or 'feature' in summary_lower) or
        ('request' in title_lower or 'request' in summary_lower) or
        ('implement' in title_lower or 'implement' in summary_lower) or
        ('add support' in title_lower or 'add support' in summary_lower) or
        ('enable' in title_lower and 'test' not in title_lower and 'feature' in summary_lower) or
        (labels and 'enhancement' in str(labels).lower())
    )

    is_bug_or_perf = any(kw in title_lower or kw in summary_lower for kw in [
        'bug', 'fail', 'error', 'crash', 'assertion', 'exception',
        'performance', 'slow', 'latency', 'timeout', 'regression',
        'accuracy', 'wrong result', 'precision', 'dtype'
    ])

    can_enable_true = False
    can_enable_false = False
    comments_list = []
    if issue_id in issue_can_enable:
        can_enable_info = issue_can_enable[issue_id]
        can_enable_list = can_enable_info.get('can_enable_list', [])
        comments_list = can_enable_info.get('comments_list', [])
        can_enable_true = any(str(val) == 'True' for val in can_enable_list)
        can_enable_false = any(str(val) == 'False' for val in can_enable_list)

    owner_transfer = ''
    action_tbd = ''
    action_tbd_reason = ''

    # Priority 1: Add to skiplist (not_target or cannot enable)
    result = action_add_to_skiplist(is_not_target, can_enable_false, reporter)
    if result[0]:
        owner_transfer, action_tbd, action_tbd_reason = result
        return (owner_transfer, action_tbd, action_tbd_reason)

    # Priority 2: Close fixed issue (all tests passed - no failures)
    result = action_close_fixed_issue(
        xpu_all_passed, stock_all_passed, e2e_all_passed, reporter, is_random,
        xpu_statuses=xpu_statuses, stock_statuses=stock_statuses, has_failed=has_failed
    )
    if result[0] or result[1]:
        owner_transfer, action_tbd, action_tbd_reason = result
        return (owner_transfer, action_tbd, action_tbd_reason)

    # Priority 2.5: Enable CUDA Test->XPU migration (CUDA enabled error case)
    # ONLY triggers when: CUDA error + CUDA case exists + XPU case NOT exists + both status blank
    result = action_enable_test(
        has_cuda_enabled_error=has_cuda_enabled_error,
        cuda_case_exists=test_case_cuda_exists,
        xpu_case_missing=not test_case_xpu_exists,  # XPU test case does NOT exist
        both_status_blank=all_statuses_empty,
        reporter=reporter
    )
    if result[0] and result[1]:
        owner_transfer, action_tbd, action_tbd_reason = result
        return (owner_transfer, action_tbd, action_tbd_reason)

    # Priority 3: Verify the issue (has PR, no failures)
    # Relaxed: trigger for ANY PR reported
    pr_reported = pr_status and str(pr_status).strip().lower() not in ['none', '']
    if pr_reported and not has_failed:
        result = action_verify_issue(pr_reported, has_failed, reporter, assignee)
    if result[0] and result[1]:
        owner_transfer, action_tbd, action_tbd_reason = result
        return (owner_transfer, action_tbd, action_tbd_reason)

    # Priority 5: Revisit PR as case failed (has PR, still failing)
    if pr_reported and has_failed:
        result = action_revisit_pr_failed(pr_reported, has_failed, assignee)
        if result[0] and result[1]:
            owner_transfer, action_tbd, action_tbd_reason = result
            return (owner_transfer, action_tbd, action_tbd_reason)

    # Priority 6: Needs Skip PR (wontfix/not_target only, NOT upstream)
    if not action_tbd and (is_wontfix or is_not_target) and not is_upstream:
        owner_transfer = assignee if assignee else reporter
        action_tbd = 'Needs Skip PR'
        action_tbd_reason = 'Issue marked as wontfix/not_target - needs skip PR'
        return (owner_transfer, action_tbd, action_tbd_reason)

    # Priority 6b: Needs Upstream Skip PR (upstream issues only)
    if not action_tbd and is_upstream:
        result = action_needs_skip_pr(is_wontfix, is_upstream, is_not_target, assignee, reporter)
        if result[0] is not None:
            owner_transfer, action_tbd, action_tbd_reason = result
            return (owner_transfer, action_tbd, action_tbd_reason)

    # Priority 6: E2E accuracy issue pending
    if not action_tbd and is_e2e_issue and is_accuracy_issue:
        owner_transfer = assignee if assignee else reporter
        action_tbd = 'E2E accuracy issue'
        action_tbd_reason = 'E2E accuracy issue pending - needs upstream investigation'
        return (owner_transfer, action_tbd, action_tbd_reason)

    # Priority 7: No Test Status in CI
    if not action_tbd:
        result = action_no_status_pending(is_public, all_statuses_empty, e2e_all_passed, reporter, assignee)
        if result[0] is not None:
            owner_transfer, action_tbd, action_tbd_reason = result
            return (owner_transfer, action_tbd, action_tbd_reason)

#    # Priority 8: Awaiting response from reporter (is_bug/perf)
#    if not action_tbd and is_bug_or_perf and not has_failed:
#        owner_transfer = reporter
#        action_tbd = 'Awaiting response from reporter'
#        action_tbd_reason = 'Bug/Perf issue pending reporter response'
#        return (owner_transfer, action_tbd, action_tbd_reason)

#    # Priority 9: Need reproduce steps (keyword based)
#    if not action_tbd and not is_feature_request:
#        title_sum = (title_raw + ' ' + summary_raw).lower()
#        if any(kw in title_sum for kw in ['no error', 'cannot reproduce', 'reproduce', 'how to']):
#            owner_transfer = reporter
#            action_tbd = 'Need reproduce steps'
#            action_tbd_reason = 'Issue needs reproducible test case from reporter'
#            return (owner_transfer, action_tbd, action_tbd_reason)

#    # Priority 10: Bug/Perf awaiting responses
#    if not action_tbd and is_bug_or_perf:
#        owner_transfer = reporter
#        action_tbd = 'Awaiting response'
#        action_tbd_reason = 'Bug/Perf issue awaiting reporter response'
#        return (owner_transfer, action_tbd, action_tbd_reason)

    # Priority 11: Need more information (accuracy/performance keywords)
    if not action_tbd and is_bug_or_perf and not has_failed:
        title_sum = (str(title_raw) + ' ' + str(summary_raw)).lower()
        #if any(kw in title_sum for kw in ['accuracy', 'performance', 'regression', 'slow', 'wrong']):
        #    owner_transfer = reporter
        #    action_tbd = 'Need more information'
        #    action_tbd_reason = 'Issue needs accuracy/performance data from reporter'
        #    return (owner_transfer, action_tbd, action_tbd_reason)

        #result = action_bug_ready_for_upstream(
        #    any(tc.get('test_file') and tc.get('test_case') for tc in test_cases_info),
        #    is_bug_or_perf,
        #    has_already_requested,
        #    reporter
        #)
        #if result[0] is not None:
        #    owner_transfer, action_tbd, action_tbd_reason = result
        #    return (owner_transfer, action_tbd, action_tbd_reason)

        #result = action_awaiting_response_from_reporter(has_already_requested, is_bug_or_perf, reporter)
        #if result[0] and result[1]:
        #    owner_transfer, action_tbd, action_tbd_reason = result
        #    return (owner_transfer, action_tbd, action_tbd_reason)

        result = action_need_more_information(
            is_feature_request, is_bug_or_perf, is_e2e_issue,
            title_lower, summary_lower, reporter
        )
        if result[0] and result[1]:
            owner_transfer, action_tbd, action_tbd_reason = result
            return (owner_transfer, action_tbd, action_tbd_reason)

    # Fallback: If no action TBD found, set to Need Investigation
    if not action_tbd:
        owner_transfer = assignee
        action_tbd = 'Need Investigation'
        action_tbd_reason = 'No specific action identified - needs investigation'
        return (owner_transfer, action_tbd, action_tbd_reason)

    return (owner_transfer, action_tbd, action_tbd_reason)


def check_info_requested_to_reporter(issue_content: str, title_raw: str = '', summary_raw: str = '') -> bool:
    """
    Check if maintainer has requested more information from reporter.
    Uses keyword-based check first, falls back to LLM if negative.

    Args:
        issue_content: Combined issue title and summary
        title_raw: Original title (for LLM fallback)
        summary_raw: Original summary (for LLM fallback)

    Returns:
        True if info was requested from reporter
    """
    if not issue_content:
        return False

    request_keywords = [
        'could you please provide',
        'please provide more',
        'can you provide additional',
        'need more information',
        'needs more info',
        'please add',
        'please share',
        'need the reproduce',
        'we need',
        'please attach',
        'please run',
        'please check',
        'please verify',
        'reproduce',
        'reproduction steps',
        'how to reproduce',
        'provide the',
        'please include',
        'could you provide',
        'can you also',
        'is there a way',
        'tell us more',
        'elaborate',
        'clarify',
    ]

    content_lower = issue_content.lower()
    keyword_found = any(kw in content_lower for kw in request_keywords)
    
    if keyword_found:
        return True
    
    return False


def is_public_branch(version_str: str) -> bool:
    """
    Check if version indicates a public branch (main, release) vs private branch/PR.

    Args:
        version_str: Version string to check

    Returns:
        True if public branch, False if private
    """
    if not version_str:
        return False

    version_lower = version_str.lower()

    if 'pr' in version_lower and 'http' in version_lower:
        return False

    if version_lower in ['main', 'master']:
        return True

    import re
    if re.match(r'^v?2\.\d+(\.\d+)?$', version_lower):
        return True

    if re.match(r'^\d+\.\d+\.\d+a\d+\+git[0-9a-f]+$', version_lower):
        return True

    if '+git' in version_lower and 'pr' not in version_lower:
        return True

    return False

import os
MINIMAX_ENDPOINT = os.environ.get("MINIMAX_ENDPOINT", "http://10.239.15.43/v1/chat/completions")
MINIMAX_API_KEY = os.environ.get("OPENCODE_API_KEY", "sk-xxxx")
MINIMAX_MODEL = "glm-latest"
PYTORCH_TEST_ROOT = os.path.expanduser("~/issue_traige/pytorch/test")
TORCH_XPU_TEST_ROOT = os.path.expanduser("~/issue_traige/torch-xpu-ops/test/xpu")


def check_info_requested_to_reporter_llm(
    issue_title: str,
    issue_summary: str,
    error_msg: str = None,
    traceback: str = None
) -> str:
    """
    Use Qwen3-32B via internal API to check if maintainer has requested more info from reporter.
    Returns: "Need reproduce steps" or "Need more information" if info was requested, empty string otherwise.
    """
    import requests
    import os
    import time
    import re

    LLM_ENDPOINT = "http://10.239.15.43/v1/chat/completions"
    LLM_API_KEY = os.environ.get("OPENCODE_API_KEY", "sk-xxxxxxxxxx")
    LLM_MODEL = "Qwen3-32B"

    issue_content = f"{issue_title} {issue_summary} {error_msg or ''} {traceback or ''}".strip()

    if not issue_content or len(issue_content) < 20:
        return ""

    prompt = f"""Analyze this PyTorch XPU GitHub issue to determine if the maintainer has requested more information from the reporter.

Issue Title: {issue_title}
Issue Summary: {issue_summary[:500] if issue_summary else 'N/A'}
Error: {error_msg[:500] if error_msg else 'N/A'}

Look at the issue description above and determine:
- Has the maintainer already requested more information (reproduce steps, error details, performance data, etc.)?
- OR does the issue have sufficient information to start debugging?

Respond with EXACTLY one of:
- "Need reproduce steps" - if maintainer needs reproduce steps from reporter
- "Need more information" - if maintainer needs other info from reporter (accuracy data, version info, etc.)
- "" (empty string) - if the issue already has sufficient information OR maintainer hasn't requested anything

YOUR ANSWER:"""

    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": "Output ONLY what action to take. Be brief. No markdown. No JSON."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0,
        "max_tokens": 60
    }

    start_time = time.time()

    try:
        response = requests.post(LLM_ENDPOINT, headers=headers, json=payload, timeout=180)
        elapsed = time.time() - start_time

        if response.status_code == 200:
            result = response.json()
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            # Clean up thinking tags - extract content after closing bracket
            content = content.replace("[TO]", "").replace("[/TO]", "").replace("[:cn]", "").strip()
            # Split by '>' or '>]' and take last part
            parts = re.split(r'>\s*', content)
            content = parts[-1].strip() if parts else content.strip()
            # Clean up any remaining thinking tags patterns
            content = re.sub(r'\[n?t\w*\]', '', content)
            content = re.sub(r'\s+', ' ', content)
            content = content.strip()
            # Take first meaningful response
            first_sentence = content.split('.')[0] if '.' in content else content
            first_sentence = first_sentence.strip()
            if len(first_sentence) < 5 or first_sentence.lower().startswith(('okay', 'let', '首先', '这个', '该')):
                # Fall back to second sentence or just return need more info
                sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', content) if s.strip()]
                content = ''
                for s in sentences:
                    s_clean = re.sub(r'\[n?t\w*\]', '', s).strip()
                    if s_clean and not s_clean.lower().startswith(('okay', 'let', '首先', '这个', '该')):
                        content = s_clean
                        break
            if not content or len(content) < 5:
                return ""
            content = content[:100].strip()
            if content.lower().startswith('okay') or content.lower().startswith('let'):
                return ""

            # Check if LLM indicates info was requested
            if "reproduce" in content.lower():
                return "Need reproduce steps"
            elif "need more information" in content.lower():
                return "Need more information"
            else:
                # No info requested - sufficient info provided
                return ""

        return ""

    except Exception as e:
        return ""


class ActionAnalyzer:
    """
    High-level interface for action_TBD analysis.
    Provides methods to determine appropriate actions for issues.
    """

    def __init__(self):
        pass

    def analyze(
        self,
        issue_id: str,
        labels: str,
        title_raw: str,
        summary_raw: str,
        reporter: str,
        assignee: str,
        test_module: str,
        xpu_statuses: Set[str],
        stock_statuses: Set[str],
        e2e_statuses: Set[str],
        issue_can_enable: Optional[Dict] = None,
        issue_duplicated_map: Optional[Dict] = None,
        pr_status: str = None,
        test_cases_info: Optional[List[Dict]] = None,
        llm_info_action: str = None,
        version_info: str = None,
        error_msg: str = None,
        test_case_cuda_exists: bool = False,
        test_case_xpu_exists: bool = True,
        has_cuda_enabled_error: bool = False
    ) -> Tuple[str, str, str]:
        """
        Analyze and determine appropriate action for an issue.

        Returns:
            Tuple of (owner_transfer, action_tbd, reason)
        """
        if issue_can_enable is None:
            issue_can_enable = {}
        if issue_duplicated_map is None:
            issue_duplicated_map = {}
        if test_cases_info is None:
            test_cases_info = []

        return analyze_action_all(
            issue_id, labels, title_raw, summary_raw,
            reporter, assignee, test_module,
            xpu_statuses, stock_statuses, e2e_statuses,
            issue_can_enable, issue_duplicated_map,
            pr_status, test_cases_info, llm_info_action,
            version_info,
            error_msg=error_msg,
            test_case_cuda_exists=test_case_cuda_exists,
            test_case_xpu_exists=test_case_xpu_exists,
            has_cuda_enabled_error=has_cuda_enabled_error
        )
