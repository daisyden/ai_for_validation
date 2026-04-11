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
- Needs Upstream Skip PR (not_target + ut_upstream)
- Needs Skip PR (wontfix / not_target)
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
    is_model_issue: bool = False
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Action: Close fixed issue (all test cases passed).

    Returns:
        Tuple of (owner_transfer, action_tbd, reason) or (None, None, None) if not applicable
    """
    if is_random:
        return (None, None, None)

    if xpu_all_passed or stock_all_passed:
        owner = reporter
        action = 'Close fixed issue'
        reason = 'All test cases passed on XPU/stock - issue is resolved'
        return (owner, action, reason)

    if e2e_all_passed:
        owner = reporter
        action = 'Close fixed issue'
        reason = 'All E2E test cases passed - issue is resolved'
        return (owner, action, reason)

    return (None, None, None)


def action_enable_test(
    issue_id: str,
    issue_can_enable: Dict[str, Dict[str, List[Any]]],
    reporter: str = None
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Action: Enable test (when test cases can be enabled on XPU).

    Returns:
        Tuple of (owner_transfer, action_tbd, reason) or (None, None, None)
    """
    if issue_id not in issue_can_enable:
        return (None, None, None)

    can_enable_info = issue_can_enable[issue_id]
    can_enable_list = can_enable_info.get('can_enable_list', [])
    comments_list = can_enable_info.get('comments_list', [])

    can_enable_true = any(str(val) == 'True' for val in can_enable_list)
    can_enable_false = any(str(val) == 'False' for val in can_enable_list)

    if can_enable_true or can_enable_false:
        owner = reporter
        action = 'Enable test' if can_enable_true else 'add to skiplist'
        comments = ' | '.join([str(c) for c in comments_list if c])
        reason = f'Case existence comments: {comments}'
        return (owner, action, reason)

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


def action_needs_upstream_skip_pr(
    is_not_target_upstream: bool,
    assignee: str = None
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Action: Needs Upstream Skip PR (not_target + ut_upstream).

    Returns:
        Tuple of (owner_transfer, action_tbd, reason) or (None, None, None) if not applicable
    """
    if is_not_target_upstream:
        owner = assignee
        action = 'Needs Upstream Skip PR (not_target + ut_upstream)'
        reason = 'Issue has not_target label and is upstream - needs skip PR upstream'
        return (owner, action, reason)

    return (None, None, None)


def action_needs_skip_pr(
    is_wontfix: bool,
    is_upstream: bool,
    is_not_target: bool,
    assignee: str = None
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Action: Needs Skip PR (wontfix / not_target).

    Returns:
        Tuple of (owner_transfer, action_tbd, reason) or (None, None, None) if not applicable
    """
    is_not_target_upstream = is_not_target and is_upstream

    if is_not_target_upstream:
        owner = assignee
        action = 'Needs Upstream Skip PR (not_target + ut_upstream)'
        reason = 'Issue has not_target label and is upstream - needs skip PR upstream'
        return (owner, action, reason)
    elif is_wontfix:
        owner = assignee
        action = 'Needs Skip PR (wontfix / not_target)'
        reason = 'Issue marked as wontfix or not_target - needs skip PR'
        return (owner, action, reason)

    return (None, None, None)


def action_upstream_investigation(
    is_upstream: bool,
    is_not_target: bool,
    is_wontfix: bool,
    assignee: str = None,
    is_e2e_issue: bool = False
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Action: Upstream investigation needed (for upstream issues without skip).

    Returns:
        Tuple of (owner_transfer, action_tbd, reason) or (None, None, None) if not applicable
    """
    is_not_target_upstream = is_not_target and is_upstream

    if is_upstream and not is_not_target_upstream and not is_wontfix:
        if is_e2e_issue:
            owner = assignee
            action = ''
            reason = 'E2E issue pending - needs upstream investigation'
            return (owner, action, reason)
        else:
            owner = assignee
            action = ''
            reason = 'Upstream issue - owner should investigate upstream fix'
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
    Action: No test status available - pending investigation.

    Returns:
        Tuple of (owner_transfer, action_tbd, reason) or (None, None, None) if not applicable
    """
    if is_public and all_statuses_empty:
        if e2e_all_passed:
            owner = reporter
            action = 'Close fixed issue'
            reason = 'All tests passed - issue is resolved'
            return (owner, action, reason)
        else:
            owner = assignee
            action = ''
            reason = 'Issue needs investigation - no test status available'
            return (owner, action, reason)

    return (None, None, None)


def action_e2e_issue(
    is_e2e_issue: bool,
    has_e2e_status: bool,
    e2e_all_passed: bool,
    reporter: str,
    assignee: str = None
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Action: E2E issue pending - needs upstream investigation.

    Returns:
        Tuple of (owner_transfer, action_tbd, reason) or (None, None, None) if not applicable
    """
    if is_e2e_issue:
        if has_e2e_status and e2e_all_passed:
            owner = reporter
            action = 'Close fixed issue'
            reason = 'All E2E test cases passed - issue is resolved'
            return (owner, action, reason)
        else:
            owner = assignee
            action = ''
            reason = 'E2E issue pending - needs upstream investigation'
            return (owner, action, reason)

    return (None, None, None)


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
    version_info: str = None
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
    is_not_target_upstream = is_not_target and is_upstream

    is_e2e_issue = test_module == 'e2e'

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

    result = action_add_to_skiplist(is_not_target, can_enable_false, reporter)
    if result[0]:
        owner_transfer, action_tbd, action_tbd_reason = result
        return (owner_transfer, action_tbd, action_tbd_reason)

    result = action_close_fixed_issue(xpu_all_passed, stock_all_passed, e2e_all_passed, reporter, is_random)
    if result[0] or result[1]:
        owner_transfer, action_tbd, action_tbd_reason = result
        return (owner_transfer, action_tbd, action_tbd_reason)

    result = action_enable_test(issue_id, issue_can_enable, reporter)
    if result[0] and result[1]:
        owner_transfer, action_tbd, action_tbd_reason = result
        return (owner_transfer, action_tbd, action_tbd_reason)

    result = action_verify_issue(pr_closed, has_failed, reporter, assignee)
    if result[0] and result[1]:
        owner_transfer, action_tbd, action_tbd_reason = result
        return (owner_transfer, action_tbd, action_tbd_reason)

    result = action_revisit_pr_failed(pr_closed, has_failed, assignee)
    if result[0] and result[1]:
        owner_transfer, action_tbd, action_tbd_reason = result
        return (owner_transfer, action_tbd, action_tbd_reason)

    result = action_needs_upstream_skip_pr(is_not_target_upstream, assignee)
    if result[0] and result[1]:
        owner_transfer, action_tbd, action_tbd_reason = result
        return (owner_transfer, action_tbd, action_tbd_reason)

    result = action_needs_skip_pr(is_wontfix, is_upstream, is_not_target, assignee)
    if result[0] and result[1]:
        owner_transfer, action_tbd, action_tbd_reason = result
        return (owner_transfer, action_tbd, action_tbd_reason)

    result = action_upstream_investigation(is_upstream, is_not_target, is_wontfix, assignee, is_e2e_issue)
    if result[0] is not None:
        owner_transfer, action_tbd, action_tbd_reason = result
        return (owner_transfer, action_tbd, action_tbd_reason)

    if not action_tbd:
        result = action_e2e_issue(is_e2e_issue, has_e2e_status, e2e_all_passed, reporter, assignee)
        if result[0] is not None:
            owner_transfer, action_tbd, action_tbd_reason = result
            return (owner_transfer, action_tbd, action_tbd_reason)

        result = action_no_status_pending(is_public, all_statuses_empty, e2e_all_passed, reporter, assignee)
        if result[0] is not None:
            owner_transfer, action_tbd, action_tbd_reason = result
            return (owner_transfer, action_tbd, action_tbd_reason)

    if not action_tbd and not is_feature_request and not is_e2e_issue:
        has_already_requested = check_info_requested_to_reporter(title_raw + ' ' + summary_raw)

        result = action_need_reproduce_steps(llm_info_action, reporter)
        if result[0] and result[1]:
            owner_transfer, action_tbd, action_tbd_reason = result
            return (owner_transfer, action_tbd, action_tbd_reason)

        result = action_llm_suggestion(llm_info_action, reporter)
        if result[0] and result[1]:
            owner_transfer, action_tbd, action_tbd_reason = result
            return (owner_transfer, action_tbd, action_tbd_reason)

        result = action_bug_ready_for_upstream(
            any(tc.get('test_file') and tc.get('test_case') for tc in test_cases_info),
            is_bug_or_perf,
            has_already_requested,
            reporter
        )
        if result[0] is not None:
            owner_transfer, action_tbd, action_tbd_reason = result
            return (owner_transfer, action_tbd, action_tbd_reason)

        result = action_awaiting_response_from_reporter(has_already_requested, is_bug_or_perf, reporter)
        if result[0] and result[1]:
            owner_transfer, action_tbd, action_tbd_reason = result
            return (owner_transfer, action_tbd, action_tbd_reason)

        result = action_need_more_information(
            is_feature_request, is_bug_or_perf, is_e2e_issue,
            title_lower, summary_lower, reporter
        )
        if result[0] and result[1]:
            owner_transfer, action_tbd, action_tbd_reason = result
            return (owner_transfer, action_tbd, action_tbd_reason)

    return (owner_transfer, action_tbd, action_tbd_reason)


def check_info_requested_to_reporter(issue_content: str) -> bool:
    """
    Check if maintainer has requested more information from reporter.

    Args:
        issue_content: Combined issue title and summary

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
    ]

    content_lower = issue_content.lower()
    return any(kw in content_lower for kw in request_keywords)


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


def check_info_requested_to_reporter_llm(
    issue_title: str,
    issue_summary: str,
    error_msg: str = None,
    traceback: str = None
) -> str:
    """
    Use Qwen3-32B via internal API to check if more info needs to be requested from reporter.
    Returns: action string (e.g., "Need reproduce steps", "Ready to analyze", "Need more information")
    """
    import requests
    import os
    import time

    LLM_ENDPOINT = "http://10.239.15.43/v1/chat/completions"
    LLM_API_KEY = os.environ.get("OPENCODE_API_KEY", "sk-xxxxxxxxxx")
    LLM_MODEL = "Qwen3-32B"

    issue_content = f"{issue_title} {issue_summary} {error_msg or ''} {traceback or ''}".strip()

    if not issue_content or len(issue_content) < 20:
        return "Need more information"

    prompt = f"""You are analyzing a PyTorch XPU GitHub issue to determine if more information is needed from the reporter.

Issue Content:
{issue_content[:1500]}

Determine:
1. Is this a feature request or enhancement?
2. Does it have sufficient error information?
3. Does it have reproduction steps?
4. Is it a clear bug with complete information?

Respond with ONE of:
- "Ready to analyze" - with sufficient info to start debugging
- "Need reproduce steps" - if a bug without clear repro steps
- "Need more information - [specific missing info]" - if the comments requesting key details
- "Feature Request - needs triage" - if it's a feature request

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
            content = content.replace("[TO]", "").replace("[/TO]", "").replace("[:cn]", "").replace("]", "").strip()

            if "Ready to analyze" in content:
                return "Ready to analyze"
            elif "reproduce" in content.lower():
                return "Need reproduce steps"
            elif "Feature" in content:
                return "Feature Request"
            else:
                return content.strip()[:60]

        return "Need more information"

    except Exception as e:
        return "Need more information"


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
        version_info: str = None
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
            version_info
        )