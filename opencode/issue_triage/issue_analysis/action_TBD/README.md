# Action TBD Analysis Module

Manages action_TBD and owner_transfer for Issues sheet updates with separate functions for different action types.

## Action Types

| Action | Condition |
|--------|-----------|
| `add to skiplist` | Not target/wontfix issues or tests that cannot be enabled |
| `Close fixed issue` | All test cases passed (XPU/stock/E2E) |
| `Enable test` | Test cases can be enabled on XPU |
| `Verify the issue` | PR closed but no failed tests |
| `Revisit the PR as case failed` | PR closed but tests still failing |
| `Needs Upstream Skip PR` | Not target + UT upstream |
| `Needs Skip PR` | Wontfix/not_target |
| `Awaiting response from reporter` | Info already requested |
| `Need reproduce steps` | LLM suggests reporter needs to provide repro steps |
| `LLM Suggestion: ...` | Other LLM suggestions for info request |

## Usage

```python
from issue_analysis.action_TBD import ActionAnalyzer, analyze_action_all

# Method 1: Using the comprehensive function
owner_transfer, action_tbd, reason = analyze_action_all(
    issue_id="12345",
    labels="wontfix",
    title_raw="Test issue",
    summary_raw="",
    reporter="user@github.com",
    assignee="dev@github.com",
    test_module="ut",
    xpu_statuses=set(),
    stock_statuses=set(),
    e2e_statuses=set(),
    issue_can_enable={},
    issue_duplicated_map={},
    pr_status="open",
    test_cases_info=[],
    llm_info_action=None,
    version_info=None
)

# Method 2: Using individual action functions
result = action_add_to_skiplist(is_not_target=True, can_enable_false=False, reporter="user@github.com")
owner, action, reason = result

result = action_close_fixed_issue(
    xpu_all_passed=True,
    stock_all_passed=False,
    e2e_all_passed=False,
    reporter="user@github.com"
)
owner, action, reason = result

result = action_enable_test(
    issue_id="12345",
    issue_can_enable={"12345": {"can_enable_list": [True], "comments_list": []}},
    reporter="user@github.com"
)
owner, action, reason = result

# Method 3: Using the class
analyzer = ActionAnalyzer()
owner, action, reason = analyzer.analyze(
    issue_id="12345",
    labels="",
    title_raw="Test",
    summary_raw="",
    reporter="user@github.com",
    assignee="dev@github.com",
    test_module="ut",
    xpu_statuses=set(),
    stock_statuses=set(),
    e2e_statuses=set(),
    issue_can_enable={},
    issue_duplicated_map={},
    pr_status="open",
    test_cases_info=[]
)
```

## Functions

### `analyze_action_all(...)`
Comprehensive action analysis checking all action types in priority order.

**Args:**
- `issue_id`: Issue identifier
- `labels`: Issue labels
- `title_raw`: Issue title
- `summary_raw`: Issue summary
- `reporter`: Issue reporter email/ID
- `assignee`: Issue assignee email/ID
- `test_module`: Test module (ut/e2e)
- `xpu_statuses`: Set of XPU test statuses
- `stock_statuses`: Set of stock CI test statuses
- `e2e_statuses`: Set of E2E test statuses
- `issue_can_enable`: Dictionary of can_enable info per issue
- `issue_duplicated_map`: Dictionary of duplicated issues
- `pr_status`: PR status
- `test_cases_info`: List of test case info dicts
- `llm_info_action`: LLM suggestion for info request
- `version_info`: Version string

**Returns:**
- `Tuple[str, str, str]`: (owner_transfer, action_tbd, reason)

### Individual Action Functions

- `action_add_to_skiplist(is_not_target, can_enable_false, reporter)`
- `action_close_fixed_issue(xpu_all_passed, stock_all_passed, e2e_all_passed, reporter, is_random)`
- `action_enable_test(issue_id, issue_can_enable, reporter)`
- `action_verify_issue(pr_closed, has_failed, reporter, assignee)`
- `action_revisit_pr_failed(pr_closed, has_failed, assignee)`
- `action_needs_upstream_skip_pr(is_not_target_upstream, assignee)`
- `action_needs_skip_pr(is_wontfix, is_upstream, is_not_target, assignee)`
- `action_upstream_investigation(is_upstream, is_not_target, is_wontfix, assignee, is_e2e_issue)`
- `action_awaiting_response_from_reporter(has_already_requested, is_bug_or_perf, reporter)`
- `action_need_reproduce_steps(llm_info_action, reporter)`
- `action_llm_suggestion(llm_info_action, reporter)`
- `action_bug_ready_for_upstream(has_test_info, is_bug_or_perf, has_reproduce_steps, reporter)`
- `action_need_more_information(is_feature_request, is_bug_or_perf, is_e2e_issue, title_lower, summary_lower, reporter)`
- `action_no_status_pending(is_public, all_statuses_empty, e2e_all_passed, reporter, assignee)`
- `action_e2e_issue(is_e2e_issue, has_e2e_status, e2e_all_passed, reporter, assignee)`

## Helper Functions

- `check_info_requested_to_reporter(issue_content) -> bool`: Check if info was requested from reporter
- `is_public_branch(version_str) -> bool`: Check if version indicates public branch