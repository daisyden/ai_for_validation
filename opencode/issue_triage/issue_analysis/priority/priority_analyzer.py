#!/usr/bin/env python3
"""
Priority analysis module for PyTorch XPU issue triaging.

Contains LLM-based priority determination logic using Qwen3-32B API.

Usage:
    from issue_analysis.priority.priority_analyzer import (
        determine_priority_llm,
        determine_priority_rules,
        determine_priority
    )

    # LLM-based priority
    priority, reason, elapsed = determine_priority_llm(
        title="Issue title",
        summary="Issue summary",
        error_msg="Error message",
        test_module="ut",
        labels="bug, regression",
        test_cases_info=[{"test_case": "test_xxx", "error_msg": "..."}]
    )

    # Rule-based priority
    priority, reason, elapsed, count = determine_priority_rules(
        title, summary, test_module, labels_str, ws_test, issue_id, MAX_LLM_PRIORITY, llm_priority_count
    )
"""

import os
import re
import time

LLM_ENDPOINT = "http://10.239.15.43/v1/chat/completions"
LLM_API_KEY = os.environ.get("OPENCODE_API_KEY", "sk-xxxxxxxxxx")
LLM_MODEL = "Qwen3-32B"


def determine_priority_llm(title, summary, error_msg, test_module, labels_str, test_cases_info):
    """
    Use Qwen3-32B via internal API to determine the priority of an issue.

    Args:
        title: Issue title
        summary: Issue summary (up to 500 chars)
        error_msg: Error message from test failure
        test_module: Test module type ('ut', 'e2e', 'build')
        labels_str: Issue labels as string
        test_cases_info: List of dicts with 'test_case', 'error_msg', 'traceback'

    Returns:
        tuple: (priority, reason, elapsed_time)
            - priority: P0/P1/P2/P3
            - reason: Detailed reason for the priority
            - elapsed_time: API call time in seconds
    """
    import requests
    import json

    tc_info_str = ""
    if test_cases_info:
        for tc in test_cases_info:
            tc_info_str += f"- {tc.get('test_case', '')}: {str(tc.get('error_msg', ''))[:80]}\n"

    prompt = f"""You are analyzing PyTorch XPU issue priority.

Title: {title}
Summary: {summary[:500]}
Test Module: {test_module or 'Unknown'}
Labels: {labels_str}

Error Info:
{error_msg[:300] if error_msg else 'N/A'}

Test Cases:
{tc_info_str}

Determine priority (P0=critical, P1=high, P2=medium, P3=low):
- P0: Build crash, regression (was passing), real model failure, security
- P1: Many test failures, e2e accuracy issue, performance regression
- P2: Few UT failures, feature gaps, minor issues
- P3: Minor, cosmetic, documentation

Return ONLY format: "P# - detailed_reason" (full sentences, 150-300 characters)

Example: "P0 - Build crash during aten.neg kernel compilation for XPU backend due to undefined reference to device-specific Triton template implementation"
Example: "P1 - E2E regression in HuggingFace models involving torch.nn.functional.scaled_dot_product_attention failing with precision mismatch on fp16 input for XPU device"
Example: "P2 - aten.dot_xpu_mkl kernel NotImplementedError when called with Long tensors, indicating dtype support gap for aten.matmul operation on XPU"

Include specific details about: ops/functions involved, dtype transitions, arguments/parameters, failure patterns, severity indicators.

YOUR ANSWER:"""

    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": "Output ONLY 'Priority - reason'. No markdown. No JSON. No thinking tags."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0,
        "max_tokens": 50
    }

    start_time = time.time()

    try:
        response = requests.post(LLM_ENDPOINT, headers=headers, json=payload, timeout=180)
        elapsed = time.time() - start_time

        if response.status_code == 200:
            result = response.json()
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            content = content.replace("<think>", "").replace("]", "").strip()

            match = re.search(r'(P[0-3])\s*[-–]\s*[^\n]+', content, re.IGNORECASE)
            if match:
                priority = match.group(1).upper()
                reason = content[match.start():].split('-')[-1].strip()[:50]
                return priority, reason, elapsed

            return "P2", "Default priority", elapsed

        return "P2", f"API Error: {response.status_code}", 0

    except Exception as e:
        return "P2", f"Error: {str(e)[:30]}", 0


def determine_priority_rules(title_raw, summary_raw, test_module, labels, ws_test, issue_id, MAX_LLM_PRIORITY, llm_priority_count):
    """
    Rule-based priority determination with heuristics.

    Args:
        title_raw: Issue title
        summary_raw: Issue summary
        test_module: Test module type ('ut', 'e2e', 'build')
        labels: Issue labels
        ws_test: Test Cases worksheet
        issue_id: Current issue ID
        MAX_LLM_PRIORITY: Max number of LLM calls allowed
        llm_priority_count: Current count of LLM calls

    Returns:
        tuple: (priority, priority_reason, llm_count, elapsed)
    """
    import time

    priority = 'P2'
    priority_reason = ''

    is_model_issue = (
        'model' in title_raw.lower() or 'model' in summary_raw.lower() or
        'application' in title_raw.lower() or 'application' in summary_raw.lower() or
        'huggingface' in title_raw.lower() or 'timm' in title_raw.lower() or 'torchbench' in title_raw.lower()
    )
    is_e2e = test_module == 'e2e'
    is_ut = test_module == 'ut'

    is_regression = (
        'regression' in str(labels).lower() or 'regression' in title_raw.lower() or
        'was pass' in summary_raw.lower() or 'previously pass' in summary_raw.lower() or
        ('before' in summary_raw.lower() and 'now' in summary_raw.lower())
    )

    is_build_crash = (
        'build' in test_module.lower() or 'build' in title_raw.lower() or
        'crash' in title_raw.lower() or 'segmentation' in title_raw.lower() or
        'segfault' in title_raw.lower() or 'signal' in summary_raw.lower()
    )

    failed_count = 0
    for tr in range(2, ws_test.max_row + 1):
        if ws_test.cell(tr, 1).value == issue_id:
            tc_status = ws_test.cell(tr, 11).value
            if tc_status in ['failed', 'error']:
                failed_count += 1

    if is_build_crash:
        priority = 'P0'
        priority_reason = 'Build crash - critical blocking issue'
    elif is_model_issue and not ('test' in title_raw.lower() and 'case' in title_raw.lower()):
        priority = 'P0'
        priority_reason = 'Impacts real model/application'
    elif is_regression:
        priority = 'P0'
        priority_reason = 'Regression - passed before but failed now'
    elif is_e2e and ('accuracy' in title_raw.lower() or 'accuracy' in summary_raw.lower() or
                    'fail' in title_raw.lower() or 'fail' in summary_raw.lower()):
        priority = 'P1'
        priority_reason = 'E2E benchmark accuracy/functionality issue'
    elif is_e2e and ('performance' in title_raw.lower() or 'slow' in title_raw.lower() or
                    'latency' in title_raw.lower()):
        priority = 'P2'
        priority_reason = 'E2E benchmark performance issue'
    elif is_ut and failed_count > 20:
        priority = 'P1'
        priority_reason = f'UT with {failed_count} failed test cases'
    else:
        priority = 'P2'
        priority_reason = 'UT issue with few failures'

    elapsed = 0
    if llm_priority_count < MAX_LLM_PRIORITY:
        test_cases_for_llm = []
        for tr in range(2, ws_test.max_row + 1):
            if ws_test.cell(tr, 1).value == issue_id:
                tc_info = {
                    'test_case': ws_test.cell(tr, 7).value,
                    'error_msg': ws_test.cell(tr, 8).value,
                    'traceback': ws_test.cell(tr, 9).value
                }
                test_cases_for_llm.append(tc_info)

        # Get error_msg from first test case for context
        error_msg = ''
        for tc in test_cases_for_llm:
            if tc.get('error_msg'):
                error_msg = str(tc.get('error_msg'))
                break

        labels_str = str(labels) if labels else ''
        llm_priority, llm_reason, elapsed = determine_priority_llm(
            title_raw, summary_raw, error_msg, test_module, labels_str, test_cases_for_llm
        )

        if llm_priority.startswith('P') and not llm_reason.startswith('API'):
            priority = llm_priority
            priority_reason = llm_reason
            llm_priority_count += 1
            print(f"  [LLM PRIORITY #{llm_priority_count}] Issue {issue_id}: {priority} - {priority_reason}")

    return priority, priority_reason, llm_priority_count, elapsed


def determine_priority(
    title,
    summary,
    error_msg=None,
    test_module=None,
    labels=None,
    test_cases_info=None,
    is_regression=False,
    is_build_crash=False,
    failed_count=0,
    is_model_issue=False,
    is_e2e=False
):
    """
    Unified priority determination combining rules and LLM.

    Priority levels:
    - P0: Build crash, regression, real model failure, security
    - P1: Many test failures, e2e accuracy issue, performance regression
    - P2: Few UT failures, feature gaps, minor issues
    - P3: Minor, cosmetic, documentation

    Args:
        title: Issue title
        summary: Issue summary
        error_msg: Error message from test
        test_module: Module type ('ut', 'e2e', 'build')
        labels: Labels list or string
        test_cases_info: List of test case dicts
        is_regression: Whether this is a regression
        is_build_crash: Whether this is a build crash
        failed_count: Number of failed test cases
        is_model_issue: Whether this impacts real models
        is_e2e: Whether this is an E2E test issue

    Returns:
        tuple: (priority, reason)
            - priority: P0, P1, P2, or P3
            - reason: Explanation for the priority
    """
    title_lower = title.lower() if title else ''
    summary_lower = summary.lower() if summary else ''

    labels_str = str(labels).lower() if labels else ''
    test_module_lower = (test_module or '').lower()

    if is_build_crash or 'build' in test_module_lower or 'crash' in title_lower or 'segmentation' in title_lower or 'segfault' in title_lower:
        return 'P0', 'Build crash - critical blocking issue'

    if is_model_issue and not ('test' in title_lower and 'case' in title_lower):
        return 'P0', 'Impacts real model/application'

    if is_regression or 'regression' in labels_str or ('before' in summary_lower and 'now' in summary_lower):
        return 'P0', 'Regression - passed before but failed now'

    is_e2e_module = test_module == 'e2e'
    if is_e2e_module or is_e2e:
        if 'accuracy' in title_lower or 'accuracy' in summary_lower or 'fail' in title_lower:
            return 'P1', 'E2E benchmark accuracy/functionality issue'
        if 'performance' in title_lower or 'slow' in title_lower or 'latency' in title_lower:
            return 'P2', 'E2E benchmark performance issue'

    if test_module == 'ut' and failed_count > 20:
        return 'P1', f'UT with {failed_count} failed test cases'

    if failed_count == 0:
        return 'P3', 'Minor issue - no test failures'

    if failed_count <= 3:
        return 'P2', f'UT issue with {failed_count} failed test cases'

    return 'P1', f'UT issue with {failed_count} failed test cases'


# Priority level constants
PRIORITY_P0 = "P0"
PRIORITY_P1 = "P1"
PRIORITY_P2 = "P2"
PRIORITY_P3 = "P3"

# Priority descriptions
PRIORITY_DESCRIPTIONS = {
    "P0": "Critical - Build crash, regression, real model failure",
    "P1": "High - Many test failures, e2e accuracy issue",
    "P2": "Medium - Few UT failures, feature gaps",
    "P3": "Low - Minor, cosmetic, documentation"
}