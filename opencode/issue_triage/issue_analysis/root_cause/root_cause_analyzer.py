"""
Root Cause Analyzer Module

LLM-based root cause analysis for PyTorch XPU issues.
Uses internal Qwen3-32B API to classify issues into categories.

Categories:
1. Memory/Shared Memory Issue
2. Dtype/Precision Issue
3. Inductor/Compilation Issue
4. DNNL/OneDNN Issue
5. Flash Attention/Specific Ops Issue
6. Distributed/Gloo Issue
7. Skip/No Test Exists
8. Backend/Device Issue
9. API/Template Mismatch
10. Feature Not Supported
11. Timeout/Performance Issue
12. Runtime Error
13. Assertion Failure
14. Type/Value Error
15. Others
"""

import os
import re
import time
import requests

LLM_ENDPOINT = "http://10.239.15.43/v1/chat/completions"
LLM_API_KEY = os.environ.get("OPENCODE_API_KEY", "sk-xxxx")
LLM_MODEL = "Qwen3-32B"
ROOT_CAUSE_LOG = os.path.expanduser('~/ai_for_validation/opencode/issue_triage/result/root_cause.txt')


def log_root_cause(issue_id: str, root_cause: str, elapsed: float):
    """Log root cause analysis result to file."""
    msg = f"Issue {issue_id}: {root_cause} ({elapsed:.2f}s)"
    print(f"  {msg}")
    with open(ROOT_CAUSE_LOG, 'a') as log:
        log.write(msg + "\n")


def analyze_root_cause_llm(
    issue_id: str,
    issue_title: str,
    issue_summary: str,
    test_file: str,
    test_class: str,
    test_case: str,
    error_msg: str,
    traceback: str,
    test_module: str = None
) -> str:
    """
    Use internal Qwen3-32B LLM to analyze root cause of an issue.

    Args:
        issue_id: GitHub issue ID
        issue_title: Title of the issue
        issue_summary: Summary/description of the issue
        test_file: Test file where failure occurred
        test_class: Test class name
        test_case: Test case name
        error_msg: Error message
        traceback: Stack trace
        test_module: Test module path

    Returns:
        Root cause description string (e.g., "Dtype/Precision Issue - ...")
    """
    if not issue_title and not issue_summary and not error_msg and not traceback:
        return ""

    prompt = f"""You are analyzing PyTorch XPU issue root cause.

Issue ID: {issue_id}
Title: {issue_title}
Summary: {issue_summary}
Test: {test_class}.{test_case}
Error: {error_msg}
Traceback: {traceback[:500] if traceback else 'N/A'}

Classify into ONE category (internal classification):
1. Memory/Shared Memory Issue
2. Dtype/Precision Issue
3. Inductor/Compilation Issue
4. DNNL/OneDNN Issue
5. Flash Attention/Specific Ops Issue
6. Distributed/Gloo Issue
7. Skip/No Test Exists
8. Backend/Device Issue
9. API/Template Mismatch
10. Feature Not Supported
11. Timeout/Performance Issue
12. Runtime Error
13. Assertion Failure
14. Type/Value Error
15. Others

Answer format: "CATEGORY - detailed_reason" (full sentences, 150-300 characters)

Include in your reason:
- Specific PyTorch ops (aten.xxx), functions (torch.nn.functional.xxx), or APIs involved
- Dtypes that caused issues (float32, bf16, fp16, int8, Long, etc.)
- Arguments or parameters that triggered the failure
- Error patterns or signature mismatches
- Device-specific context (XPU, Inductor, Triton)

Example: "Dtype/Precision Issue - aten.memory_efficient_attention kernel fails when invoked with fp32 input tensors combined with bfloat16 scaling factor on XPU device. The torch.ops.scaled_dot_product_attention call encounters dtype promotion mismatch between query (float32), key (float32), and value (bfloat16 tensors) during kernel selection phase"
Example: "Backend/Device Issue - XPU device initialization fails during kernel compilation stage with sycl::queue creation error. The test case involves torch.nn.functional.conv2d with input tensor of ndarry type requiring device placement from CPU to XPU"

YOUR ANSWER (no JSON, no thinking tags, just the answer):"""

    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": "Output ONLY 'Category - reason'. No markdown. No JSON. No thinking tags."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 400,
        "temperature": 0.0
    }

    start = time.time()
    response = None
    try:
        max_retries = 3
        for attempt in range(max_retries):
            response = requests.post(LLM_ENDPOINT, headers=headers, json=data, timeout=180)
            elapsed = time.time() - start

            if response.status_code == 200:
                break
            elif response.status_code in (403, 429):
                wait_time = 60
                print(f"    [Rate Limit] Waiting {wait_time}s (attempt {attempt+1}/{max_retries})...")
                time.sleep(wait_time)
            else:
                elapsed = time.time() - start
                log_root_cause(issue_id, f"API Error: {response.status_code}", elapsed)
                return ""

        if response is None or response.status_code != 200:
            elapsed = time.time() - start
            log_root_cause(issue_id, f"API Error: {response.status_code if response else 'No response'}", elapsed)
            return ""

        resp_data = response.json()
        content = resp_data.get("choices", [{}])[0].get("message", {}).get("content", "")

        content = content.replace("<think>", "").replace("]", "")
        content = content.replace("[/草原]", "").strip()

        match = re.search(
            r'(Memory|Dtype|Precision|Inductor|Compilation|DNNL|OneDNN|Flash|Attention|Distributed|Gloo|Backend|Device|API|Template|Mismatch|Feature|Not|Supported|Timeout|Performance|Runtime|Error|Assertion|Failure|Type|Value|Others)\s*[-_\-\u2013]\s*.{50,}',
            content,
            re.IGNORECASE
        )

        if match:
            root_cause = match.group(0).strip()
        else:
            lines = [l.strip() for l in content.split("\n") if l.strip() and len(l.strip()) > 20]
            root_cause = lines[-1][:300] if lines else content.strip()[:300]

        log_root_cause(issue_id, root_cause, time.time() - start)
        return root_cause

    except Exception as e:
        elapsed = time.time() - start
        log_root_cause(issue_id, f"Exception: {str(e)[:50]}", elapsed)
        return ""


def analyze_root_cause_keyword(
    issue_title: str,
    issue_summary: str,
    test_file: str,
    test_class: str,
    test_case: str,
    error_msg: str,
    traceback: str,
    test_module: str = None
) -> str:
    """
    Determine root cause category based on issue information using keyword matching.
    Fallback method when LLM is unavailable.

    Returns:
        Root cause description string
    """
    text = f"{issue_title} {issue_summary} {error_msg or ''} {traceback or ''}".lower()

    if any(k in text for k in ['out of memory', 'oom', 'alloc', 'memory', 'cuda out of memory']):
        if 'shared' in text:
            return "Memory/Shared Memory Issue - shared memory allocation failed"
        return "Memory/Shared Memory Issue"

    if any(k in text for k in ['permanent kill', 'xgboost']):
        return "XGBoost/External Dependency"

    if any(k in text for k in ['requires xccl', 'no xccl', 'xccl not found']):
        return "XCCL/Dependency Issue"

    if any(k in text for k in ['inductor', 'compile', 'graph break', 'symbolic', 'fx']):
        if test_file and 'test/inductor' in str(test_file).lower():
            return "Inductor / Compilation Issue"
        return "Inductor / Compilation Issue"

    if any(k in text for k in ['dynamo']) and (
        (test_file and 'test/dynamo' in str(test_file).lower()) or
        (test_file and 'testinductor' in str(test_file).lower())
    ):
        return "Inductor / Compilation Issue"

    if 'dnnl' in text or 'onednn' in text or 'mkldnn' in text:
        return "DNNL/OneDNN Issue"

    if any(k in text for k in ['flash attention', 'flash_attention', 'flashattn']) and 'attention' in text:
        return "Flash Attention / Specific Ops Issue"

    if 'distributed' in text or 'gloo' in text or ' nccl' in text or 'nccl' in text:
        if test_file and 'test/distributed' in str(test_file).lower():
            return "Distributed / Gloo Issue"

    if any(k in text for k in ['dtype', 'precision', 'accuracy', 'numerical', 'fp16', 'bf16', 'float16']):
        return "Dtype / Precision Issue"

    if 'float' in text and ('16' in text or '32' in text or 'bf' in text):
        return "Dtype / Precision Issue"

    if 'skip' in text or 'decorator' in text or ('test' in text and 'not found' in text):
        return "Skip / No Test Exists"

    if 'device' in text or 'xpu' in text and 'init' in text:
        return "Backend / Device Issue"

    if 'api' in text or 'template' in text or 'signature' in text:
        return "API / Template Mismatch"

    if any(k in text for k in ['not implemented', 'not support', 'unimplemented']):
        return "Feature Not Supported"

    if any(k in text for k in ['import error', 'no module', 'cannot find module']):
        return "Import / Dependency Issue"

    if 'assertionerror' in text or 'assert ' in text:
        if 'dtype' in text or 'precision' in text or 'float' in text:
            return "Dtype / Precision Issue"
        return "Assertion Failure"

    if 'runtimeerror' in text:
        if 'memory' in text or 'allocate' in text:
            return "Memory/Shared Memory Issue"
        if 'dtype' in text or 'cast' in text or 'convert' in text:
            return "Dtype / Precision Issue"
        if 'inductor' in text or 'compile' in text:
            return "Inductor / Compilation Issue"
        return "Runtime Error"

    if 'typeerror' in text or 'valueerror' in text:
        if 'dtype' in text or 'cast' in text:
            return "Dtype / Precision Issue"
        return "Type / Value Error"

    if 'timeout' in text:
        return "Timeout / Performance Issue"

    if test_file and 'test/nn' in str(test_file).lower():
        if test_case and ('conv' in str(test_case).lower() or 'linear' in str(test_case).lower()):
            return "DNNL / Specific Ops Issue"

    return "Others"


class RootCauseAnalyzer:
    """
    High-level interface for root cause analysis.
    Combines LLM-based analysis with keyword fallback.
    """

    def __init__(self, use_llm: bool = True):
        """
        Initialize the root cause analyzer.

        Args:
            use_llm: Whether to use LLM for analysis (default: True)
        """
        self.use_llm = use_llm

    def analyze(
        self,
        issue_id: str,
        issue_title: str,
        issue_summary: str,
        test_file: str = None,
        test_class: str = None,
        test_case: str = None,
        error_msg: str = None,
        traceback: str = None,
        test_module: str = None
    ) -> str:
        """
        Analyze root cause of an issue.

        Args:
            issue_id: GitHub issue ID
            issue_title: Issue title
            issue_summary: Issue summary/description
            test_file: Test file path
            test_class: Test class name
            test_case: Test case name
            error_msg: Error message
            traceback: Stack trace
            test_module: Test module path

        Returns:
            Root cause description string
        """
        if self.use_llm:
            result = analyze_root_cause_llm(
                issue_id, issue_title, issue_summary,
                test_file, test_class, test_case,
                error_msg, traceback, test_module
            )
            if result:
                return result

        return analyze_root_cause_keyword(
            issue_title, issue_summary,
            test_file, test_class, test_case,
            error_msg, traceback, test_module
        )


def analyze_root_cause(
    issue_title: str,
    issue_summary: str,
    test_file: str,
    test_class: str,
    test_case: str,
    error_msg: str,
    traceback: str,
    test_module: str = None
) -> str:
    """
    Analyze root cause of an issue based on available information.
    Wrapper function for backward compatibility.

    Args:
        issue_title: Issue title
        issue_summary: Issue summary
        test_file: Test file path
        test_class: Test class name
        test_case: Test case name
        error_msg: Error message
        traceback: Stack trace
        test_module: Test module path

    Returns:
        Root cause description string
    """
    return analyze_root_cause_keyword(
        issue_title, issue_summary,
        test_file, test_class, test_case,
        error_msg, traceback, test_module
    )