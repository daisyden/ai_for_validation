# Root Cause Analysis Module

LLM-based root cause analysis for PyTorch XPU issues.

## Overview

This module provides root cause analysis functionality using Qwen3-32B LLM to classify issues into categories:

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

## Usage

```python
from issue_analysis.root_cause import RootCauseAnalyzer, analyze_root_cause_llm

# Using the class
analyzer = RootCauseAnalyzer(use_llm=True)
root_cause = analyzer.analyze(
    issue_id="12345",
    issue_title=" aten.memory_efficient_attention kernel fails",
    issue_summary="",
    test_file="test/ops/test_attention.py",
    test_class="TestXPU",
    test_case="test_memory_efficient_attention_xpu",
    error_msg="NotImplementedError: kernel not implemented",
    traceback="...",
    test_module="ut"
)

# Using the function directly
root_cause = analyze_root_cause_llm(
    issue_id="12345",
    issue_title=" aten.memory_efficient_attention kernel fails",
    issue_summary="",
    test_file="test/ops/test_attention.py",
    test_class="TestXPU",
    test_case="test_memory_efficient_attention_xpu",
    error_msg="NotImplementedError: kernel not implemented",
    traceback="...",
    test_module="ut"
)

# Keyword fallback (no LLM)
root_cause = analyze_root_cause(
    issue_title=" aten.memory_efficient_attention kernel fails",
    issue_summary="",
    test_file="test/ops/test_attention.py",
    test_class="TestXPU",
    test_case="test_memory_efficient_attention_xpu",
    error_msg="NotImplementedError: kernel not implemented",
    traceback="...",
    test_module="ut"
)
```

## Results

Results are logged to `~/ai_for_validation/opencode/issue_triage/result/root_cause.txt`.