# Priority Analysis Module

## Overview

This module contains LLM-based and rule-based priority determination logic for PyTorch XPU issue triaging.

## Usage

```python
from issue_analysis.priority.priority_analyzer import (
    determine_priority,
    determine_priority_llm,
    determine_priority_rules
)

# Simple rule-based priority (fast)
priority, reason = determine_priority(
    title="Issue title",
    summary="Issue summary",
    test_module="ut",
    is_regression=True,
    failed_count=25
)
# Returns: ('P0', 'Regression - passed before but failed now')

# LLM-based priority (accurate)
priority, reason, elapsed = determine_priority_llm(
    title="Issue title",
    summary="Issue summary",
    error_msg="Error message",
    test_module="ut",
    labels_str="bug, regression",
    test_cases_info=[{"test_case": "test_xxx", "error_msg": "..."}]
)
# Returns: ('P1', 'Reason...', 2.5)

# Combined approach (rules + LLM fallback)
priority, reason, count, elapsed = determine_priority_rules(
    title_raw, summary_raw, test_module, labels, ws_test, issue_id,
    MAX_LLM_PRIORITY=500, llm_priority_count=0
)
```

## Priority Levels

| Priority | Description |
|----------|-------------|
| **P0** | Critical - Build crash, regression (was passing), real model failure, security |
| **P1** | High - Many test failures, e2e accuracy issue, performance regression |
| **P2** | Medium - Few UT failures, feature gaps, minor issues |
| **P3** | Low - Minor, cosmetic, documentation |

## Functions

| Function | Description | Returns |
|----------|-------------|---------|
| `determine_priority()` | Unified rule-based priority | (priority, reason) |
| `determine_priority_llm()` | LLM-based priority using Qwen3-32B | (priority, reason, elapsed) |
| `determine_priority_rules()` | Rules + LLM fallback | (priority, reason, count, elapsed) |

## Priority Rules

1. **P0 - Build Crash**: test_module='build' or title contains crash/segmentation/segfault
2. **P0 - Regression**: labels/summary contains "regression" or "was pass...now fail"
3. **P0 - Real Model**: title/summary contains "model" or "application" (but NOT "test case")
4. **P1 - E2E Accuracy**: e2e module + "accuracy" or "fail" in title/summary
5. **P2 - E2E Performance**: e2e module + "performance"/"slow"/"latency"
6. **P1 - Many Failures**: ut module + failed_count > 20
7. **P2 - Few Failures**: Default for UT issues with failures

## Constants

```python
PRIORITY_P0 = "P0"
PRIORITY_P1 = "P1"
PRIORITY_P2 = "P2"
PRIORITY_P3 = "P3"

PRIORITY_DESCRIPTIONS = {
    "P0": "Critical - Build crash, regression, real model failure",
    "P1": "High - Many test failures, e2e accuracy issue",
    "P2": "Medium - Few UT failures, feature gaps",
    "P3": "Low - Minor, cosmetic, documentation"
}
```

## LLM Configuration

Uses Qwen3-32B via internal API:
- Endpoint: `http://10.239.15.43/v1/chat/completions`
- API Key: From `OPENCODE_API_KEY` env var
- Model: `Qwen3-32B`
- Temperature: 0.0
- Max tokens: 50

## Example Output

```
# P0 Priority
"P0 - Build crash during aten.neg kernel compilation for XPU backend due to undefined reference to device-specific Triton template implementation"

# P1 Priority  
"P1 - E2E regression in HuggingFace models involving torch.nn.functional.scaled_dot_product_attention failing with precision mismatch on fp16 input for XPU device"

# P2 Priority
"P2 - aten.dot_xpu_mkl kernel NotImplementedError when called with Long tensors, indicating dtype support gap for aten.matmul operation on XPU"
```