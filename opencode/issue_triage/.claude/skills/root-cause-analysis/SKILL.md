# Root Cause Analysis

## Overview
Analyzes PyTorch XPU issue root causes using LLM-based analysis (Qwen3-32B) with keyword-based fallback. Adds Root Cause and Root Cause Reason columns to Issues sheet.

## Workflow
1. Load `torch_xpu_ops_issues.xlsx` with Issues and Test Cases sheets
2. Check for existing Root Cause columns or find first blank for new columns
3. For each issue, build prompt with: issue_id, title, summary, test info, error_msg, traceback
4. Send to Qwen3-32B LLM for root cause classification
5. Parse LLM response: "CATEGORY - detailed_reason" format
6. Fallback to keyword analysis if LLM fails
7. Add columns: Root Cause, Root Cause Reason

## Usage
```bash
cd /home/daisydeng/ai_for_validation/opencode/issue_triage/issue_analysis/root_cause
python3 run_root_cause.py [--excel EXCEL_FILE] [--limit N] [--force] [--max-llm N]
```

## Examples
```bash
# Run full LLM analysis on all issues
python3 run_root_cause.py

# Test with first 10 issues
python3 run_root_cause.py --limit 10

# Force overwrite existing values
python3 run_root_cause.py --force

# Limit LLM calls to 50 (then keyword fallback)
python3 run_root_cause.py --max-llm 50
```

## Root Cause Categories

| Category | Description |
|----------|-------------|
| **Memory/Shared Memory Issue** | OOM, allocation errors, shared memory failures |
| **Dtype/Precision Issue** | dtype mismatches, precision loss, numerical issues |
| **Inductor/Compilation Issue** | compilation errors, graph breaks, symbolic failures |
| **DNNL/OneDNN Issue** | oneDNN backend issues, mkldnn operations |
| **Flash Attention/Specific Ops Issue** | flash attention failures,Unsupported head_dim/dropout |
| **Distributed/Gloo Issue** | distributed ops, XCCL, process group issues |
| **Skip/No Test Exists** | skipped tests, missing test implementations |
| **Backend/Device Issue** | device initialization, XPU backend issues |
| **API/Template Mismatch** | API signature mismatches, template errors |
| **Feature Not Supported** | unimplemented features, unsupported ops |
| **Timeout/Performance Issue** | performance regression, timeout failures |
| **Runtime Error** | general runtime errors |
| **Assertion Failure** | assertion errors, test assertions failing |
| **Type/Value Error** | type/value errors, invalid arguments |

## LLM Integration

- **Primary**: Qwen3-32B internal API (`http://10.239.15.43/v1/chat/completions`)
- **Fallback**: Keyword-based analysis when LLM unavailable or fails
- **Rate limiting**: Automatic retry with 60s wait on 403/429 errors
- **Response format**: "CATEGORY - detailed_reason" (150-300 chars)

### LLM Prompt Structure
```
You are analyzing PyTorch XPU issue root cause.

Issue ID: {issue_id}
Title: {issue_title}
Summary: {issue_summary}
Test: {test_class}.{test_case}
Error: {error_msg}
Traceback: {traceback[:500]}

Classify into ONE category...

Example answers with detailed reasons including:
- Specific PyTorch ops (aten.xxx), functions (torch.nn.functional.xxx)
- Dtypes involved (float32, bf16, fp16, int8, Long, etc.)
- Arguments/parameters that triggered the failure
- Error patterns or signature mismatches
- Device-specific context (XPU, Inductor, Triton)
```

## Output
- **Root Cause column (col 23)**: Category name
- **Root Cause Reason column (col 24)**: Detailed LLM-generated explanation

## Key Features
- **LLM-powered**: Uses Qwen3-32B for intelligent root cause classification
- **Keyword fallback**: Reliable fallback when LLM unavailable
- **Detailed reasoning**: LLM provides context-specific explanations
- **Column preservation**: Uses existing columns or adds at first blank
- **Force mode**: Overwrite existing root cause values with `--force`
- **Rate limit handling**: Automatic retry on API rate limits

## Related Info
- LLM endpoint: `http://10.239.15.43/v1/chat/completions`
- API key: `OPENCODE_API_KEY` env var
- Input: Issues sheet (col A=issue_id) and Test Cases sheet
- Output: Root Cause columns at col 23-24