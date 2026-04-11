# Torch-ops Extraction Skill

Extracts torch ops from PyTorch XPU test case data in Excel format.
Uses pattern-based extraction with LLM fallback for comprehensive coverage.

## When to Use

Use when processing torch_xpu_ops_issues.xlsx files with a "Test Cases" sheet to populate the "torch-ops" column with accurate operation names.

## Usage

```bash
cd ~/ai_for_validation/opencode/issue_triage/test_result_analysis/torch-ops-extraction
python3 extract_torch_ops.py <input_file> [output_file]
python3 extract_torch_ops.py ~/result/torch_xpu_ops_issues.xlsx
```

## Workflow

### Extraction Strategy: Pattern + LLM Fallback

1. **Pattern Extraction**: Regex and rule-based extraction from:
   - Error messages (torch.ops.aten.XXX.default pattern - HIGHEST priority)
   - Test names (OpDB patterns: torch_ops_aten__xxx, __refs_xxx, etc.)
   - Test case name mappings (addmm → torch.addmm, etc.)
   - Traceback patterns

2. **LLM Fallback**: When pattern extraction returns empty, use Qwen3-32B to extract torch ops from:
   - Test file path
   - Test case name
   - Error message
   - Traceback

### Functions

```python
def extract_torch_ops(test_file, test_case, error_msg, traceback, use_llm_fallback=True):
    """Extract torch ops using all rules in priority order.
    Falls back to LLM when pattern-based extraction returns empty.
    Returns: (ops_list, llm_elapsed_time_or_None)
    """

def extract_torch_ops_with_llm(test_file, test_case, error_msg, traceback):
    """LLM-based torch ops extraction using Qwen3-32B.
    Used as fallback when pattern-based extraction returns empty.
    Returns: (ops_list, llm_elapsed_time)
    """
```

### LLM Endpoint
- URL: `http://10.239.15.43/v1/chat/completions`
- Model: Qwen3-32B
- Timeout: 60s per call

### Logging
- Tracks LLM fallback usage (count, total time, avg time)
- Progress indicator every 500 rows

## Input Format
- Excel file with "Test Cases" sheet containing columns:
  - `Test Case`: Test case name (e.g., `test_out_addmv_xpu`)
  - `Error Message`: Error message from test failure
  - `Traceback`: Python traceback

## Output Format
- Updates "Test Cases" sheet with "torch-ops" column populated
- Preserves Issues and E2E_Test_Cases sheets

## Notes
- Pattern extraction takes priority (faster)
- LLM fallback only for unmatched cases (reduces API calls)
- Timing logged for cost/performance tracking
