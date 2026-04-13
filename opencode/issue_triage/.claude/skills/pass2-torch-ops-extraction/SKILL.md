# PASS 2: Torch-Ops Extraction

## Overview
Extract torch operations from test case error messages and CI data using pattern matching with LLM fallback.

## Workflow
1. Parse error messages from test case reproduce command
2. Apply pattern matching to identify torch ops
3. Use LLM for complex cases where pattern matching fails
4. Populate the 'Torch Op' column in Test Cases sheet

## Usage
```bash
cd /home/daisydeng/ai_for_validation/opencode/issue_triage/test_result_analysis/Test_Cases
python run_processor_steps.py --steps 2
```

## Pattern Categories
- `aten::` - ATen operators (native PyTorch)
- `torch.<module>.` - torch module functions
- `F.` - torch.nn.functional
- Custom/Specific ops

## Fallback to LLM
When pattern matching fails, LLM is used to:
- Analyze error stack trace
- Identify the actual torch operation being tested
- Handle cryptic or indirect references

## Input
- Test Cases sheet with error messages/tracebacks
- CI data from test_cases_all.xlsx

## Output
- Column 5 "Torch Op" populated with extracted operations