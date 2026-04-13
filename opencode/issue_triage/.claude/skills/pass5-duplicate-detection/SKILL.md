# PASS 5: Duplicate Detection

## Overview
Detect duplicate test cases across issues using Test Class + Test Case name matching and traceback similarity.

## Workflow
1. Load all test cases from torch_xpu_ops_issues.xlsx
2. Apply duplicate detection with two criteria:
   a. **Test Class + Test Case match**: Same test class and method name
   b. **Traceback similarity**: Normalized traceback string matching
3. Group duplicates and assign duplicate group IDs
4. Populate 'duplicate_group_id' column

## Usage
```bash
cd /home/daisydeng/ai_for_validation/opencode/issue_triage/test_result_analysis/Test_Cases
python run_processor_steps.py --steps 5
```

## Duplicate Criteria
1. **Exact Match**: Same test class AND same test case name
2. **Fuzzy Match**: Traceback similarity > threshold (normalized string)

## Input
- Test Cases sheet with Test Class and Test Case columns
- Traceback information from error messages

## Output
- Column "duplicate_group_id" populated with group identifiers
- Related Issues column populated (if applicable)

## Related Files
- pass5_duplicate_detection.py

## Example
```
Original: 53 duplicates (Test Class + Test Case only)
Enhanced: 131 duplicates (+147%) with traceback similarity added
```