# Test Cases Processor Module

## Overview

This module contains all logic for processing the 'Test_Cases' sheet in `torch_xpu_ops_issues.xlsx`.

## Usage

```python
import openpyxl
from test_result.Test_Cases.test_cases_processor import process_test_cases_sheet

wb = openpyxl.load_workbook('torch_xpu_ops_issues.xlsx')
ws, issue_duplicated_map = process_test_cases_sheet(wb)
wb.save('torch_xpu_ops_issues.xlsx')
```

## Functions

| Function | Description |
|----------|-------------|
| `process_test_cases_sheet(wb)` | Main entry point - processes Test_Cases sheet |
| `get_torch_xpu_ops_xml_files()` | Load XML files from torch-xpu-ops nightly CI |
| `get_stock_xml_files()` | Load XML files from stock PyTorch XPU CI |
| `convert_test_file_to_xml_prefix(test_file)` | Convert test file path to XML prefix |
| `convert_to_stock_prefix(test_file)` | Convert test file to stock CI prefix |
| `find_best_xml_match(xml_prefix, xml_files)` | Find matching XML file |
| `get_test_result(xml_path, test_case)` | Get test result from XML |
| `parse_failure_content(content)` | Parse failure/error message |
| `load_ops_dependency()` | Load ops -> dependency mapping |
| `load_ops_dependency_rag()` | Load ops list for RAG fuzzy matching |
| `get_dependency_from_ops_fuzzy(ops_list, ops_dep_list)` | Fuzzy match ops to dependencies |
| `analyze_test_case_with_llm()` | Use opencode LLM for test existence |
| `analyze_test_case()` | Basic file-based test existence check |
| `analyze_test_case_with_llm_qwen()` | Use Qwen3-32B LLM for test existence |
| `find_duplicated_issues(ws)` | Find cross-issue duplicates |

## Columns Processed (11-23)

| Col | Header |
|-----|--------|
| 11 | status in torch-xpu-ops nightly |
| 12 | comments in torch-xpu-ops nightly |
| 13 | Commit |
| 14 | Run_id |
| 15 | XML |
| 16 | status in stock CI |
| 17 | comments in stock CI |
| 18 | cuda_case_exist |
| 19 | xpu_case_exist |
| 20 | case_existence_comments |
| 21 | can_enable_on_xpu |
| 22 | duplicated_issue |
| 23 | dependency_lib |

## Processing Steps

1. **Setup**: Add header columns 11-23
2. **Load XMLs**: Load CI results from torch-xpu-ops nightly and stock CI
3. **PASS 1**: Process CI results for all test cases (XPU nightly + stock)
4. **PASS 2**: For "not found" cases, run LLM to check CUDA/XPU existence
5. **Apply LLM**: Write case existence results (cuda_exists, xpu_exists, comments)
6. **Dependency**: Populate dependency_lib from RAG fuzzy matching
7. **Duplicates**: Mark cross-issue duplicated test cases
8. **Cleanup**: Clear old boolean values in duplicated_issue