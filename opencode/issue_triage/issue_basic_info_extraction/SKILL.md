# Skill: Collect intel/torch-xpu-ops Open Issue Information

## Overview
This skill collects open issue information from the intel/torch-xpu-ops GitHub repository and generates Excel files with issue details and test cases.

## Prerequisites
- GitHub token with appropriate permissions
- Python with openpyxl installed
- Access to PyTorch test files for test case mapping

## Workflow

### Step 1: Fetch Open Issues
Use GitHub API with token to fetch all open issues (excluding PRs):
```python
# API endpoint: https://api.github.com/repos/intel/torch-xpu-ops/issues
# Parameters: state=open, per_page=100
# Filter out pull requests (items with 'pull_request' key)
```

### Step 2: Fetch Comments
For each issue, fetch associated comments to get full context.

### Step 3: Parse Issue Data
Extract the following fields for each issue:
1. **Basic Information**: Issue ID, Title, Status, Assignee, Reporter, Labels, Created Time, Updated Time, Milestone
2. **Related PR**: Check for any linked PRs
3. **Summary**: Brief 1-2 sentence summary of the issue
4. **Type**: Classify as:
   - `feature request` - for feature requests
   - `functionality bug` - for errors/crashes/assertions
   - `performance issue` - for latency/throughput issues
   - `internal task` - for tracking tasks (from labels)
5. **Module**: Based on content keywords:
   - `distributed`, `inductor`, `dynamo`, `autograd`, `aten_ops`, `low_precision`, `optimizer`, `profiling`
6. **Test Module**: Classify as:
   - `ut` - pytest/python test commands on test/test_*.py or test/xpu/test_*.py
   - `e2e` - benchmark tests (benchmarks/dynamo/)
   - `build` - only for actual build process issues ([Win][Build], cmake, setup.py)
   - `infrastructure` - only for CI/workflow infrastructure issues (workflow errors, github action config)
7. **Dependency**: Based on keywords: transformers, AO, oneDNN, oneCCL, oneMKL, driver, Triton, oneAPI

### Step 4: Parse Test Cases
Parse test cases from issue body in these formats:
- **Format 1**: `op_ut,third_party.torch-xpu-ops.test.xpu.test_nn_xpu.TestNNDeviceTypeXPU,test_case_name`
  - Skip cases wrapped with `~~` (fixed issues)
  - Only process if starts with known test types: op_ut, op_extend, e2e, benchmark, ut
- **Format 2**: `python benchmarks/dynamo/huggingface.py ...` (e2e tests)
- **Format 3**: pytest with `-k` flag

Extract:
- Test Type
- Test File: `torch-xpu-ops/test/xpu/test_xxx_xpu.py`
- Origin Test File: mapped to pytorch test path (`test/test_xxx.py`)
- Test Class: extracted from path (e.g., `TestNNDeviceTypeXPU`)
- Test Case: the actual test name

### Step 5: Extract torch_ops
Follow these rules in order:

**Rule 1**: Match from test case name using patterns:
- `test_block_addmv` → `torch.addmv`
- `test_block_addmm` → `torch.addmm`
- `test_block_triangular_solve` → `aten.triangular_solve.X`
- `test_cudnn_attention` → `scaled_dot_product_attention`
- `test_scaled_dot_product` / `test_sdpa` → `scaled_dot_product_attention`
- `test_flash_attention` → `_flash_attention_forward`
- `test_baddbmm` → `aten.baddbmm`
- `test_bmm` → `aten.bmm`
- `test_mm` → `aten.mm`
- `test_addmm` → `aten.addmm`
- `test_addmv` → `torch.addmv`
- `test_matmul` → `aten.matmul`
- And 60+ more patterns...

**Rule 2**: Extract from error message/traceback:
- Match `aten::xxx` pattern: `aten::triangular_solve.X` → `aten.triangular_solve.X`
- Match `aten._xxx` pattern: `aten::_scaled_dot_product_efficient_attention_backward`
- Match `torch.xxx` pattern in code

**Rule 3**: Use test file context as fallback:
- `test_sparse_csr.py` → sparse operations
- `test_transformers.py` → attention/transformer ops

### Step 6: Extract Error and Traceback
- **Error Message**: Match patterns like `AssertionError`, `RuntimeError`, `ValueError`, `TypeError`, `IndexError`, `KeyError`, `ImportError`, `NotImplementedError`, `AttributeError`, `InductorError`
- **Traceback**: Extract full trace starting from "Traceback (most recent call last):" or pytest format `____ TestXXX.XXX`

### Step 7: Parse E2E Test Cases
E2E test cases are benchmark tests from huggingface, timm, or torchbench. Model lists can be found at:
- https://github.com/intel/torch-xpu-ops/tree/main/.ci/benchmarks

**Model Lists:**
- **Huggingface**: AlbertForMaskedLM, BertForMaskedLM, GPT2ForSequenceClassification, XLNetLMHeadModel, hf_Albert, hf_Bert, etc.
- **TIMM**: adv_inception_v3, convnext_base, resnet50, vit_base_patch16_224, timm_vision_transformer, etc.
- **Torchbench**: BERT_pytorch, resnet18, resnet50, vgg16, etc.

**Parse E2E Info from Issue Body:**
Extract the following fields:
1. **Benchmark**: huggingface, timm, or torchbench (identified from model name)
2. **Phase**: training or inference
3. **Dtype**: bfloat16, float16, float32, int8 (from keywords in body)
4. **Test Type**: accuracy or performance (from throughputs/performance/latency keywords)
5. **Backend**: inductor or eager (from --backend flag or context)
6. **Cudagraph**: yes or no (from disable-cudagraphs flag)
7. **Reproducer**: command to reproduce

### Step 8: Create Excel Files
Create Excel with three sheets:

**Sheet 1: Issues**
Columns: Issue ID, Title, Status, Assignee, Reporter, Labels, Created Time, Updated Time, Milestone, Summary, Type, Module, Test Module, Dependency, **PR, PR Owner, PR Status**

**Sheet 2: Test Cases (UT)**
Columns: Issue ID, Test Reproducer, Test Type, Test File, Origin Test File, Test Class, Test Case, Error Message, Traceback, torch-ops, dependency

**Sheet 3: E2E Test Cases**
Columns: Issue ID, Test Reproducer, Benchmark, Phase, Dtype, Backend, Test Type, Cudagraph, Error Message, Traceback

### Step 9: Extract PR Information (New)
For each issue, extract PR information from the issue body and comments:

1. **Extract PR references**: Parse PR URLs and PR numbers from issue body using patterns:
   - `https://github.com/pytorch/pytorch/pull/172314`
   - `https://github.com/intel/torch-xpu-ops/pull/1047`
   - `PR #1234` or `PR1234`
   - `pull request #1234`

2. **Get PR details from GitHub API**: For each PR number found:
   - Fetch PR info from `https://api.github.com/repos/pytorch/pytorch/pulls/{pr_number}`
   - Get PR state (open, closed, merged)
   - Get PR owner (user login)
   - Get PR URL

3. **Add PR columns to Issues sheet**:
   - **PR**: Comma-separated list of PR URLs
   - **PR Owner**: Comma-separated list of PR owners
   - **PR Status**: Comma-separated list of PR states

**Example**:
- Issue https://github.com/intel/torch-xpu-ops/issues/2331 mentions PR https://github.com/pytorch/pytorch/pull/172314
- The script will extract PR #172314, fetch its status (closed), and owner
- Columns will show:
  - PR: https://github.com/pytorch/pytorch/pull/172314
  - PR Owner: username
  - PR Status: closed

## File Outputs
- `$RESULT_DIR/torch_xpu_ops_issues.json` - Raw issue data
- `$RESULT_DIR/torch_xpu_ops_comments.json` - Comments data
- `$RESULT_DIR/torch_xpu_ops_issues.xlsx` - Final Excel file with PR columns
(default: `~/ai_for_validation/opencode/issue_triage/result/`)

## Key Implementation Notes
1. Use GitHub token for API authentication (set GITHUB_TOKEN environment variable)
2. Filter out pull requests (check for 'pull_request' key)
3. Skip test cases wrapped with `~~` (fixed)
4. Only classify as 'build' for actual build process issues
5. Only classify as 'infrastructure' for CI/workflow issues
6. Use specific patterns before generic ones for torch_ops
7. Extract both aten:: and torch. patterns from error messages
8. Extract PR URLs from issue body and comments using regex patterns
9. Fetch PR status and owner from GitHub API for each extracted PR

### Related Files
- `$DOC_DIR/ops_dependency.csv` - Mapping of torch ops to dependency libraries
(default: `~/issue_traige/doc/`)