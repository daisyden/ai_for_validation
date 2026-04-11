# Create torch_xpu_ops_issues.xlsx

## Overview
This skill creates the torch_xpu_ops_issues.xlsx Excel file by collecting open issues from intel/torch-xpu-ops GitHub repository and extracting test case information.

## When to Use
- When need to generate a fresh Excel file with all open issues from torch-xpu-ops
- When issues need to be re-collected from GitHub

## Workflow Steps

### Step 1: Fetch Issues from GitHub
The script automatically fetches issues if JSON files don't exist:
- API: `https://api.github.com/repos/intel/torch-xpu-ops/issues?state=open&per_page=100`
- Filters out pull requests
- Saves to: `/home/daisydeng/issue_traige/data/torch_xpu_ops_issues.json`

### Step 2: Fetch Comments
For each issue, fetch associated comments:
- API: `https://api.github.com/repos/intel/torch-xpu-ops/issues/{issue_num}/comments`
- Saves to: `/home/daisydeng/issue_traige/data/torch_xpu_ops_comments.json`

### Step 3: Parse Issue Data
Extract fields:
- **Basic Info**: Issue ID, Title, Status, Assignee, Reporter, Labels, Created/Updated Time, Milestone
- **Classification**: Type (bug/feature/performance), Module (distributed/inductor/autograd/etc), Test Module (ut/e2e/build)
- **PR Extraction**: Only PRs that fix the issue
  - Extract from issue body and comments
  - Skip intel/torch-xpu-ops PRs if "Closed with unmerged commits"
  - Skip pytorch/pytorch PRs unless they have "Merged" label

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

**Note**: Fields `Error Message`, `Traceback`, and `torch-ops` will be blank - populated later by `test_result_analysis/` skill.

### Step 5: Parse E2E Test Cases
E2E test cases are benchmark tests from huggingface, timm, or torchbench. Model lists can be found at:
- https://github.com/intel/torch-xpu-ops/tree/main/.ci/benchmarks

**Model Lists:**
- **Huggingface**: AlbertForMaskedLM, BertForMaskedLM, GPT2ForSequenceClassification, XLNetLMHeadModel, hf_Albert, hf_Bert, etc.
- **TIMM**: adv_inception_v3, convnext_base, resnet50, vit_base_patch16_224, timm_vision_transformer, etc.
- **Torchbench**: BERT_pytorch, resnet18, resnet50, vgg16, etc.

**Parse E2E Info from Issue Body:**
Extract the following fields:
1. **Benchmark**: huggingface, timm, or torchbench (identified from model name)
2. **Model**: Model name from the benchmark suite
3. **Phase**: training or inference
4. **Dtype**: bfloat16, float16, float32, int8 (from keywords in body)
5. **AMP**: auto mixed precision setting
6. **Backend**: inductor or eager (from --backend flag or context)
7. **Test Type**: accuracy or performance (from throughputs/performance/latency keywords)
8. **Cudagraph**: yes or no (from disable-cudagraphs flag)
9. **Reproducer**: command to reproduce

### Step 6: Create Excel File
Three sheets:

1. **Issues**: Columns: Issue ID, Title, Status, Assignee, Reporter, Labels, Created Time, Updated Time, Milestone, Summary, Type, Module, Test Module, Dependency, PR, PR Owner, PR Status, PR Description

2. **Test Cases** (~2017 rows): Columns populated by this skill:
   - Issue ID, Test Reproducer, Test Type, Test File, Origin Test File, Test Class, Test Case
   - Fields left blank for test_result_analysis to fill: Error Message, Traceback, torch-ops, dependency

3. **E2E Test Cases** (~119 rows): Columns: Issue ID, Test Reproducer, Benchmark, Model, Phase, Dtype, AMP, Backend, Test Type, Cudagraph, Error Message, Traceback

**Note**: Error Message, Traceback, and torch-ops are populated by `test_result_analysis/Test_Cases/test_cases_processor.py` using CI XML results.

## Usage

```bash
cd ~/ai_for_validation/opencode/issue_triage/issue_analysis/issue_basic_info_extraction
python3 generate_excel.py
```

## Output
- `$RESULT_DIR/torch_xpu_ops_issues.xlsx` (default: `~/ai_for_validation/opencode/issue_triage/result/`)

## Prerequisites
- GitHub token with repo access (set GITHUB_TOKEN env var)
- Python with: openpyxl, requests, json

## Related Skills
- ../test_result_analysis/Test_Cases: Add CI test results and case existence analysis
- ../test_result_analysis/check-xpu-test-existence: Check if XPU test exists in torch-xpu-ops
- ../test_result_analysis/check-cuda-test-existence: Check if CUDA test exists in PyTorch
  - Location: test_result_analysis/check-xpu-test-existence/
  - Location: test_result_analysis/check-cuda-test-existence/
