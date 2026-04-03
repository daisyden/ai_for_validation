# Skill: torch-xpu-ops Issue Collection and Classification

Combines intel/torch-xpu-ops issue collection with torch-ops extraction for improved classification.

## Overview
This skill first collects open issue information from the intel/torch-xpu-ops GitHub repository, then applies torch-ops extraction to improve the accuracy of torch operations classification in the generated Excel file.

## Prerequisites
- GitHub token with appropriate permissions
- Python with openpyxl installed
- Access to PyTorch test files for test case mapping

## Workflow

### Phase 1: Collect torch-xpu-ops Issues

#### Step 1: Fetch Open Issues
Use GitHub API with token to fetch all open issues (excluding PRs):
```python
# API endpoint: https://api.github.com/repos/intel/torch-xpu-ops/issues
# Parameters: state=open, per_page=100
# Filter out pull requests (items with 'pull_request' key)
```

#### Step 2: Fetch Comments
For each issue, fetch associated comments to get full context.

#### Step 3: Parse Issue Data
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

#### Step 4: Parse Test Cases
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

#### Step 5: Extract torch_ops (Initial)
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

#### Step 6: Extract Error and Traceback
- **Error Message**: Match patterns like `AssertionError`, `RuntimeError`, `ValueError`, `TypeError`, `IndexError`, `KeyError`, `ImportError`, `NotImplementedError`, `AttributeError`, `InductorError`
- **Traceback**: Extract full trace starting from "Traceback (most recent call last):" or pytest format `____ TestXXX.XXX`

#### Step 7: Parse E2E Test Cases
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

#### Step 8: Create Initial Excel File
Create Excel with three sheets:

**Sheet 1: Issues**
Columns: Issue ID, Title, Status, Assignee, Reporter, Labels, Created Time, Updated Time, Milestone, Summary, Type, Module, Test Module, Dependency

**Sheet 2: Test Cases (UT)**
Columns: Issue ID, Test Reproducer, Test Type, Test File, Origin Test File, Test Class, Test Case, Error Message, Traceback, torch-ops, dependency

**Sheet 3: E2E Test Cases**
Columns: Issue ID, Test Reproducer, Benchmark, Phase, Dtype, Backend, Test Type, Cudagraph, Error Message, Traceback

### Phase 2: Improve torch-ops Classification

#### Step 9: Load Excel and Apply Enhanced Extraction
Load the Excel file from Phase 1 and re-process the "torch-ops" column using improved extraction rules:

**Rule 1: Error message with torch.ops.aten.XXX.default pattern (HIGHEST)**
- Extract explicit op from error messages like `torch.ops.aten._convert_weight_to_int4pack.default`
- Format: `torch.ops.aten.{op_name}.default`

**Rule 2: Test name OpDB patterns**
- `torch_ops_aten__xxx` → `aten._xxx`
  - Example: `test_out_warning_torch_ops_aten__flash_attention_forward_xpu` → `aten._flash_attention_forward`
- `__refs_xxx` → `aten._xxx`
  - Example: `test_reference_numerics_extremal__refs_log10_xpu_complex64` → `aten._log10`
- `_nn_xxx` → `nn.xxx`
  - Example: `test_grad_nn_LazyConvTranspose3d_xpu_float64` → `nn.LazyConvTranspose3d`
- `_refs_xxx` → `_refs_xxx`
- `aten__xxx` → `aten.xxx`

**Rule 3: Attention-specific patterns**
- `fused_attention`, `fused_kernel` → `aten.fused_attention`
- `sdpa`, `sdp` → `aten.scaled_dot_product_attention`
- `cudnn_attention` → `aten.cudnn_attention`
- `flash_attention`, `flash_atteention` → `aten.flash_attention`
- `mem_eff_attention` → `aten.memory_efficient_attention`
- `triton_scaled_dot_product_attention` → `aten.scaled_dot_product_attention`

**Rule 4: Test case name to torch op mapping**
```
test_addmv → torch.addmv
test_addmm → torch.addmm
test_bmm → torch.bmm
test_matmul → torch.matmul
test_dot → torch.dot
test_mm → torch.mm
test_mv → torch.mv
test_Conv2d → torch.nn.functional.conv2d
test_cross_entropy → torch.nn.functional.cross_entropy
test_layernorm → torch.nn.functional.layer_norm
test_rms_norm → torch.nn.functional.rms_norm
test_softmax → torch.nn.functional.softmax
test_relu → torch.nn.functional.relu
test_gelu → torch.nn.functional.gelu
test_stft → torch.stft
test_ones → torch.ones
test_zeros → torch.zeros
test_full → torch.full
test_empty → torch.empty
test_rand → torch.rand
test_randn → torch.randn
test_randint → torch.randint
test_arange → torch.arange
test_linspace → torch.linspace
test_logspace → torch.logspace
test_tensor → torch.tensor
test_sum → torch.sum
test_mean → torch.mean
test_std → torch.std
test_neg → torch.neg
test_abs → torch.abs
test_exp → torch.exp
test_log → torch.log
test_sqrt → torch.sqrt
test_view → torch.view
test_reshape → torch.reshape
test_flatten → torch.flatten
test_squeeze → torch.squeeze
test_unsqueeze → torch.unsqueeze
test_transpose → torch.transpose
test_permute → torch.permute
test_linalg → torch.linalg
test_cholesky → torch.cholesky
test_qr → torch.qr
test_svd → torch.svd
test_norm → torch.norm
```

**Rule 5: Sparse patterns**
- `csr_matvec` → `aten.csr_matvec`
- `sparse_csr`, `SparseCSR` → `aten.sparse_csr`
- `to_sparse` → `aten.to_sparse`
- `sparse_add` → `aten.sparse_add`

**Rule 6: Other patterns**
- `vjp_linalg_xxx` → `torch.linalg.xxx`
- `rms_norm_decomp` → `aten.rms_norm`
- `grid_sampler` → `aten.grid_sampler_2d`
- `clamp_max` → `aten.clamp_max`
- `clamp_min` → `aten.clamp_min`
- `_fft_`, `fft_` → `torch.fft`
- `transformerencoder` → `torch.nn.TransformerEncoder`
- `transformer` → `torch.nn.Transformer`

#### Helper Functions

```python
def clean_op(op):
    """Remove device and dtype suffixes"""
    op = re.sub(r'_(xpu|cuda)_(float\d+|int\d+|complex\d+)$', '', op)
    op = re.sub(r'_(float\d+|int\d+|complex\d+)$', '', op)
    op = re.sub(r'_(xpu|cuda)$', '', op)
    return op

def extract_from_error_or_traceback(text):
    """Extract torch.ops.aten.XXX.default from error/traceback"""
    # Pattern: torch.ops.aten.XXX.default
    matches = re.findall(r'torch\.ops\.aten\.(\w+)\.default', text)
    for m in matches[:3]:
        found.append(f'torch.ops.aten.{m}.default')
    
    # Fallback: torch.ops.aten.XXX
    matches = re.findall(r'torch\.ops\.aten\.(\w+)', text)
    for m in matches[:3]:
        found.append(f'torch.ops.aten.{m}')
    
    # Fallback: aten::XXX
    matches = re.findall(r'aten::(\w+)', text)
    for m in matches[:3]:
        found.append(f'aten.{m}')
```

#### Step 10: Update Excel with Improved Classification
- Update the "torch-ops" column in the Test Cases sheet with the improved extraction results
- Keep other columns unchanged
- Save the updated Excel file

## File Outputs
- `/home/daisydeng/issue_traige/data/torch_xpu_ops_issues.json` - Raw issue data
- `/home/daisydeng/issue_traige/data/torch_xpu_ops_comments.json` - Comments data
- `/home/daisydeng/issue_traige/data/torch_xpu_ops_issues.xlsx` - Final Excel file with improved torch-ops classification

## Key Implementation Notes
1. Use GitHub token for API authentication
2. Filter out pull requests (check for 'pull_request' key)
3. Skip test cases wrapped with `~~` (fixed)
4. Only classify as 'build' for actual build process issues
5. Only classify as 'infrastructure' for CI/workflow issues
6. Use specific patterns before generic ones for torch_ops
7. Extract both aten:: and torch. patterns from error messages
8. Error message patterns with `.default` take highest priority in Phase 2
9. Generic categories like "ops", "tensor", "decomp" should be avoided - use specific torch op names

## Implementation Scripts

The skill includes a combined script that runs both phases:

```
~/ai_for_validation/opencode/issue_triage/torch-xpu-ops-classification/collect_and_classify.py
```

### Usage

```bash
cd ~/ai_for_validation/opencode/issue_triage/torch-xpu-ops-classification

# Run both phases (collect issues + improve classification)
python collect_and_classify.py

# Run only Phase 1 (collect issues)
python collect_and_classify.py --phase1

# Run only Phase 2 (improve classification on existing file)
python collect_and_classify.py --phase2
```

### Phase 1 Script
Located at: `~/ai_for_validation/opencode/issue_triage/torch-xpu-ops-issue-collection/generate_excel.py`

### Phase 2 Script
Located at: `~/ai_for_validation/opencode/issue_triage/torch-ops-extraction/extract_torch_ops.py`

## Related Files
- `~/issue_traige/doc/ops_dependency.csv` - Mapping of torch ops to dependency libraries

## When to Use

Use this skill when:
- Collecting new open issues from intel/torch-xpu-ops repository
- Improving torch-ops classification accuracy in existing Excel files
- Processing issues that contain complex test cases requiring enhanced extraction
