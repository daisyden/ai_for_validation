# Torch-ops Extraction Skill

Extracts torch ops from PyTorch XPU test case data in Excel format.

## When to Use

Use when processing torch_xpu_ops_issues.xlsx files with a "Test Cases" sheet to populate the "torch-ops" column with accurate operation names.

## Workflow

### Input Format
- Excel file with "Test Cases" sheet containing columns:
  - `Test Case`: Test case name (e.g., `test_out_addmv_xpu`)
  - `Error Message`: Error message from test failure
  - `Traceback`: Python traceback

### Extraction Rules (in priority order)

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

### Helper Functions

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

## Output Format

- Excel file with two sheets:
  - "Issues": Original issue data (14 columns)
  - "Test Cases": Test case data with "torch-ops" column populated

## Notes

- Unknown cases remain as "unknown" when no torch op can be identified
- Error message patterns with `.default` take highest priority
- Generic categories like "ops", "tensor", "decomp" should be avoided - use specific torch op names
- Test file path can be used as fallback for category inference
