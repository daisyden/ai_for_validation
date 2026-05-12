# Issue Category Analysis Skill

## Overview
This skill provides **deep semantic categorization** of torch-xpu-ops issues. Rather than relying on keyword matching or regex scripts, the agent must understand the issue's technical domain by reading the full issue context, tracing the failure to its source component, and applying the category taxonomy with engineering judgment.

**Authoritative Category column values** (used in tracking Excel) are the 11 buckets below. Treat these as an enum — no other values are permitted in the Excel "Category" column.

---

## Canonical Category Taxonomy (11 buckets)

### Decision Priority Order

When an issue matches multiple categories, apply in this order (first match wins):

1. `Distributed` — anything tagged `[distributed]` or involving XCCL/ProcessGroup/DDP/FSDP/DTensor/symm_mem/collective ops
2. `Flash Attention` — SDPA / flash / efficient attention kernels (unless already claimed by Distributed)
3. `Inductor` — torch.compile / Dynamo / AOTAutograd / Triton codegen / benchmark failures via inductor path
4. `TorchAO` — quantization (int4/int8/fp8/PT2E quant/torchao)
5. `Sparse` — sparse tensor formats/ops
6. `Torch Ops - gemm` — matrix multiplication family (see below)
7. `Torch Ops - eltwise` — elementwise/pointwise operations (see below)
8. `Torch Ops - reduction` — reduction operations (see below)
9. `Torch Ops - others` — other ATen/native ops not fitting gemm/eltwise/reduction
10. `Torch Runtime` — torch.xpu.* runtime, memory/OOM, profiler, RNG, streams, IPC, device management
11. `Others` — CI/infra/tracking/build/doc/test-harness/meta — the catch-all

---

## Torch Operations Subcategories

### `Torch Ops - gemm`

Operations involving matrix-matrix or matrix-vector multiplication where the computational core is a GEMM (General Matrix Multiply) or convolution:

- **Matmul family**: `torch.mm`, `torch.bmm`, `torch.matmul`, `torch.addmm`, `torch.addmv`, `torch.baddbmm`, `torch.addbmm`, `torch.dot`, `torch.mv`, `torch.einsum` (when it lowers to matmul)
- **Linear**: `torch.nn.Linear`, `F.linear`
- **Convolution**: `torch.nn.Conv1d/2d/3d`, `torch.nn.ConvTranspose1d/2d/3d`, `F.conv1d/2d/3d`, depthwise conv, grouped conv
- **BLAS/LAPACK with GEMM core**: `torch.linalg.solve`, `torch.linalg.inv`, `torch.linalg.cholesky`, `torch.linalg.lu`, `torch.linalg.svd`, `torch.linalg.qr`, `torch.linalg.pinv`, `torch.linalg.ldl_solve`
- **Quantized GEMM**: `_weight_int4pack_mm`, qlinear, qconv (if NOT routed through TorchAO/Inductor)

**How to identify**: The root cause or traceback points to oneDNN matmul/conv primitives, GEMM kernels, or linalg decomposition routines. The operation is fundamentally O(n³) or O(n²·k) in nature.

### `Torch Ops - eltwise`

Operations that apply independently to each element (or small local neighborhood for normalization):

- **Unary**: `abs`, `neg`, `exp`, `log`, `sqrt`, `rsqrt`, `sin`, `cos`, `tanh`, `sigmoid`, `relu`, `gelu`, `hardswish`, `silu`, `ceil`, `floor`, `round`, `sign`, `bitwise_not`
- **Binary**: `add`, `sub`, `mul`, `div`, `pow`, `fmod`, `remainder`, `bitwise_and/or/xor`, `maximum`, `minimum`, `where`
- **Activation functions**: All `torch.nn.functional` activations
- **Normalization** (elementwise with learned params): `batch_norm`, `layer_norm`, `group_norm`, `instance_norm` — these have reduction internally but are primarily elementwise transforms
- **Type conversion / copy**: `to()`, `copy_`, dtype casting, `clone`
- **Comparison**: `eq`, `ne`, `lt`, `gt`, `le`, `ge`, `isnan`, `isinf`
- **Channel shuffle**: `native_channel_shuffle`
- **Clamp/clip**: `torch.clamp`, `torch.clip`

**How to identify**: The operation processes each element (or small fixed window) independently. Computational complexity is O(n) in the number of elements.

### `Torch Ops - reduction`

Operations that reduce one or more dimensions, producing output smaller than input:

- **Statistical**: `sum`, `mean`, `std`, `var`, `norm`, `nansum`, `nanmean`
- **Min/Max**: `min`, `max`, `amin`, `amax`, `argmin`, `argmax`
- **Cumulative**: `cumsum`, `cumprod`, `cummax`, `cummin`
- **Sorting/selection**: `sort`, `topk`, `kthvalue`, `median`, `mode`
- **Histogram**: `histc`, `histogram`, `bincount`
- **Loss functions** (contain reduction): `nll_loss`, `cross_entropy`, `ctc_loss`, `mse_loss`, `l1_loss`, `smooth_l1_loss`
- **Logical reduction**: `all`, `any`
- **Prod**: `prod`
- **Unique**: `unique`, `unique_consecutive`
- **Batch norm statistics** (when the issue is specifically about the mean/var computation, not the elementwise transform)

**How to identify**: The output has fewer elements than the input along at least one dimension. The operation aggregates information across elements.

### `Torch Ops - others`

ATen/native operations that don't fit the above three subcategories:

- **Indexing**: `index_select`, `index_add`, `index_put`, `gather`, `scatter`, `embedding`, `index_copy`
- **FFT/spectral**: `fft`, `ifft`, `rfft`, `irfft`, `stft`
- **Pooling**: `max_pool2d/3d`, `avg_pool2d/3d`, `adaptive_avg_pool2d`
- **Reshape/view**: `view`, `reshape`, `permute`, `transpose`, `contiguous`, `expand`, `unsqueeze`, `squeeze`
- **Memory/tensor creation**: `empty`, `zeros`, `ones`, `rand`, `randn`, `arange`, `linspace`, `logspace`
- **NestedTensor**: nested tensor operations
- **DLPack**: `to_dlpack`, `from_dlpack`
- **Autograd mechanics**: `backward`, `grad`, `autograd.Function` issues specific to op correctness
- **Custom ops**: `torch.library`, custom operator registration
- **Padding**: `F.pad`, `ReflectionPad`, `ReplicationPad`
- **Upsampling**: `F.interpolate`, `Upsample`

**How to identify**: The operation transforms tensor structure (indexing, reshaping) or belongs to a specialized domain (FFT, pooling) that isn't pure GEMM, elementwise, or reduction.

---

## Deep Analysis Protocol for Category Assignment

### Step 1: Identify the Failing Component

Read the issue and answer: **What component is actually broken?**

- If the error occurs INSIDE `torch.compile` / Inductor codegen / Triton JIT → `Inductor` (even if the underlying op is a conv or matmul)
- If the error occurs in the ATen op kernel itself (eager mode or the op's SYCL implementation) → `Torch Ops - *`
- If the error occurs in ProcessGroup / collective communication → `Distributed`
- If the error occurs in the SDPA/flash-attention kernel → `Flash Attention`

### Step 2: Trace the Root Cause

Look at:
- **Stack trace**: Which file/function is at the top of the error?
- **Root cause field**: What component does our analysis point to?
- **Test file path**: `test/distributed/` → Distributed; `test_ops` → Torch Operations; `benchmarks/dynamo/` → Inductor
- **Labels**: `module: distributed`, `module: inductor`, `module: ao`, etc.

### Step 3: Apply Priority Order for Ambiguous Cases

Example disambiguations:
- "Conv accuracy fails under torch.compile" → **Inductor** (the compile path is broken, not the eager conv kernel)
- "Conv accuracy fails in eager mode" → **Torch Ops - gemm** (the conv kernel itself is wrong)
- "SDPA crashes in distributed test" → **Distributed** (priority rule: Distributed > Flash Attention)
- "INT8 quantized model perf regression" → **TorchAO** (quantization is the domain)
- "test_cow_input matmul" → **Torch Ops - gemm** (COW is the mechanism but matmul kernel is the root cause)
- "torch.topk raises error for large dim" → **Torch Ops - reduction** (topk is a selection/reduction op)
- "torch.clip doesn't raise for float16 overflow" → **Torch Ops - eltwise** (clip/clamp is elementwise)

### Step 4: Subcategorize Torch Operations

Once you've determined the issue is `Torch Ops`, decide the subcategory:

1. **Is the core computation a matrix multiply or convolution?** → `gemm`
2. **Is the core computation elementwise (O(n), independent per element)?** → `eltwise`
3. **Does the operation reduce dimensions (output smaller than input)?** → `reduction`
4. **None of the above?** → `others`

When an op has both elementwise and reduction components (e.g., `batch_norm` = mean/var reduction + normalize elementwise):
- If the bug is in the reduction statistics computation → `reduction`
- If the bug is in the elementwise normalization transform → `eltwise`
- If unclear, default to `eltwise` for normalization ops since their primary API purpose is the elementwise transform

### Step 5: Generate Category Reason

Write a concise (5-15 word) reason. Examples:
- "Convolution backward kernel accuracy (oneDNN)"
- "torch.compile Triton codegen failure on XPU"
- "ProcessGroupXCCL missing split_group API"
- "SDPA flash attention kernel crash at large seq_len"
- "index_select kernel performance (indexing op)"
- "torch.std overflow on large input (statistical reduction)"
- "channel_shuffle validation error (elementwise transform)"
- "FFT spectral op oneMKL backend failure"

---

## Anti-Patterns (DO NOT)

1. **DO NOT** categorize based on test file name alone. `test_ops.py` tests can exercise inductor paths; `test_inductor` can reveal op bugs.
2. **DO NOT** assign `Inductor` just because `torch.compile` appears in the reproducer. If the root cause is in the eager kernel and compile just exposes it, use the op category.
3. **DO NOT** assign `Torch Ops - gemm` to every linalg issue. Only linalg ops with GEMM as their computational core (solve, cholesky, svd, etc.) belong there. `linalg.norm` is a reduction.
4. **DO NOT** use keyword scripts to categorize. The same word ("attention") appears in Flash Attention issues AND in TransformerEncoder issues that are actually Inductor bugs. Semantic understanding is required.
5. **DO NOT** assign subcategories based on words in the title alone. "accuracy" in the title doesn't tell you whether it's gemm/eltwise/reduction — you must identify which op failed.

---

## Quick Mapping: Old Categories → Canonical

| Old label (deprecated) | Canonical target |
|---|---|
| Torch Operations | One of: `Torch Ops - gemm`, `Torch Ops - eltwise`, `Torch Ops - reduction`, `Torch Ops - others` |
| Test Infrastructure | `Others` (unless testing a specific op → that op's bucket) |
| Operator Implementation | `Torch Ops - *` (appropriate subcategory) |
| Build/Compilation | `Others` |
| Performance | Same bucket as the affected component |
| Numerical Accuracy | Same bucket as the affected op |
| Feature Gap | Same bucket as the feature |
| Documentation | `Others` |
| CI/CD | `Others` |
| Environment/Driver | `Torch Runtime` |
| PT2E | `Inductor` (or `TorchAO` if PT2E quantization) |
| Inductor/Compilation | `Inductor` |

---

## Skill Metadata

- **Version**: 2.0.0
- **Created**: 2026-04-20
- **Updated**: 2026-05-12
- **Requires**: Issue text, error log, root cause, stack trace, labels
- **Related Skills**: SKILL_Priority_Analysis.md, SKILL_Triage_Logic.md
