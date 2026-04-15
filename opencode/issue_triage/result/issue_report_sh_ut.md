# Torch XPU Ops Issue Report (upstream_ut only)

**Repository:** [intel/torch-xpu-ops](https://github.com/intel/torch-xpu-ops)

**CI Data Sources:**
- Torch-XPU-OPS Nightly CI: XML files from `Inductor-XPU-UT-Data-*` + `Inductor-wheel-nightly-LTS2-XPU-E2E-Data-*`
- Stock PyTorch XPU CI: XML files from `test-default-*-linux.idc.xpu_*.zip`

**Generated:** 2026-04-15 09:32:15
**Total Issues:** 74

---

## <span id='toc'>Index</span>

- [1. Summary (#1-summary)](#1-summary) - 74 issues |
- [2. Need Investigation by Category (#2-need-investigation-by-category)](#2-need-investigation-by-category) - 71 issues |
   - [Torch Runtime](#torch-runtime) - 21 issues |
   - [Inductor/Compilation](#inductor-compilation) - 15 issues |
   - [Others](#others) - 8 issues |
   - [Flash Attention/Transformer](#flash-attention-transformer) - 8 issues |
   - [Dtype/Precision](#dtype-precision) - 6 issues |
   - [Sparse](#sparse) - 6 issues |
   - [Feature Not Supported](#feature-not-supported) - 2 issues |
   - [PT2E](#pt2e) - 2 issues |
   - [TorchAO](#torchao) - 2 issues |
   - [Torch Operations](#torch-operations) - 1 issues |
- [3. Other Actions by Type (#3-other-actions-by-type)](#3-other-actions-by-type) - 3 issues |
   - [add to skiplist](#add-to-skiplist) - 3 issues |
- [4. Last Week Issues (#4-last-week-issues)](#4-last-week-issues) - 0 issues |
- [5. Stale Issues (#5-stale-issues)](#5-stale-issues) - 48 issues |
- [6. Duplicated Issues (#6-duplicated-issues)](#6-duplicated-issues) - 8 issues |
- [7. Issues with Dependency (#7-issues-with-dependency)](#7-issues-with-dependency) - 4 issues |
- [8. Statistics (#8-statistics)](#8-statistics) - Dependency stats |

---

## <span id='1-summary'>1. Summary</span>

**Total: 74 issues**

| # | Action Type | Count | Link |
|--:|-------------|-------|------|
| 1 | [Need Investigation](#need-investigation) | 71 | [View Issues](#need-investigation) |
| 2 | [add to skiplist](#add-to-skiplist) | 3 | [View Issues](#add-to-skiplist) |
| | **Total** | **74** | |

## <span id='2-need-investigation-by-category'>2. Need Investigation by Category</span>

**Total: 71 issues** - Issues requiring further investigation

### <span id='torch-runtime'>Torch Runtime</span> (21 issues)

| # | ID | Title | Action Reason | Summary | Assignee | Owner Transfer | PR Status | Test Module |
|--:|----|-------|---------------|---------|----------|----------------|-----------|-------------|
| 1 | 1893 | [upstream_ut] oneDNN accuracy issues in test_ops_xpu.py | Fix dtype precision issue - adjust numerical tolerance or use higher precision computations. Implement dtype-specific kernel or configure proper precision settings for XPU. | [upstream_ut] oneDNN accuracy issues in test_ops_xpu.py | chunhuanMeng | chunhuanMeng |  | ut |
| 2 | 2253 | the supported dtypes are not align with cuda | Fix dtype precision issue - adjust numerical tolerance or use higher precision computations. Implement dtype-specific kernel or configure proper precision settings for XPU. | the supported dtypes are not align with cuda | daisyden | daisyden |  | ut |
| 3 | 2255 | [upstream_ut] RuntimeError: Long is not supported in oneDNN | PR closed but no failed tests - verify if issue still reproduces | [upstream_ut] RuntimeError: Long is not supported in oneDNN | daisyden | daisyden |  | ut |
| 4 | 2287 | [upstream_ut] test_python_ref issues | Investigate the issue with detailed traceback analysis. Run test case to reproduce and identify root cause for XPU fix. | [upstream_ut] test_python_ref issues | yucai-intel | yucai-intel |  | ut |
| 5 | 2295 | [upstream_ut][xpu][test]nn/test_embedding.py::TestEmbeddingNNDeviceTypeXPU::test_embedding_bag_device_xpu_int32_int32_float64 meet AssertionError: Tensor-likes are not close! | Investigate test failure - analyze traceback and fix the root cause. Run test with detailed logging to identify the specific failure point. | [upstream_ut][xpu][test]nn/test_embedding.py::TestEmbeddingNNDeviceTypeXPU::test_embedding_bag_device_xpu_int32_int32_float64 meet AssertionError: Ten | yucai-intel | yucai-intel |  | ut |
| 6 | 2301 | [upstream_ut] dtypes not align with OpInfo | Fix dtype precision issue - adjust numerical tolerance or use higher precision computations. Implement dtype-specific kernel or configure proper precision settings for XPU. | [upstream_ut] dtypes not align with OpInfo | daisyden | daisyden |  | ut |
| 7 | 2436 | [upstream_ut] AttributeError: 'NoneType' object has no attribute 'clone' | Fix error: 'NoneType' object has no attribute 'clone'... Investigate root cause and implement proper fix for XPU backend. | [upstream_ut] AttributeError: 'NoneType' object has no attribute 'clone' | daisyden | daisyden |  | ut |
| 8 | 2512 | [upstream_ut] RuntimeError: _histc_xpu does not have a deterministic implementation, but you set 'torch.use_deter | Implement XPU-specific kernel or backend dispatch for the affected operation. Add proper device type check and XPU kernel implementation in native_functions.yaml. | [upstream_ut] RuntimeError: _histc_xpu does not have a deterministic implementation, but you set 'torch.use_deter | chunhuanMeng | chunhuanMeng |  | ut |
| 9 | 2531 | [upstream_ut] AssertionError: Torch not compiled with CUDA enabled | Implement XPU-specific kernel or backend dispatch for the affected operation. Add proper device type check and XPU kernel implementation in native_functions.yaml. | [upstream_ut] AssertionError: Torch not compiled with CUDA enabled | daisyden | daisyden |  | ut |
| 10 | 2536 | Title: [upstream_ut] AttributeError: module 'torch._C' has no attribute | All test cases passed on XPU/stock - issue is resolved | Title: [upstream_ut] AttributeError: module 'torch._C' has no attribute | daisyden | daisyden |  | ut |
| 11 | 2541 | Title: [upstream_ut] RuntimeError: could not construct a memory descriptor using strides | All test cases passed on XPU/stock - issue is resolved | Title: [upstream_ut] RuntimeError: could not construct a memory descriptor using strides | yucai-intel | yucai-intel |  | ut |
| 12 | 2697 | Title: [upstream_ut] RuntimeError: Expected to find ", 0, " but did not find it | Investigate test failure - analyze traceback and fix the root cause. Run test with detailed logging to identify the specific failure point. | Title: [upstream_ut] RuntimeError: Expected to find ", 0, " but did not find it | chunhuanMeng | chunhuanMeng |  | e2e |
| 13 | 2698 | Title: [upstream_ut] RuntimeError: FlashAttentionForwardXPU only support headdim 64,96,128,192 | Fix flash attention operation for XPU - handle head dimension and dropout constraints. Update FlashAttentionForwardXPU kernel to support required configurations. | Title: [upstream_ut] RuntimeError: FlashAttentionForwardXPU only support headdim 64,96,128,192 | chunhuanMeng, LuFinch | chunhuanMeng, LuFinch |  | ut |
| 14 | 2720 | [upstream_ut] RuntimeError: false INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/native/DispatchStub.cpp":275 | Investigate test failure - analyze traceback and fix the root cause. Run test with detailed logging to identify the specific failure point. | [upstream_ut] RuntimeError: false INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/native/DispatchStub.cpp":275 | CuiYifeng | CuiYifeng |  | ut |
| 15 | 2800 | AttributeError: 'torch._C._XpuDeviceProperties' object has no attribute 'major' | Implement XPU-specific kernel or backend dispatch for the affected operation. Add proper device type check and XPU kernel implementation in native_functions.yaml. | AttributeError: 'torch._C._XpuDeviceProperties' object has no attribute 'major' | guangyey | guangyey |  | ut |
| 16 | 2891 | RuntimeError: Expected to find "(262144, 0, 512, 1" but did not find it | Implement XPU-specific kernel or backend dispatch for the affected operation. Add proper device type check and XPU kernel implementation in native_functions.yaml. | RuntimeError: Expected to find "(262144, 0, 512, 1" but did not find it | chunhuanMeng | chunhuanMeng |  | e2e |
| 17 | 2999 | KeyError: 'eager_numerics.use_pytorch_libdevice' | Implement XPU-specific kernel or backend dispatch for the affected operation. Add proper device type check and XPU kernel implementation in native_functions.yaml. | KeyError: 'eager_numerics.use_pytorch_libdevice' | daisyden | daisyden |  | ut |
| 18 | 3004 | TypeError: _xpu_recordMemoryHistory(): incompatible function arguments | Investigate test failure - analyze traceback and fix the root cause. Run test with detailed logging to identify the specific failure point. | TypeError: _xpu_recordMemoryHistory(): incompatible function arguments | guangyey | guangyey |  | ut |
| 19 | 3128 | [upstream_ut] AssertionError: RuntimeError not raised by <lambda> | Fix error: RuntimeError not raised by <lambda>... Investigate root cause and implement proper fix for XPU backend. | [upstream_ut] AssertionError: RuntimeError not raised by <lambda> | daisyden | daisyden |  | ut |
| 20 | 3129 | [upstream_ut] AssertionError: UserWarning not triggered | Investigate test failure - analyze traceback and fix the root cause. Run test with detailed logging to identify the specific failure point. | [upstream_ut] AssertionError: UserWarning not triggered | daisyden | daisyden |  | ut |
| 21 | 3132 | [upstream_ut] transfomers test reports RuntimeError: No available kernel. Aborting execution. | Fix flash attention operation for XPU - handle head dimension and dropout constraints. Update FlashAttentionForwardXPU kernel to support required configurations. | [upstream_ut] transfomers test reports RuntimeError: No available kernel. Aborting execution. | LuFinch | LuFinch |  | ut |
| | | **Subtotal: 21 issues** | | | | | |

### <span id='inductor-compilation'>Inductor/Compilation</span> (15 issues)

| # | ID | Title | Action Reason | Summary | Assignee | Owner Transfer | PR Status | Test Module |
|--:|----|-------|---------------|---------|----------|----------------|-----------|-------------|
| 22 | 2024 | AssertionError: Torch not compiled with CUDA enabled | Implement XPU-specific kernel or backend dispatch for the affected operation. Add proper device type check and XPU kernel implementation in native_functions.yaml. | AssertionError: Torch not compiled with CUDA enabled | daisyden | daisyden |  | ut |
| 23 | 2169 | Frame size comparison failed in test_size_comparison_no_recompile | Add test case to skiplist or mark as expected failure (xfail) for XPU. Verify test exists in torch-xpu-ops and add to skip list if not applicable to XPU. | Frame size comparison failed in test_size_comparison_no_recompile | guangyey | guangyey |  | ut |
| 24 | 2329 | [upstream_ut] feature missing: get_device_tflops and get_drams_gbps | PR closed but no failed tests - verify if issue still reproduces | [upstream_ut] feature missing: get_device_tflops and get_drams_gbps | etaf | etaf |  | ut |
| 25 | 2554 | [upstream_ut] AssertionError: AssertionError not raised | Investigate test failure - analyze traceback and fix the root cause. Run test with detailed logging to identify the specific failure point. | [upstream_ut] AssertionError: AssertionError not raised | daisyden | daisyden |  | ut |
| 26 | 2609 | [upstream_ut] torch._inductor.exc.InductorError: CppCompileError: C++ compile error | Fix Inductor compilation issue - implement proper XPU lowering or decomposition. Add XPU-specific inductor lowering or fix fallback/decomposition conflict. | [upstream_ut] torch._inductor.exc.InductorError: CppCompileError: C++ compile error | daisyden | daisyden |  | ut |
| 27 | 2611 | [upstream_ut] AssertionError: Tensor-likes are not equal! in test_compile_subprocess | Fix Inductor compilation issue - implement proper XPU lowering or decomposition. Add XPU-specific inductor lowering or fix fallback/decomposition conflict. | [upstream_ut] AssertionError: Tensor-likes are not equal! in test_compile_subprocess | daisyden | daisyden |  | ut |
| 28 | 2613 | [upstream_ut] AssertionError: Tensor-likes are not equal! in test_compile_subprocess.py | Investigate test failure - analyze traceback and fix the root cause. Run test with detailed logging to identify the specific failure point. | [upstream_ut] AssertionError: Tensor-likes are not equal! in test_compile_subprocess.py | daisyden | daisyden |  | ut |
| 29 | 2620 | [upstream_ut] AssertionError: dtype is needed to compute eps1 when eps1 is unset | Investigate test failure - analyze traceback and fix the root cause. Run test with detailed logging to identify the specific failure point. | [upstream_ut] AssertionError: dtype is needed to compute eps1 when eps1 is unset | daisyden | daisyden |  | ut |
| 30 | 2694 | Title: [upstream_ut] AssertionError: Tensor-likes are not equal! with test_randint tests | PR closed but no failed tests - verify if issue still reproduces | Title: [upstream_ut] AssertionError: Tensor-likes are not equal! with test_randint tests | daisyden | daisyden |  | ut |
| 31 | 2806 | CompiledAOTI need XPU support | PR closed but no failed tests - verify if issue still reproduces | CompiledAOTI need XPU support | daisyden | daisyden |  | ut |
| 32 | 2810 | AssertionError: Object comparison failed: Decimal('2.938735877055718769921841343055614194546[51 chars]-39') != Decimal('0') | Investigate test failure - analyze traceback and fix the root cause. Run test with detailed logging to identify the specific failure point. | AssertionError: Object comparison failed: Decimal('2.938735877055718769921841343055614194546[51 chars]-39') != Decimal('0') | daisyden | daisyden |  | ut |
| 33 | 2888 | torch._inductor.exc.InductorError: AssertionError: Conversions between float8_e5m2 and float8_e4m3fn is not supported! | Fix dtype precision issue - adjust numerical tolerance or use higher precision computations. Implement dtype-specific kernel or configure proper precision settings for XPU. | torch._inductor.exc.InductorError: AssertionError: Conversions between float8_e5m2 and float8_e4m3fn is not supported! | Stonepia | Stonepia |  | ut |
| 34 | 2958 | AssertionError of test_dtensor_basic_compile | Investigate test failure - analyze traceback and fix the root cause. Run test with detailed logging to identify the specific failure point. | AssertionError of test_dtensor_basic_compile | daisyden | daisyden |  | ut |
| 35 | 2997 | AssertionError of test_linear_and_cel_max_autotune | Investigate test failure - analyze traceback and fix the root cause. Run test with detailed logging to identify the specific failure point. | AssertionError of test_linear_and_cel_max_autotune | etaf | etaf |  | ut |
| 36 | 3187 | PyTorch XPU gpu_cpp_wrapper fails with InductorError NotImplementedError | Fix Inductor XPU wrapper: Implement gpu_cpp_wrapper support for XPU in torch/_inductor/codegen/wrapper.py - add XPU-specific code generation. | PyTorch XPU gpu_cpp_wrapper fails with InductorError NotImplementedError | CuiYifeng | CuiYifeng |  | ut |
| | | **Subtotal: 15 issues** | | | | | |

### <span id='others'>Others</span> (8 issues)

| # | ID | Title | Action Reason | Summary | Assignee | Owner Transfer | PR Status | Test Module |
|--:|----|-------|---------------|---------|----------|----------------|-----------|-------------|
| 37 | 2015 | inf is returned by nn.TransformerEncoderLayer | Fix dtype precision issue - adjust numerical tolerance or use higher precision computations. Implement dtype-specific kernel or configure proper precision settings for XPU. | inf is returned by nn.TransformerEncoderLayer | yucai-intel | yucai-intel |  | ut |
| 38 | 2270 | Backend Compatibility Error in test/xpu/test_decomp.py | Implement attention operation for XPU backend or enable FlashAttentionForwardXPU. Register aten._efficient_attention_forward or flash_attention_forward for XPU. | Backend Compatibility Error in test/xpu/test_decomp.py | LuFinch | LuFinch |  | ut |
| 39 | 2285 | Support efficient attention | Fix flash attention operation for XPU - handle head dimension and dropout constraints. Update FlashAttentionForwardXPU kernel to support required configurations. | Support efficient attention | chunhuanMeng | chunhuanMeng |  | ut |
| 40 | 2783 | [Bug Skip]: Key "xpu" is missing from dict "driver" in test_svd | Add test case to skiplist or mark as expected failure (xfail) for XPU. Verify test exists in torch-xpu-ops and add to skip list if not applicable to XPU. | [Bug Skip]: Key "xpu" is missing from dict "driver" in test_svd | daisyden | daisyden |  | ut |
| 41 | 3033 | [Bug Skip]: Softmax tolerance | Investigate the issue with detailed traceback analysis. Run test case to reproduce and identify root cause for XPU fix. | [Bug Skip]: Softmax tolerance | chunhuanMeng | chunhuanMeng |  | ut |
| 42 | 3140 | [upstream_ut] RuntimeError: FlashAttentionForwardXPU does not only support dropout > 0.0 yet | Fix flash attention operation for XPU - handle head dimension and dropout constraints. Update FlashAttentionForwardXPU kernel to support required configurations. | [upstream_ut] RuntimeError: FlashAttentionForwardXPU does not only support dropout > 0.0 yet | LuFinch | LuFinch |  | ut |
| 43 | 3143 | NotImplementedError: The operator 'aten::_scaled_dot_product_efficient_attention_backward' is not currently implemented for the XPU device. | Implement attention operation for XPU backend or enable FlashAttentionForwardXPU. Register aten._efficient_attention_forward or flash_attention_forward for XPU. | NotImplementedError: The operator 'aten::_scaled_dot_product_efficient_attention_backward' is not currently implemented for the XPU device. | LuFinch | LuFinch |  | ut |
| 44 | 3170 | Unskip test_bmm_windows_error_xpu_float64 | Fix error: /__w/torch-xpu-ops/torch-xpu-ops/pytorch/third_party/torch-xpu-ops/test/xpu/test_sparse_xpu.py:1965:... Investigate root cause and implement proper fix for XPU backend. | Unskip test_bmm_windows_error_xpu_float64 | jenniew | jenniew |  | ut |
| | | **Subtotal: 8 issues** | | | | | |

### <span id='flash-attention-transformer'>Flash Attention/Transformer</span> (8 issues)

| # | ID | Title | Action Reason | Summary | Assignee | Owner Transfer | PR Status | Test Module |
|--:|----|-------|---------------|---------|----------|----------------|-----------|-------------|
| 45 | 2442 | [Bug Skip]: NotImplementedError: Could not run 'aten::_flash_attention_forward' with arguments from the 'CPU' backend | Fix flash attention operation for XPU - handle head dimension and dropout constraints. Update FlashAttentionForwardXPU kernel to support required configurations. | [Bug Skip]: NotImplementedError: Could not run 'aten::_flash_attention_forward' with arguments from the 'CPU' backend | daisyden, LuFinch | daisyden, LuFinch |  | ut |
| 46 | 2802 | Three aten._scaled_dot_product_flash_attention issues | Fix flash attention operation for XPU - handle head dimension and dropout constraints. Update FlashAttentionForwardXPU kernel to support required configurations. | Three aten._scaled_dot_product_flash_attention issues | LuFinch | LuFinch |  | ut |
| 47 | 2853 | [upstream_ut] torch.ops.aten._flash_attention_forward lack of support for XPU. | Fix flash attention operation for XPU - handle head dimension and dropout constraints. Update FlashAttentionForwardXPU kernel to support required configurations. | [upstream_ut] torch.ops.aten._flash_attention_forward lack of support for XPU. | LuFinch | LuFinch |  | ut |
| 48 | 3007 | AssertionError: Scalars are not equal! with test_flash_attention_dynamic | PR closed but no failed tests - verify if issue still reproduces | AssertionError: Scalars are not equal! with test_flash_attention_dynamic | daisyden | daisyden |  | e2e |
| 49 | 3126 | [upstream_ut] Two NestedTensor issue with flash attention | Fix flash attention operation for XPU - handle head dimension and dropout constraints. Update FlashAttentionForwardXPU kernel to support required configurations. | [upstream_ut] Two NestedTensor issue with flash attention | daisyden | daisyden |  | ut |
| 50 | 3133 | [upstream_ut] RuntimeError: scaled_dot_product_attention: If inputs are nested tensors they must be contiguous | Fix flash attention operation for XPU - handle head dimension and dropout constraints. Update FlashAttentionForwardXPU kernel to support required configurations. | [upstream_ut] RuntimeError: scaled_dot_product_attention: If inputs are nested tensors they must be contiguous | daisyden | daisyden |  | ut |
| 51 | 3136 | [upstream_ut] AssertionError: False is not true in test_transformers | Fix flash attention operation for XPU - handle head dimension and dropout constraints. Update FlashAttentionForwardXPU kernel to support required configurations. | [upstream_ut] AssertionError: False is not true in test_transformers | LuFinch | LuFinch |  | ut |
| 52 | 3141 | [upstream_ut] RuntimeError: FlashAttentionForwardXPU only support headdim 64,96,128,192 | Fix flash attention operation for XPU - handle head dimension and dropout constraints. Update FlashAttentionForwardXPU kernel to support required configurations. | [upstream_ut] RuntimeError: FlashAttentionForwardXPU only support headdim 64,96,128,192 | LuFinch | LuFinch |  | ut |
| | | **Subtotal: 8 issues** | | | | | |

### <span id='dtype-precision'>Dtype/Precision</span> (6 issues)

| # | ID | Title | Action Reason | Summary | Assignee | Owner Transfer | PR Status | Test Module |
|--:|----|-------|---------------|---------|----------|----------------|-----------|-------------|
| 53 | 2482 | test_dtypes issue introduced by pytorch test sample input updates | Investigate AMP inference accuracy - check gradient scaling and mixed precision implementation. Add AMP-specific tolerance adjustment for XPU or verify cuDNN/MKLDNN backend configuration. | test_dtypes issue introduced by pytorch test sample input updates | daisyden | daisyden |  | ut |
| 54 | 2615 | [Bug Skip]: New failures RuntimeError: Unsupported dtype Half / RuntimeError: Unsupported dtype torch.float16 | Fix precision issue for float16 dtype - adjust numerical tolerance or use higher precision intermediate. Implement fp16-specific kernel with stable computation or add torchao precision tuning. | [Bug Skip]: New failures RuntimeError: Unsupported dtype Half / RuntimeError: Unsupported dtype torch.float16 | CuiYifeng | CuiYifeng |  | ut |
| 55 | 3137 | [upstream_ut] RuntimeError: expected scalar type Half but found Float | Fix dtype precision issue - adjust numerical tolerance or use higher precision computations. Implement dtype-specific kernel or configure proper precision settings for XPU. | [upstream_ut] RuntimeError: expected scalar type Half but found Float | LuFinch | LuFinch |  | ut |
| 56 | 3163 | [Bug Skip]: Object comparison failed: torch.int64 != torch.int32 in test_sparse_add | Fix sparse index dtype: In torch/sparse/__init__.py, fix crow_indices dtype conversion - ensure int64 to int32 alignment for XPU. | [Bug Skip]: Object comparison failed: torch.int64 != torch.int32 in test_sparse_add | chunhuanMeng | chunhuanMeng |  | ut |
| 57 | 3177 | Accuracy gap of BF16/FP16 test_block_addmm | Fix block_addmm BF16 accuracy: In torch/sparse/_triton/ops.py, adjust precision tolerance or fix BF16 computation in CSR block_addmm on XPU. | Accuracy gap of BF16/FP16 test_block_addmm | jenniew | jenniew |  | ut |
| 58 | 3238 | The supported dtypes of _refs.stft is not aligned to stft | Align stft dtypes: In torch/_decomp/decompositions.py and torch/signal/windows.py, align supported dtypes for _refs.stft with stft - add complex32 support. | The supported dtypes of _refs.stft is not aligned to stft | daisyden | daisyden |  | ut |
| | | **Subtotal: 6 issues** | | | | | |

### <span id='sparse'>Sparse</span> (6 issues)

| # | ID | Title | Action Reason | Summary | Assignee | Owner Transfer | PR Status | Test Module |
|--:|----|-------|---------------|---------|----------|----------------|-----------|-------------|
| 59 | 2214 | test/test_sparse.py::TestSparseAnyXPU::test_gradcheck_mm expected error message not match | Implement XPU-specific kernel or backend dispatch for the affected operation. Add proper device type check and XPU kernel implementation in native_functions.yaml. | test/test_sparse.py::TestSparseAnyXPU::test_gradcheck_mm expected error message not match | jenniew | jenniew |  | ut |
| 60 | 2229 | test/test_sparse_csr.py::TestSparseCompressedCPU::test_invalid_input meet message not match | Fix error: /__w/torch-xpu-ops/torch-xpu-ops/pytorch/third_party/torch-xpu-ops/test/xpu/test_sparse_csr_xpu.py:1... Investigate root cause and implement proper fix for XPU backend. | test/test_sparse_csr.py::TestSparseCompressedCPU::test_invalid_input meet message not match | jenniew | jenniew |  | ut |
| 61 | 2244 | test/test_sparse_csr.py::TestSparseCSRXPU::test_block_addmm meet RuntimeError: empty_sparse_compressed expected sparse compressed (non-block) tensor layout but got SparseBsr | Fix error: Tensor-likes are not close!... Investigate root cause and implement proper fix for XPU backend. | test/test_sparse_csr.py::TestSparseCSRXPU::test_block_addmm meet RuntimeError: empty_sparse_compressed expected sparse compressed (non-block) tensor l | jenniew | jenniew |  | ut |
| 62 | 2245 | oneDNN matmul received incorrect shape in test/test_sparse_csr.py::TestSparseCSRXPU::test_addmm_errors_xpu_float32 | Fix OneDNN/TorchAO compatibility - update kernel or use compatible version. Update OneDNN backend or adjust quantization configuration for XPU. | oneDNN matmul received incorrect shape in test/test_sparse_csr.py::TestSparseCSRXPU::test_addmm_errors_xpu_float32 | CuiYifeng | CuiYifeng |  | ut |
| 63 | 2283 | [upstream_ut] sparse._sampled_addmm is not supported | Implement XPU-specific kernel or backend dispatch for the affected operation. Add proper device type check and XPU kernel implementation in native_functions.yaml. | [upstream_ut] sparse._sampled_addmm is not supported | jenniew | jenniew |  | ut |
| 64 | 3166 | test_consistency_SparseCSR failures | Fix sparse operation for XPU - implement proper Triton kernel for XPU. Update sparse CSR/BSR kernel to support XPU device properly. | test_consistency_SparseCSR failures | yucai-intel | yucai-intel |  | ut |
| | | **Subtotal: 6 issues** | | | | | |

### <span id='feature-not-supported'>Feature Not Supported</span> (2 issues)

| # | ID | Title | Action Reason | Summary | Assignee | Owner Transfer | PR Status | Test Module |
|--:|----|-------|---------------|---------|----------|----------------|-----------|-------------|
| 65 | 2376 | [Bug Skip]: NotImplementedError: "logaddexp_xpu" not implemented for 'Complex' | Fix dtype precision issue - adjust numerical tolerance or use higher precision computations. Implement dtype-specific kernel or configure proper precision settings for XPU. | [Bug Skip]: NotImplementedError: "logaddexp_xpu" not implemented for 'Complex' | CuiYifeng | CuiYifeng |  | ut |
| 66 | 3142 | [upstream_ut] RuntimeError: The sycl_ext_oneapi_work_group_scratch_memory feature is not yet available for use with SYCL Graph extension. | Fix memory management on XPU: Check memory allocation/deallocation in the operation's XPU kernel implementation. | [upstream_ut] RuntimeError: The sycl_ext_oneapi_work_group_scratch_memory feature is not yet available for use with SYCL Graph extension. | LuFinch | LuFinch |  | ut |
| | | **Subtotal: 2 issues** | | | | | |

### <span id='pt2e'>PT2E</span> (2 issues)

| # | ID | Title | Action Reason | Summary | Assignee | Owner Transfer | PR Status | Test Module |
|--:|----|-------|---------------|---------|----------|----------------|-----------|-------------|
| 67 | 2715 | [upstream_ut] torch._dynamo.exc.Unsupported: Attempted to inline function marked as skipped | Fix Inductor compilation issue - implement proper XPU lowering or decomposition. Add XPU-specific inductor lowering or fix fallback/decomposition conflict. | [upstream_ut] torch._dynamo.exc.Unsupported: Attempted to inline function marked as skipped | CuiYifeng | CuiYifeng |  | ut |
| 68 | 3006 | AssertionError: '.to(tl.float16)' unexpectedly found in '# AOT ID | Investigate test failure - analyze traceback and fix the root cause. Run test with detailed logging to identify the specific failure point. | AssertionError: '.to(tl.float16)' unexpectedly found in '# AOT ID | CuiYifeng | CuiYifeng |  | e2e |
| | | **Subtotal: 2 issues** | | | | | |

### <span id='torchao'>TorchAO</span> (2 issues)

| # | ID | Title | Action Reason | Summary | Assignee | Owner Transfer | PR Status | Test Module |
|--:|----|-------|---------------|---------|----------|----------------|-----------|-------------|
| 69 | 2532 | Title: [upstream_ut] AssertionError: wrong number of dimensions2 for op: torch.ops.aten._convert_weight_to_int4pack.defa | Fix OneDNN/TorchAO compatibility - update kernel or use compatible version. Update OneDNN backend or adjust quantization configuration for XPU. | Title: [upstream_ut] AssertionError: wrong number of dimensions2 for op: torch.ops.aten._convert_weight_to_int4pack.defa | yucai-intel | yucai-intel |  | ut |
| 70 | 2578 | [TorchAO][UT] test/quantization/test_quant_api.py AssertionError: SQNR -2.90625 is too low | Fix OneDNN/TorchAO compatibility - update kernel or use compatible version. Update OneDNN backend or adjust quantization configuration for XPU. | [TorchAO][UT] test/quantization/test_quant_api.py AssertionError: SQNR -2.90625 is too low | Stonepia | Stonepia |  | build |
| | | **Subtotal: 2 issues** | | | | | |

### <span id='torch-operations'>Torch Operations</span> (1 issues)

| # | ID | Title | Action Reason | Summary | Assignee | Owner Transfer | PR Status | Test Module |
|--:|----|-------|---------------|---------|----------|----------------|-----------|-------------|
| 71 | 3131 | [upstream_ut] NotImplementedError: The operator 'aten::_scaled_dot_product_efficient_attention_backward' is not c | Implement attention operation for XPU backend or enable FlashAttentionForwardXPU. Register aten._efficient_attention_forward or flash_attention_forward for XPU. | [upstream_ut] NotImplementedError: The operator 'aten::_scaled_dot_product_efficient_attention_backward' is not c | chunhuanMeng | chunhuanMeng |  | ut |
| | | **Subtotal: 1 issues** | | | | | |

[Back to Index](#toc) |

## <span id='3-other-actions-by-type'>3. Other Actions by Type</span>

**Total: 3 issues** - Actions other than Need Investigation

### <span id='add-to-skiplist'>add to skiplist</span> (3 issues)

| # | ID | Title | Action Reason | Summary | Assignee | Owner Transfer | PR Status | Test Module |
|--:|----|-------|---------------|---------|----------|----------------|-----------|-------------|
| 1 | 2164 | skip test_no_cuda_monkeypatch as it is cuda specific | Issue marked as not_target/wontfix - should be skipped for XPU enablement | skip test_no_cuda_monkeypatch as it is cuda specific | daisyden | daisyden |  | ut |
| 2 | 2309 | unsupported ops with PYTORCH_ENABLE_XPU_FALLBACK unset | Issue marked as not_target/wontfix - should be skipped for XPU enablement | unsupported ops with PYTORCH_ENABLE_XPU_FALLBACK unset | daisyden | daisyden |  | ut |
| 3 | 3127 | [upstream_ut] AssertionError: AssertionError not raised | Issue marked as not_target/wontfix - should be skipped for XPU enablement | [upstream_ut] AssertionError: AssertionError not raised | daisyden | daisyden |  | ut |
| | | **Subtotal: 3 issues** | | | | | |

[Back to Index](#toc) |

## <span id='4-last-week-issues'>4. Last Week Issues</span>

**Total: 0 issues** - Issues created in the last 7 days

No issues created in the last 7 days.

[Back to Index](#toc) |

## <span id='5-stale-issues'>5. Stale Issues</span>

**Total: 48 issues** - Issues not updated in 2+ weeks

| # | ID | Title | Action Reason | Summary | Category | Updated Time | Days Since Update |
|--:|----|-------|---------------|---------|----------|---------------|-------------------|
| 1 | 2015 | inf is returned by nn.TransformerEncoderLayer | Fix dtype precision issue - adjust numerical tolerance or use higher precision computations. Implement dtype-specific kernel or configure proper precision settings for XPU. | inf is returned by nn.TransformerEncoderLayer | Others | 2026-03-02T07:16:07Z | 44 |
| 2 | 2164 | skip test_no_cuda_monkeypatch as it is cuda specific | Issue marked as not_target/wontfix - should be skipped for XPU enablement | skip test_no_cuda_monkeypatch as it is cuda specific | Torch Runtime | 2026-03-25T07:57:55Z | 21 |
| 3 | 2214 | test/test_sparse.py::TestSparseAnyXPU::test_gradcheck_mm expected error message not match | Implement XPU-specific kernel or backend dispatch for the affected operation. Add proper device type check and XPU kernel implementation in native_functions.yaml. | test/test_sparse.py::TestSparseAnyXPU::test_gradcheck_mm expected error message not match | Sparse | 2026-03-19T03:53:55Z | 27 |
| 4 | 2229 | test/test_sparse_csr.py::TestSparseCompressedCPU::test_invalid_input meet message not match | Fix error: /__w/torch-xpu-ops/torch-xpu-ops/pytorch/third_party/torch-xpu-ops/test/xpu/test_sparse_csr_xpu.py:1... Investigate root cause and implement proper fix for XPU backend. | test/test_sparse_csr.py::TestSparseCompressedCPU::test_invalid_input meet message not match | Sparse | 2026-03-12T21:59:46Z | 34 |
| 5 | 2244 | test/test_sparse_csr.py::TestSparseCSRXPU::test_block_addmm meet RuntimeError: empty_sparse_compressed expected sparse compressed (non-block) tensor layout but got SparseBsr | Fix error: Tensor-likes are not close!... Investigate root cause and implement proper fix for XPU backend. | test/test_sparse_csr.py::TestSparseCSRXPU::test_block_addmm meet RuntimeError: empty_sparse_compressed expected sparse compressed (non-block) tensor l | Sparse | 2026-03-19T01:48:59Z | 27 |
| 6 | 2245 | oneDNN matmul received incorrect shape in test/test_sparse_csr.py::TestSparseCSRXPU::test_addmm_errors_xpu_float32 | Fix OneDNN/TorchAO compatibility - update kernel or use compatible version. Update OneDNN backend or adjust quantization configuration for XPU. | oneDNN matmul received incorrect shape in test/test_sparse_csr.py::TestSparseCSRXPU::test_addmm_errors_xpu_float32 | Sparse | 2026-03-24T13:32:56Z | 22 |
| 7 | 2253 | the supported dtypes are not align with cuda | Fix dtype precision issue - adjust numerical tolerance or use higher precision computations. Implement dtype-specific kernel or configure proper precision settings for XPU. | the supported dtypes are not align with cuda | Torch Runtime | 2026-03-02T07:04:32Z | 44 |
| 8 | 2255 | [upstream_ut] RuntimeError: Long is not supported in oneDNN | PR closed but no failed tests - verify if issue still reproduces | [upstream_ut] RuntimeError: Long is not supported in oneDNN | Torch Runtime | 2026-03-02T07:12:52Z | 44 |
| 9 | 2270 | Backend Compatibility Error in test/xpu/test_decomp.py | Implement attention operation for XPU backend or enable FlashAttentionForwardXPU. Register aten._efficient_attention_forward or flash_attention_forward for XPU. | Backend Compatibility Error in test/xpu/test_decomp.py | Others | 2026-01-28T08:27:17Z | 77 |
| 10 | 2285 | Support efficient attention | Fix flash attention operation for XPU - handle head dimension and dropout constraints. Update FlashAttentionForwardXPU kernel to support required configurations. | Support efficient attention | Others | 2026-03-02T06:52:42Z | 44 |
| 11 | 2287 | [upstream_ut] test_python_ref issues | Investigate the issue with detailed traceback analysis. Run test case to reproduce and identify root cause for XPU fix. | [upstream_ut] test_python_ref issues | Torch Runtime | 2026-03-30T06:04:57Z | 16 |
| 12 | 2295 | [upstream_ut][xpu][test]nn/test_embedding.py::TestEmbeddingNNDeviceTypeXPU::test_embedding_bag_device_xpu_int32_int32_float64 meet AssertionError: Tensor-likes are not close! | Investigate test failure - analyze traceback and fix the root cause. Run test with detailed logging to identify the specific failure point. | [upstream_ut][xpu][test]nn/test_embedding.py::TestEmbeddingNNDeviceTypeXPU::test_embedding_bag_device_xpu_int32_int32_float64 meet AssertionError: Ten | Torch Runtime | 2026-03-30T02:00:03Z | 16 |
| 13 | 2301 | [upstream_ut] dtypes not align with OpInfo | Fix dtype precision issue - adjust numerical tolerance or use higher precision computations. Implement dtype-specific kernel or configure proper precision settings for XPU. | [upstream_ut] dtypes not align with OpInfo | Torch Runtime | 2026-03-02T07:18:58Z | 44 |
| 14 | 2309 | unsupported ops with PYTORCH_ENABLE_XPU_FALLBACK unset | Issue marked as not_target/wontfix - should be skipped for XPU enablement | unsupported ops with PYTORCH_ENABLE_XPU_FALLBACK unset | Others | 2026-03-19T05:31:22Z | 27 |
| 15 | 2329 | [upstream_ut] feature missing: get_device_tflops and get_drams_gbps | PR closed but no failed tests - verify if issue still reproduces | [upstream_ut] feature missing: get_device_tflops and get_drams_gbps | Inductor/Compilation | 2026-03-04T07:37:55Z | 42 |
| 16 | 2376 | [Bug Skip]: NotImplementedError: "logaddexp_xpu" not implemented for 'Complex' | Fix dtype precision issue - adjust numerical tolerance or use higher precision computations. Implement dtype-specific kernel or configure proper precision settings for XPU. | [Bug Skip]: NotImplementedError: "logaddexp_xpu" not implemented for 'Complex' | Feature Not Supported | 2026-03-19T01:26:41Z | 27 |
| 17 | 2482 | test_dtypes issue introduced by pytorch test sample input updates | Investigate AMP inference accuracy - check gradient scaling and mixed precision implementation. Add AMP-specific tolerance adjustment for XPU or verify cuDNN/MKLDNN backend configuration. | test_dtypes issue introduced by pytorch test sample input updates | Dtype/Precision | 2026-03-02T06:58:45Z | 44 |
| 18 | 2512 | [upstream_ut] RuntimeError: _histc_xpu does not have a deterministic implementation, but you set 'torch.use_deter | Implement XPU-specific kernel or backend dispatch for the affected operation. Add proper device type check and XPU kernel implementation in native_functions.yaml. | [upstream_ut] RuntimeError: _histc_xpu does not have a deterministic implementation, but you set 'torch.use_deter | Torch Runtime | 2026-03-19T07:34:19Z | 27 |
| 19 | 2531 | [upstream_ut] AssertionError: Torch not compiled with CUDA enabled | Implement XPU-specific kernel or backend dispatch for the affected operation. Add proper device type check and XPU kernel implementation in native_functions.yaml. | [upstream_ut] AssertionError: Torch not compiled with CUDA enabled | Torch Runtime | 2026-02-04T20:22:00Z | 70 |
| 20 | 2536 | Title: [upstream_ut] AttributeError: module 'torch._C' has no attribute | All test cases passed on XPU/stock - issue is resolved | Title: [upstream_ut] AttributeError: module 'torch._C' has no attribute | Torch Runtime | 2026-04-01T21:05:46Z | 14 |
| 21 | 2541 | Title: [upstream_ut] RuntimeError: could not construct a memory descriptor using strides | All test cases passed on XPU/stock - issue is resolved | Title: [upstream_ut] RuntimeError: could not construct a memory descriptor using strides | Torch Runtime | 2026-03-09T09:19:33Z | 37 |
| 22 | 2554 | [upstream_ut] AssertionError: AssertionError not raised | Investigate test failure - analyze traceback and fix the root cause. Run test with detailed logging to identify the specific failure point. | [upstream_ut] AssertionError: AssertionError not raised | Inductor/Compilation | 2026-03-24T07:41:02Z | 22 |
| 23 | 2578 | [TorchAO][UT] test/quantization/test_quant_api.py AssertionError: SQNR -2.90625 is too low | Fix OneDNN/TorchAO compatibility - update kernel or use compatible version. Update OneDNN backend or adjust quantization configuration for XPU. | [TorchAO][UT] test/quantization/test_quant_api.py AssertionError: SQNR -2.90625 is too low | TorchAO | 2026-03-25T03:16:16Z | 21 |
| 24 | 2609 | [upstream_ut] torch._inductor.exc.InductorError: CppCompileError: C++ compile error | Fix Inductor compilation issue - implement proper XPU lowering or decomposition. Add XPU-specific inductor lowering or fix fallback/decomposition conflict. | [upstream_ut] torch._inductor.exc.InductorError: CppCompileError: C++ compile error | Inductor/Compilation | 2025-12-29T07:32:22Z | 107 |
| 25 | 2611 | [upstream_ut] AssertionError: Tensor-likes are not equal! in test_compile_subprocess | Fix Inductor compilation issue - implement proper XPU lowering or decomposition. Add XPU-specific inductor lowering or fix fallback/decomposition conflict. | [upstream_ut] AssertionError: Tensor-likes are not equal! in test_compile_subprocess | Inductor/Compilation | 2025-12-29T07:33:29Z | 107 |
| 26 | 2613 | [upstream_ut] AssertionError: Tensor-likes are not equal! in test_compile_subprocess.py | Investigate test failure - analyze traceback and fix the root cause. Run test with detailed logging to identify the specific failure point. | [upstream_ut] AssertionError: Tensor-likes are not equal! in test_compile_subprocess.py | Inductor/Compilation | 2025-12-29T08:50:16Z | 107 |
| 27 | 2620 | [upstream_ut] AssertionError: dtype is needed to compute eps1 when eps1 is unset | Investigate test failure - analyze traceback and fix the root cause. Run test with detailed logging to identify the specific failure point. | [upstream_ut] AssertionError: dtype is needed to compute eps1 when eps1 is unset | Inductor/Compilation | 2025-12-29T07:32:53Z | 107 |
| 28 | 2694 | Title: [upstream_ut] AssertionError: Tensor-likes are not equal! with test_randint tests | PR closed but no failed tests - verify if issue still reproduces | Title: [upstream_ut] AssertionError: Tensor-likes are not equal! with test_randint tests | Inductor/Compilation | 2026-01-29T02:25:00Z | 76 |
| 29 | 2697 | Title: [upstream_ut] RuntimeError: Expected to find ", 0, " but did not find it | Investigate test failure - analyze traceback and fix the root cause. Run test with detailed logging to identify the specific failure point. | Title: [upstream_ut] RuntimeError: Expected to find ", 0, " but did not find it | Torch Runtime | 2026-01-07T08:10:59Z | 98 |
| 30 | 2715 | [upstream_ut] torch._dynamo.exc.Unsupported: Attempted to inline function marked as skipped | Fix Inductor compilation issue - implement proper XPU lowering or decomposition. Add XPU-specific inductor lowering or fix fallback/decomposition conflict. | [upstream_ut] torch._dynamo.exc.Unsupported: Attempted to inline function marked as skipped | PT2E | 2026-03-26T03:08:50Z | 20 |
| 31 | 2783 | [Bug Skip]: Key "xpu" is missing from dict "driver" in test_svd | Add test case to skiplist or mark as expected failure (xfail) for XPU. Verify test exists in torch-xpu-ops and add to skip list if not applicable to XPU. | [Bug Skip]: Key "xpu" is missing from dict "driver" in test_svd | Others | 2026-01-28T08:31:27Z | 77 |
| 32 | 2800 | AttributeError: 'torch._C._XpuDeviceProperties' object has no attribute 'major' | Implement XPU-specific kernel or backend dispatch for the affected operation. Add proper device type check and XPU kernel implementation in native_functions.yaml. | AttributeError: 'torch._C._XpuDeviceProperties' object has no attribute 'major' | Torch Runtime | 2026-04-01T08:02:28Z | 14 |
| 33 | 2806 | CompiledAOTI need XPU support | PR closed but no failed tests - verify if issue still reproduces | CompiledAOTI need XPU support | Inductor/Compilation | 2026-03-25T14:07:39Z | 21 |
| 34 | 2888 | torch._inductor.exc.InductorError: AssertionError: Conversions between float8_e5m2 and float8_e4m3fn is not supported! | Fix dtype precision issue - adjust numerical tolerance or use higher precision computations. Implement dtype-specific kernel or configure proper precision settings for XPU. | torch._inductor.exc.InductorError: AssertionError: Conversions between float8_e5m2 and float8_e4m3fn is not supported! | Inductor/Compilation | 2026-03-25T03:28:41Z | 21 |
| 35 | 2891 | RuntimeError: Expected to find "(262144, 0, 512, 1" but did not find it | Implement XPU-specific kernel or backend dispatch for the affected operation. Add proper device type check and XPU kernel implementation in native_functions.yaml. | RuntimeError: Expected to find "(262144, 0, 512, 1" but did not find it | Torch Runtime | 2026-03-25T09:14:22Z | 21 |
| 36 | 2958 | AssertionError of test_dtensor_basic_compile | Investigate test failure - analyze traceback and fix the root cause. Run test with detailed logging to identify the specific failure point. | AssertionError of test_dtensor_basic_compile | Inductor/Compilation | 2026-03-25T13:33:16Z | 21 |
| 37 | 2997 | AssertionError of test_linear_and_cel_max_autotune | Investigate test failure - analyze traceback and fix the root cause. Run test with detailed logging to identify the specific failure point. | AssertionError of test_linear_and_cel_max_autotune | Inductor/Compilation | 2026-03-06T14:20:04Z | 40 |
| 38 | 2999 | KeyError: 'eager_numerics.use_pytorch_libdevice' | Implement XPU-specific kernel or backend dispatch for the affected operation. Add proper device type check and XPU kernel implementation in native_functions.yaml. | KeyError: 'eager_numerics.use_pytorch_libdevice' | Torch Runtime | 2026-03-25T12:51:44Z | 21 |
| 39 | 3004 | TypeError: _xpu_recordMemoryHistory(): incompatible function arguments | Investigate test failure - analyze traceback and fix the root cause. Run test with detailed logging to identify the specific failure point. | TypeError: _xpu_recordMemoryHistory(): incompatible function arguments | Torch Runtime | 2026-03-09T05:20:13Z | 37 |
| 40 | 3007 | AssertionError: Scalars are not equal! with test_flash_attention_dynamic | PR closed but no failed tests - verify if issue still reproduces | AssertionError: Scalars are not equal! with test_flash_attention_dynamic | Flash Attention/Transformer | 2026-04-01T08:09:56Z | 14 |
| 41 | 3033 | [Bug Skip]: Softmax tolerance | Investigate the issue with detailed traceback analysis. Run test case to reproduce and identify root cause for XPU fix. | [Bug Skip]: Softmax tolerance | Others | 2026-03-17T07:27:00Z | 29 |
| 42 | 3131 | [upstream_ut] NotImplementedError: The operator 'aten::_scaled_dot_product_efficient_attention_backward' is not c | Implement attention operation for XPU backend or enable FlashAttentionForwardXPU. Register aten._efficient_attention_forward or flash_attention_forward for XPU. | [upstream_ut] NotImplementedError: The operator 'aten::_scaled_dot_product_efficient_attention_backward' is not c | Torch Operations | 2026-03-24T06:27:04Z | 22 |
| 43 | 3140 | [upstream_ut] RuntimeError: FlashAttentionForwardXPU does not only support dropout > 0.0 yet | Fix flash attention operation for XPU - handle head dimension and dropout constraints. Update FlashAttentionForwardXPU kernel to support required configurations. | [upstream_ut] RuntimeError: FlashAttentionForwardXPU does not only support dropout > 0.0 yet | Others | 2026-03-24T06:33:36Z | 22 |
| 44 | 3141 | [upstream_ut] RuntimeError: FlashAttentionForwardXPU only support headdim 64,96,128,192 | Fix flash attention operation for XPU - handle head dimension and dropout constraints. Update FlashAttentionForwardXPU kernel to support required configurations. | [upstream_ut] RuntimeError: FlashAttentionForwardXPU only support headdim 64,96,128,192 | Flash Attention/Transformer | 2026-03-24T06:32:57Z | 22 |
| 45 | 3163 | [Bug Skip]: Object comparison failed: torch.int64 != torch.int32 in test_sparse_add | Fix sparse index dtype: In torch/sparse/__init__.py, fix crow_indices dtype conversion - ensure int64 to int32 alignment for XPU. | [Bug Skip]: Object comparison failed: torch.int64 != torch.int32 in test_sparse_add | Dtype/Precision | 2026-03-25T02:19:59Z | 21 |
| 46 | 3166 | test_consistency_SparseCSR failures | Fix sparse operation for XPU - implement proper Triton kernel for XPU. Update sparse CSR/BSR kernel to support XPU device properly. | test_consistency_SparseCSR failures | Sparse | 2026-03-24T15:31:42Z | 22 |
| 47 | 3170 | Unskip test_bmm_windows_error_xpu_float64 | Fix error: /__w/torch-xpu-ops/torch-xpu-ops/pytorch/third_party/torch-xpu-ops/test/xpu/test_sparse_xpu.py:1965:... Investigate root cause and implement proper fix for XPU backend. | Unskip test_bmm_windows_error_xpu_float64 | Others | 2026-03-25T08:48:38Z | 21 |
| 48 | 3187 | PyTorch XPU gpu_cpp_wrapper fails with InductorError NotImplementedError | Fix Inductor XPU wrapper: Implement gpu_cpp_wrapper support for XPU in torch/_inductor/codegen/wrapper.py - add XPU-specific code generation. | PyTorch XPU gpu_cpp_wrapper fails with InductorError NotImplementedError | Inductor/Compilation | 2026-03-30T06:57:11Z | 16 |
| | | **Subtotal: 48 issues** | | | | | |

[Back to Index](#toc) |

## <span id='6-duplicated-issues'>6. Duplicated Issues</span>

**Total: 8 issues** - Issues sharing test cases with other issues

| # | ID | Title | Summary | Assignee | Priority | Root Cause | Dependency | Duplicated With | Test Module |
|--:|----|-------|---------|----------|---------|-----------|-----------|----------------|-------------|
| 1 | 2015 | inf is returned by nn.TransformerEncoderLayer | inf is returned by nn.TransformerEncoderLayer | yucai-intel | P2 |  | None | 2006 | ut |
| 2 | 2024 | AssertionError: Torch not compiled with CUDA enabled | AssertionError: Torch not compiled with CUDA enabled | daisyden | P2 |  | None | 2444 | ut |
| 3 | 2255 | [upstream_ut] RuntimeError: Long is not supported in oneDNN | [upstream_ut] RuntimeError: Long is not supported in oneDNN | daisyden | P2 |  | None | 2301 | ut |
| 4 | 2285 | Support efficient attention | Support efficient attention | chunhuanMeng | P2 |  | None | 2853 | ut |
| 5 | 2301 | [upstream_ut] dtypes not align with OpInfo | [upstream_ut] dtypes not align with OpInfo | daisyden | P2 |  | None | 2255 | ut |
| 6 | 2536 | Title: [upstream_ut] AttributeError: module 'torch._C' has no attribute | Title: [upstream_ut] AttributeError: module 'torch._C' has no attribute | daisyden | P2 |  | None | 2508 | ut |
| 7 | 2715 | [upstream_ut] torch._dynamo.exc.Unsupported: Attempted to inline function marked as skipped | [upstream_ut] torch._dynamo.exc.Unsupported: Attempted to inline function marked as skipped | CuiYifeng | P2 |  | None | 3286 | ut |
| 8 | 2853 | [upstream_ut] torch.ops.aten._flash_attention_forward lack of support for XPU. | [upstream_ut] torch.ops.aten._flash_attention_forward lack of support for XPU. | LuFinch | P2 |  | None | 2285 | ut |
| | | **Subtotal: 8 issues** | | | | | | |

[Back to Index](#toc) |

## <span id='7-issues-with-dependency'>7. Issues with Dependency</span>

**Total: 4 issues** - Issues with external dependencies

| # | ID | Title | Summary | Assignee | Priority | Root Cause | Category | Dependency | PR Status | Test Module |
|--:|----|-------|---------|----------|---------|-----------|----------|------------|-----------|-------------|
| 1 | 2329 | [upstream_ut] feature missing: get_device_tflops and get_drams_gbps | [upstream_ut] feature missing: get_device_tflops and get_drams_gbps | etaf | P2 |  | Inductor/Compilation | Triton |  | ut |
| 2 | 2611 | [upstream_ut] AssertionError: Tensor-likes are not equal! in test_compile_subprocess | [upstream_ut] AssertionError: Tensor-likes are not equal! in test_compile_subprocess | daisyden | P2 |  | Inductor/Compilation | driver |  | ut |
| 3 | 2613 | [upstream_ut] AssertionError: Tensor-likes are not equal! in test_compile_subprocess.py | [upstream_ut] AssertionError: Tensor-likes are not equal! in test_compile_subprocess.py | daisyden | P2 |  | Inductor/Compilation | driver |  | ut |
| 4 | 2800 | AttributeError: 'torch._C._XpuDeviceProperties' object has no attribute 'major' | AttributeError: 'torch._C._XpuDeviceProperties' object has no attribute 'major' | guangyey | P2 |  | Torch Runtime | oneAPI |  | ut |
| | | **Subtotal: 4 issues** | | | | | | | |

[Back to Index](#toc) |

## <span id='8-statistics'>8. Statistics</span>

### <span id='stats-dependency'>By Dependency</span>

| Dependency | Count |
|------------|-------|
| driver | 2 |
| oneAPI | 1 |
| Triton | 1 |

[Back to Index](#toc) |

---
*Report generated with 74 issues*
