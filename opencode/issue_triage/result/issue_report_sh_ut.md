# Torch XPU Ops Issue Report (upstream_ut only)

**Repository:** [intel/torch-xpu-ops](https://github.com/intel/torch-xpu-ops)

**CI Data Sources:**
- Torch-XPU-OPS Nightly CI: XML files from `Inductor-XPU-UT-Data-*` + `Inductor-wheel-nightly-LTS2-XPU-E2E-Data-*`
- Stock PyTorch XPU CI: XML files from `test-default-*-linux.idc.xpu_*.zip`

**Generated:** 2026-04-15 20:56:57
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

| # | ID | Title | Priority | Priority Reason | Action Reason | Summary | Assignee | Test Module | Related PR |
|--:|----|-------|----------|---------------|---------------|---------|----------|-------------|-------------|
| 1 | 3132 | [upstream_ut] transfomers test reports RuntimeError: No available kernel. Aborting execution. | P2 | UT issue with few failures | Fix flash attention operation for XPU - handle head dimension and dropout constraints. Update FlashAttentionForwardXPU kernel to support required configurations. | [upstream_ut] transfomers test reports RuntimeError: No available kernel. Aborting execution. | LuFinch | ut |  |
| 2 | 3129 | [upstream_ut] AssertionError: UserWarning not triggered | P2 | UT issue with few failures | Investigate test failure - analyze traceback and fix the root cause. Run test with detailed logging to identify the specific failure point. | [upstream_ut] AssertionError: UserWarning not triggered | daisyden | ut |  |
| 3 | 3128 | [upstream_ut] AssertionError: RuntimeError not raised by <lambda> | P2 | UT issue with few failures | Fix error: RuntimeError not raised by <lambda>... Investigate root cause and implement proper fix for XPU backend. | [upstream_ut] AssertionError: RuntimeError not raised by <lambda> | daisyden | ut |  |
| 4 | 3004 | TypeError: _xpu_recordMemoryHistory(): incompatible function arguments | P2 | UT issue with few failures | Investigate test failure - analyze traceback and fix the root cause. Run test with detailed logging to identify the specific failure point. | TypeError: _xpu_recordMemoryHistory(): incompatible function arguments | guangyey | ut |  |
| 5 | 2999 | KeyError: 'eager_numerics.use_pytorch_libdevice' | P2 | UT issue with few failures | Implement XPU-specific kernel or backend dispatch for the affected operation. Add proper device type check and XPU kernel implementation in native_functions.yaml. | KeyError: 'eager_numerics.use_pytorch_libdevice' | daisyden | ut |  |
| 6 | 2891 | RuntimeError: Expected to find "(262144, 0, 512, 1" but did not find it | P2 | UT issue with few failures | Implement XPU-specific kernel or backend dispatch for the affected operation. Add proper device type check and XPU kernel implementation in native_functions.yaml. | RuntimeError: Expected to find "(262144, 0, 512, 1" but did not find it | chunhuanMeng | e2e |  |
| 7 | 2800 | AttributeError: 'torch._C._XpuDeviceProperties' object has no attribute 'major' | P2 | UT issue with few failures | Implement XPU-specific kernel or backend dispatch for the affected operation. Add proper device type check and XPU kernel implementation in native_functions.yaml. | AttributeError: 'torch._C._XpuDeviceProperties' object has no attribute 'major' | guangyey | ut |  |
| 8 | 2720 | [upstream_ut] RuntimeError: false INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/native/DispatchStub.cpp":275 | P2 | UT issue with few failures | Investigate test failure - analyze traceback and fix the root cause. Run test with detailed logging to identify the specific failure point. | [upstream_ut] RuntimeError: false INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/native/DispatchStub.cpp":275 | CuiYifeng | ut |  |
| 9 | 2698 | Title: [upstream_ut] RuntimeError: FlashAttentionForwardXPU only support headdim 64,96,128,192 | P2 | UT issue with few failures | Fix flash attention operation for XPU - handle head dimension and dropout constraints. Update FlashAttentionForwardXPU kernel to support required configurations. | Title: [upstream_ut] RuntimeError: FlashAttentionForwardXPU only support headdim 64,96,128,192 | chunhuanMeng, LuFinch | ut |  |
| 10 | 2697 | Title: [upstream_ut] RuntimeError: Expected to find ", 0, " but did not find it | P2 | UT issue with few failures | Investigate test failure - analyze traceback and fix the root cause. Run test with detailed logging to identify the specific failure point. | Title: [upstream_ut] RuntimeError: Expected to find ", 0, " but did not find it | chunhuanMeng | e2e |  |
| 11 | 2541 | Title: [upstream_ut] RuntimeError: could not construct a memory descriptor using strides | P2 | UT issue with few failures | All test cases passed on XPU/stock - issue is resolved | Title: [upstream_ut] RuntimeError: could not construct a memory descriptor using strides | yucai-intel | ut |  |
| 12 | 2536 | Title: [upstream_ut] AttributeError: module 'torch._C' has no attribute | P2 | UT issue with few failures | All test cases passed on XPU/stock - issue is resolved | Title: [upstream_ut] AttributeError: module 'torch._C' has no attribute | daisyden | ut |  |
| 13 | 2531 | [upstream_ut] AssertionError: Torch not compiled with CUDA enabled | P2 | UT issue with few failures | Implement XPU-specific kernel or backend dispatch for the affected operation. Add proper device type check and XPU kernel implementation in native_functions.yaml. | [upstream_ut] AssertionError: Torch not compiled with CUDA enabled | daisyden | ut |  |
| 14 | 2512 | [upstream_ut] RuntimeError: _histc_xpu does not have a deterministic implementation, but you set 'torch.use_deter | P2 | UT issue with few failures | Implement XPU-specific kernel or backend dispatch for the affected operation. Add proper device type check and XPU kernel implementation in native_functions.yaml. | [upstream_ut] RuntimeError: _histc_xpu does not have a deterministic implementation, but you set 'torch.use_deter | chunhuanMeng | ut |  |
| 15 | 2436 | [upstream_ut] AttributeError: 'NoneType' object has no attribute 'clone' | P2 | UT issue with few failures | Fix error: 'NoneType' object has no attribute 'clone'... Investigate root cause and implement proper fix for XPU backend. | [upstream_ut] AttributeError: 'NoneType' object has no attribute 'clone' | daisyden | ut |  |
| 16 | 2301 | [upstream_ut] dtypes not align with OpInfo | P2 | UT issue with few failures | Fix dtype precision issue - adjust numerical tolerance or use higher precision computations. Implement dtype-specific kernel or configure proper precision settings for XPU. | [upstream_ut] dtypes not align with OpInfo | daisyden | ut |  |
| 17 | 2295 | [upstream_ut][xpu][test]nn/test_embedding.py::TestEmbeddingNNDeviceTypeXPU::test_embedding_bag_device_xpu_int32_int32_float64 meet AssertionError: Tensor-likes are not close! | P2 | UT issue with few failures | Investigate test failure - analyze traceback and fix the root cause. Run test with detailed logging to identify the specific failure point. | [upstream_ut][xpu][test]nn/test_embedding.py::TestEmbeddingNNDeviceTypeXPU::test_embedding_bag_device_xpu_int32_int32_float64 meet AssertionError: Ten | yucai-intel | ut |  |
| 18 | 2287 | [upstream_ut] test_python_ref issues | P2 | UT issue with few failures | Investigate the issue with detailed traceback analysis. Run test case to reproduce and identify root cause for XPU fix. | [upstream_ut] test_python_ref issues | yucai-intel | ut |  |
| 19 | 2255 | [upstream_ut] RuntimeError: Long is not supported in oneDNN | P2 | UT issue with few failures | PR closed but no failed tests - verify if issue still reproduces | [upstream_ut] RuntimeError: Long is not supported in oneDNN | daisyden | ut |  |
| 20 | 2253 | the supported dtypes are not align with cuda | P2 | UT issue with few failures | Fix dtype precision issue - adjust numerical tolerance or use higher precision computations. Implement dtype-specific kernel or configure proper precision settings for XPU. | the supported dtypes are not align with cuda | daisyden | ut |  |
| 21 | 1893 | [upstream_ut] oneDNN accuracy issues in test_ops_xpu.py | P2 | UT issue with few failures | Fix dtype precision issue - adjust numerical tolerance or use higher precision computations. Implement dtype-specific kernel or configure proper precision settings for XPU. | [upstream_ut] oneDNN accuracy issues in test_ops_xpu.py | chunhuanMeng | ut |  |
| | | **Subtotal: 21 issues** | | | | | | | |

### <span id='inductor-compilation'>Inductor/Compilation</span> (15 issues)

| # | ID | Title | Priority | Priority Reason | Action Reason | Summary | Assignee | Test Module | Related PR |
|--:|----|-------|----------|---------------|---------------|---------|----------|-------------|-------------|
| 22 | 3187 | PyTorch XPU gpu_cpp_wrapper fails with InductorError NotImplementedError | P2 | UT issue with few failures | Fix Inductor XPU wrapper: Implement gpu_cpp_wrapper support for XPU in torch/_inductor/codegen/wrapper.py - add XPU-specific code generation. | PyTorch XPU gpu_cpp_wrapper fails with InductorError NotImplementedError | CuiYifeng | ut |  |
| 23 | 2997 | AssertionError of test_linear_and_cel_max_autotune | P2 | UT issue with few failures | Investigate test failure - analyze traceback and fix the root cause. Run test with detailed logging to identify the specific failure point. | AssertionError of test_linear_and_cel_max_autotune | etaf | ut |  |
| 24 | 2958 | AssertionError of test_dtensor_basic_compile | P2 | UT issue with few failures | Investigate test failure - analyze traceback and fix the root cause. Run test with detailed logging to identify the specific failure point. | AssertionError of test_dtensor_basic_compile | daisyden | ut |  |
| 25 | 2888 | torch._inductor.exc.InductorError: AssertionError: Conversions between float8_e5m2 and float8_e4m3fn is not supported! | P2 | UT issue with few failures | Fix dtype precision issue - adjust numerical tolerance or use higher precision computations. Implement dtype-specific kernel or configure proper precision settings for XPU. | torch._inductor.exc.InductorError: AssertionError: Conversions between float8_e5m2 and float8_e4m3fn is not supported! | Stonepia | ut |  |
| 26 | 2810 | AssertionError: Object comparison failed: Decimal('2.938735877055718769921841343055614194546[51 chars]-39') != Decimal('0') | P2 | UT issue with few failures | Investigate test failure - analyze traceback and fix the root cause. Run test with detailed logging to identify the specific failure point. | AssertionError: Object comparison failed: Decimal('2.938735877055718769921841343055614194546[51 chars]-39') != Decimal('0') | daisyden | ut |  |
| 27 | 2806 | CompiledAOTI need XPU support | P2 | UT issue with few failures | PR closed but no failed tests - verify if issue still reproduces | CompiledAOTI need XPU support | daisyden | ut |  |
| 28 | 2694 | Title: [upstream_ut] AssertionError: Tensor-likes are not equal! with test_randint tests | P2 | UT issue with few failures | PR closed but no failed tests - verify if issue still reproduces | Title: [upstream_ut] AssertionError: Tensor-likes are not equal! with test_randint tests | daisyden | ut |  |
| 29 | 2620 | [upstream_ut] AssertionError: dtype is needed to compute eps1 when eps1 is unset | P2 | UT issue with few failures | Investigate test failure - analyze traceback and fix the root cause. Run test with detailed logging to identify the specific failure point. | [upstream_ut] AssertionError: dtype is needed to compute eps1 when eps1 is unset | daisyden | ut |  |
| 30 | 2613 | [upstream_ut] AssertionError: Tensor-likes are not equal! in test_compile_subprocess.py | P2 | UT issue with few failures | Investigate test failure - analyze traceback and fix the root cause. Run test with detailed logging to identify the specific failure point. | [upstream_ut] AssertionError: Tensor-likes are not equal! in test_compile_subprocess.py | daisyden | ut |  |
| 31 | 2611 | [upstream_ut] AssertionError: Tensor-likes are not equal! in test_compile_subprocess | P2 | UT issue with few failures | Fix Inductor compilation issue - implement proper XPU lowering or decomposition. Add XPU-specific inductor lowering or fix fallback/decomposition conflict. | [upstream_ut] AssertionError: Tensor-likes are not equal! in test_compile_subprocess | daisyden | ut |  |
| 32 | 2609 | [upstream_ut] torch._inductor.exc.InductorError: CppCompileError: C++ compile error | P2 | UT issue with few failures | Fix Inductor compilation issue - implement proper XPU lowering or decomposition. Add XPU-specific inductor lowering or fix fallback/decomposition conflict. | [upstream_ut] torch._inductor.exc.InductorError: CppCompileError: C++ compile error | daisyden | ut |  |
| 33 | 2554 | [upstream_ut] AssertionError: AssertionError not raised | P2 | UT issue with few failures | Investigate test failure - analyze traceback and fix the root cause. Run test with detailed logging to identify the specific failure point. | [upstream_ut] AssertionError: AssertionError not raised | daisyden | ut |  |
| 34 | 2329 | [upstream_ut] feature missing: get_device_tflops and get_drams_gbps | P2 | UT issue with few failures | PR closed but no failed tests - verify if issue still reproduces | [upstream_ut] feature missing: get_device_tflops and get_drams_gbps | etaf | ut |  |
| 35 | 2169 | Frame size comparison failed in test_size_comparison_no_recompile | P2 | UT issue with few failures | Add test case to skiplist or mark as expected failure (xfail) for XPU. Verify test exists in torch-xpu-ops and add to skip list if not applicable to XPU. | Frame size comparison failed in test_size_comparison_no_recompile | guangyey | ut |  |
| 36 | 2024 | AssertionError: Torch not compiled with CUDA enabled | P2 | UT issue with few failures | Implement XPU-specific kernel or backend dispatch for the affected operation. Add proper device type check and XPU kernel implementation in native_functions.yaml. | AssertionError: Torch not compiled with CUDA enabled | daisyden | ut |  |
| | | **Subtotal: 15 issues** | | | | | | | |

### <span id='others'>Others</span> (8 issues)

| # | ID | Title | Priority | Priority Reason | Action Reason | Summary | Assignee | Test Module | Related PR |
|--:|----|-------|----------|---------------|---------------|---------|----------|-------------|-------------|
| 37 | 3170 | Unskip test_bmm_windows_error_xpu_float64 | P2 | UT issue with few failures | Fix error: /__w/torch-xpu-ops/torch-xpu-ops/pytorch/third_party/torch-xpu-ops/test/xpu/test_sparse_xpu.py:1965:... Investigate root cause and implement proper fix for XPU backend. | Unskip test_bmm_windows_error_xpu_float64 | jenniew | ut |  |
| 38 | 3143 | NotImplementedError: The operator 'aten::_scaled_dot_product_efficient_attention_backward' is not currently implemented for the XPU device. | P2 | UT issue with few failures | Implement attention operation for XPU backend or enable FlashAttentionForwardXPU. Register aten._efficient_attention_forward or flash_attention_forward for XPU. | NotImplementedError: The operator 'aten::_scaled_dot_product_efficient_attention_backward' is not currently implemented for the XPU device. | LuFinch | ut |  |
| 39 | 3140 | [upstream_ut] RuntimeError: FlashAttentionForwardXPU does not only support dropout > 0.0 yet | P2 | UT issue with few failures | Fix flash attention operation for XPU - handle head dimension and dropout constraints. Update FlashAttentionForwardXPU kernel to support required configurations. | [upstream_ut] RuntimeError: FlashAttentionForwardXPU does not only support dropout > 0.0 yet | LuFinch | ut |  |
| 40 | 3033 | [Bug Skip]: Softmax tolerance | P2 | UT issue with few failures | Investigate the issue with detailed traceback analysis. Run test case to reproduce and identify root cause for XPU fix. | [Bug Skip]: Softmax tolerance | chunhuanMeng | ut |  |
| 41 | 2783 | [Bug Skip]: Key "xpu" is missing from dict "driver" in test_svd | P2 | UT issue with few failures | Add test case to skiplist or mark as expected failure (xfail) for XPU. Verify test exists in torch-xpu-ops and add to skip list if not applicable to XPU. | [Bug Skip]: Key "xpu" is missing from dict "driver" in test_svd | daisyden | ut |  |
| 42 | 2285 | Support efficient attention | P2 | UT issue with few failures | Fix flash attention operation for XPU - handle head dimension and dropout constraints. Update FlashAttentionForwardXPU kernel to support required configurations. | Support efficient attention | chunhuanMeng | ut |  |
| 43 | 2270 | Backend Compatibility Error in test/xpu/test_decomp.py | P2 | UT issue with few failures | Implement attention operation for XPU backend or enable FlashAttentionForwardXPU. Register aten._efficient_attention_forward or flash_attention_forward for XPU. | Backend Compatibility Error in test/xpu/test_decomp.py | LuFinch | ut |  |
| 44 | 2015 | inf is returned by nn.TransformerEncoderLayer | P2 | UT issue with few failures | Fix dtype precision issue - adjust numerical tolerance or use higher precision computations. Implement dtype-specific kernel or configure proper precision settings for XPU. | inf is returned by nn.TransformerEncoderLayer | yucai-intel | ut |  |
| | | **Subtotal: 8 issues** | | | | | | | |

### <span id='flash-attention-transformer'>Flash Attention/Transformer</span> (8 issues)

| # | ID | Title | Priority | Priority Reason | Action Reason | Summary | Assignee | Test Module | Related PR |
|--:|----|-------|----------|---------------|---------------|---------|----------|-------------|-------------|
| 45 | 3141 | [upstream_ut] RuntimeError: FlashAttentionForwardXPU only support headdim 64,96,128,192 | P2 | UT issue with few failures | Fix flash attention operation for XPU - handle head dimension and dropout constraints. Update FlashAttentionForwardXPU kernel to support required configurations. | [upstream_ut] RuntimeError: FlashAttentionForwardXPU only support headdim 64,96,128,192 | LuFinch | ut |  |
| 46 | 3136 | [upstream_ut] AssertionError: False is not true in test_transformers | P2 | UT issue with few failures | Fix flash attention operation for XPU - handle head dimension and dropout constraints. Update FlashAttentionForwardXPU kernel to support required configurations. | [upstream_ut] AssertionError: False is not true in test_transformers | LuFinch | ut |  |
| 47 | 3133 | [upstream_ut] RuntimeError: scaled_dot_product_attention: If inputs are nested tensors they must be contiguous | P2 | UT issue with few failures | Fix flash attention operation for XPU - handle head dimension and dropout constraints. Update FlashAttentionForwardXPU kernel to support required configurations. | [upstream_ut] RuntimeError: scaled_dot_product_attention: If inputs are nested tensors they must be contiguous | daisyden | ut |  |
| 48 | 3126 | [upstream_ut] Two NestedTensor issue with flash attention | P2 | UT issue with few failures | Fix flash attention operation for XPU - handle head dimension and dropout constraints. Update FlashAttentionForwardXPU kernel to support required configurations. | [upstream_ut] Two NestedTensor issue with flash attention | daisyden | ut |  |
| 49 | 3007 | AssertionError: Scalars are not equal! with test_flash_attention_dynamic | P2 | UT issue with few failures | PR closed but no failed tests - verify if issue still reproduces | AssertionError: Scalars are not equal! with test_flash_attention_dynamic | daisyden | e2e |  |
| 50 | 2853 | [upstream_ut] torch.ops.aten._flash_attention_forward lack of support for XPU. | P2 | UT issue with few failures | Fix flash attention operation for XPU - handle head dimension and dropout constraints. Update FlashAttentionForwardXPU kernel to support required configurations. | [upstream_ut] torch.ops.aten._flash_attention_forward lack of support for XPU. | LuFinch | ut |  |
| 51 | 2802 | Three aten._scaled_dot_product_flash_attention issues | P2 | UT issue with few failures | Fix flash attention operation for XPU - handle head dimension and dropout constraints. Update FlashAttentionForwardXPU kernel to support required configurations. | Three aten._scaled_dot_product_flash_attention issues | LuFinch | ut |  |
| 52 | 2442 | [Bug Skip]: NotImplementedError: Could not run 'aten::_flash_attention_forward' with arguments from the 'CPU' backend | P2 | UT issue with few failures | Fix flash attention operation for XPU - handle head dimension and dropout constraints. Update FlashAttentionForwardXPU kernel to support required configurations. | [Bug Skip]: NotImplementedError: Could not run 'aten::_flash_attention_forward' with arguments from the 'CPU' backend | daisyden, LuFinch | ut |  |
| | | **Subtotal: 8 issues** | | | | | | | |

### <span id='dtype-precision'>Dtype/Precision</span> (6 issues)

| # | ID | Title | Priority | Priority Reason | Action Reason | Summary | Assignee | Test Module | Related PR |
|--:|----|-------|----------|---------------|---------------|---------|----------|-------------|-------------|
| 53 | 3238 | The supported dtypes of _refs.stft is not aligned to stft | P2 | UT issue with few failures | Align stft dtypes: In torch/_decomp/decompositions.py and torch/signal/windows.py, align supported dtypes for _refs.stft with stft - add complex32 support. | The supported dtypes of _refs.stft is not aligned to stft | daisyden | ut |  |
| 54 | 3177 | Accuracy gap of BF16/FP16 test_block_addmm | P2 | UT issue with few failures | Fix block_addmm BF16 accuracy: In torch/sparse/_triton/ops.py, adjust precision tolerance or fix BF16 computation in CSR block_addmm on XPU. | Accuracy gap of BF16/FP16 test_block_addmm | jenniew | ut |  |
| 55 | 3163 | [Bug Skip]: Object comparison failed: torch.int64 != torch.int32 in test_sparse_add | P2 | UT issue with few failures | Fix sparse index dtype: In torch/sparse/__init__.py, fix crow_indices dtype conversion - ensure int64 to int32 alignment for XPU. | [Bug Skip]: Object comparison failed: torch.int64 != torch.int32 in test_sparse_add | chunhuanMeng | ut |  |
| 56 | 3137 | [upstream_ut] RuntimeError: expected scalar type Half but found Float | P2 | UT issue with few failures | Fix dtype precision issue - adjust numerical tolerance or use higher precision computations. Implement dtype-specific kernel or configure proper precision settings for XPU. | [upstream_ut] RuntimeError: expected scalar type Half but found Float | LuFinch | ut |  |
| 57 | 2615 | [Bug Skip]: New failures RuntimeError: Unsupported dtype Half / RuntimeError: Unsupported dtype torch.float16 | P2 | UT issue with few failures | Fix precision issue for float16 dtype - adjust numerical tolerance or use higher precision intermediate. Implement fp16-specific kernel with stable computation or add torchao precision tuning. | [Bug Skip]: New failures RuntimeError: Unsupported dtype Half / RuntimeError: Unsupported dtype torch.float16 | CuiYifeng | ut |  |
| 58 | 2482 | test_dtypes issue introduced by pytorch test sample input updates | P2 | UT issue with few failures | Investigate AMP inference accuracy - check gradient scaling and mixed precision implementation. Add AMP-specific tolerance adjustment for XPU or verify cuDNN/MKLDNN backend configuration. | test_dtypes issue introduced by pytorch test sample input updates | daisyden | ut |  |
| | | **Subtotal: 6 issues** | | | | | | | |

### <span id='sparse'>Sparse</span> (6 issues)

| # | ID | Title | Priority | Priority Reason | Action Reason | Summary | Assignee | Test Module | Related PR |
|--:|----|-------|----------|---------------|---------------|---------|----------|-------------|-------------|
| 59 | 3166 | test_consistency_SparseCSR failures | P2 | UT issue with few failures | Fix sparse operation for XPU - implement proper Triton kernel for XPU. Update sparse CSR/BSR kernel to support XPU device properly. | test_consistency_SparseCSR failures | yucai-intel | ut |  |
| 60 | 2283 | [upstream_ut] sparse._sampled_addmm is not supported | P2 | UT issue with few failures | Implement XPU-specific kernel or backend dispatch for the affected operation. Add proper device type check and XPU kernel implementation in native_functions.yaml. | [upstream_ut] sparse._sampled_addmm is not supported | jenniew | ut |  |
| 61 | 2245 | oneDNN matmul received incorrect shape in test/test_sparse_csr.py::TestSparseCSRXPU::test_addmm_errors_xpu_float32 | P2 | UT issue with few failures | Fix OneDNN/TorchAO compatibility - update kernel or use compatible version. Update OneDNN backend or adjust quantization configuration for XPU. | oneDNN matmul received incorrect shape in test/test_sparse_csr.py::TestSparseCSRXPU::test_addmm_errors_xpu_float32 | CuiYifeng | ut |  |
| 62 | 2244 | test/test_sparse_csr.py::TestSparseCSRXPU::test_block_addmm meet RuntimeError: empty_sparse_compressed expected sparse compressed (non-block) tensor layout but got SparseBsr | P2 | UT issue with few failures | Fix error: Tensor-likes are not close!... Investigate root cause and implement proper fix for XPU backend. | test/test_sparse_csr.py::TestSparseCSRXPU::test_block_addmm meet RuntimeError: empty_sparse_compressed expected sparse compressed (non-block) tensor l | jenniew | ut |  |
| 63 | 2229 | test/test_sparse_csr.py::TestSparseCompressedCPU::test_invalid_input meet message not match | P2 | UT issue with few failures | Fix error: /__w/torch-xpu-ops/torch-xpu-ops/pytorch/third_party/torch-xpu-ops/test/xpu/test_sparse_csr_xpu.py:1... Investigate root cause and implement proper fix for XPU backend. | test/test_sparse_csr.py::TestSparseCompressedCPU::test_invalid_input meet message not match | jenniew | ut |  |
| 64 | 2214 | test/test_sparse.py::TestSparseAnyXPU::test_gradcheck_mm expected error message not match | P2 | UT issue with few failures | Implement XPU-specific kernel or backend dispatch for the affected operation. Add proper device type check and XPU kernel implementation in native_functions.yaml. | test/test_sparse.py::TestSparseAnyXPU::test_gradcheck_mm expected error message not match | jenniew | ut |  |
| | | **Subtotal: 6 issues** | | | | | | | |

### <span id='feature-not-supported'>Feature Not Supported</span> (2 issues)

| # | ID | Title | Priority | Priority Reason | Action Reason | Summary | Assignee | Test Module | Related PR |
|--:|----|-------|----------|---------------|---------------|---------|----------|-------------|-------------|
| 65 | 3142 | [upstream_ut] RuntimeError: The sycl_ext_oneapi_work_group_scratch_memory feature is not yet available for use with SYCL Graph extension. | P2 | UT issue with few failures | Fix memory management on XPU: Check memory allocation/deallocation in the operation's XPU kernel implementation. | [upstream_ut] RuntimeError: The sycl_ext_oneapi_work_group_scratch_memory feature is not yet available for use with SYCL Graph extension. | LuFinch | ut |  |
| 66 | 2376 | [Bug Skip]: NotImplementedError: "logaddexp_xpu" not implemented for 'Complex' | P2 | UT issue with few failures | Fix dtype precision issue - adjust numerical tolerance or use higher precision computations. Implement dtype-specific kernel or configure proper precision settings for XPU. | [Bug Skip]: NotImplementedError: "logaddexp_xpu" not implemented for 'Complex' | CuiYifeng | ut |  |
| | | **Subtotal: 2 issues** | | | | | | | |

### <span id='pt2e'>PT2E</span> (2 issues)

| # | ID | Title | Priority | Priority Reason | Action Reason | Summary | Assignee | Test Module | Related PR |
|--:|----|-------|----------|---------------|---------------|---------|----------|-------------|-------------|
| 67 | 3006 | AssertionError: '.to(tl.float16)' unexpectedly found in '# AOT ID | P2 | UT issue with few failures | Investigate test failure - analyze traceback and fix the root cause. Run test with detailed logging to identify the specific failure point. | AssertionError: '.to(tl.float16)' unexpectedly found in '# AOT ID | CuiYifeng | e2e |  |
| 68 | 2715 | [upstream_ut] torch._dynamo.exc.Unsupported: Attempted to inline function marked as skipped | P2 | UT issue with few failures | Fix Inductor compilation issue - implement proper XPU lowering or decomposition. Add XPU-specific inductor lowering or fix fallback/decomposition conflict. | [upstream_ut] torch._dynamo.exc.Unsupported: Attempted to inline function marked as skipped | CuiYifeng | ut |  |
| | | **Subtotal: 2 issues** | | | | | | | |

### <span id='torchao'>TorchAO</span> (2 issues)

| # | ID | Title | Priority | Priority Reason | Action Reason | Summary | Assignee | Test Module | Related PR |
|--:|----|-------|----------|---------------|---------------|---------|----------|-------------|-------------|
| 69 | 2578 | [TorchAO][UT] test/quantization/test_quant_api.py AssertionError: SQNR -2.90625 is too low | P0 | Build crash - critical blocking issue | Fix OneDNN/TorchAO compatibility - update kernel or use compatible version. Update OneDNN backend or adjust quantization configuration for XPU. | [TorchAO][UT] test/quantization/test_quant_api.py AssertionError: SQNR -2.90625 is too low | Stonepia | build |  |
| 70 | 2532 | Title: [upstream_ut] AssertionError: wrong number of dimensions2 for op: torch.ops.aten._convert_weight_to_int4pack.defa | P2 | UT issue with few failures | Fix OneDNN/TorchAO compatibility - update kernel or use compatible version. Update OneDNN backend or adjust quantization configuration for XPU. | Title: [upstream_ut] AssertionError: wrong number of dimensions2 for op: torch.ops.aten._convert_weight_to_int4pack.defa | yucai-intel | ut |  |
| | | **Subtotal: 2 issues** | | | | | | | |

### <span id='torch-operations'>Torch Operations</span> (1 issues)

| # | ID | Title | Priority | Priority Reason | Action Reason | Summary | Assignee | Test Module | Related PR |
|--:|----|-------|----------|---------------|---------------|---------|----------|-------------|-------------|
| 71 | 3131 | [upstream_ut] NotImplementedError: The operator 'aten::_scaled_dot_product_efficient_attention_backward' is not c | P2 | UT issue with few failures | Implement attention operation for XPU backend or enable FlashAttentionForwardXPU. Register aten._efficient_attention_forward or flash_attention_forward for XPU. | [upstream_ut] NotImplementedError: The operator 'aten::_scaled_dot_product_efficient_attention_backward' is not c | chunhuanMeng | ut |  |
| | | **Subtotal: 1 issues** | | | | | | | |

[Back to Index](#toc) |

## <span id='3-other-actions-by-type'>3. Other Actions by Type</span>

**Total: 3 issues** - Actions other than Need Investigation

### <span id='add-to-skiplist'>add to skiplist</span> (3 issues)

| # | ID | Title | Priority | Priority Reason | Action Reason | Summary | Assignee | Test Module | Related PR |
|--:|----|-------|----------|---------------|---------------|---------|----------|-------------|-------------|
| 1 | 3127 | [upstream_ut] AssertionError: AssertionError not raised | P2 | UT issue with few failures | Issue marked as not_target/wontfix - should be skipped for XPU enablement | [upstream_ut] AssertionError: AssertionError not raised | daisyden | ut |  |
| 2 | 2309 | unsupported ops with PYTORCH_ENABLE_XPU_FALLBACK unset | P2 | UT issue with few failures | Issue marked as not_target/wontfix - should be skipped for XPU enablement | unsupported ops with PYTORCH_ENABLE_XPU_FALLBACK unset | daisyden | ut |  |
| 3 | 2164 | skip test_no_cuda_monkeypatch as it is cuda specific | P2 | UT issue with few failures | Issue marked as not_target/wontfix - should be skipped for XPU enablement | skip test_no_cuda_monkeypatch as it is cuda specific | daisyden | ut |  |
| | | **Subtotal: 3 issues** | | | | | | | |

[Back to Index](#toc) |

## <span id='4-last-week-issues'>4. Last Week Issues</span>

**Total: 0 issues** - Issues created in the last 7 days

No issues created in the last 7 days.

[Back to Index](#toc) |

## <span id='5-stale-issues'>5. Stale Issues</span>

**Total: 48 issues** - Issues not updated in 2+ weeks

| # | ID | Title | Priority | Priority Reason | Action Reason | Summary | Category | Updated Time | Days Since Update | Related PR |
|--:|----|-------|----------|---------------|---------------|---------|----------|---------------|-------------------|-------------|
| 4 | 2578 | [TorchAO][UT] test/quantization/test_quant_api.py AssertionError: SQNR -2.90625 is too low | P0 | Build crash - critical blocking issue | Fix OneDNN/TorchAO compatibility - update kernel or use compatible version. Update OneDNN backend or adjust quantization configuration for XPU. | [TorchAO][UT] test/quantization/test_quant_api.py AssertionError: SQNR -2.90625 is too low | TorchAO | 2026-03-25T03:16:16Z | 21 |  |
| 5 | 3187 | PyTorch XPU gpu_cpp_wrapper fails with InductorError NotImplementedError | P2 | UT issue with few failures | Fix Inductor XPU wrapper: Implement gpu_cpp_wrapper support for XPU in torch/_inductor/codegen/wrapper.py - add XPU-specific code generation. | PyTorch XPU gpu_cpp_wrapper fails with InductorError NotImplementedError | Inductor/Compilation | 2026-03-30T06:57:11Z | 16 |  |
| 6 | 3170 | Unskip test_bmm_windows_error_xpu_float64 | P2 | UT issue with few failures | Fix error: /__w/torch-xpu-ops/torch-xpu-ops/pytorch/third_party/torch-xpu-ops/test/xpu/test_sparse_xpu.py:1965:... Investigate root cause and implement proper fix for XPU backend. | Unskip test_bmm_windows_error_xpu_float64 | Others | 2026-03-25T08:48:38Z | 21 |  |
| 7 | 3166 | test_consistency_SparseCSR failures | P2 | UT issue with few failures | Fix sparse operation for XPU - implement proper Triton kernel for XPU. Update sparse CSR/BSR kernel to support XPU device properly. | test_consistency_SparseCSR failures | Sparse | 2026-03-24T15:31:42Z | 22 |  |
| 8 | 3163 | [Bug Skip]: Object comparison failed: torch.int64 != torch.int32 in test_sparse_add | P2 | UT issue with few failures | Fix sparse index dtype: In torch/sparse/__init__.py, fix crow_indices dtype conversion - ensure int64 to int32 alignment for XPU. | [Bug Skip]: Object comparison failed: torch.int64 != torch.int32 in test_sparse_add | Dtype/Precision | 2026-03-25T02:19:59Z | 21 |  |
| 9 | 3141 | [upstream_ut] RuntimeError: FlashAttentionForwardXPU only support headdim 64,96,128,192 | P2 | UT issue with few failures | Fix flash attention operation for XPU - handle head dimension and dropout constraints. Update FlashAttentionForwardXPU kernel to support required configurations. | [upstream_ut] RuntimeError: FlashAttentionForwardXPU only support headdim 64,96,128,192 | Flash Attention/Transformer | 2026-03-24T06:32:57Z | 22 |  |
| 10 | 3140 | [upstream_ut] RuntimeError: FlashAttentionForwardXPU does not only support dropout > 0.0 yet | P2 | UT issue with few failures | Fix flash attention operation for XPU - handle head dimension and dropout constraints. Update FlashAttentionForwardXPU kernel to support required configurations. | [upstream_ut] RuntimeError: FlashAttentionForwardXPU does not only support dropout > 0.0 yet | Others | 2026-03-24T06:33:36Z | 22 |  |
| 11 | 3131 | [upstream_ut] NotImplementedError: The operator 'aten::_scaled_dot_product_efficient_attention_backward' is not c | P2 | UT issue with few failures | Implement attention operation for XPU backend or enable FlashAttentionForwardXPU. Register aten._efficient_attention_forward or flash_attention_forward for XPU. | [upstream_ut] NotImplementedError: The operator 'aten::_scaled_dot_product_efficient_attention_backward' is not c | Torch Operations | 2026-03-24T06:27:04Z | 22 |  |
| 12 | 3033 | [Bug Skip]: Softmax tolerance | P2 | UT issue with few failures | Investigate the issue with detailed traceback analysis. Run test case to reproduce and identify root cause for XPU fix. | [Bug Skip]: Softmax tolerance | Others | 2026-03-17T07:27:00Z | 29 |  |
| 13 | 3007 | AssertionError: Scalars are not equal! with test_flash_attention_dynamic | P2 | UT issue with few failures | PR closed but no failed tests - verify if issue still reproduces | AssertionError: Scalars are not equal! with test_flash_attention_dynamic | Flash Attention/Transformer | 2026-04-01T08:09:56Z | 14 |  |
| 14 | 3004 | TypeError: _xpu_recordMemoryHistory(): incompatible function arguments | P2 | UT issue with few failures | Investigate test failure - analyze traceback and fix the root cause. Run test with detailed logging to identify the specific failure point. | TypeError: _xpu_recordMemoryHistory(): incompatible function arguments | Torch Runtime | 2026-03-09T05:20:13Z | 37 |  |
| 15 | 2999 | KeyError: 'eager_numerics.use_pytorch_libdevice' | P2 | UT issue with few failures | Implement XPU-specific kernel or backend dispatch for the affected operation. Add proper device type check and XPU kernel implementation in native_functions.yaml. | KeyError: 'eager_numerics.use_pytorch_libdevice' | Torch Runtime | 2026-03-25T12:51:44Z | 21 |  |
| 16 | 2997 | AssertionError of test_linear_and_cel_max_autotune | P2 | UT issue with few failures | Investigate test failure - analyze traceback and fix the root cause. Run test with detailed logging to identify the specific failure point. | AssertionError of test_linear_and_cel_max_autotune | Inductor/Compilation | 2026-03-06T14:20:04Z | 40 |  |
| 17 | 2958 | AssertionError of test_dtensor_basic_compile | P2 | UT issue with few failures | Investigate test failure - analyze traceback and fix the root cause. Run test with detailed logging to identify the specific failure point. | AssertionError of test_dtensor_basic_compile | Inductor/Compilation | 2026-03-25T13:33:16Z | 21 |  |
| 18 | 2891 | RuntimeError: Expected to find "(262144, 0, 512, 1" but did not find it | P2 | UT issue with few failures | Implement XPU-specific kernel or backend dispatch for the affected operation. Add proper device type check and XPU kernel implementation in native_functions.yaml. | RuntimeError: Expected to find "(262144, 0, 512, 1" but did not find it | Torch Runtime | 2026-03-25T09:14:22Z | 21 |  |
| 19 | 2888 | torch._inductor.exc.InductorError: AssertionError: Conversions between float8_e5m2 and float8_e4m3fn is not supported! | P2 | UT issue with few failures | Fix dtype precision issue - adjust numerical tolerance or use higher precision computations. Implement dtype-specific kernel or configure proper precision settings for XPU. | torch._inductor.exc.InductorError: AssertionError: Conversions between float8_e5m2 and float8_e4m3fn is not supported! | Inductor/Compilation | 2026-03-25T03:28:41Z | 21 |  |
| 20 | 2806 | CompiledAOTI need XPU support | P2 | UT issue with few failures | PR closed but no failed tests - verify if issue still reproduces | CompiledAOTI need XPU support | Inductor/Compilation | 2026-03-25T14:07:39Z | 21 |  |
| 21 | 2800 | AttributeError: 'torch._C._XpuDeviceProperties' object has no attribute 'major' | P2 | UT issue with few failures | Implement XPU-specific kernel or backend dispatch for the affected operation. Add proper device type check and XPU kernel implementation in native_functions.yaml. | AttributeError: 'torch._C._XpuDeviceProperties' object has no attribute 'major' | Torch Runtime | 2026-04-01T08:02:28Z | 14 |  |
| 22 | 2783 | [Bug Skip]: Key "xpu" is missing from dict "driver" in test_svd | P2 | UT issue with few failures | Add test case to skiplist or mark as expected failure (xfail) for XPU. Verify test exists in torch-xpu-ops and add to skip list if not applicable to XPU. | [Bug Skip]: Key "xpu" is missing from dict "driver" in test_svd | Others | 2026-01-28T08:31:27Z | 77 |  |
| 23 | 2715 | [upstream_ut] torch._dynamo.exc.Unsupported: Attempted to inline function marked as skipped | P2 | UT issue with few failures | Fix Inductor compilation issue - implement proper XPU lowering or decomposition. Add XPU-specific inductor lowering or fix fallback/decomposition conflict. | [upstream_ut] torch._dynamo.exc.Unsupported: Attempted to inline function marked as skipped | PT2E | 2026-03-26T03:08:50Z | 20 |  |
| 24 | 2697 | Title: [upstream_ut] RuntimeError: Expected to find ", 0, " but did not find it | P2 | UT issue with few failures | Investigate test failure - analyze traceback and fix the root cause. Run test with detailed logging to identify the specific failure point. | Title: [upstream_ut] RuntimeError: Expected to find ", 0, " but did not find it | Torch Runtime | 2026-01-07T08:10:59Z | 98 |  |
| 25 | 2694 | Title: [upstream_ut] AssertionError: Tensor-likes are not equal! with test_randint tests | P2 | UT issue with few failures | PR closed but no failed tests - verify if issue still reproduces | Title: [upstream_ut] AssertionError: Tensor-likes are not equal! with test_randint tests | Inductor/Compilation | 2026-01-29T02:25:00Z | 76 |  |
| 26 | 2620 | [upstream_ut] AssertionError: dtype is needed to compute eps1 when eps1 is unset | P2 | UT issue with few failures | Investigate test failure - analyze traceback and fix the root cause. Run test with detailed logging to identify the specific failure point. | [upstream_ut] AssertionError: dtype is needed to compute eps1 when eps1 is unset | Inductor/Compilation | 2025-12-29T07:32:53Z | 107 |  |
| 27 | 2613 | [upstream_ut] AssertionError: Tensor-likes are not equal! in test_compile_subprocess.py | P2 | UT issue with few failures | Investigate test failure - analyze traceback and fix the root cause. Run test with detailed logging to identify the specific failure point. | [upstream_ut] AssertionError: Tensor-likes are not equal! in test_compile_subprocess.py | Inductor/Compilation | 2025-12-29T08:50:16Z | 107 |  |
| 28 | 2611 | [upstream_ut] AssertionError: Tensor-likes are not equal! in test_compile_subprocess | P2 | UT issue with few failures | Fix Inductor compilation issue - implement proper XPU lowering or decomposition. Add XPU-specific inductor lowering or fix fallback/decomposition conflict. | [upstream_ut] AssertionError: Tensor-likes are not equal! in test_compile_subprocess | Inductor/Compilation | 2025-12-29T07:33:29Z | 107 |  |
| 29 | 2609 | [upstream_ut] torch._inductor.exc.InductorError: CppCompileError: C++ compile error | P2 | UT issue with few failures | Fix Inductor compilation issue - implement proper XPU lowering or decomposition. Add XPU-specific inductor lowering or fix fallback/decomposition conflict. | [upstream_ut] torch._inductor.exc.InductorError: CppCompileError: C++ compile error | Inductor/Compilation | 2025-12-29T07:32:22Z | 107 |  |
| 30 | 2554 | [upstream_ut] AssertionError: AssertionError not raised | P2 | UT issue with few failures | Investigate test failure - analyze traceback and fix the root cause. Run test with detailed logging to identify the specific failure point. | [upstream_ut] AssertionError: AssertionError not raised | Inductor/Compilation | 2026-03-24T07:41:02Z | 22 |  |
| 31 | 2541 | Title: [upstream_ut] RuntimeError: could not construct a memory descriptor using strides | P2 | UT issue with few failures | All test cases passed on XPU/stock - issue is resolved | Title: [upstream_ut] RuntimeError: could not construct a memory descriptor using strides | Torch Runtime | 2026-03-09T09:19:33Z | 37 |  |
| 32 | 2536 | Title: [upstream_ut] AttributeError: module 'torch._C' has no attribute | P2 | UT issue with few failures | All test cases passed on XPU/stock - issue is resolved | Title: [upstream_ut] AttributeError: module 'torch._C' has no attribute | Torch Runtime | 2026-04-01T21:05:46Z | 14 |  |
| 33 | 2531 | [upstream_ut] AssertionError: Torch not compiled with CUDA enabled | P2 | UT issue with few failures | Implement XPU-specific kernel or backend dispatch for the affected operation. Add proper device type check and XPU kernel implementation in native_functions.yaml. | [upstream_ut] AssertionError: Torch not compiled with CUDA enabled | Torch Runtime | 2026-02-04T20:22:00Z | 70 |  |
| 34 | 2512 | [upstream_ut] RuntimeError: _histc_xpu does not have a deterministic implementation, but you set 'torch.use_deter | P2 | UT issue with few failures | Implement XPU-specific kernel or backend dispatch for the affected operation. Add proper device type check and XPU kernel implementation in native_functions.yaml. | [upstream_ut] RuntimeError: _histc_xpu does not have a deterministic implementation, but you set 'torch.use_deter | Torch Runtime | 2026-03-19T07:34:19Z | 27 |  |
| 35 | 2482 | test_dtypes issue introduced by pytorch test sample input updates | P2 | UT issue with few failures | Investigate AMP inference accuracy - check gradient scaling and mixed precision implementation. Add AMP-specific tolerance adjustment for XPU or verify cuDNN/MKLDNN backend configuration. | test_dtypes issue introduced by pytorch test sample input updates | Dtype/Precision | 2026-03-02T06:58:45Z | 44 |  |
| 36 | 2376 | [Bug Skip]: NotImplementedError: "logaddexp_xpu" not implemented for 'Complex' | P2 | UT issue with few failures | Fix dtype precision issue - adjust numerical tolerance or use higher precision computations. Implement dtype-specific kernel or configure proper precision settings for XPU. | [Bug Skip]: NotImplementedError: "logaddexp_xpu" not implemented for 'Complex' | Feature Not Supported | 2026-03-19T01:26:41Z | 27 |  |
| 37 | 2329 | [upstream_ut] feature missing: get_device_tflops and get_drams_gbps | P2 | UT issue with few failures | PR closed but no failed tests - verify if issue still reproduces | [upstream_ut] feature missing: get_device_tflops and get_drams_gbps | Inductor/Compilation | 2026-03-04T07:37:55Z | 42 |  |
| 38 | 2309 | unsupported ops with PYTORCH_ENABLE_XPU_FALLBACK unset | P2 | UT issue with few failures | Issue marked as not_target/wontfix - should be skipped for XPU enablement | unsupported ops with PYTORCH_ENABLE_XPU_FALLBACK unset | Others | 2026-03-19T05:31:22Z | 27 |  |
| 39 | 2301 | [upstream_ut] dtypes not align with OpInfo | P2 | UT issue with few failures | Fix dtype precision issue - adjust numerical tolerance or use higher precision computations. Implement dtype-specific kernel or configure proper precision settings for XPU. | [upstream_ut] dtypes not align with OpInfo | Torch Runtime | 2026-03-02T07:18:58Z | 44 |  |
| 40 | 2295 | [upstream_ut][xpu][test]nn/test_embedding.py::TestEmbeddingNNDeviceTypeXPU::test_embedding_bag_device_xpu_int32_int32_float64 meet AssertionError: Tensor-likes are not close! | P2 | UT issue with few failures | Investigate test failure - analyze traceback and fix the root cause. Run test with detailed logging to identify the specific failure point. | [upstream_ut][xpu][test]nn/test_embedding.py::TestEmbeddingNNDeviceTypeXPU::test_embedding_bag_device_xpu_int32_int32_float64 meet AssertionError: Ten | Torch Runtime | 2026-03-30T02:00:03Z | 16 |  |
| 41 | 2287 | [upstream_ut] test_python_ref issues | P2 | UT issue with few failures | Investigate the issue with detailed traceback analysis. Run test case to reproduce and identify root cause for XPU fix. | [upstream_ut] test_python_ref issues | Torch Runtime | 2026-03-30T06:04:57Z | 16 |  |
| 42 | 2285 | Support efficient attention | P2 | UT issue with few failures | Fix flash attention operation for XPU - handle head dimension and dropout constraints. Update FlashAttentionForwardXPU kernel to support required configurations. | Support efficient attention | Others | 2026-03-02T06:52:42Z | 44 |  |
| 43 | 2270 | Backend Compatibility Error in test/xpu/test_decomp.py | P2 | UT issue with few failures | Implement attention operation for XPU backend or enable FlashAttentionForwardXPU. Register aten._efficient_attention_forward or flash_attention_forward for XPU. | Backend Compatibility Error in test/xpu/test_decomp.py | Others | 2026-01-28T08:27:17Z | 77 |  |
| 44 | 2255 | [upstream_ut] RuntimeError: Long is not supported in oneDNN | P2 | UT issue with few failures | PR closed but no failed tests - verify if issue still reproduces | [upstream_ut] RuntimeError: Long is not supported in oneDNN | Torch Runtime | 2026-03-02T07:12:52Z | 44 |  |
| 45 | 2253 | the supported dtypes are not align with cuda | P2 | UT issue with few failures | Fix dtype precision issue - adjust numerical tolerance or use higher precision computations. Implement dtype-specific kernel or configure proper precision settings for XPU. | the supported dtypes are not align with cuda | Torch Runtime | 2026-03-02T07:04:32Z | 44 |  |
| 46 | 2245 | oneDNN matmul received incorrect shape in test/test_sparse_csr.py::TestSparseCSRXPU::test_addmm_errors_xpu_float32 | P2 | UT issue with few failures | Fix OneDNN/TorchAO compatibility - update kernel or use compatible version. Update OneDNN backend or adjust quantization configuration for XPU. | oneDNN matmul received incorrect shape in test/test_sparse_csr.py::TestSparseCSRXPU::test_addmm_errors_xpu_float32 | Sparse | 2026-03-24T13:32:56Z | 22 |  |
| 47 | 2244 | test/test_sparse_csr.py::TestSparseCSRXPU::test_block_addmm meet RuntimeError: empty_sparse_compressed expected sparse compressed (non-block) tensor layout but got SparseBsr | P2 | UT issue with few failures | Fix error: Tensor-likes are not close!... Investigate root cause and implement proper fix for XPU backend. | test/test_sparse_csr.py::TestSparseCSRXPU::test_block_addmm meet RuntimeError: empty_sparse_compressed expected sparse compressed (non-block) tensor l | Sparse | 2026-03-19T01:48:59Z | 27 |  |
| 48 | 2229 | test/test_sparse_csr.py::TestSparseCompressedCPU::test_invalid_input meet message not match | P2 | UT issue with few failures | Fix error: /__w/torch-xpu-ops/torch-xpu-ops/pytorch/third_party/torch-xpu-ops/test/xpu/test_sparse_csr_xpu.py:1... Investigate root cause and implement proper fix for XPU backend. | test/test_sparse_csr.py::TestSparseCompressedCPU::test_invalid_input meet message not match | Sparse | 2026-03-12T21:59:46Z | 34 |  |
| 49 | 2214 | test/test_sparse.py::TestSparseAnyXPU::test_gradcheck_mm expected error message not match | P2 | UT issue with few failures | Implement XPU-specific kernel or backend dispatch for the affected operation. Add proper device type check and XPU kernel implementation in native_functions.yaml. | test/test_sparse.py::TestSparseAnyXPU::test_gradcheck_mm expected error message not match | Sparse | 2026-03-19T03:53:55Z | 27 |  |
| 50 | 2164 | skip test_no_cuda_monkeypatch as it is cuda specific | P2 | UT issue with few failures | Issue marked as not_target/wontfix - should be skipped for XPU enablement | skip test_no_cuda_monkeypatch as it is cuda specific | Torch Runtime | 2026-03-25T07:57:55Z | 21 |  |
| 51 | 2015 | inf is returned by nn.TransformerEncoderLayer | P2 | UT issue with few failures | Fix dtype precision issue - adjust numerical tolerance or use higher precision computations. Implement dtype-specific kernel or configure proper precision settings for XPU. | inf is returned by nn.TransformerEncoderLayer | Others | 2026-03-02T07:16:07Z | 44 |  |
| | | **Subtotal: 51 issues** | | | | | | | | |

[Back to Index](#toc) |

## <span id='6-duplicated-issues'>6. Duplicated Issues</span>

**Total: 8 issues** - Issues sharing test cases with other issues

| # | ID | Title | Priority | Priority Reason | Summary | Assignee | Root Cause | Dependency | Duplicated With | Test Module | Related PR |
|--:|----|-------|----------|---------------|---------|----------|---------|-----------|----------------|-------------|-------------|
| 52 | 2853 | [upstream_ut] torch.ops.aten._flash_attention_forward lack of support for XPU. | P2 | UT issue with few failures | [upstream_ut] torch.ops.aten._flash_attention_forward lack of support for XPU. | LuFinch |  | None | 2285 | ut |  |
| 53 | 2715 | [upstream_ut] torch._dynamo.exc.Unsupported: Attempted to inline function marked as skipped | P2 | UT issue with few failures | [upstream_ut] torch._dynamo.exc.Unsupported: Attempted to inline function marked as skipped | CuiYifeng |  | None | 3286 | ut |  |
| 54 | 2536 | Title: [upstream_ut] AttributeError: module 'torch._C' has no attribute | P2 | UT issue with few failures | Title: [upstream_ut] AttributeError: module 'torch._C' has no attribute | daisyden |  | None | 2508 | ut |  |
| 55 | 2301 | [upstream_ut] dtypes not align with OpInfo | P2 | UT issue with few failures | [upstream_ut] dtypes not align with OpInfo | daisyden |  | None | 2255 | ut |  |
| 56 | 2285 | Support efficient attention | P2 | UT issue with few failures | Support efficient attention | chunhuanMeng |  | None | 2853 | ut |  |
| 57 | 2255 | [upstream_ut] RuntimeError: Long is not supported in oneDNN | P2 | UT issue with few failures | [upstream_ut] RuntimeError: Long is not supported in oneDNN | daisyden |  | None | 2301 | ut |  |
| 58 | 2024 | AssertionError: Torch not compiled with CUDA enabled | P2 | UT issue with few failures | AssertionError: Torch not compiled with CUDA enabled | daisyden |  | None | 2444 | ut |  |
| 59 | 2015 | inf is returned by nn.TransformerEncoderLayer | P2 | UT issue with few failures | inf is returned by nn.TransformerEncoderLayer | yucai-intel |  | None | 2006 | ut |  |
| | | **Subtotal: 59 issues** | | | | | | | | |

[Back to Index](#toc) |

## <span id='7-issues-with-dependency'>7. Issues with Dependency</span>

**Total: 4 issues** - Issues with external dependencies

| # | ID | Title | Priority | Priority Reason | Summary | Assignee | Root Cause | Category | Dependency | Test Module | Related PR |
|--:|----|-------|----------|---------------|---------|----------|---------|----------|------------|-------------|-------------|
| 60 | 2800 | AttributeError: 'torch._C._XpuDeviceProperties' object has no attribute 'major' | P2 | UT issue with few failures | AttributeError: 'torch._C._XpuDeviceProperties' object has no attribute 'major' | guangyey |  | Torch Runtime | oneAPI | ut |  |
| 61 | 2613 | [upstream_ut] AssertionError: Tensor-likes are not equal! in test_compile_subprocess.py | P2 | UT issue with few failures | [upstream_ut] AssertionError: Tensor-likes are not equal! in test_compile_subprocess.py | daisyden |  | Inductor/Compilation | driver | ut |  |
| 62 | 2611 | [upstream_ut] AssertionError: Tensor-likes are not equal! in test_compile_subprocess | P2 | UT issue with few failures | [upstream_ut] AssertionError: Tensor-likes are not equal! in test_compile_subprocess | daisyden |  | Inductor/Compilation | driver | ut |  |
| 63 | 2329 | [upstream_ut] feature missing: get_device_tflops and get_drams_gbps | P2 | UT issue with few failures | [upstream_ut] feature missing: get_device_tflops and get_drams_gbps | etaf |  | Inductor/Compilation | Triton | ut |  |
| | | **Subtotal: 63 issues** | | | | | | | | |

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
