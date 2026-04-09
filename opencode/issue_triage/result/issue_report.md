# Torch XPU Ops Issue Report

**Repository:** [intel/torch-xpu-ops](https://github.com/intel/torch-xpu-ops)

**Issues Extracted:** April 5, 2026 (from GitHub issue data: `torch_xpu_ops_issues.json`)

**CI Data Sources:**
- Torch-XPU-OPS Nightly CI: XML files from `Inductor-XPU-UT-Data-*` + `Inductor-wheel-nightly-LTS2-XPU-E2E-Data-*` (commit: `a2d516a58c64f18b76880f3a77efbc02885d65af`)
- Stock PyTorch XPU CI: XML files from `test-default-*-linux.idc.xpu_*.zip` (run IDs: 69741866812, 69741866834, 69741866862, 69741866911, etc.)

Generated: 2026-04-08 20:55:59

## Summary

| Category | Count |
|----------|-------|
| Action Required | 349 |
| No Assignee | 3 |
| Duplicated Issues | 42 |
| With Dependency | 5 |
| Others | 18 |
| **Total** | 417 |

---

## Statistics

### By Test Module

| Test Module | Count |
|-------------|-------|
| ut | 371 |
| e2e | 39 |
| build | 7 |

### By Module

| Module | Count |
|--------|-------|
| aten_ops | 303 |
| distributed | 39 |
| inductor | 30 |
| AO | 21 |
| unknown | 16 |
| profiling | 5 |
| low_precision | 3 |

### By Dependency

| Dependency | Count |
|------------|-------|
| oneAPI | 13 |
| driver | 10 |
| oneDNN | 8 |
| Triton | 5 |
| oneCCL | 1 |

### By Action TBD

| Action TBD | Count |
|------------|-------|
| Need reproduce steps (Only for bugs or performance issue) | 167 |
| Needs PyTorch Repo Changes (upstream) | 123 |
| Need more information - error logs and reproduction steps | 73 |
| Close fixed issue | 17 |
| Revisit the PR as case failed | 4 |
| add to skiplist | 4 |
| Verify the issue | 3 |

### By Category

| Category | Count |
|----------|-------|
| Others | 115 |
| Distributed | 105 |
| Dtype / Precision Related | 56 |
| Flash Attention / Transformer Related | 50 |
| TorchAO | 38 |
| Inductor / Compilation Related | 26 |
| Sparse Operations Related | 14 |
| PT2E | 13 |

### By Priority

| Priority | Count |
|----------|-------|
| P0 | 50 |
| P1 | 18 |
| P2 | 349 |

### By Root Cause

| Root Cause | Count |
|------------|-------|
| Others - insufficient information to determine root cause | 7 |
| Failure - Tensor-likes are not close! | 4 |
| Skip/No Test Exists - no test or error information provided | 4 |
| Skip/No Test Exists - no test or error details provided | 3 |
| Backend/Device Issue - inputs are not on the same XPU device | 2 |
| Failure - RuntimeError not raised as expected in test | 2 |
| Backend/Device Issue - _share_fd_ is not available on XPU device | 2 |
| Backend/Device Issue - missing kernel for xpu in DispatchStub | 2 |
| Failure - test assertion failed with False is not true | 2 |
| Skip/No Test Exists - no test or error traceback provided | 2 |
| Backend/Device Issue - XPU device initialization or compatibility failure | 1 |
| 5 - Flash Attention/Specific Ops Issue - Error in _scaled_dot_product_fused_atte | 1 |
| Backend/Device Issue - Huggingface test models failed to run on XPU, indicating  | 1 |
| Timeout/Performance Issue - performance tests failed due to regression in execut | 1 |
| Failure - assertion 'False is not true' failed in test | 1 |
| Failure - mismatch in expected IR code for XPU backend operations | 1 |
| Distributed/Gloo Issue - No backend for the parent process group or its backend  | 1 |
| Failure - AssertionError not raised for Placement (Shard(dim=2),) in test | 1 |
| Inductor/Compilation Issue - Dynamo failed to compile FX node with fake tensors  | 1 |
| Timeout/Performance Issue - high latency in XPU event (~0.1ms) indicates a perfo | 1 |
| Backend/Device Issue - SYCL kernel build failure on Windows for XPU | 1 |
| Backend/Device Issue - XPU ops show non-determinism and inconsistency compared t | 1 |
| Backend/Device Issue - Cyclic dependencies during build with BUILD_SEPARATE_OPS= | 1 |
| Backend/Device Issue - Tensors are on different devices (xpu:0 vs cpu) | 1 |
| 10 - vitals feature is not supported, cases should be disabled | 1 |
| Backend/Device Issue - test crashed on XPU backend execution | 1 |
| Backend/Device Issue - Incorrect strides related to XPU device handling | 1 |
| Inductor/Compilation Issue - Assertion failure due to conflicting fallback and d | 1 |
| Failure - test expects a specific condition to be true but it failed during execution. | 1 |
| Backend/Device Issue - eager_two_runs_differ on ARC XPU backend | 1 |
| Backend/Device Issue - XPU-specific test failure in vmap with SDPA operation | 1 |
| Dtype/Precision Issue - Accuracy gap in BF16/FP16 test | 1 |
| Failure - test assertion failed for Conv2d groups output comparison | 1 |
| Failure - Assertion cur_target >= 0 && cur_target < n_classes failed in cross entropy loss test | 1 |
| Backend/Device Issue - XPU specific failure with Timm models in Triton. | 1 |
| device-specific backend discrepancy. | 1 |
| Backend/Device Issue - XPU specific failure with Huggingface model accuracy | 1 |
| Failure - Expects xpu:0 but got xpu:1 | 1 |
| Backend/Device Issue - Requires mslk >= 1.0.0 for XPU support with int4 quantiza | 1 |
| Skip/No Test Exists - test is skipped or not applicable for XPU backend | 1 |
| Skip/No Test Exists - test was skipped on 2026-3-21 | 1 |
| DNNL/OneDNN Issue - wrong results from oneDNN conv2d kernels due to data pointer | 1 |
| Backend/Device Issue - XPU tensor-like comparison failure in test | 1 |
| Backend/Device Issue - invalid device string 'xpu:foo' indicates a problem with  | 1 |
| Distributed/Gloo Issue - ProcessGroupXCCL object lacks '_set_default_timeout' at | 1 |
| Distributed/Gloo Issue - related to NCCL logging and trace handling in distribut | 1 |
| Backend/Device Issue - VISIBLE_DEVICE support is related to device visibility an | 1 |
| Supported - XPU does not support NestedTensor for SDPA operations. | 1 |
| Timeout/Performance Issue - INT4 RTN Flex-attention experienced a 5% performance | 1 |
| Backend/Device Issue - nvml support blocking test cases on XPU | 1 |
| Backend/Device Issue - torch.library.register_autocast does not support XPU devi | 1 |
| Skip/No Test Exists - test is marked to be skipped with no valid test to execute | 1 |
| Distributed/Gloo Issue - multithread support in distributed operations is affect | 1 |
| Supported - Sparse CSR gemm-like operations are not supported yet. | 1 |
| 10 - Feature Not Supported | 1 |
| DNNL/OneDNN Issue - Performance drop with oneDNN 3.11.1 in Dynamic INT8 for Llam | 1 |
| Skip/No Test Exists - test is skipped expecting current_work_stream is not null | 1 |
| Backend/Device Issue - Implementation required for XPU backend | 1 |
| Backend/Device Issue - Profiler result discrepancy on B70 device. | 1 |
| Failure - Torch not compiled with CUDA enabled assertion error | 1 |
| Skip/No Test Exists - test is skipped due to Softmax tolerance issue | 1 |
| Skip/No Test Exists - test was skipped due to failure with no detailed error pro | 1 |
| Failure - Expected and actual decomposition outputs do not match. | 1 |
| Others - clang-tidy checks enablement is a code quality/linter configuration task, not a runtime or functional issue. | 1 |
| Distributed/Gloo Issue - batch_isend_irecv compatibility issue on B60/XCCL | 1 |
| Distributed/Gloo Issue - all_to_all_single compatibility issue on B60/XCCL | 1 |
| Inductor/Compilation Issue - TorchRuntimeError during fake tensor call in Dynamo | 1 |
| Skip/No Test Exists - test unexpectedly succeeded and should have been skipped | 1 |
| Backend/Device Issue - XPU specific failure during fp32 training accuracy check | 1 |
| Backend/Device Issue - ZE_RESULT_ERROR_MODULE_BUILD_FAILURE indicates a problem  | 1 |
| Distributed/Gloo Issue - AssertionError in test/distributed/test_c10d_xccl.py re | 1 |
| Distributed/Gloo Issue - KeyError in distributed test related to c10d_xccl | 1 |
| Failure - Scalars are not equal in test case | 1 |
| Timeout/Performance Issue - test experienced a timeout in distributed execution  | 1 |
| Distributed/Gloo Issue - feature gaps in distributed testing for XPU with test_c | 1 |
| Timeout/Performance Issue - Example code timed out during test execution. | 1 |
| Skip/No Test Exists - test is marked as a skip with no valid test implementation | 1 |
| Dtype/Precision Issue - float16 training accuracy test failure | 1 |
| Dtype/Precision Issue - bfloat16 accuracy failure in model training | 1 |
| Inductor/Compilation Issue - SYCL compilation flag not working as expected for T | 1 |
| Backend/Device Issue - XPU benchmark enabling issue | 1 |
| Skip/No Test Exists - test is marked to be skipped with no valid test implementa | 1 |
| Backend/Device Issue - Fatal Python error on Windows unit tests related to XPU b | 1 |
| Timeout/Performance Issue - Models performance dropped ~10% - 15% | 1 |
| Timeout/Performance Issue - inference amp performance dropped ~15% | 1 |
| Timeout/Performance Issue - inference fp32 performance dropped ~25% | 1 |
| Inductor/Compilation Issue - performance regression in XLNetLMHeadModel with amp | 1 |
| Failure - encountered AssertionError during training | 1 |
| Skip/No Test Exists - test is skipped due to RuntimeError | 1 |
| Backend/Device Issue - fail_to_run on XPU for volo_d1_224 inference | 1 |
| Dtype/Precision Issue - fp32 inference accuracy failure | 1 |
| Dtype/Precision Issue - amp_bf16 training accuracy failure | 1 |
| Backend/Device Issue - pass_fds not supported on Windows | 1 |
| Dtype/Precision Issue - test failure related to complex64 data type and abs oper | 1 |
| Timeout/Performance Issue - models performance regression in testcases | 1 |
| Backend/Device Issue - XPU stub uses incorrect type for nan_to_num parameters | 1 |
| Skip/No Test Exists - test contains failed ops and is skipped | 1 |
| Failure - test_fsdp_overlap.py assertion failed with "False is not true" | 1 |
| Skip/No Test Exists - Test is marked as skipped or not executed | 1 |
| Dtype/Precision Issue - accuracy issue with float8 operations | 1 |
| Skip/No Test Exists - test_xpu new failures marked as skipped or no test availab | 1 |
| Flash Attention/Specific Ops Issue - _flash_attention_forward not supported on X | 1 |
| Skip/No Test Exists - test was skipped or not present | 1 |
| Skip/No Test Exists - test was skipped or does not exist | 1 |
| Dtype/Precision Issue - Tensor comparison failed due to small absolute differenc | 1 |
| Failure - Tensor-likes not close in Muon optimizer test | 1 |
| Timeout/Performance Issue - 20% performance drop in next token generation with D | 1 |
| Failure - mismatch between expected and actual error message | 1 |
| Dtype/Precision Issue - torch.logcumsumexp returns NaNs for complex64 input | 1 |
| Error - output shape mismatch during broadcasting | 1 |
| Failure - Expected and actual trace outputs do not match. | 1 |
| Error - source tensor shape mismatch during to_dense() for Sparse CSR backend | 1 |
| Others - Copy error not raised in test_dlpack.py test case | 1 |
| Dtype/Precision Issue - integer input causes error with deterministic algorithm  | 1 |
| Skip/No Test Exists - Key "xpu" is missing, indicating the test is skipped or no | 1 |
| Dtype/Precision Issue - accuracy failures in logspace operation | 1 |
| Skip/No Test Exists - test is marked as a skip with no detailed error traceback  | 1 |
| DNNL/OneDNN Issue - failed test cases related to oneDNN with 3.11 compared to 3. | 1 |
| Failure - test_control_flow_xpu.py got AssertionError | 1 |
| Memory/Shared Memory Issue - investigate memory layout performance for MaxPool2d | 1 |
| Failure - RuntimeError not raised as expected in test case | 1 |
| Skip/No Test Exists - test was skipped due to random failure标记 | 1 |
| Skip/No Test Exists - test was skipped due to changes in tolerance values causin | 1 |
| Timeout/Performance Issue - hf_Roberta_base model performance failed for both AS | 1 |
| Distributed/Gloo Issue - input tensor size mismatch in distributed context | 1 |
| Distributed/Gloo Issue - missing attribute '_gather' in distributed context | 1 |
| Timeout/Performance Issue - 50% performance drop in INT4 RTN Flex-attention with | 1 |
| Skip/No Test Exists - test is marked as skipped due to random failures | 1 |
| Backend/Device Issue - aten::flip not implemented for QuantizedXPU backend | 1 |
| Mismatch - INT4 GPTQ failed due to TorchAO API change. | 1 |
| Timeout/Performance Issue - Work ran time out after 0 milliseconds in distribute | 1 |
| Distributed/Gloo Issue - Barrier Timeout Error in distributed testing | 1 |
| Distributed/Gloo Issue - Hang issues with test_distributed_spawn.py suggest a pr | 1 |
| Failure - cur_target out of bounds assertion failed | 1 |
| Distributed/Gloo Issue - Accuracy issues in distributed testing with test_distri | 1 |
| Dtype/Precision Issue - XPU Autocast does not support fp32 dtypes | 1 |
| Others - Random failure with no traceback or specific error provided | 1 |
| Error - 'NoneType' object has no attribute 'clone' due to missing object reference | 1 |
| Failure - Tensor-likes are not close! in test_vmap_exhaustive_addmv_xpu_float32 | 1 |
| Backend/Device Issue - XPU related failure in test cases on Windows with BMG | 1 |
| Distributed/Gloo Issue - Backend xccl does not implement getBackendOptions. | 1 |
| Backend/Device Issue - failure occurs specifically on BMG WSL2 XPU backend | 1 |
| Timeout/Performance Issue - hf_Reformer performance drop reported. | 1 |
| Timeout/Performance Issue - inference performance drop | 1 |
| Inductor/Compilation Issue - Performance impact caused by TORCHINDUCTOR_ONLINE_S | 1 |
| Distributed/Gloo Issue - test_schedule_multiproc.py is hanging during distribute | 1 |
| Others - incomplete traceback and insufficient information to determine root cause | 1 |
| Failure - test_to() failed during rnn isinstance() check | 1 |
| Failure - Scalars are not equal! | 1 |
| Dtype/Precision Issue - Unsupported dtype Half / torch.float16 | 1 |
| Inductor/Compilation Issue - Adding freezing pattern for fusing int4 mm kernel i | 1 |
| Timeout/Performance Issue - First token latency drops significantly with change  | 1 |
| Timeout/Performance Issue - INT4 GPTQ shows worse performance compared with RTN  | 1 |
| Dtype/Precision Issue - accuracy fluctuations due to INT4 quantization methods ( | 1 |
| Skip/No Test Exists - Test was skipped due to random crashed cases. | 1 |
| Others - warning treated as error but no traceback or specific error provided | 1 |
| Backend/Device Issue - XPU device check failure in test | 1 |
| DNNL/OneDNN Issue - could not construct a memory descriptor using strides | 1 |
| Backend/Device Issue - Attempt to instantiate dummy base class CUDAGraph on XPU | 1 |
| DNNL/OneDNN Issue - Float8_e4m3fnuz is not supported in oneDNN. | 1 |
| Others - Test expects failure but passed unexpectedly, no specific error trace provided. | 1 |
| Backend/Device Issue - missing attribute '_scatter' in torch._C for XPU device | 1 |
| Backend/Device Issue - missing XPU-specific attribute in torch._C for tunableop  | 1 |
| Mismatch - Test method 'test_qsoftmax' is missing in 'TestQuantizedOpsXPU' class. | 1 |
| Failure - wrong number of dimensions for int4 conversion op | 1 |
| Backend/Device Issue - map2_ is only implemented on CPU tensors, indicating lack | 1 |
| Memory/Shared Memory Issue - _share_fd_ is not available on XPU, indicating a sh | 1 |
| Backend/Device Issue - _histc_xpu lacks deterministic implementation on XPU back | 1 |
| Error - tensor size exceeds int32_t maximum limit | 1 |
| Failure - Scalars are not equal assertion error in test | 1 |
| Dtype/Precision Issue - test_dtypes issue related to data types in conv_transpos | 1 |
| Backend/Device Issue - different output on BMG and PVC devices | 1 |
| Skip/No Test Exists - test_cuda.py gaps indicate missing or skipped tests. | 1 |
| Skip/No Test Exists - test_matmul_cuda.py gaps indicate missing or skipped tests | 1 |
| Backend/Device Issue - Host stuck due to excessive kernel submissions on XPU | 1 |
| Timeout/Performance Issue - UT hang indicates a timeout or performance bottlenec | 1 |
| Backend/Device Issue - SYCL runtime not detected on XPU | 1 |
| Memory/Shared Memory Issue - failure in test_share_memory_xpu related to shared  | 1 |
| Failure - mismatch between expected and actual error messages | 1 |
| Memory/Shared Memory Issue - UR backend returns out of resources error (40) indi | 1 |
| Flash Attention/Specific Ops Issue - aten::_flash_attention_forward not implemen | 1 |
| DNNL/OneDNN Issue - Test failure related to oneDNN decomposition on XPU with flo | 1 |
| Error - 'NoneType' object has no attribute 'clone' due to missing object handling | 1 |
| Skip/No Test Exists - test is marked as a bug skip or not implemented properly | 1 |
| Error - Nested tensor operation with non-nested tensor input | 1 |
| Backend/Device Issue - XPU support missing for NestedTensor operations | 1 |
| Failure - Booleans mismatch assertion error | 1 |
| Backend/Device Issue - XPU support required for tf32_on_and_off() test | 1 |
| Memory/Shared Memory Issue - XPU out of memory error occurred | 1 |
| Backend/Device Issue - SDPA uses different backend compared with IPEX on XPU | 1 |
| Error - Data corruption detected during test execution | 1 |
| Dtype/Precision Issue - "logaddexp_xpu" not implemented for 'Complex' dtype | 1 |
| Backend/Device Issue - missing library version for XPU backend compatibility | 1 |
| Failure - Roofline estimation requires CUDA capabilities assertion failed | 1 |
| Skip/No Test Exists - test is empty or not applicable | 1 |
| Backend/Device Issue - XPU benchmark enabling issue in TorchAO | 1 |
| Backend/Device Issue - Training on XPU with TorchAO and native PyTorch is not fu | 1 |
| Supported - Float8 training is not supported on XPU. | 1 |
| Supported - FP8 conv is not supported yet in TorchAO | 1 |
| Backend/Device Issue - MOE training not enabled on XPU | 1 |
| Backend/Device Issue - aten::_efficient_attention_forward not supported on CPU b | 1 |
| Flash Attention/Specific Ops Issue - test involves aten._flash_attention_forward | 1 |
| Backend/Device Issue - XPU trace event timing discrepancy | 1 |
| Backend/Device Issue - XPU profiler warning during fork process execution | 1 |
| Mismatch - Mismatch in output of aten.view.default between FakeTensor and concrete Tensors. | 1 |
| Backend/Device Issue - Torch not compiled with CUDA enabled for Intel GPU suppor | 1 |
| DNNL/OneDNN Issue - oneDNN matmul primitive descriptor creation failure | 1 |
| Error - empty_sparse_compressed expects non-block layout but received SparseBsr | 1 |
| Error - forward gradient size mismatch with original Tensor | 1 |
| DNNL/OneDNN Issue - could not create a primitive descriptor for deconvolution fo | 1 |
| Backend/Device Issue - unexpected warning from non-optimal triton kernel paramet | 1 |
| Memory/Shared Memory Issue - sdpa backward kernel needed to reduce memory usage | 1 |
| Backend/Device Issue - inputs are not on the same GPU device | 1 |
| Backend/Device Issue - device mismatch between crow_indices and col_indices on X | 1 |
| Backend/Device Issue - unknown SPIR-V extension 'SPV_INTEL_subgroup_matrix_multi | 1 |
| Dtype/Precision Issue - float8_e4m3fn precision overflow | 1 |
| Timeout/Performance Issue - AO Performance issue track | 1 |
| Others - Looking for use case example of torch-xpu-ops.lib in SYCL C++ extension | 1 |
| Backend/Device Issue - FP8/MXFP8 Ops related to XPU and CUDA alignment | 1 |
| Timeout/Performance Issue - RTN performance regression in next-token latency for | 1 |
| Failure - assert vr is not None error encountered | 1 |
| Flash Attention/Specific Ops Issue - request to support flash attention op on XP | 1 |
| Memory/Shared Memory Issue - register spill in reduction and norm operations | 1 |
| Memory/Shared Memory Issue - register spill in DistributionElementwiseKernelFunc | 1 |
| Backend/Device Issue - functionality not working on BMG for PyTorch profiling | 1 |
| Failure - Mul tiheadAttention does not support NestedTensor outside of its fast path | 1 |
| Failure - Scalars are not equal in test assertion | 1 |
| Failure - Scalars are not equal in test comparison | 1 |
| Failure - test_flatten_mesh_3d encountered an assertion error | 1 |
| Distributed/Gloo Issue - distributed UT cases need support from sac_estimator.py | 1 |
| Timeout/Performance Issue - BMG d2h copy is significantly slower compared to PVC | 1 |
| Memory/Shared Memory Issue - XPU and CUDA show different memory allocation outpu | 1 |
| Memory/Shared Memory Issue - Avoiding copy in FFT kernels relates to memory hand | 1 |
| Backend/Device Issue - Exception Code 0xC0000005 indicates an access violation l | 1 |
| Distributed/Gloo Issue - related to Distributed Data Parallel update example | 1 |
| Backend/Device Issue - XPU functions in yaml related to upstream backend issues | 1 |
| Backend/Device Issue - torch.xpu.is_available() initializes GPU context unintent | 1 |
| Backend/Device Issue - nd_item::barrier is deprecated on XPU backend. | 1 |
| device-specific backend problem. | 1 |
| Dtype/Precision Issue - inf returned using float16 in TransformerEncoderLayer on | 1 |
| Backend/Device Issue - work-group size exceeds device limitations on XPU | 1 |
| Memory/Shared Memory Issue - error originated from shared memory connection in t | 1 |
| Memory/Shared Memory Issue - related to memory efficient optimizers in TorchAO | 1 |
| Mismatch - torch.xpu._sleep is not implemented or available in the current setup. | 1 |
| Failure - Tensor values not equal or close in test_depthwise_conv_64bit_indexing_xpu | 1 |
| Backend/Device Issue - CUDA not available on the system | 1 |
| Error - cannot create weak reference to 'torch.Event' object | 1 |
| Backend/Device Issue - segfault related to XPU device operation in test | 1 |
| Backend/Device Issue - XPU backend for torch.linalg.cholesky is not implemented | 1 |
| Backend/Device Issue - Implementation required for XPU dequantization of CUDA in | 1 |
| Backend/Device Issue - XPU backend for torch.linalg.pinv is not implemented | 1 |
| Backend/Device Issue - XPU backend for torch.linalg.svd not implemented | 1 |
| Backend/Device Issue - XPU backend implementation missing for torch.linalg.qr | 1 |
| precision-related failure in performance test | 1 |
| Backend/Device Issue - fail_accuracy on XPU for specific models | 1 |
| Dtype/Precision Issue - amp_fp16 inference accuracy failure | 1 |
| Memory/Shared Memory Issue - extra copy caused by channel last aten::hardswish_ | 1 |
| Dtype/Precision Issue - INT8 accuracy lower than FP32 due to precision reduction | 1 |
| Backend/Device Issue - Accuracy issue on XPU for Torch-xpu-ops UT | 1 |
| 11 - Timeout/Performance Issue - Torch XPU Profiler is not reliable | 1 |
| Others - New kernels for concat, no specific error provided. | 1 |
| compilation-related task or issue. | 1 |
| Backend/Device Issue - XPU does not support backward or grad for SDPA operation | 1 |
| Memory/Shared Memory Issue - torch.xpu.empty_cache() introduces memory leak | 1 |
| Backend/Device Issue - missing attribute '_sleep' in torch.xpu module | 1 |
| Mismatch - Request for an API to query GPU type (iGPU/dGPU) is missing or not properly implemented. | 1 |
| Memory/Shared Memory Issue - missing op support for `model.share_memory()` indic | 1 |
| Distributed/Gloo Issue - Accuracy gap in _composable/fsdp on Xelink suggests a d | 1 |
| Backend/Device Issue - inconsistent oneAPI versions cause XPU backend errors | 1 |
| Others - building warning tracking issue | 1 |
| Backend/Device Issue - aten::_grouped_mm not implemented for XPU device | 1 |
| Distributed/Gloo Issue - ReduceOp.PREMUL_SUM not supported with XCCL | 1 |
| Distributed/Gloo Issue - Operator lacks a sharding strategy in distributed conte | 1 |
| Distributed/Gloo Issue - mixed torch.Tensor and DTensor in distributed operation | 1 |
| Backend/Device Issue - XPU device does not support the 'symm_mem::fused_scaled_m | 1 |
| Distributed/Gloo Issue - 'fused_all_gather_scaled_matmul' not found in graph dur | 1 |
| Failure - 'fused_all_gather_matmul' not found in AOT ID list | 1 |
| Backend/Device Issue - XPU device does not support the 'symm_mem::fused_matmul_r | 1 |
| Backend/Device Issue - coredump related to XPU operations in Torch-xpu-ops UT | 1 |
| Backend/Device Issue - test cases hang on BMG Ubuntu related to XPU backend | 1 |
| Memory/Shared Memory Issue - Out of memory (OOM) leading to tensor context break | 1 |
| Backend/Device Issue - unexpected error on XPU for LNL Windows | 1 |
| Skip/No Test Exists - No test was implemented or executed. | 1 |
| Dtype/Precision Issue - value cannot be converted to at::BFloat16 without overfl | 1 |
| Backend/Device Issue - related to SYCL RT and kernel-specific max work group siz | 1 |
| Backend/Device Issue - model code forces use of CUDA instead of XPU | 1 |
| Backend/Device Issue - xpu not supported | 1 |
| Others - abstract utility functions in ATen operator implementation | 1 |
| Backend/Device Issue - register spill evaluation in SYCL kernel on XPU | 1 |

### 1. Information Required

Information needed from reporters to proceed with issue triage.

| ID | Title | Owner | Owner Transferred | Required Info | Priority | Action Reason | Category | Root Cause | PR | Module | Test Module |
|---|-------|-------|-------------------|--------------|---------|--------|----------|-----------|-----|--------|-------------|
| [1171](https://github.com/intel/torch-xpu-ops/issues/1171) | LNL Windows got unexpected error me | xuhancn, chunhuanMeng | daisyden | Need reproduce steps (Only for bugs or performance issue) | P2 | Missing reproduce steps for bug/performance issue | Others | Backend/Device Issue - unexpected error on XPU for LNL Windows |  | aten_ops | ut |
| [1324](https://github.com/intel/torch-xpu-ops/issues/1324) | [Win] UR Error when OOM and break t | Stonepia | Stonepia | Need reproduce steps (Only for bugs or performance issue) | P2 | Missing reproduce steps for bug/performance issue | Others | Memory/Shared Memory Issue - Out of memory (OOM) leading to tensor context break |  | aten_ops | ut |
| [1548](https://github.com/intel/torch-xpu-ops/issues/1548) | [distributed] AssertionError: 'fuse | Chao1Han | PenghuiCheng | Need reproduce steps (Only for bugs or performance issue) | P2 | Missing reproduce steps for bug/performance issue | Distributed | Failure - 'fused_all_gather_matmul' not found in AOT ID list |  | distributed | ut |
| [1549](https://github.com/intel/torch-xpu-ops/issues/1549) | [distributed] AssertionError: 'fuse | Chao1Han | PenghuiCheng | Need reproduce steps (Only for bugs or performance issue) | P2 | Missing reproduce steps for bug/performance issue | Distributed | Distributed/Gloo Issue - 'fused_all_gather_scaled_matmul' not found in graph dur |  | distributed | ut |
| [1555](https://github.com/intel/torch-xpu-ops/issues/1555) | [distributed] RuntimeError: aten.ad | chuanqi129 | PenghuiCheng | Need reproduce steps (Only for bugs or performance issue) | P2 | Missing reproduce steps for bug/performance issue | Distributed | Distributed/Gloo Issue - mixed torch.Tensor and DTensor in distributed operation |  | distributed | ut |
| [1571](https://github.com/intel/torch-xpu-ops/issues/1571) | [distributed] ValueError: Cannot us | zhangxiaoli73 | daisyden | Need reproduce steps (Only for bugs or performance issue) | P2 | Missing reproduce steps for bug/performance issue | Distributed | Distributed/Gloo Issue - ReduceOp.PREMUL_SUM not supported with XCCL |  | distributed | ut |
| [1649](https://github.com/intel/torch-xpu-ops/issues/1649) | [cpp extension] Provide a clear err | dvrogozh | ZhaoqiongZ | Need reproduce steps (Only for bugs or performance issue) | P2 | Missing reproduce steps for bug/performance issue | Others | Backend/Device Issue - inconsistent oneAPI versions cause XPU backend errors |  | aten_ops | ut |
| [1661](https://github.com/intel/torch-xpu-ops/issues/1661) | [distributed] Accuracy gap in _comp | githubsgi | PenghuiCheng | Need reproduce steps (Only for bugs or performance issue) | P2 | Missing reproduce steps for bug/performance issue | Distributed | Distributed/Gloo Issue - Accuracy gap in _composable/fsdp on Xelink suggests a d |  | distributed | ut |
| [1727](https://github.com/intel/torch-xpu-ops/issues/1727) | [distributed] AttributeError: modul | guangyey | PenghuiCheng | Need reproduce steps (Only for bugs or performance issue) | P2 | Missing reproduce steps for bug/performance issue | Distributed | Backend/Device Issue - missing attribute '_sleep' in torch.xpu module |  | distributed | ut |
| [1749](https://github.com/intel/torch-xpu-ops/issues/1749) | transformers UT failure in XPU beca | LuFinch | sywangyi | Need reproduce steps (Only for bugs or performance issue) | P2 | Missing reproduce steps for bug/performance issue | Flash Attention / Transformer Related | Backend/Device Issue - XPU does not support backward or grad for SDPA operation |  | aten_ops | ut |
| [1784](https://github.com/intel/torch-xpu-ops/issues/1784) | [Performance] Torch XPU Profiler is | jfedorov | liangan1 | Need reproduce steps (Only for bugs or performance issue) | P2 | Missing reproduce steps for bug/performance issue | Others | 11 - Timeout/Performance Issue - Torch XPU Profiler is not reliable |  | profiling | ut |
| [1833](https://github.com/intel/torch-xpu-ops/issues/1833) | [PT2E] INT8 accuracy is lower than  | chunhuanMeng | mengfei25 | Need reproduce steps (Only for bugs or performance issue) | P2 | Missing reproduce steps for bug/performance issue | TorchAO | Dtype/Precision Issue - INT8 accuracy lower than FP32 due to precision reduction |  | low_precision | ut |
| [1877](https://github.com/intel/torch-xpu-ops/issues/1877) | Torchbench model squeezenet1_1 and  | Silv3S | libohao1201 | Need reproduce steps (Only for bugs or performance issue) | P0 | Missing reproduce steps for bug/performance issue | Dtype / Precision Related | Backend/Device Issue - fail_accuracy on XPU for specific models |  | aten_ops | ut |
| [1969](https://github.com/intel/torch-xpu-ops/issues/1969) | torch._dynamo.exc.InternalTorchDyna | guangyey | shangerxin | Need reproduce steps (Only for bugs or performance issue) | P2 | Missing reproduce steps for bug/performance issue | PT2E | Error - cannot create weak reference to 'torch.Event' object |  | aten_ops | ut |
| [1970](https://github.com/intel/torch-xpu-ops/issues/1970) | torch._dynamo.exc.BackendCompilerFa | None | shangerxin | Need reproduce steps (Only for bugs or performance issue) | P2 | Missing reproduce steps for bug/performance issue | PT2E | Backend/Device Issue - CUDA not available on the system |  | aten_ops | ut |
| [2004](https://github.com/intel/torch-xpu-ops/issues/2004) | [distributed][shared_tensor] test\d | libohao1201 | libohao1201 | Need reproduce steps (Only for bugs or performance issue) | P2 | Missing reproduce steps for bug/performance issue | Distributed | Memory/Shared Memory Issue - error originated from shared memory connection in t |  | distributed | ut |
| [2022](https://github.com/intel/torch-xpu-ops/issues/2022) | [Windows] [CI] [UT] AssertionError: | None | RUIJIEZHONG66166 | Need reproduce steps (Only for bugs or performance issue) | P2 | Missing reproduce steps for bug/performance issue | Dtype / Precision Related | Failure - Tensor-likes are not close! |  | aten_ops | ut |
| [2024](https://github.com/intel/torch-xpu-ops/issues/2024) | AssertionError: Torch not compiled  | daisyden | mengfei25 | Need reproduce steps (Only for bugs or performance issue) | P2 | Missing reproduce steps for bug/performance issue | Distributed |  |  | aten_ops | ut |
| [2157](https://github.com/intel/torch-xpu-ops/issues/2157) | BMG d2h copy is very slow compare t | jianyizh, mengfei25 | jianyizh | Need reproduce steps (Only for bugs or performance issue) | P2 | Missing reproduce steps for bug/performance issue | Others | Timeout/Performance Issue - BMG d2h copy is significantly slower compared to PVC |  | aten_ops | ut |
| [2165](https://github.com/intel/torch-xpu-ops/issues/2165) | [distributed] test_device_mesh.py:: | jemitche1 | zxd1997066 | Need reproduce steps (Only for bugs or performance issue) | P2 | Missing reproduce steps for bug/performance issue | Distributed | Failure - test_flatten_mesh_3d encountered an assertion error |  | distributed | ut |
| [2169](https://github.com/intel/torch-xpu-ops/issues/2169) | Frame size comparison failed in tes | guangyey | daisyden | Need reproduce steps (Only for bugs or performance issue) | P2 | Missing reproduce steps for bug/performance issue | Distributed | Failure - Scalars are not equal in test comparison |  | aten_ops | ut |
| [2182](https://github.com/intel/torch-xpu-ops/issues/2182) | test_transform_bias_rescale_qkv_nes | PawelSwider2000 | wincent8 | Need reproduce steps (Only for bugs or performance issue) | P2 | Missing reproduce steps for bug/performance issue | Distributed | Failure - Scalars are not equal in test assertion |  | aten_ops | ut |
| [2201](https://github.com/intel/torch-xpu-ops/issues/2201) | [TorchAO][BMG] When using paged att | Stonepia | MingxuZh | Need reproduce steps (Only for bugs or performance issue) | P2 | Missing reproduce steps for bug/performance issue | TorchAO | Failure - assert vr is not None error encountered |  | AO | ut |
| [2217](https://github.com/intel/torch-xpu-ops/issues/2217) | AO Performance issue track | Stonepia | liangan1 | Need reproduce steps (Only for bugs or performance issue) | P2 | Missing reproduce steps for bug/performance issue | Others | Timeout/Performance Issue - AO Performance issue track |  | AO | ut |
| [2219](https://github.com/intel/torch-xpu-ops/issues/2219) | float8_e4m3fn precision overflow | CuiYifeng, yucai-intel | jiqing-feng | Need reproduce steps (Only for bugs or performance issue) | P2 | Missing reproduce steps for bug/performance issue | Dtype / Precision Related | Dtype/Precision Issue - float8_e4m3fn precision overflow |  | aten_ops | ut |
| [2239](https://github.com/intel/torch-xpu-ops/issues/2239) | Exception: could not create a primi | wpietka | zxd1997066 | Need reproduce steps (Only for bugs or performance issue) | P2 | Missing reproduce steps for bug/performance issue | Distributed | DNNL/OneDNN Issue - could not create a primitive descriptor for deconvolution fo |  | aten_ops | ut |
| [2240](https://github.com/intel/torch-xpu-ops/issues/2240) | RuntimeError: Trying to set a forwa | gplutop7 | zxd1997066 | Need reproduce steps (Only for bugs or performance issue) | P2 | Missing reproduce steps for bug/performance issue | Distributed | Error - forward gradient size mismatch with original Tensor |  | aten_ops | ut |
| [2245](https://github.com/intel/torch-xpu-ops/issues/2245) | oneDNN matmul received incorrect sh | CuiYifeng | wincent8 | Need reproduce steps (Only for bugs or performance issue) | P2 | Missing reproduce steps for bug/performance issue | Distributed | DNNL/OneDNN Issue - oneDNN matmul primitive descriptor creation failure |  | aten_ops | ut |
| [2263](https://github.com/intel/torch-xpu-ops/issues/2263) | [xpu][bug] XPU Trace event ends too | PawelSwider2000 | chuanqi129 | Need reproduce steps (Only for bugs or performance issue) | P2 | Missing reproduce steps for bug/performance issue | Others | Backend/Device Issue - XPU trace event timing discrepancy |  | profiling | ut |
| [2340](https://github.com/intel/torch-xpu-ops/issues/2340) | [distributed][_tools] AssertionErro | githubsgi | zxd1997066 | Need reproduce steps (Only for bugs or performance issue) | P2 | Missing reproduce steps for bug/performance issue | Distributed | Failure - Roofline estimation requires CUDA capabilities assertion failed |  | distributed | ut |
| [2389](https://github.com/intel/torch-xpu-ops/issues/2389) | [Bug Skip]: RuntimeError: Data corr | PatrykWilczewski | kaileiyx | Need reproduce steps (Only for bugs or performance issue) | P2 | Missing reproduce steps for bug/performance issue | Distributed | Error - Data corruption detected during test execution |  | aten_ops | ut |
| [2392](https://github.com/intel/torch-xpu-ops/issues/2392) | [Bug Skip]: torch.OutOfMemoryError: | xuhancn | RUIJIEZHONG66166 | Need reproduce steps (Only for bugs or performance issue) | P2 | Missing reproduce steps for bug/performance issue | Others | Memory/Shared Memory Issue - XPU out of memory error occurred |  | aten_ops | ut |
| [2404](https://github.com/intel/torch-xpu-ops/issues/2404) | [distributed][checkpoint] Assertion | None | zxd1997066 | Need reproduce steps (Only for bugs or performance issue) | P2 | Missing reproduce steps for bug/performance issue | Distributed | Failure - Booleans mismatch assertion error |  | aten_ops | ut |
| [2425](https://github.com/intel/torch-xpu-ops/issues/2425) | [upstream_ut]  RuntimeError: Expect | BBBela | daisyden | Need reproduce steps (Only for bugs or performance issue) | P2 | Missing reproduce steps for bug/performance issue | Dtype / Precision Related | Error - Nested tensor operation with non-nested tensor input |  | aten_ops | ut |
| [2434](https://github.com/intel/torch-xpu-ops/issues/2434) | [Bug Skip]: New failures 2025-11-28 | AKloniecki | mengfei25 | Need reproduce steps (Only for bugs or performance issue) | P2 | Missing reproduce steps for bug/performance issue | TorchAO | Skip/No Test Exists - test is marked as a bug skip or not implemented properly |  | aten_ops | ut |
| [2439](https://github.com/intel/torch-xpu-ops/issues/2439) | [oneDNN] TestDecompXPU.test_quick_a | libohao1201 | mengfei25 | Need reproduce steps (Only for bugs or performance issue) | P2 | Missing reproduce steps for bug/performance issue | Dtype / Precision Related | DNNL/OneDNN Issue - Test failure related to oneDNN decomposition on XPU with flo |  | aten_ops | ut |
| [2440](https://github.com/intel/torch-xpu-ops/issues/2440) | [For UT failures classify] Save ref | None | RUIJIEZHONG66166 | Need reproduce steps (Only for bugs or performance issue) | P2 | Missing reproduce steps for bug/performance issue | Others | Skip/No Test Exists - no test or error details provided |  | inductor | ut |
| [2444](https://github.com/intel/torch-xpu-ops/issues/2444) | [upstream_ut]  RuntimeError: UR bac | Silv3S | wincent8 | Need reproduce steps (Only for bugs or performance issue) | P2 | Missing reproduce steps for bug/performance issue | Dtype / Precision Related | Memory/Shared Memory Issue - UR backend returns out of resources error (40) indi |  | aten_ops | ut |
| [2446](https://github.com/intel/torch-xpu-ops/issues/2446) | [Bug Skip]: AssertionError: "Simula | None | kaileiyx | Need reproduce steps (Only for bugs or performance issue) | P2 | Missing reproduce steps for bug/performance issue | Distributed | Failure - mismatch between expected and actual error messages |  | aten_ops | ut |
| [2447](https://github.com/intel/torch-xpu-ops/issues/2447) | test_share_memory_xpu failure | None | daisyden | Need reproduce steps (Only for bugs or performance issue) | P2 | Missing reproduce steps for bug/performance issue | Others | Memory/Shared Memory Issue - failure in test_share_memory_xpu related to shared  |  | aten_ops | ut |
| [2463](https://github.com/intel/torch-xpu-ops/issues/2463) | [Bug Skip]: OSError: SYCL runtime i | xuhancn | RUIJIEZHONG66166 | Need reproduce steps (Only for bugs or performance issue) | P2 | Missing reproduce steps for bug/performance issue | Others | Backend/Device Issue - SYCL runtime not detected on XPU |  | aten_ops | ut |
| [2479](https://github.com/intel/torch-xpu-ops/issues/2479) | [Bug] torch.rand output different r | Stonepia, CuiYifeng | zufangzhu | Need reproduce steps (Only for bugs or performance issue) | P2 | Missing reproduce steps for bug/performance issue | Others | Backend/Device Issue - different output on BMG and PVC devices |  | aten_ops | ut |
| [2491](https://github.com/intel/torch-xpu-ops/issues/2491) | [upstream_ut]  AssertionError: Fals | PatrykWilczewski | libohao1201 | Need reproduce steps (Only for bugs or performance issue) | P2 | Missing reproduce steps for bug/performance issue | Dtype / Precision Related | Failure - test assertion failed with False is not true |  | aten_ops | ut |
| [2494](https://github.com/intel/torch-xpu-ops/issues/2494) | [upstream_ut]  AssertionError: Scal | PawelSwider2000 | libohao1201 | Need reproduce steps (Only for bugs or performance issue) | P2 | Missing reproduce steps for bug/performance issue | Dtype / Precision Related | Failure - Scalars are not equal assertion error in test |  | aten_ops | ut |
| [2510](https://github.com/intel/torch-xpu-ops/issues/2510) | [upstream_ut]  RuntimeError: Expect | PawelSwider2000 | libohao1201 | Need reproduce steps (Only for bugs or performance issue) | P2 | Missing reproduce steps for bug/performance issue | Dtype / Precision Related | Error - tensor size exceeds int32_t maximum limit |  | aten_ops | ut |
| [2513](https://github.com/intel/torch-xpu-ops/issues/2513) | [upstream_ut]  RuntimeError: _share | gplutop7 | libohao1201 | Need reproduce steps (Only for bugs or performance issue) | P2 | Missing reproduce steps for bug/performance issue | Others | Memory/Shared Memory Issue - _share_fd_ is not available on XPU, indicating a sh |  | aten_ops | ut |
| [2531](https://github.com/intel/torch-xpu-ops/issues/2531) | [upstream_ut]  AssertionError: Torc | daisyden | daisyden | Need reproduce steps (Only for bugs or performance issue) | P2 | Missing reproduce steps for bug/performance issue | Distributed |  |  | unknown | ut |
| [2532](https://github.com/intel/torch-xpu-ops/issues/2532) | Title: [upstream_ut]  AssertionErro | yucai-intel | daisyden | Need reproduce steps (Only for bugs or performance issue) | P2 | Missing reproduce steps for bug/performance issue | Distributed | Failure - wrong number of dimensions for int4 conversion op |  | aten_ops | ut |
| [2533](https://github.com/intel/torch-xpu-ops/issues/2533) | Title: [upstream_ut]  AttributeErro | astachowiczhabana | daisyden | Need reproduce steps (Only for bugs or performance issue) | P2 | Missing reproduce steps for bug/performance issue | Distributed | Mismatch - Test method 'test_qsoftmax' is missing in 'TestQuantizedOpsXPU' class. |  | aten_ops | ut |
| [2535](https://github.com/intel/torch-xpu-ops/issues/2535) | Title: [upstream_ut]  AttributeErro | Silv3S | daisyden | Need reproduce steps (Only for bugs or performance issue) | P2 | Missing reproduce steps for bug/performance issue | Distributed | Backend/Device Issue - missing XPU-specific attribute in torch._C for tunableop  |  | aten_ops | ut |
| [2536](https://github.com/intel/torch-xpu-ops/issues/2536) | Title: [upstream_ut]  AttributeErro | daisyden | daisyden | Need reproduce steps (Only for bugs or performance issue) | P2 | Missing reproduce steps for bug/performance issue | Distributed | Backend/Device Issue - missing attribute '_scatter' in torch._C for XPU device |  | aten_ops | ut |
| [2537](https://github.com/intel/torch-xpu-ops/issues/2537) | Title: [upstream_ut]  Failed: Unexp | PatrykWilczewski | daisyden | Need reproduce steps (Only for bugs or performance issue) | P2 | Missing reproduce steps for bug/performance issue | Others | Others - Test expects failure but passed unexpectedly, no specific error trace provided. |  | aten_ops | ut |
| [2538](https://github.com/intel/torch-xpu-ops/issues/2538) | Title: [upstream_ut]  RuntimeError: | Silv3S | daisyden | Need reproduce steps (Only for bugs or performance issue) | P2 | Missing reproduce steps for bug/performance issue | Distributed | DNNL/OneDNN Issue - Float8_e4m3fnuz is not supported in oneDNN. |  | aten_ops | ut |
| [2539](https://github.com/intel/torch-xpu-ops/issues/2539) | Title: [upstream_ut]  RuntimeError: | None | daisyden | Need reproduce steps (Only for bugs or performance issue) | P2 | Missing reproduce steps for bug/performance issue | Distributed | Backend/Device Issue - Attempt to instantiate dummy base class CUDAGraph on XPU |  | aten_ops | ut |
| [2541](https://github.com/intel/torch-xpu-ops/issues/2541) | Title: [upstream_ut]  RuntimeError: | yucai-intel | daisyden | Need reproduce steps (Only for bugs or performance issue) | P2 | Missing reproduce steps for bug/performance issue | Distributed | DNNL/OneDNN Issue - could not construct a memory descriptor using strides |  | aten_ops | ut |
| [2560](https://github.com/intel/torch-xpu-ops/issues/2560) | [UT] "RuntimeError: iter.device(arg | CuiYifeng | libohao1201 | Need reproduce steps (Only for bugs or performance issue) | P2 | Missing reproduce steps for bug/performance issue | Others | Backend/Device Issue - XPU device check failure in test |  | aten_ops | ut |
| [2562](https://github.com/intel/torch-xpu-ops/issues/2562) | Warning as Error | chunhuanMeng | EikanWang | Need reproduce steps (Only for bugs or performance issue) | P2 | Missing reproduce steps for bug/performance issue | Others | Others - warning treated as error but no traceback or specific error provided |  | aten_ops | ut |
| [2572](https://github.com/intel/torch-xpu-ops/issues/2572) | [TorchAO][UT] test/dtypes/test_affi | xiaowangintel | zxd1997066 | Need reproduce steps (Only for bugs or performance issue) | P0 | Missing reproduce steps for bug/performance issue | TorchAO | Failure - Tensor-likes are not close! |  | AO | build |
| [2580](https://github.com/intel/torch-xpu-ops/issues/2580) | [TorchAO][UT] test/test_low_bit_opt | arlesniak | zxd1997066 | Need reproduce steps (Only for bugs or performance issue) | P0 | Missing reproduce steps for bug/performance issue | TorchAO | Failure - Tensor-likes are not close! |  | AO | build |
| [2595](https://github.com/intel/torch-xpu-ops/issues/2595) | [Bug Skip]: Random crashed cases 20 | None | CuiYifeng | Need reproduce steps (Only for bugs or performance issue) | P0 | Missing reproduce steps for bug/performance issue | Sparse Operations Related | Skip/No Test Exists - Test was skipped due to random crashed cases. |  | aten_ops | ut |
| [2596](https://github.com/intel/torch-xpu-ops/issues/2596) | [TorchAO][BMG]Observed accuracy flu | None | LifengWang | Need reproduce steps (Only for bugs or performance issue) | P0 | Missing reproduce steps for bug/performance issue | TorchAO | Dtype/Precision Issue - accuracy fluctuations due to INT4 quantization methods ( |  | AO | ut |
| [2597](https://github.com/intel/torch-xpu-ops/issues/2597) | [TorchAO][BMG] INT4 GPTQ shows wors | xiaowangintel | LifengWang | Need reproduce steps (Only for bugs or performance issue) | P2 | Missing reproduce steps for bug/performance issue | TorchAO | Timeout/Performance Issue - INT4 GPTQ shows worse performance compared with RTN  |  | AO | ut |
| [2615](https://github.com/intel/torch-xpu-ops/issues/2615) | [Bug Skip]: New failures RuntimeErr | CuiYifeng | kaileiyx | Need reproduce steps (Only for bugs or performance issue) | P2 | Missing reproduce steps for bug/performance issue | Distributed | Dtype/Precision Issue - Unsupported dtype Half / torch.float16 |  | aten_ops | ut |
| [2630](https://github.com/intel/torch-xpu-ops/issues/2630) | Title: [upstream_ut]  AssertionErro | jmamzax | daisyden | Need reproduce steps (Only for bugs or performance issue) | P2 | Missing reproduce steps for bug/performance issue | Distributed | Failure - Scalars are not equal! |  | aten_ops | ut |
| [2639](https://github.com/intel/torch-xpu-ops/issues/2639) | test_to() failed during rnn isinsta | Silv3S | daisyden | Need reproduce steps (Only for bugs or performance issue) | P2 | Missing reproduce steps for bug/performance issue | Distributed | Failure - test_to() failed during rnn isinstance() check | [PR](https://github.com/intel/torch-xpu-ops/pull/3193) | aten_ops | ut |
| [2645](https://github.com/intel/torch-xpu-ops/issues/2645) | [Bug Skip]: [regression] RuntimeErr | CuiYifeng | kaileiyx | Need reproduce steps (Only for bugs or performance issue) | P0 | Missing reproduce steps for bug/performance issue | Inductor / Compilation Related | Backend/Device Issue - missing kernel for xpu in DispatchStub |  | aten_ops | ut |
| [2669](https://github.com/intel/torch-xpu-ops/issues/2669) | [upstream_ut]  AssertionError: Tens | tszulist-hbn | daisyden | Need reproduce steps (Only for bugs or performance issue) | P2 | Missing reproduce steps for bug/performance issue | Distributed | Failure - Tensor-likes are not close! in test_vmap_exhaustive_addmv_xpu_float32 |  | aten_ops | ut |
| [2676](https://github.com/intel/torch-xpu-ops/issues/2676) | Random failure in CI test | None | daisyden | Need reproduce steps (Only for bugs or performance issue) | P2 | Missing reproduce steps for bug/performance issue | Dtype / Precision Related | Others - Random failure with no traceback or specific error provided |  | unknown | ut |
| [2680](https://github.com/intel/torch-xpu-ops/issues/2680) | XPU Autocast does not support  fp32 | CuiYifeng | kaixuanliu | Need reproduce steps (Only for bugs or performance issue) | P2 | Missing reproduce steps for bug/performance issue | Dtype / Precision Related | Dtype/Precision Issue - XPU Autocast does not support fp32 dtypes |  | aten_ops | ut |
| [2686](https://github.com/intel/torch-xpu-ops/issues/2686) | [distributed] Accuracy issues with  | frost-intel | madhumitha0102 | Need reproduce steps (Only for bugs or performance issue) | P2 | Missing reproduce steps for bug/performance issue | Distributed | Distributed/Gloo Issue - Accuracy issues in distributed testing with test_distri |  | distributed | ut |
| [2689](https://github.com/intel/torch-xpu-ops/issues/2689) | [LNL][Windows] AssertionError: 'Ass | tadkrawiec | kaileiyx | Need reproduce steps (Only for bugs or performance issue) | P2 | Missing reproduce steps for bug/performance issue | Dtype / Precision Related | Failure - cur_target out of bounds assertion failed |  | aten_ops | ut |
| [2701](https://github.com/intel/torch-xpu-ops/issues/2701) | [distributed] Barrier Timeout Error | syedshahbaaz | madhumitha0102 | Need reproduce steps (Only for bugs or performance issue) | P2 | Missing reproduce steps for bug/performance issue | Distributed | Distributed/Gloo Issue - Barrier Timeout Error in distributed testing |  | distributed | ut |
| [2702](https://github.com/intel/torch-xpu-ops/issues/2702) | [distributed] RuntimeError: Work ra | syedshahbaaz | madhumitha0102 | Need reproduce steps (Only for bugs or performance issue) | P2 | Missing reproduce steps for bug/performance issue | Distributed | Timeout/Performance Issue - Work ran time out after 0 milliseconds in distribute |  | distributed | ut |
| [2707](https://github.com/intel/torch-xpu-ops/issues/2707) | [TorchAO][BMG] INT4 GPTQ failed due | xiaowangintel | LifengWang | Need reproduce steps (Only for bugs or performance issue) | P2 | Missing reproduce steps for bug/performance issue | TorchAO | Mismatch - INT4 GPTQ failed due to TorchAO API change. |  | AO | ut |
| [2720](https://github.com/intel/torch-xpu-ops/issues/2720) | [upstream_ut] RuntimeError: false I | CuiYifeng | wincent8 | Need reproduce steps (Only for bugs or performance issue) | P2 | Missing reproduce steps for bug/performance issue | Distributed | Backend/Device Issue - missing kernel for xpu in DispatchStub |  | aten_ops | ut |
| [2729](https://github.com/intel/torch-xpu-ops/issues/2729) | [Bug Skip]: Random failures 2026WW0 | Silv3S | CuiYifeng | Need reproduce steps (Only for bugs or performance issue) | P2 | Missing reproduce steps for bug/performance issue | Sparse Operations Related | Skip/No Test Exists - test is marked as skipped due to random failures |  | unknown | ut |
| [2734](https://github.com/intel/torch-xpu-ops/issues/2734) | [TorchAO][BMG] INT4 RTN Flex-attent | Stonepia, hoshibara | LifengWang | Need reproduce steps (Only for bugs or performance issue) | P2 | Missing reproduce steps for bug/performance issue | TorchAO | Timeout/Performance Issue - 50% performance drop in INT4 RTN Flex-attention with | [PR](https://github.com/pytorch/pytorch/pull/160480, https://github.com/pytorch/pytorch/pull/172316) | AO | ut |
| [2737](https://github.com/intel/torch-xpu-ops/issues/2737) | [distributed] AttributeError: modul | None | zxd1997066 | Need reproduce steps (Only for bugs or performance issue) | P2 | Missing reproduce steps for bug/performance issue | Distributed | Distributed/Gloo Issue - missing attribute '_gather' in distributed context |  | distributed | ut |
| [2738](https://github.com/intel/torch-xpu-ops/issues/2738) | [distributed] test_c10d_spawn_nccl. | jenniew | zxd1997066 | Need reproduce steps (Only for bugs or performance issue) | P2 | Missing reproduce steps for bug/performance issue | Distributed | Distributed/Gloo Issue - input tensor size mismatch in distributed context |  | distributed | ut |
| [2744](https://github.com/intel/torch-xpu-ops/issues/2744) | [Bug Skip]: extended test failures  | pbielak | daisyden | Need reproduce steps (Only for bugs or performance issue) | P2 | Missing reproduce steps for bug/performance issue | Others | Skip/No Test Exists - test was skipped due to changes in tolerance values causin |  | aten_ops | ut |
| [2751](https://github.com/intel/torch-xpu-ops/issues/2751) | [Bug Skip]: Random failures 2026WW0 | None | CuiYifeng | Need reproduce steps (Only for bugs or performance issue) | P2 | Missing reproduce steps for bug/performance issue | Sparse Operations Related | Skip/No Test Exists - test was skipped due to random failure标记 |  | unknown | ut |
| [2759](https://github.com/intel/torch-xpu-ops/issues/2759) | [Bug Skip]: New failed cases 2026-1 | AKloniecki | kaileiyx | Need reproduce steps (Only for bugs or performance issue) | P2 | Missing reproduce steps for bug/performance issue | Distributed | Failure - RuntimeError not raised as expected in test case |  | aten_ops | ut |
| [2766](https://github.com/intel/torch-xpu-ops/issues/2766) | MaxPool2d - investigate memory layo | BBBela | pbielak | Need reproduce steps (Only for bugs or performance issue) | P2 | Missing reproduce steps for bug/performance issue | Others | Memory/Shared Memory Issue - investigate memory layout performance for MaxPool2d |  | aten_ops | ut |
| [2767](https://github.com/intel/torch-xpu-ops/issues/2767) | [UT] test_control_flow_xpu.py got A | PatrykWilczewski | libohao1201 | Need reproduce steps (Only for bugs or performance issue) | P1 | Missing reproduce steps for bug/performance issue | PT2E | Failure - test_control_flow_xpu.py got AssertionError |  | aten_ops | ut |
| [2769](https://github.com/intel/torch-xpu-ops/issues/2769) | [oneDNN] New failed test cases with | LuFinch | mengfei25 | Need reproduce steps (Only for bugs or performance issue) | P2 | Missing reproduce steps for bug/performance issue | Others | DNNL/OneDNN Issue - failed test cases related to oneDNN with 3.11 compared to 3. |  | aten_ops | ut |
| [2777](https://github.com/intel/torch-xpu-ops/issues/2777) | [Bug Skip]: Random failures 2026WW0 | AKloniecki | CuiYifeng | Need reproduce steps (Only for bugs or performance issue) | P2 | Missing reproduce steps for bug/performance issue | Sparse Operations Related | Skip/No Test Exists - test is marked as a skip with no detailed error traceback  |  | unknown | ut |
| [2779](https://github.com/intel/torch-xpu-ops/issues/2779) | Accuracy failures in logspace op | PawelSwider2000 | PawelSwider2000 | Need reproduce steps (Only for bugs or performance issue) | P2 | Missing reproduce steps for bug/performance issue | Distributed | Dtype/Precision Issue - accuracy failures in logspace operation |  | aten_ops | ut |
| [2783](https://github.com/intel/torch-xpu-ops/issues/2783) | [Bug Skip]: Key "xpu" is missing fr | daisyden | CuiYifeng | Need reproduce steps (Only for bugs or performance issue) | P2 | Missing reproduce steps for bug/performance issue | Dtype / Precision Related | Skip/No Test Exists - Key "xpu" is missing, indicating the test is skipped or no |  | aten_ops | ut |
| [2795](https://github.com/intel/torch-xpu-ops/issues/2795) | Histc raises error with integer inp | CuiYifeng | YangKai0616 | Need reproduce steps (Only for bugs or performance issue) | P2 | Missing reproduce steps for bug/performance issue | Others | Dtype/Precision Issue - integer input causes error with deterministic algorithm  |  | aten_ops | ut |
| [2797](https://github.com/intel/torch-xpu-ops/issues/2797) | Copy error is not raise on test_dlp | None | shangerxin | Need reproduce steps (Only for bugs or performance issue) | P2 | Missing reproduce steps for bug/performance issue | Others | Others - Copy error not raised in test_dlpack.py test case |  | aten_ops | ut |
| [2801](https://github.com/intel/torch-xpu-ops/issues/2801) | to_dense() for Sparse CSR backend c | jenniew | jenniew | Need reproduce steps (Only for bugs or performance issue) | P2 | Missing reproduce steps for bug/performance issue | Sparse Operations Related | Error - source tensor shape mismatch during to_dense() for Sparse CSR backend |  | aten_ops | ut |
| [2811](https://github.com/intel/torch-xpu-ops/issues/2811) | [Bug Skip]: [Regression] failed cas | jmamzax | kaileiyx | Need reproduce steps (Only for bugs or performance issue) | P0 | Missing reproduce steps for bug/performance issue | Distributed | Failure - Expected and actual trace outputs do not match. |  | aten_ops | ut |
| [2815](https://github.com/intel/torch-xpu-ops/issues/2815) | RuntimeError: output with shape [2] | PawelSwider2000 | Silv3S | Need reproduce steps (Only for bugs or performance issue) | P2 | Missing reproduce steps for bug/performance issue | Others | Error - output shape mismatch during broadcasting |  | aten_ops | ut |
| [2823](https://github.com/intel/torch-xpu-ops/issues/2823) | [TorchAO][BMG] Llama-3.2-1B-Instruc | xiaowangintel, lchen2331 | LifengWang | Need reproduce steps (Only for bugs or performance issue) | P2 | Missing reproduce steps for bug/performance issue | TorchAO | Timeout/Performance Issue - 20% performance drop in next token generation with D |  | AO | ut |
| [2845](https://github.com/intel/torch-xpu-ops/issues/2845) | [Bug Skip]:[UT] [Windows] failed ca | None | kaileiyx | Need reproduce steps (Only for bugs or performance issue) | P2 | Missing reproduce steps for bug/performance issue | Others | Skip/No Test Exists - test was skipped or does not exist |  | aten_ops | ut |
| [2852](https://github.com/intel/torch-xpu-ops/issues/2852) | [Bug Skip]: New UT failures in 0206 | None | chuanqi129 | Need reproduce steps (Only for bugs or performance issue) | P2 | Missing reproduce steps for bug/performance issue | Others | Skip/No Test Exists - test was skipped or not present |  | aten_ops | ut |
| [2858](https://github.com/intel/torch-xpu-ops/issues/2858) | [Bug Skip]: test_xpu new failures | None | RUIJIEZHONG66166 | Need reproduce steps (Only for bugs or performance issue) | P2 | Missing reproduce steps for bug/performance issue | Others | Skip/No Test Exists - test_xpu new failures marked as skipped or no test availab |  | unknown | ut |
| [2862](https://github.com/intel/torch-xpu-ops/issues/2862) | accuracy issue with test_float8_sca | tszulist-hbn | daisyden | Need reproduce steps (Only for bugs or performance issue) | P2 | Missing reproduce steps for bug/performance issue | Dtype / Precision Related | Dtype/Precision Issue - accuracy issue with float8 operations |  | aten_ops | ut |
| [2871](https://github.com/intel/torch-xpu-ops/issues/2871) | [distributed][fsdp] test_fsdp_overl | songhappy | zxd1997066 | Need reproduce steps (Only for bugs or performance issue) | P2 | Missing reproduce steps for bug/performance issue | Distributed | Failure - test_fsdp_overlap.py assertion failed with "False is not true" |  | distributed | ut |
| [2873](https://github.com/intel/torch-xpu-ops/issues/2873) | [Bug Skip]: test_repos.py contains  | PawelSwider2000 | shangerxin | Need reproduce steps (Only for bugs or performance issue) | P2 | Missing reproduce steps for bug/performance issue | PT2E | Skip/No Test Exists - test contains failed ops and is skipped |  | aten_ops | ut |
| [2879](https://github.com/intel/torch-xpu-ops/issues/2879) | RuntimeError: _share_fd_: only avai | Silv3S | Silv3S | Need reproduce steps (Only for bugs or performance issue) | P2 | Missing reproduce steps for bug/performance issue | Others | Backend/Device Issue - _share_fd_ is not available on XPU device |  | aten_ops | ut |
| [2914](https://github.com/intel/torch-xpu-ops/issues/2914) | Test case test/test_autograd.py::Te | None | shangerxin | Need reproduce steps (Only for bugs or performance issue) | P2 | Missing reproduce steps for bug/performance issue | Dtype / Precision Related | Failure - Tensor-likes are not close! |  | aten_ops | ut |
| [2920](https://github.com/intel/torch-xpu-ops/issues/2920) | Failing Test Cases in Nightly Wheel | Silv3S | BBBela | Need reproduce steps (Only for bugs or performance issue) | P2 | Missing reproduce steps for bug/performance issue | Others | Others - insufficient information to determine root cause |  | aten_ops | ut |
| [2921](https://github.com/intel/torch-xpu-ops/issues/2921) | [abs][complex64] - new failing test | AKloniecki | BBBela | Need reproduce steps (Only for bugs or performance issue) | P2 | Missing reproduce steps for bug/performance issue | Distributed | Dtype/Precision Issue - test failure related to complex64 data type and abs oper |  | aten_ops | ut |
| [2942](https://github.com/intel/torch-xpu-ops/issues/2942) | [Windows] Unit tests got Fatal pyth | xuhancn, Stonepia | mengfei25 | Need reproduce steps (Only for bugs or performance issue) | P2 | Missing reproduce steps for bug/performance issue | Others | Backend/Device Issue - Fatal Python error on Windows unit tests related to XPU b |  | aten_ops | ut |
| [2946](https://github.com/intel/torch-xpu-ops/issues/2946) | [Bug Skip]: Random failures 2026WW0 | None | CuiYifeng | Need reproduce steps (Only for bugs or performance issue) | P2 | Missing reproduce steps for bug/performance issue | TorchAO | Skip/No Test Exists - test is marked to be skipped with no valid test implementa |  | unknown | ut |
| [2965](https://github.com/intel/torch-xpu-ops/issues/2965) | [Bug Skip]: Random failures 2026WW1 | None | CuiYifeng | Need reproduce steps (Only for bugs or performance issue) | P2 | Missing reproduce steps for bug/performance issue | Sparse Operations Related | Skip/No Test Exists - test is marked as a skip with no valid test implementation |  | unknown | ut |
| [2968](https://github.com/intel/torch-xpu-ops/issues/2968) | [distributed] timeout issue in test | frost-intel | zxd1997066 | Need reproduce steps (Only for bugs or performance issue) | P2 | Missing reproduce steps for bug/performance issue | Distributed | Timeout/Performance Issue - test experienced a timeout in distributed execution  |  | distributed | ut |
| [2969](https://github.com/intel/torch-xpu-ops/issues/2969) | [distributed] AssertionError: Scala | frost-intel | zxd1997066 | Need reproduce steps (Only for bugs or performance issue) | P2 | Missing reproduce steps for bug/performance issue | Distributed | Failure - Scalars are not equal in test case |  | distributed | ut |
| [2971](https://github.com/intel/torch-xpu-ops/issues/2971) | [distributed] KeyError in test/dist | newtdms, frost-intel | zxd1997066 | Need reproduce steps (Only for bugs or performance issue) | P2 | Missing reproduce steps for bug/performance issue | Distributed | Distributed/Gloo Issue - KeyError in distributed test related to c10d_xccl |  | distributed | ut |
| [2972](https://github.com/intel/torch-xpu-ops/issues/2972) | [distributed] AssertionError: Value | newtdms | zxd1997066 | Need reproduce steps (Only for bugs or performance issue) | P2 | Missing reproduce steps for bug/performance issue | Distributed | Distributed/Gloo Issue - AssertionError in test/distributed/test_c10d_xccl.py re |  | distributed | ut |
| [2993](https://github.com/intel/torch-xpu-ops/issues/2993) | [Bug Skip]: Unexpected success of t | gplutop7 | CuiYifeng | Need reproduce steps (Only for bugs or performance issue) | P2 | Missing reproduce steps for bug/performance issue | Others | Skip/No Test Exists - test unexpectedly succeeded and should have been skipped |  | aten_ops | ut |
| [3000](https://github.com/intel/torch-xpu-ops/issues/3000) | [Bug Skip]: RuntimeError: _share_fd | gplutop7 | zxd1997066 | Need reproduce steps (Only for bugs or performance issue) | P2 | Missing reproduce steps for bug/performance issue | Others | Backend/Device Issue - _share_fd_ is not available on XPU device |  | aten_ops | ut |
| [3010](https://github.com/intel/torch-xpu-ops/issues/3010) | [distributed][tensor] test_random_o | jenniew | zxd1997066 | Need reproduce steps (Only for bugs or performance issue) | P2 | Missing reproduce steps for bug/performance issue | Distributed | Inductor/Compilation Issue - TorchRuntimeError during fake tensor call in Dynamo |  | distributed | ut |
| [3025](https://github.com/intel/torch-xpu-ops/issues/3025) | New failing test in Nightly Wheel t | None | BBBela | Need reproduce steps (Only for bugs or performance issue) | P2 | Missing reproduce steps for bug/performance issue | Distributed | Failure - Expected and actual decomposition outputs do not match. |  | aten_ops | ut |
| [3030](https://github.com/intel/torch-xpu-ops/issues/3030) | [Bug Skip] test/test_modules.py::Te | gplutop7 | shangerxin | Need reproduce steps (Only for bugs or performance issue) | P2 | Missing reproduce steps for bug/performance issue | Others | Skip/No Test Exists - test was skipped due to failure with no detailed error pro |  | aten_ops | ut |
| [3032](https://github.com/intel/torch-xpu-ops/issues/3032) | [TorchAO][UT] failures in test/prot | Stonepia | zxd1997066 | Need reproduce steps (Only for bugs or performance issue) | P0 | Missing reproduce steps for bug/performance issue | TorchAO | Others - insufficient information to determine root cause |  | AO | build |
| [3033](https://github.com/intel/torch-xpu-ops/issues/3033) | [Bug Skip]: Softmax tolerance | chunhuanMeng | chunhuanMeng | Need reproduce steps (Only for bugs or performance issue) | P2 | Missing reproduce steps for bug/performance issue | Others | Skip/No Test Exists - test is skipped due to Softmax tolerance issue |  | aten_ops | ut |
| [3074](https://github.com/intel/torch-xpu-ops/issues/3074) | [Bug Skip] test_dlpack_exchange_api | AKloniecki | shangerxin | Need reproduce steps (Only for bugs or performance issue) | P2 | Missing reproduce steps for bug/performance issue | Others | Skip/No Test Exists - test is skipped expecting current_work_stream is not null |  | aten_ops | ut |
| [3076](https://github.com/intel/torch-xpu-ops/issues/3076) | [TorchAO][BMG] Llama-3.2-1B-Instruc | None | LifengWang | Need reproduce steps (Only for bugs or performance issue) | P2 | Missing reproduce steps for bug/performance issue | TorchAO | DNNL/OneDNN Issue - Performance drop with oneDNN 3.11.1 in Dynamic INT8 for Llam |  | aten_ops | ut |
| [3083](https://github.com/intel/torch-xpu-ops/issues/3083) | [Bug Skip]: Random failures 2026WW1 | None | CuiYifeng | Need reproduce steps (Only for bugs or performance issue) | P2 | Missing reproduce steps for bug/performance issue | Others | Skip/No Test Exists - test is marked to be skipped with no valid test to execute |  | unknown | ut |
| [3088](https://github.com/intel/torch-xpu-ops/issues/3088) | [TorchAO][BMG] INT4 RTN Flex-attent | Stonepia | LifengWang | Need reproduce steps (Only for bugs or performance issue) | P2 | Missing reproduce steps for bug/performance issue | TorchAO | Timeout/Performance Issue - INT4 RTN Flex-attention experienced a 5% performance |  | AO | ut |
| [3101](https://github.com/intel/torch-xpu-ops/issues/3101) | [distributed] 'torch._C._distribute | None | zxd1997066 | Need reproduce steps (Only for bugs or performance issue) | P2 | Missing reproduce steps for bug/performance issue | Distributed | Distributed/Gloo Issue - ProcessGroupXCCL object lacks '_set_default_timeout' at |  | distributed | ut |
| [3102](https://github.com/intel/torch-xpu-ops/issues/3102) | [distributed] RuntimeError: Invalid | None | zxd1997066 | Need reproduce steps (Only for bugs or performance issue) | P2 | Missing reproduce steps for bug/performance issue | Distributed | Backend/Device Issue - invalid device string 'xpu:foo' indicates a problem with  |  | distributed | ut |
| [3121](https://github.com/intel/torch-xpu-ops/issues/3121) | [Bug Skip]: CUDA specific UT test_f | None | CuiYifeng | Need reproduce steps (Only for bugs or performance issue) | P1 | Missing reproduce steps for bug/performance issue | Dtype / Precision Related | Skip/No Test Exists - test is skipped or not applicable for XPU backend |  | aten_ops | ut |
| [3124](https://github.com/intel/torch-xpu-ops/issues/3124) | [TorchAO][Bug] ImportError: Require | None | FRAMEEE17 | Need reproduce steps (Only for bugs or performance issue) | P2 | Missing reproduce steps for bug/performance issue | TorchAO | Backend/Device Issue - Requires mslk >= 1.0.0 for XPU support with int4 quantiza |  | aten_ops | ut |
| [3139](https://github.com/intel/torch-xpu-ops/issues/3139) | [distributed][_composable] Assertio | Kanya-Mo | zxd1997066 | Need reproduce steps (Only for bugs or performance issue) | P2 | Missing reproduce steps for bug/performance issue | Distributed | Failure - Expects xpu:0 but got xpu:1 |  | distributed | ut |
| [3149](https://github.com/intel/torch-xpu-ops/issues/3149) | New failure in test_rms_norm_decomp | None | BBBela | Need reproduce steps (Only for bugs or performance issue) | P2 | Missing reproduce steps for bug/performance issue | Flash Attention / Transformer Related | Failure - RuntimeError not raised as expected in test |  | aten_ops | ut |
| [3174](https://github.com/intel/torch-xpu-ops/issues/3174) | [Bug Skip]: Accuracy failure of tes | None | CuiYifeng | Need reproduce steps (Only for bugs or performance issue) | P2 | Missing reproduce steps for bug/performance issue | Distributed | Failure - test assertion failed for Conv2d groups output comparison |  | aten_ops | ut |
| [3178](https://github.com/intel/torch-xpu-ops/issues/3178) | New failed test cases 2026-03-25 | pponikox | BBBela | Need reproduce steps (Only for bugs or performance issue) | P2 | Missing reproduce steps for bug/performance issue | Flash Attention / Transformer Related | Backend/Device Issue - XPU-specific test failure in vmap with SDPA operation |  | aten_ops | ut |
| [3195](https://github.com/intel/torch-xpu-ops/issues/3195) | test_sdpa_unbacked_no_dde_xpu crash | None | daisyden | Need reproduce steps (Only for bugs or performance issue) | P0 | Missing reproduce steps for bug/performance issue | Flash Attention / Transformer Related | Backend/Device Issue - test crashed on XPU backend execution |  | aten_ops | ut |
| [3206](https://github.com/intel/torch-xpu-ops/issues/3206) | [Bug Skip]: [new failures]RuntimeEr | tszulist-hbn | kaileiyx | Need reproduce steps (Only for bugs or performance issue) | P2 | Missing reproduce steps for bug/performance issue | Others | Backend/Device Issue - Tensors are on different devices (xpu:0 vs cpu) |  | aten_ops | ut |
| [3209](https://github.com/intel/torch-xpu-ops/issues/3209) | [Win][Build] There is Cyclic depend | Copilot | NeoZhangJianyu | Need reproduce steps (Only for bugs or performance issue) | P0 | Missing reproduce steps for bug/performance issue | Others | Backend/Device Issue - Cyclic dependencies during build with BUILD_SEPARATE_OPS= |  | aten_ops | build |
| [3224](https://github.com/intel/torch-xpu-ops/issues/3224) | [Win][Build] Building SYCL (Device) | chunhuanMeng | anmyachev | Need reproduce steps (Only for bugs or performance issue) | P0 | Missing reproduce steps for bug/performance issue | Inductor / Compilation Related | Backend/Device Issue - SYCL kernel build failure on Windows for XPU |  | aten_ops | build |
| [3227](https://github.com/intel/torch-xpu-ops/issues/3227) | torch xpu event has ~0.1ms latency, | guangyey | jianyizh | Need reproduce steps (Only for bugs or performance issue) | P2 | Missing reproduce steps for bug/performance issue | Others | Timeout/Performance Issue - high latency in XPU event (~0.1ms) indicates a perfo |  | aten_ops | ut |
| [3231](https://github.com/intel/torch-xpu-ops/issues/3231) | Dynamo failed to run FX node with f | None | daisyden | Need reproduce steps (Only for bugs or performance issue) | P2 | Missing reproduce steps for bug/performance issue | PT2E | Inductor/Compilation Issue - Dynamo failed to compile FX node with fake tensors  |  | aten_ops | ut |
| [3232](https://github.com/intel/torch-xpu-ops/issues/3232) | [distributed][tensor] AssertionErro | None | zxd1997066 | Need reproduce steps (Only for bugs or performance issue) | P2 | Missing reproduce steps for bug/performance issue | Distributed | Failure - AssertionError not raised for Placement (Shard(dim=2),) in test |  | distributed | ut |
| [3233](https://github.com/intel/torch-xpu-ops/issues/3233) | [distributed] RuntimeError: No back | None | zxd1997066 | Need reproduce steps (Only for bugs or performance issue) | P2 | Missing reproduce steps for bug/performance issue | Distributed | Distributed/Gloo Issue - No backend for the parent process group or its backend  |  | distributed | ut |
| [3236](https://github.com/intel/torch-xpu-ops/issues/3236) | AssertionError: 'def [28 chars]n un | jmamzax | jmamzax | Need reproduce steps (Only for bugs or performance issue) | P2 | Missing reproduce steps for bug/performance issue | Dtype / Precision Related | Failure - mismatch in expected IR code for XPU backend operations |  | aten_ops | ut |
| [3242](https://github.com/intel/torch-xpu-ops/issues/3242) | AssertionError: Torch not compiled  | None | zxd1997066 | Need reproduce steps (Only for bugs or performance issue) | P2 | Missing reproduce steps for bug/performance issue | Inductor / Compilation Related |  |  | aten_ops | ut |
| [3243](https://github.com/intel/torch-xpu-ops/issues/3243) | AssertionError: False is not true | pponikox | zxd1997066 | Need reproduce steps (Only for bugs or performance issue) | P2 | Missing reproduce steps for bug/performance issue | Dtype / Precision Related | Failure - assertion 'False is not true' failed in test |  | aten_ops | ut |
| [3258](https://github.com/intel/torch-xpu-ops/issues/3258) | Error in op: torch.ops.aten._scaled | None | bjarzemb | Need reproduce steps (Only for bugs or performance issue) | P2 | Missing reproduce steps for bug/performance issue | Flash Attention / Transformer Related | 5 - Flash Attention/Specific Ops Issue - Error in _scaled_dot_product_fused_atte |  | aten_ops | ut |
| [3259](https://github.com/intel/torch-xpu-ops/issues/3259) | New failed test cases 2026-04-02 | SlawomirLaba | Silv3S | Need reproduce steps (Only for bugs or performance issue) | P2 | Missing reproduce steps for bug/performance issue | Flash Attention / Transformer Related | Backend/Device Issue - XPU device initialization or compatibility failure |  | aten_ops | ut |
| [1059](https://github.com/intel/torch-xpu-ops/issues/1059) | SYCL RT: Using recommended shortcut | CuiYifeng, jianyizh | fengyuan14 | error logs and reproduction steps | P2 |  | Distributed | Backend/Device Issue - related to SYCL RT and kernel-specific max work group siz |  | aten_ops | ut |
| [1165](https://github.com/intel/torch-xpu-ops/issues/1165) | [CI] Add a test of PyTorch XPU with | RUIJIEZHONG66166 | dvrogozh | error logs and reproduction steps | P0 |  | Flash Attention / Transformer Related | Skip/No Test Exists - No test was implemented or executed. |  | aten_ops | build |
| [1510](https://github.com/intel/torch-xpu-ops/issues/1510) | Some test cases will be hang on BMG | Stonepia, mengfei25 | mengfei25 | error logs and reproduction steps | P2 |  | Others | Backend/Device Issue - test cases hang on BMG Ubuntu related to XPU backend |  | aten_ops | ut |
| [1587](https://github.com/intel/torch-xpu-ops/issues/1587) | Keep track on the latest CUDA op im | CuiYifeng, yucai-intel | toyxu | error logs and reproduction steps | P2 |  | Others | Skip/No Test Exists - no test or error traceback provided |  | aten_ops | ut |
| [1594](https://github.com/intel/torch-xpu-ops/issues/1594) | Keep track on the building warning | CuiYifeng, chunhuanMeng | toyxu | error logs and reproduction steps | P0 |  | Others | Others - building warning tracking issue |  | aten_ops | ut |
| [1645](https://github.com/intel/torch-xpu-ops/issues/1645) | [For Comparison] Save reference com | mengfei25 | mengfei25 | error logs and reproduction steps | P2 |  | Others | Skip/No Test Exists - no test or error information provided |  | inductor | ut |
| [1678](https://github.com/intel/torch-xpu-ops/issues/1678) | missing op support for `model.share | None | jafraustro | error logs and reproduction steps | P0 |  | Others | Memory/Shared Memory Issue - missing op support for `model.share_memory()` indic |  | aten_ops | ut |
| [1689](https://github.com/intel/torch-xpu-ops/issues/1689) | [For op Perf Comparison] Save refer | None | RUIJIEZHONG66166 | error logs and reproduction steps | P2 |  | Others | Skip/No Test Exists - no test or error information provided |  | aten_ops | ut |
| [1722](https://github.com/intel/torch-xpu-ops/issues/1722) | Ask an API to query GPU type(iGPU/d | guangyey | xuhancn | error logs and reproduction steps | P2 |  | Others | Mismatch - Request for an API to query GPU type (iGPU/dGPU) is missing or not properly implemented. |  | aten_ops | ut |
| [1729](https://github.com/intel/torch-xpu-ops/issues/1729) | Validation Check List | chuanqi129 | EikanWang | error logs and reproduction steps | P2 |  | Others | Skip/No Test Exists - no test or error information provided |  | aten_ops | ut |
| [1745](https://github.com/intel/torch-xpu-ops/issues/1745) | torch.xpu.empty_cache() introduces  | guangyey | songhappy | error logs and reproduction steps | P2 |  | Others | Memory/Shared Memory Issue - torch.xpu.empty_cache() introduces memory leak |  | aten_ops | ut |
| [1762](https://github.com/intel/torch-xpu-ops/issues/1762) | Add an ocloc AOT target compilation | chunhuanMeng | jingxu10 | error logs and reproduction steps | P2 |  | PT2E | compilation-related task or issue. |  | aten_ops | ut |
| [1764](https://github.com/intel/torch-xpu-ops/issues/1764) | New kernels for concat | yucai-intel | jianyizh | error logs and reproduction steps | P2 |  | Inductor / Compilation Related | Others - New kernels for concat, no specific error provided. |  | aten_ops | ut |
| [1856](https://github.com/intel/torch-xpu-ops/issues/1856) | channel last aten::hardswish_ will  | chunhuanMeng | jianyizh | error logs and reproduction steps | P2 |  | Others | Memory/Shared Memory Issue - extra copy caused by channel last aten::hardswish_ |  | aten_ops | ut |
| [1962](https://github.com/intel/torch-xpu-ops/issues/1962) | [upstream_ut] segfault with test_fa | jenniew, mengfei25 | daisyden | error logs and reproduction steps | P0 |  | Sparse Operations Related | Backend/Device Issue - segfault related to XPU device operation in test |  | aten_ops | ut |
| [1986](https://github.com/intel/torch-xpu-ops/issues/1986) | torch.xpu._sleep is missing, | guangyey | githubsgi | error logs and reproduction steps | P2 |  | Others | Mismatch - torch.xpu._sleep is not implemented or available in the current setup. |  | aten_ops | ut |
| [1996](https://github.com/intel/torch-xpu-ops/issues/1996) | [TorchAO]  Memory Efficient Optimiz | arlesniak | liangan1 | error logs and reproduction steps | P2 |  | TorchAO | Memory/Shared Memory Issue - related to memory efficient optimizers in TorchAO |  | AO | ut |
| [2113](https://github.com/intel/torch-xpu-ops/issues/2113) | Update example for Distributed Data | songhappy | luoyu-intel | error logs and reproduction steps | P2 |  | Distributed | Distributed/Gloo Issue - related to Distributed Data Parallel update example |  | distributed | ut |
| [2142](https://github.com/intel/torch-xpu-ops/issues/2142) | XPU max_memory_allocated have diffe | guangyey | jiqing-feng | error logs and reproduction steps | P2 |  | Others | Memory/Shared Memory Issue - XPU and CUDA show different memory allocation outpu |  | aten_ops | ut |
| [2163](https://github.com/intel/torch-xpu-ops/issues/2163) | 3 distributed UT cases need to be s | githubsgi | libohao1201 | error logs and reproduction steps | P2 |  | Distributed | Distributed/Gloo Issue - distributed UT cases need support from sac_estimator.py |  | distributed | ut |
| [2195](https://github.com/intel/torch-xpu-ops/issues/2195) | Tools in pti should get 100% functi | aostrowski-hbn | jianyizh | error logs and reproduction steps | P2 |  | Others | Backend/Device Issue - functionality not working on BMG for PyTorch profiling |  | profiling | ut |
| [2200](https://github.com/intel/torch-xpu-ops/issues/2200) | support flash attention op on XPU d | ElaineBao | Zjq9409 | error logs and reproduction steps | P2 |  | Flash Attention / Transformer Related | Flash Attention/Specific Ops Issue - request to support flash attention op on XP |  | aten_ops | ut |
| [2215](https://github.com/intel/torch-xpu-ops/issues/2215) | Find use case example for torch-xpu | dvrogozh | dvrogozh | error logs and reproduction steps | P2 |  | Dtype / Precision Related | Others - Looking for use case example of torch-xpu-ops.lib in SYCL C++ extension |  | aten_ops | ut |
| [2229](https://github.com/intel/torch-xpu-ops/issues/2229) | test/test_sparse_csr.py::TestSparse | jenniew | wincent8 | error logs and reproduction steps | P2 |  | Distributed | Backend/Device Issue - device mismatch between crow_indices and col_indices on X |  | aten_ops | ut |
| [2232](https://github.com/intel/torch-xpu-ops/issues/2232) | sdpa backward kernel is required to | None | xin3he | error logs and reproduction steps | P2 |  | Flash Attention / Transformer Related | Memory/Shared Memory Issue - sdpa backward kernel needed to reduce memory usage |  | aten_ops | ut |
| [2250](https://github.com/intel/torch-xpu-ops/issues/2250) | Found mismatch when comparing the o | astachowiczhabana | daisyden | error logs and reproduction steps | P2 |  | Flash Attention / Transformer Related | Mismatch - Mismatch in output of aten.view.default between FakeTensor and concrete Tensors. |  | aten_ops | ut |
| [2261](https://github.com/intel/torch-xpu-ops/issues/2261) | [xpu][profiler] Run with fork proce | moksiuc | chuanqi129 | error logs and reproduction steps | P2 |  | Others | Backend/Device Issue - XPU profiler warning during fork process execution |  | profiling | ut |
| [2323](https://github.com/intel/torch-xpu-ops/issues/2323) | [TorchAO] MOE training enabling on  | riverliuintel | liangan1 | error logs and reproduction steps | P2 |  | TorchAO | Backend/Device Issue - MOE training not enabled on XPU |  | AO | ut |
| [2324](https://github.com/intel/torch-xpu-ops/issues/2324) | [TorchAO] FP8 conv support | Stonepia | liangan1 | error logs and reproduction steps | P2 |  | TorchAO | Supported - FP8 conv is not supported yet in TorchAO |  | AO | ut |
| [2325](https://github.com/intel/torch-xpu-ops/issues/2325) | [TorchAO] Float8 training support o | arlesniak, riverliuintel | liangan1 | error logs and reproduction steps | P2 |  | TorchAO | Supported - Float8 training is not supported on XPU. |  | AO | ut |
| [2326](https://github.com/intel/torch-xpu-ops/issues/2326) | [TorchAO] MX training  native PyTor | riverliuintel | liangan1 | error logs and reproduction steps | P2 |  | TorchAO | Backend/Device Issue - Training on XPU with TorchAO and native PyTorch is not fu |  | AO | ut |
| [2327](https://github.com/intel/torch-xpu-ops/issues/2327) | [TorchAO] benchmark enabling on XPU | None | liangan1 | error logs and reproduction steps | P2 |  | TorchAO | Backend/Device Issue - XPU benchmark enabling issue in TorchAO |  | aten_ops | ut |
| [2333](https://github.com/intel/torch-xpu-ops/issues/2333) | [Don't merge] Collect the new passe | None | mengfei25 | error logs and reproduction steps | P2 |  | Others | Skip/No Test Exists - test is empty or not applicable |  | aten_ops | ut |
| [2349](https://github.com/intel/torch-xpu-ops/issues/2349) | [oneAPI][backward compatibility] li | riverliuintel | dvrogozh | error logs and reproduction steps | P2 |  | Others | Backend/Device Issue - missing library version for XPU backend compatibility |  | aten_ops | ut |
| [2390](https://github.com/intel/torch-xpu-ops/issues/2390) | SDPA in pytorch use different backe | LuFinch | jiqing-feng | error logs and reproduction steps | P2 |  | Flash Attention / Transformer Related | Backend/Device Issue - SDPA uses different backend compared with IPEX on XPU |  | aten_ops | ut |
| [2400](https://github.com/intel/torch-xpu-ops/issues/2400) | [ut_upstream] tf32_on_and_off() nee | chunhuanMeng | daisyden | error logs and reproduction steps | P2 |  | Others | Backend/Device Issue - XPU support required for tf32_on_and_off() test |  | aten_ops | ut |
| [2412](https://github.com/intel/torch-xpu-ops/issues/2412) | Some NestedTensor missing XPU suppo | yucai-intel | daisyden | error logs and reproduction steps | P2 |  | Others | Backend/Device Issue - XPU support missing for NestedTensor operations |  | aten_ops | ut |
| [2465](https://github.com/intel/torch-xpu-ops/issues/2465) | [windows] ut hang | tadkrawiec | bjarzemb | error logs and reproduction steps | P2 |  | Others | Timeout/Performance Issue - UT hang indicates a timeout or performance bottlenec |  | aten_ops | ut |
| [2467](https://github.com/intel/torch-xpu-ops/issues/2467) | Host may stuck when submit too many | jianyizh | jianyizh | error logs and reproduction steps | P2 |  | Inductor / Compilation Related | Backend/Device Issue - Host stuck due to excessive kernel submissions on XPU |  | aten_ops | ut |
| [2469](https://github.com/intel/torch-xpu-ops/issues/2469) | test_matmul_cuda.py gaps | CuiYifeng, guangyey | daisyden | error logs and reproduction steps | P2 |  | Others | Skip/No Test Exists - test_matmul_cuda.py gaps indicate missing or skipped tests |  | aten_ops | ut |
| [2471](https://github.com/intel/torch-xpu-ops/issues/2471) | test_cuda.py gaps | guangyey | daisyden | error logs and reproduction steps | P2 |  | Others | Skip/No Test Exists - test_cuda.py gaps indicate missing or skipped tests. |  | aten_ops | ut |
| [2605](https://github.com/intel/torch-xpu-ops/issues/2605) | [int4][inductor] Add freezing patte | None | liangan1 | error logs and reproduction steps | P2 |  | TorchAO | Inductor/Compilation Issue - Adding freezing pattern for fusing int4 mm kernel i |  | aten_ops | ut |
| [2640](https://github.com/intel/torch-xpu-ops/issues/2640) | random issue test_vjpvjp_index_redu | wpietka | daisyden | error logs and reproduction steps | P2 |  | Distributed | Others - incomplete traceback and insufficient information to determine root cause |  | aten_ops | ut |
| [2649](https://github.com/intel/torch-xpu-ops/issues/2649) | [distributed][pipelining] test_sche | syedshahbaaz | zxd1997066 | error logs and reproduction steps | P2 |  | Distributed | Distributed/Gloo Issue - test_schedule_multiproc.py is hanging during distribute |  | distributed | ut |
| [2700](https://github.com/intel/torch-xpu-ops/issues/2700) | [distributed] Hang issues with test | syedshahbaaz | madhumitha0102 | error logs and reproduction steps | P2 |  | Distributed | Distributed/Gloo Issue - Hang issues with test_distributed_spawn.py suggest a pr |  | distributed | ut |
| [2816](https://github.com/intel/torch-xpu-ops/issues/2816) | torch.logcumsumexp incorrectly retu | Silv3S | Silv3S | error logs and reproduction steps | P2 |  | Dtype / Precision Related | Dtype/Precision Issue - torch.logcumsumexp returns NaNs for complex64 input |  | aten_ops | ut |
| [2853](https://github.com/intel/torch-xpu-ops/issues/2853) | [upstream_ut] torch.ops.aten._flash | LuFinch | BBBela | error logs and reproduction steps | P2 |  | Distributed | Flash Attention/Specific Ops Issue - _flash_attention_forward not supported on X |  | aten_ops | ut |
| [2899](https://github.com/intel/torch-xpu-ops/issues/2899) | Update nan_to_num XPU stub to use s | None | cleonard530 | error logs and reproduction steps | P2 |  | Dtype / Precision Related | Backend/Device Issue - XPU stub uses incorrect type for nan_to_num parameters |  | aten_ops | ut |
| [2948](https://github.com/intel/torch-xpu-ops/issues/2948) | [AO] Benchmark enabling on XPU | None | liangan1 | error logs and reproduction steps | P2 |  | Others | Backend/Device Issue - XPU benchmark enabling issue |  | AO | ut |
| [2950](https://github.com/intel/torch-xpu-ops/issues/2950) | SYCL compilation flag -fsycl-id-que | BBBela | BBBela | error logs and reproduction steps | P2 |  | Inductor / Compilation Related | Inductor/Compilation Issue - SYCL compilation flag not working as expected for T |  | aten_ops | ut |
| [3021](https://github.com/intel/torch-xpu-ops/issues/3021) | [distributed] all_to_all_single Com | zhangxiaoli73 | xiangyuT | error logs and reproduction steps | P2 |  | Distributed | Distributed/Gloo Issue - all_to_all_single compatibility issue on B60/XCCL |  | distributed | ut |
| [3022](https://github.com/intel/torch-xpu-ops/issues/3022) | [distributed] batch_isend_irecv Com | zhangxiaoli73 | xiangyuT | error logs and reproduction steps | P2 |  | Distributed | Distributed/Gloo Issue - batch_isend_irecv compatibility issue on B60/XCCL |  | distributed | ut |
| [3024](https://github.com/intel/torch-xpu-ops/issues/3024) | Enable clang-tidy checks | Silv3S | Silv3S | error logs and reproduction steps | P2 |  | Others | Others - clang-tidy checks enablement is a code quality/linter configuration task, not a runtime or functional issue. |  | aten_ops | ut |
| [3048](https://github.com/intel/torch-xpu-ops/issues/3048) | Profiler result is not correct on B | aostrowski-hbn | jianyizh | error logs and reproduction steps | P2 |  | Others | Backend/Device Issue - Profiler result discrepancy on B70 device. |  | profiling | ut |
| [3081](https://github.com/intel/torch-xpu-ops/issues/3081) | Sparse CSR gemm-like ops have not b | None | daisyden | error logs and reproduction steps | P2 |  | Sparse Operations Related | Supported - Sparse CSR gemm-like operations are not supported yet. |  | aten_ops | ut |
| [3082](https://github.com/intel/torch-xpu-ops/issues/3082) | multithread support in distributed | None | daisyden | error logs and reproduction steps | P2 |  | Distributed | Distributed/Gloo Issue - multithread support in distributed operations is affect |  | distributed | ut |
| [3084](https://github.com/intel/torch-xpu-ops/issues/3084) | torch.library.register_autocast doe | None | daisyden | error logs and reproduction steps | P2 |  | Others | Backend/Device Issue - torch.library.register_autocast does not support XPU devi |  | aten_ops | ut |
| [3086](https://github.com/intel/torch-xpu-ops/issues/3086) | nvml support blocks some test cases | None | daisyden | error logs and reproduction steps | P2 |  | Others | Backend/Device Issue - nvml support blocking test cases on XPU |  | aten_ops | ut |
| [3093](https://github.com/intel/torch-xpu-ops/issues/3093) | XPU does not support NestedTensor f | None | daisyden | error logs and reproduction steps | P2 |  | Flash Attention / Transformer Related | Supported - XPU does not support NestedTensor for SDPA operations. |  | aten_ops | ut |
| [3096](https://github.com/intel/torch-xpu-ops/issues/3096) | VISIBLE_DEVICE support | None | daisyden | error logs and reproduction steps | P2 |  | Others | Backend/Device Issue - VISIBLE_DEVICE support is related to device visibility an |  | aten_ops | ut |
| [3100](https://github.com/intel/torch-xpu-ops/issues/3100) | [distributed] /handler/dump_nccl_tr | None | zxd1997066 | error logs and reproduction steps | P2 |  | Distributed | Distributed/Gloo Issue - related to NCCL logging and trace handling in distribut |  | distributed | ut |
| [3103](https://github.com/intel/torch-xpu-ops/issues/3103) | Tensor-likes are not equal for test | BBBela | BBBela | error logs and reproduction steps | P2 |  | Dtype / Precision Related | Backend/Device Issue - XPU tensor-like comparison failure in test |  | aten_ops | ut |
| [3180](https://github.com/intel/torch-xpu-ops/issues/3180) | [E2E] Timm/Torchbench models got "e | None | libohao1201 | error logs and reproduction steps | P0 |  | Others | Backend/Device Issue - eager_two_runs_differ on ARC XPU backend |  | aten_ops | ut |
| [3189](https://github.com/intel/torch-xpu-ops/issues/3189) | Task Tracker | guangyey | guangyey | error logs and reproduction steps | P2 |  | Others | Skip/No Test Exists - no test or error details provided |  | aten_ops | ut |
| [3194](https://github.com/intel/torch-xpu-ops/issues/3194) | Incorrect strides in TestCommonXPU, | AKloniecki | AKloniecki | error logs and reproduction steps | P2 |  | Distributed | Backend/Device Issue - Incorrect strides related to XPU device handling |  | aten_ops | ut |
| [3196](https://github.com/intel/torch-xpu-ops/issues/3196) | vitals is not supported, the cases  | libohao1201 | daisyden | error logs and reproduction steps | P2 |  | Others | 10 - vitals feature is not supported, cases should be disabled |  | aten_ops | ut |
| [3216](https://github.com/intel/torch-xpu-ops/issues/3216) | [OPs] Some ops of XPU have non-dete | CuiYifeng | YangKai0616 | error logs and reproduction steps | P2 |  | Others | Backend/Device Issue - XPU ops show non-determinism and inconsistency compared t |  | aten_ops | ut |

### 1.1 Close fixed issue

| ID | Title | Owner | Owner Transferred | TBD | Priority | Action Reason | Category | Root Cause | PR | Module | Test Module |
|---|-------|-------|-------------------|-----|---------|--------|----------|-----------|-----|--------|-------------|
| [1624](https://github.com/intel/torch-xpu-ops/issues/1624) | [DONT CLOSE] Known UT Issue list | None | RUIJIEZHONG66166 | Close fixed issue | P2 | All test cases now passing | Dtype / Precision Related |  |  | distributed | ut |
| [2496](https://github.com/intel/torch-xpu-ops/issues/2496) | [upstream_ut]  Segmentation fault w | astachowiczhabana | libohao1201 | Close fixed issue | P0 | All test cases now passing | Dtype / Precision Related |  |  | aten_ops | ut |
| [2518](https://github.com/intel/torch-xpu-ops/issues/2518) | [upstream_ut]  TypeError: Creating  | astachowiczhabana | libohao1201 | Close fixed issue | P2 | All test cases now passing | Others |  |  | aten_ops | ut |
| [2592](https://github.com/intel/torch-xpu-ops/issues/2592) | [release/2.10] models got fail_accu | mengfei25 | mengfei25 | Close fixed issue | P0 | All test cases now passing | Dtype / Precision Related |  |  | aten_ops | e2e |
| [2619](https://github.com/intel/torch-xpu-ops/issues/2619) | [release/2.10] Some models inductor | jianyizh, weishi-deng | mengfei25 | Close fixed issue | P0 | All test cases now passing | Inductor / Compilation Related |  |  | aten_ops | e2e |
| [2953](https://github.com/intel/torch-xpu-ops/issues/2953) | [release/2.11][wsl] huggingface TrO | xuhancn | bjarzemb | Close fixed issue | P0 | All test cases now passing | Others |  |  | aten_ops | e2e |
| [2981](https://github.com/intel/torch-xpu-ops/issues/2981) | [release/2.11] T5 models performanc | jianyizh, weishi-deng | mengfei25 | Close fixed issue | P0 | All test cases now passing | Others |  |  | aten_ops | e2e |
| [3011](https://github.com/intel/torch-xpu-ops/issues/3011) | [upstream_ut] torch.OutOfMemoryErro | None | Silv3S | Close fixed issue | P2 | All test cases now passing | Others |  |  | aten_ops | ut |
| [3013](https://github.com/intel/torch-xpu-ops/issues/3013) | [upstream_ut] RuntimeError: Kernel  | None | Silv3S | Close fixed issue | P2 | All test cases now passing | Inductor / Compilation Related |  |  | unknown | ut |
| [3014](https://github.com/intel/torch-xpu-ops/issues/3014) | [upstream_ut] AssertionError: False | None | Silv3S | Close fixed issue | P2 | All test cases now passing | Dtype / Precision Related |  |  | unknown | ut |
| [3058](https://github.com/intel/torch-xpu-ops/issues/3058) | [E2E] hf_GPT2_large amp_fp16/amp_bf | weishi-deng | kaileiyx | Close fixed issue | P1 | All test cases now passing | Flash Attention / Transformer Related |  |  | aten_ops | e2e |
| [3106](https://github.com/intel/torch-xpu-ops/issues/3106) | Worker crashes when running TestDec | BBBela | BBBela | Close fixed issue | P0 | All test cases now passing | Others |  |  | aten_ops | ut |
| [3157](https://github.com/intel/torch-xpu-ops/issues/3157) | XPU out of memory. Tried to allocat | kdrozd-dev | kdrozd-dev | Close fixed issue | P2 | All test cases now passing | Others |  |  | aten_ops | ut |
| [3158](https://github.com/intel/torch-xpu-ops/issues/3158) | AttributeError: module 'triton.comp | tadkrawiec | kdrozd-dev | Close fixed issue | P2 | All test cases now passing | Inductor / Compilation Related |  |  | aten_ops | ut |
| [3161](https://github.com/intel/torch-xpu-ops/issues/3161) | Exception: Tensor-likes are not clo | tadkrawiec | kdrozd-dev | Close fixed issue | P2 | All test cases now passing | Dtype / Precision Related |  |  | aten_ops | ut |
| [3166](https://github.com/intel/torch-xpu-ops/issues/3166) | test_consistency_SparseCSR failures | yucai-intel | CuiYifeng | Close fixed issue | P2 | All test cases now passing | TorchAO |  |  | aten_ops | ut |

### 1.2 Needs PyTorch Repo Changes (upstream)

| ID | Title | Owner | Owner Transferred | TBD | Priority | Action Reason | Category | Root Cause | PR | Module | Test Module |
|---|-------|-------|-------------------|-----|---------|--------|----------|-----------|-----|--------|-------------|
| [489](https://github.com/intel/torch-xpu-ops/issues/489) | Moco NotImplementedError: xpu not s | weishi-deng | weishi-deng | Needs PyTorch Repo Changes (upstream) | P2 | Requires upstream fix in PyTorch | Others | Backend/Device Issue - xpu not supported |  | aten_ops | e2e |
| [492](https://github.com/intel/torch-xpu-ops/issues/492) | Timm_efficientdet NotImplementedErr | weishi-deng | weishi-deng | Needs PyTorch Repo Changes (upstream) | P0 | Requires upstream fix in PyTorch | Others | Backend/Device Issue - model code forces use of CUDA instead of XPU |  | aten_ops | e2e |
| [1159](https://github.com/intel/torch-xpu-ops/issues/1159) | [LNL Windows][Test by CD Nightly Wh | Stonepia | Stonepia | Needs PyTorch Repo Changes (upstream) | P0 | Requires upstream fix in PyTorch | Flash Attention / Transformer Related | Dtype/Precision Issue - value cannot be converted to at::BFloat16 without overfl |  | aten_ops | e2e |
| [1505](https://github.com/intel/torch-xpu-ops/issues/1505) | [ARC-WSL-Ubuntu24.04] 15 Timm model | xuhancn, Stonepia | xuhancn, Stonepia | Needs PyTorch Repo Changes (upstream) | P0 | Requires upstream fix in PyTorch | Dtype / Precision Related |  |  | inductor | e2e |
| [1519](https://github.com/intel/torch-xpu-ops/issues/1519) | [PVC][PT2.7][ABI=1][Torch-xpu-ops U | None | None | Needs PyTorch Repo Changes (upstream) | P2 | Requires upstream fix in PyTorch | Others | Backend/Device Issue - coredump related to XPU operations in Torch-xpu-ops UT |  | aten_ops | ut |
| [1778](https://github.com/intel/torch-xpu-ops/issues/1778) | [Infra] Show known issues for accur | mengfei25 | mengfei25 | Needs PyTorch Repo Changes (upstream) | P1 | Requires upstream fix in PyTorch | Dtype / Precision Related | Skip/No Test Exists - no test or error details provided |  | unknown | e2e |
| [1818](https://github.com/intel/torch-xpu-ops/issues/1818) | [BMG-Windows][PT2.8]Torch-xpu-ops U | kdrozd-dev | kdrozd-dev | Needs PyTorch Repo Changes (upstream) | P2 | Requires upstream fix in PyTorch | Dtype / Precision Related | Backend/Device Issue - Accuracy issue on XPU for Torch-xpu-ops UT |  | aten_ops | ut |
| [1866](https://github.com/intel/torch-xpu-ops/issues/1866) | [release 2.8]Torchbench vision_mask | BartoszKokoszko | BartoszKokoszko | Needs PyTorch Repo Changes (upstream) | P0 | Requires upstream fix in PyTorch | Dtype / Precision Related | Dtype/Precision Issue - amp_fp16 inference accuracy failure |  | aten_ops | e2e |
| [1894](https://github.com/intel/torch-xpu-ops/issues/1894) | [Linux][PT2E] performance test got  | jenniew | jenniew | Needs PyTorch Repo Changes (upstream) | P1 | Requires upstream fix in PyTorch | TorchAO | precision-related failure in performance test |  | low_precision | e2e |
| [1963](https://github.com/intel/torch-xpu-ops/issues/1963) | [upstream_ut] MetadataMismatchError | pbielak | pbielak | Needs PyTorch Repo Changes (upstream) | P2 | Requires upstream fix in PyTorch | Others |  | [PR](https://github.com/pytorch/pytorch/pull/178277, https://github.com/pytorch/pytorch/pull/175965) | aten_ops | ut |
| [2055](https://github.com/intel/torch-xpu-ops/issues/2055) | New huggingface LLM models issues | jianyizh, mengfei25 | jianyizh, mengfei25 | Needs PyTorch Repo Changes (upstream) | P0 | Requires upstream fix in PyTorch | Others | Others - insufficient information to determine root cause |  | aten_ops | e2e |
| [2058](https://github.com/intel/torch-xpu-ops/issues/2058) | [release/2.9] llama_v2_7b_16h amp i | jianyizh | jianyizh | Needs PyTorch Repo Changes (upstream) | P0 | Requires upstream fix in PyTorch | Flash Attention / Transformer Related | device-specific backend problem. |  | aten_ops | e2e |
| [2128](https://github.com/intel/torch-xpu-ops/issues/2128) | [2.9][BMG-Windows][Torchbench] spee | chuanqi129 | chuanqi129 | Needs PyTorch Repo Changes (upstream) | P0 | Requires upstream fix in PyTorch | Dtype / Precision Related | Backend/Device Issue - Exception Code 0xC0000005 indicates an access violation l |  | aten_ops | ut |
| [2132](https://github.com/intel/torch-xpu-ops/issues/2132) | [2.9][BMG-Windows][Torch-xpu-ops UT | pbielak | pbielak | Needs PyTorch Repo Changes (upstream) | P2 | Requires upstream fix in PyTorch | Inductor / Compilation Related |  |  | aten_ops | ut |
| [2202](https://github.com/intel/torch-xpu-ops/issues/2202) | [TorchAO][BMG] The RTN performance  | Stonepia | Stonepia | Needs PyTorch Repo Changes (upstream) | P0 | Requires upstream fix in PyTorch | TorchAO | Timeout/Performance Issue - RTN performance regression in next-token latency for |  | AO | ut |
| [2214](https://github.com/intel/torch-xpu-ops/issues/2214) | test/test_sparse.py::TestSparseAnyX | jenniew | jenniew | Needs PyTorch Repo Changes (upstream) | P2 | Requires upstream fix in PyTorch | Distributed |  |  | aten_ops | ut |
| [2234](https://github.com/intel/torch-xpu-ops/issues/2234) | [upstream_ut] AssertionError: Runti | Silv3S | Silv3S | Needs PyTorch Repo Changes (upstream) | P2 | Requires upstream fix in PyTorch | Distributed |  |  | aten_ops | ut |
| [2248](https://github.com/intel/torch-xpu-ops/issues/2248) | [upstream_ut] test_cow failures | gplutop7 | gplutop7 | Needs PyTorch Repo Changes (upstream) | P2 | Requires upstream fix in PyTorch | Distributed |  |  | aten_ops | ut |
| [2251](https://github.com/intel/torch-xpu-ops/issues/2251) | [upstream_ut] test_fake_autocase go | astachowiczhabana | astachowiczhabana | Needs PyTorch Repo Changes (upstream) | P2 | Requires upstream fix in PyTorch | Dtype / Precision Related |  |  | aten_ops | ut |
| [2255](https://github.com/intel/torch-xpu-ops/issues/2255) | [upstream_ut] RuntimeError: Long is | daisyden | daisyden | Needs PyTorch Repo Changes (upstream) | P2 | Requires upstream fix in PyTorch | Flash Attention / Transformer Related |  |  | aten_ops | ut |
| [2283](https://github.com/intel/torch-xpu-ops/issues/2283) | [upstream_ut] sparse._sampled_addmm | jenniew | jenniew | Needs PyTorch Repo Changes (upstream) | P1 | Requires upstream fix in PyTorch | Distributed |  |  | aten_ops | ut |
| [2287](https://github.com/intel/torch-xpu-ops/issues/2287) | [upstream_ut] test_python_ref issue | yucai-intel | yucai-intel | Needs PyTorch Repo Changes (upstream) | P2 | Requires upstream fix in PyTorch | Others |  |  | aten_ops | ut |
| [2295](https://github.com/intel/torch-xpu-ops/issues/2295) | [upstream_ut][xpu][test]nn/test_emb | yucai-intel | yucai-intel | Needs PyTorch Repo Changes (upstream) | P2 | Requires upstream fix in PyTorch | Dtype / Precision Related |  |  | inductor | ut |
| [2301](https://github.com/intel/torch-xpu-ops/issues/2301) | [upstream_ut] dtypes not align with | daisyden | daisyden | Needs PyTorch Repo Changes (upstream) | P2 | Requires upstream fix in PyTorch | Distributed |  |  | aten_ops | ut |
| [2329](https://github.com/intel/torch-xpu-ops/issues/2329) | [upstream_ut] feature missing: get_ | etaf | etaf | Needs PyTorch Repo Changes (upstream) | P2 | Requires upstream fix in PyTorch | Others |  |  | inductor | ut |
| [2359](https://github.com/intel/torch-xpu-ops/issues/2359) | [upstream_ut] GradcheckError: Backw | BBBela | BBBela | Needs PyTorch Repo Changes (upstream) | P2 | Requires upstream fix in PyTorch | Distributed |  |  | aten_ops | ut |
| [2512](https://github.com/intel/torch-xpu-ops/issues/2512) | [upstream_ut]  RuntimeError: _histc | chunhuanMeng | libohao1201 | Needs PyTorch Repo Changes (upstream) | P2 | Requires upstream fix in PyTorch | Dtype / Precision Related | Backend/Device Issue - _histc_xpu lacks deterministic implementation on XPU back |  | aten_ops | ut |
| [2519](https://github.com/intel/torch-xpu-ops/issues/2519) | [upstream_ut]  TypeError: map2_ is  | Silv3S | libohao1201 | Needs PyTorch Repo Changes (upstream) | P2 | Requires upstream fix in PyTorch | Others | Backend/Device Issue - map2_ is only implemented on CPU tensors, indicating lack |  | aten_ops | ut |
| [2554](https://github.com/intel/torch-xpu-ops/issues/2554) | [upstream_ut]  AssertionError: Asse | daisyden | daisyden | Needs PyTorch Repo Changes (upstream) | P2 | Requires upstream fix in PyTorch | PT2E |  |  | inductor | ut |
| [2570](https://github.com/intel/torch-xpu-ops/issues/2570) | crash in sdpa. | LuFinch | sywangyi | Needs PyTorch Repo Changes (upstream) | P0 | Requires upstream fix in PyTorch | Flash Attention / Transformer Related | Others - insufficient information to determine root cause |  | aten_ops | ut |
| [2578](https://github.com/intel/torch-xpu-ops/issues/2578) | [TorchAO][UT] test/quantization/tes | Stonepia | Stonepia | Needs PyTorch Repo Changes (upstream) | P0 | Requires upstream fix in PyTorch | TorchAO |  |  | AO | build |
| [2598](https://github.com/intel/torch-xpu-ops/issues/2598) | [TorchAO][BMG]The first token laten | Stonepia | Stonepia | Needs PyTorch Repo Changes (upstream) | P2 | Requires upstream fix in PyTorch | TorchAO | Timeout/Performance Issue - First token latency drops significantly with change  |  | AO | ut |
| [2609](https://github.com/intel/torch-xpu-ops/issues/2609) | [upstream_ut]  torch._inductor.exc. | daisyden | daisyden | Needs PyTorch Repo Changes (upstream) | P2 | Requires upstream fix in PyTorch | PT2E |  | [PR](https://github.com/pytorch/pytorch/pull/171154) | inductor | ut |
| [2620](https://github.com/intel/torch-xpu-ops/issues/2620) | [upstream_ut]  AssertionError: dtyp | daisyden | daisyden | Needs PyTorch Repo Changes (upstream) | P2 | Requires upstream fix in PyTorch | TorchAO |  |  | inductor | ut |
| [2650](https://github.com/intel/torch-xpu-ops/issues/2650) | [OOB Performance] The performance i | jianyizh | jianyizh | Needs PyTorch Repo Changes (upstream) | P0 | Requires upstream fix in PyTorch | Inductor / Compilation Related | Inductor/Compilation Issue - Performance impact caused by TORCHINDUCTOR_ONLINE_S |  | aten_ops | e2e |
| [2654](https://github.com/intel/torch-xpu-ops/issues/2654) | [BMG][OOB] t5 inference performance | jianyizh | jianyizh | Needs PyTorch Repo Changes (upstream) | P0 | Requires upstream fix in PyTorch | Dtype / Precision Related | Timeout/Performance Issue - inference performance drop |  | aten_ops | e2e |
| [2655](https://github.com/intel/torch-xpu-ops/issues/2655) | [BMG][OOB] hf_Reformer performance  | jianyizh | jianyizh | Needs PyTorch Repo Changes (upstream) | P0 | Requires upstream fix in PyTorch | Others | Timeout/Performance Issue - hf_Reformer performance drop reported. |  | aten_ops | e2e |
| [2656](https://github.com/intel/torch-xpu-ops/issues/2656) | [release/2.10] models got fail_accu | None | None | Needs PyTorch Repo Changes (upstream) | P0 | Requires upstream fix in PyTorch | Dtype / Precision Related | Backend/Device Issue - failure occurs specifically on BMG WSL2 XPU backend |  | aten_ops | ut |
| [2660](https://github.com/intel/torch-xpu-ops/issues/2660) | [release/2.10][Windows][BMG] New fa | tadkrawiec | tadkrawiec | Needs PyTorch Repo Changes (upstream) | P2 | Requires upstream fix in PyTorch | Others | Others - insufficient information to determine root cause |  | aten_ops | ut |
| [2662](https://github.com/intel/torch-xpu-ops/issues/2662) | [release/2.10][Windows][BMG] New fa | tadkrawiec, kdrozd-dev | tadkrawiec, kdrozd-dev | Needs PyTorch Repo Changes (upstream) | P2 | Requires upstream fix in PyTorch | Others | Backend/Device Issue - XPU related failure in test cases on Windows with BMG |  | aten_ops | ut |
| [2663](https://github.com/intel/torch-xpu-ops/issues/2663) | test_sparse_semi_structured.py gaps | None | None | Needs PyTorch Repo Changes (upstream) | P2 | Requires upstream fix in PyTorch | Sparse Operations Related |  |  | aten_ops | ut |
| [2670](https://github.com/intel/torch-xpu-ops/issues/2670) | [upstream_ut]  RuntimeError: could  | tszulist-hbn | tszulist-hbn | Needs PyTorch Repo Changes (upstream) | P2 | Requires upstream fix in PyTorch | Distributed |  |  | aten_ops | ut |
| [2693](https://github.com/intel/torch-xpu-ops/issues/2693) | Title: [upstream_ut]  AssertionErro | hoshibara | hoshibara | Needs PyTorch Repo Changes (upstream) | P2 | Requires upstream fix in PyTorch | Distributed |  |  | inductor | ut |
| [2696](https://github.com/intel/torch-xpu-ops/issues/2696) | Title: [upstream_ut]  RuntimeError: | chunhuanMeng | chunhuanMeng | Needs PyTorch Repo Changes (upstream) | P2 | Requires upstream fix in PyTorch | Flash Attention / Transformer Related |  |  | inductor | e2e |
| [2697](https://github.com/intel/torch-xpu-ops/issues/2697) | Title: [upstream_ut]  RuntimeError: | chunhuanMeng | chunhuanMeng | Needs PyTorch Repo Changes (upstream) | P2 | Requires upstream fix in PyTorch | Flash Attention / Transformer Related |  |  | inductor | e2e |
| [2698](https://github.com/intel/torch-xpu-ops/issues/2698) | Title: [upstream_ut]  RuntimeError: | chunhuanMeng, LuFinch | chunhuanMeng, LuFinch | Needs PyTorch Repo Changes (upstream) | P2 | Requires upstream fix in PyTorch | PT2E |  |  | inductor | ut |
| [2704](https://github.com/intel/torch-xpu-ops/issues/2704) | Title: [upstream_ut]  AssertionErro | kdrozd-dev | kdrozd-dev | Needs PyTorch Repo Changes (upstream) | P2 | Requires upstream fix in PyTorch | Flash Attention / Transformer Related |  | [PR](https://github.com/pytorch/pytorch/pull/177636) | aten_ops | ut |
| [2712](https://github.com/intel/torch-xpu-ops/issues/2712) | [upstream_ut]  RuntimeError: Cannot | tszulist-hbn | tszulist-hbn | Needs PyTorch Repo Changes (upstream) | P2 | Requires upstream fix in PyTorch | Others |  |  | aten_ops | ut |
| [2714](https://github.com/intel/torch-xpu-ops/issues/2714) | [upstream_ut]  AssertionError: Obje | Silv3S | Silv3S | Needs PyTorch Repo Changes (upstream) | P2 | Requires upstream fix in PyTorch | Distributed |  |  | aten_ops | ut |
| [2715](https://github.com/intel/torch-xpu-ops/issues/2715) | [upstream_ut]  torch._dynamo.exc.Un | CuiYifeng | CuiYifeng | Needs PyTorch Repo Changes (upstream) | P2 | Requires upstream fix in PyTorch | PT2E |  |  | aten_ops | ut |
| [2722](https://github.com/intel/torch-xpu-ops/issues/2722) | [Bug Skip]: NotImplementedError: Co | Silv3S | CuiYifeng | Needs PyTorch Repo Changes (upstream) | P2 | Requires upstream fix in PyTorch | TorchAO | Backend/Device Issue - aten::flip not implemented for QuantizedXPU backend |  | aten_ops | ut |
| [2742](https://github.com/intel/torch-xpu-ops/issues/2742) | [Linux][PT2E] hf_Roberta_base model | chunhuanMeng | chunhuanMeng | Needs PyTorch Repo Changes (upstream) | P0 | Requires upstream fix in PyTorch | Flash Attention / Transformer Related | Timeout/Performance Issue - hf_Roberta_base model performance failed for both AS |  | aten_ops | e2e |
| [2798](https://github.com/intel/torch-xpu-ops/issues/2798) | Test case  test/test_dlpack.py::Tes | None | None | Needs PyTorch Repo Changes (upstream) | P2 | Requires upstream fix in PyTorch | Others |  |  | aten_ops | ut |
| [2800](https://github.com/intel/torch-xpu-ops/issues/2800) | AttributeError: 'torch._C._XpuDevic | guangyey | guangyey | Needs PyTorch Repo Changes (upstream) | P2 | Requires upstream fix in PyTorch | Flash Attention / Transformer Related |  |  | inductor | ut |
| [2802](https://github.com/intel/torch-xpu-ops/issues/2802) | Three aten._scaled_dot_product_flas | LuFinch | LuFinch | Needs PyTorch Repo Changes (upstream) | P2 | Requires upstream fix in PyTorch | Flash Attention / Transformer Related |  |  | inductor | ut |
| [2806](https://github.com/intel/torch-xpu-ops/issues/2806) | CompiledAOTI need XPU support | daisyden | daisyden | Needs PyTorch Repo Changes (upstream) | P2 | Requires upstream fix in PyTorch | PT2E |  |  | inductor | ut |
| [2810](https://github.com/intel/torch-xpu-ops/issues/2810) | AssertionError: Object comparison f | daisyden | daisyden | Needs PyTorch Repo Changes (upstream) | P2 | Requires upstream fix in PyTorch | Dtype / Precision Related |  |  | inductor | ut |
| [2888](https://github.com/intel/torch-xpu-ops/issues/2888) | torch._inductor.exc.InductorError:  | Stonepia | Stonepia | Needs PyTorch Repo Changes (upstream) | P2 | Requires upstream fix in PyTorch | Inductor / Compilation Related |  |  | inductor | ut |
| [2891](https://github.com/intel/torch-xpu-ops/issues/2891) | RuntimeError: Expected to find "(26 | chunhuanMeng | chunhuanMeng | Needs PyTorch Repo Changes (upstream) | P2 | Requires upstream fix in PyTorch | Others |  |  | inductor | e2e |
| [2907](https://github.com/intel/torch-xpu-ops/issues/2907) | [release/2.11] Models performance r | xuhancn | xuhancn | Needs PyTorch Repo Changes (upstream) | P0 | Requires upstream fix in PyTorch | Others | Timeout/Performance Issue - models performance regression in testcases |  | aten_ops | ut |
| [2908](https://github.com/intel/torch-xpu-ops/issues/2908) | [release/2.11] Model fail_accuracy  | xuhancn | xuhancn | Needs PyTorch Repo Changes (upstream) | P1 | Requires upstream fix in PyTorch | Dtype / Precision Related | Others - insufficient information to determine root cause |  | aten_ops | e2e |
| [2912](https://github.com/intel/torch-xpu-ops/issues/2912) | [release/2.11] UT extended 220 new  | None | None | Needs PyTorch Repo Changes (upstream) | P2 | Requires upstream fix in PyTorch | Others | Others - insufficient information to determine root cause |  | aten_ops | ut |
| [2918](https://github.com/intel/torch-xpu-ops/issues/2918) | [XPU][upstream_ut][COW] Skip non-su | gplutop7 | gplutop7 | Needs PyTorch Repo Changes (upstream) | P2 | Requires upstream fix in PyTorch | Others |  | [PR](https://github.com/pytorch/pytorch/pull/174670) | aten_ops | ut |
| [2919](https://github.com/intel/torch-xpu-ops/issues/2919) | [XPU][upstream_ut][COW] Fix materia | gplutop7 | gplutop7 | Needs PyTorch Repo Changes (upstream) | P2 | Requires upstream fix in PyTorch | Others |  |  | aten_ops | ut |
| [2922](https://github.com/intel/torch-xpu-ops/issues/2922) | [release/2.11] UT inductor Assertio | tadkrawiec | tadkrawiec | Needs PyTorch Repo Changes (upstream) | P2 | Requires upstream fix in PyTorch | Inductor / Compilation Related | Backend/Device Issue - pass_fds not supported on Windows |  | aten_ops | ut |
| [2924](https://github.com/intel/torch-xpu-ops/issues/2924) | [release/2.11] xcit_large_24_p8_224 | jianyizh, mengfei25 | jianyizh, mengfei25 | Needs PyTorch Repo Changes (upstream) | P1 | Requires upstream fix in PyTorch | Dtype / Precision Related | Dtype/Precision Issue - amp_bf16 training accuracy failure |  | aten_ops | e2e |
| [2928](https://github.com/intel/torch-xpu-ops/issues/2928) | [release/2.11] pyhpc_turbulent_kine | jianyizh | jianyizh | Needs PyTorch Repo Changes (upstream) | P1 | Requires upstream fix in PyTorch | Dtype / Precision Related | Dtype/Precision Issue - fp32 inference accuracy failure |  | aten_ops | e2e |
| [2929](https://github.com/intel/torch-xpu-ops/issues/2929) | [release/2.11] volo_d1_224 inferenc | jianyizh | jianyizh | Needs PyTorch Repo Changes (upstream) | P1 | Requires upstream fix in PyTorch | Dtype / Precision Related | Backend/Device Issue - fail_to_run on XPU for volo_d1_224 inference |  | aten_ops | e2e |
| [2930](https://github.com/intel/torch-xpu-ops/issues/2930) | [release/2.11] UT skip test_binary_ | None | None | Needs PyTorch Repo Changes (upstream) | P2 | Requires upstream fix in PyTorch | Others | Skip/No Test Exists - test is skipped due to RuntimeError |  | aten_ops | ut |
| [2932](https://github.com/intel/torch-xpu-ops/issues/2932) | [release/2.11] jx_nest_base and vol | jianyizh | jianyizh | Needs PyTorch Repo Changes (upstream) | P2 | Requires upstream fix in PyTorch | Dtype / Precision Related | Failure - encountered AssertionError during training |  | aten_ops | e2e |
| [2935](https://github.com/intel/torch-xpu-ops/issues/2935) | [release/2.11][inductor] huggingfac | jianyizh | jianyizh | Needs PyTorch Repo Changes (upstream) | P0 | Requires upstream fix in PyTorch | Inductor / Compilation Related | Inductor/Compilation Issue - performance regression in XLNetLMHeadModel with amp |  | aten_ops | e2e |
| [2938](https://github.com/intel/torch-xpu-ops/issues/2938) | [release/2.11] basic_gnn_gin and ba | jianyizh | jianyizh | Needs PyTorch Repo Changes (upstream) | P2 | Requires upstream fix in PyTorch | Dtype / Precision Related | Timeout/Performance Issue - inference fp32 performance dropped ~25% |  | aten_ops | e2e |
| [2939](https://github.com/intel/torch-xpu-ops/issues/2939) | [release/2.11] gmlp_s16_224 inferen | jianyizh | jianyizh | Needs PyTorch Repo Changes (upstream) | P2 | Requires upstream fix in PyTorch | Flash Attention / Transformer Related | Timeout/Performance Issue - inference amp performance dropped ~15% |  | aten_ops | e2e |
| [2940](https://github.com/intel/torch-xpu-ops/issues/2940) | [release/2.11] Models performance d | jianyizh, LuFinch | jianyizh, LuFinch | Needs PyTorch Repo Changes (upstream) | P0 | Requires upstream fix in PyTorch | Others | Timeout/Performance Issue - Models performance dropped ~10% - 15% |  | aten_ops | e2e |
| [2952](https://github.com/intel/torch-xpu-ops/issues/2952) | [release/2.11][wsl] timm_models_acc | weishi-deng | weishi-deng | Needs PyTorch Repo Changes (upstream) | P0 | Requires upstream fix in PyTorch | Dtype / Precision Related | Dtype/Precision Issue - bfloat16 accuracy failure in model training |  | aten_ops | ut |
| [2958](https://github.com/intel/torch-xpu-ops/issues/2958) | AssertionError of test_dtensor_basi | daisyden | daisyden | Needs PyTorch Repo Changes (upstream) | P2 | Requires upstream fix in PyTorch | Inductor / Compilation Related |  |  | inductor | ut |
| [2960](https://github.com/intel/torch-xpu-ops/issues/2960) | [release/2.11] timm_models_xcit_lar | None | None | Needs PyTorch Repo Changes (upstream) | P0 | Requires upstream fix in PyTorch | Dtype / Precision Related | Dtype/Precision Issue - float16 training accuracy test failure |  | aten_ops | ut |
| [2979](https://github.com/intel/torch-xpu-ops/issues/2979) | eca_halonext26ts got RuntimeError:  | None | None | Needs PyTorch Repo Changes (upstream) | P0 | Requires upstream fix in PyTorch | Others | Backend/Device Issue - ZE_RESULT_ERROR_MODULE_BUILD_FAILURE indicates a problem  |  | aten_ops | e2e |
| [2984](https://github.com/intel/torch-xpu-ops/issues/2984) | [release/2.11] sebotnet33ts_256 fp3 | jianyizh, weishi-deng | jianyizh, weishi-deng | Needs PyTorch Repo Changes (upstream) | P1 | Requires upstream fix in PyTorch | Dtype / Precision Related | Backend/Device Issue - XPU specific failure during fp32 training accuracy check |  | aten_ops | e2e |
| [2997](https://github.com/intel/torch-xpu-ops/issues/2997) | AssertionError of test_linear_and_c | etaf | etaf | Needs PyTorch Repo Changes (upstream) | P2 | Requires upstream fix in PyTorch | Flash Attention / Transformer Related |  |  | inductor | ut |
| [2999](https://github.com/intel/torch-xpu-ops/issues/2999) | KeyError: 'eager_numerics.use_pytor | daisyden | daisyden | Needs PyTorch Repo Changes (upstream) | P2 | Requires upstream fix in PyTorch | Others |  |  | inductor | ut |
| [3004](https://github.com/intel/torch-xpu-ops/issues/3004) | TypeError: _xpu_recordMemoryHistory | guangyey | guangyey | Needs PyTorch Repo Changes (upstream) | P2 | Requires upstream fix in PyTorch | Others |  |  | inductor | ut |
| [3006](https://github.com/intel/torch-xpu-ops/issues/3006) | AssertionError: '.to(tl.float16)' u | CuiYifeng | CuiYifeng | Needs PyTorch Repo Changes (upstream) | P2 | Requires upstream fix in PyTorch | PT2E |  |  | inductor | e2e |
| [3041](https://github.com/intel/torch-xpu-ops/issues/3041) | AssertionError: Expected len(flat_d | Silv3S | Silv3S | Needs PyTorch Repo Changes (upstream) | P2 | Requires upstream fix in PyTorch | Dtype / Precision Related |  |  | aten_ops | ut |
| [3077](https://github.com/intel/torch-xpu-ops/issues/3077) | [Bug Skip] test_dlpack.py::TestTorc | AKloniecki | AKloniecki | Needs PyTorch Repo Changes (upstream) | P2 | Requires upstream fix in PyTorch | Others |  |  | aten_ops | ut |
| [3094](https://github.com/intel/torch-xpu-ops/issues/3094) | XPUGraph tree support | None | None | Needs PyTorch Repo Changes (upstream) | P2 | Requires upstream fix in PyTorch | Inductor / Compilation Related |  |  | inductor | ut |
| [3095](https://github.com/intel/torch-xpu-ops/issues/3095) | cutlass support blocks some unit te | None | None | Needs PyTorch Repo Changes (upstream) | P2 | Requires upstream fix in PyTorch | Others |  |  | inductor | ut |
| [3126](https://github.com/intel/torch-xpu-ops/issues/3126) | [upstream_ut]  Two NestedTensor iss | LuFinch | LuFinch | Needs PyTorch Repo Changes (upstream) | P2 | Requires upstream fix in PyTorch | Flash Attention / Transformer Related |  |  | aten_ops | ut |
| [3127](https://github.com/intel/torch-xpu-ops/issues/3127) | [upstream_ut]  AssertionError: Asse | LuFinch | LuFinch | Needs PyTorch Repo Changes (upstream) | P2 | Requires upstream fix in PyTorch | Flash Attention / Transformer Related |  |  | aten_ops | ut |
| [3128](https://github.com/intel/torch-xpu-ops/issues/3128) | [upstream_ut]  AssertionError: Runt | LuFinch | LuFinch | Needs PyTorch Repo Changes (upstream) | P2 | Requires upstream fix in PyTorch | Distributed |  |  | aten_ops | ut |
| [3129](https://github.com/intel/torch-xpu-ops/issues/3129) | [upstream_ut]  AssertionError: User | LuFinch | LuFinch | Needs PyTorch Repo Changes (upstream) | P2 | Requires upstream fix in PyTorch | Flash Attention / Transformer Related |  |  | aten_ops | ut |
| [3130](https://github.com/intel/torch-xpu-ops/issues/3130) | [upstream_ut]  AssertionError: tens | LuFinch | LuFinch | Needs PyTorch Repo Changes (upstream) | P2 | Requires upstream fix in PyTorch | Flash Attention / Transformer Related |  |  | aten_ops | ut |
| [3131](https://github.com/intel/torch-xpu-ops/issues/3131) | [upstream_ut]  NotImplementedError: | chunhuanMeng | chunhuanMeng | Needs PyTorch Repo Changes (upstream) | P2 | Requires upstream fix in PyTorch | Flash Attention / Transformer Related |  |  | aten_ops | ut |
| [3132](https://github.com/intel/torch-xpu-ops/issues/3132) | [upstream_ut]  transfomers test rep | LuFinch | LuFinch | Needs PyTorch Repo Changes (upstream) | P2 | Requires upstream fix in PyTorch | Flash Attention / Transformer Related |  |  | aten_ops | ut |
| [3133](https://github.com/intel/torch-xpu-ops/issues/3133) | [upstream_ut]  RuntimeError: scaled | LuFinch | LuFinch | Needs PyTorch Repo Changes (upstream) | P2 | Requires upstream fix in PyTorch | Flash Attention / Transformer Related |  |  | aten_ops | ut |
| [3137](https://github.com/intel/torch-xpu-ops/issues/3137) | [upstream_ut]  RuntimeError: expect | LuFinch | LuFinch | Needs PyTorch Repo Changes (upstream) | P2 | Requires upstream fix in PyTorch | Flash Attention / Transformer Related |  |  | aten_ops | ut |
| [3140](https://github.com/intel/torch-xpu-ops/issues/3140) | [upstream_ut]  RuntimeError: FlashA | LuFinch | LuFinch | Needs PyTorch Repo Changes (upstream) | P2 | Requires upstream fix in PyTorch | Flash Attention / Transformer Related |  |  | aten_ops | ut |
| [3141](https://github.com/intel/torch-xpu-ops/issues/3141) | [upstream_ut]  RuntimeError: FlashA | LuFinch | LuFinch | Needs PyTorch Repo Changes (upstream) | P2 | Requires upstream fix in PyTorch | Flash Attention / Transformer Related |  |  | aten_ops | ut |
| [3142](https://github.com/intel/torch-xpu-ops/issues/3142) | [upstream_ut]  RuntimeError: The sy | LuFinch | LuFinch | Needs PyTorch Repo Changes (upstream) | P2 | Requires upstream fix in PyTorch | Flash Attention / Transformer Related |  |  | aten_ops | ut |
| [3143](https://github.com/intel/torch-xpu-ops/issues/3143) | NotImplementedError: The operator ' | LuFinch | LuFinch | Needs PyTorch Repo Changes (upstream) | P2 | Requires upstream fix in PyTorch | Flash Attention / Transformer Related |  |  | aten_ops | ut |
| [3148](https://github.com/intel/torch-xpu-ops/issues/3148) | [Triton] Huggingface openai/whisper | None | None | Needs PyTorch Repo Changes (upstream) | P0 | Requires upstream fix in PyTorch | Inductor / Compilation Related | Backend/Device Issue - XPU specific failure with Huggingface model accuracy |  | aten_ops | e2e |
| [3151](https://github.com/intel/torch-xpu-ops/issues/3151) | [Triton] Timm_models  rexnet_100 /  | None | None | Needs PyTorch Repo Changes (upstream) | P0 | Requires upstream fix in PyTorch | Inductor / Compilation Related | Backend/Device Issue - XPU specific failure with Timm models in Triton. |  | aten_ops | e2e |
| [3163](https://github.com/intel/torch-xpu-ops/issues/3163) | [Bug Skip]: Object comparison faile | chunhuanMeng | chunhuanMeng | Needs PyTorch Repo Changes (upstream) | P2 | Requires upstream fix in PyTorch | Distributed |  |  | aten_ops | ut |
| [3165](https://github.com/intel/torch-xpu-ops/issues/3165) | test_sparse_csr_xpu.py::TestSparseC | None | None | Needs PyTorch Repo Changes (upstream) | P2 | Requires upstream fix in PyTorch | Sparse Operations Related |  |  | aten_ops | ut |
| [3167](https://github.com/intel/torch-xpu-ops/issues/3167) | NotImplementedError: Could not run  | tszulist-hbn | tszulist-hbn | Needs PyTorch Repo Changes (upstream) | P1 | Requires upstream fix in PyTorch | Sparse Operations Related |  |  | aten_ops | ut |
| [3168](https://github.com/intel/torch-xpu-ops/issues/3168) | NotImplementedError: Could not run  | jenniew | jenniew | Needs PyTorch Repo Changes (upstream) | P1 | Requires upstream fix in PyTorch | Sparse Operations Related |  |  | aten_ops | ut |
| [3169](https://github.com/intel/torch-xpu-ops/issues/3169) | NotImplementedError: Could not run  | jkosnox | jkosnox | Needs PyTorch Repo Changes (upstream) | P2 | Requires upstream fix in PyTorch | Sparse Operations Related |  |  | aten_ops | ut |
| [3170](https://github.com/intel/torch-xpu-ops/issues/3170) | Unskip test_bmm_windows_error_xpu_f | jenniew | jenniew | Needs PyTorch Repo Changes (upstream) | P2 | Requires upstream fix in PyTorch | Others |  |  | aten_ops | ut |
| [3187](https://github.com/intel/torch-xpu-ops/issues/3187) | PyTorch XPU gpu_cpp_wrapper fails w | CuiYifeng | CuiYifeng | Needs PyTorch Repo Changes (upstream) | P2 | Requires upstream fix in PyTorch | Inductor / Compilation Related |  |  | aten_ops | ut |
| [3191](https://github.com/intel/torch-xpu-ops/issues/3191) | torch._inductor.exc.InductorError:  | EikanWang, Copilot | EikanWang, Copilot | Needs PyTorch Repo Changes (upstream) | P2 | Requires upstream fix in PyTorch | Inductor / Compilation Related | Inductor/Compilation Issue - Assertion failure due to conflicting fallback and d |  | aten_ops | e2e |
| [3229](https://github.com/intel/torch-xpu-ops/issues/3229) | RuntimeError: No viable backend for | None | None | Needs PyTorch Repo Changes (upstream) | P2 | Requires upstream fix in PyTorch | Flash Attention / Transformer Related |  |  | aten_ops | ut |
| [3238](https://github.com/intel/torch-xpu-ops/issues/3238) | The supported dtypes of _refs.stft  | CuiYifeng | CuiYifeng | Needs PyTorch Repo Changes (upstream) | P2 | Requires upstream fix in PyTorch | Dtype / Precision Related |  |  | aten_ops | ut |
| [3247](https://github.com/intel/torch-xpu-ops/issues/3247) | NotImplementedError: "dot_xpu_mkl"  | Silv3S | Silv3S | Needs PyTorch Repo Changes (upstream) | P2 | Requires upstream fix in PyTorch | Others |  |  | aten_ops | ut |
| [3255](https://github.com/intel/torch-xpu-ops/issues/3255) | [Linux][PT2E][Regression] Some perf | None | None | Needs PyTorch Repo Changes (upstream) | P0 | Requires upstream fix in PyTorch | TorchAO | Timeout/Performance Issue - performance tests failed due to regression in execut |  | aten_ops | e2e |
| [3257](https://github.com/intel/torch-xpu-ops/issues/3257) | [Linux][E2E][Regression] Huggingfac | None | None | Needs PyTorch Repo Changes (upstream) | P0 | Requires upstream fix in PyTorch | Others | Backend/Device Issue - Huggingface test models failed to run on XPU, indicating  |  | aten_ops | e2e |

### 1.3 Revisit the PR as case failed

| ID | Title | Owner | Owner Transferred | TBD | Priority | Action Reason | Category | Root Cause | PR | Module | Test Module |
|---|-------|-------|-------------------|-----|---------|--------|----------|-----------|-----|--------|-------------|
| [3246](https://github.com/intel/torch-xpu-ops/issues/3246) | AssertionError: Booleans mismatch:  | BartoszKokoszko | BartoszKokoszko | Revisit the PR as case failed | P2 | Test failed after PR merge | Distributed |  | [PR](https://github.com/intel/torch-xpu-ops/pull/3249) | aten_ops | ut |

### 1.4 Verify the issue

| ID | Title | Owner | Owner Transferred | TBD | Priority | Action Reason | Category | Root Cause | PR | Module | Test Module |
|---|-------|-------|-------------------|-----|---------|--------|----------|-----------|-----|--------|-------------|
| [2331](https://github.com/intel/torch-xpu-ops/issues/2331) | [upstream_ut] AssertionError: Scala | hoshibara | daisyden | Verify the issue | P2 | PR closed, needs verification | Dtype / Precision Related |  | [PR](https://github.com/pytorch/pytorch/pull/172314) | inductor | ut |
| [2694](https://github.com/intel/torch-xpu-ops/issues/2694) | Title: [upstream_ut]  AssertionErro | daisyden | daisyden | Verify the issue | P2 | PR closed, needs verification | Distributed |  | [PR](https://github.com/pytorch/pytorch/pull/171773) | inductor | ut |
| [3007](https://github.com/intel/torch-xpu-ops/issues/3007) | AssertionError: Scalars are not equ | daisyden | daisyden | Verify the issue | P2 | PR closed, needs verification | Flash Attention / Transformer Related |  | [PR](https://github.com/pytorch/pytorch/pull/178369) | inductor | e2e |

### 1.5 add to skiplist

| ID | Title | Owner | Owner Transferred | TBD | Priority | Action Reason | Category | Root Cause | PR | Module | Test Module |
|---|-------|-------|-------------------|-----|---------|--------|----------|-----------|-----|--------|-------------|
| [2164](https://github.com/intel/torch-xpu-ops/issues/2164) | skip test_no_cuda_monkeypatch as it | daisyden | daisyden | add to skiplist | P2 | Labeled as not target/wontfix | Distributed |  |  | aten_ops | ut |
| [2309](https://github.com/intel/torch-xpu-ops/issues/2309) | unsupported ops with PYTORCH_ENABLE | daisyden | daisyden | add to skiplist | P2 | Labeled as not target/wontfix | TorchAO |  |  | aten_ops | ut |
| [2472](https://github.com/intel/torch-xpu-ops/issues/2472) | [upstream_ut]  NotImplementedError: | Silv3S | daisyden | add to skiplist | P2 | Labeled as not target/wontfix | PT2E |  |  | aten_ops | ut |
| [2508](https://github.com/intel/torch-xpu-ops/issues/2508) | TypedStorage / TypedTensors depreca | Silv3S | libohao1201 | add to skiplist | P1 | Labeled as not target/wontfix | TorchAO |  | [PR](https://github.com/intel/torch-xpu-ops/pull/3260) | aten_ops | ut |

### Issues without Assignee

| ID | Title | Owner | Owner Transferred | TBD | Priority | Reason | Category | Root Cause | PR | Module | Test Module |
|---|-------|-------|-------------------|-----|---------|--------|----------|-----------|-----|--------|-------------|
| [3080](https://github.com/intel/torch-xpu-ops/issues/3080) | cudagraph tests blocked by feature  | None | chuanqi | assign owner | P2 | UT issue with few failures | Inductor / Compilation Related | 10 - Feature Not Supported |  | aten_ops | ut |
| [2199](https://github.com/intel/torch-xpu-ops/issues/2199) | Fix reduction and norm register spi | None | chuanqi | assign owner | P2 | UT issue with few failures | Others | Memory/Shared Memory Issue - register spill in reduction and norm operations |  | aten_ops | ut |
| [2196](https://github.com/intel/torch-xpu-ops/issues/2196) | Fix DistributionElementwiseKernelFu | None | chuanqi | assign owner | P2 | UT issue with few failures | Inductor / Compilation Related | Memory/Shared Memory Issue - register spill in DistributionElementwiseKernelFunc |  | aten_ops | ut |

---

## 2. Duplicated Issues

Issues that share test cases with other issues.

| ID | Title | Owner | Reporter | Duplicated With | Priority | Reason | Category | Root Cause | PR | Module | Test Module |
|---|-------|-------|----------|-----------------|---------|--------|----------|-----------|-----|--------|-------------|
| [1893](https://github.com/intel/torch-xpu-ops/issues/1893) | [upstream_ut] oneDNN accuracy  | chunhuanMeng | daisyden | 1951 | P2 | UT issue with few failures | Distributed |  |  | aten_ops | ut |
| [1951](https://github.com/intel/torch-xpu-ops/issues/1951) | Functionality issues in TestCo | AKloniecki | daisyden | 1893 | P2 | UT issue with few failures | Distributed |  |  | aten_ops | ut |
| [1973](https://github.com/intel/torch-xpu-ops/issues/1973) | AssertionError: Scalars or Ten | gplutop7 | mengfei25 | 2837,2840 | P2 | UT issue with few failures | Distributed | Failure - Tensor values not equal or close in test_depthwise_conv_64bit_indexing_xpu |  | aten_ops | ut |
| [2006](https://github.com/intel/torch-xpu-ops/issues/2006) | work-item/workgroup issue in s | BartoszKokoszko | daisyden | 2257 | P2 | UT issue with few failures | TorchAO | Backend/Device Issue - work-group size exceeds device limitations on XPU |  | aten_ops | ut |
| [2015](https://github.com/intel/torch-xpu-ops/issues/2015) | inf is returned by nn.Transfor | yucai-intel | daisyden | 2186,2529 | P2 | UT issue with few failures | Flash Attention / Transformer Related | Dtype/Precision Issue - inf returned using float16 in TransformerEncoderLayer on |  | aten_ops | ut |
| [2186](https://github.com/intel/torch-xpu-ops/issues/2186) | AssertionError: Mul tiheadAtte | daisyden | daisyden | 2015 | P2 | UT issue with few failures | Flash Attention / Transformer Related | Failure - Mul tiheadAttention does not support NestedTensor outside of its fast path |  | aten_ops | ut |
| [2220](https://github.com/intel/torch-xpu-ops/issues/2220) | test/test_sparse_csr.py::TestS | None | wincent8 | 2246 | P2 | UT issue with few failures | TorchAO | Backend/Device Issue - unknown SPIR-V extension 'SPV_INTEL_subgroup_matrix_multi |  | aten_ops | ut |
| [2230](https://github.com/intel/torch-xpu-ops/issues/2230) | test_sparse_csr.py::TestSparse | None | wincent8 | 2246,3175,3176 | P1 | UT with 28 failed test cases | Distributed | Backend/Device Issue - inputs are not on the same GPU device |  | aten_ops | ut |
| [2235](https://github.com/intel/torch-xpu-ops/issues/2235) | test/test_sparse_csr.py::TestS | None | wincent8 | 3047 | P2 | UT issue with few failures | Distributed | Backend/Device Issue - unexpected warning from non-optimal triton kernel paramet |  | aten_ops | ut |
| [2238](https://github.com/intel/torch-xpu-ops/issues/2238) | Exception: Tensor-likes are no | BBBela | zxd1997066 | 3105 | P2 | UT issue with few failures | Distributed |  | [PR](https://github.com/intel/torch-xpu-ops/pull/2886) | aten_ops | ut |
| [2244](https://github.com/intel/torch-xpu-ops/issues/2244) | test/test_sparse_csr.py::TestS | jenniew | wincent8 | 3177 | P2 | UT issue with few failures | Distributed | Error - empty_sparse_compressed expects non-block layout but received SparseBsr |  | aten_ops | ut |
| [2246](https://github.com/intel/torch-xpu-ops/issues/2246) | torch/sparse/_triton_ops*.py n | None | wincent8 | 2220,2230 | P1 | UT with 33 failed test cases | Distributed | Backend/Device Issue - Torch not compiled with CUDA enabled for Intel GPU suppor |  | unknown | ut |
| [2253](https://github.com/intel/torch-xpu-ops/issues/2253) | the supported dtypes are not a | daisyden | daisyden | 2482 | P2 | UT issue with few failures | Dtype / Precision Related |  |  | aten_ops | ut |
| [2257](https://github.com/intel/torch-xpu-ops/issues/2257) | Accuracy failures in test/xpu/ | pbielak | zxd1997066 | 2006 | P1 | UT with 40 failed test cases | Distributed |  | [PR](https://github.com/intel/torch-xpu-ops/pull/2808) | aten_ops | ut |
| [2270](https://github.com/intel/torch-xpu-ops/issues/2270) | Backend Compatibility Error in | LuFinch | libohao1201 | 2442 | P2 | UT issue with few failures | Distributed | Flash Attention/Specific Ops Issue - test involves aten._flash_attention_forward |  | aten_ops | ut |
| [2285](https://github.com/intel/torch-xpu-ops/issues/2285) | Support efficient attention | chunhuanMeng | daisyden | 2358 | P2 | UT issue with few failures | Flash Attention / Transformer Related | Backend/Device Issue - aten::_efficient_attention_forward not supported on CPU b |  | aten_ops | ut |
| [2358](https://github.com/intel/torch-xpu-ops/issues/2358) | test/test_view_ops.py::TestOld | Silv3S | wincent8 | 2285 | P2 | UT issue with few failures | TorchAO |  |  | aten_ops | ut |
| [2436](https://github.com/intel/torch-xpu-ops/issues/2436) | [upstream_ut]  AttributeError: | daisyden | daisyden | 2675 | P1 | UT with 51 failed test cases | Flash Attention / Transformer Related | Error - 'NoneType' object has no attribute 'clone' due to missing object handling |  | aten_ops | ut |
| [2442](https://github.com/intel/torch-xpu-ops/issues/2442) | [Bug Skip]: NotImplementedErro | daisyden, LuFinch | CuiYifeng | 2270 | P2 | UT issue with few failures | Flash Attention / Transformer Related | Flash Attention/Specific Ops Issue - aten::_flash_attention_forward not implemen |  | aten_ops | ut |
| [2482](https://github.com/intel/torch-xpu-ops/issues/2482) | test_dtypes issue introduced b | daisyden | daisyden | 2253 | P2 | UT issue with few failures | Dtype / Precision Related | Dtype/Precision Issue - test_dtypes issue related to data types in conv_transpos |  | aten_ops | ut |
| [2529](https://github.com/intel/torch-xpu-ops/issues/2529) | [upstream_ut]  AssertionError: | Silv3S | daisyden | 2015,3136 | P2 | UT issue with few failures | Distributed | Failure - test assertion failed with False is not true |  | aten_ops | ut |
| [2530](https://github.com/intel/torch-xpu-ops/issues/2530) | Title: [upstream_ut]  Assertio | PatrykWilczewski | daisyden | 2817 | P2 | UT issue with few failures | Distributed | Failure - RuntimeError not raised as expected in test |  | aten_ops | ut |
| [2611](https://github.com/intel/torch-xpu-ops/issues/2611) | [upstream_ut]  AssertionError: | daisyden | daisyden | 2613 | P2 | UT issue with few failures | Distributed |  |  | inductor | ut |
| [2613](https://github.com/intel/torch-xpu-ops/issues/2613) | [upstream_ut]  AssertionError: | daisyden | daisyden | 2611 | P2 | UT issue with few failures | Distributed |  |  | inductor | ut |
| [2618](https://github.com/intel/torch-xpu-ops/issues/2618) | [Bug Skip]: [regression] Asser | jmamzax | kaileiyx | 3089 | P0 | Regression - passed before but failed now | Distributed |  | [PR](https://github.com/numpy/numpy/pull/22525) | unknown | ut |
| [2675](https://github.com/intel/torch-xpu-ops/issues/2675) | [Bug Skip]: AttributeError: 'N | pponikox | kaileiyx | 2436 | P2 | UT issue with few failures | Flash Attention / Transformer Related | Error - 'NoneType' object has no attribute 'clone' due to missing object reference |  | aten_ops | ut |
| [2817](https://github.com/intel/torch-xpu-ops/issues/2817) | Expected error message is diff | kdrozd-dev | Silv3S | 2530 | P2 | UT issue with few failures | Dtype / Precision Related | Failure - mismatch between expected and actual error message |  | aten_ops | ut |
| [2837](https://github.com/intel/torch-xpu-ops/issues/2837) | Accuracy issue for Muon optimi | Silv3S | kdrozd-dev | 1973 | P2 | UT issue with few failures | Distributed | Failure - Tensor-likes not close in Muon optimizer test |  | aten_ops | ut |
| [2840](https://github.com/intel/torch-xpu-ops/issues/2840) | Accuracy issue with 64 bit ind | SlawomirLaba, Silv3S | kdrozd-dev | 1973 | P2 | UT issue with few failures | Distributed | Dtype/Precision Issue - Tensor comparison failed due to small absolute differenc |  | aten_ops | ut |
| [2869](https://github.com/intel/torch-xpu-ops/issues/2869) | [Bug Skip]: New UT failure in  | None | RUIJIEZHONG66166 | 3160 | P2 | UT issue with few failures | Flash Attention / Transformer Related | Skip/No Test Exists - Test is marked as skipped or not executed |  | aten_ops | ut |
| [2966](https://github.com/intel/torch-xpu-ops/issues/2966) | [Bug Skip]: [Regression]2026-3 | jmamzax | kaileiyx | 3114 | P0 | Regression - passed before but failed now | Distributed | Timeout/Performance Issue - Example code timed out during test execution. |  | aten_ops | ut |
| [3047](https://github.com/intel/torch-xpu-ops/issues/3047) | [Bug Skip]: [Regression]UT fai | None | kaileiyx | 2235 | P0 | Regression - passed before but failed now | Flash Attention / Transformer Related | Failure - Torch not compiled with CUDA enabled assertion error |  | unknown | ut |
| [3089](https://github.com/intel/torch-xpu-ops/issues/3089) | AssertionError: Torch not comp | jmamzax | jmamzax | 2618 | P2 | UT issue with few failures | TorchAO |  |  | unknown | ut |
| [3105](https://github.com/intel/torch-xpu-ops/issues/3105) | Wrong results from oneDNN conv | BBBela | BBBela | 2238 | P2 | UT issue with few failures | Distributed | DNNL/OneDNN Issue - wrong results from oneDNN conv2d kernels due to data pointer |  | aten_ops | ut |
| [3114](https://github.com/intel/torch-xpu-ops/issues/3114) | [Bug Skip]: Failure skip on 20 | None | guangyey | 2966 | P2 | UT issue with few failures | Flash Attention / Transformer Related | Skip/No Test Exists - test was skipped on 2026-3-21 |  | aten_ops | ut |
| [3136](https://github.com/intel/torch-xpu-ops/issues/3136) | [upstream_ut]  AssertionError: | LuFinch | daisyden | 2529 | P2 | UT issue with few failures | Flash Attention / Transformer Related |  |  | aten_ops | ut |
| [3156](https://github.com/intel/torch-xpu-ops/issues/3156) | AssertionError: 'Assertion cur | kdrozd-dev | kdrozd-dev | 3184 | P2 | UT issue with few failures | Dtype / Precision Related | Failure - Assertion cur_target >= 0 && cur_target < n_classes failed in cross entropy loss test |  | aten_ops | ut |
| [3160](https://github.com/intel/torch-xpu-ops/issues/3160) | compiler not found (Windows) | kdrozd-dev | kdrozd-dev | 2869 | P2 | UT issue with few failures | Inductor / Compilation Related |  |  | aten_ops | ut |
| [3175](https://github.com/intel/torch-xpu-ops/issues/3175) | [Bug Skip]: ValueError: sample | None | CuiYifeng | 2230 | P2 | UT issue with few failures | Sparse Operations Related | Backend/Device Issue - inputs are not on the same XPU device |  | aten_ops | ut |
| [3176](https://github.com/intel/torch-xpu-ops/issues/3176) | [Bug Skip]: ValueError: _scale | None | CuiYifeng | 2230 | P2 | UT issue with few failures | Flash Attention / Transformer Related | Backend/Device Issue - inputs are not on the same XPU device |  | aten_ops | ut |
| [3177](https://github.com/intel/torch-xpu-ops/issues/3177) | Accuracy gap of BF16/FP16 test | jenniew | CuiYifeng | 2244 | P2 | UT issue with few failures | Distributed | Dtype/Precision Issue - Accuracy gap in BF16/FP16 test |  | aten_ops | ut |
| [3184](https://github.com/intel/torch-xpu-ops/issues/3184) | New failing UTs: test_cross_en | wpietka | BBBela | 3156 | P2 | UT issue with few failures | Dtype / Precision Related | Failure - test expects a specific condition to be true but it failed during execution. |  | aten_ops | ut |

---

## 3. Issues with Dependency

Issues that have dependencies on other components (non-empty).

| ID | Title | Owner | Type | Priority | Reason | Category | Root Cause | Dependency | PR | Labels |
|---|-------|------|------|---------|--------|----------|-----------|------------|-----|--------|
| [1547](https://github.com/intel/torch-xpu-ops/issues/1547) | [distributed] NotImplemen | Chao1Han | feature request | P2 | UT issue with few failures | Distributed | Backend/Device Issue - XPU device does not support the 'symm_mem::fused_matmul_r | oneAPI |  | module: distributed, dependency component: oneAPI |
| [1551](https://github.com/intel/torch-xpu-ops/issues/1551) | [distributed] NotImplemen | Chao1Han | feature request | P2 | UT issue with few failures | Distributed | Backend/Device Issue - XPU device does not support the 'symm_mem::fused_scaled_m | oneAPI |  | module: distributed, dependency component: oneAPI |
| [1556](https://github.com/intel/torch-xpu-ops/issues/1556) | [distributed] NotImplemen | pkourdis | feature request | P2 | UT issue with few failures | Distributed | Distributed/Gloo Issue - Operator lacks a sharding strategy in distributed conte | oneDNN |  | module: distributed, dependency component: oneDNN |
| [1912](https://github.com/intel/torch-xpu-ops/issues/1912) | Implement the torch.ops.a | liangan1 | feature request | P2 | UT issue with few failures | TorchAO | Backend/Device Issue - Implementation required for XPU dequantization of CUDA in | oneDNN |  | dependency component: oneDNN |
| [2089](https://github.com/intel/torch-xpu-ops/issues/2089) | need an implementation th | guangyey | feature request | P2 | UT issue with few failures | Others | Backend/Device Issue - torch.xpu.is_available() initializes GPU context unintent | driver |  | dependency component: driver |

---

## 4. Others

Issues that don't fall into the categories above.

| ID | Title | Owner | Priority | Reason | Category | Root Cause | Labels | PR | Module | Test Module |
|---|-------|-------|---------|--------|----------|-----------|--------|-----|--------|-------------|
| [3150](https://github.com/intel/torch-xpu-ops/issues/3150) | [Task] Align XPU kernel's impl | guangyey | P2 | UT issue with few failures | Inductor / Compilation Related | device-specific backend discrepancy. |  |  | aten_ops | ut |
| [3060](https://github.com/intel/torch-xpu-ops/issues/3060) | Implement torch._scaled_groupe | Stonepia, liangan1 | P2 | UT issue with few failures | Others | Backend/Device Issue - Implementation required for XPU backend | module: quant |  | low_precision | ut |
| [2967](https://github.com/intel/torch-xpu-ops/issues/2967) | [distributed] feature gaps in  | frost-intel | P2 | UT issue with few failures | Distributed | Distributed/Gloo Issue - feature gaps in distributed testing for XPU with test_c | bug, module: distributed |  | distributed | ut |
| [2659](https://github.com/intel/torch-xpu-ops/issues/2659) | [distributed] test_dist2.py Ru | Chao1Han | P2 | UT issue with few failures | Distributed | Distributed/Gloo Issue - Backend xccl does not implement getBackendOptions. | module: distributed |  | distributed | ut |
| [2376](https://github.com/intel/torch-xpu-ops/issues/2376) | [Bug Skip]: NotImplementedErro | CuiYifeng | P2 | UT issue with few failures | Dtype / Precision Related | Dtype/Precision Issue - "logaddexp_xpu" not implemented for 'Complex' dtype | module: ut, skipped |  | aten_ops | ut |
| [2207](https://github.com/intel/torch-xpu-ops/issues/2207) | Enable FP8/MXFP8 Ops with requ | CuiYifeng | P2 | UT issue with few failures | TorchAO | Backend/Device Issue - FP8/MXFP8 Ops related to XPU and CUDA alignment | dtype: float8 |  | aten_ops | ut |
| [2140](https://github.com/intel/torch-xpu-ops/issues/2140) | Consider how to avoid copy in  | CuiYifeng | P2 | UT issue with few failures | Inductor / Compilation Related | Memory/Shared Memory Issue - Avoiding copy in FFT kernels relates to memory hand | enhancement |  | aten_ops | ut |
| [2127](https://github.com/intel/torch-xpu-ops/issues/2127) | Path Coverage enhancement | CuiYifeng | P2 | UT issue with few failures | Others | Skip/No Test Exists - no test or error information provided | enhancement |  | aten_ops | ut |
| [2098](https://github.com/intel/torch-xpu-ops/issues/2098) | Upstream XPU functions in yaml | guangyey | P2 | UT issue with few failures | Others | Backend/Device Issue - XPU functions in yaml related to upstream backend issues | enhancement |  | aten_ops | ut |
| [2086](https://github.com/intel/torch-xpu-ops/issues/2086) | nd_item::barrier has been depr | dvrogozh | P2 | UT issue with few failures | Others | Backend/Device Issue - nd_item::barrier is deprecated on XPU backend. | enhancement |  | aten_ops | ut |
| [2063](https://github.com/intel/torch-xpu-ops/issues/2063) | Avoid using out-of-date term | CuiYifeng | P2 | UT issue with few failures | Others | Skip/No Test Exists - no test or error traceback provided | enhancement |  | aten_ops | ut |
| [1936](https://github.com/intel/torch-xpu-ops/issues/1936) | implement torch.linalg.cholesk | mwiktor-intel | P2 | UT issue with few failures | Others | Backend/Device Issue - XPU backend for torch.linalg.cholesky is not implemented | module: op impl, bug_fix_stage5 |  | aten_ops | ut |
| [1902](https://github.com/intel/torch-xpu-ops/issues/1902) | implement torch.linalg.pinv xp | mwiktor-intel | P2 | UT issue with few failures | Others | Backend/Device Issue - XPU backend for torch.linalg.pinv is not implemented | module: op impl, bug_fix_stage5 |  | aten_ops | ut |
| [1901](https://github.com/intel/torch-xpu-ops/issues/1901) | implement torch.linalg.svd xpu | CuiYifeng | P2 | UT issue with few failures | Others | Backend/Device Issue - XPU backend for torch.linalg.svd not implemented | module: op impl |  | aten_ops | ut |
| [1900](https://github.com/intel/torch-xpu-ops/issues/1900) | implement torch.linalg.qr xpu  | pbielak | P2 | UT issue with few failures | Others | Backend/Device Issue - XPU backend implementation missing for torch.linalg.qr | module: op impl, bug_fix_stage3 |  | aten_ops | ut |
| [1574](https://github.com/intel/torch-xpu-ops/issues/1574) | The operator 'aten::_grouped_m | Stonepia | P2 | UT issue with few failures | Others | Backend/Device Issue - aten::_grouped_mm not implemented for XPU device | module: ao |  | AO | ut |
| [208](https://github.com/intel/torch-xpu-ops/issues/208) | Abstract utility functions use | CuiYifeng | P2 | UT issue with few failures | Others | Others - abstract utility functions in ATen operator implementation | enhancement, module: op impl, long term |  | aten_ops | ut |
| [146](https://github.com/intel/torch-xpu-ops/issues/146) | Evaluate register spill in SYC | CuiYifeng, jianyizh, mengfei25 | P2 | UT issue with few failures | Inductor / Compilation Related | Backend/Device Issue - register spill evaluation in SYCL kernel on XPU | enhancement |  | aten_ops | ut |

---

## 5. Recent Issues (Last 10 Days)

Issues created in the last 10 days (as of 2026-04-08).

| ID | Title | Status | Owner | Priority | Reason | Category | Root Cause | Labels | Module | Test Module |
|---|-------|--------|-------|---------|--------|----------|-----------|--------|--------|-------------|
| [3259](https://github.com/intel/torch-xpu-ops/issues/3259) | New failed test cases 2026-04-02 | open | SlawomirLaba | P2 | UT issue with few failures | Flash Attention / Transformer Related | Backend/Device Issue - XPU device initialization or compatibility failure | skipped | aten_ops | ut |
| [3258](https://github.com/intel/torch-xpu-ops/issues/3258) | Error in op: torch.ops.aten._scaled_dot_ | open | None | P2 | UT issue with few failures | Flash Attention / Transformer Related | 5 - Flash Attention/Specific Ops Issue - Error in _scaled_dot_product_fused_atte |  | aten_ops | ut |
| [3257](https://github.com/intel/torch-xpu-ops/issues/3257) | [Linux][E2E][Regression] Huggingface tes | open | None | P0 | Impacts real model/application | Others | Backend/Device Issue - Huggingface test models failed to run on XPU, indicating  |  | aten_ops | e2e |
| [3255](https://github.com/intel/torch-xpu-ops/issues/3255) | [Linux][PT2E][Regression] Some performan | open | None | P0 | Regression - passed before but failed now | TorchAO | Timeout/Performance Issue - performance tests failed due to regression in execut |  | aten_ops | e2e |
| [3247](https://github.com/intel/torch-xpu-ops/issues/3247) | NotImplementedError: "dot_xpu_mkl" not i | open | Silv3S | P2 | UT issue with few failures | Others |  | ut_upstream | aten_ops | ut |
| [3246](https://github.com/intel/torch-xpu-ops/issues/3246) | AssertionError: Booleans mismatch: True  | open | BartoszKokoszko | P2 | UT issue with few failures | Distributed |  | skipped | aten_ops | ut |
| [3243](https://github.com/intel/torch-xpu-ops/issues/3243) | AssertionError: False is not true | open | pponikox | P2 | UT issue with few failures | Dtype / Precision Related | Failure - assertion 'False is not true' failed in test | module: ut, skipped | aten_ops | ut |
| [3242](https://github.com/intel/torch-xpu-ops/issues/3242) | AssertionError: Torch not compiled with  | open | None | P2 | UT issue with few failures | Inductor / Compilation Related |  | module: ut, skipped | aten_ops | ut |
| [3238](https://github.com/intel/torch-xpu-ops/issues/3238) | The supported dtypes of _refs.stft is no | open | CuiYifeng | P2 | UT issue with few failures | Dtype / Precision Related |  | ut_upstream | aten_ops | ut |
| [3236](https://github.com/intel/torch-xpu-ops/issues/3236) | AssertionError: 'def [28 chars]n unbind  | open | jmamzax | P2 | UT issue with few failures | Dtype / Precision Related | Failure - mismatch in expected IR code for XPU backend operations | bug_fix_stage5 | aten_ops | ut |
| [3233](https://github.com/intel/torch-xpu-ops/issues/3233) | [distributed] RuntimeError: No backend f | open | None | P2 | UT issue with few failures | Distributed | Distributed/Gloo Issue - No backend for the parent process group or its backend  | bug, module: distributed | distributed | ut |
| [3232](https://github.com/intel/torch-xpu-ops/issues/3232) | [distributed][tensor] AssertionError: As | open | None | P2 | UT issue with few failures | Distributed | Failure - AssertionError not raised for Placement (Shard(dim=2),) in test | bug, module: distributed | distributed | ut |
| [3231](https://github.com/intel/torch-xpu-ops/issues/3231) | Dynamo failed to run FX node with fake t | open | None | P2 | UT issue with few failures | PT2E | Inductor/Compilation Issue - Dynamo failed to compile FX node with fake tensors  | module: ut, skipped | aten_ops | ut |
| [3229](https://github.com/intel/torch-xpu-ops/issues/3229) | RuntimeError: No viable backend for scal | open | None | P2 | UT issue with few failures | Flash Attention / Transformer Related |  | skipped, ut_upstream | aten_ops | ut |
| [3227](https://github.com/intel/torch-xpu-ops/issues/3227) | torch xpu event has ~0.1ms latency, whic | open | guangyey | P2 | UT issue with few failures | Others | Timeout/Performance Issue - high latency in XPU event (~0.1ms) indicates a perfo |  | aten_ops | ut |
| [3224](https://github.com/intel/torch-xpu-ops/issues/3224) | [Win][Build] Building SYCL (Device) obje | open | chunhuanMeng | P0 | Build crash - critical blocking issue | Inductor / Compilation Related | Backend/Device Issue - SYCL kernel build failure on Windows for XPU |  | aten_ops | build |
| [3216](https://github.com/intel/torch-xpu-ops/issues/3216) | [OPs] Some ops of XPU have non-determini | open | CuiYifeng | P2 | UT issue with few failures | Others | Backend/Device Issue - XPU ops show non-determinism and inconsistency compared t |  | aten_ops | ut |
| [3209](https://github.com/intel/torch-xpu-ops/issues/3209) | [Win][Build] There is Cyclic dependencie | open | Copilot | P0 | Build crash - critical blocking issue | Others | Backend/Device Issue - Cyclic dependencies during build with BUILD_SEPARATE_OPS= |  | aten_ops | build |
| [3206](https://github.com/intel/torch-xpu-ops/issues/3206) | [Bug Skip]: [new failures]RuntimeError:  | open | tszulist-hbn | P2 | UT issue with few failures | Others | Backend/Device Issue - Tensors are on different devices (xpu:0 vs cpu) | skipped | aten_ops | ut |
