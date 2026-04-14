# Torch XPU Ops Issue Report

**Repository:** [intel/torch-xpu-ops](https://github.com/intel/torch-xpu-ops)

**CI Data Sources:**
- Torch-XPU-OPS Nightly CI: XML files from `Inductor-XPU-UT-Data-*` + `Inductor-wheel-nightly-LTS2-XPU-E2E-Data-*`
- Stock PyTorch XPU CI: XML files from `test-default-*-linux.idc.xpu_*.zip`

**Generated:** 2026-04-14 09:32:16
**Total Issues:** 374 (excluded 10 enhancement issues)

---

## <span id='toc'>Index</span>

- [1. Summary (#1-summary)](#1-summary) - 374 issues |
- [2. Need Investigation by Category (#2-need-investigation-by-category)](#2-need-investigation-by-category) - 336 issues |
   - [unknown](#unknown) - 68 issues |
      - [Implement XPU kernel or backend support](#unknown-implement-xpu-kernel-or-backend-support) - 30 issues |
      - [Fix memory management on XPU](#unknown-fix-memory-management-on-xpu) - 6 issues |
      - [Investigate and fix test failure](#unknown-investigate-and-fix-test-failure) - 6 issues |
      - [Fix distributed backend for XPU](#unknown-fix-distributed-backend-for-xpu) - 6 issues |
      - [Fix dtype/precision issue](#unknown-fix-dtype-precision-issue) - 5 issues |
      - [Add to skiplist](#unknown-add-to-skiplist) - 4 issues |
      - [Fix precision/accuracy issue](#unknown-fix-precision-accuracy-issue) - 4 issues |
      - [Fix attention/SDPA operation on XPU](#unknown-fix-attention-sdpa-operation-on-xpu) - 2 issues |
      - [Fix error handling](#unknown-fix-error-handling) - 2 issues |
      - [Unknown Action](#unknown-unknown-action) - 2 issues |
      - [Fix performance issue](#unknown-fix-performance-issue) - 1 issues |
   - [Torch Operations](#torch-operations) - 54 issues |
   - [Dtype/Precision](#dtype-precision) - 47 issues |
   - [Others](#others) - 30 issues |
   - [Inductor/Compilation](#inductor-compilation) - 25 issues |
   - [Distributed](#distributed) - 24 issues |
   - [Flash Attention/Transformer](#flash-attention-transformer) - 23 issues |
   - [TorchAO](#torchao) - 21 issues |
   - [Sparse](#sparse) - 13 issues |
   - [Feature Not Supported](#feature-not-supported) - 10 issues |
   - [Torch Runtime](#torch-runtime) - 8 issues |
   - [PT2E](#pt2e) - 4 issues |
   - [Skip/No Test Exists](#skip-no-test-exists) - 3 issues |
   - [Build/Compilation](#build-compilation) - 2 issues |
   - [Performance](#performance) - 2 issues |
   - [Accuracy](#accuracy) - 1 issues |
   - [Profiler](#profiler) - 1 issues |
- [3. Other Actions by Type (#3-other-actions-by-type)](#3-other-actions-by-type) - 38 issues |
   - [Close fixed issue](#close-fixed-issue) - 17 issues |
   - [Revisit the PR as case failed](#revisit-the-pr-as-case-failed) - 2 issues |
   - [add to skiplist](#add-to-skiplist) - 5 issues |
   - [Verify the issue](#verify-the-issue) - 14 issues |
- [4. Duplicated Issues (#4-duplicated-issues)](#4-duplicated-issues) - 38 issues |
- [5. Issues with Dependency (#5-issues-with-dependency)](#5-issues-with-dependency) - 35 issues |
- [6. Statistics (#6-statistics)](#6-statistics) - Dependency stats |

---

## <span id='1-summary'>1. Summary</span>

**Total: 374 issues** (excluded 10 enhancement issues)

| # | Action Type | Count | Link |
|--:|-------------|-------|------|
| 1 | [Need Investigation](#need-investigation) | 336 | [View Issues](#need-investigation) |
| 2 | [Close fixed issue](#close-fixed-issue) | 17 | [View Issues](#close-fixed-issue) |
| 3 | [Verify the issue](#verify-the-issue) | 14 | [View Issues](#verify-the-issue) |
| 4 | [add to skiplist](#add-to-skiplist) | 5 | [View Issues](#add-to-skiplist) |
| 5 | [Revisit the PR as case failed](#revisit-the-pr-as-case-failed) | 2 | [View Issues](#revisit-the-pr-as-case-failed) |
| | **Total** | **374** | |

## <span id='2-need-investigation-by-category'>2. Need Investigation by Category</span>

**Total: 336 issues** - Issues requiring further investigation

### <span id='unknown'>unknown</span> (68 issues)

**Grouped by Action Type:**

#### <span id='unknown-implement-xpu-kernel-or-backend-support'>Implement XPU kernel or backend support</span> (30 issues)
| # | ID | Title | Action Reason | Summary | Assignee | Owner Transfer | PR Status | Test Module |
|--:|----|-------|---------------|---------|----------|----------------|-----------|-------------|
| 1 | 2230 | test_sparse_csr.py::TestSparseCompressedTritonKernelsXPU::test_triton_ meet ValueError: all inputs are expected to be on the same GPU device. | Implement XPU kernel or backend support: Add proper XPU dispatch in native_functions.yaml or implement the missing kernel in the appropriate aten module. | test_sparse_csr.py::TestSparseCompressedTritonKernelsXPU::test_triton_ meet ValueError: all inputs are expected to be on the same GPU device. | None | None |  | ut |
| 2 | 2220 | test/test_sparse_csr.py::TestSparseCompressedTritonKernelsXPU::test_triton_bsr_scatter_mm_blocksize_16_xpu_bfloat16 will meet InvalidModule: Invalid SPIR-V module: input SPIR-V module uses unknown extension 'SPV_INTEL_subgroup_matrix_multiply_accumulate' | Implement XPU kernel or backend support: Add proper XPU dispatch in native_functions.yaml or implement the missing kernel in the appropriate aten module. | test/test_sparse_csr.py::TestSparseCompressedTritonKernelsXPU::test_triton_bsr_scatter_mm_blocksize_16_xpu_bfloat16 will meet InvalidModule: Invalid S | None | None |  | ut |
| 3 | 2215 | Find use case example for torch-xpu-ops.lib in sycl cpp extension | Implement XPU kernel or backend support: Add proper XPU dispatch in native_functions.yaml or implement the missing kernel in the appropriate aten module. | Find use case example for torch-xpu-ops.lib in sycl cpp extension | dvrogozh | dvrogozh |  | ut |
| 4 | 2214 | test/test_sparse.py::TestSparseAnyXPU::test_gradcheck_mm expected error message not match | Implement XPU kernel or backend support: Add proper XPU dispatch in native_functions.yaml or implement the missing kernel in the appropriate aten module. | test/test_sparse.py::TestSparseAnyXPU::test_gradcheck_mm expected error message not match | jenniew | jenniew |  | ut |
| 5 | 2142 | XPU max_memory_allocated have different output with CUDA | Implement XPU kernel or backend support: Add proper XPU dispatch in native_functions.yaml or implement the missing kernel in the appropriate aten module. | XPU max_memory_allocated have different output with CUDA | guangyey | guangyey |  | ut |
| 6 | 2132 | [2.9][BMG-Windows][Torch-xpu-ops UT] 1 case failed with "AssertionError: Torch not compiled with CUDA enabled " | Implement XPU kernel or backend support: Add proper XPU dispatch in native_functions.yaml or implement the missing kernel in the appropriate aten module. | [2.9][BMG-Windows][Torch-xpu-ops UT] 1 case failed with "AssertionError: Torch not compiled with CUDA enabled " | pbielak | pbielak |  | ut |
| 7 | 2089 | need an implementation that won't initialize gpu context for torch.xpu.is_available() | Implement XPU kernel or backend support: Add proper XPU dispatch in native_functions.yaml or implement the missing kernel in the appropriate aten module. | need an implementation that won't initialize gpu context for torch.xpu.is_available() | guangyey | guangyey |  | ut |
| 8 | 2024 | AssertionError: Torch not compiled with CUDA enabled | Implement XPU kernel or backend support: Add proper XPU dispatch in native_functions.yaml or implement the missing kernel in the appropriate aten module. | AssertionError: Torch not compiled with CUDA enabled | daisyden | daisyden |  | ut |
| 9 | 2015 | inf is returned by nn.TransformerEncoderLayer | Implement XPU kernel or backend support: Add proper XPU dispatch in native_functions.yaml or implement the missing kernel in the appropriate aten module. | inf is returned by nn.TransformerEncoderLayer | yucai-intel | yucai-intel | open;merged | ut |
| 10 | 2006 | work-item/workgroup issue in softmax/unsampling/nonzero | Implement XPU kernel or backend support: Add proper XPU dispatch in native_functions.yaml or implement the missing kernel in the appropriate aten module. | work-item/workgroup issue in softmax/unsampling/nonzero | BartoszKokoszko | BartoszKokoszko | open;merged | ut |
| 11 | 1986 | torch.xpu._sleep is missing, | Implement XPU kernel or backend support: Add proper XPU dispatch in native_functions.yaml or implement the missing kernel in the appropriate aten module. | torch.xpu._sleep is missing, | guangyey | guangyey |  | ut |
| 12 | 1970 | torch._dynamo.exc.BackendCompilerFailed: backend='inductor' raised: RuntimeError: CUDA not available | Implement XPU kernel or backend support: Add proper XPU dispatch in native_functions.yaml or implement the missing kernel in the appropriate aten module. | torch._dynamo.exc.BackendCompilerFailed: backend='inductor' raised: RuntimeError: CUDA not available | None | None |  | ut |
| 13 | 1951 | Functionality issues in TestCommon.test_out. | Implement XPU kernel or backend support: Add proper XPU dispatch in native_functions.yaml or implement the missing kernel in the appropriate aten module. | Functionality issues in TestCommon.test_out. | AKloniecki | AKloniecki |  | ut |
| 14 | 1936 | implement torch.linalg.cholesky xpu backend | Implement XPU kernel or backend support: Add proper XPU dispatch in native_functions.yaml or implement the missing kernel in the appropriate aten module. | implement torch.linalg.cholesky xpu backend | mwiktor-intel | mwiktor-intel |  | ut |
| 15 | 1912 | Implement the torch.ops.aten._weight_int4pack_mm for dequantizing the  CUDA int4 layout | Implement XPU kernel or backend support: Add proper XPU dispatch in native_functions.yaml or implement the missing kernel in the appropriate aten module. | Implement the torch.ops.aten._weight_int4pack_mm for dequantizing the  CUDA int4 layout | liangan1 | liangan1 |  | ut |
| 16 | 1902 | implement torch.linalg.pinv xpu backend | Implement XPU kernel or backend support: Add proper XPU dispatch in native_functions.yaml or implement the missing kernel in the appropriate aten module. | implement torch.linalg.pinv xpu backend | mwiktor-intel | mwiktor-intel |  | ut |
| 17 | 1901 | implement torch.linalg.svd xpu backend | Implement XPU kernel or backend support: Add proper XPU dispatch in native_functions.yaml or implement the missing kernel in the appropriate aten module. | implement torch.linalg.svd xpu backend | CuiYifeng | CuiYifeng |  | ut |
| 18 | 1900 | implement torch.linalg.qr xpu backend | Implement XPU kernel or backend support: Add proper XPU dispatch in native_functions.yaml or implement the missing kernel in the appropriate aten module. | implement torch.linalg.qr xpu backend | pbielak | pbielak |  | ut |
| 19 | 1784 | [Performance] Torch XPU Profiler is not reliable | Implement XPU kernel or backend support: Add proper XPU dispatch in native_functions.yaml or implement the missing kernel in the appropriate aten module. | [Performance] Torch XPU Profiler is not reliable | jfedorov, aostrowski-hbn | jfedorov, aostrowski-hbn |  | ut |
| 20 | 1762 | Add an ocloc AOT target compilation test in cmake | Implement XPU kernel or backend support: Add proper XPU dispatch in native_functions.yaml or implement the missing kernel in the appropriate aten module. | Add an ocloc AOT target compilation test in cmake | chunhuanMeng | chunhuanMeng |  | ut |
| 21 | 1749 | transformers UT failure in XPU because SDPA check error "Backward or grad to be supported" | Implement XPU kernel or backend support: Add proper XPU dispatch in native_functions.yaml or implement the missing kernel in the appropriate aten module. | transformers UT failure in XPU because SDPA check error "Backward or grad to be supported" | LuFinch | LuFinch |  | ut |
| 22 | 1727 | [distributed] AttributeError: module 'torch.xpu' has no attribute '_sleep' | Implement XPU kernel or backend support: Add proper XPU dispatch in native_functions.yaml or implement the missing kernel in the appropriate aten module. | [distributed] AttributeError: module 'torch.xpu' has no attribute '_sleep' | guangyey | guangyey |  | ut |
| 23 | 1722 | Ask an API to query GPU type(iGPU/dGPU). | Implement XPU kernel or backend support: Add proper XPU dispatch in native_functions.yaml or implement the missing kernel in the appropriate aten module. | Ask an API to query GPU type(iGPU/dGPU). | guangyey | guangyey |  | ut |
| 24 | 1649 | [cpp extension] Provide a clear error message when using inconsistent oneapi versions. | Implement XPU kernel or backend support: Add proper XPU dispatch in native_functions.yaml or implement the missing kernel in the appropriate aten module. | [cpp extension] Provide a clear error message when using inconsistent oneapi versions. | dvrogozh | dvrogozh |  | ut |
| 25 | 1645 | [For Comparison] Save reference comparison run id | Implement XPU kernel or backend support: Add proper XPU dispatch in native_functions.yaml or implement the missing kernel in the appropriate aten module. | [For Comparison] Save reference comparison run id | mengfei25 | mengfei25 |  | ut |
| 26 | 1574 | The operator 'aten::_grouped_mm' is not currently implemented for the XPU device. | Implement XPU kernel or backend support: Add proper XPU dispatch in native_functions.yaml or implement the missing kernel in the appropriate aten module. | The operator 'aten::_grouped_mm' is not currently implemented for the XPU device. | Stonepia, LuFinch | Stonepia, LuFinch |  | ut |
| 27 | 1551 | [distributed] NotImplementedError: The operator 'symm_mem::fused_scaled_matmul_reduce_scatter' is not currently implemented for the XPU device. | Implement XPU kernel or backend support: Add proper XPU dispatch in native_functions.yaml or implement the missing kernel in the appropriate aten module. | [distributed] NotImplementedError: The operator 'symm_mem::fused_scaled_matmul_reduce_scatter' is not currently implemented for the XPU device. | Chao1Han | Chao1Han |  | ut |
| 28 | 1547 | [distributed] NotImplementedError: The operator 'symm_mem::fused_matmul_reduce_scatter' is not currently implemented for the XPU device | Implement XPU kernel or backend support: Add proper XPU dispatch in native_functions.yaml or implement the missing kernel in the appropriate aten module. | [distributed] NotImplementedError: The operator 'symm_mem::fused_matmul_reduce_scatter' is not currently implemented for the XPU device | Chao1Han | Chao1Han |  | ut |
| 29 | 1059 | SYCL RT: Using recommended shortcut API for kernel specific max work group size. | Implement XPU kernel or backend support: Add proper XPU dispatch in native_functions.yaml or implement the missing kernel in the appropriate aten module. | SYCL RT: Using recommended shortcut API for kernel specific max work group size. | CuiYifeng, jianyizh | CuiYifeng, jianyizh |  | ut |
| 30 | 489 | Moco NotImplementedError: xpu not supported | Implement XPU kernel or backend support: Add proper XPU dispatch in native_functions.yaml or implement the missing kernel in the appropriate aten module. | Moco NotImplementedError: xpu not supported | weishi-deng | weishi-deng |  | e2e |
| | | **Subtotal: 30 issues** | | | | | |

#### <span id='unknown-fix-memory-management-on-xpu'>Fix memory management on XPU</span> (6 issues)
| # | ID | Title | Action Reason | Summary | Assignee | Owner Transfer | PR Status | Test Module |
|--:|----|-------|---------------|---------|----------|----------------|-----------|-------------|
| 1 | 2232 | sdpa backward kernel is required to reduce memory usage | Fix memory management on XPU: Check memory allocation/deallocation in the operation implementation. | sdpa backward kernel is required to reduce memory usage | None | None |  | ut |
| 2 | 2004 | [distributed][shared_tensor] test\distributed\_shard\shared_tensor\test_sharded_tensor.py has 12 cases failed with "RuntimeError: eof (this error originated at tensorpipe/transport/shm/connection_impl.cc:259)" | Fix memory management on XPU: Check memory allocation/deallocation in the operation implementation. | [distributed][shared_tensor] test\distributed\_shard\shared_tensor\test_sharded_tensor.py has 12 cases failed with "RuntimeError: eof (this error orig | libohao1201 | libohao1201 |  | ut |
| 3 | 1996 | [TorchAO]  Memory Efficient Optimizers | Fix memory management on XPU: Check memory allocation/deallocation in the operation implementation. | [TorchAO]  Memory Efficient Optimizers | None | None |  | ut |
| 4 | 1856 | channel last aten::hardswish_ will call extra copy | Fix memory management on XPU: Check memory allocation/deallocation in the operation implementation. | channel last aten::hardswish_ will call extra copy | chunhuanMeng | chunhuanMeng |  | ut |
| 5 | 1678 | missing op support for `model.share_memory()` | Fix memory management on XPU: Check memory allocation/deallocation in the operation implementation. | missing op support for `model.share_memory()` | None | None |  | ut |
| 6 | 1324 | [Win] UR Error when OOM and break the tensor context | Fix memory management on XPU: Check memory allocation/deallocation in the operation implementation. | [Win] UR Error when OOM and break the tensor context | Stonepia | Stonepia |  | ut |
| | | **Subtotal: 6 issues** | | | | | |

#### <span id='unknown-investigate-and-fix-test-failure'>Investigate and fix test failure</span> (6 issues)
| # | ID | Title | Action Reason | Summary | Assignee | Owner Transfer | PR Status | Test Module |
|--:|----|-------|---------------|---------|----------|----------------|-----------|-------------|
| 1 | 2229 | test/test_sparse_csr.py::TestSparseCompressedCPU::test_invalid_input meet message not match | Investigate and fix test failure: Analyze traceback and implement proper fix in the operation implementation or test. | test/test_sparse_csr.py::TestSparseCompressedCPU::test_invalid_input meet message not match | jenniew | jenniew |  | ut |
| 2 | 2201 | [TorchAO][BMG] When using paged attention backend, all cases failed with "assert vr is not None" | Investigate and fix test failure: Analyze traceback and implement proper fix in the operation implementation or test. | [TorchAO][BMG] When using paged attention backend, all cases failed with "assert vr is not None" | Stonepia | Stonepia |  | ut |
| 3 | 2186 | AssertionError: Mul tiheadAttention does not support NestedTensor outside of its fast path | Investigate and fix test failure: Analyze traceback and implement proper fix in the operation implementation or test. | AssertionError: Mul tiheadAttention does not support NestedTensor outside of its fast path | daisyden | daisyden |  | ut |
| 4 | 2182 | test_transform_bias_rescale_qkv_nested_xpu_float32 failed with AssertionError: Scalars are not equal! | Investigate and fix test failure: Analyze traceback and implement proper fix in the operation implementation or test. | test_transform_bias_rescale_qkv_nested_xpu_float32 failed with AssertionError: Scalars are not equal! | SlawomirLaba, PawelSwider2000 | SlawomirLaba, PawelSwider2000 |  | ut |
| 5 | 2165 | [distributed] test_device_mesh.py::TestDeviceMeshGetItem::test_flatten_mesh_3d AssertionError | Investigate and fix test failure: Analyze traceback and implement proper fix in the operation implementation or test. | [distributed] test_device_mesh.py::TestDeviceMeshGetItem::test_flatten_mesh_3d AssertionError | jemitche1 | jemitche1 |  | ut |
| 6 | 1973 | AssertionError: Scalars or Tensor-likes are not equal or close! | Investigate and fix test failure: Analyze traceback and implement proper fix in the operation implementation or test. | AssertionError: Scalars or Tensor-likes are not equal or close! | gplutop7 | gplutop7 |  | ut |
| | | **Subtotal: 6 issues** | | | | | |

#### <span id='unknown-fix-distributed-backend-for-xpu'>Fix distributed backend for XPU</span> (6 issues)
| # | ID | Title | Action Reason | Summary | Assignee | Owner Transfer | PR Status | Test Module |
|--:|----|-------|---------------|---------|----------|----------------|-----------|-------------|
| 1 | 2163 | 3 distributed UT cases need to be supported by - https://github.com/pytorch/pytorch/blob/main/torch/distributed/_tools/sac_estimator.py | Fix distributed backend for XPU: Update process group initialization for XPU in torch/distributed/distributed_c10d.py or implement XPU-compatible collective operations. | 3 distributed UT cases need to be supported by - https://github.com/pytorch/pytorch/blob/main/torch/distributed/_tools/sac_estimator.py | githubsgi | githubsgi |  | ut |
| 2 | 1661 | [distributed] Accuracy gap in _composable/fsdp on Xelink | Fix distributed backend for XPU: Update process group initialization for XPU in torch/distributed/distributed_c10d.py or implement XPU-compatible collective operations. | [distributed] Accuracy gap in _composable/fsdp on Xelink | githubsgi | githubsgi |  | ut |
| 3 | 1571 | [distributed] ValueError: Cannot use ReduceOp.PREMUL_SUM with XCCL | Fix distributed backend for XPU: Update process group initialization for XPU in torch/distributed/distributed_c10d.py or implement XPU-compatible collective operations. | [distributed] ValueError: Cannot use ReduceOp.PREMUL_SUM with XCCL | zhangxiaoli73 | zhangxiaoli73 |  | ut |
| 4 | 1555 | [distributed] RuntimeError: aten.add.Tensor: got mixed torch.Tensor and DTensor, need to convert all torch.Tensor to DTensor before calling distributed operators! | Fix distributed backend for XPU: Update process group initialization for XPU in torch/distributed/distributed_c10d.py or implement XPU-compatible collective operations. | [distributed] RuntimeError: aten.add.Tensor: got mixed torch.Tensor and DTensor, need to convert all torch.Tensor to DTensor before calling distribute | chuanqi129 | chuanqi129 |  | ut |
| 5 | 1549 | [distributed] AssertionError: 'fused_all_gather_scaled_matmul' not found in 'graph():\n......' | Fix distributed backend for XPU: Update process group initialization for XPU in torch/distributed/distributed_c10d.py or implement XPU-compatible collective operations. | [distributed] AssertionError: 'fused_all_gather_scaled_matmul' not found in 'graph():\n......' | Chao1Han | Chao1Han |  | ut |
| 6 | 1548 | [distributed] AssertionError: 'fused_all_gather_matmul' not found in '# AOT ID: [\'2_inference\']\n......' | Fix distributed backend for XPU: Update process group initialization for XPU in torch/distributed/distributed_c10d.py or implement XPU-compatible collective operations. | [distributed] AssertionError: 'fused_all_gather_matmul' not found in '# AOT ID: [\'2_inference\']\n......' | Chao1Han | Chao1Han |  | ut |
| | | **Subtotal: 6 issues** | | | | | |

#### <span id='unknown-fix-dtype-precision-issue'>Fix dtype/precision issue</span> (5 issues)
| # | ID | Title | Action Reason | Summary | Assignee | Owner Transfer | PR Status | Test Module |
|--:|----|-------|---------------|---------|----------|----------------|-----------|-------------|
| 1 | 2207 | Enable FP8/MXFP8 Ops with requests and CUDA alignment | Fix dtype/precision issue: Adjust tolerance in test file or implement proper dtype handling in the operation kernel for XPU. | Enable FP8/MXFP8 Ops with requests and CUDA alignment | Stonepia, CuiYifeng, LuFinch | Stonepia, CuiYifeng, LuFinch |  | ut |
| 2 | 1894 | [Linux][PT2E] performance test got failed, int8 ASYMM and int8 SYMM | Fix dtype/precision issue: Adjust tolerance in test file or implement proper dtype handling in the operation kernel for XPU. | [Linux][PT2E] performance test got failed, int8 ASYMM and int8 SYMM | jenniew | jenniew |  | e2e |
| 3 | 1866 | [release 2.8]Torchbench vision_maskrcnn (amp_b16/amp_fp16 inference) got fail_accuracy | Fix dtype/precision issue: Adjust tolerance in test file or implement proper dtype handling in the operation kernel for XPU. | [release 2.8]Torchbench vision_maskrcnn (amp_b16/amp_fp16 inference) got fail_accuracy | BartoszKokoszko | BartoszKokoszko |  | e2e |
| 4 | 1818 | [BMG-Windows][PT2.8]Torch-xpu-ops UT got accuracy issue | Fix dtype/precision issue: Adjust tolerance in test file or implement proper dtype handling in the operation kernel for XPU. | [BMG-Windows][PT2.8]Torch-xpu-ops UT got accuracy issue | kdrozd-dev | kdrozd-dev | open | ut |
| 5 | 1159 | [LNL Windows][Test by CD Nightly Wheels] hugging face model - DebertaForQuestionAnswering && DebertaV2ForMaskedLM failed with RuntimeError: value cannot be converted to type at::BFloat16 without overflow   | Fix dtype/precision issue: Adjust tolerance in test file or implement proper dtype handling in the operation kernel for XPU. | [LNL Windows][Test by CD Nightly Wheels] hugging face model - DebertaForQuestionAnswering && DebertaV2ForMaskedLM failed with RuntimeError: value cann | Stonepia | Stonepia |  | e2e |
| | | **Subtotal: 5 issues** | | | | | |

#### <span id='unknown-add-to-skiplist'>Add to skiplist</span> (4 issues)
| # | ID | Title | Action Reason | Summary | Assignee | Owner Transfer | PR Status | Test Module |
|--:|----|-------|---------------|---------|----------|----------------|-----------|-------------|
| 1 | 2169 | Frame size comparison failed in test_size_comparison_no_recompile | Add to skiplist: Mark test as xfail or add skip decorator in the test file for XPU if not applicable. | Frame size comparison failed in test_size_comparison_no_recompile | guangyey | guangyey |  | ut |
| 2 | 2113 | Update example for Distributed Data Parallel | Add to skiplist: Mark test as xfail or add skip decorator in the test file for XPU if not applicable. | Update example for Distributed Data Parallel | songhappy | songhappy |  | ut |
| 3 | 1729 | Validation Check List | Add to skiplist: Mark test as xfail or add skip decorator in the test file for XPU if not applicable. | Validation Check List | chuanqi129 | chuanqi129 |  | ut |
| 4 | 1689 | [For op Perf Comparison] Save reference comparison run id | Add to skiplist: Mark test as xfail or add skip decorator in the test file for XPU if not applicable. | [For op Perf Comparison] Save reference comparison run id | None | None |  | ut |
| | | **Subtotal: 4 issues** | | | | | |

#### <span id='unknown-fix-precision-accuracy-issue'>Fix precision/accuracy issue</span> (4 issues)
| # | ID | Title | Action Reason | Summary | Assignee | Owner Transfer | PR Status | Test Module |
|--:|----|-------|---------------|---------|----------|----------------|-----------|-------------|
| 1 | 2128 | [2.9][BMG-Windows][Torchbench] speeach_transforer accuracy_training failed with Exception Code: 0xC0000005 when using torchbench pinned by pytorch2.8 | Fix precision/accuracy issue: Adjust tolerance or implement proper dtype handling for XPU. | [2.9][BMG-Windows][Torchbench] speeach_transforer accuracy_training failed with Exception Code: 0xC0000005 when using torchbench pinned by pytorch2.8 | chuanqi129 | chuanqi129 |  | ut |
| 2 | 1893 | [upstream_ut] oneDNN accuracy issues in test_ops_xpu.py | Fix precision/accuracy issue: Adjust tolerance or implement proper dtype handling for XPU. | [upstream_ut] oneDNN accuracy issues in test_ops_xpu.py | chunhuanMeng | chunhuanMeng |  | ut |
| 3 | 1877 | Torchbench model squeezenet1_1 and functorch_dp_cifar10  got fail_accuracy | Fix precision/accuracy issue: Adjust tolerance or implement proper dtype handling for XPU. | Torchbench model squeezenet1_1 and functorch_dp_cifar10  got fail_accuracy | Silv3S | Silv3S |  | ut |
| 4 | 1505 | [ARC-WSL-Ubuntu24.04] 15 Timm models got fail_accuracy | Fix precision/accuracy issue: Adjust tolerance or implement proper dtype handling for XPU. | [ARC-WSL-Ubuntu24.04] 15 Timm models got fail_accuracy | None | None |  | e2e |
| | | **Subtotal: 4 issues** | | | | | |

#### <span id='unknown-fix-attention-sdpa-operation-on-xpu'>Fix attention/SDPA operation on XPU</span> (2 issues)
| # | ID | Title | Action Reason | Summary | Assignee | Owner Transfer | PR Status | Test Module |
|--:|----|-------|---------------|---------|----------|----------------|-----------|-------------|
| 1 | 2200 | support flash attention op on XPU device | Fix attention/SDPA operation on XPU: Implement proper scaled_dot_product_attention dispatch or add XPU fallback in native_functions.yaml. | support flash attention op on XPU device | ElaineBao | ElaineBao |  | ut |
| 2 | 1556 | [distributed] NotImplementedError: Operator aten._scaled_dot_product_fused_attention_overrideable.default does not have a sharding strategy registered. | Fix attention/SDPA operation on XPU: Implement proper scaled_dot_product_attention dispatch or add XPU fallback in native_functions.yaml. | [distributed] NotImplementedError: Operator aten._scaled_dot_product_fused_attention_overrideable.default does not have a sharding strategy registered | pkourdis | pkourdis |  | ut |
| | | **Subtotal: 2 issues** | | | | | |

#### <span id='unknown-fix-error-handling'>Fix error handling</span> (2 issues)
| # | ID | Title | Action Reason | Summary | Assignee | Owner Transfer | PR Status | Test Module |
|--:|----|-------|---------------|---------|----------|----------------|-----------|-------------|
| 1 | 1969 | torch._dynamo.exc.InternalTorchDynamoError: TypeError: cannot create weak reference to 'torch.Event' object | Fix error handling: Implement proper error handling or add XPU-specific error path in the operation. | torch._dynamo.exc.InternalTorchDynamoError: TypeError: cannot create weak reference to 'torch.Event' object | guangyey | guangyey |  | ut |
| 2 | 492 | Timm_efficientdet NotImplementedError: The original model code forces the use of CUDA. | Fix error handling: Implement proper error handling or add XPU-specific error path in the operation. | Timm_efficientdet NotImplementedError: The original model code forces the use of CUDA. | weishi-deng | weishi-deng |  | e2e |
| | | **Subtotal: 2 issues** | | | | | |

#### <span id='unknown-unknown-action'>Unknown Action</span> (2 issues)
| # | ID | Title | Action Reason | Summary | Assignee | Owner Transfer | PR Status | Test Module |
|--:|----|-------|---------------|---------|----------|----------------|-----------|-------------|
| 1 | 1963 | [upstream_ut] MetadataMismatchError in TestFakeTensor of test_ops.py | This is not a PyTorch code issue - may require CI or documentation changes. | [upstream_ut] MetadataMismatchError in TestFakeTensor of test_ops.py | pbielak | pbielak | open;open | ut |
| 2 | 1587 | Keep track on the latest CUDA op impl | This is not a PyTorch code issue - may require CI or documentation changes. | Keep track on the latest CUDA op impl | CuiYifeng, yucai-intel | CuiYifeng, yucai-intel |  | ut |
| | | **Subtotal: 2 issues** | | | | | |

#### <span id='unknown-fix-performance-issue'>Fix performance issue</span> (1 issues)
| # | ID | Title | Action Reason | Summary | Assignee | Owner Transfer | PR Status | Test Module |
|--:|----|-------|---------------|---------|----------|----------------|-----------|-------------|
| 1 | 2217 | AO Performance issue track | Fix performance issue: Implement optimized XPU kernel or enable existing optimization path for the operation. | AO Performance issue track | Stonepia | Stonepia |  | ut |
| | | **Subtotal: 1 issues** | | | | | |

| | | **Category Total: 68 issues** | | | | | |

### <span id='torch-operations'>Torch Operations</span> (54 issues)

| # | ID | Title | Action Reason | Summary | Assignee | Owner Transfer | PR Status | Test Module |
|--:|----|-------|---------------|---------|----------|----------------|-----------|-------------|
| 1 | 2239 | Exception: could not create a primitive descriptor for the deconvolution forward propagation primitive. in test/functorch/test_ops.py | Fix convolution operation on XPU: Add proper backend selection or implement missing XPU kernel. | Exception: could not create a primitive descriptor for the deconvolution forward propagation primitive. in test/functorch/test_ops.py | wpietka | wpietka | open;open | ut |
| 2 | 2240 | RuntimeError: Trying to set a forward gradient that has a different size than that of the original Tensor, this is not supported. in test/functorch/test_ops.py | Fix error handling: Implement proper error handling or add XPU-specific error path in the operation. | RuntimeError: Trying to set a forward gradient that has a different size than that of the original Tensor, this is not supported. in test/functorch/te | gplutop7 | gplutop7 |  | ut |
| 3 | 2248 | [upstream_ut] test_cow failures | Implement XPU kernel or backend support: Add proper XPU dispatch in native_functions.yaml or implement the missing kernel in the appropriate aten module. | [upstream_ut] test_cow failures | gplutop7 | gplutop7 | closed;closed;open;open;merged;open | ut |
| 4 | 2358 | test/test_view_ops.py::TestOldViewOpsXPU::test_ravel_xpu meet NotImplementedError: Could not run 'aten::_empty_affine_quantized' with arguments from the 'QuantizedXPU' backend | Implement XPU kernel or backend support: Add proper XPU dispatch in native_functions.yaml or implement the missing kernel in the appropriate aten module. | test/test_view_ops.py::TestOldViewOpsXPU::test_ravel_xpu meet NotImplementedError: Could not run 'aten::_empty_affine_quantized' with arguments from t | Silv3S | Silv3S |  | ut |
| 5 | 2359 | [upstream_ut] GradcheckError: Backward is not reentrant | Implement XPU kernel or backend support: Add proper XPU dispatch in native_functions.yaml or implement the missing kernel in the appropriate aten module. | [upstream_ut] GradcheckError: Backward is not reentrant | BBBela | BBBela | open | ut |
| 6 | 2425 | [upstream_ut]  RuntimeError: Expected both self and other to be nested, but got a nested self and non-nested other
 | Fix error handling: Implement proper error handling or add XPU-specific error path in the operation. | [upstream_ut]  RuntimeError: Expected both self and other to be nested, but got a nested self and non-nested other
 | BBBela | BBBela | open | ut |
| 7 | 2436 | [upstream_ut]  AttributeError: 'NoneType' object has no attribute 'clone' 
 | Fix error handling: Implement proper error handling or add XPU-specific error path in the operation. | [upstream_ut]  AttributeError: 'NoneType' object has no attribute 'clone' 
 | daisyden | daisyden |  | ut |
| 8 | 2446 | [Bug Skip]: AssertionError: "Simulate error" does not match "grad can be implicitly created only for scalar outputs" | Investigate and fix test failure: Analyze traceback and implement proper fix in the operation implementation or test. | [Bug Skip]: AssertionError: "Simulate error" does not match "grad can be implicitly created only for scalar outputs" | BBBela | BBBela | open | ut |
| 9 | 2479 | [Bug] torch.rand output different result on bmg and pvc | Implement XPU kernel or backend support: Add proper XPU dispatch in native_functions.yaml or implement the missing kernel in the appropriate aten module. | [Bug] torch.rand output different result on bmg and pvc | Stonepia, CuiYifeng | Stonepia, CuiYifeng |  | ut |
| 10 | 2491 | [upstream_ut]  AssertionError: False is not true 
 | Investigate and fix test failure: Analyze traceback and implement proper fix in the operation implementation or test. | [upstream_ut]  AssertionError: False is not true 
 | PatrykWilczewski | PatrykWilczewski | open | ut |
| 11 | 2512 | [upstream_ut]  RuntimeError: _histc_xpu does not have a deterministic implementation, but you set 'torch.use_deter
 | Implement XPU kernel or backend support: Add proper XPU dispatch in native_functions.yaml or implement the missing kernel in the appropriate aten module. | [upstream_ut]  RuntimeError: _histc_xpu does not have a deterministic implementation, but you set 'torch.use_deter
 | chunhuanMeng | chunhuanMeng |  | ut |
| 12 | 2513 | [upstream_ut]  RuntimeError: _share_fd_: only available on CPU 
 | Implement XPU kernel or backend support: Add proper XPU dispatch in native_functions.yaml or implement the missing kernel in the appropriate aten module. | [upstream_ut]  RuntimeError: _share_fd_: only available on CPU 
 | gplutop7 | gplutop7 |  | ut |
| 13 | 2518 | [upstream_ut]  TypeError: Creating a Tensor subclass from a class that does not inherit from Tensor is not possibl
 | Fix error handling: Implement proper error handling or add XPU-specific error path in the operation. | [upstream_ut]  TypeError: Creating a Tensor subclass from a class that does not inherit from Tensor is not possibl
 | astachowiczhabana | astachowiczhabana |  | ut |
| 14 | 2519 | [upstream_ut]  TypeError: map2_ is only implemented on CPU tensors 
 | Implement XPU kernel or backend support: Add proper XPU dispatch in native_functions.yaml or implement the missing kernel in the appropriate aten module. | [upstream_ut]  TypeError: map2_ is only implemented on CPU tensors 
 | Silv3S | Silv3S |  | ut |
| 15 | 2537 | Title: [upstream_ut]  Failed: Unexpected success | Implement XPU kernel or backend support: Add proper XPU dispatch in native_functions.yaml or implement the missing kernel in the appropriate aten module. | Title: [upstream_ut]  Failed: Unexpected success | PatrykWilczewski | PatrykWilczewski |  | ut |
| 16 | 2539 | Title: [upstream_ut]  RuntimeError: Tried to instantiate dummy base class CUDAGraph | Implement XPU kernel or backend support: Add proper XPU dispatch in native_functions.yaml or implement the missing kernel in the appropriate aten module. | Title: [upstream_ut]  RuntimeError: Tried to instantiate dummy base class CUDAGraph | BBBela | BBBela |  | ut |
| 17 | 2560 | [UT] "RuntimeError: iter.device(arg).is_xpu()"  in test_torch_xpu.py | Implement XPU kernel or backend support: Add proper XPU dispatch in native_functions.yaml or implement the missing kernel in the appropriate aten module. | [UT] "RuntimeError: iter.device(arg).is_xpu()"  in test_torch_xpu.py | CuiYifeng | CuiYifeng |  | ut |
| 18 | 2639 | test_to() failed during rnn isinstance() check | Implement XPU kernel or backend support: Add proper XPU dispatch in native_functions.yaml or implement the missing kernel in the appropriate aten module. | test_to() failed during rnn isinstance() check | Silv3S | Silv3S | open | ut |
| 19 | 2670 | [upstream_ut]  RuntimeError: could not create a primitive descriptor for the deconvolution forward propagation in functorch/test_vmap.py | Fix convolution operation on XPU: Add proper backend selection or implement missing XPU kernel. | [upstream_ut]  RuntimeError: could not create a primitive descriptor for the deconvolution forward propagation in functorch/test_vmap.py | tszulist-hbn | tszulist-hbn |  | ut |
| 20 | 2675 | [Bug Skip]: AttributeError: 'NoneType' object has no attribute 'clone' | Add to skiplist: Mark test as xfail or add skip decorator in the test file for XPU if not applicable. | [Bug Skip]: AttributeError: 'NoneType' object has no attribute 'clone' | pponikox | pponikox |  | ut |
| 21 | 2712 | [upstream_ut]  RuntimeError: Cannot swap t2 because it has weakref associated with it ; RuntimeError: _apply(): Co
 | Fix error handling: Implement proper error handling or add XPU-specific error path in the operation. | [upstream_ut]  RuntimeError: Cannot swap t2 because it has weakref associated with it ; RuntimeError: _apply(): Co
 | tszulist-hbn | tszulist-hbn |  | ut |
| 22 | 2720 | [upstream_ut] RuntimeError: false INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/native/DispatchStub.cpp":275 | Investigate and fix test failure: Analyze traceback and implement proper fix in the operation implementation or test. | [upstream_ut] RuntimeError: false INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/native/DispatchStub.cpp":275 | CuiYifeng | CuiYifeng |  | ut |
| 23 | 2722 | [Bug Skip]: NotImplementedError: Could not run 'aten::flip' with arguments from the 'QuantizedXPU' backend | Implement XPU kernel or backend support: Add proper XPU dispatch in native_functions.yaml or implement the missing kernel in the appropriate aten module. | [Bug Skip]: NotImplementedError: Could not run 'aten::flip' with arguments from the 'QuantizedXPU' backend | Silv3S | Silv3S | open | ut |
| 24 | 2766 | MaxPool2d - investigate memory layout performance | Fix performance issue: Implement optimized XPU kernel or enable existing optimization path for the operation. | MaxPool2d - investigate memory layout performance | BBBela | BBBela |  | ut |
| 25 | 2767 | [UT] test_control_flow_xpu.py got AssertionError | Investigate and fix test failure: Analyze traceback and implement proper fix in the operation implementation or test. | [UT] test_control_flow_xpu.py got AssertionError | PatrykWilczewski | PatrykWilczewski |  | ut |
| 26 | 2783 | [Bug Skip]: Key "xpu" is missing from dict "driver" in test_svd | Add to skiplist: Mark test as xfail or add skip decorator in the test file for XPU if not applicable. | [Bug Skip]: Key "xpu" is missing from dict "driver" in test_svd | daisyden | daisyden |  | ut |
| 27 | 2795 | Histc raises error with integer input when deterministic algorithm is enabled | Fix dtype/precision issue: Adjust tolerance in test file or implement proper dtype handling in the operation kernel for XPU. | Histc raises error with integer input when deterministic algorithm is enabled | CuiYifeng | CuiYifeng |  | ut |
| 28 | 2798 | Test case  test/test_dlpack.py::TestTorchDlPackCPU::test_numpy_cross_device_transfer_cpu failed with assert error. 'cpu'!='xpu' | Investigate and fix test failure: Analyze traceback and implement proper fix in the operation implementation or test. | Test case  test/test_dlpack.py::TestTorchDlPackCPU::test_numpy_cross_device_transfer_cpu failed with assert error. 'cpu'!='xpu' | None | None |  | ut |
| 29 | 2815 | RuntimeError: output with shape [2] doesn't match the broadcast shape [2, 2] | Fix error handling: Implement proper error handling or add XPU-specific error path in the operation. | RuntimeError: output with shape [2] doesn't match the broadcast shape [2, 2] | PawelSwider2000 | PawelSwider2000 |  | ut |
| 30 | 2817 | Expected error message is different than actual | Implement XPU kernel or backend support: Add proper XPU dispatch in native_functions.yaml or implement the missing kernel in the appropriate aten module. | Expected error message is different than actual | kdrozd-dev | kdrozd-dev |  | ut |
| 31 | 2869 | [Bug Skip]: New UT failure in 0209 nightly windows. | Investigate and fix test failure: Analyze traceback and implement proper fix in the operation implementation or test. | [Bug Skip]: New UT failure in 0209 nightly windows. | None | None |  | ut |
| 32 | 2879 | RuntimeError: _share_fd_: only available on CPU | Implement XPU kernel or backend support: Add proper XPU dispatch in native_functions.yaml or implement the missing kernel in the appropriate aten module. | RuntimeError: _share_fd_: only available on CPU | Silv3S | Silv3S |  | ut |
| 33 | 2919 | [XPU][upstream_ut][COW] Fix materialization in remaining TestCompositeComplianceXPU tests | Implement XPU kernel or backend support: Add proper XPU dispatch in native_functions.yaml or implement the missing kernel in the appropriate aten module. | [XPU][upstream_ut][COW] Fix materialization in remaining TestCompositeComplianceXPU tests | gplutop7 | gplutop7 | open;open | ut |
| 34 | 2950 | SYCL compilation flag -fsycl-id-queries-fit-in-int does not work as expected for TriuTril kernel. | Fix Inductor XPU compilation: Add proper lowering in torch/_inductor/lowering.py or fix decomposition path for the specific operator. | SYCL compilation flag -fsycl-id-queries-fit-in-int does not work as expected for TriuTril kernel. | BBBela | BBBela |  | ut |
| 35 | 3000 | [Bug Skip]: RuntimeError: _share_fd_: only available on CPU in test_dataloader_xpu.py | Implement XPU kernel or backend support: Add proper XPU dispatch in native_functions.yaml or implement the missing kernel in the appropriate aten module. | [Bug Skip]: RuntimeError: _share_fd_: only available on CPU in test_dataloader_xpu.py | gplutop7 | gplutop7 |  | ut |
| 36 | 3013 | [upstream_ut] RuntimeError: Kernel is incompatible with all devices in devs | Implement XPU kernel or backend support: Add proper XPU dispatch in native_functions.yaml or implement the missing kernel in the appropriate aten module. | [upstream_ut] RuntimeError: Kernel is incompatible with all devices in devs | None | None |  | ut |
| 37 | 3060 | Implement torch._scaled_grouped_mm for xpu backend | Implement XPU kernel or backend support: Add proper XPU dispatch in native_functions.yaml or implement the missing kernel in the appropriate aten module. | Implement torch._scaled_grouped_mm for xpu backend | Stonepia, liangan1 | Stonepia, liangan1 |  | ut |
| 38 | 3077 | [Bug Skip] test_dlpack.py::TestTorchDlPackXPU::test_no_copy_xpu - RuntimeError: Can't get ATen device for XPU without XPU data. | Implement XPU kernel or backend support: Add proper XPU dispatch in native_functions.yaml or implement the missing kernel in the appropriate aten module. | [Bug Skip] test_dlpack.py::TestTorchDlPackXPU::test_no_copy_xpu - RuntimeError: Can't get ATen device for XPU without XPU data. | AKloniecki | AKloniecki |  | ut |
| 39 | 3089 | AssertionError: Torch not compiled with CUDA enabled | Implement XPU kernel or backend support: Add proper XPU dispatch in native_functions.yaml or implement the missing kernel in the appropriate aten module. | AssertionError: Torch not compiled with CUDA enabled | jmamzax | jmamzax |  | ut |
| 40 | 3121 | [Bug Skip]: CUDA specific UT test_fft_half_and_chalf_not_power_of_two_error | Fix STFT/FFT precision on XPU: Use float32 intermediate or adjust test tolerance for float16/bfloat16. | [Bug Skip]: CUDA specific UT test_fft_half_and_chalf_not_power_of_two_error | None | None |  | ut |
| 41 | 3128 | [upstream_ut]  AssertionError: RuntimeError not raised by <lambda> 
 | Fix error handling: Implement proper error handling or add XPU-specific error path in the operation. | [upstream_ut]  AssertionError: RuntimeError not raised by <lambda> 
 | daisyden | daisyden |  | ut |
| 42 | 3143 | NotImplementedError: The operator 'aten::_scaled_dot_product_efficient_attention_backward' is not currently implemented for the XPU device. | Fix error handling: Implement proper error handling or add XPU-specific error path in the operation. | NotImplementedError: The operator 'aten::_scaled_dot_product_efficient_attention_backward' is not currently implemented for the XPU device. | LuFinch | LuFinch |  | ut |
| 43 | 3150 | [Task] Align XPU kernel's implementation to stock PyTorch | Implement XPU kernel or backend support: Add proper XPU dispatch in native_functions.yaml or implement the missing kernel in the appropriate aten module. | [Task] Align XPU kernel's implementation to stock PyTorch | guangyey | guangyey |  | ut |
| 44 | 3167 | NotImplementedError: Could not run 'aten::triangular_solve.X' with arguments from the 'SparseCsrXPU' backend | Implement XPU kernel or backend support: Add proper XPU dispatch in native_functions.yaml or implement the missing kernel in the appropriate aten module. | NotImplementedError: Could not run 'aten::triangular_solve.X' with arguments from the 'SparseCsrXPU' backend | tszulist-hbn | tszulist-hbn |  | ut |
| 45 | 3169 | NotImplementedError: Could not run 'aten::hspmm' with arguments from the 'SparseXPU' backend | Implement XPU kernel or backend support: Add proper XPU dispatch in native_functions.yaml or implement the missing kernel in the appropriate aten module. | NotImplementedError: Could not run 'aten::hspmm' with arguments from the 'SparseXPU' backend | jkosnox | jkosnox |  | ut |
| 46 | 3170 | Unskip test_bmm_windows_error_xpu_float64 | Implement XPU kernel or backend support: Add proper XPU dispatch in native_functions.yaml or implement the missing kernel in the appropriate aten module. | Unskip test_bmm_windows_error_xpu_float64 | jenniew | jenniew |  | ut |
| 47 | 3175 | [Bug Skip]: ValueError: sampled_addmm(): all inputs are expected to be on the same GPU device | Implement XPU kernel or backend support: Add proper XPU dispatch in native_functions.yaml or implement the missing kernel in the appropriate aten module. | [Bug Skip]: ValueError: sampled_addmm(): all inputs are expected to be on the same GPU device | None | None |  | ut |
| 48 | 3194 | Incorrect strides in TestCommonXPU,test_out_addmv_xpu_float32 | Implement XPU kernel or backend support: Add proper XPU dispatch in native_functions.yaml or implement the missing kernel in the appropriate aten module. | Incorrect strides in TestCommonXPU,test_out_addmv_xpu_float32 | AKloniecki | AKloniecki |  | ut |
| 49 | 3216 | [OPs] Some ops of XPU have non-determinism and are inconsistent with CUDA behavior. | Implement XPU kernel or backend support: Add proper XPU dispatch in native_functions.yaml or implement the missing kernel in the appropriate aten module. | [OPs] Some ops of XPU have non-determinism and are inconsistent with CUDA behavior. | CuiYifeng | CuiYifeng |  | ut |
| 50 | 3243 | AssertionError: False is not true | Investigate and fix test failure: Analyze traceback and implement proper fix in the operation implementation or test. | AssertionError: False is not true | pponikox | pponikox |  | ut |
| 51 | 3247 | NotImplementedError: "dot_xpu_mkl" not implemented for 'Long' | Fix dtype/precision issue: Adjust tolerance in test file or implement proper dtype handling in the operation kernel for XPU. | NotImplementedError: "dot_xpu_mkl" not implemented for 'Long' | Silv3S | Silv3S |  | ut |
| 52 | 3259 | New failed test cases 2026-04-02 | Implement XPU kernel or backend support: Add proper XPU dispatch in native_functions.yaml or implement the missing kernel in the appropriate aten module. | New failed test cases 2026-04-02 | SlawomirLaba | SlawomirLaba |  | ut |
| 53 | 3266 | [RFC] Migrate XPU kernel math functions from std::/:: to sycl::/sycl::native:: namespace | Implement XPU kernel or backend support: Add proper XPU dispatch in native_functions.yaml or implement the missing kernel in the appropriate aten module. | [RFC] Migrate XPU kernel math functions from std::/:: to sycl::/sycl::native:: namespace | None | None |  | ut |
| 54 | 3284 | Optimize torch.nn.functional.one_hot | Fix performance issue: Implement optimized XPU kernel or enable existing optimization path for the operation. | Optimize torch.nn.functional.one_hot | Silv3S | Silv3S | open | ut |
| | | **Subtotal: 54 issues** | | | | | |

### <span id='dtype-precision'>Dtype/Precision</span> (47 issues)

| # | ID | Title | Action Reason | Summary | Assignee | Owner Transfer | PR Status | Test Module |
|--:|----|-------|---------------|---------|----------|----------------|-----------|-------------|
| 55 | 2234 | [upstream_ut] AssertionError: RuntimeError not raised : Expected RuntimeError when doing an unsafe cast from a result of dtype torch.float32 into an out= with dtype torch.long | Fix dtype/precision issue: Adjust tolerance in test file or implement proper dtype handling in the operation kernel for XPU. | [upstream_ut] AssertionError: RuntimeError not raised : Expected RuntimeError when doing an unsafe cast from a result of dtype torch.float32 into an o | Silv3S | Silv3S |  | ut |
| 56 | 2238 | Exception: Tensor-likes are not close! in test/functorch/test_ops.py | Fix dtype/precision issue: Adjust tolerance in test file or implement proper dtype handling in the operation kernel for XPU. | Exception: Tensor-likes are not close! in test/functorch/test_ops.py | BBBela | BBBela | merged;merged;open;merged | ut |
| 57 | 2251 | [upstream_ut] test_fake_autocase got Exception: Dtypes torch.float32 and torch.float16 are not equal! | Fix dtype/precision issue: Adjust tolerance in test file or implement proper dtype handling in the operation kernel for XPU. | [upstream_ut] test_fake_autocase got Exception: Dtypes torch.float32 and torch.float16 are not equal! | astachowiczhabana | astachowiczhabana |  | ut |
| 58 | 2253 | the supported dtypes are not align with cuda | Fix dtype/precision issue: Adjust tolerance in test file or implement proper dtype handling in the operation kernel for XPU. | the supported dtypes are not align with cuda | daisyden | daisyden |  | ut |
| 59 | 2257 | Accuracy failures in test/xpu/test_unary_ufuncs_xpu.py | Fix dtype/precision issue: Adjust tolerance in test file or implement proper dtype handling in the operation kernel for XPU. | Accuracy failures in test/xpu/test_unary_ufuncs_xpu.py | pbielak | pbielak | closed;open;merged;open;merged;open | ut |
| 60 | 2287 | [upstream_ut] test_python_ref issues | This is not a PyTorch code issue - may require CI or documentation changes. | [upstream_ut] test_python_ref issues | yucai-intel | yucai-intel | closed;open | ut |
| 61 | 2301 | [upstream_ut] dtypes not align with OpInfo | Fix dtype/precision issue: Adjust tolerance in test file or implement proper dtype handling in the operation kernel for XPU. | [upstream_ut] dtypes not align with OpInfo | daisyden | daisyden |  | ut |
| 62 | 2482 | test_dtypes issue introduced by pytorch test sample input updates | Fix dtype/precision issue: Adjust tolerance in test file or implement proper dtype handling in the operation kernel for XPU. | test_dtypes issue introduced by pytorch test sample input updates | daisyden | daisyden |  | ut |
| 63 | 2510 | [upstream_ut]  RuntimeError: Expected output.numel() <= std::numeric_limits<int32_t>::max() to be true, but got fa
 | Fix error handling: Implement proper error handling or add XPU-specific error path in the operation. | [upstream_ut]  RuntimeError: Expected output.numel() <= std::numeric_limits<int32_t>::max() to be true, but got fa
 | SlawomirLaba, PawelSwider2000 | SlawomirLaba, PawelSwider2000 |  | ut |
| 64 | 2529 | [upstream_ut]  AssertionError: False is not true | Investigate and fix test failure: Analyze traceback and implement proper fix in the operation implementation or test. | [upstream_ut]  AssertionError: False is not true | Silv3S | Silv3S |  | ut |
| 65 | 2615 | [Bug Skip]: New failures RuntimeError: Unsupported dtype Half / RuntimeError: Unsupported dtype torch.float16 | Fix dtype/precision issue: Adjust tolerance in test file or implement proper dtype handling in the operation kernel for XPU. | [Bug Skip]: New failures RuntimeError: Unsupported dtype Half / RuntimeError: Unsupported dtype torch.float16 | CuiYifeng | CuiYifeng |  | ut |
| 66 | 2618 | [Bug Skip]: [regression] AssertionError: Scalars are not close! AssertionError: Tensor-likes are not close! | Implement XPU kernel or backend support: Add proper XPU dispatch in native_functions.yaml or implement the missing kernel in the appropriate aten module. | [Bug Skip]: [regression] AssertionError: Scalars are not close! AssertionError: Tensor-likes are not close! | jmamzax | jmamzax |  | ut |
| 67 | 2630 | Title: [upstream_ut]  AssertionError: Scalars are not equal! | Implement XPU kernel or backend support: Add proper XPU dispatch in native_functions.yaml or implement the missing kernel in the appropriate aten module. | Title: [upstream_ut]  AssertionError: Scalars are not equal! | jmamzax | jmamzax |  | ut |
| 68 | 2640 | random issue test_vjpvjp_index_reduce_prod_xpu_float32 | Fix dtype/precision issue: Adjust tolerance in test file or implement proper dtype handling in the operation kernel for XPU. | random issue test_vjpvjp_index_reduce_prod_xpu_float32 | wpietka | wpietka |  | ut |
| 69 | 2654 | [BMG][OOB] t5 inference performance drop 2 | Fix performance issue: Implement optimized XPU kernel or enable existing optimization path for the operation. | [BMG][OOB] t5 inference performance drop 2 | jianyizh | jianyizh |  | e2e |
| 70 | 2655 | [BMG][OOB] hf_Reformer performance drop | Fix performance issue: Implement optimized XPU kernel or enable existing optimization path for the operation. | [BMG][OOB] hf_Reformer performance drop | jianyizh | jianyizh |  | e2e |
| 71 | 2669 | [upstream_ut]  AssertionError: Tensor-likes are not close! in functorch/test_vmap.py | Fix dtype/precision issue: Adjust tolerance in test file or implement proper dtype handling in the operation kernel for XPU. | [upstream_ut]  AssertionError: Tensor-likes are not close! in functorch/test_vmap.py | tszulist-hbn | tszulist-hbn |  | ut |
| 72 | 2676 | Random failure in CI test | Implement XPU kernel or backend support: Add proper XPU dispatch in native_functions.yaml or implement the missing kernel in the appropriate aten module. | Random failure in CI test | BBBela | BBBela |  | ut |
| 73 | 2680 | XPU Autocast does not support  fp32 dtypes | Fix dtype/precision issue: Adjust tolerance in test file or implement proper dtype handling in the operation kernel for XPU. | XPU Autocast does not support  fp32 dtypes | CuiYifeng | CuiYifeng |  | ut |
| 74 | 2689 | [LNL][Windows] AssertionError: 'Assertion `cur_target >= 0 && cur_target < n_classes` failed'  not found in 'PYTORCH_API_USAGE torch.python | Investigate and fix test failure: Analyze traceback and implement proper fix in the operation implementation or test. | [LNL][Windows] AssertionError: 'Assertion `cur_target >= 0 && cur_target < n_classes` failed'  not found in 'PYTORCH_API_USAGE torch.python | draghan, tadkrawiec | draghan, tadkrawiec |  | ut |
| 75 | 2714 | [upstream_ut]  AssertionError: Object comparison failed: torch.float32 != torch.float64 
 | Fix dtype/precision issue: Adjust tolerance in test file or implement proper dtype handling in the operation kernel for XPU. | [upstream_ut]  AssertionError: Object comparison failed: torch.float32 != torch.float64 
 | Silv3S | Silv3S |  | ut |
| 76 | 2759 | [Bug Skip]: New failed cases 2026-1-22 | Investigate and fix test failure: Analyze traceback and implement proper fix in the operation implementation or test. | [Bug Skip]: New failed cases 2026-1-22 | AKloniecki | AKloniecki |  | ut |
| 77 | 2779 | Accuracy failures in logspace op | Fix dtype/precision issue: Adjust tolerance in test file or implement proper dtype handling in the operation kernel for XPU. | Accuracy failures in logspace op | PawelSwider2000 | PawelSwider2000 |  | ut |
| 78 | 2816 | torch.logcumsumexp incorrectly returns NaNs for complex64 input | Fix dtype/precision issue: Adjust tolerance in test file or implement proper dtype handling in the operation kernel for XPU. | torch.logcumsumexp incorrectly returns NaNs for complex64 input | Silv3S | Silv3S |  | ut |
| 79 | 2837 | Accuracy issue for Muon optimizer | Fix dtype/precision issue: Adjust tolerance in test file or implement proper dtype handling in the operation kernel for XPU. | Accuracy issue for Muon optimizer | Silv3S | Silv3S |  | ut |
| 80 | 2840 | Accuracy issue with 64 bit indexing depthwise_conv | Fix dtype/precision issue: Adjust tolerance in test file or implement proper dtype handling in the operation kernel for XPU. | Accuracy issue with 64 bit indexing depthwise_conv | SlawomirLaba, Silv3S | SlawomirLaba, Silv3S |  | ut |
| 81 | 2858 | [Bug Skip]: test_xpu new failures | Add to skiplist: Mark test as xfail or add skip decorator in the test file for XPU if not applicable. | [Bug Skip]: test_xpu new failures | None | None |  | ut |
| 82 | 2914 | Test case test/test_autograd.py::TestAutogradMultipleDispatchCPU::test_view_copy_cpu' failed with error AssertionError: Tensor-likes are not close! | Investigate and fix test failure: Analyze traceback and implement proper fix in the operation implementation or test. | Test case test/test_autograd.py::TestAutogradMultipleDispatchCPU::test_view_copy_cpu' failed with error AssertionError: Tensor-likes are not close! | None | None |  | ut |
| 83 | 2924 | [release/2.11] xcit_large_24_p8_224 amp_bf16 training got fail_accuracy | Fix dtype/precision issue: Adjust tolerance in test file or implement proper dtype handling in the operation kernel for XPU. | [release/2.11] xcit_large_24_p8_224 amp_bf16 training got fail_accuracy | jianyizh, mengfei25 | jianyizh, mengfei25 |  | e2e |
| 84 | 2928 | [release/2.11] pyhpc_turbulent_kinetic_energy fp32 inference got fail_accuracy | Fix dtype/precision issue: Adjust tolerance in test file or implement proper dtype handling in the operation kernel for XPU. | [release/2.11] pyhpc_turbulent_kinetic_energy fp32 inference got fail_accuracy | jianyizh | jianyizh |  | e2e |
| 85 | 2938 | [release/2.11] basic_gnn_gin and basic_gnn_sage inference fp32 performance dropped ~25% | Fix performance issue: Implement optimized XPU kernel or enable existing optimization path for the operation. | [release/2.11] basic_gnn_gin and basic_gnn_sage inference fp32 performance dropped ~25% | jianyizh | jianyizh |  | e2e |
| 86 | 2952 | [release/2.11][wsl] timm_models_accuracy_training_bfloat16 convnextv2_nano.fcmae_ft_in22k_in1k fail_accuracy | Fix dtype/precision issue: Adjust tolerance in test file or implement proper dtype handling in the operation kernel for XPU. | [release/2.11][wsl] timm_models_accuracy_training_bfloat16 convnextv2_nano.fcmae_ft_in22k_in1k fail_accuracy | weishi-deng | weishi-deng |  | ut |
| 87 | 2953 | [release/2.11][wsl] huggingface TrOCRForCausalLM and XGLMForCausalLM pass but has RuntimeError: value cannot be converted to type float without overflow | Fix dtype/precision issue: Adjust tolerance in test file or implement proper dtype handling in the operation kernel for XPU. | [release/2.11][wsl] huggingface TrOCRForCausalLM and XGLMForCausalLM pass but has RuntimeError: value cannot be converted to type float without overfl | None | None |  | e2e |
| 88 | 2960 | [release/2.11] timm_models_xcit_large_24_p8_224_float16_training accuracy test failed on PTL Windows | Fix dtype/precision issue: Adjust tolerance in test file or implement proper dtype handling in the operation kernel for XPU. | [release/2.11] timm_models_xcit_large_24_p8_224_float16_training accuracy test failed on PTL Windows | pfierek, tadkrawiec | pfierek, tadkrawiec |  | ut |
| 89 | 2984 | [release/2.11] sebotnet33ts_256 fp32 training got fail_accuracy | Fix precision/accuracy issue: Adjust tolerance or implement proper dtype handling for XPU. | [release/2.11] sebotnet33ts_256 fp32 training got fail_accuracy | jianyizh, weishi-deng | jianyizh, weishi-deng |  | e2e |
| 90 | 3033 | [Bug Skip]: Softmax tolerance | Implement XPU kernel or backend support: Add proper XPU dispatch in native_functions.yaml or implement the missing kernel in the appropriate aten module. | [Bug Skip]: Softmax tolerance | chunhuanMeng | chunhuanMeng |  | ut |
| 91 | 3041 | AssertionError: Expected len(flat_diff_results) > 0 in test_fake_crossref_backward_amp_normal_number_mean_xpu_float32 | Investigate and fix test failure: Analyze traceback and implement proper fix in the operation implementation or test. | AssertionError: Expected len(flat_diff_results) > 0 in test_fake_crossref_backward_amp_normal_number_mean_xpu_float32 | Silv3S | Silv3S |  | ut |
| 92 | 3058 | [E2E] hf_GPT2_large amp_fp16/amp_bf16  training got  fail_accuracy | Fix dtype/precision issue: Adjust tolerance in test file or implement proper dtype handling in the operation kernel for XPU. | [E2E] hf_GPT2_large amp_fp16/amp_bf16  training got  fail_accuracy | weishi-deng | weishi-deng |  | e2e |
| 93 | 3103 | Tensor-likes are not equal for test_backward_nn_functional_conv3d_xpu_float32 | Fix dtype/precision issue: Adjust tolerance in test file or implement proper dtype handling in the operation kernel for XPU. | Tensor-likes are not equal for test_backward_nn_functional_conv3d_xpu_float32 | BBBela | BBBela | open | ut |
| 94 | 3137 | [upstream_ut]  RuntimeError: expected scalar type Half but found Float 
 | Fix dtype/precision issue: Adjust tolerance in test file or implement proper dtype handling in the operation kernel for XPU. | [upstream_ut]  RuntimeError: expected scalar type Half but found Float 
 | LuFinch | LuFinch | open | ut |
| 95 | 3156 | AssertionError: 'Assertion cur_target >= 0 && cur_target <   n_classes failed' not found | Implement XPU kernel or backend support: Add proper XPU dispatch in native_functions.yaml or implement the missing kernel in the appropriate aten module. | AssertionError: 'Assertion cur_target >= 0 && cur_target <   n_classes failed' not found | kdrozd-dev | kdrozd-dev |  | ut |
| 96 | 3163 | [Bug Skip]: Object comparison failed: torch.int64 != torch.int32 in test_sparse_add | Fix dtype/precision issue: Adjust tolerance in test file or implement proper dtype handling in the operation kernel for XPU. | [Bug Skip]: Object comparison failed: torch.int64 != torch.int32 in test_sparse_add | chunhuanMeng | chunhuanMeng |  | ut |
| 97 | 3177 | Accuracy gap of BF16/FP16 test_block_addmm | Fix dtype/precision issue: Adjust tolerance in test file or implement proper dtype handling in the operation kernel for XPU. | Accuracy gap of BF16/FP16 test_block_addmm | jenniew | jenniew | open | ut |
| 98 | 3184 | New failing UTs: test_cross_entropy_loss_2d_out_of_bounds_class_index_xpu | Implement XPU kernel or backend support: Add proper XPU dispatch in native_functions.yaml or implement the missing kernel in the appropriate aten module. | New failing UTs: test_cross_entropy_loss_2d_out_of_bounds_class_index_xpu | wpietka | wpietka |  | ut |
| 99 | 3238 | The supported dtypes of _refs.stft is not aligned to stft | Fix dtype/precision issue: Adjust tolerance in test file or implement proper dtype handling in the operation kernel for XPU. | The supported dtypes of _refs.stft is not aligned to stft | daisyden | daisyden |  | ut |
| 100 | 3290 | huggingface amp_fp16 inference accuracy openai/whisper-tiny got fail_accuracy | Fix dtype/precision issue: Adjust tolerance in test file or implement proper dtype handling in the operation kernel for XPU. | huggingface amp_fp16 inference accuracy openai/whisper-tiny got fail_accuracy | jianyizh, weishi-deng | jianyizh, weishi-deng |  | e2e |
| 101 | 3296 | accuracy gap of stft in float16 | Fix dtype/precision issue: Adjust tolerance in test file or implement proper dtype handling in the operation kernel for XPU. | accuracy gap of stft in float16 | None | None |  | ut |
| | | **Subtotal: 47 issues** | | | | | |

### <span id='others'>Others</span> (30 issues)

| # | ID | Title | Action Reason | Summary | Assignee | Owner Transfer | PR Status | Test Module |
|--:|----|-------|---------------|---------|----------|----------------|-----------|-------------|
| 102 | 2261 | [xpu][profiler] Run with fork process has extra warning | Implement XPU kernel or backend support: Add proper XPU dispatch in native_functions.yaml or implement the missing kernel in the appropriate aten module. | [xpu][profiler] Run with fork process has extra warning | moksiuc | moksiuc |  | ut |
| 103 | 2349 | [oneAPI][backward compatibility] libur_loader.so.0: version `LIBUR_LOADER_0.11' not found | Implement XPU kernel or backend support: Add proper XPU dispatch in native_functions.yaml or implement the missing kernel in the appropriate aten module. | [oneAPI][backward compatibility] libur_loader.so.0: version `LIBUR_LOADER_0.11' not found | riverliuintel | riverliuintel |  | ut |
| 104 | 2389 | [Bug Skip]: RuntimeError: Data corruption detected | Fix error handling: Implement proper error handling or add XPU-specific error path in the operation. | [Bug Skip]: RuntimeError: Data corruption detected | PatrykWilczewski | PatrykWilczewski |  | ut |
| 105 | 2434 | [Bug Skip]: New failures 2025-11-28 | Add to skiplist: Mark test as xfail or add skip decorator in the test file for XPU if not applicable. | [Bug Skip]: New failures 2025-11-28 | AKloniecki | AKloniecki | merged;open | ut |
| 106 | 2465 | [windows] ut hang | Fix performance issue: Implement optimized XPU kernel or enable existing optimization path for the operation. | [windows] ut hang | tadkrawiec, mganczarenko | tadkrawiec, mganczarenko |  | ut |
| 107 | 2562 | Warning as Error | This is not a PyTorch code issue - may require CI or documentation changes. | Warning as Error | chunhuanMeng | chunhuanMeng |  | ut |
| 108 | 2595 | [Bug Skip]: Random crashed cases 2025-12-17 | Investigate and fix test failure: Analyze traceback and implement proper fix in the operation implementation or test. | [Bug Skip]: Random crashed cases 2025-12-17 | BBBela | BBBela |  | ut |
| 109 | 2656 | [release/2.10] models got fail_accuracy on BMG WSL2 | Fix precision/accuracy issue: Adjust tolerance or implement proper dtype handling for XPU. | [release/2.10] models got fail_accuracy on BMG WSL2 | None | None |  | ut |
| 110 | 2660 | [release/2.10][Windows][BMG] New failed test cases | Implement XPU kernel or backend support: Add proper XPU dispatch in native_functions.yaml or implement the missing kernel in the appropriate aten module. | [release/2.10][Windows][BMG] New failed test cases | pfierek, tadkrawiec, eryk-roch | pfierek, tadkrawiec, eryk-roch |  | ut |
| 111 | 2662 | [release/2.10][Windows][BMG] New failed test cases and 2.9 also failed but pvc passed | Implement XPU kernel or backend support: Add proper XPU dispatch in native_functions.yaml or implement the missing kernel in the appropriate aten module. | [release/2.10][Windows][BMG] New failed test cases and 2.9 also failed but pvc passed | tadkrawiec, kdrozd-dev | tadkrawiec, kdrozd-dev | open | ut |
| 112 | 2729 | [Bug Skip]: Random failures 2026WW03 | Add to skiplist: Mark test as xfail or add skip decorator in the test file for XPU if not applicable. | [Bug Skip]: Random failures 2026WW03 | Silv3S | Silv3S |  | ut |
| 113 | 2769 | [oneDNN] New failed test cases with 3.11 compared with 3.10 | Implement XPU kernel or backend support: Add proper XPU dispatch in native_functions.yaml or implement the missing kernel in the appropriate aten module. | [oneDNN] New failed test cases with 3.11 compared with 3.10 | LuFinch | LuFinch |  | ut |
| 114 | 2907 | [release/2.11] Models performance regression for 5 testcases | Fix performance issue: Implement optimized XPU kernel or enable existing optimization path for the operation. | [release/2.11] Models performance regression for 5 testcases | xuhancn | xuhancn |  | ut |
| 115 | 2908 | [release/2.11] Model fail_accuracy for 5 testcases | Fix precision/accuracy issue: Adjust tolerance or implement proper dtype handling for XPU. | [release/2.11] Model fail_accuracy for 5 testcases | xuhancn | xuhancn |  | e2e |
| 116 | 2912 | [release/2.11] UT extended 220 new failures | This is not a PyTorch code issue - may require CI or documentation changes. | [release/2.11] UT extended 220 new failures | None | None |  | ut |
| 117 | 2942 | [Windows] Unit tests got Fatal python error | Implement XPU kernel or backend support: Add proper XPU dispatch in native_functions.yaml or implement the missing kernel in the appropriate aten module. | [Windows] Unit tests got Fatal python error | xuhancn, Stonepia | xuhancn, Stonepia |  | ut |
| 118 | 2965 | [Bug Skip]: Random failures 2026WW10 | Add to skiplist: Mark test as xfail or add skip decorator in the test file for XPU if not applicable. | [Bug Skip]: Random failures 2026WW10 | None | None |  | ut |
| 119 | 2966 | [Bug Skip]: [Regression]2026-3-2 ut failures | Add to skiplist: Mark test as xfail or add skip decorator in the test file for XPU if not applicable. | [Bug Skip]: [Regression]2026-3-2 ut failures | jmamzax | jmamzax |  | ut |
| 120 | 3014 | [upstream_ut] AssertionError: False is not true | Investigate and fix test failure: Analyze traceback and implement proper fix in the operation implementation or test. | [upstream_ut] AssertionError: False is not true | None | None |  | ut |
| 121 | 3025 | New failing test in Nightly Wheel test_decomp_xpu.HasDecompTest,test_has_decomposition | Implement XPU kernel or backend support: Add proper XPU dispatch in native_functions.yaml or implement the missing kernel in the appropriate aten module. | New failing test in Nightly Wheel test_decomp_xpu.HasDecompTest,test_has_decomposition | None | None |  | ut |
| 122 | 3030 | [Bug Skip] test/test_modules.py::TestModuleXPU::test_cpu_gpu_parity_nn_ConvTranspose2d_xpu_complex32 failed with | Add to skiplist: Mark test as xfail or add skip decorator in the test file for XPU if not applicable. | [Bug Skip] test/test_modules.py::TestModuleXPU::test_cpu_gpu_parity_nn_ConvTranspose2d_xpu_complex32 failed with | gplutop7 | gplutop7 | open | ut |
| 123 | 3048 | Profiler result is not correct on B70 | Implement XPU kernel or backend support: Add proper XPU dispatch in native_functions.yaml or implement the missing kernel in the appropriate aten module. | Profiler result is not correct on B70 | aostrowski-hbn | aostrowski-hbn |  | ut |
| 124 | 3074 | [Bug Skip] test_dlpack_exchange_api expect current_work_stream is NOT null | Add to skiplist: Mark test as xfail or add skip decorator in the test file for XPU if not applicable. | [Bug Skip] test_dlpack_exchange_api expect current_work_stream is NOT null | AKloniecki | AKloniecki |  | ut |
| 125 | 3083 | [Bug Skip]: Random failures 2026WW12 | Add to skiplist: Mark test as xfail or add skip decorator in the test file for XPU if not applicable. | [Bug Skip]: Random failures 2026WW12 | None | None |  | ut |
| 126 | 3086 | nvml support blocks some test cases | Implement XPU kernel or backend support: Add proper XPU dispatch in native_functions.yaml or implement the missing kernel in the appropriate aten module. | nvml support blocks some test cases | None | None |  | ut |
| 127 | 3129 | [upstream_ut]  AssertionError: UserWarning not triggered 
 | Investigate and fix test failure: Analyze traceback and implement proper fix in the operation implementation or test. | [upstream_ut]  AssertionError: UserWarning not triggered 
 | daisyden | daisyden |  | ut |
| 128 | 3180 | [E2E] Timm/Torchbench models got "eager_two_runs_differ" on ARC | Implement XPU kernel or backend support: Add proper XPU dispatch in native_functions.yaml or implement the missing kernel in the appropriate aten module. | [E2E] Timm/Torchbench models got "eager_two_runs_differ" on ARC | None | None |  | ut |
| 129 | 3189 | Task Tracker | Implement XPU kernel or backend support: Add proper XPU dispatch in native_functions.yaml or implement the missing kernel in the appropriate aten module. | Task Tracker | guangyey | guangyey |  | ut |
| 130 | 3286 | New failing test case after enabling tests from test_ctx_manager_xpu.py | Implement XPU kernel or backend support: Add proper XPU dispatch in native_functions.yaml or implement the missing kernel in the appropriate aten module. | New failing test case after enabling tests from test_ctx_manager_xpu.py | BBBela | BBBela | open | ut |
| 131 | 3300 | [CI] When creating PR, several pull workflows are launched and then all but one are immediately cancelled. | This is not a PyTorch code issue - may require CI or documentation changes. | [CI] When creating PR, several pull workflows are launched and then all but one are immediately cancelled. | None | None |  | ut |
| | | **Subtotal: 30 issues** | | | | | |

### <span id='inductor-compilation'>Inductor/Compilation</span> (25 issues)

| # | ID | Title | Action Reason | Summary | Assignee | Owner Transfer | PR Status | Test Module |
|--:|----|-------|---------------|---------|----------|----------------|-----------|-------------|
| 132 | 2295 | [upstream_ut][xpu][test]nn/test_embedding.py::TestEmbeddingNNDeviceTypeXPU::test_embedding_bag_device_xpu_int32_int32_float64 meet AssertionError: Tensor-likes are not close! | Investigate and fix test failure: Analyze traceback and implement proper fix in the operation implementation or test. | [upstream_ut][xpu][test]nn/test_embedding.py::TestEmbeddingNNDeviceTypeXPU::test_embedding_bag_device_xpu_int32_int32_float64 meet AssertionError: Ten | yucai-intel | yucai-intel |  | ut |
| 133 | 2554 | [upstream_ut]  AssertionError: AssertionError not raised 
 | Investigate and fix test failure: Analyze traceback and implement proper fix in the operation implementation or test. | [upstream_ut]  AssertionError: AssertionError not raised 
 | daisyden | daisyden |  | ut |
| 134 | 2609 | [upstream_ut]  torch._inductor.exc.InductorError: CppCompileError: C++ compile error 
 | Fix Inductor XPU compilation: Add proper lowering in torch/_inductor/lowering.py or fix decomposition path for the specific operator. | [upstream_ut]  torch._inductor.exc.InductorError: CppCompileError: C++ compile error 
 | daisyden | daisyden | open | ut |
| 135 | 2611 | [upstream_ut]  AssertionError: Tensor-likes are not equal! in test_compile_subprocess | Fix Inductor XPU compilation: Add proper lowering in torch/_inductor/lowering.py or fix decomposition path for the specific operator. | [upstream_ut]  AssertionError: Tensor-likes are not equal! in test_compile_subprocess | daisyden | daisyden |  | ut |
| 136 | 2613 | [upstream_ut]  AssertionError: Tensor-likes are not equal! in test_compile_subprocess.py | Investigate and fix test failure: Analyze traceback and implement proper fix in the operation implementation or test. | [upstream_ut]  AssertionError: Tensor-likes are not equal! in test_compile_subprocess.py | daisyden | daisyden |  | ut |
| 137 | 2620 | [upstream_ut]  AssertionError: dtype is needed to compute eps1 when eps1 is unset 
 | Investigate and fix test failure: Analyze traceback and implement proper fix in the operation implementation or test. | [upstream_ut]  AssertionError: dtype is needed to compute eps1 when eps1 is unset 
 | daisyden | daisyden |  | ut |
| 138 | 2697 | Title: [upstream_ut]  RuntimeError: Expected to find ", 0, " but did not find it | Fix error handling: Implement proper error handling or add XPU-specific error path in the operation. | Title: [upstream_ut]  RuntimeError: Expected to find ", 0, " but did not find it | chunhuanMeng | chunhuanMeng |  | e2e |
| 139 | 2715 | [upstream_ut]  torch._dynamo.exc.Unsupported: Attempted to inline function marked as skipped 
 | Fix Inductor XPU compilation: Add proper lowering in torch/_inductor/lowering.py or fix decomposition path for the specific operator. | [upstream_ut]  torch._dynamo.exc.Unsupported: Attempted to inline function marked as skipped 
 | CuiYifeng | CuiYifeng |  | ut |
| 140 | 2800 | AttributeError: 'torch._C._XpuDeviceProperties' object has no attribute 'major' | Implement XPU kernel or backend support: Add proper XPU dispatch in native_functions.yaml or implement the missing kernel in the appropriate aten module. | AttributeError: 'torch._C._XpuDeviceProperties' object has no attribute 'major' | guangyey | guangyey |  | ut |
| 141 | 2810 | AssertionError: Object comparison failed: Decimal('2.938735877055718769921841343055614194546[51 chars]-39') != Decimal('0') | Investigate and fix test failure: Analyze traceback and implement proper fix in the operation implementation or test. | AssertionError: Object comparison failed: Decimal('2.938735877055718769921841343055614194546[51 chars]-39') != Decimal('0') | daisyden | daisyden |  | ut |
| 142 | 2873 | [Bug Skip]: test_repos.py contains several failed ops | Add to skiplist: Mark test as xfail or add skip decorator in the test file for XPU if not applicable. | [Bug Skip]: test_repos.py contains several failed ops | PawelSwider2000 | PawelSwider2000 |  | ut |
| 143 | 2888 | torch._inductor.exc.InductorError: AssertionError: Conversions between float8_e5m2 and float8_e4m3fn is not supported! | Fix dtype/precision issue: Adjust tolerance in test file or implement proper dtype handling in the operation kernel for XPU. | torch._inductor.exc.InductorError: AssertionError: Conversions between float8_e5m2 and float8_e4m3fn is not supported! | Stonepia | Stonepia |  | ut |
| 144 | 2891 | RuntimeError: Expected to find "(262144, 0, 512, 1" but did not find it | Implement XPU kernel or backend support: Add proper XPU dispatch in native_functions.yaml or implement the missing kernel in the appropriate aten module. | RuntimeError: Expected to find "(262144, 0, 512, 1" but did not find it | chunhuanMeng | chunhuanMeng |  | e2e |
| 145 | 2922 | [release/2.11] UT inductor AssertionError: pass_fds not supported on Windows. | Investigate and fix test failure: Analyze traceback and implement proper fix in the operation implementation or test. | [release/2.11] UT inductor AssertionError: pass_fds not supported on Windows. | tadkrawiec | tadkrawiec |  | ut |
| 146 | 2935 | [release/2.11][inductor] huggingface amp_fp16 and float16 training XLNetLMHeadModel perf regression | Fix Inductor XPU compilation: Add proper lowering in torch/_inductor/lowering.py or fix decomposition path for the specific operator. | [release/2.11][inductor] huggingface amp_fp16 and float16 training XLNetLMHeadModel perf regression | jianyizh | jianyizh |  | e2e |
| 147 | 2958 | AssertionError of test_dtensor_basic_compile | Investigate and fix test failure: Analyze traceback and implement proper fix in the operation implementation or test. | AssertionError of test_dtensor_basic_compile | daisyden | daisyden |  | ut |
| 148 | 2997 | AssertionError of test_linear_and_cel_max_autotune | Investigate and fix test failure: Analyze traceback and implement proper fix in the operation implementation or test. | AssertionError of test_linear_and_cel_max_autotune | etaf | etaf |  | ut |
| 149 | 2999 | KeyError: 'eager_numerics.use_pytorch_libdevice' | Implement XPU kernel or backend support: Add proper XPU dispatch in native_functions.yaml or implement the missing kernel in the appropriate aten module. | KeyError: 'eager_numerics.use_pytorch_libdevice' | daisyden | daisyden |  | ut |
| 150 | 3004 | TypeError: _xpu_recordMemoryHistory(): incompatible function arguments | Fix error handling: Implement proper error handling or add XPU-specific error path in the operation. | TypeError: _xpu_recordMemoryHistory(): incompatible function arguments | guangyey | guangyey |  | ut |
| 151 | 3094 | XPUGraph tree support | Implement XPU kernel or backend support: Add proper XPU dispatch in native_functions.yaml or implement the missing kernel in the appropriate aten module. | XPUGraph tree support | None | None |  | ut |
| 152 | 3095 | cutlass support blocks some unit test cases | Implement XPU kernel or backend support: Add proper XPU dispatch in native_functions.yaml or implement the missing kernel in the appropriate aten module. | cutlass support blocks some unit test cases | None | None |  | ut |
| 153 | 3148 | [Triton] Huggingface openai/whisper-tiny got fail_accuracy | Fix precision/accuracy issue: Adjust tolerance or implement proper dtype handling for XPU. | [Triton] Huggingface openai/whisper-tiny got fail_accuracy | None | None |  | e2e |
| 154 | 3151 | [Triton] Timm_models  rexnet_100 / fbnetv3_b / sebotnet33ts_256 got fail_accuracy | Fix precision/accuracy issue: Adjust tolerance or implement proper dtype handling for XPU. | [Triton] Timm_models  rexnet_100 / fbnetv3_b / sebotnet33ts_256 got fail_accuracy | None | None |  | e2e |
| 155 | 3187 | PyTorch XPU gpu_cpp_wrapper fails with InductorError NotImplementedError | Fix Inductor XPU compilation: Add proper lowering in torch/_inductor/lowering.py or fix decomposition path for the specific operator. | PyTorch XPU gpu_cpp_wrapper fails with InductorError NotImplementedError | CuiYifeng | CuiYifeng |  | ut |
| 156 | 3191 | torch._inductor.exc.InductorError: AssertionError: both a fallback and a decomp for same op: aten.index_add.default | Fix Inductor XPU compilation: Add proper lowering in torch/_inductor/lowering.py or fix decomposition path for the specific operator. | torch._inductor.exc.InductorError: AssertionError: both a fallback and a decomp for same op: aten.index_add.default | EikanWang, Copilot | EikanWang, Copilot |  | e2e |
| | | **Subtotal: 25 issues** | | | | | |

### <span id='distributed'>Distributed</span> (24 issues)

| # | ID | Title | Action Reason | Summary | Assignee | Owner Transfer | PR Status | Test Module |
|--:|----|-------|---------------|---------|----------|----------------|-----------|-------------|
| 157 | 2340 | [distributed][_tools] AssertionError: Roofline estimation needs to access CUDA capabilities to make estimations | Investigate and fix test failure: Analyze traceback and implement proper fix in the operation implementation or test. | [distributed][_tools] AssertionError: Roofline estimation needs to access CUDA capabilities to make estimations | githubsgi | githubsgi |  | ut |
| 158 | 2404 | [distributed][checkpoint] AssertionError: Booleans mismatch: False is not True | Investigate and fix test failure: Analyze traceback and implement proper fix in the operation implementation or test. | [distributed][checkpoint] AssertionError: Booleans mismatch: False is not True | None | None |  | ut |
| 159 | 2659 | [distributed] test_dist2.py RuntimeError: Backend xccl does not implement getBackendOptions. | Implement XPU kernel or backend support: Add proper XPU dispatch in native_functions.yaml or implement the missing kernel in the appropriate aten module. | [distributed] test_dist2.py RuntimeError: Backend xccl does not implement getBackendOptions. | Chao1Han | Chao1Han |  | ut |
| 160 | 2686 | [distributed] Accuracy issues with test_distributed_spawn.py | Fix distributed operation on XPU: Update process group initialization or implement XPU-compatible collectives. | [distributed] Accuracy issues with test_distributed_spawn.py | frost-intel | frost-intel |  | ut |
| 161 | 2700 | [distributed] Hang issues with test_distributed_spawn.py | Fix distributed operation on XPU: Update process group initialization or implement XPU-compatible collectives. | [distributed] Hang issues with test_distributed_spawn.py | syedshahbaaz | syedshahbaaz |  | ut |
| 162 | 2701 | [distributed] Barrier Timeout Error with test_distributed_spawn.py | Fix distributed operation on XPU: Update process group initialization or implement XPU-compatible collectives. | [distributed] Barrier Timeout Error with test_distributed_spawn.py | syedshahbaaz | syedshahbaaz |  | ut |
| 163 | 2702 | [distributed] RuntimeError: Work ran time out after 0 milliseconds with test_distributed_spawn.py | Fix distributed operation on XPU: Update process group initialization or implement XPU-compatible collectives. | [distributed] RuntimeError: Work ran time out after 0 milliseconds with test_distributed_spawn.py | syedshahbaaz | syedshahbaaz |  | ut |
| 164 | 2737 | [distributed] AttributeError: module 'torch._C' has no attribute '_gather' | Fix distributed backend for XPU: Update process group initialization for XPU in torch/distributed/distributed_c10d.py or implement XPU-compatible collective operations. | [distributed] AttributeError: module 'torch._C' has no attribute '_gather' | None | None |  | ut |
| 165 | 2738 | [distributed] test_c10d_spawn_nccl.py ValueError: input tensor must be the same size as output size times world size | Fix distributed backend for XPU: Update process group initialization for XPU in torch/distributed/distributed_c10d.py or implement XPU-compatible collective operations. | [distributed] test_c10d_spawn_nccl.py ValueError: input tensor must be the same size as output size times world size | jenniew | jenniew |  | ut |
| 166 | 2968 | [distributed] timeout issue in test/distributed/test_c10d_xccl.py | Fix distributed backend for XPU: Update process group initialization for XPU in torch/distributed/distributed_c10d.py or implement XPU-compatible collective operations. | [distributed] timeout issue in test/distributed/test_c10d_xccl.py | frost-intel | frost-intel |  | ut |
| 167 | 2969 | [distributed] AssertionError: Scalars are not equal! in test/distributed/test_c10d_xccl.py | Investigate and fix test failure: Analyze traceback and implement proper fix in the operation implementation or test. | [distributed] AssertionError: Scalars are not equal! in test/distributed/test_c10d_xccl.py | frost-intel | frost-intel |  | ut |
| 168 | 2972 | [distributed] AssertionError: ValueError not raised in test/distributed/test_c10d_xccl.py | Fix distributed backend for XPU: Update process group initialization for XPU in torch/distributed/distributed_c10d.py or implement XPU-compatible collective operations. | [distributed] AssertionError: ValueError not raised in test/distributed/test_c10d_xccl.py | newtdms | newtdms |  | ut |
| 169 | 3021 | [distributed] all_to_all_single Compatibility Issue on B60/XCCL | Fix distributed backend for XPU: Update process group initialization for XPU in torch/distributed/distributed_c10d.py or implement XPU-compatible collective operations. | [distributed] all_to_all_single Compatibility Issue on B60/XCCL | zhangxiaoli73 | zhangxiaoli73 |  | ut |
| 170 | 3022 | [distributed] batch_isend_irecv Compatibility Issue on B60/XCCL | Fix distributed backend for XPU: Update process group initialization for XPU in torch/distributed/distributed_c10d.py or implement XPU-compatible collective operations. | [distributed] batch_isend_irecv Compatibility Issue on B60/XCCL | zhangxiaoli73 | zhangxiaoli73 |  | ut |
| 171 | 3082 | multithread support in distributed | Fix distributed backend for XPU: Update process group initialization for XPU in torch/distributed/distributed_c10d.py or implement XPU-compatible collective operations. | multithread support in distributed | None | None |  | ut |
| 172 | 3100 | [distributed] /handler/dump_nccl_trace_pickle and nccl_log need in distributed ut tests | Fix distributed backend for XPU: Update process group initialization for XPU in torch/distributed/distributed_c10d.py or implement XPU-compatible collective operations. | [distributed] /handler/dump_nccl_trace_pickle and nccl_log need in distributed ut tests | None | None |  | ut |
| 173 | 3101 | [distributed] 'torch._C._distributed_c10d.ProcessGroupXCCL' object has no attribute '_set_default_timeout' in test_dynamo_distributed.py | Fix distributed backend for XPU: Update process group initialization for XPU in torch/distributed/distributed_c10d.py or implement XPU-compatible collective operations. | [distributed] 'torch._C._distributed_c10d.ProcessGroupXCCL' object has no attribute '_set_default_timeout' in test_dynamo_distributed.py | None | None |  | ut |
| 174 | 3102 | [distributed] RuntimeError: Invalid device string: 'xpu:foo' in test_sharding_spec.py | Implement XPU kernel or backend support: Add proper XPU dispatch in native_functions.yaml or implement the missing kernel in the appropriate aten module. | [distributed] RuntimeError: Invalid device string: 'xpu:foo' in test_sharding_spec.py | None | None |  | ut |
| 175 | 3139 | [distributed][_composable] AssertionError: Expects xpu:0 but got xpu:1 | Investigate and fix test failure: Analyze traceback and implement proper fix in the operation implementation or test. | [distributed][_composable] AssertionError: Expects xpu:0 but got xpu:1 | Kanya-Mo | Kanya-Mo |  | ut |
| 176 | 3232 | [distributed][tensor] AssertionError: AssertionError not raised : Placement (Shard(dim=2),) in test/distributed/tensor/test_attention.py | Investigate and fix test failure: Analyze traceback and implement proper fix in the operation implementation or test. | [distributed][tensor] AssertionError: AssertionError not raised : Placement (Shard(dim=2),) in test/distributed/tensor/test_attention.py | Kanya-Mo | Kanya-Mo |  | ut |
| 177 | 3233 | [distributed] RuntimeError: No backend for the parent process group or its backend does not support splitting in test/distributed/test_device_mesh.py | Fix distributed operation on XPU: Update process group initialization or implement XPU-compatible collectives. | [distributed] RuntimeError: No backend for the parent process group or its backend does not support splitting in test/distributed/test_device_mesh.py | songhappy | songhappy |  | ut |
| 178 | 3270 | [distributed][tensor] RuntimeError: Invalid scaling configuration in test_matrix_ops.py | Fix distributed backend for XPU: Update process group initialization for XPU in torch/distributed/distributed_c10d.py or implement XPU-compatible collective operations. | [distributed][tensor] RuntimeError: Invalid scaling configuration in test_matrix_ops.py | syedshahbaaz | syedshahbaaz |  | ut |
| 179 | 3305 | [distributed] shrink operation support in test/distributed/test_c10d_xccl.py | Fix distributed backend for XPU: Update process group initialization for XPU in torch/distributed/distributed_c10d.py or implement XPU-compatible collective operations. | [distributed] shrink operation support in test/distributed/test_c10d_xccl.py | None | None |  | ut |
| 180 | 3306 | [distributed] no attribute '_reset_fr_recording_xccl' in test/distributed/test_c10d_xccl.py | Add XCCL flight recorder API: In torch/csrc/distributed/c10d/init.cpp, add `module.def("_reset_fr_recording_xccl", []() { ::c10d::reset_xccl_trace(); });` similar to _reset_fr_recording_nccl at line 4249. | [distributed] no attribute '_reset_fr_recording_xccl' in test/distributed/test_c10d_xccl.py | None | None |  | ut |
| | | **Subtotal: 24 issues** | | | | | |

### <span id='flash-attention-transformer'>Flash Attention/Transformer</span> (23 issues)

| # | ID | Title | Action Reason | Summary | Assignee | Owner Transfer | PR Status | Test Module |
|--:|----|-------|---------------|---------|----------|----------------|-----------|-------------|
| 181 | 2270 | Backend Compatibility Error in test/xpu/test_decomp.py | Implement XPU kernel or backend support: Add proper XPU dispatch in native_functions.yaml or implement the missing kernel in the appropriate aten module. | Backend Compatibility Error in test/xpu/test_decomp.py | LuFinch | LuFinch |  | ut |
| 182 | 2285 | Support efficient attention | Fix attention/SDPA operation on XPU: Implement proper scaled_dot_product_attention dispatch or add XPU fallback in native_functions.yaml. | Support efficient attention | chunhuanMeng | chunhuanMeng |  | ut |
| 183 | 2390 | SDPA in pytorch use different backend compared with ipex | Implement XPU kernel or backend support: Add proper XPU dispatch in native_functions.yaml or implement the missing kernel in the appropriate aten module. | SDPA in pytorch use different backend compared with ipex | LuFinch | LuFinch |  | ut |
| 184 | 2442 | [Bug Skip]: NotImplementedError: Could not run 'aten::_flash_attention_forward' with arguments from the 'CPU' backend | Fix attention/SDPA operation on XPU: Implement proper scaled_dot_product_attention dispatch or add XPU fallback in native_functions.yaml. | [Bug Skip]: NotImplementedError: Could not run 'aten::_flash_attention_forward' with arguments from the 'CPU' backend | daisyden, LuFinch | daisyden, LuFinch |  | ut |
| 185 | 2570 | crash in sdpa. | Implement XPU kernel or backend support: Add proper XPU dispatch in native_functions.yaml or implement the missing kernel in the appropriate aten module. | crash in sdpa. | LuFinch | LuFinch |  | ut |
| 186 | 2693 | Title: [upstream_ut]  AssertionError: Scalars are not equal! | Investigate and fix test failure: Analyze traceback and implement proper fix in the operation implementation or test. | Title: [upstream_ut]  AssertionError: Scalars are not equal! | hoshibara | hoshibara |  | ut |
| 187 | 2698 | Title: [upstream_ut]  RuntimeError: FlashAttentionForwardXPU only support headdim 64,96,128,192 | Fix Flash Attention on XPU: Enable or implement FlashAttentionForwardXPU with proper head_dim and dropout support. | Title: [upstream_ut]  RuntimeError: FlashAttentionForwardXPU only support headdim 64,96,128,192 | chunhuanMeng, LuFinch | chunhuanMeng, LuFinch |  | ut |
| 188 | 2802 | Three aten._scaled_dot_product_flash_attention issues | Fix attention/SDPA operation on XPU: Implement proper scaled_dot_product_attention dispatch or add XPU fallback in native_functions.yaml. | Three aten._scaled_dot_product_flash_attention issues | LuFinch | LuFinch | open | ut |
| 189 | 2853 | [upstream_ut] torch.ops.aten._flash_attention_forward lack of support for XPU. | Fix attention/SDPA operation on XPU: Implement proper scaled_dot_product_attention dispatch or add XPU fallback in native_functions.yaml. | [upstream_ut] torch.ops.aten._flash_attention_forward lack of support for XPU. | LuFinch | LuFinch |  | ut |
| 190 | 3093 | XPU does not support NestedTensor for SDPA operations. | Implement XPU kernel or backend support: Add proper XPU dispatch in native_functions.yaml or implement the missing kernel in the appropriate aten module. | XPU does not support NestedTensor for SDPA operations. | None | None |  | ut |
| 191 | 3126 | [upstream_ut]  Two NestedTensor issue with flash attention | Fix Flash Attention on XPU: Enable or implement FlashAttentionForwardXPU with proper head_dim and dropout support. | [upstream_ut]  Two NestedTensor issue with flash attention | daisyden | daisyden |  | ut |
| 192 | 3131 | [upstream_ut]  NotImplementedError: The operator 'aten::_scaled_dot_product_efficient_attention_backward' is not c
 | Fix error handling: Implement proper error handling or add XPU-specific error path in the operation. | [upstream_ut]  NotImplementedError: The operator 'aten::_scaled_dot_product_efficient_attention_backward' is not c
 | chunhuanMeng | chunhuanMeng |  | ut |
| 193 | 3132 | [upstream_ut]  transfomers test reports RuntimeError: No available kernel. Aborting execution.  | Implement XPU kernel or backend support: Add proper XPU dispatch in native_functions.yaml or implement the missing kernel in the appropriate aten module. | [upstream_ut]  transfomers test reports RuntimeError: No available kernel. Aborting execution.  | LuFinch | LuFinch | open | ut |
| 194 | 3133 | [upstream_ut]  RuntimeError: scaled_dot_product_attention: If inputs are nested tensors they must be contiguous 
 | Fix error handling: Implement proper error handling or add XPU-specific error path in the operation. | [upstream_ut]  RuntimeError: scaled_dot_product_attention: If inputs are nested tensors they must be contiguous 
 | daisyden | daisyden |  | ut |
| 195 | 3136 | [upstream_ut]  AssertionError: False is not true in test_transformers | Investigate and fix test failure: Analyze traceback and implement proper fix in the operation implementation or test. | [upstream_ut]  AssertionError: False is not true in test_transformers | LuFinch | LuFinch | open | ut |
| 196 | 3140 | [upstream_ut]  RuntimeError: FlashAttentionForwardXPU does not only support dropout > 0.0 yet 
 | Fix Flash Attention on XPU: Enable or implement FlashAttentionForwardXPU with proper head_dim and dropout support. | [upstream_ut]  RuntimeError: FlashAttentionForwardXPU does not only support dropout > 0.0 yet 
 | LuFinch | LuFinch |  | ut |
| 197 | 3141 | [upstream_ut]  RuntimeError: FlashAttentionForwardXPU only support headdim 64,96,128,192 
 | Fix Flash Attention on XPU: Enable or implement FlashAttentionForwardXPU with proper head_dim and dropout support. | [upstream_ut]  RuntimeError: FlashAttentionForwardXPU only support headdim 64,96,128,192 
 | LuFinch | LuFinch |  | ut |
| 198 | 3176 | [Bug Skip]: ValueError: _scaled_dot_product_attention(): all inputs are expected to be on the same GPU device | Implement XPU kernel or backend support: Add proper XPU dispatch in native_functions.yaml or implement the missing kernel in the appropriate aten module. | [Bug Skip]: ValueError: _scaled_dot_product_attention(): all inputs are expected to be on the same GPU device | None | None |  | ut |
| 199 | 3178 | New failed test cases 2026-03-25 | Implement XPU kernel or backend support: Add proper XPU dispatch in native_functions.yaml or implement the missing kernel in the appropriate aten module. | New failed test cases 2026-03-25 | pponikox | pponikox |  | ut |
| 200 | 3195 | test_sdpa_unbacked_no_dde_xpu crashed | Implement XPU kernel or backend support: Add proper XPU dispatch in native_functions.yaml or implement the missing kernel in the appropriate aten module. | test_sdpa_unbacked_no_dde_xpu crashed | None | None |  | ut |
| 201 | 3229 | RuntimeError: No viable backend for scaled_dot_product_attention was found | Implement XPU kernel or backend support: Add proper XPU dispatch in native_functions.yaml or implement the missing kernel in the appropriate aten module. | RuntimeError: No viable backend for scaled_dot_product_attention was found | tszulist-hbn | tszulist-hbn |  | ut |
| 202 | 3231 | Dynamo failed to run FX node with fake tensors: call_function <built-in function scaled_dot_product_attention> | Fix attention/SDPA operation on XPU: Implement proper scaled_dot_product_attention dispatch or add XPU fallback in native_functions.yaml. | Dynamo failed to run FX node with fake tensors: call_function <built-in function scaled_dot_product_attention> | None | None |  | ut |
| 203 | 3258 | huggingface accuracy inference Error in op: torch.ops.aten._scaled_dot_product_fused_attention_overrideable.default | Fix attention/SDPA operation on XPU: Implement proper scaled_dot_product_attention dispatch or add XPU fallback in native_functions.yaml. | huggingface accuracy inference Error in op: torch.ops.aten._scaled_dot_product_fused_attention_overrideable.default | LuFinch | LuFinch | merged;merged | ut |
| | | **Subtotal: 23 issues** | | | | | |

### <span id='torchao'>TorchAO</span> (21 issues)

| # | ID | Title | Action Reason | Summary | Assignee | Owner Transfer | PR Status | Test Module |
|--:|----|-------|---------------|---------|----------|----------------|-----------|-------------|
| 204 | 2323 | [TorchAO] MOE training enabling on XPU | Implement XPU kernel or backend support: Add proper XPU dispatch in native_functions.yaml or implement the missing kernel in the appropriate aten module. | [TorchAO] MOE training enabling on XPU | riverliuintel | riverliuintel |  | ut |
| 205 | 2324 | [TorchAO] FP8 conv support | Fix dtype/precision issue: Adjust tolerance in test file or implement proper dtype handling in the operation kernel for XPU. | [TorchAO] FP8 conv support | Stonepia | Stonepia |  | ut |
| 206 | 2325 | [TorchAO] Float8 training support on XPU | Fix dtype/precision issue: Adjust tolerance in test file or implement proper dtype handling in the operation kernel for XPU. | [TorchAO] Float8 training support on XPU | arlesniak, riverliuintel | arlesniak, riverliuintel |  | ut |
| 207 | 2326 | [TorchAO] MX training  native PyTorch on XPU | Implement XPU kernel or backend support: Add proper XPU dispatch in native_functions.yaml or implement the missing kernel in the appropriate aten module. | [TorchAO] MX training  native PyTorch on XPU | riverliuintel | riverliuintel |  | ut |
| 208 | 2327 | [TorchAO] benchmark enabling on XPU | Implement XPU kernel or backend support: Add proper XPU dispatch in native_functions.yaml or implement the missing kernel in the appropriate aten module. | [TorchAO] benchmark enabling on XPU | None | None |  | ut |
| 209 | 2532 | Title: [upstream_ut]  AssertionError: wrong number of dimensions2 for op: torch.ops.aten._convert_weight_to_int4pack.defa | Investigate and fix test failure: Analyze traceback and implement proper fix in the operation implementation or test. | Title: [upstream_ut]  AssertionError: wrong number of dimensions2 for op: torch.ops.aten._convert_weight_to_int4pack.defa | yucai-intel | yucai-intel |  | ut |
| 210 | 2572 | [TorchAO][UT] test/dtypes/test_affine_quantized.py AssertionError: Tensor-likes are not close! | Fix dtype/precision issue: Adjust tolerance in test file or implement proper dtype handling in the operation kernel for XPU. | [TorchAO][UT] test/dtypes/test_affine_quantized.py AssertionError: Tensor-likes are not close! | xiaowangintel | xiaowangintel |  | build |
| 211 | 2578 | [TorchAO][UT] test/quantization/test_quant_api.py AssertionError: SQNR -2.90625 is too low | Investigate and fix test failure: Analyze traceback and implement proper fix in the operation implementation or test. | [TorchAO][UT] test/quantization/test_quant_api.py AssertionError: SQNR -2.90625 is too low | Stonepia | Stonepia |  | build |
| 212 | 2580 | [TorchAO][UT] test/test_low_bit_optim.py AssertionError: Tensor-likes are not close! | Investigate and fix test failure: Analyze traceback and implement proper fix in the operation implementation or test. | [TorchAO][UT] test/test_low_bit_optim.py AssertionError: Tensor-likes are not close! | arlesniak | arlesniak |  | build |
| 213 | 2597 | [TorchAO][BMG] INT4 GPTQ shows worse performance compared with RTN and AWQ. | Fix performance issue: Implement optimized XPU kernel or enable existing optimization path for the operation. | [TorchAO][BMG] INT4 GPTQ shows worse performance compared with RTN and AWQ. | xiaowangintel | xiaowangintel |  | ut |
| 214 | 2598 | [TorchAO][BMG]The first token latency of Qwen2.5-1.5B-Instruct drops 10%+ when max-new-tokens changes from 2 to 1. | Fix performance issue: Implement optimized XPU kernel or enable existing optimization path for the operation. | [TorchAO][BMG]The first token latency of Qwen2.5-1.5B-Instruct drops 10%+ when max-new-tokens changes from 2 to 1. | Stonepia | Stonepia |  | ut |
| 215 | 2605 | [int4][inductor] Add freezing pattern for fusing int4 mm kernel as #170341 | Fix Inductor XPU compilation: Add proper lowering in torch/_inductor/lowering.py or fix decomposition path for the specific operator. | [int4][inductor] Add freezing pattern for fusing int4 mm kernel as #170341 | None | None |  | ut |
| 216 | 2707 | [TorchAO][BMG] INT4 GPTQ failed due to TorchAO API change. | Implement XPU kernel or backend support: Add proper XPU dispatch in native_functions.yaml or implement the missing kernel in the appropriate aten module. | [TorchAO][BMG] INT4 GPTQ failed due to TorchAO API change. | xiaowangintel | xiaowangintel |  | ut |
| 217 | 2823 | [TorchAO][BMG] Llama-3.2-1B-Instruct Dynamic INT8 got 20% performance drop on next token performance with 0122 nightly whl | Fix performance issue: Implement optimized XPU kernel or enable existing optimization path for the operation. | [TorchAO][BMG] Llama-3.2-1B-Instruct Dynamic INT8 got 20% performance drop on next token performance with 0122 nightly whl | xiaowangintel, lchen2331 | xiaowangintel, lchen2331 |  | ut |
| 218 | 2862 | accuracy issue with test_float8_scale_fast_accum_xpu | Fix dtype/precision issue: Adjust tolerance in test file or implement proper dtype handling in the operation kernel for XPU. | accuracy issue with test_float8_scale_fast_accum_xpu | tszulist-hbn | tszulist-hbn |  | ut |
| 219 | 2948 | [AO] Benchmark enabling on XPU | Implement XPU kernel or backend support: Add proper XPU dispatch in native_functions.yaml or implement the missing kernel in the appropriate aten module. | [AO] Benchmark enabling on XPU | None | None |  | ut |
| 220 | 2993 | [Bug Skip]: Unexpected success of test_cpu_gpu_parity_nn_ConvTranspose3d_xpu_complex32 | Add to skiplist: Mark test as xfail or add skip decorator in the test file for XPU if not applicable. | [Bug Skip]: Unexpected success of test_cpu_gpu_parity_nn_ConvTranspose3d_xpu_complex32 | gplutop7 | gplutop7 |  | ut |
| 221 | 3032 | [TorchAO][UT] failures in test/prototype/safetensors/test_safetensors_support.py | Implement XPU kernel or backend support: Add proper XPU dispatch in native_functions.yaml or implement the missing kernel in the appropriate aten module. | [TorchAO][UT] failures in test/prototype/safetensors/test_safetensors_support.py | Stonepia | Stonepia |  | build |
| 222 | 3076 | [TorchAO][BMG] Llama-3.2-1B-Instruct Dynamic INT8 got 10% performance drop with oneDNN 3.11.1 | Implement XPU kernel or backend support: Add proper XPU dispatch in native_functions.yaml or implement the missing kernel in the appropriate aten module. | [TorchAO][BMG] Llama-3.2-1B-Instruct Dynamic INT8 got 10% performance drop with oneDNN 3.11.1 | None | None |  | ut |
| 223 | 3088 | [TorchAO][BMG] INT4 RTN Flex-attention got 5% performance drop | Fix performance issue: Implement optimized XPU kernel or enable existing optimization path for the operation. | [TorchAO][BMG] INT4 RTN Flex-attention got 5% performance drop | Stonepia | Stonepia |  | ut |
| 224 | 3124 | [TorchAO][Bug] ImportError: Requires mslk >= 1.0.0 when calling save_pretrained_torchao with qat_scheme="int4" on Qwen3-4B | Implement XPU kernel or backend support: Add proper XPU dispatch in native_functions.yaml or implement the missing kernel in the appropriate aten module. | [TorchAO][Bug] ImportError: Requires mslk >= 1.0.0 when calling save_pretrained_torchao with qat_scheme="int4" on Qwen3-4B | None | None |  | ut |
| | | **Subtotal: 21 issues** | | | | | |

### <span id='sparse'>Sparse</span> (13 issues)

| # | ID | Title | Action Reason | Summary | Assignee | Owner Transfer | PR Status | Test Module |
|--:|----|-------|---------------|---------|----------|----------------|-----------|-------------|
| 225 | 2235 | test/test_sparse_csr.py::TestSparseCompressedTritonKernelsXPU::test_triton_bsr_dense_addmm_meta_xpu meet unexpected warning | Implement sparse operation for XPU: Add missing sparse kernel implementation for the operation. | test/test_sparse_csr.py::TestSparseCompressedTritonKernelsXPU::test_triton_bsr_dense_addmm_meta_xpu meet unexpected warning | None | None |  | ut |
| 226 | 2244 | test/test_sparse_csr.py::TestSparseCSRXPU::test_block_addmm meet RuntimeError: empty_sparse_compressed expected sparse compressed (non-block) tensor layout but got SparseBsr | Fix error handling: Implement proper error handling or add XPU-specific error path in the operation. | test/test_sparse_csr.py::TestSparseCSRXPU::test_block_addmm meet RuntimeError: empty_sparse_compressed expected sparse compressed (non-block) tensor l | jenniew | jenniew |  | ut |
| 227 | 2245 | oneDNN matmul received incorrect shape in test/test_sparse_csr.py::TestSparseCSRXPU::test_addmm_errors_xpu_float32 | Implement sparse operation for XPU: Add missing sparse kernel implementation for the operation. | oneDNN matmul received incorrect shape in test/test_sparse_csr.py::TestSparseCSRXPU::test_addmm_errors_xpu_float32 | CuiYifeng | CuiYifeng |  | ut |
| 228 | 2246 | torch/sparse/_triton_ops*.py need to be ported to enable for Intel GPU for test_sparse and test_sparse_csr cases | Implement XPU kernel or backend support: Add proper XPU dispatch in native_functions.yaml or implement the missing kernel in the appropriate aten module. | torch/sparse/_triton_ops*.py need to be ported to enable for Intel GPU for test_sparse and test_sparse_csr cases | None | None |  | ut |
| 229 | 2663 | test_sparse_semi_structured.py gaps | Add to skiplist: Mark test as xfail or add skip decorator in the test file for XPU if not applicable. | test_sparse_semi_structured.py gaps | None | None |  | ut |
| 230 | 2751 | [Bug Skip]: Random failures 2026WW04 | Add to skiplist: Mark test as xfail or add skip decorator in the test file for XPU if not applicable. | [Bug Skip]: Random failures 2026WW04 | None | None |  | ut |
| 231 | 2777 | [Bug Skip]: Random failures 2026WW05 | Add to skiplist: Mark test as xfail or add skip decorator in the test file for XPU if not applicable. | [Bug Skip]: Random failures 2026WW05 | AKloniecki | AKloniecki |  | ut |
| 232 | 2801 | to_dense() for Sparse CSR backend cannot broadcast batch dim for indices. failed with: RuntimeError: source tensor shape must match self tensor shape, excluding the specified dimension. Got self.shape = [x, x] source.shape = [x] | Fix error handling: Implement proper error handling or add XPU-specific error path in the operation. | to_dense() for Sparse CSR backend cannot broadcast batch dim for indices. failed with: RuntimeError: source tensor shape must match self tensor shape, | jenniew | jenniew |  | ut |
| 233 | 2921 | [abs][complex64] - new failing test cases caused by PyTorch changes. | Implement XPU kernel or backend support: Add proper XPU dispatch in native_functions.yaml or implement the missing kernel in the appropriate aten module. | [abs][complex64] - new failing test cases caused by PyTorch changes. | AKloniecki | AKloniecki |  | ut |
| 234 | 2946 | [Bug Skip]: Random failures 2026WW09 | Add to skiplist: Mark test as xfail or add skip decorator in the test file for XPU if not applicable. | [Bug Skip]: Random failures 2026WW09 | BBBela | BBBela |  | ut |
| 235 | 3081 | Sparse CSR gemm-like ops have not been supported yet | Implement sparse operation for XPU: Add missing sparse kernel implementation for the operation. | Sparse CSR gemm-like ops have not been supported yet | tszulist-hbn | tszulist-hbn |  | ut |
| 236 | 3165 | test_sparse_csr_xpu.py::TestSparseCompressedTritonKernelsXPU::test_triton_bsr_softmax meet RuntimeError: ZE_RESULT_ERROR_INVALID_KERNEL_NAME | Implement sparse operation for XPU: Add missing sparse kernel implementation for the operation. | test_sparse_csr_xpu.py::TestSparseCompressedTritonKernelsXPU::test_triton_bsr_softmax meet RuntimeError: ZE_RESULT_ERROR_INVALID_KERNEL_NAME | None | None |  | ut |
| 237 | 3166 | test_consistency_SparseCSR failures | Implement sparse operation for XPU: Add missing sparse kernel implementation for the operation. | test_consistency_SparseCSR failures | yucai-intel | yucai-intel |  | ut |
| | | **Subtotal: 13 issues** | | | | | |

### <span id='feature-not-supported'>Feature Not Supported</span> (10 issues)

| # | ID | Title | Action Reason | Summary | Assignee | Owner Transfer | PR Status | Test Module |
|--:|----|-------|---------------|---------|----------|----------------|-----------|-------------|
| 238 | 2283 | [upstream_ut] sparse._sampled_addmm is not supported | Implement XPU kernel or backend support: Add proper XPU dispatch in native_functions.yaml or implement the missing kernel in the appropriate aten module. | [upstream_ut] sparse._sampled_addmm is not supported | jenniew | jenniew |  | ut |
| 239 | 2376 | [Bug Skip]: NotImplementedError: "logaddexp_xpu" not implemented for 'Complex' | Fix dtype/precision issue: Adjust tolerance in test file or implement proper dtype handling in the operation kernel for XPU. | [Bug Skip]: NotImplementedError: "logaddexp_xpu" not implemented for 'Complex' | CuiYifeng | CuiYifeng |  | ut |
| 240 | 2400 | [ut_upstream] tf32_on_and_off() need xpu support | Implement XPU kernel or backend support: Add proper XPU dispatch in native_functions.yaml or implement the missing kernel in the appropriate aten module. | [ut_upstream] tf32_on_and_off() need xpu support | chunhuanMeng | chunhuanMeng |  | ut |
| 241 | 2412 | Some NestedTensor missing XPU support | Implement XPU kernel or backend support: Add proper XPU dispatch in native_functions.yaml or implement the missing kernel in the appropriate aten module. | Some NestedTensor missing XPU support | yucai-intel | yucai-intel | closed;open | ut |
| 242 | 2531 | [upstream_ut]  AssertionError: Torch not compiled with CUDA enabled | Implement XPU kernel or backend support: Add proper XPU dispatch in native_functions.yaml or implement the missing kernel in the appropriate aten module. | [upstream_ut]  AssertionError: Torch not compiled with CUDA enabled | daisyden | daisyden |  | ut |
| 243 | 2918 | [XPU][upstream_ut][COW] Skip non-supported ops (jiterator + histogramdd) | Add to skiplist: Mark test as xfail or add skip decorator in the test file for XPU if not applicable. | [XPU][upstream_ut][COW] Skip non-supported ops (jiterator + histogramdd) | gplutop7 | gplutop7 | closed;open | ut |
| 244 | 3080 | cudagraph tests blocked by feature gap | Implement XPU kernel or backend support: Add proper XPU dispatch in native_functions.yaml or implement the missing kernel in the appropriate aten module. | cudagraph tests blocked by feature gap | None | None |  | ut |
| 245 | 3084 | torch.library.register_autocast does not support xpu | Implement XPU kernel or backend support: Add proper XPU dispatch in native_functions.yaml or implement the missing kernel in the appropriate aten module. | torch.library.register_autocast does not support xpu | None | None |  | ut |
| 246 | 3142 | [upstream_ut]  RuntimeError: The sycl_ext_oneapi_work_group_scratch_memory feature is not yet available for use with SYCL Graph extension. | Implement XPU kernel or backend support: Add proper XPU dispatch in native_functions.yaml or implement the missing kernel in the appropriate aten module. | [upstream_ut]  RuntimeError: The sycl_ext_oneapi_work_group_scratch_memory feature is not yet available for use with SYCL Graph extension. | LuFinch | LuFinch |  | ut |
| 247 | 3196 | vitals is not supported, the cases should be disabled | Implement XPU kernel or backend support: Add proper XPU dispatch in native_functions.yaml or implement the missing kernel in the appropriate aten module. | vitals is not supported, the cases should be disabled | libohao1201 | libohao1201 |  | ut |
| | | **Subtotal: 10 issues** | | | | | |

### <span id='torch-runtime'>Torch Runtime</span> (8 issues)

| # | ID | Title | Action Reason | Summary | Assignee | Owner Transfer | PR Status | Test Module |
|--:|----|-------|---------------|---------|----------|----------------|-----------|-------------|
| 248 | 2444 | [upstream_ut]  RuntimeError: UR backend failed. UR backend returns:40 (UR_RESULT_ERROR_OUT_OF_RESOURCES) ; Runtime
 | Implement XPU kernel or backend support: Add proper XPU dispatch in native_functions.yaml or implement the missing kernel in the appropriate aten module. | [upstream_ut]  RuntimeError: UR backend failed. UR backend returns:40 (UR_RESULT_ERROR_OUT_OF_RESOURCES) ; Runtime
 | Silv3S | Silv3S |  | ut |
| 249 | 2467 | Host may stuck when submit too many kernels when event recording | Implement XPU kernel or backend support: Add proper XPU dispatch in native_functions.yaml or implement the missing kernel in the appropriate aten module. | Host may stuck when submit too many kernels when event recording | jianyizh | jianyizh |  | ut |
| 250 | 2496 | [upstream_ut]  Segmentation fault when running test_torch.TestTorch and test_torch.TestTorchDeviceType at the same tiem. | Fix memory management on XPU: Check memory allocation/deallocation in the operation implementation. | [upstream_ut]  Segmentation fault when running test_torch.TestTorch and test_torch.TestTorchDeviceType at the same tiem. | astachowiczhabana | astachowiczhabana |  | ut |
| 251 | 2979 | eca_halonext26ts got RuntimeError: ZE_RESULT_ERROR_MODULE_BUILD_FAILURE | Implement XPU kernel or backend support: Add proper XPU dispatch in native_functions.yaml or implement the missing kernel in the appropriate aten module. | eca_halonext26ts got RuntimeError: ZE_RESULT_ERROR_MODULE_BUILD_FAILURE | None | None |  | e2e |
| 252 | 3011 | [upstream_ut] torch.OutOfMemoryError: XPU out of memory | Fix memory management on XPU: Check memory allocation/deallocation in the operation implementation. | [upstream_ut] torch.OutOfMemoryError: XPU out of memory | None | None |  | ut |
| 253 | 3096 | VISIBLE_DEVICE support | Implement XPU kernel or backend support: Add proper XPU dispatch in native_functions.yaml or implement the missing kernel in the appropriate aten module. | VISIBLE_DEVICE support | None | None |  | ut |
| 254 | 3114 | [Bug Skip]: Failure skip on 2026-3-21 | Add to skiplist: Mark test as xfail or add skip decorator in the test file for XPU if not applicable. | [Bug Skip]: Failure skip on 2026-3-21 | None | None |  | ut |
| 255 | 3227 | torch xpu event has ~0.1ms latency, which is too large | Fix performance issue: Implement optimized XPU kernel or enable existing optimization path for the operation. | torch xpu event has ~0.1ms latency, which is too large | guangyey | guangyey |  | ut |
| | | **Subtotal: 8 issues** | | | | | |

### <span id='pt2e'>PT2E</span> (4 issues)

| # | ID | Title | Action Reason | Summary | Assignee | Owner Transfer | PR Status | Test Module |
|--:|----|-------|---------------|---------|----------|----------------|-----------|-------------|
| 256 | 2250 | Found mismatch when comparing the output of aten.view.default on FakeTensor and concrete Tensors | Fix dtype/precision issue: Adjust tolerance in test file or implement proper dtype handling in the operation kernel for XPU. | Found mismatch when comparing the output of aten.view.default on FakeTensor and concrete Tensors | astachowiczhabana | astachowiczhabana |  | ut |
| 257 | 2742 | [Linux][PT2E] hf_Roberta_base model performance ASYMM and SYMM both failed | Fix performance issue: Implement optimized XPU kernel or enable existing optimization path for the operation. | [Linux][PT2E] hf_Roberta_base model performance ASYMM and SYMM both failed | chunhuanMeng | chunhuanMeng |  | e2e |
| 258 | 3006 | AssertionError: '.to(tl.float16)' unexpectedly found in '# AOT ID | Investigate and fix test failure: Analyze traceback and implement proper fix in the operation implementation or test. | AssertionError: '.to(tl.float16)' unexpectedly found in '# AOT ID | CuiYifeng | CuiYifeng |  | e2e |
| 259 | 3010 | [distributed][tensor] test_random_ops.py torch._dynamo.exc.TorchRuntimeError: RuntimeError when making fake tensor call | Fix Inductor XPU compilation: Add proper lowering in torch/_inductor/lowering.py or fix decomposition path for the specific operator. | [distributed][tensor] test_random_ops.py torch._dynamo.exc.TorchRuntimeError: RuntimeError when making fake tensor call | jenniew | jenniew |  | ut |
| | | **Subtotal: 4 issues** | | | | | |

### <span id='skip-no-test-exists'>Skip/No Test Exists</span> (3 issues)

| # | ID | Title | Action Reason | Summary | Assignee | Owner Transfer | PR Status | Test Module |
|--:|----|-------|---------------|---------|----------|----------------|-----------|-------------|
| 260 | 1778 | [Infra] Show known issues for accuracy test | Add to skiplist: Mark test as xfail or add skip decorator in the test file for XPU if not applicable. | [Infra] Show known issues for accuracy test | mengfei25 | mengfei25 |  | e2e |
| 261 | 2471 | test_cuda.py gaps | Add to skiplist: Mark test as xfail or add skip decorator in the test file for XPU if not applicable. | test_cuda.py gaps | guangyey | guangyey |  | ut |
| 262 | 2533 | Title: [upstream_ut]  AttributeError: 'TestQuantizedOpsXPU' object has no attribute 'test_qsoftmax' | Add to skiplist: Mark test as xfail or add skip decorator in the test file for XPU if not applicable. | Title: [upstream_ut]  AttributeError: 'TestQuantizedOpsXPU' object has no attribute 'test_qsoftmax' | astachowiczhabana | astachowiczhabana |  | ut |
| | | **Subtotal: 3 issues** | | | | | |

### <span id='build-compilation'>Build/Compilation</span> (2 issues)

| # | ID | Title | Action Reason | Summary | Assignee | Owner Transfer | PR Status | Test Module |
|--:|----|-------|---------------|---------|----------|----------------|-----------|-------------|
| 263 | 3209 | [Win][Build] There is Cyclic dependencies error when build with BUILD_SEPARATE_OPS=true | Implement XPU kernel or backend support: Add proper XPU dispatch in native_functions.yaml or implement the missing kernel in the appropriate aten module. | [Win][Build] There is Cyclic dependencies error when build with BUILD_SEPARATE_OPS=true | Copilot | Copilot |  | build |
| 264 | 3224 | [Win][Build] Building SYCL (Device) object torch_xpu_ops_sycl_kernels_gen_NMSKernel.cpp.obj failed on Windows | Implement XPU kernel or backend support: Add proper XPU dispatch in native_functions.yaml or implement the missing kernel in the appropriate aten module. | [Win][Build] Building SYCL (Device) object torch_xpu_ops_sycl_kernels_gen_NMSKernel.cpp.obj failed on Windows | chunhuanMeng | chunhuanMeng |  | build |
| | | **Subtotal: 2 issues** | | | | | |

### <span id='performance'>Performance</span> (2 issues)

| # | ID | Title | Action Reason | Summary | Assignee | Owner Transfer | PR Status | Test Module |
|--:|----|-------|---------------|---------|----------|----------------|-----------|-------------|
| 265 | 2939 | [release/2.11] gmlp_s16_224 inference amp performance dropped ~15% | Fix performance issue: Implement optimized XPU kernel or enable existing optimization path for the operation. | [release/2.11] gmlp_s16_224 inference amp performance dropped ~15% | jianyizh | jianyizh |  | e2e |
| 266 | 2981 | [release/2.11] T5 models performance dropped ~20% | Fix performance issue: Implement optimized XPU kernel or enable existing optimization path for the operation. | [release/2.11] T5 models performance dropped ~20% | jianyizh, weishi-deng | jianyizh, weishi-deng | merged;merged | e2e |
| | | **Subtotal: 2 issues** | | | | | |

### <span id='accuracy'>Accuracy</span> (1 issues)

| # | ID | Title | Action Reason | Summary | Assignee | Owner Transfer | PR Status | Test Module |
|--:|----|-------|---------------|---------|----------|----------------|-----------|-------------|
| 267 | 2592 | [release/2.10] models got fail_accuracy | Implement XPU kernel or backend support: Add proper XPU dispatch in native_functions.yaml or implement the missing kernel in the appropriate aten module. | [release/2.10] models got fail_accuracy | mengfei25 | mengfei25 | merged;closed | e2e |
| | | **Subtotal: 1 issues** | | | | | |

### <span id='profiler'>Profiler</span> (1 issues)

| # | ID | Title | Action Reason | Summary | Assignee | Owner Transfer | PR Status | Test Module |
|--:|----|-------|---------------|---------|----------|----------------|-----------|-------------|
| 268 | 2263 | [xpu][bug] XPU Trace event ends too late! | Implement XPU kernel or backend support: Add proper XPU dispatch in native_functions.yaml or implement the missing kernel in the appropriate aten module. | [xpu][bug] XPU Trace event ends too late! | PawelSwider2000 | PawelSwider2000 | open | ut |
| | | **Subtotal: 1 issues** | | | | | |

[Back to Index](#toc) |

## <span id='3-other-actions-by-type'>3. Other Actions by Type</span>

**Total: 38 issues** - Actions other than Need Investigation

### <span id='close-fixed-issue'>Close fixed issue</span> (17 issues)

| # | ID | Title | Action Reason | Summary | Assignee | Owner Transfer | PR Status | Test Module |
|--:|----|-------|---------------|---------|----------|----------------|-----------|-------------|
| 1 | 1624 | [DONT CLOSE] Known UT Issue list | All test cases passed on XPU/stock - issue is resolved | [DONT CLOSE] Known UT Issue list | None | RUIJIEZHONG66166 |  | ut |
| 2 | 2022 | [Windows] [CI] [UT] AssertionError: Tensor-likes are not close! | All test cases passed on XPU/stock - issue is resolved | [Windows] [CI] [UT] AssertionError: Tensor-likes are not close! | None | RUIJIEZHONG66166 |  | ut |
| 3 | 2392 | [Bug Skip]: torch.OutOfMemoryError: XPU out of memory | All test cases passed on XPU/stock - issue is resolved | [Bug Skip]: torch.OutOfMemoryError: XPU out of memory | xuhancn | RUIJIEZHONG66166 |  | ut |
| 4 | 2463 | [Bug Skip]: OSError: SYCL runtime is not dected. | All test cases passed on XPU/stock - issue is resolved | [Bug Skip]: OSError: SYCL runtime is not dected. | xuhancn | RUIJIEZHONG66166 |  | ut |
| 5 | 2530 | Title: [upstream_ut]  AssertionError: RuntimeError not raised | All test cases passed on XPU/stock - issue is resolved | Title: [upstream_ut]  AssertionError: RuntimeError not raised | PatrykWilczewski | daisyden |  | ut |
| 6 | 2536 | Title: [upstream_ut]  AttributeError: module 'torch._C' has no attribute | All test cases passed on XPU/stock - issue is resolved | Title: [upstream_ut]  AttributeError: module 'torch._C' has no attribute | daisyden | daisyden |  | ut |
| 7 | 2541 | Title: [upstream_ut]  RuntimeError: could not construct a memory descriptor using strides | All test cases passed on XPU/stock - issue is resolved | Title: [upstream_ut]  RuntimeError: could not construct a memory descriptor using strides | yucai-intel | daisyden |  | ut |
| 8 | 2811 | [Bug Skip]: [Regression] failed cases 2026-2-2 | All test cases passed on XPU/stock - issue is resolved | [Bug Skip]: [Regression] failed cases 2026-2-2 | jmamzax | kaileiyx | merged | ut |
| 9 | 2845 | [Bug Skip]:[UT] [Windows] failed cases 2026-2-4 | All test cases passed on XPU/stock - issue is resolved | [Bug Skip]:[UT] [Windows] failed cases 2026-2-4 | None | kaileiyx |  | ut |
| 10 | 2852 | [Bug Skip]: New UT failures in 0206 nightly on Windows | All test cases passed on XPU/stock - issue is resolved | [Bug Skip]: New UT failures in 0206 nightly on Windows | None | chuanqi129 |  | ut |
| 11 | 3106 | Worker crashes when running TestDecompXPU,test_quick_core_backward_baddbmm_xpu_float64 in CI. | All test cases passed on XPU/stock - issue is resolved | Worker crashes when running TestDecompXPU,test_quick_core_backward_baddbmm_xpu_float64 in CI. | BBBela | BBBela |  | ut |
| 12 | 3158 | AttributeError: module 'triton.compiler' has no attribute 'OutOfResources' | All test cases passed on XPU/stock - issue is resolved | AttributeError: module 'triton.compiler' has no attribute 'OutOfResources' | tadkrawiec | kdrozd-dev |  | ut |
| 13 | 3160 | compiler not found (Windows) | All test cases passed on XPU/stock - issue is resolved | compiler not found (Windows) | kdrozd-dev | kdrozd-dev |  | ut |
| 14 | 3161 | Exception: Tensor-likes are not close! - test_vjp_linalg_tensorsolve_xpu_float32 | All test cases passed on XPU/stock - issue is resolved | Exception: Tensor-likes are not close! - test_vjp_linalg_tensorsolve_xpu_float32 | tadkrawiec | kdrozd-dev |  | ut |
| 15 | 3174 | [Bug Skip]: Accuracy failure of test_Conv2d_groups_nobias | All test cases passed on XPU/stock - issue is resolved | [Bug Skip]: Accuracy failure of test_Conv2d_groups_nobias | pbielak | CuiYifeng |  | ut |
| 16 | 3267 | New failed test cases 2026-04-06 | All test cases passed on XPU/stock - issue is resolved | New failed test cases 2026-04-06 | None | zxd1997066 |  | ut |
| 17 | 3280 | [Bug Skip]: New UT failure in 0406 nightly windows. | All test cases passed on XPU/stock - issue is resolved | [Bug Skip]: New UT failure in 0406 nightly windows. | None | RUIJIEZHONG66166 |  | ut |
| | | **Subtotal: 17 issues** | | | | | |

### <span id='verify-the-issue'>Verify the issue</span> (14 issues)

| # | ID | Title | Action Reason | Summary | Assignee | Owner Transfer | PR Status | Test Module |
|--:|----|-------|---------------|---------|----------|----------------|-----------|-------------|
| 18 | 1171 | LNL Windows got unexpected error message | PR closed but no failed tests - verify if issue still reproduces | LNL Windows got unexpected error message | xuhancn, chunhuanMeng | daisyden | closed | ut |
| 19 | 1594 | Keep track on the building warning | PR closed but no failed tests - verify if issue still reproduces | Keep track on the building warning | CuiYifeng, chunhuanMeng | toyxu | merged | ut |
| 20 | 2219 | float8_e4m3fn precision overflow | PR closed but no failed tests - verify if issue still reproduces | float8_e4m3fn precision overflow | CuiYifeng, yucai-intel | jiqing-feng | merged | ut |
| 21 | 2255 | [upstream_ut] RuntimeError: Long is not supported in oneDNN | PR closed but no failed tests - verify if issue still reproduces | [upstream_ut] RuntimeError: Long is not supported in oneDNN | daisyden | daisyden | closed | ut |
| 22 | 2329 | [upstream_ut] feature missing: get_device_tflops and get_drams_gbps | PR closed but no failed tests - verify if issue still reproduces | [upstream_ut] feature missing: get_device_tflops and get_drams_gbps | etaf | daisyden | merged | ut |
| 23 | 2331 | [upstream_ut] AssertionError: Scalars are not equal! with test_prune_configs_over_shared_memory_limit | PR closed but no failed tests - verify if issue still reproduces | [upstream_ut] AssertionError: Scalars are not equal! with test_prune_configs_over_shared_memory_limit | hoshibara | daisyden | merged | ut |
| 24 | 2439 | [oneDNN] TestDecompXPU.test_quick_addmv_xpu_float64 got fail accuracy result | PR closed but no failed tests - verify if issue still reproduces | [oneDNN] TestDecompXPU.test_quick_addmv_xpu_float64 got fail accuracy result | libohao1201 | mengfei25 | merged | ut |
| 25 | 2535 | Title: [upstream_ut]  AttributeError: module 'torch._C' has no attribute '_cuda_tunableop_get_rotating_buffer_size' | PR closed but no failed tests - verify if issue still reproduces | Title: [upstream_ut]  AttributeError: module 'torch._C' has no attribute '_cuda_tunableop_get_rotating_buffer_size' | Silv3S, BartoszKokoszko | daisyden | merged | ut |
| 26 | 2619 | [release/2.10] Some models inductor performance dropped ~ 10% - 30% | PR closed but no failed tests - verify if issue still reproduces | [release/2.10] Some models inductor performance dropped ~ 10% - 30% | jianyizh, weishi-deng, mengfei25 | mengfei25 | merged | e2e |
| 27 | 2649 | [distributed][pipelining] test_schedule_multiproc.py hang issue | PR closed but no failed tests - verify if issue still reproduces | [distributed][pipelining] test_schedule_multiproc.py hang issue | syedshahbaaz | zxd1997066 | merged | ut |
| 28 | 2694 | Title: [upstream_ut]  AssertionError: Tensor-likes are not equal! with test_randint tests | PR closed but no failed tests - verify if issue still reproduces | Title: [upstream_ut]  AssertionError: Tensor-likes are not equal! with test_randint tests | daisyden | daisyden | merged | ut |
| 29 | 2744 | [Bug Skip]: extended test failures when test_compare_cpu atol and rtol changed | PR closed but no failed tests - verify if issue still reproduces | [Bug Skip]: extended test failures when test_compare_cpu atol and rtol changed | pbielak | daisyden | merged | ut |
| 30 | 2806 | CompiledAOTI need XPU support | PR closed but no failed tests - verify if issue still reproduces | CompiledAOTI need XPU support | daisyden | daisyden | merged | ut |
| 31 | 3007 | AssertionError: Scalars are not equal! with test_flash_attention_dynamic | PR closed but no failed tests - verify if issue still reproduces | AssertionError: Scalars are not equal! with test_flash_attention_dynamic | daisyden | daisyden | merged | e2e |
| | | **Subtotal: 14 issues** | | | | | |

### <span id='add-to-skiplist'>add to skiplist</span> (5 issues)

| # | ID | Title | Action Reason | Summary | Assignee | Owner Transfer | PR Status | Test Module |
|--:|----|-------|---------------|---------|----------|----------------|-----------|-------------|
| 32 | 2164 | skip test_no_cuda_monkeypatch as it is cuda specific | Issue marked as not_target/wontfix - should be skipped for XPU enablement | skip test_no_cuda_monkeypatch as it is cuda specific | daisyden | daisyden |  | ut |
| 33 | 2309 | unsupported ops with PYTORCH_ENABLE_XPU_FALLBACK unset | Issue marked as not_target/wontfix - should be skipped for XPU enablement | unsupported ops with PYTORCH_ENABLE_XPU_FALLBACK unset | daisyden | daisyden |  | ut |
| 34 | 2472 | [upstream_ut]  NotImplementedError: The operator 'aten::_cudnn_rnn' is not currently implemented for the XPU devic
 | Issue marked as not_target/wontfix - should be skipped for XPU enablement | [upstream_ut]  NotImplementedError: The operator 'aten::_cudnn_rnn' is not currently implemented for the XPU devic
 | Silv3S | daisyden |  | ut |
| 35 | 2508 | TypedStorage / TypedTensors deprecation | Issue marked as not_target/wontfix - should be skipped for XPU enablement | TypedStorage / TypedTensors deprecation | Silv3S | libohao1201 | open | ut |
| 36 | 3127 | [upstream_ut]  AssertionError: AssertionError not raised 
 | Issue marked as not_target/wontfix - should be skipped for XPU enablement | [upstream_ut]  AssertionError: AssertionError not raised 
 | daisyden | daisyden |  | ut |
| | | **Subtotal: 5 issues** | | | | | |

### <span id='revisit-the-pr-as-case-failed'>Revisit the PR as case failed</span> (2 issues)

| # | ID | Title | Action Reason | Summary | Assignee | Owner Transfer | PR Status | Test Module |
|--:|----|-------|---------------|---------|----------|----------------|-----------|-------------|
| 37 | 3242 | AssertionError: Torch not compiled with CUDA enabled | PR closed but tests still failing - revisit PR for fix | AssertionError: Torch not compiled with CUDA enabled | Silv3S | Silv3S | merged | ut |
| 38 | 3246 | AssertionError: Booleans mismatch: True is not False | PR closed but tests still failing - revisit PR for fix | AssertionError: Booleans mismatch: True is not False | BartoszKokoszko | BartoszKokoszko | merged | ut |
| | | **Subtotal: 2 issues** | | | | | |

[Back to Index](#toc) |

## <span id='4-duplicated-issues'>4. Duplicated Issues</span>

**Total: 38 issues** - Issues sharing test cases with other issues

| # | ID | Title | Summary | Assignee | Priority | Root Cause | Dependency | Duplicated With | Test Module |
|--:|----|-------|---------|----------|---------|-----------|-----------|----------------|-------------|
| 1 | 1893 | [upstream_ut] oneDNN accuracy issues in test_ops_xpu.py | [upstream_ut] oneDNN accuracy issues in test_ops_xpu.py | chunhuanMeng | P2 | DNNL/OneDNN Issue | None | 1951 | ut |
| 2 | 1951 | Functionality issues in TestCommon.test_out. | Functionality issues in TestCommon.test_out. | AKloniecki | P2 | Backend/Device Issue | None | 1893 | ut |
| 3 | 1973 | AssertionError: Scalars or Tensor-likes are not equal or close! | AssertionError: Scalars or Tensor-likes are not equal or close! | gplutop7 | P2 | Failure | None | 2837 | ut |
| 4 | 2006 | work-item/workgroup issue in softmax/unsampling/nonzero | work-item/workgroup issue in softmax/unsampling/nonzero | BartoszKokoszko | P2 | Backend/Device Issue | None | 2257 | ut |
| 5 | 2015 | inf is returned by nn.TransformerEncoderLayer | inf is returned by nn.TransformerEncoderLayer | yucai-intel | P2 | precision-related instability in the layer's computation, likely due to low-precision (fp16) operati | None | 2529 | ut |
| 6 | 2186 | AssertionError: Mul tiheadAttention does not support NestedTensor outside of its fast path | AssertionError: Mul tiheadAttention does not support NestedTensor outside of its fast path | daisyden | P2 | Failure | oneDNN | 2015 | ut |
| 7 | 2220 | test/test_sparse_csr.py::TestSparseCompressedTritonKernelsXPU::test_triton_bsr_scatter_mm_blocksize_16_xpu_bfloat16 will meet InvalidModule: Invalid SPIR-V module: input SPIR-V module uses unknown extension 'SPV_INTEL_subgroup_matrix_multiply_accumulate' | test/test_sparse_csr.py::TestSparseCompressedTritonKernelsXPU::test_triton_bsr_scatter_mm_blocksize_16_xpu_bfloat16 will meet InvalidModule: Invalid S | None | P2 | Backend/Device Issue | Triton | 2246 | ut |
| 8 | 2230 | test_sparse_csr.py::TestSparseCompressedTritonKernelsXPU::test_triton_ meet ValueError: all inputs are expected to be on the same GPU device. | test_sparse_csr.py::TestSparseCompressedTritonKernelsXPU::test_triton_ meet ValueError: all inputs are expected to be on the same GPU device. | None | P2 | Backend/Device Issue | None | 3176 | ut |
| 9 | 2244 | test/test_sparse_csr.py::TestSparseCSRXPU::test_block_addmm meet RuntimeError: empty_sparse_compressed expected sparse compressed (non-block) tensor layout but got SparseBsr | test/test_sparse_csr.py::TestSparseCSRXPU::test_block_addmm meet RuntimeError: empty_sparse_compressed expected sparse compressed (non-block) tensor l | jenniew | P2 | Error | None | 3177 | ut |
| 10 | 2246 | torch/sparse/_triton_ops*.py need to be ported to enable for Intel GPU for test_sparse and test_sparse_csr cases | torch/sparse/_triton_ops*.py need to be ported to enable for Intel GPU for test_sparse and test_sparse_csr cases | None | P2 | Backend/Device Issue | None | 2230 | ut |
| 11 | 2253 | the supported dtypes are not align with cuda | the supported dtypes are not align with cuda | daisyden | P2 | Dtype/Precision Issue | None | 2482 | ut |
| 12 | 2257 | Accuracy failures in test/xpu/test_unary_ufuncs_xpu.py | Accuracy failures in test/xpu/test_unary_ufuncs_xpu.py | pbielak | P2 | Dtype/Precision Issue | None | 2006 | ut |
| 13 | 2270 | Backend Compatibility Error in test/xpu/test_decomp.py | Backend Compatibility Error in test/xpu/test_decomp.py | LuFinch | P2 | flash_attention_forward operator is not available for the CPU backend, and the test attempts to run  | None | 2442 | ut |
| 14 | 2285 | Support efficient attention | Support efficient attention | chunhuanMeng | P2 | attention_forward_xpu_float16 fails on XPU because the aten._efficient_attention_forward op is not p | None | 2358 | ut |
| 15 | 2358 | test/test_view_ops.py::TestOldViewOpsXPU::test_ravel_xpu meet NotImplementedError: Could not run 'aten::_empty_affine_quantized' with arguments from the 'QuantizedXPU' backend | test/test_view_ops.py::TestOldViewOpsXPU::test_ravel_xpu meet NotImplementedError: Could not run 'aten::_empty_affine_quantized' with arguments from t | Silv3S | P2 | Backend/Device Issue | None | 2285 | ut |
| 16 | 2436 | [upstream_ut]  AttributeError: 'NoneType' object has no attribute 'clone' 
 | [upstream_ut]  AttributeError: 'NoneType' object has no attribute 'clone' 
 | daisyden | P1 | Error | None | 2675 | ut |
| 17 | 2442 | [Bug Skip]: NotImplementedError: Could not run 'aten::_flash_attention_forward' with arguments from the 'CPU' backend | [Bug Skip]: NotImplementedError: Could not run 'aten::_flash_attention_forward' with arguments from the 'CPU' backend | daisyden, LuFinch | P2 | flash_attention_forward is invoked with CPU backend arguments in a test expecting XPU execution. The | None | 2270 | ut |
| 18 | 2482 | test_dtypes issue introduced by pytorch test sample input updates | test_dtypes issue introduced by pytorch test sample input updates | daisyden | P2 | Dtype/Precision Issue | None | 2253 | ut |
| 19 | 2529 | [upstream_ut]  AssertionError: False is not true | [upstream_ut]  AssertionError: False is not true | Silv3S | P2 | Failure | None | 2015 | ut |
| 20 | 2530 | Title: [upstream_ut]  AssertionError: RuntimeError not raised | Title: [upstream_ut]  AssertionError: RuntimeError not raised | PatrykWilczewski | P2 | Mismatch | None | 2817 | ut |
| 21 | 2618 | [Bug Skip]: [regression] AssertionError: Scalars are not close! AssertionError: Tensor-likes are not close! | [Bug Skip]: [regression] AssertionError: Scalars are not close! AssertionError: Tensor-likes are not close! | jmamzax | P0 | device-specific misconfiguration on XPU. The test attempts to run a CUDA-related max pooling operati | None | 3089 | ut |
| 22 | 2675 | [Bug Skip]: AttributeError: 'NoneType' object has no attribute 'clone' | [Bug Skip]: AttributeError: 'NoneType' object has no attribute 'clone' | pponikox | P2 | Skip/No Test Exists | None | 2436 | ut |
| 23 | 2817 | Expected error message is different than actual | Expected error message is different than actual | kdrozd-dev | P2 | Mismatch | None | 2530 | ut |
| 24 | 2837 | Accuracy issue for Muon optimizer | Accuracy issue for Muon optimizer | Silv3S | P2 | Dtype/Precision Issue | None | 1973 | ut |
| 25 | 2840 | Accuracy issue with 64 bit indexing depthwise_conv | Accuracy issue with 64 bit indexing depthwise_conv | SlawomirLaba, Silv3S | P2 | Dtype/Precision Issue | oneDNN | 1973 | ut |
| 26 | 2845 | [Bug Skip]:[UT] [Windows] failed cases 2026-2-4 | [Bug Skip]:[UT] [Windows] failed cases 2026-2-4 | None | P2 | Skip/No Test Exists | None | 2852 | ut |
| 27 | 2852 | [Bug Skip]: New UT failures in 0206 nightly on Windows | [Bug Skip]: New UT failures in 0206 nightly on Windows | None | P2 | Skip/No Test Exists | None | 2845 | ut |
| 28 | 2869 | [Bug Skip]: New UT failure in 0209 nightly windows. | [Bug Skip]: New UT failure in 0209 nightly windows. | None | P2 | Failure | None | 3160 | ut |
| 29 | 2966 | [Bug Skip]: [Regression]2026-3-2 ut failures | [Bug Skip]: [Regression]2026-3-2 ut failures | jmamzax | P0 | Skip/No Test Exists | None | 3114 | ut |
| 30 | 3089 | AssertionError: Torch not compiled with CUDA enabled | AssertionError: Torch not compiled with CUDA enabled | jmamzax | P2 | Backend/Device Issue | None | 2618 | ut |
| 31 | 3114 | [Bug Skip]: Failure skip on 2026-3-21 | [Bug Skip]: Failure skip on 2026-3-21 | None | P2 | Skip/No Test Exists | None | 2966 | ut |
| 32 | 3136 | [upstream_ut]  AssertionError: False is not true in test_transformers | [upstream_ut]  AssertionError: False is not true in test_transformers | LuFinch | P2 | Failure | None | 2529 | ut |
| 33 | 3156 | AssertionError: 'Assertion cur_target >= 0 && cur_target <   n_classes failed' not found | AssertionError: 'Assertion cur_target >= 0 && cur_target <   n_classes failed' not found | kdrozd-dev | P2 | API_USAGE error related to torch.python.import. This suggests an XPU-specific backend issue with err | None | 3184 | ut |
| 34 | 3160 | compiler not found (Windows) | compiler not found (Windows) | kdrozd-dev | P2 | Backend/Device Issue | None | 2869 | ut |
| 35 | 3175 | [Bug Skip]: ValueError: sampled_addmm(): all inputs are expected to be on the same GPU device | [Bug Skip]: ValueError: sampled_addmm(): all inputs are expected to be on the same GPU device | None | P2 | Backend/Device Issue | None | 2230 | ut |
| 36 | 3176 | [Bug Skip]: ValueError: _scaled_dot_product_attention(): all inputs are expected to be on the same GPU device | [Bug Skip]: ValueError: _scaled_dot_product_attention(): all inputs are expected to be on the same GPU device | None | P2 | Backend/Device Issue | None | 2230 | ut |
| 37 | 3177 | Accuracy gap of BF16/FP16 test_block_addmm | Accuracy gap of BF16/FP16 test_block_addmm | jenniew | P2 | Dtype/Precision Issue | None | 2244 | ut |
| 38 | 3184 | New failing UTs: test_cross_entropy_loss_2d_out_of_bounds_class_index_xpu | New failing UTs: test_cross_entropy_loss_2d_out_of_bounds_class_index_xpu | wpietka | P2 | API_USAGE error related to torch.python.import. This suggests a device-specific handling or compilat | None | 3156 | ut |
| | | **Subtotal: 38 issues** | | | | | | |

[Back to Index](#toc) |

## <span id='5-issues-with-dependency'>5. Issues with Dependency</span>

**Total: 35 issues** - Issues with external dependencies

| # | ID | Title | Summary | Assignee | Priority | Root Cause | Category | Dependency | PR Status | Test Module |
|--:|----|-------|---------|----------|---------|-----------|----------|------------|-----------|-------------|
| 1 | 1059 | SYCL RT: Using recommended shortcut API for kernel specific max work group size. | SYCL RT: Using recommended shortcut API for kernel specific max work group size. | CuiYifeng, jianyizh | P2 | backend-specific, involving SYCL kernel configuration without direct PyTorch op or API involvement. | unknown | oneAPI |  | ut |
| 2 | 1171 | LNL Windows got unexpected error message | LNL Windows got unexpected error message | xuhancn, chunhuanMeng | P2 | device-specific failure in kernel initialization or execution. The error lacks a traceback, suggesti | unknown | driver | closed | ut |
| 3 | 1324 | [Win] UR Error when OOM and break the tensor context | [Win] UR Error when OOM and break the tensor context | Stonepia | P2 | Memory/Shared Memory Issue | unknown | oneAPI |  | ut |
| 4 | 1547 | [distributed] NotImplementedError: The operator 'symm_mem::fused_matmul_reduce_scatter' is not currently implemented for the XPU device | [distributed] NotImplementedError: The operator 'symm_mem::fused_matmul_reduce_scatter' is not currently implemented for the XPU device | Chao1Han | P2 | Backend/Device Issue | unknown | oneAPI |  | ut |
| 5 | 1548 | [distributed] AssertionError: 'fused_all_gather_matmul' not found in '# AOT ID: [\'2_inference\']\n......' | [distributed] AssertionError: 'fused_all_gather_matmul' not found in '# AOT ID: [\'2_inference\']\n......' | Chao1Han | P2 | Distributed/Gloo Issue | unknown | oneAPI |  | ut |
| 6 | 1549 | [distributed] AssertionError: 'fused_all_gather_scaled_matmul' not found in 'graph():\n......' | [distributed] AssertionError: 'fused_all_gather_scaled_matmul' not found in 'graph():\n......' | Chao1Han | P2 | Distributed/Gloo Issue | unknown | oneAPI |  | ut |
| 7 | 1551 | [distributed] NotImplementedError: The operator 'symm_mem::fused_scaled_matmul_reduce_scatter' is not currently implemented for the XPU device. | [distributed] NotImplementedError: The operator 'symm_mem::fused_scaled_matmul_reduce_scatter' is not currently implemented for the XPU device. | Chao1Han | P2 | Backend/Device Issue | unknown | oneAPI |  | ut |
| 8 | 1555 | [distributed] RuntimeError: aten.add.Tensor: got mixed torch.Tensor and DTensor, need to convert all torch.Tensor to DTensor before calling distributed operators! | [distributed] RuntimeError: aten.add.Tensor: got mixed torch.Tensor and DTensor, need to convert all torch.Tensor to DTensor before calling distribute | chuanqi129 | P2 | Distributed/Gloo Issue | unknown | oneDNN |  | ut |
| 9 | 1556 | [distributed] NotImplementedError: Operator aten._scaled_dot_product_fused_attention_overrideable.default does not have a sharding strategy registered. | [distributed] NotImplementedError: Operator aten._scaled_dot_product_fused_attention_overrideable.default does not have a sharding strategy registered | pkourdis | P2 | attention_overrideable.default is invoked without a registered sharding strategy. This operator is u | unknown | oneDNN |  | ut |
| 10 | 1624 | [DONT CLOSE] Known UT Issue list | [DONT CLOSE] Known UT Issue list | None | P2 | Others | unknown | oneCCL |  | ut |
| 11 | 1649 | [cpp extension] Provide a clear error message when using inconsistent oneapi versions. | [cpp extension] Provide a clear error message when using inconsistent oneapi versions. | dvrogozh | P2 | Backend/Device Issue | unknown | oneAPI |  | ut |
| 12 | 1722 | Ask an API to query GPU type(iGPU/dGPU). | Ask an API to query GPU type(iGPU/dGPU). | guangyey | P2 | Mismatch | unknown | oneAPI |  | ut |
| 13 | 1727 | [distributed] AttributeError: module 'torch.xpu' has no attribute '_sleep' | [distributed] AttributeError: module 'torch.xpu' has no attribute '_sleep' | guangyey | P2 | Backend/Device Issue | unknown | oneAPI |  | ut |
| 14 | 1912 | Implement the torch.ops.aten._weight_int4pack_mm for dequantizing the  CUDA int4 layout | Implement the torch.ops.aten._weight_int4pack_mm for dequantizing the  CUDA int4 layout | liangan1 | P2 | device-specific and requires backend support. The operation is not yet supported on XPU, leading to  | unknown | oneDNN |  | ut |
| 15 | 1986 | torch.xpu._sleep is missing, | torch.xpu._sleep is missing, | guangyey | P2 | Mismatch | unknown | oneAPI |  | ut |
| 16 | 2089 | need an implementation that won't initialize gpu context for torch.xpu.is_available() | need an implementation that won't initialize gpu context for torch.xpu.is_available() | guangyey | P2 | Backend/Device Issue | unknown | driver |  | ut |
| 17 | 2186 | AssertionError: Mul tiheadAttention does not support NestedTensor outside of its fast path | AssertionError: Mul tiheadAttention does not support NestedTensor outside of its fast path | daisyden | P2 | Failure | unknown | oneDNN |  | ut |
| 18 | 2200 | support flash attention op on XPU device | support flash attention op on XPU device | ElaineBao | P2 | flash_attention op in the XPU backend. No test case or error traceback is provided, but the request  | unknown | oneDNN |  | ut |
| 19 | 2220 | test/test_sparse_csr.py::TestSparseCompressedTritonKernelsXPU::test_triton_bsr_scatter_mm_blocksize_16_xpu_bfloat16 will meet InvalidModule: Invalid SPIR-V module: input SPIR-V module uses unknown extension 'SPV_INTEL_subgroup_matrix_multiply_accumulate' | test/test_sparse_csr.py::TestSparseCompressedTritonKernelsXPU::test_triton_bsr_scatter_mm_blocksize_16_xpu_bfloat16 will meet InvalidModule: Invalid S | None | P2 | Backend/Device Issue | unknown | Triton |  | ut |
| 20 | 2261 | [xpu][profiler] Run with fork process has extra warning | [xpu][profiler] Run with fork process has extra warning | moksiuc | P2 | device-specific issue during process spawning. This occurs regardless of specific ops or dtypes, sug | Others | oneAPI |  | ut |
| 21 | 2329 | [upstream_ut] feature missing: get_device_tflops and get_drams_gbps | [upstream_ut] feature missing: get_device_tflops and get_drams_gbps | etaf | P2 | Supported | Feature Not Supported | Triton | merged | ut |
| 22 | 2331 | [upstream_ut] AssertionError: Scalars are not equal! with test_prune_configs_over_shared_memory_limit | [upstream_ut] AssertionError: Scalars are not equal! with test_prune_configs_over_shared_memory_limit | hoshibara | P2 | memory_limit_do_pruning_True fails with an AssertionError due to shared memory limit violations on X | Inductor/Compilation | oneAPI | merged | ut |
| 23 | 2439 | [oneDNN] TestDecompXPU.test_quick_addmv_xpu_float64 got fail accuracy result | [oneDNN] TestDecompXPU.test_quick_addmv_xpu_float64 got fail accuracy result | libohao1201 | P2 | DNNL/OneDNN Issue | Dtype/Precision | oneDNN | merged | ut |
| 24 | 2467 | Host may stuck when submit too many kernels when event recording | Host may stuck when submit too many kernels when event recording | jianyizh | P2 | Backend/Device Issue | Torch Runtime | driver |  | ut |
| 25 | 2570 | crash in sdpa. | crash in sdpa. | LuFinch | P0 | Backend/Device Issue | Flash Attention/Transformer | oneDNN |  | ut |
| 26 | 2611 | [upstream_ut]  AssertionError: Tensor-likes are not equal! in test_compile_subprocess | [upstream_ut]  AssertionError: Tensor-likes are not equal! in test_compile_subprocess | daisyden | P2 | Inductor/Compilation Issue | Inductor/Compilation | driver |  | ut |
| 27 | 2613 | [upstream_ut]  AssertionError: Tensor-likes are not equal! in test_compile_subprocess.py | [upstream_ut]  AssertionError: Tensor-likes are not equal! in test_compile_subprocess.py | daisyden | P2 | Failure | Inductor/Compilation | driver |  | ut |
| 28 | 2655 | [BMG][OOB] hf_Reformer performance drop | [BMG][OOB] hf_Reformer performance drop | jianyizh | P0 | Timeout/Performance Issue | Dtype/Precision | Triton |  | e2e |
| 29 | 2769 | [oneDNN] New failed test cases with 3.11 compared with 3.10 | [oneDNN] New failed test cases with 3.11 compared with 3.10 | LuFinch | P2 | DNNL/OneDNN Issue | Others | oneDNN |  | ut |
| 30 | 2800 | AttributeError: 'torch._C._XpuDeviceProperties' object has no attribute 'major' | AttributeError: 'torch._C._XpuDeviceProperties' object has no attribute 'major' | guangyey | P2 | Backend/Device Issue | Inductor/Compilation | oneAPI |  | ut |
| 31 | 2840 | Accuracy issue with 64 bit indexing depthwise_conv | Accuracy issue with 64 bit indexing depthwise_conv | SlawomirLaba, Silv3S | P2 | Dtype/Precision Issue | Dtype/Precision | oneDNN |  | ut |
| 32 | 2924 | [release/2.11] xcit_large_24_p8_224 amp_bf16 training got fail_accuracy | [release/2.11] xcit_large_24_p8_224 amp_bf16 training got fail_accuracy | jianyizh, mengfei25 | P1 | Dtype/Precision Issue | Dtype/Precision | Triton |  | e2e |
| 33 | 2979 | eca_halonext26ts got RuntimeError: ZE_RESULT_ERROR_MODULE_BUILD_FAILURE | eca_halonext26ts got RuntimeError: ZE_RESULT_ERROR_MODULE_BUILD_FAILURE | None | P0 | ERROR_MODULE_BUILD_FAILURE occurs during XPU kernel compilation for eca_halonext26ts model. The erro | Torch Runtime | driver |  | e2e |
| 34 | 3148 | [Triton] Huggingface openai/whisper-tiny got fail_accuracy | [Triton] Huggingface openai/whisper-tiny got fail_accuracy | None | P1 | device-specific handling of attention operations, possibly involving aten._scaled_dot_product_attent | Inductor/Compilation | Triton |  | e2e |
| 35 | 3151 | [Triton] Timm_models  rexnet_100 / fbnetv3_b / sebotnet33ts_256 got fail_accuracy | [Triton] Timm_models  rexnet_100 / fbnetv3_b / sebotnet33ts_256 got fail_accuracy | None | P0 | device-specific backend issue. The error likely stems from improper model execution or kernel behavi | Inductor/Compilation | Triton |  | e2e |
| | | **Subtotal: 35 issues** | | | | | | | |

[Back to Index](#toc) |

## <span id='6-statistics'>6. Statistics</span>

### <span id='stats-dependency'>By Dependency</span>

| Dependency | Count |
|------------|-------|
| oneAPI | 13 |
| oneDNN | 9 |
| Triton | 6 |
| driver | 6 |
| oneCCL | 1 |

[Back to Index](#toc) |

---
*Report generated with 374 issues (excluded 10 enhancement-labeled issues)*
