# Torch XPU Ops UT Issue Report

**Repository:** [intel/torch-xpu-ops](https://github.com/intel/torch-xpu-ops)

**Report Type:** UT (Unit Test) Issues Only

**Generated:** 2026-04-08 22:51:49

---

## Index

1. [Summary](#1-summary)
2. [Statistics](#2-statistics)
   - [By Module](#by-module)
   - [By Action TBD](#by-action-tbd)
   - [By Category](#by-category)
   - [By Priority](#by-priority)
3. [New Submitted Issues (Past Week)](#3-new-submitted-issues-past-week)
4. [Action Required](#4-action-required)
   - [Reporter Actions](#reporter-actions)
     - [Information Required](#information-required)
     - [Close Fixed Issue](#close-fixed-issue)
     - [Enable Test](#enable-test)
     - [Add to Skiplist](#add-to-skiplist)
     - [Verify the Issue](#verify-the-issue)
     - [Need Reproduce Steps](#need-reproduce-steps)
   - [Engineer Actions](#engineer-actions)
     - [Needs PyTorch Repo Changes (upstream)](#needs-pytorch-repo-changes-upstream)
     - [Revisit the PR as Case Failed](#revisit-the-pr-as-case-failed)
5. [By Category](#5-by-category)
6. [Duplicated Issues](#6-duplicated-issues)
7. [Issues with Dependency](#7-issues-with-dependency)

---

## 1. Summary {#1-summary}

| Category | Count |
|----------|-------|
| Action Required | 303 |
| No Assignee | 3 |
| Duplicated Issues | 42 |
| With Dependency | 5 |
| Others | 18 |
| **Total UT Issues** | 371 |

---

## 2. Statistics {#2-statistics}

### By Module

| Module | Count |
|--------|-------|
| aten_ops | 269 |
| distributed | 39 |
| inductor | 24 |
| AO | 17 |
| unknown | 15 |
| profiling | 5 |
| low_precision | 2 |

### By Action TBD

| Action TBD | Count |
|------------|-------|
| Need reproduce steps (Only for bugs or performance issue) | 162 |
| Needs PyTorch Repo Changes (upstream) | 89 |
| Need more information - error logs and reproduction steps | 72 |
| Close fixed issue | 12 |
| Revisit the PR as case failed | 4 |
| add to skiplist | 4 |
| Verify the issue | 2 |

### By Category

| Category | Count |
|----------|-------|
| Distributed | 105 |
| Others | 104 |
| Dtype / Precision Related | 44 |
| Flash Attention / Transformer Related | 41 |
| TorchAO | 32 |
| Inductor / Compilation Related | 19 |
| Sparse Operations Related | 14 |
| PT2E | 12 |

### By Priority

| Priority | Count |
|----------|-------|
| P0 | 22 |
| P1 | 10 |
| P2 | 339 |

---

## 3. New Submitted Issues (Past Week) {#3-new-submitted-issues-past-week}

Issues created in the past 7 days (as of 2026-04-08).

| ID | Title | Status | Owner | Priority | Reason | Category | Root Cause | Labels | Module |
|---|-------|--------|-------|---------|--------|----------|-----------|--------|--------|
| [3259](https://github.com/intel/torch-xpu-ops/issues/3259) | New failed test cases 2026-04-02 | open | SlawomirLaba | P2 | UT issue with few failures | Flash Attention / Transformer Related | Backend/Device Issue - XPU device initialization or compatibility failure | skipped | aten_ops |
| [3258](https://github.com/intel/torch-xpu-ops/issues/3258) | Error in op: torch.ops.aten._scaled_dot_ | open | None | P2 | UT issue with few failures | Flash Attention / Transformer Related | 5 - Flash Attention/Specific Ops Issue - Error in _scaled_dot_product_fused_atte |  | aten_ops |

---

## 4. Action Required {#4-action-required}

### Reporter Actions {#reporter-actions}

#### Information Required {#information-required}

| ID | Title | Owner | Owner Transferred | Priority | Category | Root Cause | PR |
|---|-------|-------|-------------------|---------|----------|-----------|-----|
| [1059](https://github.com/intel/torch-xpu-ops/issues/1059) | SYCL RT: Using recommended shortcut API  | CuiYifeng, jianyizh | fengyuan14 | P2 | Distributed | Backend/Device Issue - related to SYCL RT and kernel-specific max work group siz |  |
| [1510](https://github.com/intel/torch-xpu-ops/issues/1510) | Some test cases will be hang on BMG Ubun | Stonepia, mengfei25 | mengfei25 | P2 | Others | Backend/Device Issue - test cases hang on BMG Ubuntu related to XPU backend |  |
| [1587](https://github.com/intel/torch-xpu-ops/issues/1587) | Keep track on the latest CUDA op impl | CuiYifeng, yucai-intel | toyxu | P2 | Others | Skip/No Test Exists - no test or error traceback provided |  |
| [1594](https://github.com/intel/torch-xpu-ops/issues/1594) | Keep track on the building warning | CuiYifeng, chunhuanMeng | toyxu | P0 | Others | Others - building warning tracking issue |  |
| [1645](https://github.com/intel/torch-xpu-ops/issues/1645) | [For Comparison] Save reference comparis | mengfei25 | mengfei25 | P2 | Others | Skip/No Test Exists - no test or error information provided |  |
| [1678](https://github.com/intel/torch-xpu-ops/issues/1678) | missing op support for `model.share_memo | None | jafraustro | P0 | Others | Memory/Shared Memory Issue - missing op support for `model.share_memory()` indic |  |
| [1689](https://github.com/intel/torch-xpu-ops/issues/1689) | [For op Perf Comparison] Save reference  | None | RUIJIEZHONG66166 | P2 | Others | Skip/No Test Exists - no test or error information provided |  |
| [1722](https://github.com/intel/torch-xpu-ops/issues/1722) | Ask an API to query GPU type(iGPU/dGPU). | guangyey | xuhancn | P2 | Others | Mismatch - Request for an API to query GPU type (iGPU/dGPU) is missing or not properly implemented. |  |
| [1729](https://github.com/intel/torch-xpu-ops/issues/1729) | Validation Check List | chuanqi129 | EikanWang | P2 | Others | Skip/No Test Exists - no test or error information provided |  |
| [1745](https://github.com/intel/torch-xpu-ops/issues/1745) | torch.xpu.empty_cache() introduces memor | guangyey | songhappy | P2 | Others | Memory/Shared Memory Issue - torch.xpu.empty_cache() introduces memory leak |  |
| [1762](https://github.com/intel/torch-xpu-ops/issues/1762) | Add an ocloc AOT target compilation test | chunhuanMeng | jingxu10 | P2 | PT2E | compilation-related task or issue. |  |
| [1764](https://github.com/intel/torch-xpu-ops/issues/1764) | New kernels for concat | yucai-intel | jianyizh | P2 | Inductor / Compilation Related | Others - New kernels for concat, no specific error provided. |  |
| [1856](https://github.com/intel/torch-xpu-ops/issues/1856) | channel last aten::hardswish_ will call  | chunhuanMeng | jianyizh | P2 | Others | Memory/Shared Memory Issue - extra copy caused by channel last aten::hardswish_ |  |
| [1962](https://github.com/intel/torch-xpu-ops/issues/1962) | [upstream_ut] segfault with test_fake_cr | jenniew, mengfei25 | daisyden | P0 | Sparse Operations Related | Backend/Device Issue - segfault related to XPU device operation in test |  |
| [1986](https://github.com/intel/torch-xpu-ops/issues/1986) | torch.xpu._sleep is missing, | guangyey | githubsgi | P2 | Others | Mismatch - torch.xpu._sleep is not implemented or available in the current setup. |  |
| [1996](https://github.com/intel/torch-xpu-ops/issues/1996) | [TorchAO]  Memory Efficient Optimizers | arlesniak | liangan1 | P2 | TorchAO | Memory/Shared Memory Issue - related to memory efficient optimizers in TorchAO |  |
| [2113](https://github.com/intel/torch-xpu-ops/issues/2113) | Update example for Distributed Data Para | songhappy | luoyu-intel | P2 | Distributed | Distributed/Gloo Issue - related to Distributed Data Parallel update example |  |
| [2142](https://github.com/intel/torch-xpu-ops/issues/2142) | XPU max_memory_allocated have different  | guangyey | jiqing-feng | P2 | Others | Memory/Shared Memory Issue - XPU and CUDA show different memory allocation outpu |  |
| [2163](https://github.com/intel/torch-xpu-ops/issues/2163) | 3 distributed UT cases need to be suppor | githubsgi | libohao1201 | P2 | Distributed | Distributed/Gloo Issue - distributed UT cases need support from sac_estimator.py |  |
| [2195](https://github.com/intel/torch-xpu-ops/issues/2195) | Tools in pti should get 100% functionali | aostrowski-hbn | jianyizh | P2 | Others | Backend/Device Issue - functionality not working on BMG for PyTorch profiling |  |
| [2200](https://github.com/intel/torch-xpu-ops/issues/2200) | support flash attention op on XPU device | ElaineBao | Zjq9409 | P2 | Flash Attention / Transformer Related | Flash Attention/Specific Ops Issue - request to support flash attention op on XP |  |
| [2215](https://github.com/intel/torch-xpu-ops/issues/2215) | Find use case example for torch-xpu-ops. | dvrogozh | dvrogozh | P2 | Dtype / Precision Related | Others - Looking for use case example of torch-xpu-ops.lib in SYCL C++ extension |  |
| [2229](https://github.com/intel/torch-xpu-ops/issues/2229) | test/test_sparse_csr.py::TestSparseCompr | jenniew | wincent8 | P2 | Distributed | Backend/Device Issue - device mismatch between crow_indices and col_indices on X |  |
| [2232](https://github.com/intel/torch-xpu-ops/issues/2232) | sdpa backward kernel is required to redu | None | xin3he | P2 | Flash Attention / Transformer Related | Memory/Shared Memory Issue - sdpa backward kernel needed to reduce memory usage |  |
| [2250](https://github.com/intel/torch-xpu-ops/issues/2250) | Found mismatch when comparing the output | astachowiczhabana | daisyden | P2 | Flash Attention / Transformer Related | Mismatch - Mismatch in output of aten.view.default between FakeTensor and concrete Tensors. |  |
| [2261](https://github.com/intel/torch-xpu-ops/issues/2261) | [xpu][profiler] Run with fork process ha | moksiuc | chuanqi129 | P2 | Others | Backend/Device Issue - XPU profiler warning during fork process execution |  |
| [2323](https://github.com/intel/torch-xpu-ops/issues/2323) | [TorchAO] MOE training enabling on XPU | riverliuintel | liangan1 | P2 | TorchAO | Backend/Device Issue - MOE training not enabled on XPU |  |
| [2324](https://github.com/intel/torch-xpu-ops/issues/2324) | [TorchAO] FP8 conv support | Stonepia | liangan1 | P2 | TorchAO | Supported - FP8 conv is not supported yet in TorchAO |  |
| [2325](https://github.com/intel/torch-xpu-ops/issues/2325) | [TorchAO] Float8 training support on XPU | arlesniak, riverliuintel | liangan1 | P2 | TorchAO | Supported - Float8 training is not supported on XPU. |  |
| [2326](https://github.com/intel/torch-xpu-ops/issues/2326) | [TorchAO] MX training  native PyTorch on | riverliuintel | liangan1 | P2 | TorchAO | Backend/Device Issue - Training on XPU with TorchAO and native PyTorch is not fu |  |
| [2327](https://github.com/intel/torch-xpu-ops/issues/2327) | [TorchAO] benchmark enabling on XPU | None | liangan1 | P2 | TorchAO | Backend/Device Issue - XPU benchmark enabling issue in TorchAO |  |
| [2333](https://github.com/intel/torch-xpu-ops/issues/2333) | [Don't merge] Collect the new passed cas | None | mengfei25 | P2 | Others | Skip/No Test Exists - test is empty or not applicable |  |
| [2349](https://github.com/intel/torch-xpu-ops/issues/2349) | [oneAPI][backward compatibility] libur_l | riverliuintel | dvrogozh | P2 | Others | Backend/Device Issue - missing library version for XPU backend compatibility |  |
| [2390](https://github.com/intel/torch-xpu-ops/issues/2390) | SDPA in pytorch use different backend co | LuFinch | jiqing-feng | P2 | Flash Attention / Transformer Related | Backend/Device Issue - SDPA uses different backend compared with IPEX on XPU |  |
| [2400](https://github.com/intel/torch-xpu-ops/issues/2400) | [ut_upstream] tf32_on_and_off() need xpu | chunhuanMeng | daisyden | P2 | Others | Backend/Device Issue - XPU support required for tf32_on_and_off() test |  |
| [2412](https://github.com/intel/torch-xpu-ops/issues/2412) | Some NestedTensor missing XPU support | yucai-intel | daisyden | P2 | Others | Backend/Device Issue - XPU support missing for NestedTensor operations |  |
| [2465](https://github.com/intel/torch-xpu-ops/issues/2465) | [windows] ut hang | tadkrawiec | bjarzemb | P2 | Others | Timeout/Performance Issue - UT hang indicates a timeout or performance bottlenec |  |
| [2467](https://github.com/intel/torch-xpu-ops/issues/2467) | Host may stuck when submit too many kern | jianyizh | jianyizh | P2 | Inductor / Compilation Related | Backend/Device Issue - Host stuck due to excessive kernel submissions on XPU |  |
| [2469](https://github.com/intel/torch-xpu-ops/issues/2469) | test_matmul_cuda.py gaps | CuiYifeng, guangyey | daisyden | P2 | Others | Skip/No Test Exists - test_matmul_cuda.py gaps indicate missing or skipped tests |  |
| [2471](https://github.com/intel/torch-xpu-ops/issues/2471) | test_cuda.py gaps | guangyey | daisyden | P2 | Others | Skip/No Test Exists - test_cuda.py gaps indicate missing or skipped tests. |  |
| [2605](https://github.com/intel/torch-xpu-ops/issues/2605) | [int4][inductor] Add freezing pattern fo | None | liangan1 | P2 | TorchAO | Inductor/Compilation Issue - Adding freezing pattern for fusing int4 mm kernel i |  |
| [2640](https://github.com/intel/torch-xpu-ops/issues/2640) | random issue test_vjpvjp_index_reduce_pr | wpietka | daisyden | P2 | Distributed | Others - incomplete traceback and insufficient information to determine root cause |  |
| [2649](https://github.com/intel/torch-xpu-ops/issues/2649) | [distributed][pipelining] test_schedule_ | syedshahbaaz | zxd1997066 | P2 | Distributed | Distributed/Gloo Issue - test_schedule_multiproc.py is hanging during distribute |  |
| [2700](https://github.com/intel/torch-xpu-ops/issues/2700) | [distributed] Hang issues with test_dist | syedshahbaaz | madhumitha0102 | P2 | Distributed | Distributed/Gloo Issue - Hang issues with test_distributed_spawn.py suggest a pr |  |
| [2816](https://github.com/intel/torch-xpu-ops/issues/2816) | torch.logcumsumexp incorrectly returns N | Silv3S | Silv3S | P2 | Dtype / Precision Related | Dtype/Precision Issue - torch.logcumsumexp returns NaNs for complex64 input |  |
| [2853](https://github.com/intel/torch-xpu-ops/issues/2853) | [upstream_ut] torch.ops.aten._flash_atte | LuFinch | BBBela | P2 | Distributed | Flash Attention/Specific Ops Issue - _flash_attention_forward not supported on X |  |
| [2899](https://github.com/intel/torch-xpu-ops/issues/2899) | Update nan_to_num XPU stub to use std::o | None | cleonard530 | P2 | Dtype / Precision Related | Backend/Device Issue - XPU stub uses incorrect type for nan_to_num parameters |  |
| [2948](https://github.com/intel/torch-xpu-ops/issues/2948) | [AO] Benchmark enabling on XPU | None | liangan1 | P2 | Others | Backend/Device Issue - XPU benchmark enabling issue |  |
| [2950](https://github.com/intel/torch-xpu-ops/issues/2950) | SYCL compilation flag -fsycl-id-queries- | BBBela | BBBela | P2 | Inductor / Compilation Related | Inductor/Compilation Issue - SYCL compilation flag not working as expected for T |  |
| [3021](https://github.com/intel/torch-xpu-ops/issues/3021) | [distributed] all_to_all_single Compatib | zhangxiaoli73 | xiangyuT | P2 | Distributed | Distributed/Gloo Issue - all_to_all_single compatibility issue on B60/XCCL |  |
| [3022](https://github.com/intel/torch-xpu-ops/issues/3022) | [distributed] batch_isend_irecv Compatib | zhangxiaoli73 | xiangyuT | P2 | Distributed | Distributed/Gloo Issue - batch_isend_irecv compatibility issue on B60/XCCL |  |
| [3024](https://github.com/intel/torch-xpu-ops/issues/3024) | Enable clang-tidy checks | Silv3S | Silv3S | P2 | Others | Others - clang-tidy checks enablement is a code quality/linter configuration task, not a runtime or functional issue. |  |
| [3048](https://github.com/intel/torch-xpu-ops/issues/3048) | Profiler result is not correct on B70 | aostrowski-hbn | jianyizh | P2 | Others | Backend/Device Issue - Profiler result discrepancy on B70 device. |  |
| [3081](https://github.com/intel/torch-xpu-ops/issues/3081) | Sparse CSR gemm-like ops have not been s | None | daisyden | P2 | Sparse Operations Related | Supported - Sparse CSR gemm-like operations are not supported yet. |  |
| [3082](https://github.com/intel/torch-xpu-ops/issues/3082) | multithread support in distributed | None | daisyden | P2 | Distributed | Distributed/Gloo Issue - multithread support in distributed operations is affect |  |
| [3084](https://github.com/intel/torch-xpu-ops/issues/3084) | torch.library.register_autocast does not | None | daisyden | P2 | Others | Backend/Device Issue - torch.library.register_autocast does not support XPU devi |  |
| [3086](https://github.com/intel/torch-xpu-ops/issues/3086) | nvml support blocks some test cases | None | daisyden | P2 | Others | Backend/Device Issue - nvml support blocking test cases on XPU |  |
| [3093](https://github.com/intel/torch-xpu-ops/issues/3093) | XPU does not support NestedTensor for SD | None | daisyden | P2 | Flash Attention / Transformer Related | Supported - XPU does not support NestedTensor for SDPA operations. |  |
| [3096](https://github.com/intel/torch-xpu-ops/issues/3096) | VISIBLE_DEVICE support | None | daisyden | P2 | Others | Backend/Device Issue - VISIBLE_DEVICE support is related to device visibility an |  |
| [3100](https://github.com/intel/torch-xpu-ops/issues/3100) | [distributed] /handler/dump_nccl_trace_p | None | zxd1997066 | P2 | Distributed | Distributed/Gloo Issue - related to NCCL logging and trace handling in distribut |  |
| [3103](https://github.com/intel/torch-xpu-ops/issues/3103) | Tensor-likes are not equal for test_back | BBBela | BBBela | P2 | Dtype / Precision Related | Backend/Device Issue - XPU tensor-like comparison failure in test |  |
| [3180](https://github.com/intel/torch-xpu-ops/issues/3180) | [E2E] Timm/Torchbench models got "eager_ | None | libohao1201 | P0 | Others | Backend/Device Issue - eager_two_runs_differ on ARC XPU backend |  |
| [3189](https://github.com/intel/torch-xpu-ops/issues/3189) | Task Tracker | guangyey | guangyey | P2 | Others | Skip/No Test Exists - no test or error details provided |  |
| [3194](https://github.com/intel/torch-xpu-ops/issues/3194) | Incorrect strides in TestCommonXPU,test_ | AKloniecki | AKloniecki | P2 | Distributed | Backend/Device Issue - Incorrect strides related to XPU device handling |  |
| [3196](https://github.com/intel/torch-xpu-ops/issues/3196) | vitals is not supported, the cases shoul | libohao1201 | daisyden | P2 | Others | 10 - vitals feature is not supported, cases should be disabled |  |
| [3216](https://github.com/intel/torch-xpu-ops/issues/3216) | [OPs] Some ops of XPU have non-determini | CuiYifeng | YangKai0616 | P2 | Others | Backend/Device Issue - XPU ops show non-determinism and inconsistency compared t |  |
| [1171](https://github.com/intel/torch-xpu-ops/issues/1171) | LNL Windows got unexpected error message | xuhancn, chunhuanMeng | daisyden | P2 | Others | Backend/Device Issue - unexpected error on XPU for LNL Windows |  |
| [1324](https://github.com/intel/torch-xpu-ops/issues/1324) | [Win] UR Error when OOM and break the te | Stonepia | Stonepia | P2 | Others | Memory/Shared Memory Issue - Out of memory (OOM) leading to tensor context break |  |
| [1548](https://github.com/intel/torch-xpu-ops/issues/1548) | [distributed] AssertionError: 'fused_all | Chao1Han | PenghuiCheng | P2 | Distributed | Failure - 'fused_all_gather_matmul' not found in AOT ID list |  |
| [1549](https://github.com/intel/torch-xpu-ops/issues/1549) | [distributed] AssertionError: 'fused_all | Chao1Han | PenghuiCheng | P2 | Distributed | Distributed/Gloo Issue - 'fused_all_gather_scaled_matmul' not found in graph dur |  |
| [1555](https://github.com/intel/torch-xpu-ops/issues/1555) | [distributed] RuntimeError: aten.add.Ten | chuanqi129 | PenghuiCheng | P2 | Distributed | Distributed/Gloo Issue - mixed torch.Tensor and DTensor in distributed operation |  |
| [1571](https://github.com/intel/torch-xpu-ops/issues/1571) | [distributed] ValueError: Cannot use Red | zhangxiaoli73 | daisyden | P2 | Distributed | Distributed/Gloo Issue - ReduceOp.PREMUL_SUM not supported with XCCL |  |
| [1649](https://github.com/intel/torch-xpu-ops/issues/1649) | [cpp extension] Provide a clear error me | dvrogozh | ZhaoqiongZ | P2 | Others | Backend/Device Issue - inconsistent oneAPI versions cause XPU backend errors |  |
| [1661](https://github.com/intel/torch-xpu-ops/issues/1661) | [distributed] Accuracy gap in _composabl | githubsgi | PenghuiCheng | P2 | Distributed | Distributed/Gloo Issue - Accuracy gap in _composable/fsdp on Xelink suggests a d |  |
| [1727](https://github.com/intel/torch-xpu-ops/issues/1727) | [distributed] AttributeError: module 'to | guangyey | PenghuiCheng | P2 | Distributed | Backend/Device Issue - missing attribute '_sleep' in torch.xpu module |  |
| [1749](https://github.com/intel/torch-xpu-ops/issues/1749) | transformers UT failure in XPU because S | LuFinch | sywangyi | P2 | Flash Attention / Transformer Related | Backend/Device Issue - XPU does not support backward or grad for SDPA operation |  |
| [1784](https://github.com/intel/torch-xpu-ops/issues/1784) | [Performance] Torch XPU Profiler is not  | jfedorov | liangan1 | P2 | Others | 11 - Timeout/Performance Issue - Torch XPU Profiler is not reliable |  |
| [1833](https://github.com/intel/torch-xpu-ops/issues/1833) | [PT2E] INT8 accuracy is lower than FP32 | chunhuanMeng | mengfei25 | P2 | TorchAO | Dtype/Precision Issue - INT8 accuracy lower than FP32 due to precision reduction |  |
| [1877](https://github.com/intel/torch-xpu-ops/issues/1877) | Torchbench model squeezenet1_1 and funct | Silv3S | libohao1201 | P0 | Dtype / Precision Related | Backend/Device Issue - fail_accuracy on XPU for specific models |  |
| [1969](https://github.com/intel/torch-xpu-ops/issues/1969) | torch._dynamo.exc.InternalTorchDynamoErr | guangyey | shangerxin | P2 | PT2E | Error - cannot create weak reference to 'torch.Event' object |  |
| [1970](https://github.com/intel/torch-xpu-ops/issues/1970) | torch._dynamo.exc.BackendCompilerFailed: | None | shangerxin | P2 | PT2E | Backend/Device Issue - CUDA not available on the system |  |
| [2004](https://github.com/intel/torch-xpu-ops/issues/2004) | [distributed][shared_tensor] test\distri | libohao1201 | libohao1201 | P2 | Distributed | Memory/Shared Memory Issue - error originated from shared memory connection in t |  |
| [2022](https://github.com/intel/torch-xpu-ops/issues/2022) | [Windows] [CI] [UT] AssertionError: Tens | None | RUIJIEZHONG66166 | P2 | Dtype / Precision Related | Failure - Tensor-likes are not close! |  |
| [2024](https://github.com/intel/torch-xpu-ops/issues/2024) | AssertionError: Torch not compiled with  | daisyden | mengfei25 | P2 | Distributed |  |  |
| [2157](https://github.com/intel/torch-xpu-ops/issues/2157) | BMG d2h copy is very slow compare to pvc | jianyizh, mengfei25 | jianyizh | P2 | Others | Timeout/Performance Issue - BMG d2h copy is significantly slower compared to PVC |  |
| [2165](https://github.com/intel/torch-xpu-ops/issues/2165) | [distributed] test_device_mesh.py::TestD | jemitche1 | zxd1997066 | P2 | Distributed | Failure - test_flatten_mesh_3d encountered an assertion error |  |
| [2169](https://github.com/intel/torch-xpu-ops/issues/2169) | Frame size comparison failed in test_siz | guangyey | daisyden | P2 | Distributed | Failure - Scalars are not equal in test comparison |  |
| [2182](https://github.com/intel/torch-xpu-ops/issues/2182) | test_transform_bias_rescale_qkv_nested_x | PawelSwider2000 | wincent8 | P2 | Distributed | Failure - Scalars are not equal in test assertion |  |
| [2201](https://github.com/intel/torch-xpu-ops/issues/2201) | [TorchAO][BMG] When using paged attentio | Stonepia | MingxuZh | P2 | TorchAO | Failure - assert vr is not None error encountered |  |
| [2217](https://github.com/intel/torch-xpu-ops/issues/2217) | AO Performance issue track | Stonepia | liangan1 | P2 | Others | Timeout/Performance Issue - AO Performance issue track |  |
| [2219](https://github.com/intel/torch-xpu-ops/issues/2219) | float8_e4m3fn precision overflow | CuiYifeng, yucai-intel | jiqing-feng | P2 | Dtype / Precision Related | Dtype/Precision Issue - float8_e4m3fn precision overflow |  |
| [2239](https://github.com/intel/torch-xpu-ops/issues/2239) | Exception: could not create a primitive  | wpietka | zxd1997066 | P2 | Distributed | DNNL/OneDNN Issue - could not create a primitive descriptor for deconvolution fo |  |
| [2240](https://github.com/intel/torch-xpu-ops/issues/2240) | RuntimeError: Trying to set a forward gr | gplutop7 | zxd1997066 | P2 | Distributed | Error - forward gradient size mismatch with original Tensor |  |
| [2245](https://github.com/intel/torch-xpu-ops/issues/2245) | oneDNN matmul received incorrect shape i | CuiYifeng | wincent8 | P2 | Distributed | DNNL/OneDNN Issue - oneDNN matmul primitive descriptor creation failure |  |
| [2263](https://github.com/intel/torch-xpu-ops/issues/2263) | [xpu][bug] XPU Trace event ends too late | PawelSwider2000 | chuanqi129 | P2 | Others | Backend/Device Issue - XPU trace event timing discrepancy |  |
| [2340](https://github.com/intel/torch-xpu-ops/issues/2340) | [distributed][_tools] AssertionError: Ro | githubsgi | zxd1997066 | P2 | Distributed | Failure - Roofline estimation requires CUDA capabilities assertion failed |  |
| [2389](https://github.com/intel/torch-xpu-ops/issues/2389) | [Bug Skip]: RuntimeError: Data corruptio | PatrykWilczewski | kaileiyx | P2 | Distributed | Error - Data corruption detected during test execution |  |
| [2392](https://github.com/intel/torch-xpu-ops/issues/2392) | [Bug Skip]: torch.OutOfMemoryError: XPU  | xuhancn | RUIJIEZHONG66166 | P2 | Others | Memory/Shared Memory Issue - XPU out of memory error occurred |  |
| [2404](https://github.com/intel/torch-xpu-ops/issues/2404) | [distributed][checkpoint] AssertionError | None | zxd1997066 | P2 | Distributed | Failure - Booleans mismatch assertion error |  |
| [2425](https://github.com/intel/torch-xpu-ops/issues/2425) | [upstream_ut]  RuntimeError: Expected bo | BBBela | daisyden | P2 | Dtype / Precision Related | Error - Nested tensor operation with non-nested tensor input |  |
| [2434](https://github.com/intel/torch-xpu-ops/issues/2434) | [Bug Skip]: New failures 2025-11-28 | AKloniecki | mengfei25 | P2 | TorchAO | Skip/No Test Exists - test is marked as a bug skip or not implemented properly |  |
| [2439](https://github.com/intel/torch-xpu-ops/issues/2439) | [oneDNN] TestDecompXPU.test_quick_addmv_ | libohao1201 | mengfei25 | P2 | Dtype / Precision Related | DNNL/OneDNN Issue - Test failure related to oneDNN decomposition on XPU with flo |  |
| [2440](https://github.com/intel/torch-xpu-ops/issues/2440) | [For UT failures classify] Save referenc | None | RUIJIEZHONG66166 | P2 | Others | Skip/No Test Exists - no test or error details provided |  |
| [2444](https://github.com/intel/torch-xpu-ops/issues/2444) | [upstream_ut]  RuntimeError: UR backend  | Silv3S | wincent8 | P2 | Dtype / Precision Related | Memory/Shared Memory Issue - UR backend returns out of resources error (40) indi |  |
| [2446](https://github.com/intel/torch-xpu-ops/issues/2446) | [Bug Skip]: AssertionError: "Simulate er | None | kaileiyx | P2 | Distributed | Failure - mismatch between expected and actual error messages |  |
| [2447](https://github.com/intel/torch-xpu-ops/issues/2447) | test_share_memory_xpu failure | None | daisyden | P2 | Others | Memory/Shared Memory Issue - failure in test_share_memory_xpu related to shared  |  |
| [2463](https://github.com/intel/torch-xpu-ops/issues/2463) | [Bug Skip]: OSError: SYCL runtime is not | xuhancn | RUIJIEZHONG66166 | P2 | Others | Backend/Device Issue - SYCL runtime not detected on XPU |  |
| [2479](https://github.com/intel/torch-xpu-ops/issues/2479) | [Bug] torch.rand output different result | Stonepia, CuiYifeng | zufangzhu | P2 | Others | Backend/Device Issue - different output on BMG and PVC devices |  |
| [2491](https://github.com/intel/torch-xpu-ops/issues/2491) | [upstream_ut]  AssertionError: False is  | PatrykWilczewski | libohao1201 | P2 | Dtype / Precision Related | Failure - test assertion failed with False is not true |  |
| [2494](https://github.com/intel/torch-xpu-ops/issues/2494) | [upstream_ut]  AssertionError: Scalars a | PawelSwider2000 | libohao1201 | P2 | Dtype / Precision Related | Failure - Scalars are not equal assertion error in test |  |
| [2510](https://github.com/intel/torch-xpu-ops/issues/2510) | [upstream_ut]  RuntimeError: Expected ou | PawelSwider2000 | libohao1201 | P2 | Dtype / Precision Related | Error - tensor size exceeds int32_t maximum limit |  |
| [2513](https://github.com/intel/torch-xpu-ops/issues/2513) | [upstream_ut]  RuntimeError: _share_fd_: | gplutop7 | libohao1201 | P2 | Others | Memory/Shared Memory Issue - _share_fd_ is not available on XPU, indicating a sh |  |
| [2531](https://github.com/intel/torch-xpu-ops/issues/2531) | [upstream_ut]  AssertionError: Torch not | daisyden | daisyden | P2 | Distributed |  |  |
| [2532](https://github.com/intel/torch-xpu-ops/issues/2532) | Title: [upstream_ut]  AssertionError: wr | yucai-intel | daisyden | P2 | Distributed | Failure - wrong number of dimensions for int4 conversion op |  |
| [2533](https://github.com/intel/torch-xpu-ops/issues/2533) | Title: [upstream_ut]  AttributeError: 'T | astachowiczhabana | daisyden | P2 | Distributed | Mismatch - Test method 'test_qsoftmax' is missing in 'TestQuantizedOpsXPU' class. |  |
| [2535](https://github.com/intel/torch-xpu-ops/issues/2535) | Title: [upstream_ut]  AttributeError: mo | Silv3S | daisyden | P2 | Distributed | Backend/Device Issue - missing XPU-specific attribute in torch._C for tunableop  |  |
| [2536](https://github.com/intel/torch-xpu-ops/issues/2536) | Title: [upstream_ut]  AttributeError: mo | daisyden | daisyden | P2 | Distributed | Backend/Device Issue - missing attribute '_scatter' in torch._C for XPU device |  |
| [2537](https://github.com/intel/torch-xpu-ops/issues/2537) | Title: [upstream_ut]  Failed: Unexpected | PatrykWilczewski | daisyden | P2 | Others | Others - Test expects failure but passed unexpectedly, no specific error trace provided. |  |
| [2538](https://github.com/intel/torch-xpu-ops/issues/2538) | Title: [upstream_ut]  RuntimeError: Floa | Silv3S | daisyden | P2 | Distributed | DNNL/OneDNN Issue - Float8_e4m3fnuz is not supported in oneDNN. |  |
| [2539](https://github.com/intel/torch-xpu-ops/issues/2539) | Title: [upstream_ut]  RuntimeError: Trie | None | daisyden | P2 | Distributed | Backend/Device Issue - Attempt to instantiate dummy base class CUDAGraph on XPU |  |
| [2541](https://github.com/intel/torch-xpu-ops/issues/2541) | Title: [upstream_ut]  RuntimeError: coul | yucai-intel | daisyden | P2 | Distributed | DNNL/OneDNN Issue - could not construct a memory descriptor using strides |  |
| [2560](https://github.com/intel/torch-xpu-ops/issues/2560) | [UT] "RuntimeError: iter.device(arg).is_ | CuiYifeng | libohao1201 | P2 | Others | Backend/Device Issue - XPU device check failure in test |  |
| [2562](https://github.com/intel/torch-xpu-ops/issues/2562) | Warning as Error | chunhuanMeng | EikanWang | P2 | Others | Others - warning treated as error but no traceback or specific error provided |  |
| [2595](https://github.com/intel/torch-xpu-ops/issues/2595) | [Bug Skip]: Random crashed cases 2025-12 | None | CuiYifeng | P0 | Sparse Operations Related | Skip/No Test Exists - Test was skipped due to random crashed cases. |  |
| [2596](https://github.com/intel/torch-xpu-ops/issues/2596) | [TorchAO][BMG]Observed accuracy fluctuat | None | LifengWang | P0 | TorchAO | Dtype/Precision Issue - accuracy fluctuations due to INT4 quantization methods ( |  |
| [2597](https://github.com/intel/torch-xpu-ops/issues/2597) | [TorchAO][BMG] INT4 GPTQ shows worse per | xiaowangintel | LifengWang | P2 | TorchAO | Timeout/Performance Issue - INT4 GPTQ shows worse performance compared with RTN  |  |
| [2615](https://github.com/intel/torch-xpu-ops/issues/2615) | [Bug Skip]: New failures RuntimeError: U | CuiYifeng | kaileiyx | P2 | Distributed | Dtype/Precision Issue - Unsupported dtype Half / torch.float16 |  |
| [2630](https://github.com/intel/torch-xpu-ops/issues/2630) | Title: [upstream_ut]  AssertionError: Sc | jmamzax | daisyden | P2 | Distributed | Failure - Scalars are not equal! |  |
| [2639](https://github.com/intel/torch-xpu-ops/issues/2639) | test_to() failed during rnn isinstance() | Silv3S | daisyden | P2 | Distributed | Failure - test_to() failed during rnn isinstance() check | [PR](https://github.com/intel/torch-xpu-ops/pull/3193) |
| [2645](https://github.com/intel/torch-xpu-ops/issues/2645) | [Bug Skip]: [regression] RuntimeError: f | CuiYifeng | kaileiyx | P0 | Inductor / Compilation Related | Backend/Device Issue - missing kernel for xpu in DispatchStub |  |
| [2669](https://github.com/intel/torch-xpu-ops/issues/2669) | [upstream_ut]  AssertionError: Tensor-li | tszulist-hbn | daisyden | P2 | Distributed | Failure - Tensor-likes are not close! in test_vmap_exhaustive_addmv_xpu_float32 |  |
| [2676](https://github.com/intel/torch-xpu-ops/issues/2676) | Random failure in CI test | None | daisyden | P2 | Dtype / Precision Related | Others - Random failure with no traceback or specific error provided |  |
| [2680](https://github.com/intel/torch-xpu-ops/issues/2680) | XPU Autocast does not support  fp32 dtyp | CuiYifeng | kaixuanliu | P2 | Dtype / Precision Related | Dtype/Precision Issue - XPU Autocast does not support fp32 dtypes |  |
| [2686](https://github.com/intel/torch-xpu-ops/issues/2686) | [distributed] Accuracy issues with test_ | frost-intel | madhumitha0102 | P2 | Distributed | Distributed/Gloo Issue - Accuracy issues in distributed testing with test_distri |  |
| [2689](https://github.com/intel/torch-xpu-ops/issues/2689) | [LNL][Windows] AssertionError: 'Assertio | tadkrawiec | kaileiyx | P2 | Dtype / Precision Related | Failure - cur_target out of bounds assertion failed |  |
| [2701](https://github.com/intel/torch-xpu-ops/issues/2701) | [distributed] Barrier Timeout Error with | syedshahbaaz | madhumitha0102 | P2 | Distributed | Distributed/Gloo Issue - Barrier Timeout Error in distributed testing |  |
| [2702](https://github.com/intel/torch-xpu-ops/issues/2702) | [distributed] RuntimeError: Work ran tim | syedshahbaaz | madhumitha0102 | P2 | Distributed | Timeout/Performance Issue - Work ran time out after 0 milliseconds in distribute |  |
| [2707](https://github.com/intel/torch-xpu-ops/issues/2707) | [TorchAO][BMG] INT4 GPTQ failed due to T | xiaowangintel | LifengWang | P2 | TorchAO | Mismatch - INT4 GPTQ failed due to TorchAO API change. |  |
| [2720](https://github.com/intel/torch-xpu-ops/issues/2720) | [upstream_ut] RuntimeError: false INTERN | CuiYifeng | wincent8 | P2 | Distributed | Backend/Device Issue - missing kernel for xpu in DispatchStub |  |
| [2729](https://github.com/intel/torch-xpu-ops/issues/2729) | [Bug Skip]: Random failures 2026WW03 | Silv3S | CuiYifeng | P2 | Sparse Operations Related | Skip/No Test Exists - test is marked as skipped due to random failures |  |
| [2734](https://github.com/intel/torch-xpu-ops/issues/2734) | [TorchAO][BMG] INT4 RTN Flex-attention g | Stonepia, hoshibara | LifengWang | P2 | TorchAO | Timeout/Performance Issue - 50% performance drop in INT4 RTN Flex-attention with | [PR](https://github.com/pytorch/pytorch/pull/160480, https://github.com/pytorch/pytorch/pull/172316) |
| [2737](https://github.com/intel/torch-xpu-ops/issues/2737) | [distributed] AttributeError: module 'to | None | zxd1997066 | P2 | Distributed | Distributed/Gloo Issue - missing attribute '_gather' in distributed context |  |
| [2738](https://github.com/intel/torch-xpu-ops/issues/2738) | [distributed] test_c10d_spawn_nccl.py Va | jenniew | zxd1997066 | P2 | Distributed | Distributed/Gloo Issue - input tensor size mismatch in distributed context |  |
| [2744](https://github.com/intel/torch-xpu-ops/issues/2744) | [Bug Skip]: extended test failures when  | pbielak | daisyden | P2 | Others | Skip/No Test Exists - test was skipped due to changes in tolerance values causin |  |
| [2751](https://github.com/intel/torch-xpu-ops/issues/2751) | [Bug Skip]: Random failures 2026WW04 | None | CuiYifeng | P2 | Sparse Operations Related | Skip/No Test Exists - test was skipped due to random failure标记 |  |
| [2759](https://github.com/intel/torch-xpu-ops/issues/2759) | [Bug Skip]: New failed cases 2026-1-22 | AKloniecki | kaileiyx | P2 | Distributed | Failure - RuntimeError not raised as expected in test case |  |
| [2766](https://github.com/intel/torch-xpu-ops/issues/2766) | MaxPool2d - investigate memory layout pe | BBBela | pbielak | P2 | Others | Memory/Shared Memory Issue - investigate memory layout performance for MaxPool2d |  |
| [2767](https://github.com/intel/torch-xpu-ops/issues/2767) | [UT] test_control_flow_xpu.py got Assert | PatrykWilczewski | libohao1201 | P1 | PT2E | Failure - test_control_flow_xpu.py got AssertionError |  |
| [2769](https://github.com/intel/torch-xpu-ops/issues/2769) | [oneDNN] New failed test cases with 3.11 | LuFinch | mengfei25 | P2 | Others | DNNL/OneDNN Issue - failed test cases related to oneDNN with 3.11 compared to 3. |  |
| [2777](https://github.com/intel/torch-xpu-ops/issues/2777) | [Bug Skip]: Random failures 2026WW05 | AKloniecki | CuiYifeng | P2 | Sparse Operations Related | Skip/No Test Exists - test is marked as a skip with no detailed error traceback  |  |
| [2779](https://github.com/intel/torch-xpu-ops/issues/2779) | Accuracy failures in logspace op | PawelSwider2000 | PawelSwider2000 | P2 | Distributed | Dtype/Precision Issue - accuracy failures in logspace operation |  |
| [2783](https://github.com/intel/torch-xpu-ops/issues/2783) | [Bug Skip]: Key "xpu" is missing from di | daisyden | CuiYifeng | P2 | Dtype / Precision Related | Skip/No Test Exists - Key "xpu" is missing, indicating the test is skipped or no |  |
| [2795](https://github.com/intel/torch-xpu-ops/issues/2795) | Histc raises error with integer input wh | CuiYifeng | YangKai0616 | P2 | Others | Dtype/Precision Issue - integer input causes error with deterministic algorithm  |  |
| [2797](https://github.com/intel/torch-xpu-ops/issues/2797) | Copy error is not raise on test_dlpack.p | None | shangerxin | P2 | Others | Others - Copy error not raised in test_dlpack.py test case |  |
| [2801](https://github.com/intel/torch-xpu-ops/issues/2801) | to_dense() for Sparse CSR backend cannot | jenniew | jenniew | P2 | Sparse Operations Related | Error - source tensor shape mismatch during to_dense() for Sparse CSR backend |  |
| [2811](https://github.com/intel/torch-xpu-ops/issues/2811) | [Bug Skip]: [Regression] failed cases 20 | jmamzax | kaileiyx | P0 | Distributed | Failure - Expected and actual trace outputs do not match. |  |
| [2815](https://github.com/intel/torch-xpu-ops/issues/2815) | RuntimeError: output with shape [2] does | PawelSwider2000 | Silv3S | P2 | Others | Error - output shape mismatch during broadcasting |  |
| [2823](https://github.com/intel/torch-xpu-ops/issues/2823) | [TorchAO][BMG] Llama-3.2-1B-Instruct Dyn | xiaowangintel, lchen2331 | LifengWang | P2 | TorchAO | Timeout/Performance Issue - 20% performance drop in next token generation with D |  |
| [2845](https://github.com/intel/torch-xpu-ops/issues/2845) | [Bug Skip]:[UT] [Windows] failed cases 2 | None | kaileiyx | P2 | Others | Skip/No Test Exists - test was skipped or does not exist |  |
| [2852](https://github.com/intel/torch-xpu-ops/issues/2852) | [Bug Skip]: New UT failures in 0206 nigh | None | chuanqi129 | P2 | Others | Skip/No Test Exists - test was skipped or not present |  |
| [2858](https://github.com/intel/torch-xpu-ops/issues/2858) | [Bug Skip]: test_xpu new failures | None | RUIJIEZHONG66166 | P2 | Others | Skip/No Test Exists - test_xpu new failures marked as skipped or no test availab |  |
| [2862](https://github.com/intel/torch-xpu-ops/issues/2862) | accuracy issue with test_float8_scale_fa | tszulist-hbn | daisyden | P2 | Dtype / Precision Related | Dtype/Precision Issue - accuracy issue with float8 operations |  |
| [2871](https://github.com/intel/torch-xpu-ops/issues/2871) | [distributed][fsdp] test_fsdp_overlap.py | songhappy | zxd1997066 | P2 | Distributed | Failure - test_fsdp_overlap.py assertion failed with "False is not true" |  |
| [2873](https://github.com/intel/torch-xpu-ops/issues/2873) | [Bug Skip]: test_repos.py contains sever | PawelSwider2000 | shangerxin | P2 | PT2E | Skip/No Test Exists - test contains failed ops and is skipped |  |
| [2879](https://github.com/intel/torch-xpu-ops/issues/2879) | RuntimeError: _share_fd_: only available | Silv3S | Silv3S | P2 | Others | Backend/Device Issue - _share_fd_ is not available on XPU device |  |
| [2914](https://github.com/intel/torch-xpu-ops/issues/2914) | Test case test/test_autograd.py::TestAut | None | shangerxin | P2 | Dtype / Precision Related | Failure - Tensor-likes are not close! |  |
| [2920](https://github.com/intel/torch-xpu-ops/issues/2920) | Failing Test Cases in Nightly Wheel [202 | Silv3S | BBBela | P2 | Others | Others - insufficient information to determine root cause |  |
| [2921](https://github.com/intel/torch-xpu-ops/issues/2921) | [abs][complex64] - new failing test case | AKloniecki | BBBela | P2 | Distributed | Dtype/Precision Issue - test failure related to complex64 data type and abs oper |  |
| [2942](https://github.com/intel/torch-xpu-ops/issues/2942) | [Windows] Unit tests got Fatal python er | xuhancn, Stonepia | mengfei25 | P2 | Others | Backend/Device Issue - Fatal Python error on Windows unit tests related to XPU b |  |
| [2946](https://github.com/intel/torch-xpu-ops/issues/2946) | [Bug Skip]: Random failures 2026WW09 | None | CuiYifeng | P2 | TorchAO | Skip/No Test Exists - test is marked to be skipped with no valid test implementa |  |
| [2965](https://github.com/intel/torch-xpu-ops/issues/2965) | [Bug Skip]: Random failures 2026WW10 | None | CuiYifeng | P2 | Sparse Operations Related | Skip/No Test Exists - test is marked as a skip with no valid test implementation |  |
| [2968](https://github.com/intel/torch-xpu-ops/issues/2968) | [distributed] timeout issue in test/dist | frost-intel | zxd1997066 | P2 | Distributed | Timeout/Performance Issue - test experienced a timeout in distributed execution  |  |
| [2969](https://github.com/intel/torch-xpu-ops/issues/2969) | [distributed] AssertionError: Scalars ar | frost-intel | zxd1997066 | P2 | Distributed | Failure - Scalars are not equal in test case |  |
| [2971](https://github.com/intel/torch-xpu-ops/issues/2971) | [distributed] KeyError in test/distribut | newtdms, frost-intel | zxd1997066 | P2 | Distributed | Distributed/Gloo Issue - KeyError in distributed test related to c10d_xccl |  |
| [2972](https://github.com/intel/torch-xpu-ops/issues/2972) | [distributed] AssertionError: ValueError | newtdms | zxd1997066 | P2 | Distributed | Distributed/Gloo Issue - AssertionError in test/distributed/test_c10d_xccl.py re |  |
| [2993](https://github.com/intel/torch-xpu-ops/issues/2993) | [Bug Skip]: Unexpected success of test_c | gplutop7 | CuiYifeng | P2 | Others | Skip/No Test Exists - test unexpectedly succeeded and should have been skipped |  |
| [3000](https://github.com/intel/torch-xpu-ops/issues/3000) | [Bug Skip]: RuntimeError: _share_fd_: on | gplutop7 | zxd1997066 | P2 | Others | Backend/Device Issue - _share_fd_ is not available on XPU device |  |
| [3010](https://github.com/intel/torch-xpu-ops/issues/3010) | [distributed][tensor] test_random_ops.py | jenniew | zxd1997066 | P2 | Distributed | Inductor/Compilation Issue - TorchRuntimeError during fake tensor call in Dynamo |  |
| [3025](https://github.com/intel/torch-xpu-ops/issues/3025) | New failing test in Nightly Wheel test_d | None | BBBela | P2 | Distributed | Failure - Expected and actual decomposition outputs do not match. |  |
| [3030](https://github.com/intel/torch-xpu-ops/issues/3030) | [Bug Skip] test/test_modules.py::TestMod | gplutop7 | shangerxin | P2 | Others | Skip/No Test Exists - test was skipped due to failure with no detailed error pro |  |
| [3033](https://github.com/intel/torch-xpu-ops/issues/3033) | [Bug Skip]: Softmax tolerance | chunhuanMeng | chunhuanMeng | P2 | Others | Skip/No Test Exists - test is skipped due to Softmax tolerance issue |  |
| [3074](https://github.com/intel/torch-xpu-ops/issues/3074) | [Bug Skip] test_dlpack_exchange_api expe | AKloniecki | shangerxin | P2 | Others | Skip/No Test Exists - test is skipped expecting current_work_stream is not null |  |
| [3076](https://github.com/intel/torch-xpu-ops/issues/3076) | [TorchAO][BMG] Llama-3.2-1B-Instruct Dyn | None | LifengWang | P2 | TorchAO | DNNL/OneDNN Issue - Performance drop with oneDNN 3.11.1 in Dynamic INT8 for Llam |  |
| [3083](https://github.com/intel/torch-xpu-ops/issues/3083) | [Bug Skip]: Random failures 2026WW12 | None | CuiYifeng | P2 | Others | Skip/No Test Exists - test is marked to be skipped with no valid test to execute |  |
| [3088](https://github.com/intel/torch-xpu-ops/issues/3088) | [TorchAO][BMG] INT4 RTN Flex-attention g | Stonepia | LifengWang | P2 | TorchAO | Timeout/Performance Issue - INT4 RTN Flex-attention experienced a 5% performance |  |
| [3101](https://github.com/intel/torch-xpu-ops/issues/3101) | [distributed] 'torch._C._distributed_c10 | None | zxd1997066 | P2 | Distributed | Distributed/Gloo Issue - ProcessGroupXCCL object lacks '_set_default_timeout' at |  |
| [3102](https://github.com/intel/torch-xpu-ops/issues/3102) | [distributed] RuntimeError: Invalid devi | None | zxd1997066 | P2 | Distributed | Backend/Device Issue - invalid device string 'xpu:foo' indicates a problem with  |  |
| [3121](https://github.com/intel/torch-xpu-ops/issues/3121) | [Bug Skip]: CUDA specific UT test_fft_ha | None | CuiYifeng | P1 | Dtype / Precision Related | Skip/No Test Exists - test is skipped or not applicable for XPU backend |  |
| [3124](https://github.com/intel/torch-xpu-ops/issues/3124) | [TorchAO][Bug] ImportError: Requires msl | None | FRAMEEE17 | P2 | TorchAO | Backend/Device Issue - Requires mslk >= 1.0.0 for XPU support with int4 quantiza |  |
| [3139](https://github.com/intel/torch-xpu-ops/issues/3139) | [distributed][_composable] AssertionErro | Kanya-Mo | zxd1997066 | P2 | Distributed | Failure - Expects xpu:0 but got xpu:1 |  |
| [3149](https://github.com/intel/torch-xpu-ops/issues/3149) | New failure in test_rms_norm_decomp_acce | None | BBBela | P2 | Flash Attention / Transformer Related | Failure - RuntimeError not raised as expected in test |  |
| [3174](https://github.com/intel/torch-xpu-ops/issues/3174) | [Bug Skip]: Accuracy failure of test_Con | None | CuiYifeng | P2 | Distributed | Failure - test assertion failed for Conv2d groups output comparison |  |
| [3178](https://github.com/intel/torch-xpu-ops/issues/3178) | New failed test cases 2026-03-25 | pponikox | BBBela | P2 | Flash Attention / Transformer Related | Backend/Device Issue - XPU-specific test failure in vmap with SDPA operation |  |
| [3195](https://github.com/intel/torch-xpu-ops/issues/3195) | test_sdpa_unbacked_no_dde_xpu crashed | None | daisyden | P0 | Flash Attention / Transformer Related | Backend/Device Issue - test crashed on XPU backend execution |  |
| [3206](https://github.com/intel/torch-xpu-ops/issues/3206) | [Bug Skip]: [new failures]RuntimeError:  | tszulist-hbn | kaileiyx | P2 | Others | Backend/Device Issue - Tensors are on different devices (xpu:0 vs cpu) |  |
| [3227](https://github.com/intel/torch-xpu-ops/issues/3227) | torch xpu event has ~0.1ms latency, whic | guangyey | jianyizh | P2 | Others | Timeout/Performance Issue - high latency in XPU event (~0.1ms) indicates a perfo |  |
| [3231](https://github.com/intel/torch-xpu-ops/issues/3231) | Dynamo failed to run FX node with fake t | None | daisyden | P2 | PT2E | Inductor/Compilation Issue - Dynamo failed to compile FX node with fake tensors  |  |
| [3232](https://github.com/intel/torch-xpu-ops/issues/3232) | [distributed][tensor] AssertionError: As | None | zxd1997066 | P2 | Distributed | Failure - AssertionError not raised for Placement (Shard(dim=2),) in test |  |
| [3233](https://github.com/intel/torch-xpu-ops/issues/3233) | [distributed] RuntimeError: No backend f | None | zxd1997066 | P2 | Distributed | Distributed/Gloo Issue - No backend for the parent process group or its backend  |  |
| [3236](https://github.com/intel/torch-xpu-ops/issues/3236) | AssertionError: 'def [28 chars]n unbind  | jmamzax | jmamzax | P2 | Dtype / Precision Related | Failure - mismatch in expected IR code for XPU backend operations |  |
| [3242](https://github.com/intel/torch-xpu-ops/issues/3242) | AssertionError: Torch not compiled with  | None | zxd1997066 | P2 | Inductor / Compilation Related |  |  |
| [3243](https://github.com/intel/torch-xpu-ops/issues/3243) | AssertionError: False is not true | pponikox | zxd1997066 | P2 | Dtype / Precision Related | Failure - assertion 'False is not true' failed in test |  |
| [3258](https://github.com/intel/torch-xpu-ops/issues/3258) | Error in op: torch.ops.aten._scaled_dot_ | None | bjarzemb | P2 | Flash Attention / Transformer Related | 5 - Flash Attention/Specific Ops Issue - Error in _scaled_dot_product_fused_atte |  |
| [3259](https://github.com/intel/torch-xpu-ops/issues/3259) | New failed test cases 2026-04-02 | SlawomirLaba | Silv3S | P2 | Flash Attention / Transformer Related | Backend/Device Issue - XPU device initialization or compatibility failure |  |

#### Close Fixed Issue {#close-fixed-issue}

| ID | Title | Owner | Owner Transferred | Priority | Category | Root Cause | PR |
|---|-------|-------|-------------------|---------|----------|-----------|-----|
| [1624](https://github.com/intel/torch-xpu-ops/issues/1624) | [DONT CLOSE] Known UT Issue list | None | RUIJIEZHONG66166 | P2 | Dtype / Precision Related |  |  |
| [2496](https://github.com/intel/torch-xpu-ops/issues/2496) | [upstream_ut]  Segmentation fault when r | astachowiczhabana | libohao1201 | P0 | Dtype / Precision Related |  |  |
| [2518](https://github.com/intel/torch-xpu-ops/issues/2518) | [upstream_ut]  TypeError: Creating a Ten | astachowiczhabana | libohao1201 | P2 | Others |  |  |
| [3011](https://github.com/intel/torch-xpu-ops/issues/3011) | [upstream_ut] torch.OutOfMemoryError: XP | None | Silv3S | P2 | Others |  |  |
| [3013](https://github.com/intel/torch-xpu-ops/issues/3013) | [upstream_ut] RuntimeError: Kernel is in | None | Silv3S | P2 | Inductor / Compilation Related |  |  |
| [3014](https://github.com/intel/torch-xpu-ops/issues/3014) | [upstream_ut] AssertionError: False is n | None | Silv3S | P2 | Dtype / Precision Related |  |  |
| [3106](https://github.com/intel/torch-xpu-ops/issues/3106) | Worker crashes when running TestDecompXP | BBBela | BBBela | P0 | Others |  |  |
| [3157](https://github.com/intel/torch-xpu-ops/issues/3157) | XPU out of memory. Tried to allocate 32. | kdrozd-dev | kdrozd-dev | P2 | Others |  |  |
| [3158](https://github.com/intel/torch-xpu-ops/issues/3158) | AttributeError: module 'triton.compiler' | tadkrawiec | kdrozd-dev | P2 | Inductor / Compilation Related |  |  |
| [3161](https://github.com/intel/torch-xpu-ops/issues/3161) | Exception: Tensor-likes are not close! - | tadkrawiec | kdrozd-dev | P2 | Dtype / Precision Related |  |  |
| [3166](https://github.com/intel/torch-xpu-ops/issues/3166) | test_consistency_SparseCSR failures | yucai-intel | CuiYifeng | P2 | TorchAO |  |  |

#### Enable Test {#enable-test}

| ID | Title | Owner | Owner Transferred | Priority | Category | Root Cause | PR |
|---|-------|-------|-------------------|---------|----------|-----------|-----|

#### Add to Skiplist {#add-to-skiplist}

| ID | Title | Owner | Owner Transferred | Priority | Category | Root Cause | PR |
|---|-------|-------|-------------------|---------|----------|-----------|-----|
| [2164](https://github.com/intel/torch-xpu-ops/issues/2164) | skip test_no_cuda_monkeypatch as it is c | daisyden | daisyden | P2 | Distributed |  |  |
| [2309](https://github.com/intel/torch-xpu-ops/issues/2309) | unsupported ops with PYTORCH_ENABLE_XPU_ | daisyden | daisyden | P2 | TorchAO |  |  |
| [2472](https://github.com/intel/torch-xpu-ops/issues/2472) | [upstream_ut]  NotImplementedError: The  | Silv3S | daisyden | P2 | PT2E |  |  |
| [2508](https://github.com/intel/torch-xpu-ops/issues/2508) | TypedStorage / TypedTensors deprecation | Silv3S | libohao1201 | P1 | TorchAO |  | [PR](https://github.com/intel/torch-xpu-ops/pull/3260) |

#### Verify the Issue {#verify-the-issue}

| ID | Title | Owner | Owner Transferred | Priority | Category | Root Cause | PR |
|---|-------|-------|-------------------|---------|----------|-----------|-----|
| [2331](https://github.com/intel/torch-xpu-ops/issues/2331) | [upstream_ut] AssertionError: Scalars ar | hoshibara | daisyden | P2 | Dtype / Precision Related |  | [PR](https://github.com/pytorch/pytorch/pull/172314) |
| [2694](https://github.com/intel/torch-xpu-ops/issues/2694) | Title: [upstream_ut]  AssertionError: Te | daisyden | daisyden | P2 | Distributed |  | [PR](https://github.com/pytorch/pytorch/pull/171773) |

#### Need Reproduce Steps {#need-reproduce-steps}

| ID | Title | Owner | Owner Transferred | Priority | Category | Root Cause | PR |
|---|-------|-------|-------------------|---------|----------|-----------|-----|

### Engineer Actions {#engineer-actions}

#### Needs PyTorch Repo Changes (upstream) {#needs-pytorch-repo-changes-upstream}

| ID | Title | Owner | Owner Transferred | Priority | Category | Root Cause | PR |
|---|-------|-------|-------------------|---------|----------|-----------|-----|
| [1519](https://github.com/intel/torch-xpu-ops/issues/1519) | [PVC][PT2.7][ABI=1][Torch-xpu-ops UT][ww | None | None | P2 | Others | Backend/Device Issue - coredump related to XPU operations in Torch-xpu-ops UT |  |
| [1818](https://github.com/intel/torch-xpu-ops/issues/1818) | [BMG-Windows][PT2.8]Torch-xpu-ops UT got | kdrozd-dev | kdrozd-dev | P2 | Dtype / Precision Related | Backend/Device Issue - Accuracy issue on XPU for Torch-xpu-ops UT |  |
| [1963](https://github.com/intel/torch-xpu-ops/issues/1963) | [upstream_ut] MetadataMismatchError in T | pbielak | pbielak | P2 | Others |  | [PR](https://github.com/pytorch/pytorch/pull/178277, https://github.com/pytorch/pytorch/pull/175965) |
| [2128](https://github.com/intel/torch-xpu-ops/issues/2128) | [2.9][BMG-Windows][Torchbench] speeach_t | chuanqi129 | chuanqi129 | P0 | Dtype / Precision Related | Backend/Device Issue - Exception Code 0xC0000005 indicates an access violation l |  |
| [2132](https://github.com/intel/torch-xpu-ops/issues/2132) | [2.9][BMG-Windows][Torch-xpu-ops UT] 1 c | pbielak | pbielak | P2 | Inductor / Compilation Related |  |  |
| [2202](https://github.com/intel/torch-xpu-ops/issues/2202) | [TorchAO][BMG] The RTN performance of Ll | Stonepia | Stonepia | P0 | TorchAO | Timeout/Performance Issue - RTN performance regression in next-token latency for |  |
| [2214](https://github.com/intel/torch-xpu-ops/issues/2214) | test/test_sparse.py::TestSparseAnyXPU::t | jenniew | jenniew | P2 | Distributed |  |  |
| [2234](https://github.com/intel/torch-xpu-ops/issues/2234) | [upstream_ut] AssertionError: RuntimeErr | Silv3S | Silv3S | P2 | Distributed |  |  |
| [2248](https://github.com/intel/torch-xpu-ops/issues/2248) | [upstream_ut] test_cow failures | gplutop7 | gplutop7 | P2 | Distributed |  |  |
| [2251](https://github.com/intel/torch-xpu-ops/issues/2251) | [upstream_ut] test_fake_autocase got Exc | astachowiczhabana | astachowiczhabana | P2 | Dtype / Precision Related |  |  |
| [2255](https://github.com/intel/torch-xpu-ops/issues/2255) | [upstream_ut] RuntimeError: Long is not  | daisyden | daisyden | P2 | Flash Attention / Transformer Related |  |  |
| [2283](https://github.com/intel/torch-xpu-ops/issues/2283) | [upstream_ut] sparse._sampled_addmm is n | jenniew | jenniew | P1 | Distributed |  |  |
| [2287](https://github.com/intel/torch-xpu-ops/issues/2287) | [upstream_ut] test_python_ref issues | yucai-intel | yucai-intel | P2 | Others |  |  |
| [2295](https://github.com/intel/torch-xpu-ops/issues/2295) | [upstream_ut][xpu][test]nn/test_embeddin | yucai-intel | yucai-intel | P2 | Dtype / Precision Related |  |  |
| [2301](https://github.com/intel/torch-xpu-ops/issues/2301) | [upstream_ut] dtypes not align with OpIn | daisyden | daisyden | P2 | Distributed |  |  |
| [2329](https://github.com/intel/torch-xpu-ops/issues/2329) | [upstream_ut] feature missing: get_devic | etaf | etaf | P2 | Others |  |  |
| [2359](https://github.com/intel/torch-xpu-ops/issues/2359) | [upstream_ut] GradcheckError: Backward i | BBBela | BBBela | P2 | Distributed |  |  |
| [2512](https://github.com/intel/torch-xpu-ops/issues/2512) | [upstream_ut]  RuntimeError: _histc_xpu  | chunhuanMeng | libohao1201 | P2 | Dtype / Precision Related | Backend/Device Issue - _histc_xpu lacks deterministic implementation on XPU back |  |
| [2519](https://github.com/intel/torch-xpu-ops/issues/2519) | [upstream_ut]  TypeError: map2_ is only  | Silv3S | libohao1201 | P2 | Others | Backend/Device Issue - map2_ is only implemented on CPU tensors, indicating lack |  |
| [2554](https://github.com/intel/torch-xpu-ops/issues/2554) | [upstream_ut]  AssertionError: Assertion | daisyden | daisyden | P2 | PT2E |  |  |
| [2570](https://github.com/intel/torch-xpu-ops/issues/2570) | crash in sdpa. | LuFinch | sywangyi | P0 | Flash Attention / Transformer Related | Others - insufficient information to determine root cause |  |
| [2598](https://github.com/intel/torch-xpu-ops/issues/2598) | [TorchAO][BMG]The first token latency of | Stonepia | Stonepia | P2 | TorchAO | Timeout/Performance Issue - First token latency drops significantly with change  |  |
| [2609](https://github.com/intel/torch-xpu-ops/issues/2609) | [upstream_ut]  torch._inductor.exc.Induc | daisyden | daisyden | P2 | PT2E |  | [PR](https://github.com/pytorch/pytorch/pull/171154) |
| [2620](https://github.com/intel/torch-xpu-ops/issues/2620) | [upstream_ut]  AssertionError: dtype is  | daisyden | daisyden | P2 | TorchAO |  |  |
| [2656](https://github.com/intel/torch-xpu-ops/issues/2656) | [release/2.10] models got fail_accuracy  | None | None | P0 | Dtype / Precision Related | Backend/Device Issue - failure occurs specifically on BMG WSL2 XPU backend |  |
| [2660](https://github.com/intel/torch-xpu-ops/issues/2660) | [release/2.10][Windows][BMG] New failed  | tadkrawiec | tadkrawiec | P2 | Others | Others - insufficient information to determine root cause |  |
| [2662](https://github.com/intel/torch-xpu-ops/issues/2662) | [release/2.10][Windows][BMG] New failed  | tadkrawiec, kdrozd-dev | tadkrawiec, kdrozd-dev | P2 | Others | Backend/Device Issue - XPU related failure in test cases on Windows with BMG |  |
| [2663](https://github.com/intel/torch-xpu-ops/issues/2663) | test_sparse_semi_structured.py gaps | None | None | P2 | Sparse Operations Related |  |  |
| [2670](https://github.com/intel/torch-xpu-ops/issues/2670) | [upstream_ut]  RuntimeError: could not c | tszulist-hbn | tszulist-hbn | P2 | Distributed |  |  |
| [2693](https://github.com/intel/torch-xpu-ops/issues/2693) | Title: [upstream_ut]  AssertionError: Sc | hoshibara | hoshibara | P2 | Distributed |  |  |
| [2698](https://github.com/intel/torch-xpu-ops/issues/2698) | Title: [upstream_ut]  RuntimeError: Flas | chunhuanMeng, LuFinch | chunhuanMeng, LuFinch | P2 | PT2E |  |  |
| [2704](https://github.com/intel/torch-xpu-ops/issues/2704) | Title: [upstream_ut]  AssertionError: As | kdrozd-dev | kdrozd-dev | P2 | Flash Attention / Transformer Related |  | [PR](https://github.com/pytorch/pytorch/pull/177636) |
| [2712](https://github.com/intel/torch-xpu-ops/issues/2712) | [upstream_ut]  RuntimeError: Cannot swap | tszulist-hbn | tszulist-hbn | P2 | Others |  |  |
| [2714](https://github.com/intel/torch-xpu-ops/issues/2714) | [upstream_ut]  AssertionError: Object co | Silv3S | Silv3S | P2 | Distributed |  |  |
| [2715](https://github.com/intel/torch-xpu-ops/issues/2715) | [upstream_ut]  torch._dynamo.exc.Unsuppo | CuiYifeng | CuiYifeng | P2 | PT2E |  |  |
| [2722](https://github.com/intel/torch-xpu-ops/issues/2722) | [Bug Skip]: NotImplementedError: Could n | Silv3S | CuiYifeng | P2 | TorchAO | Backend/Device Issue - aten::flip not implemented for QuantizedXPU backend |  |
| [2798](https://github.com/intel/torch-xpu-ops/issues/2798) | Test case  test/test_dlpack.py::TestTorc | None | None | P2 | Others |  |  |
| [2800](https://github.com/intel/torch-xpu-ops/issues/2800) | AttributeError: 'torch._C._XpuDeviceProp | guangyey | guangyey | P2 | Flash Attention / Transformer Related |  |  |
| [2802](https://github.com/intel/torch-xpu-ops/issues/2802) | Three aten._scaled_dot_product_flash_att | LuFinch | LuFinch | P2 | Flash Attention / Transformer Related |  |  |
| [2806](https://github.com/intel/torch-xpu-ops/issues/2806) | CompiledAOTI need XPU support | daisyden | daisyden | P2 | PT2E |  |  |
| [2810](https://github.com/intel/torch-xpu-ops/issues/2810) | AssertionError: Object comparison failed | daisyden | daisyden | P2 | Dtype / Precision Related |  |  |
| [2888](https://github.com/intel/torch-xpu-ops/issues/2888) | torch._inductor.exc.InductorError: Asser | Stonepia | Stonepia | P2 | Inductor / Compilation Related |  |  |
| [2907](https://github.com/intel/torch-xpu-ops/issues/2907) | [release/2.11] Models performance regres | xuhancn | xuhancn | P0 | Others | Timeout/Performance Issue - models performance regression in testcases |  |
| [2912](https://github.com/intel/torch-xpu-ops/issues/2912) | [release/2.11] UT extended 220 new failu | None | None | P2 | Others | Others - insufficient information to determine root cause |  |
| [2918](https://github.com/intel/torch-xpu-ops/issues/2918) | [XPU][upstream_ut][COW] Skip non-support | gplutop7 | gplutop7 | P2 | Others |  | [PR](https://github.com/pytorch/pytorch/pull/174670) |
| [2919](https://github.com/intel/torch-xpu-ops/issues/2919) | [XPU][upstream_ut][COW] Fix materializat | gplutop7 | gplutop7 | P2 | Others |  |  |
| [2922](https://github.com/intel/torch-xpu-ops/issues/2922) | [release/2.11] UT inductor AssertionErro | tadkrawiec | tadkrawiec | P2 | Inductor / Compilation Related | Backend/Device Issue - pass_fds not supported on Windows |  |
| [2930](https://github.com/intel/torch-xpu-ops/issues/2930) | [release/2.11] UT skip test_binary_ufunc | None | None | P2 | Others | Skip/No Test Exists - test is skipped due to RuntimeError |  |
| [2952](https://github.com/intel/torch-xpu-ops/issues/2952) | [release/2.11][wsl] timm_models_accuracy | weishi-deng | weishi-deng | P0 | Dtype / Precision Related | Dtype/Precision Issue - bfloat16 accuracy failure in model training |  |
| [2958](https://github.com/intel/torch-xpu-ops/issues/2958) | AssertionError of test_dtensor_basic_com | daisyden | daisyden | P2 | Inductor / Compilation Related |  |  |
| [2960](https://github.com/intel/torch-xpu-ops/issues/2960) | [release/2.11] timm_models_xcit_large_24 | None | None | P0 | Dtype / Precision Related | Dtype/Precision Issue - float16 training accuracy test failure |  |
| [2997](https://github.com/intel/torch-xpu-ops/issues/2997) | AssertionError of test_linear_and_cel_ma | etaf | etaf | P2 | Flash Attention / Transformer Related |  |  |
| [2999](https://github.com/intel/torch-xpu-ops/issues/2999) | KeyError: 'eager_numerics.use_pytorch_li | daisyden | daisyden | P2 | Others |  |  |
| [3004](https://github.com/intel/torch-xpu-ops/issues/3004) | TypeError: _xpu_recordMemoryHistory(): i | guangyey | guangyey | P2 | Others |  |  |
| [3041](https://github.com/intel/torch-xpu-ops/issues/3041) | AssertionError: Expected len(flat_diff_r | Silv3S | Silv3S | P2 | Dtype / Precision Related |  |  |
| [3077](https://github.com/intel/torch-xpu-ops/issues/3077) | [Bug Skip] test_dlpack.py::TestTorchDlPa | AKloniecki | AKloniecki | P2 | Others |  |  |
| [3094](https://github.com/intel/torch-xpu-ops/issues/3094) | XPUGraph tree support | None | None | P2 | Inductor / Compilation Related |  |  |
| [3095](https://github.com/intel/torch-xpu-ops/issues/3095) | cutlass support blocks some unit test ca | None | None | P2 | Others |  |  |
| [3126](https://github.com/intel/torch-xpu-ops/issues/3126) | [upstream_ut]  Two NestedTensor issue wi | LuFinch | LuFinch | P2 | Flash Attention / Transformer Related |  |  |
| [3127](https://github.com/intel/torch-xpu-ops/issues/3127) | [upstream_ut]  AssertionError: Assertion | LuFinch | LuFinch | P2 | Flash Attention / Transformer Related |  |  |
| [3128](https://github.com/intel/torch-xpu-ops/issues/3128) | [upstream_ut]  AssertionError: RuntimeEr | LuFinch | LuFinch | P2 | Distributed |  |  |
| [3129](https://github.com/intel/torch-xpu-ops/issues/3129) | [upstream_ut]  AssertionError: UserWarni | LuFinch | LuFinch | P2 | Flash Attention / Transformer Related |  |  |
| [3130](https://github.com/intel/torch-xpu-ops/issues/3130) | [upstream_ut]  AssertionError: tensor(Tr | LuFinch | LuFinch | P2 | Flash Attention / Transformer Related |  |  |
| [3131](https://github.com/intel/torch-xpu-ops/issues/3131) | [upstream_ut]  NotImplementedError: The  | chunhuanMeng | chunhuanMeng | P2 | Flash Attention / Transformer Related |  |  |
| [3132](https://github.com/intel/torch-xpu-ops/issues/3132) | [upstream_ut]  transfomers test reports  | LuFinch | LuFinch | P2 | Flash Attention / Transformer Related |  |  |
| [3133](https://github.com/intel/torch-xpu-ops/issues/3133) | [upstream_ut]  RuntimeError: scaled_dot_ | LuFinch | LuFinch | P2 | Flash Attention / Transformer Related |  |  |
| [3137](https://github.com/intel/torch-xpu-ops/issues/3137) | [upstream_ut]  RuntimeError: expected sc | LuFinch | LuFinch | P2 | Flash Attention / Transformer Related |  |  |
| [3140](https://github.com/intel/torch-xpu-ops/issues/3140) | [upstream_ut]  RuntimeError: FlashAttent | LuFinch | LuFinch | P2 | Flash Attention / Transformer Related |  |  |
| [3141](https://github.com/intel/torch-xpu-ops/issues/3141) | [upstream_ut]  RuntimeError: FlashAttent | LuFinch | LuFinch | P2 | Flash Attention / Transformer Related |  |  |
| [3142](https://github.com/intel/torch-xpu-ops/issues/3142) | [upstream_ut]  RuntimeError: The sycl_ex | LuFinch | LuFinch | P2 | Flash Attention / Transformer Related |  |  |
| [3143](https://github.com/intel/torch-xpu-ops/issues/3143) | NotImplementedError: The operator 'aten: | LuFinch | LuFinch | P2 | Flash Attention / Transformer Related |  |  |
| [3163](https://github.com/intel/torch-xpu-ops/issues/3163) | [Bug Skip]: Object comparison failed: to | chunhuanMeng | chunhuanMeng | P2 | Distributed |  |  |
| [3165](https://github.com/intel/torch-xpu-ops/issues/3165) | test_sparse_csr_xpu.py::TestSparseCompre | None | None | P2 | Sparse Operations Related |  |  |
| [3167](https://github.com/intel/torch-xpu-ops/issues/3167) | NotImplementedError: Could not run 'aten | tszulist-hbn | tszulist-hbn | P1 | Sparse Operations Related |  |  |
| [3168](https://github.com/intel/torch-xpu-ops/issues/3168) | NotImplementedError: Could not run 'aten | jenniew | jenniew | P1 | Sparse Operations Related |  |  |
| [3169](https://github.com/intel/torch-xpu-ops/issues/3169) | NotImplementedError: Could not run 'aten | jkosnox | jkosnox | P2 | Sparse Operations Related |  |  |
| [3170](https://github.com/intel/torch-xpu-ops/issues/3170) | Unskip test_bmm_windows_error_xpu_float6 | jenniew | jenniew | P2 | Others |  |  |
| [3187](https://github.com/intel/torch-xpu-ops/issues/3187) | PyTorch XPU gpu_cpp_wrapper fails with I | CuiYifeng | CuiYifeng | P2 | Inductor / Compilation Related |  |  |
| [3229](https://github.com/intel/torch-xpu-ops/issues/3229) | RuntimeError: No viable backend for scal | None | None | P2 | Flash Attention / Transformer Related |  |  |
| [3238](https://github.com/intel/torch-xpu-ops/issues/3238) | The supported dtypes of _refs.stft is no | CuiYifeng | CuiYifeng | P2 | Dtype / Precision Related |  |  |
| [3247](https://github.com/intel/torch-xpu-ops/issues/3247) | NotImplementedError: "dot_xpu_mkl" not i | Silv3S | Silv3S | P2 | Others |  |  |

#### Revisit the PR as Case Failed {#revisit-the-pr-as-case-failed}

| ID | Title | Owner | Owner Transferred | Priority | Category | Root Cause | PR |
|---|-------|-------|-------------------|---------|----------|-----------|-----|
| [3246](https://github.com/intel/torch-xpu-ops/issues/3246) | AssertionError: Booleans mismatch: True  | BartoszKokoszko | BartoszKokoszko | P2 | Distributed |  | [PR](https://github.com/intel/torch-xpu-ops/pull/3249) |

---

## 5. By Category {#5-by-category}

#### Distributed (#distributed)

| ID | Title | Status | Owner | Priority | Root Cause | PR | Labels | Module |
|---|-------|--------|-------|---------|-----------|-----|--------|--------|
| [1059](https://github.com/intel/torch-xpu-ops/issues/1059) | SYCL RT: Using recommended sho | open | CuiYifeng, jianyizh | P2 | Backend/Device Issue - related to SYCL RT and kernel-specific max work group siz |  | dependency component: one.. | aten_ops |
| [1547](https://github.com/intel/torch-xpu-ops/issues/1547) | [distributed] NotImplementedEr | open | Chao1Han | P2 | Backend/Device Issue - XPU device does not support the 'symm_mem::fused_matmul_r |  | module: distributed, depe.. | distributed |
| [1548](https://github.com/intel/torch-xpu-ops/issues/1548) | [distributed] AssertionError:  | open | Chao1Han | P2 | Failure - 'fused_all_gather_matmul' not found in AOT ID list |  | module: distributed, depe.. | distributed |
| [1549](https://github.com/intel/torch-xpu-ops/issues/1549) | [distributed] AssertionError:  | open | Chao1Han | P2 | Distributed/Gloo Issue - 'fused_all_gather_scaled_matmul' not found in graph dur |  | module: distributed, depe.. | distributed |
| [1551](https://github.com/intel/torch-xpu-ops/issues/1551) | [distributed] NotImplementedEr | open | Chao1Han | P2 | Backend/Device Issue - XPU device does not support the 'symm_mem::fused_scaled_m |  | module: distributed, depe.. | distributed |
| [1555](https://github.com/intel/torch-xpu-ops/issues/1555) | [distributed] RuntimeError: at | open | chuanqi129 | P2 | Distributed/Gloo Issue - mixed torch.Tensor and DTensor in distributed operation |  | module: distributed, depe.. | distributed |
| [1556](https://github.com/intel/torch-xpu-ops/issues/1556) | [distributed] NotImplementedEr | open | pkourdis | P2 | Distributed/Gloo Issue - Operator lacks a sharding strategy in distributed conte |  | module: distributed, depe.. | distributed |
| [1571](https://github.com/intel/torch-xpu-ops/issues/1571) | [distributed] ValueError: Cann | open | zhangxiaoli73 | P2 | Distributed/Gloo Issue - ReduceOp.PREMUL_SUM not supported with XCCL |  | module: distributed | distributed |
| [1661](https://github.com/intel/torch-xpu-ops/issues/1661) | [distributed] Accuracy gap in  | open | githubsgi | P2 | Distributed/Gloo Issue - Accuracy gap in _composable/fsdp on Xelink suggests a d |  | bug, module: distributed | distributed |
| [1727](https://github.com/intel/torch-xpu-ops/issues/1727) | [distributed] AttributeError:  | open | guangyey | P2 | Backend/Device Issue - missing attribute '_sleep' in torch.xpu module |  | module: distributed, depe.. | distributed |
| [1893](https://github.com/intel/torch-xpu-ops/issues/1893) | [upstream_ut] oneDNN accuracy  | open | chunhuanMeng | P2 |  |  | skipped, ut_upstream | aten_ops |
| [1951](https://github.com/intel/torch-xpu-ops/issues/1951) | Functionality issues in TestCo | open | AKloniecki | P2 |  |  | module: ut, skipped, ut_u.. | aten_ops |
| [1973](https://github.com/intel/torch-xpu-ops/issues/1973) | AssertionError: Scalars or Ten | open | gplutop7 | P2 | Failure - Tensor values not equal or close in test_depthwise_conv_64bit_indexing_xpu |  | hw: PVC, module: ut, skip.. | aten_ops |
| [2004](https://github.com/intel/torch-xpu-ops/issues/2004) | [distributed][shared_tensor] t | open | libohao1201 | P2 | Memory/Shared Memory Issue - error originated from shared memory connection in t |  | bug, module: distributed | distributed |
| [2024](https://github.com/intel/torch-xpu-ops/issues/2024) | AssertionError: Torch not comp | open | daisyden | P2 |  |  | module: ut, skipped | aten_ops |
| [2113](https://github.com/intel/torch-xpu-ops/issues/2113) | Update example for Distributed | open | songhappy | P2 | Distributed/Gloo Issue - related to Distributed Data Parallel update example |  | module: distributed | distributed |
| [2163](https://github.com/intel/torch-xpu-ops/issues/2163) | 3 distributed UT cases need to | open | githubsgi | P2 | Distributed/Gloo Issue - distributed UT cases need support from sac_estimator.py |  | module: distributed | distributed |
| [2164](https://github.com/intel/torch-xpu-ops/issues/2164) | skip test_no_cuda_monkeypatch  | open | daisyden | P2 |  |  | wontfix, skipped | aten_ops |
| [2165](https://github.com/intel/torch-xpu-ops/issues/2165) | [distributed] test_device_mesh | open | jemitche1 | P2 | Failure - test_flatten_mesh_3d encountered an assertion error |  | bug, module: distributed | distributed |
| [2169](https://github.com/intel/torch-xpu-ops/issues/2169) | Frame size comparison failed i | open | guangyey | P2 | Failure - Scalars are not equal in test comparison |  | skipped | aten_ops |
| [2182](https://github.com/intel/torch-xpu-ops/issues/2182) | test_transform_bias_rescale_qk | open | PawelSwider2000 | P2 | Failure - Scalars are not equal in test assertion |  | Accuracy, module: ut, ski.. | aten_ops |
| [2214](https://github.com/intel/torch-xpu-ops/issues/2214) | test/test_sparse.py::TestSpars | open | jenniew | P2 |  |  | skipped, ut_upstream | aten_ops |
| [2229](https://github.com/intel/torch-xpu-ops/issues/2229) | test/test_sparse_csr.py::TestS | open | jenniew | P2 | Backend/Device Issue - device mismatch between crow_indices and col_indices on X |  | skipped | aten_ops |
| [2230](https://github.com/intel/torch-xpu-ops/issues/2230) | test_sparse_csr.py::TestSparse | open | None | P1 | Backend/Device Issue - inputs are not on the same GPU device |  | skipped | aten_ops |
| [2234](https://github.com/intel/torch-xpu-ops/issues/2234) | [upstream_ut] AssertionError:  | open | Silv3S | P2 |  |  | module: ut, skipped, ut_u.. | aten_ops |
| [2235](https://github.com/intel/torch-xpu-ops/issues/2235) | test/test_sparse_csr.py::TestS | open | None | P2 | Backend/Device Issue - unexpected warning from non-optimal triton kernel paramet |  | skipped | aten_ops |
| [2238](https://github.com/intel/torch-xpu-ops/issues/2238) | Exception: Tensor-likes are no | open | BBBela | P2 |  | [PR](https://github.com/intel/torch-xpu-ops/pull/2886) | skipped, bug_fix_stage3 | aten_ops |
| [2239](https://github.com/intel/torch-xpu-ops/issues/2239) | Exception: could not create a  | open | wpietka | P2 | DNNL/OneDNN Issue - could not create a primitive descriptor for deconvolution fo |  | skipped, bug_fix_stage5 | aten_ops |
| [2240](https://github.com/intel/torch-xpu-ops/issues/2240) | RuntimeError: Trying to set a  | open | gplutop7 | P2 | Error - forward gradient size mismatch with original Tensor |  | skipped, bug_fix_stage3 | aten_ops |
| [2244](https://github.com/intel/torch-xpu-ops/issues/2244) | test/test_sparse_csr.py::TestS | open | jenniew | P2 | Error - empty_sparse_compressed expects non-block layout but received SparseBsr |  | module: ut, skipped | aten_ops |
| [2245](https://github.com/intel/torch-xpu-ops/issues/2245) | oneDNN matmul received incorre | open | CuiYifeng | P2 | DNNL/OneDNN Issue - oneDNN matmul primitive descriptor creation failure |  | module: ut, skipped | aten_ops |
| [2246](https://github.com/intel/torch-xpu-ops/issues/2246) | torch/sparse/_triton_ops*.py n | open | None | P1 | Backend/Device Issue - Torch not compiled with CUDA enabled for Intel GPU suppor |  | skipped | unknown |
| [2248](https://github.com/intel/torch-xpu-ops/issues/2248) | [upstream_ut] test_cow failure | open | gplutop7 | P2 |  |  | skipped, bug_fix_stage3, .. | aten_ops |
| [2257](https://github.com/intel/torch-xpu-ops/issues/2257) | Accuracy failures in test/xpu/ | open | pbielak | P1 |  | [PR](https://github.com/intel/torch-xpu-ops/pull/2808) | skipped, bug_fix_stage4 | aten_ops |
| [2270](https://github.com/intel/torch-xpu-ops/issues/2270) | Backend Compatibility Error in | open | LuFinch | P2 | Flash Attention/Specific Ops Issue - test involves aten._flash_attention_forward |  | module: ut, skipped | aten_ops |
| [2283](https://github.com/intel/torch-xpu-ops/issues/2283) | [upstream_ut] sparse._sampled_ | open | jenniew | P1 |  |  | skipped, ut_upstream | aten_ops |
| [2301](https://github.com/intel/torch-xpu-ops/issues/2301) | [upstream_ut] dtypes not align | open | daisyden | P2 |  |  | skipped, ut_upstream | aten_ops |
| [2340](https://github.com/intel/torch-xpu-ops/issues/2340) | [distributed][_tools] Assertio | open | githubsgi | P2 | Failure - Roofline estimation requires CUDA capabilities assertion failed |  | bug, duplicate, module: d.. | distributed |
| [2359](https://github.com/intel/torch-xpu-ops/issues/2359) | [upstream_ut] GradcheckError:  | open | BBBela | P2 |  |  | skipped, ut_upstream | aten_ops |
| [2389](https://github.com/intel/torch-xpu-ops/issues/2389) | [Bug Skip]: RuntimeError: Data | open | PatrykWilczewski | P2 | Error - Data corruption detected during test execution |  | skipped, bug_fix_stage4, .. | aten_ops |
| [2404](https://github.com/intel/torch-xpu-ops/issues/2404) | [distributed][checkpoint] Asse | open | None | P2 | Failure - Booleans mismatch assertion error |  | bug | aten_ops |
| [2446](https://github.com/intel/torch-xpu-ops/issues/2446) | [Bug Skip]: AssertionError: "S | open | None | P2 | Failure - mismatch between expected and actual error messages |  | skipped, random | aten_ops |
| [2529](https://github.com/intel/torch-xpu-ops/issues/2529) | [upstream_ut]  AssertionError: | open | Silv3S | P2 | Failure - test assertion failed with False is not true |  | skipped, port_from_skipli.. | aten_ops |
| [2530](https://github.com/intel/torch-xpu-ops/issues/2530) | Title: [upstream_ut]  Assertio | open | PatrykWilczewski | P2 | Failure - RuntimeError not raised as expected in test |  | skipped, bug_fix_stage5, .. | aten_ops |
| [2531](https://github.com/intel/torch-xpu-ops/issues/2531) | [upstream_ut]  AssertionError: | open | daisyden | P2 |  |  | skipped, port_from_skipli.. | unknown |
| [2532](https://github.com/intel/torch-xpu-ops/issues/2532) | Title: [upstream_ut]  Assertio | open | yucai-intel | P2 | Failure - wrong number of dimensions for int4 conversion op |  | skipped, port_from_skipli.. | aten_ops |
| [2533](https://github.com/intel/torch-xpu-ops/issues/2533) | Title: [upstream_ut]  Attribut | open | astachowiczhabana | P2 | Mismatch - Test method 'test_qsoftmax' is missing in 'TestQuantizedOpsXPU' class. |  | skipped, port_from_skipli.. | aten_ops |
| [2535](https://github.com/intel/torch-xpu-ops/issues/2535) | Title: [upstream_ut]  Attribut | open | Silv3S | P2 | Backend/Device Issue - missing XPU-specific attribute in torch._C for tunableop  |  | skipped, port_from_skipli.. | aten_ops |
| [2536](https://github.com/intel/torch-xpu-ops/issues/2536) | Title: [upstream_ut]  Attribut | open | daisyden | P2 | Backend/Device Issue - missing attribute '_scatter' in torch._C for XPU device |  | skipped, port_from_skipli.. | aten_ops |
| [2538](https://github.com/intel/torch-xpu-ops/issues/2538) | Title: [upstream_ut]  RuntimeE | open | Silv3S | P2 | DNNL/OneDNN Issue - Float8_e4m3fnuz is not supported in oneDNN. |  | skipped, port_from_skipli.. | aten_ops |
| [2539](https://github.com/intel/torch-xpu-ops/issues/2539) | Title: [upstream_ut]  RuntimeE | open | None | P2 | Backend/Device Issue - Attempt to instantiate dummy base class CUDAGraph on XPU |  | skipped, port_from_skipli.. | aten_ops |
| [2541](https://github.com/intel/torch-xpu-ops/issues/2541) | Title: [upstream_ut]  RuntimeE | open | yucai-intel | P2 | DNNL/OneDNN Issue - could not construct a memory descriptor using strides |  | skipped, port_from_skipli.. | aten_ops |
| [2611](https://github.com/intel/torch-xpu-ops/issues/2611) | [upstream_ut]  AssertionError: | open | daisyden | P2 |  |  | dependency component: dri.. | inductor |
| [2613](https://github.com/intel/torch-xpu-ops/issues/2613) | [upstream_ut]  AssertionError: | open | daisyden | P2 |  |  | dependency component: dri.. | inductor |
| [2615](https://github.com/intel/torch-xpu-ops/issues/2615) | [Bug Skip]: New failures Runti | open | CuiYifeng | P2 | Dtype/Precision Issue - Unsupported dtype Half / torch.float16 |  | module: ut, skipped | aten_ops |
| [2618](https://github.com/intel/torch-xpu-ops/issues/2618) | [Bug Skip]: [regression] Asser | open | jmamzax | P0 |  | [PR](https://github.com/numpy/numpy/pull/22525) | skipped, bug_fix_stage5 | unknown |
| [2630](https://github.com/intel/torch-xpu-ops/issues/2630) | Title: [upstream_ut]  Assertio | open | jmamzax | P2 | Failure - Scalars are not equal! |  | skipped, port_from_skipli.. | aten_ops |
| [2639](https://github.com/intel/torch-xpu-ops/issues/2639) | test_to() failed during rnn is | open | Silv3S | P2 | Failure - test_to() failed during rnn isinstance() check | [PR](https://github.com/intel/torch-xpu-ops/pull/3193) | skipped | aten_ops |
| [2640](https://github.com/intel/torch-xpu-ops/issues/2640) | random issue test_vjpvjp_index | open | wpietka | P2 | Others - incomplete traceback and insufficient information to determine root cause |  | skipped, random | aten_ops |
| [2649](https://github.com/intel/torch-xpu-ops/issues/2649) | [distributed][pipelining] test | open | syedshahbaaz | P2 | Distributed/Gloo Issue - test_schedule_multiproc.py is hanging during distribute |  | bug, module: distributed | distributed |
| [2659](https://github.com/intel/torch-xpu-ops/issues/2659) | [distributed] test_dist2.py Ru | open | Chao1Han | P2 | Distributed/Gloo Issue - Backend xccl does not implement getBackendOptions. |  | module: distributed | distributed |
| [2669](https://github.com/intel/torch-xpu-ops/issues/2669) | [upstream_ut]  AssertionError: | open | tszulist-hbn | P2 | Failure - Tensor-likes are not close! in test_vmap_exhaustive_addmv_xpu_float32 |  | skipped | aten_ops |
| [2670](https://github.com/intel/torch-xpu-ops/issues/2670) | [upstream_ut]  RuntimeError: c | open | tszulist-hbn | P2 |  |  | skipped, ut_upstream | aten_ops |
| [2686](https://github.com/intel/torch-xpu-ops/issues/2686) | [distributed] Accuracy issues  | open | frost-intel | P2 | Distributed/Gloo Issue - Accuracy issues in distributed testing with test_distri |  | bug, module: distributed | distributed |
| [2693](https://github.com/intel/torch-xpu-ops/issues/2693) | Title: [upstream_ut]  Assertio | open | hoshibara | P2 |  |  | module: inductor, skipped.. | inductor |
| [2694](https://github.com/intel/torch-xpu-ops/issues/2694) | Title: [upstream_ut]  Assertio | open | daisyden | P2 |  | [PR](https://github.com/pytorch/pytorch/pull/171773) | module: inductor, skipped.. | inductor |
| [2700](https://github.com/intel/torch-xpu-ops/issues/2700) | [distributed] Hang issues with | open | syedshahbaaz | P2 | Distributed/Gloo Issue - Hang issues with test_distributed_spawn.py suggest a pr |  | bug, module: distributed | distributed |
| [2701](https://github.com/intel/torch-xpu-ops/issues/2701) | [distributed] Barrier Timeout  | open | syedshahbaaz | P2 | Distributed/Gloo Issue - Barrier Timeout Error in distributed testing |  | bug, module: distributed | distributed |
| [2702](https://github.com/intel/torch-xpu-ops/issues/2702) | [distributed] RuntimeError: Wo | open | syedshahbaaz | P2 | Timeout/Performance Issue - Work ran time out after 0 milliseconds in distribute |  | bug, module: distributed | distributed |
| [2714](https://github.com/intel/torch-xpu-ops/issues/2714) | [upstream_ut]  AssertionError: | open | Silv3S | P2 |  |  | skipped, ut_upstream | aten_ops |
| [2720](https://github.com/intel/torch-xpu-ops/issues/2720) | [upstream_ut] RuntimeError: fa | open | CuiYifeng | P2 | Backend/Device Issue - missing kernel for xpu in DispatchStub |  | skipped | aten_ops |
| [2737](https://github.com/intel/torch-xpu-ops/issues/2737) | [distributed] AttributeError:  | open | None | P2 | Distributed/Gloo Issue - missing attribute '_gather' in distributed context |  | bug, module: distributed | distributed |
| [2738](https://github.com/intel/torch-xpu-ops/issues/2738) | [distributed] test_c10d_spawn_ | open | jenniew | P2 | Distributed/Gloo Issue - input tensor size mismatch in distributed context |  | bug, module: distributed | distributed |
| [2759](https://github.com/intel/torch-xpu-ops/issues/2759) | [Bug Skip]: New failed cases 2 | open | AKloniecki | P2 | Failure - RuntimeError not raised as expected in test case |  | skipped | aten_ops |
| [2779](https://github.com/intel/torch-xpu-ops/issues/2779) | Accuracy failures in logspace  | open | PawelSwider2000 | P2 | Dtype/Precision Issue - accuracy failures in logspace operation |  | module: ut, skipped, bug_.. | aten_ops |
| [2811](https://github.com/intel/torch-xpu-ops/issues/2811) | [Bug Skip]: [Regression] faile | open | jmamzax | P0 | Failure - Expected and actual trace outputs do not match. |  | skipped, bug_fix_stage5 | aten_ops |
| [2837](https://github.com/intel/torch-xpu-ops/issues/2837) | Accuracy issue for Muon optimi | open | Silv3S | P2 | Failure - Tensor-likes not close in Muon optimizer test |  | skipped, bug_fix_stage5 | aten_ops |
| [2840](https://github.com/intel/torch-xpu-ops/issues/2840) | Accuracy issue with 64 bit ind | open | SlawomirLaba, Silv3S | P2 | Dtype/Precision Issue - Tensor comparison failed due to small absolute differenc |  | skipped, bug_fix_stage5 | aten_ops |
| [2853](https://github.com/intel/torch-xpu-ops/issues/2853) | [upstream_ut] torch.ops.aten._ | open | LuFinch | P2 | Flash Attention/Specific Ops Issue - _flash_attention_forward not supported on X |  | skipped | aten_ops |
| [2871](https://github.com/intel/torch-xpu-ops/issues/2871) | [distributed][fsdp] test_fsdp_ | open | songhappy | P2 | Failure - test_fsdp_overlap.py assertion failed with "False is not true" |  | bug, module: distributed | distributed |
| [2921](https://github.com/intel/torch-xpu-ops/issues/2921) | [abs][complex64] - new failing | open | AKloniecki | P2 | Dtype/Precision Issue - test failure related to complex64 data type and abs oper |  | skipped | aten_ops |
| [2966](https://github.com/intel/torch-xpu-ops/issues/2966) | [Bug Skip]: [Regression]2026-3 | open | jmamzax | P0 | Timeout/Performance Issue - Example code timed out during test execution. |  | skipped, bug_fix_stage5, .. | aten_ops |
| [2967](https://github.com/intel/torch-xpu-ops/issues/2967) | [distributed] feature gaps in  | open | frost-intel | P2 | Distributed/Gloo Issue - feature gaps in distributed testing for XPU with test_c |  | bug, module: distributed | distributed |
| [2968](https://github.com/intel/torch-xpu-ops/issues/2968) | [distributed] timeout issue in | open | frost-intel | P2 | Timeout/Performance Issue - test experienced a timeout in distributed execution  |  | bug, module: distributed | distributed |
| [2969](https://github.com/intel/torch-xpu-ops/issues/2969) | [distributed] AssertionError:  | open | frost-intel | P2 | Failure - Scalars are not equal in test case |  | bug, module: distributed | distributed |
| [2971](https://github.com/intel/torch-xpu-ops/issues/2971) | [distributed] KeyError in test | open | newtdms, frost-intel | P2 | Distributed/Gloo Issue - KeyError in distributed test related to c10d_xccl |  | bug, module: distributed | distributed |
| [2972](https://github.com/intel/torch-xpu-ops/issues/2972) | [distributed] AssertionError:  | open | newtdms | P2 | Distributed/Gloo Issue - AssertionError in test/distributed/test_c10d_xccl.py re |  | bug, module: distributed | distributed |
| [3010](https://github.com/intel/torch-xpu-ops/issues/3010) | [distributed][tensor] test_ran | open | jenniew | P2 | Inductor/Compilation Issue - TorchRuntimeError during fake tensor call in Dynamo |  | bug, module: distributed | distributed |
| [3021](https://github.com/intel/torch-xpu-ops/issues/3021) | [distributed] all_to_all_singl | open | zhangxiaoli73 | P2 | Distributed/Gloo Issue - all_to_all_single compatibility issue on B60/XCCL |  | module: distributed | distributed |
| [3022](https://github.com/intel/torch-xpu-ops/issues/3022) | [distributed] batch_isend_irec | open | zhangxiaoli73 | P2 | Distributed/Gloo Issue - batch_isend_irecv compatibility issue on B60/XCCL |  | module: distributed | distributed |
| [3025](https://github.com/intel/torch-xpu-ops/issues/3025) | New failing test in Nightly Wh | open | None | P2 | Failure - Expected and actual decomposition outputs do not match. |  | skipped, random | aten_ops |
| [3082](https://github.com/intel/torch-xpu-ops/issues/3082) | multithread support in distrib | open | None | P2 | Distributed/Gloo Issue - multithread support in distributed operations is affect |  | module: distributed, modu.. | distributed |
| [3100](https://github.com/intel/torch-xpu-ops/issues/3100) | [distributed] /handler/dump_nc | open | None | P2 | Distributed/Gloo Issue - related to NCCL logging and trace handling in distribut |  | module: distributed | distributed |
| [3101](https://github.com/intel/torch-xpu-ops/issues/3101) | [distributed] 'torch._C._distr | open | None | P2 | Distributed/Gloo Issue - ProcessGroupXCCL object lacks '_set_default_timeout' at |  | module: distributed | distributed |
| [3102](https://github.com/intel/torch-xpu-ops/issues/3102) | [distributed] RuntimeError: In | open | None | P2 | Backend/Device Issue - invalid device string 'xpu:foo' indicates a problem with  |  | module: distributed | distributed |
| [3105](https://github.com/intel/torch-xpu-ops/issues/3105) | Wrong results from oneDNN conv | open | BBBela | P2 | DNNL/OneDNN Issue - wrong results from oneDNN conv2d kernels due to data pointer |  | hw: PVC, module: ut, skip.. | aten_ops |
| [3128](https://github.com/intel/torch-xpu-ops/issues/3128) | [upstream_ut]  AssertionError: | open | LuFinch | P2 |  |  | module: ut, skipped, ut_u.. | aten_ops |
| [3139](https://github.com/intel/torch-xpu-ops/issues/3139) | [distributed][_composable] Ass | open | Kanya-Mo | P2 | Failure - Expects xpu:0 but got xpu:1 |  | bug, module: distributed | distributed |
| [3163](https://github.com/intel/torch-xpu-ops/issues/3163) | [Bug Skip]: Object comparison  | open | chunhuanMeng | P2 |  |  | skipped, ut_upstream | aten_ops |
| [3174](https://github.com/intel/torch-xpu-ops/issues/3174) | [Bug Skip]: Accuracy failure o | open | None | P2 | Failure - test assertion failed for Conv2d groups output comparison |  | module: ut, skipped | aten_ops |
| [3177](https://github.com/intel/torch-xpu-ops/issues/3177) | Accuracy gap of BF16/FP16 test | open | jenniew | P2 | Dtype/Precision Issue - Accuracy gap in BF16/FP16 test |  | skipped | aten_ops |
| [3194](https://github.com/intel/torch-xpu-ops/issues/3194) | Incorrect strides in TestCommo | open | AKloniecki | P2 | Backend/Device Issue - Incorrect strides related to XPU device handling |  |  | aten_ops |
| [3232](https://github.com/intel/torch-xpu-ops/issues/3232) | [distributed][tensor] Assertio | open | None | P2 | Failure - AssertionError not raised for Placement (Shard(dim=2),) in test |  | bug, module: distributed | distributed |
| [3233](https://github.com/intel/torch-xpu-ops/issues/3233) | [distributed] RuntimeError: No | open | None | P2 | Distributed/Gloo Issue - No backend for the parent process group or its backend  |  | bug, module: distributed | distributed |
| [3246](https://github.com/intel/torch-xpu-ops/issues/3246) | AssertionError: Booleans misma | open | BartoszKokoszko | P2 |  | [PR](https://github.com/intel/torch-xpu-ops/pull/3249) | skipped | aten_ops |

#### Dtype / Precision Related (#dtype---precision-related)

| ID | Title | Status | Owner | Priority | Root Cause | PR | Labels | Module |
|---|-------|--------|-------|---------|-----------|-----|--------|--------|
| [1624](https://github.com/intel/torch-xpu-ops/issues/1624) | [DONT CLOSE] Known UT Issue li | open | None | P2 |  |  | module: distributed, modu.. | distributed |
| [1818](https://github.com/intel/torch-xpu-ops/issues/1818) | [BMG-Windows][PT2.8]Torch-xpu- | open | kdrozd-dev | P2 | Backend/Device Issue - Accuracy issue on XPU for Torch-xpu-ops UT |  | os: Windows, hw: BMG, bug.. | aten_ops |
| [1877](https://github.com/intel/torch-xpu-ops/issues/1877) | Torchbench model squeezenet1_1 | open | Silv3S | P0 | Backend/Device Issue - fail_accuracy on XPU for specific models |  | Accuracy, hw: BMG, hw: PV.. | aten_ops |
| [2022](https://github.com/intel/torch-xpu-ops/issues/2022) | [Windows] [CI] [UT] AssertionE | open | None | P2 | Failure - Tensor-likes are not close! |  | os: Windows, module: ut, .. | aten_ops |
| [2128](https://github.com/intel/torch-xpu-ops/issues/2128) | [2.9][BMG-Windows][Torchbench] | open | chuanqi129 | P0 | Backend/Device Issue - Exception Code 0xC0000005 indicates an access violation l |  |  | aten_ops |
| [2215](https://github.com/intel/torch-xpu-ops/issues/2215) | Find use case example for torc | open | dvrogozh | P2 | Others - Looking for use case example of torch-xpu-ops.lib in SYCL C++ extension |  |  | aten_ops |
| [2219](https://github.com/intel/torch-xpu-ops/issues/2219) | float8_e4m3fn precision overfl | open | CuiYifeng, yucai-intel | P2 | Dtype/Precision Issue - float8_e4m3fn precision overflow |  |  | aten_ops |
| [2251](https://github.com/intel/torch-xpu-ops/issues/2251) | [upstream_ut] test_fake_autoca | open | astachowiczhabana | P2 |  |  | duplicate, module: ut, sk.. | aten_ops |
| [2253](https://github.com/intel/torch-xpu-ops/issues/2253) | the supported dtypes are not a | open | daisyden | P2 |  |  | duplicate, skipped, ut_up.. | aten_ops |
| [2295](https://github.com/intel/torch-xpu-ops/issues/2295) | [upstream_ut][xpu][test]nn/tes | open | yucai-intel | P2 |  |  | module: inductor, skipped.. | inductor |
| [2331](https://github.com/intel/torch-xpu-ops/issues/2331) | [upstream_ut] AssertionError:  | open | hoshibara | P2 |  | [PR](https://github.com/pytorch/pytorch/pull/172314) | dependency component: one.. | inductor |
| [2376](https://github.com/intel/torch-xpu-ops/issues/2376) | [Bug Skip]: NotImplementedErro | open | CuiYifeng | P2 | Dtype/Precision Issue - "logaddexp_xpu" not implemented for 'Complex' dtype |  | module: ut, skipped | aten_ops |
| [2425](https://github.com/intel/torch-xpu-ops/issues/2425) | [upstream_ut]  RuntimeError: E | open | BBBela | P2 | Error - Nested tensor operation with non-nested tensor input |  | skipped, bug_fix_stage4 | aten_ops |
| [2439](https://github.com/intel/torch-xpu-ops/issues/2439) | [oneDNN] TestDecompXPU.test_qu | open | libohao1201 | P2 | DNNL/OneDNN Issue - Test failure related to oneDNN decomposition on XPU with flo |  | dependency component: one.. | aten_ops |
| [2444](https://github.com/intel/torch-xpu-ops/issues/2444) | [upstream_ut]  RuntimeError: U | open | Silv3S | P2 | Memory/Shared Memory Issue - UR backend returns out of resources error (40) indi |  | skipped | aten_ops |
| [2482](https://github.com/intel/torch-xpu-ops/issues/2482) | test_dtypes issue introduced b | open | daisyden | P2 | Dtype/Precision Issue - test_dtypes issue related to data types in conv_transpos |  | skipped | aten_ops |
| [2491](https://github.com/intel/torch-xpu-ops/issues/2491) | [upstream_ut]  AssertionError: | open | PatrykWilczewski | P2 | Failure - test assertion failed with False is not true |  | skipped, bug_fix_stage5 | aten_ops |
| [2494](https://github.com/intel/torch-xpu-ops/issues/2494) | [upstream_ut]  AssertionError: | open | PawelSwider2000 | P2 | Failure - Scalars are not equal assertion error in test |  | skipped | aten_ops |
| [2496](https://github.com/intel/torch-xpu-ops/issues/2496) | [upstream_ut]  Segmentation fa | open | astachowiczhabana | P0 |  |  | skipped | aten_ops |
| [2510](https://github.com/intel/torch-xpu-ops/issues/2510) | [upstream_ut]  RuntimeError: E | open | PawelSwider2000 | P2 | Error - tensor size exceeds int32_t maximum limit |  | skipped | aten_ops |
| [2512](https://github.com/intel/torch-xpu-ops/issues/2512) | [upstream_ut]  RuntimeError: _ | open | chunhuanMeng | P2 | Backend/Device Issue - _histc_xpu lacks deterministic implementation on XPU back |  | skipped | aten_ops |
| [2656](https://github.com/intel/torch-xpu-ops/issues/2656) | [release/2.10] models got fail | open | None | P0 | Backend/Device Issue - failure occurs specifically on BMG WSL2 XPU backend |  | os: Windows | aten_ops |
| [2676](https://github.com/intel/torch-xpu-ops/issues/2676) | Random failure in CI test | open | None | P2 | Others - Random failure with no traceback or specific error provided |  | skipped, random | unknown |
| [2680](https://github.com/intel/torch-xpu-ops/issues/2680) | XPU Autocast does not support  | open | CuiYifeng | P2 | Dtype/Precision Issue - XPU Autocast does not support fp32 dtypes |  |  | aten_ops |
| [2689](https://github.com/intel/torch-xpu-ops/issues/2689) | [LNL][Windows] AssertionError: | open | tadkrawiec | P2 | Failure - cur_target out of bounds assertion failed |  | os: Windows, module: ut | aten_ops |
| [2783](https://github.com/intel/torch-xpu-ops/issues/2783) | [Bug Skip]: Key "xpu" is missi | open | daisyden | P2 | Skip/No Test Exists - Key "xpu" is missing, indicating the test is skipped or no |  | module: ut, skipped | aten_ops |
| [2810](https://github.com/intel/torch-xpu-ops/issues/2810) | AssertionError: Object compari | open | daisyden | P2 |  |  | module: inductor, ut_upst.. | inductor |
| [2816](https://github.com/intel/torch-xpu-ops/issues/2816) | torch.logcumsumexp incorrectly | open | Silv3S | P2 | Dtype/Precision Issue - torch.logcumsumexp returns NaNs for complex64 input |  | Ready for merge, skipped,.. | aten_ops |
| [2817](https://github.com/intel/torch-xpu-ops/issues/2817) | Expected error message is diff | open | kdrozd-dev | P2 | Failure - mismatch between expected and actual error message |  | skipped, bug_fix_stage5 | aten_ops |
| [2862](https://github.com/intel/torch-xpu-ops/issues/2862) | accuracy issue with test_float | open | tszulist-hbn | P2 | Dtype/Precision Issue - accuracy issue with float8 operations |  |  | aten_ops |
| [2899](https://github.com/intel/torch-xpu-ops/issues/2899) | Update nan_to_num XPU stub to  | open | None | P2 | Backend/Device Issue - XPU stub uses incorrect type for nan_to_num parameters |  |  | aten_ops |
| [2914](https://github.com/intel/torch-xpu-ops/issues/2914) | Test case test/test_autograd.p | open | None | P2 | Failure - Tensor-likes are not close! |  |  | aten_ops |
| [2952](https://github.com/intel/torch-xpu-ops/issues/2952) | [release/2.11][wsl] timm_model | open | weishi-deng | P0 | Dtype/Precision Issue - bfloat16 accuracy failure in model training |  | Accuracy, hw: BMG | aten_ops |
| [2960](https://github.com/intel/torch-xpu-ops/issues/2960) | [release/2.11] timm_models_xci | open | None | P0 | Dtype/Precision Issue - float16 training accuracy test failure |  |  | aten_ops |
| [3014](https://github.com/intel/torch-xpu-ops/issues/3014) | [upstream_ut] AssertionError:  | open | None | P2 |  |  | bug_fix_stage5 | unknown |
| [3041](https://github.com/intel/torch-xpu-ops/issues/3041) | AssertionError: Expected len(f | open | Silv3S | P2 |  |  | ut_upstream | aten_ops |
| [3103](https://github.com/intel/torch-xpu-ops/issues/3103) | Tensor-likes are not equal for | open | BBBela | P2 | Backend/Device Issue - XPU tensor-like comparison failure in test |  | module: ut, skipped, rand.. | aten_ops |
| [3121](https://github.com/intel/torch-xpu-ops/issues/3121) | [Bug Skip]: CUDA specific UT t | open | None | P1 | Skip/No Test Exists - test is skipped or not applicable for XPU backend |  | skipped | aten_ops |
| [3156](https://github.com/intel/torch-xpu-ops/issues/3156) | AssertionError: 'Assertion cur | open | kdrozd-dev | P2 | Failure - Assertion cur_target >= 0 && cur_target < n_classes failed in cross entropy loss test |  |  | aten_ops |
| [3161](https://github.com/intel/torch-xpu-ops/issues/3161) | Exception: Tensor-likes are no | open | tadkrawiec | P2 |  |  | os: Windows | aten_ops |
| [3184](https://github.com/intel/torch-xpu-ops/issues/3184) | New failing UTs: test_cross_en | open | wpietka | P2 | Failure - test expects a specific condition to be true but it failed during execution. |  | module: ut, skipped | aten_ops |
| [3236](https://github.com/intel/torch-xpu-ops/issues/3236) | AssertionError: 'def [28 chars | open | jmamzax | P2 | Failure - mismatch in expected IR code for XPU backend operations |  | bug_fix_stage5 | aten_ops |
| [3238](https://github.com/intel/torch-xpu-ops/issues/3238) | The supported dtypes of _refs. | open | CuiYifeng | P2 |  |  | ut_upstream | aten_ops |
| [3243](https://github.com/intel/torch-xpu-ops/issues/3243) | AssertionError: False is not t | open | pponikox | P2 | Failure - assertion 'False is not true' failed in test |  | module: ut, skipped | aten_ops |

#### Flash Attention / Transformer Related (#flash-attention---transformer-related)

| ID | Title | Status | Owner | Priority | Root Cause | PR | Labels | Module |
|---|-------|--------|-------|---------|-----------|-----|--------|--------|
| [1749](https://github.com/intel/torch-xpu-ops/issues/1749) | transformers UT failure in XPU | open | LuFinch | P2 | Backend/Device Issue - XPU does not support backward or grad for SDPA operation |  |  | aten_ops |
| [2015](https://github.com/intel/torch-xpu-ops/issues/2015) | inf is returned by nn.Transfor | open | yucai-intel | P2 | Dtype/Precision Issue - inf returned using float16 in TransformerEncoderLayer on |  | skipped | aten_ops |
| [2186](https://github.com/intel/torch-xpu-ops/issues/2186) | AssertionError: Mul tiheadAtte | open | daisyden | P2 | Failure - Mul tiheadAttention does not support NestedTensor outside of its fast path |  | dependency component: one.. | aten_ops |
| [2200](https://github.com/intel/torch-xpu-ops/issues/2200) | support flash attention op on  | open | ElaineBao | P2 | Flash Attention/Specific Ops Issue - request to support flash attention op on XP |  | dependency component: one.. | aten_ops |
| [2232](https://github.com/intel/torch-xpu-ops/issues/2232) | sdpa backward kernel is requir | open | None | P2 | Memory/Shared Memory Issue - sdpa backward kernel needed to reduce memory usage |  |  | aten_ops |
| [2250](https://github.com/intel/torch-xpu-ops/issues/2250) | Found mismatch when comparing  | open | astachowiczhabana | P2 | Mismatch - Mismatch in output of aten.view.default between FakeTensor and concrete Tensors. |  | skipped, bug_fix_stage3 | aten_ops |
| [2255](https://github.com/intel/torch-xpu-ops/issues/2255) | [upstream_ut] RuntimeError: Lo | open | daisyden | P2 |  |  | skipped, ut_upstream | aten_ops |
| [2285](https://github.com/intel/torch-xpu-ops/issues/2285) | Support efficient attention | open | chunhuanMeng | P2 | Backend/Device Issue - aten::_efficient_attention_forward not supported on CPU b |  | skipped | aten_ops |
| [2390](https://github.com/intel/torch-xpu-ops/issues/2390) | SDPA in pytorch use different  | open | LuFinch | P2 | Backend/Device Issue - SDPA uses different backend compared with IPEX on XPU |  |  | aten_ops |
| [2436](https://github.com/intel/torch-xpu-ops/issues/2436) | [upstream_ut]  AttributeError: | open | daisyden | P1 | Error - 'NoneType' object has no attribute 'clone' due to missing object handling |  | skipped, dependency compo.. | aten_ops |
| [2442](https://github.com/intel/torch-xpu-ops/issues/2442) | [Bug Skip]: NotImplementedErro | open | daisyden, LuFinch | P2 | Flash Attention/Specific Ops Issue - aten::_flash_attention_forward not implemen |  | skipped, not_target | aten_ops |
| [2570](https://github.com/intel/torch-xpu-ops/issues/2570) | crash in sdpa. | open | LuFinch | P0 | Others - insufficient information to determine root cause |  | dependency component: one.. | aten_ops |
| [2675](https://github.com/intel/torch-xpu-ops/issues/2675) | [Bug Skip]: AttributeError: 'N | open | pponikox | P2 | Error - 'NoneType' object has no attribute 'clone' due to missing object reference |  | skipped, bug_fix_stage5 | aten_ops |
| [2704](https://github.com/intel/torch-xpu-ops/issues/2704) | Title: [upstream_ut]  Assertio | open | kdrozd-dev | P2 |  | [PR](https://github.com/pytorch/pytorch/pull/177636) | skipped, ut_upstream, bug.. | aten_ops |
| [2800](https://github.com/intel/torch-xpu-ops/issues/2800) | AttributeError: 'torch._C._Xpu | open | guangyey | P2 |  |  | dependency component: one.. | inductor |
| [2802](https://github.com/intel/torch-xpu-ops/issues/2802) | Three aten._scaled_dot_product | open | LuFinch | P2 |  |  | module: inductor, ut_upst.. | inductor |
| [2869](https://github.com/intel/torch-xpu-ops/issues/2869) | [Bug Skip]: New UT failure in  | open | None | P2 | Skip/No Test Exists - Test is marked as skipped or not executed |  | skipped_windows | aten_ops |
| [2997](https://github.com/intel/torch-xpu-ops/issues/2997) | AssertionError of test_linear_ | open | etaf | P2 |  |  | module: inductor, ut_upst.. | inductor |
| [3047](https://github.com/intel/torch-xpu-ops/issues/3047) | [Bug Skip]: [Regression]UT fai | open | None | P0 | Failure - Torch not compiled with CUDA enabled assertion error |  | skipped | unknown |
| [3093](https://github.com/intel/torch-xpu-ops/issues/3093) | XPU does not support NestedTen | open | None | P2 | Supported - XPU does not support NestedTensor for SDPA operations. |  | module: ut | aten_ops |
| [3114](https://github.com/intel/torch-xpu-ops/issues/3114) | [Bug Skip]: Failure skip on 20 | open | None | P2 | Skip/No Test Exists - test was skipped on 2026-3-21 |  | skipped, random | aten_ops |
| [3126](https://github.com/intel/torch-xpu-ops/issues/3126) | [upstream_ut]  Two NestedTenso | open | LuFinch | P2 |  |  | module: ut, skipped, ut_u.. | aten_ops |
| [3127](https://github.com/intel/torch-xpu-ops/issues/3127) | [upstream_ut]  AssertionError: | open | LuFinch | P2 |  |  | module: ut, skipped, ut_u.. | aten_ops |
| [3129](https://github.com/intel/torch-xpu-ops/issues/3129) | [upstream_ut]  AssertionError: | open | LuFinch | P2 |  |  | module: ut, skipped, ut_u.. | aten_ops |
| [3130](https://github.com/intel/torch-xpu-ops/issues/3130) | [upstream_ut]  AssertionError: | open | LuFinch | P2 |  |  | module: ut, skipped, ut_u.. | aten_ops |
| [3131](https://github.com/intel/torch-xpu-ops/issues/3131) | [upstream_ut]  NotImplementedE | open | chunhuanMeng | P2 |  |  | module: ut, skipped, ut_u.. | aten_ops |
| [3132](https://github.com/intel/torch-xpu-ops/issues/3132) | [upstream_ut]  transfomers tes | open | LuFinch | P2 |  |  | module: ut, skipped, ut_u.. | aten_ops |
| [3133](https://github.com/intel/torch-xpu-ops/issues/3133) | [upstream_ut]  RuntimeError: s | open | LuFinch | P2 |  |  | module: ut, skipped, ut_u.. | aten_ops |
| [3136](https://github.com/intel/torch-xpu-ops/issues/3136) | [upstream_ut]  AssertionError: | open | LuFinch | P2 |  |  | module: ut, skipped, ut_u.. | aten_ops |
| [3137](https://github.com/intel/torch-xpu-ops/issues/3137) | [upstream_ut]  RuntimeError: e | open | LuFinch | P2 |  |  | module: ut, skipped, ut_u.. | aten_ops |
| [3140](https://github.com/intel/torch-xpu-ops/issues/3140) | [upstream_ut]  RuntimeError: F | open | LuFinch | P2 |  |  | module: ut, skipped, ut_u.. | aten_ops |
| [3141](https://github.com/intel/torch-xpu-ops/issues/3141) | [upstream_ut]  RuntimeError: F | open | LuFinch | P2 |  |  | module: ut, skipped, ut_u.. | aten_ops |
| [3142](https://github.com/intel/torch-xpu-ops/issues/3142) | [upstream_ut]  RuntimeError: T | open | LuFinch | P2 |  |  | module: ut, skipped, ut_u.. | aten_ops |
| [3143](https://github.com/intel/torch-xpu-ops/issues/3143) | NotImplementedError: The opera | open | LuFinch | P2 |  |  | module: ut, skipped, ut_u.. | aten_ops |
| [3149](https://github.com/intel/torch-xpu-ops/issues/3149) | New failure in test_rms_norm_d | open | None | P2 | Failure - RuntimeError not raised as expected in test |  | module: ut, skipped | aten_ops |
| [3176](https://github.com/intel/torch-xpu-ops/issues/3176) | [Bug Skip]: ValueError: _scale | open | None | P2 | Backend/Device Issue - inputs are not on the same XPU device |  | skipped | aten_ops |
| [3178](https://github.com/intel/torch-xpu-ops/issues/3178) | New failed test cases 2026-03- | open | pponikox | P2 | Backend/Device Issue - XPU-specific test failure in vmap with SDPA operation |  | module: ut, skipped | aten_ops |
| [3195](https://github.com/intel/torch-xpu-ops/issues/3195) | test_sdpa_unbacked_no_dde_xpu  | open | None | P0 | Backend/Device Issue - test crashed on XPU backend execution |  | skipped, random | aten_ops |
| [3229](https://github.com/intel/torch-xpu-ops/issues/3229) | RuntimeError: No viable backen | open | None | P2 |  |  | skipped, ut_upstream | aten_ops |
| [3258](https://github.com/intel/torch-xpu-ops/issues/3258) | Error in op: torch.ops.aten._s | open | None | P2 | 5 - Flash Attention/Specific Ops Issue - Error in _scaled_dot_product_fused_atte |  |  | aten_ops |
| [3259](https://github.com/intel/torch-xpu-ops/issues/3259) | New failed test cases 2026-04- | open | SlawomirLaba | P2 | Backend/Device Issue - XPU device initialization or compatibility failure |  | skipped | aten_ops |

#### Inductor / Compilation Related (#inductor---compilation-related)

| ID | Title | Status | Owner | Priority | Root Cause | PR | Labels | Module |
|---|-------|--------|-------|---------|-----------|-----|--------|--------|
| [146](https://github.com/intel/torch-xpu-ops/issues/146) | Evaluate register spill in SYC | open | CuiYifeng, jianyizh, mengfei25 | P2 | Backend/Device Issue - register spill evaluation in SYCL kernel on XPU |  | enhancement | aten_ops |
| [1764](https://github.com/intel/torch-xpu-ops/issues/1764) | New kernels for concat | open | yucai-intel | P2 | Others - New kernels for concat, no specific error provided. |  | performance, kernel_optim.. | aten_ops |
| [2132](https://github.com/intel/torch-xpu-ops/issues/2132) | [2.9][BMG-Windows][Torch-xpu-o | open | pbielak | P2 |  |  | module: ut | aten_ops |
| [2140](https://github.com/intel/torch-xpu-ops/issues/2140) | Consider how to avoid copy in  | open | CuiYifeng | P2 | Memory/Shared Memory Issue - Avoiding copy in FFT kernels relates to memory hand |  | enhancement | aten_ops |
| [2196](https://github.com/intel/torch-xpu-ops/issues/2196) | Fix DistributionElementwiseKer | open | None | P2 | Memory/Shared Memory Issue - register spill in DistributionElementwiseKernelFunc |  | enhancement | aten_ops |
| [2467](https://github.com/intel/torch-xpu-ops/issues/2467) | Host may stuck when submit too | open | jianyizh | P2 | Backend/Device Issue - Host stuck due to excessive kernel submissions on XPU |  | dependency component: dri.. | aten_ops |
| [2645](https://github.com/intel/torch-xpu-ops/issues/2645) | [Bug Skip]: [regression] Runti | open | CuiYifeng | P0 | Backend/Device Issue - missing kernel for xpu in DispatchStub |  | skipped | aten_ops |
| [2888](https://github.com/intel/torch-xpu-ops/issues/2888) | torch._inductor.exc.InductorEr | open | Stonepia | P2 |  |  | module: inductor, ut_upst.. | inductor |
| [2922](https://github.com/intel/torch-xpu-ops/issues/2922) | [release/2.11] UT inductor Ass | open | tadkrawiec | P2 | Backend/Device Issue - pass_fds not supported on Windows |  | os: Windows | aten_ops |
| [2950](https://github.com/intel/torch-xpu-ops/issues/2950) | SYCL compilation flag -fsycl-i | open | BBBela | P2 | Inductor/Compilation Issue - SYCL compilation flag not working as expected for T |  |  | aten_ops |
| [2958](https://github.com/intel/torch-xpu-ops/issues/2958) | AssertionError of test_dtensor | open | daisyden | P2 |  |  | module: inductor, ut_upst.. | inductor |
| [3013](https://github.com/intel/torch-xpu-ops/issues/3013) | [upstream_ut] RuntimeError: Ke | open | None | P2 |  |  | bug_fix_stage5 | unknown |
| [3080](https://github.com/intel/torch-xpu-ops/issues/3080) | cudagraph tests blocked by fea | open | None | P2 | 10 - Feature Not Supported |  | module: ut | aten_ops |
| [3094](https://github.com/intel/torch-xpu-ops/issues/3094) | XPUGraph tree support | open | None | P2 |  |  | module: inductor, ut_upst.. | inductor |
| [3150](https://github.com/intel/torch-xpu-ops/issues/3150) | [Task] Align XPU kernel's impl | open | guangyey | P2 | device-specific backend discrepancy. |  |  | aten_ops |
| [3158](https://github.com/intel/torch-xpu-ops/issues/3158) | AttributeError: module 'triton | open | tadkrawiec | P2 |  |  |  | aten_ops |
| [3160](https://github.com/intel/torch-xpu-ops/issues/3160) | compiler not found (Windows) | open | kdrozd-dev | P2 |  |  | os: Windows | aten_ops |
| [3187](https://github.com/intel/torch-xpu-ops/issues/3187) | PyTorch XPU gpu_cpp_wrapper fa | open | CuiYifeng | P2 |  |  | ut_upstream | aten_ops |
| [3242](https://github.com/intel/torch-xpu-ops/issues/3242) | AssertionError: Torch not comp | open | None | P2 |  |  | module: ut, skipped | aten_ops |

#### Others (#others)

| ID | Title | Status | Owner | Priority | Root Cause | PR | Labels | Module |
|---|-------|--------|-------|---------|-----------|-----|--------|--------|
| [208](https://github.com/intel/torch-xpu-ops/issues/208) | Abstract utility functions use | open | CuiYifeng | P2 | Others - abstract utility functions in ATen operator implementation |  | enhancement, module: op i.. | aten_ops |
| [1171](https://github.com/intel/torch-xpu-ops/issues/1171) | LNL Windows got unexpected err | open | xuhancn, chunhuanMeng | P2 | Backend/Device Issue - unexpected error on XPU for LNL Windows |  | client, os: Windows, hw :.. | aten_ops |
| [1324](https://github.com/intel/torch-xpu-ops/issues/1324) | [Win] UR Error when OOM and br | open | Stonepia | P2 | Memory/Shared Memory Issue - Out of memory (OOM) leading to tensor context break |  | client, os: Windows, modu.. | aten_ops |
| [1510](https://github.com/intel/torch-xpu-ops/issues/1510) | Some test cases will be hang o | open | Stonepia, mengfei25 | P2 | Backend/Device Issue - test cases hang on BMG Ubuntu related to XPU backend |  | bug, client, os: Ubuntu, .. | aten_ops |
| [1519](https://github.com/intel/torch-xpu-ops/issues/1519) | [PVC][PT2.7][ABI=1][Torch-xpu- | open | None | P2 | Backend/Device Issue - coredump related to XPU operations in Torch-xpu-ops UT |  | dependency component: dri.. | aten_ops |
| [1574](https://github.com/intel/torch-xpu-ops/issues/1574) | The operator 'aten::_grouped_m | open | Stonepia | P2 | Backend/Device Issue - aten::_grouped_mm not implemented for XPU device |  | module: ao | AO |
| [1587](https://github.com/intel/torch-xpu-ops/issues/1587) | Keep track on the latest CUDA  | open | CuiYifeng, yucai-intel | P2 | Skip/No Test Exists - no test or error traceback provided |  | kernel_optimization | aten_ops |
| [1594](https://github.com/intel/torch-xpu-ops/issues/1594) | Keep track on the building war | open | CuiYifeng, chunhuanMeng | P0 | Others - building warning tracking issue |  | module: build | aten_ops |
| [1645](https://github.com/intel/torch-xpu-ops/issues/1645) | [For Comparison] Save referenc | open | mengfei25 | P2 | Skip/No Test Exists - no test or error information provided |  | module: infra | inductor |
| [1649](https://github.com/intel/torch-xpu-ops/issues/1649) | [cpp extension] Provide a clea | open | dvrogozh | P2 | Backend/Device Issue - inconsistent oneAPI versions cause XPU backend errors |  | dependency component: one.. | aten_ops |
| [1678](https://github.com/intel/torch-xpu-ops/issues/1678) | missing op support for `model. | open | None | P0 | Memory/Shared Memory Issue - missing op support for `model.share_memory()` indic |  | bug_fix_stage3 | aten_ops |
| [1689](https://github.com/intel/torch-xpu-ops/issues/1689) | [For op Perf Comparison] Save  | open | None | P2 | Skip/No Test Exists - no test or error information provided |  | module: infra | aten_ops |
| [1722](https://github.com/intel/torch-xpu-ops/issues/1722) | Ask an API to query GPU type(i | open | guangyey | P2 | Mismatch - Request for an API to query GPU type (iGPU/dGPU) is missing or not properly implemented. |  | dependency component: one.. | aten_ops |
| [1729](https://github.com/intel/torch-xpu-ops/issues/1729) | Validation Check List | open | chuanqi129 | P2 | Skip/No Test Exists - no test or error information provided |  | module: infra | aten_ops |
| [1745](https://github.com/intel/torch-xpu-ops/issues/1745) | torch.xpu.empty_cache() introd | open | guangyey | P2 | Memory/Shared Memory Issue - torch.xpu.empty_cache() introduces memory leak |  | module: core | aten_ops |
| [1784](https://github.com/intel/torch-xpu-ops/issues/1784) | [Performance] Torch XPU Profil | open | jfedorov | P2 | 11 - Timeout/Performance Issue - Torch XPU Profiler is not reliable |  | module: profiler | profiling |
| [1856](https://github.com/intel/torch-xpu-ops/issues/1856) | channel last aten::hardswish_  | open | chunhuanMeng | P2 | Memory/Shared Memory Issue - extra copy caused by channel last aten::hardswish_ |  | performance, hw: BMG | aten_ops |
| [1900](https://github.com/intel/torch-xpu-ops/issues/1900) | implement torch.linalg.qr xpu  | open | pbielak | P2 | Backend/Device Issue - XPU backend implementation missing for torch.linalg.qr |  | module: op impl, bug_fix_.. | aten_ops |
| [1901](https://github.com/intel/torch-xpu-ops/issues/1901) | implement torch.linalg.svd xpu | open | CuiYifeng | P2 | Backend/Device Issue - XPU backend for torch.linalg.svd not implemented |  | module: op impl | aten_ops |
| [1902](https://github.com/intel/torch-xpu-ops/issues/1902) | implement torch.linalg.pinv xp | open | mwiktor-intel | P2 | Backend/Device Issue - XPU backend for torch.linalg.pinv is not implemented |  | module: op impl, bug_fix_.. | aten_ops |
| [1936](https://github.com/intel/torch-xpu-ops/issues/1936) | implement torch.linalg.cholesk | open | mwiktor-intel | P2 | Backend/Device Issue - XPU backend for torch.linalg.cholesky is not implemented |  | module: op impl, bug_fix_.. | aten_ops |
| [1963](https://github.com/intel/torch-xpu-ops/issues/1963) | [upstream_ut] MetadataMismatch | open | pbielak | P2 |  | [PR](https://github.com/pytorch/pytorch/pull/178277, https://github.com/pytorch/pytorch/pull/175965) | module: ut, ut_upstream | aten_ops |
| [1986](https://github.com/intel/torch-xpu-ops/issues/1986) | torch.xpu._sleep is missing, | open | guangyey | P2 | Mismatch - torch.xpu._sleep is not implemented or available in the current setup. |  | dependency component: one.. | aten_ops |
| [2063](https://github.com/intel/torch-xpu-ops/issues/2063) | Avoid using out-of-date term | open | CuiYifeng | P2 | Skip/No Test Exists - no test or error traceback provided |  | enhancement | aten_ops |
| [2086](https://github.com/intel/torch-xpu-ops/issues/2086) | nd_item::barrier has been depr | open | dvrogozh | P2 | Backend/Device Issue - nd_item::barrier is deprecated on XPU backend. |  | enhancement | aten_ops |
| [2089](https://github.com/intel/torch-xpu-ops/issues/2089) | need an implementation that wo | open | guangyey | P2 | Backend/Device Issue - torch.xpu.is_available() initializes GPU context unintent |  | dependency component: dri.. | aten_ops |
| [2098](https://github.com/intel/torch-xpu-ops/issues/2098) | Upstream XPU functions in yaml | open | guangyey | P2 | Backend/Device Issue - XPU functions in yaml related to upstream backend issues |  | enhancement | aten_ops |
| [2127](https://github.com/intel/torch-xpu-ops/issues/2127) | Path Coverage enhancement | open | CuiYifeng | P2 | Skip/No Test Exists - no test or error information provided |  | enhancement | aten_ops |
| [2142](https://github.com/intel/torch-xpu-ops/issues/2142) | XPU max_memory_allocated have  | open | guangyey | P2 | Memory/Shared Memory Issue - XPU and CUDA show different memory allocation outpu |  | bug | aten_ops |
| [2157](https://github.com/intel/torch-xpu-ops/issues/2157) | BMG d2h copy is very slow comp | open | jianyizh, mengfei25 | P2 | Timeout/Performance Issue - BMG d2h copy is significantly slower compared to PVC |  | performance, dependency c.. | aten_ops |
| [2195](https://github.com/intel/torch-xpu-ops/issues/2195) | Tools in pti should get 100% f | open | aostrowski-hbn | P2 | Backend/Device Issue - functionality not working on BMG for PyTorch profiling |  | module: profiler | profiling |
| [2199](https://github.com/intel/torch-xpu-ops/issues/2199) | Fix reduction and norm registe | open | None | P2 | Memory/Shared Memory Issue - register spill in reduction and norm operations |  | enhancement | aten_ops |
| [2217](https://github.com/intel/torch-xpu-ops/issues/2217) | AO Performance issue track | open | Stonepia | P2 | Timeout/Performance Issue - AO Performance issue track |  | module: ao | AO |
| [2261](https://github.com/intel/torch-xpu-ops/issues/2261) | [xpu][profiler] Run with fork  | open | moksiuc | P2 | Backend/Device Issue - XPU profiler warning during fork process execution |  | dependency component: one.. | profiling |
| [2263](https://github.com/intel/torch-xpu-ops/issues/2263) | [xpu][bug] XPU Trace event end | open | PawelSwider2000 | P2 | Backend/Device Issue - XPU trace event timing discrepancy |  | module: profiler | profiling |
| [2287](https://github.com/intel/torch-xpu-ops/issues/2287) | [upstream_ut] test_python_ref  | open | yucai-intel | P2 |  |  | module: ut, ut_upstream | aten_ops |
| [2329](https://github.com/intel/torch-xpu-ops/issues/2329) | [upstream_ut] feature missing: | open | etaf | P2 |  |  | duplicate, dependency com.. | inductor |
| [2333](https://github.com/intel/torch-xpu-ops/issues/2333) | [Don't merge] Collect the new  | open | None | P2 | Skip/No Test Exists - test is empty or not applicable |  | module: infra | aten_ops |
| [2349](https://github.com/intel/torch-xpu-ops/issues/2349) | [oneAPI][backward compatibilit | open | riverliuintel | P2 | Backend/Device Issue - missing library version for XPU backend compatibility |  |  | aten_ops |
| [2392](https://github.com/intel/torch-xpu-ops/issues/2392) | [Bug Skip]: torch.OutOfMemoryE | open | xuhancn | P2 | Memory/Shared Memory Issue - XPU out of memory error occurred |  | skipped_windows | aten_ops |
| [2400](https://github.com/intel/torch-xpu-ops/issues/2400) | [ut_upstream] tf32_on_and_off( | open | chunhuanMeng | P2 | Backend/Device Issue - XPU support required for tf32_on_and_off() test |  |  | aten_ops |
| [2412](https://github.com/intel/torch-xpu-ops/issues/2412) | Some NestedTensor missing XPU  | open | yucai-intel | P2 | Backend/Device Issue - XPU support missing for NestedTensor operations |  | module: ut | aten_ops |
| [2440](https://github.com/intel/torch-xpu-ops/issues/2440) | [For UT failures classify] Sav | open | None | P2 | Skip/No Test Exists - no test or error details provided |  | module: infra | inductor |
| [2447](https://github.com/intel/torch-xpu-ops/issues/2447) | test_share_memory_xpu failure | open | None | P2 | Memory/Shared Memory Issue - failure in test_share_memory_xpu related to shared  |  | skipped | aten_ops |
| [2463](https://github.com/intel/torch-xpu-ops/issues/2463) | [Bug Skip]: OSError: SYCL runt | open | xuhancn | P2 | Backend/Device Issue - SYCL runtime not detected on XPU |  | skipped_windows | aten_ops |
| [2465](https://github.com/intel/torch-xpu-ops/issues/2465) | [windows] ut hang | open | tadkrawiec | P2 | Timeout/Performance Issue - UT hang indicates a timeout or performance bottlenec |  | os: Windows | aten_ops |
| [2469](https://github.com/intel/torch-xpu-ops/issues/2469) | test_matmul_cuda.py gaps | open | CuiYifeng, guangyey | P2 | Skip/No Test Exists - test_matmul_cuda.py gaps indicate missing or skipped tests |  |  | aten_ops |
| [2471](https://github.com/intel/torch-xpu-ops/issues/2471) | test_cuda.py gaps | open | guangyey | P2 | Skip/No Test Exists - test_cuda.py gaps indicate missing or skipped tests. |  |  | aten_ops |
| [2479](https://github.com/intel/torch-xpu-ops/issues/2479) | [Bug] torch.rand output differ | open | Stonepia, CuiYifeng | P2 | Backend/Device Issue - different output on BMG and PVC devices |  |  | aten_ops |
| [2513](https://github.com/intel/torch-xpu-ops/issues/2513) | [upstream_ut]  RuntimeError: _ | open | gplutop7 | P2 | Memory/Shared Memory Issue - _share_fd_ is not available on XPU, indicating a sh |  | skipped | aten_ops |
| [2518](https://github.com/intel/torch-xpu-ops/issues/2518) | [upstream_ut]  TypeError: Crea | open | astachowiczhabana | P2 |  |  | skipped | aten_ops |
| [2519](https://github.com/intel/torch-xpu-ops/issues/2519) | [upstream_ut]  TypeError: map2 | open | Silv3S | P2 | Backend/Device Issue - map2_ is only implemented on CPU tensors, indicating lack |  | skipped | aten_ops |
| [2537](https://github.com/intel/torch-xpu-ops/issues/2537) | Title: [upstream_ut]  Failed:  | open | PatrykWilczewski | P2 | Others - Test expects failure but passed unexpectedly, no specific error trace provided. |  | skipped, port_from_skipli.. | aten_ops |
| [2560](https://github.com/intel/torch-xpu-ops/issues/2560) | [UT] "RuntimeError: iter.devic | open | CuiYifeng | P2 | Backend/Device Issue - XPU device check failure in test |  | bug | aten_ops |
| [2562](https://github.com/intel/torch-xpu-ops/issues/2562) | Warning as Error | open | chunhuanMeng | P2 | Others - warning treated as error but no traceback or specific error provided |  |  | aten_ops |
| [2660](https://github.com/intel/torch-xpu-ops/issues/2660) | [release/2.10][Windows][BMG] N | open | tadkrawiec | P2 | Others - insufficient information to determine root cause |  | os: Windows, hw: BMG, mod.. | aten_ops |
| [2662](https://github.com/intel/torch-xpu-ops/issues/2662) | [release/2.10][Windows][BMG] N | open | tadkrawiec, kdrozd-dev | P2 | Backend/Device Issue - XPU related failure in test cases on Windows with BMG |  | os: Windows, hw: BMG, mod.. | aten_ops |
| [2712](https://github.com/intel/torch-xpu-ops/issues/2712) | [upstream_ut]  RuntimeError: C | open | tszulist-hbn | P2 |  |  | skipped, ut_upstream | aten_ops |
| [2744](https://github.com/intel/torch-xpu-ops/issues/2744) | [Bug Skip]: extended test fail | open | pbielak | P2 | Skip/No Test Exists - test was skipped due to changes in tolerance values causin |  | skipped | aten_ops |
| [2766](https://github.com/intel/torch-xpu-ops/issues/2766) | MaxPool2d - investigate memory | open | BBBela | P2 | Memory/Shared Memory Issue - investigate memory layout performance for MaxPool2d |  |  | aten_ops |
| [2769](https://github.com/intel/torch-xpu-ops/issues/2769) | [oneDNN] New failed test cases | open | LuFinch | P2 | DNNL/OneDNN Issue - failed test cases related to oneDNN with 3.11 compared to 3. |  | hw: PVC, dependency compo.. | aten_ops |
| [2795](https://github.com/intel/torch-xpu-ops/issues/2795) | Histc raises error with intege | open | CuiYifeng | P2 | Dtype/Precision Issue - integer input causes error with deterministic algorithm  |  |  | aten_ops |
| [2797](https://github.com/intel/torch-xpu-ops/issues/2797) | Copy error is not raise on tes | open | None | P2 | Others - Copy error not raised in test_dlpack.py test case |  |  | aten_ops |
| [2798](https://github.com/intel/torch-xpu-ops/issues/2798) | Test case  test/test_dlpack.py | open | None | P2 |  |  | ut_upstream | aten_ops |
| [2815](https://github.com/intel/torch-xpu-ops/issues/2815) | RuntimeError: output with shap | open | PawelSwider2000 | P2 | Error - output shape mismatch during broadcasting |  | skipped, bug_fix_stage5 | aten_ops |
| [2845](https://github.com/intel/torch-xpu-ops/issues/2845) | [Bug Skip]:[UT] [Windows] fail | open | None | P2 | Skip/No Test Exists - test was skipped or does not exist |  | skipped_windows | aten_ops |
| [2852](https://github.com/intel/torch-xpu-ops/issues/2852) | [Bug Skip]: New UT failures in | open | None | P2 | Skip/No Test Exists - test was skipped or not present |  | skipped_windows | aten_ops |
| [2858](https://github.com/intel/torch-xpu-ops/issues/2858) | [Bug Skip]: test_xpu new failu | open | None | P2 | Skip/No Test Exists - test_xpu new failures marked as skipped or no test availab |  | skipped_windows | unknown |
| [2879](https://github.com/intel/torch-xpu-ops/issues/2879) | RuntimeError: _share_fd_: only | open | Silv3S | P2 | Backend/Device Issue - _share_fd_ is not available on XPU device |  | bug_fix_stage5 | aten_ops |
| [2907](https://github.com/intel/torch-xpu-ops/issues/2907) | [release/2.11] Models performa | open | xuhancn | P0 | Timeout/Performance Issue - models performance regression in testcases |  |  | aten_ops |
| [2912](https://github.com/intel/torch-xpu-ops/issues/2912) | [release/2.11] UT extended 220 | open | None | P2 | Others - insufficient information to determine root cause |  | os: Windows, hw: BMG | aten_ops |
| [2918](https://github.com/intel/torch-xpu-ops/issues/2918) | [XPU][upstream_ut][COW] Skip n | open | gplutop7 | P2 |  | [PR](https://github.com/pytorch/pytorch/pull/174670) | skipped, bug_fix_stage3, .. | aten_ops |
| [2919](https://github.com/intel/torch-xpu-ops/issues/2919) | [XPU][upstream_ut][COW] Fix ma | open | gplutop7 | P2 |  |  | bug_fix_stage3, ut_upstre.. | aten_ops |
| [2920](https://github.com/intel/torch-xpu-ops/issues/2920) | Failing Test Cases in Nightly  | open | Silv3S | P2 | Others - insufficient information to determine root cause |  | skipped | aten_ops |
| [2930](https://github.com/intel/torch-xpu-ops/issues/2930) | [release/2.11] UT skip test_bi | open | None | P2 | Skip/No Test Exists - test is skipped due to RuntimeError |  |  | aten_ops |
| [2942](https://github.com/intel/torch-xpu-ops/issues/2942) | [Windows] Unit tests got Fatal | open | xuhancn, Stonepia | P2 | Backend/Device Issue - Fatal Python error on Windows unit tests related to XPU b |  | os: Windows | aten_ops |
| [2948](https://github.com/intel/torch-xpu-ops/issues/2948) | [AO] Benchmark enabling on XPU | open | None | P2 | Backend/Device Issue - XPU benchmark enabling issue |  | module: ao, bug_fix_stage.. | AO |
| [2993](https://github.com/intel/torch-xpu-ops/issues/2993) | [Bug Skip]: Unexpected success | open | gplutop7 | P2 | Skip/No Test Exists - test unexpectedly succeeded and should have been skipped |  | skipped | aten_ops |
| [2999](https://github.com/intel/torch-xpu-ops/issues/2999) | KeyError: 'eager_numerics.use_ | open | daisyden | P2 |  |  | module: inductor, ut_upst.. | inductor |
| [3000](https://github.com/intel/torch-xpu-ops/issues/3000) | [Bug Skip]: RuntimeError: _sha | open | gplutop7 | P2 | Backend/Device Issue - _share_fd_ is not available on XPU device |  | skipped | aten_ops |
| [3004](https://github.com/intel/torch-xpu-ops/issues/3004) | TypeError: _xpu_recordMemoryHi | open | guangyey | P2 |  |  | module: inductor, ut_upst.. | inductor |
| [3011](https://github.com/intel/torch-xpu-ops/issues/3011) | [upstream_ut] torch.OutOfMemor | open | None | P2 |  |  | bug_fix_stage5 | aten_ops |
| [3024](https://github.com/intel/torch-xpu-ops/issues/3024) | Enable clang-tidy checks | open | Silv3S | P2 | Others - clang-tidy checks enablement is a code quality/linter configuration task, not a runtime or functional issue. |  | bug_fix_stage5 | aten_ops |
| [3030](https://github.com/intel/torch-xpu-ops/issues/3030) | [Bug Skip] test/test_modules.p | open | gplutop7 | P2 | Skip/No Test Exists - test was skipped due to failure with no detailed error pro |  | skipped | aten_ops |
| [3033](https://github.com/intel/torch-xpu-ops/issues/3033) | [Bug Skip]: Softmax tolerance | open | chunhuanMeng | P2 | Skip/No Test Exists - test is skipped due to Softmax tolerance issue |  | skipped, random | aten_ops |
| [3048](https://github.com/intel/torch-xpu-ops/issues/3048) | Profiler result is not correct | open | aostrowski-hbn | P2 | Backend/Device Issue - Profiler result discrepancy on B70 device. |  | module: profiler | profiling |
| [3060](https://github.com/intel/torch-xpu-ops/issues/3060) | Implement torch._scaled_groupe | open | Stonepia, liangan1 | P2 | Backend/Device Issue - Implementation required for XPU backend |  | module: quant | low_precision |
| [3074](https://github.com/intel/torch-xpu-ops/issues/3074) | [Bug Skip] test_dlpack_exchang | open | AKloniecki | P2 | Skip/No Test Exists - test is skipped expecting current_work_stream is not null |  | skipped | aten_ops |
| [3077](https://github.com/intel/torch-xpu-ops/issues/3077) | [Bug Skip] test_dlpack.py::Tes | open | AKloniecki | P2 |  |  | ut_upstream | aten_ops |
| [3083](https://github.com/intel/torch-xpu-ops/issues/3083) | [Bug Skip]: Random failures 20 | open | None | P2 | Skip/No Test Exists - test is marked to be skipped with no valid test to execute |  | skipped, random | unknown |
| [3084](https://github.com/intel/torch-xpu-ops/issues/3084) | torch.library.register_autocas | open | None | P2 | Backend/Device Issue - torch.library.register_autocast does not support XPU devi |  | module: ut | aten_ops |
| [3086](https://github.com/intel/torch-xpu-ops/issues/3086) | nvml support blocks some test  | open | None | P2 | Backend/Device Issue - nvml support blocking test cases on XPU |  | module: ut | aten_ops |
| [3095](https://github.com/intel/torch-xpu-ops/issues/3095) | cutlass support blocks some un | open | None | P2 |  |  | module: inductor, ut_upst.. | inductor |
| [3096](https://github.com/intel/torch-xpu-ops/issues/3096) | VISIBLE_DEVICE support | open | None | P2 | Backend/Device Issue - VISIBLE_DEVICE support is related to device visibility an |  | module: ut | aten_ops |
| [3106](https://github.com/intel/torch-xpu-ops/issues/3106) | Worker crashes when running Te | open | BBBela | P0 |  |  | module: ut, skipped, bug_.. | aten_ops |
| [3157](https://github.com/intel/torch-xpu-ops/issues/3157) | XPU out of memory. Tried to al | open | kdrozd-dev | P2 |  |  |  | aten_ops |
| [3170](https://github.com/intel/torch-xpu-ops/issues/3170) | Unskip test_bmm_windows_error_ | open | jenniew | P2 |  |  | skipped, ut_upstream | aten_ops |
| [3180](https://github.com/intel/torch-xpu-ops/issues/3180) | [E2E] Timm/Torchbench models g | open | None | P0 | Backend/Device Issue - eager_two_runs_differ on ARC XPU backend |  |  | aten_ops |
| [3189](https://github.com/intel/torch-xpu-ops/issues/3189) | Task Tracker | open | guangyey | P2 | Skip/No Test Exists - no test or error details provided |  |  | aten_ops |
| [3196](https://github.com/intel/torch-xpu-ops/issues/3196) | vitals is not supported, the c | open | libohao1201 | P2 | 10 - vitals feature is not supported, cases should be disabled |  | skipped | aten_ops |
| [3206](https://github.com/intel/torch-xpu-ops/issues/3206) | [Bug Skip]: [new failures]Runt | open | tszulist-hbn | P2 | Backend/Device Issue - Tensors are on different devices (xpu:0 vs cpu) |  | skipped | aten_ops |
| [3216](https://github.com/intel/torch-xpu-ops/issues/3216) | [OPs] Some ops of XPU have non | open | CuiYifeng | P2 | Backend/Device Issue - XPU ops show non-determinism and inconsistency compared t |  |  | aten_ops |
| [3227](https://github.com/intel/torch-xpu-ops/issues/3227) | torch xpu event has ~0.1ms lat | open | guangyey | P2 | Timeout/Performance Issue - high latency in XPU event (~0.1ms) indicates a perfo |  |  | aten_ops |
| [3247](https://github.com/intel/torch-xpu-ops/issues/3247) | NotImplementedError: "dot_xpu_ | open | Silv3S | P2 |  |  | ut_upstream | aten_ops |

#### PT2E (#pt2e)

| ID | Title | Status | Owner | Priority | Root Cause | PR | Labels | Module |
|---|-------|--------|-------|---------|-----------|-----|--------|--------|
| [1762](https://github.com/intel/torch-xpu-ops/issues/1762) | Add an ocloc AOT target compil | open | chunhuanMeng | P2 | compilation-related task or issue. |  | module: build | aten_ops |
| [1969](https://github.com/intel/torch-xpu-ops/issues/1969) | torch._dynamo.exc.InternalTorc | open | guangyey | P2 | Error - cannot create weak reference to 'torch.Event' object |  | module: ut | aten_ops |
| [1970](https://github.com/intel/torch-xpu-ops/issues/1970) | torch._dynamo.exc.BackendCompi | open | None | P2 | Backend/Device Issue - CUDA not available on the system |  | module: ut | aten_ops |
| [2472](https://github.com/intel/torch-xpu-ops/issues/2472) | [upstream_ut]  NotImplementedE | open | Silv3S | P2 |  |  | wontfix, module: ut, skip.. | aten_ops |
| [2554](https://github.com/intel/torch-xpu-ops/issues/2554) | [upstream_ut]  AssertionError: | open | daisyden | P2 |  |  | module: inductor, skipped | inductor |
| [2609](https://github.com/intel/torch-xpu-ops/issues/2609) | [upstream_ut]  torch._inductor | open | daisyden | P2 |  | [PR](https://github.com/pytorch/pytorch/pull/171154) | module: inductor, skipped.. | inductor |
| [2698](https://github.com/intel/torch-xpu-ops/issues/2698) | Title: [upstream_ut]  RuntimeE | open | chunhuanMeng, LuFinch | P2 |  |  | module: inductor, skipped.. | inductor |
| [2715](https://github.com/intel/torch-xpu-ops/issues/2715) | [upstream_ut]  torch._dynamo.e | open | CuiYifeng | P2 |  |  | skipped, ut_upstream | aten_ops |
| [2767](https://github.com/intel/torch-xpu-ops/issues/2767) | [UT] test_control_flow_xpu.py  | open | PatrykWilczewski | P1 | Failure - test_control_flow_xpu.py got AssertionError |  | module: ut, skipped, bug_.. | aten_ops |
| [2806](https://github.com/intel/torch-xpu-ops/issues/2806) | CompiledAOTI need XPU support | open | daisyden | P2 |  |  | module: inductor, ut_upst.. | inductor |
| [2873](https://github.com/intel/torch-xpu-ops/issues/2873) | [Bug Skip]: test_repos.py cont | open | PawelSwider2000 | P2 | Skip/No Test Exists - test contains failed ops and is skipped |  | skipped | aten_ops |
| [3231](https://github.com/intel/torch-xpu-ops/issues/3231) | Dynamo failed to run FX node w | open | None | P2 | Inductor/Compilation Issue - Dynamo failed to compile FX node with fake tensors  |  | module: ut, skipped | aten_ops |

#### Sparse Operations Related (#sparse-operations-related)

| ID | Title | Status | Owner | Priority | Root Cause | PR | Labels | Module |
|---|-------|--------|-------|---------|-----------|-----|--------|--------|
| [1962](https://github.com/intel/torch-xpu-ops/issues/1962) | [upstream_ut] segfault with te | open | jenniew, mengfei25 | P0 | Backend/Device Issue - segfault related to XPU device operation in test |  | dependency component: dri.. | aten_ops |
| [2595](https://github.com/intel/torch-xpu-ops/issues/2595) | [Bug Skip]: Random crashed cas | open | None | P0 | Skip/No Test Exists - Test was skipped due to random crashed cases. |  | skipped, random | aten_ops |
| [2663](https://github.com/intel/torch-xpu-ops/issues/2663) | test_sparse_semi_structured.py | open | None | P2 |  |  | module: ut, ut_upstream | aten_ops |
| [2729](https://github.com/intel/torch-xpu-ops/issues/2729) | [Bug Skip]: Random failures 20 | open | Silv3S | P2 | Skip/No Test Exists - test is marked as skipped due to random failures |  | skipped, bug_fix_stage5, .. | unknown |
| [2751](https://github.com/intel/torch-xpu-ops/issues/2751) | [Bug Skip]: Random failures 20 | open | None | P2 | Skip/No Test Exists - test was skipped due to random failure标记 |  | skipped, random | unknown |
| [2777](https://github.com/intel/torch-xpu-ops/issues/2777) | [Bug Skip]: Random failures 20 | open | AKloniecki | P2 | Skip/No Test Exists - test is marked as a skip with no detailed error traceback  |  | skipped, random | unknown |
| [2801](https://github.com/intel/torch-xpu-ops/issues/2801) | to_dense() for Sparse CSR back | open | jenniew | P2 | Error - source tensor shape mismatch during to_dense() for Sparse CSR backend |  |  | aten_ops |
| [2965](https://github.com/intel/torch-xpu-ops/issues/2965) | [Bug Skip]: Random failures 20 | open | None | P2 | Skip/No Test Exists - test is marked as a skip with no valid test implementation |  | hw: PVC, skipped, random | unknown |
| [3081](https://github.com/intel/torch-xpu-ops/issues/3081) | Sparse CSR gemm-like ops have  | open | None | P2 | Supported - Sparse CSR gemm-like operations are not supported yet. |  | module: ut | aten_ops |
| [3165](https://github.com/intel/torch-xpu-ops/issues/3165) | test_sparse_csr_xpu.py::TestSp | open | None | P2 |  |  | skipped, ut_upstream | aten_ops |
| [3167](https://github.com/intel/torch-xpu-ops/issues/3167) | NotImplementedError: Could not | open | tszulist-hbn | P1 |  |  | skipped, ut_upstream | aten_ops |
| [3168](https://github.com/intel/torch-xpu-ops/issues/3168) | NotImplementedError: Could not | open | jenniew | P1 |  |  | skipped, ut_upstream | aten_ops |
| [3169](https://github.com/intel/torch-xpu-ops/issues/3169) | NotImplementedError: Could not | open | jkosnox | P2 |  |  | skipped, ut_upstream | aten_ops |
| [3175](https://github.com/intel/torch-xpu-ops/issues/3175) | [Bug Skip]: ValueError: sample | open | None | P2 | Backend/Device Issue - inputs are not on the same XPU device |  | skipped | aten_ops |

#### TorchAO (#torchao)

| ID | Title | Status | Owner | Priority | Root Cause | PR | Labels | Module |
|---|-------|--------|-------|---------|-----------|-----|--------|--------|
| [1833](https://github.com/intel/torch-xpu-ops/issues/1833) | [PT2E] INT8 accuracy is lower  | open | chunhuanMeng | P2 | Dtype/Precision Issue - INT8 accuracy lower than FP32 due to precision reduction |  | Accuracy, module: quant, .. | low_precision |
| [1912](https://github.com/intel/torch-xpu-ops/issues/1912) | Implement the torch.ops.aten._ | open | liangan1 | P2 | Backend/Device Issue - Implementation required for XPU dequantization of CUDA in |  | dependency component: one.. | aten_ops |
| [1996](https://github.com/intel/torch-xpu-ops/issues/1996) | [TorchAO]  Memory Efficient Op | open | arlesniak | P2 | Memory/Shared Memory Issue - related to memory efficient optimizers in TorchAO |  | module: ao | AO |
| [2006](https://github.com/intel/torch-xpu-ops/issues/2006) | work-item/workgroup issue in s | open | BartoszKokoszko | P2 | Backend/Device Issue - work-group size exceeds device limitations on XPU |  | module: ut, skipped | aten_ops |
| [2201](https://github.com/intel/torch-xpu-ops/issues/2201) | [TorchAO][BMG] When using page | open | Stonepia | P2 | Failure - assert vr is not None error encountered |  | module: ao | AO |
| [2202](https://github.com/intel/torch-xpu-ops/issues/2202) | [TorchAO][BMG] The RTN perform | open | Stonepia | P0 | Timeout/Performance Issue - RTN performance regression in next-token latency for |  | performance, regression, .. | AO |
| [2207](https://github.com/intel/torch-xpu-ops/issues/2207) | Enable FP8/MXFP8 Ops with requ | open | CuiYifeng | P2 | Backend/Device Issue - FP8/MXFP8 Ops related to XPU and CUDA alignment |  | dtype: float8 | aten_ops |
| [2220](https://github.com/intel/torch-xpu-ops/issues/2220) | test/test_sparse_csr.py::TestS | open | None | P2 | Backend/Device Issue - unknown SPIR-V extension 'SPV_INTEL_subgroup_matrix_multi |  | duplicate, module: depend.. | aten_ops |
| [2309](https://github.com/intel/torch-xpu-ops/issues/2309) | unsupported ops with PYTORCH_E | open | daisyden | P2 |  |  | wontfix, module: op impl,.. | aten_ops |
| [2323](https://github.com/intel/torch-xpu-ops/issues/2323) | [TorchAO] MOE training enablin | open | riverliuintel | P2 | Backend/Device Issue - MOE training not enabled on XPU |  | module: ao | AO |
| [2324](https://github.com/intel/torch-xpu-ops/issues/2324) | [TorchAO] FP8 conv support | open | Stonepia | P2 | Supported - FP8 conv is not supported yet in TorchAO |  | module: ao | AO |
| [2325](https://github.com/intel/torch-xpu-ops/issues/2325) | [TorchAO] Float8 training supp | open | arlesniak, riverliuintel | P2 | Supported - Float8 training is not supported on XPU. |  | module: ao | AO |
| [2326](https://github.com/intel/torch-xpu-ops/issues/2326) | [TorchAO] MX training  native  | open | riverliuintel | P2 | Backend/Device Issue - Training on XPU with TorchAO and native PyTorch is not fu |  | module: ao | AO |
| [2327](https://github.com/intel/torch-xpu-ops/issues/2327) | [TorchAO] benchmark enabling o | open | None | P2 | Backend/Device Issue - XPU benchmark enabling issue in TorchAO |  |  | aten_ops |
| [2358](https://github.com/intel/torch-xpu-ops/issues/2358) | test/test_view_ops.py::TestOld | open | Silv3S | P2 |  |  | Ready for merge, ut_upstr.. | aten_ops |
| [2434](https://github.com/intel/torch-xpu-ops/issues/2434) | [Bug Skip]: New failures 2025- | open | AKloniecki | P2 | Skip/No Test Exists - test is marked as a bug skip or not implemented properly |  | module: ut, skipped, bug_.. | aten_ops |
| [2508](https://github.com/intel/torch-xpu-ops/issues/2508) | TypedStorage / TypedTensors de | open | Silv3S | P1 |  | [PR](https://github.com/intel/torch-xpu-ops/pull/3260) | wontfix, skipped | aten_ops |
| [2596](https://github.com/intel/torch-xpu-ops/issues/2596) | [TorchAO][BMG]Observed accurac | open | None | P0 | Dtype/Precision Issue - accuracy fluctuations due to INT4 quantization methods ( |  | module: ao | AO |
| [2597](https://github.com/intel/torch-xpu-ops/issues/2597) | [TorchAO][BMG] INT4 GPTQ shows | open | xiaowangintel | P2 | Timeout/Performance Issue - INT4 GPTQ shows worse performance compared with RTN  |  | module: ao | AO |
| [2598](https://github.com/intel/torch-xpu-ops/issues/2598) | [TorchAO][BMG]The first token  | open | Stonepia | P2 | Timeout/Performance Issue - First token latency drops significantly with change  |  | module: ao | AO |
| [2605](https://github.com/intel/torch-xpu-ops/issues/2605) | [int4][inductor] Add freezing  | open | None | P2 | Inductor/Compilation Issue - Adding freezing pattern for fusing int4 mm kernel i |  |  | aten_ops |
| [2620](https://github.com/intel/torch-xpu-ops/issues/2620) | [upstream_ut]  AssertionError: | open | daisyden | P2 |  |  | module: inductor, skipped.. | inductor |
| [2707](https://github.com/intel/torch-xpu-ops/issues/2707) | [TorchAO][BMG] INT4 GPTQ faile | open | xiaowangintel | P2 | Mismatch - INT4 GPTQ failed due to TorchAO API change. |  | module: ao | AO |
| [2722](https://github.com/intel/torch-xpu-ops/issues/2722) | [Bug Skip]: NotImplementedErro | open | Silv3S | P2 | Backend/Device Issue - aten::flip not implemented for QuantizedXPU backend |  | skipped, bug_fix_stage5 | aten_ops |
| [2734](https://github.com/intel/torch-xpu-ops/issues/2734) | [TorchAO][BMG] INT4 RTN Flex-a | open | Stonepia, hoshibara | P2 | Timeout/Performance Issue - 50% performance drop in INT4 RTN Flex-attention with | [PR](https://github.com/pytorch/pytorch/pull/160480, https://github.com/pytorch/pytorch/pull/172316) | module: ao | AO |
| [2823](https://github.com/intel/torch-xpu-ops/issues/2823) | [TorchAO][BMG] Llama-3.2-1B-In | open | xiaowangintel, lchen2331 | P2 | Timeout/Performance Issue - 20% performance drop in next token generation with D |  | module: ao | AO |
| [2946](https://github.com/intel/torch-xpu-ops/issues/2946) | [Bug Skip]: Random failures 20 | open | None | P2 | Skip/No Test Exists - test is marked to be skipped with no valid test implementa |  | skipped, random | unknown |
| [3076](https://github.com/intel/torch-xpu-ops/issues/3076) | [TorchAO][BMG] Llama-3.2-1B-In | open | None | P2 | DNNL/OneDNN Issue - Performance drop with oneDNN 3.11.1 in Dynamic INT8 for Llam |  |  | aten_ops |
| [3088](https://github.com/intel/torch-xpu-ops/issues/3088) | [TorchAO][BMG] INT4 RTN Flex-a | open | Stonepia | P2 | Timeout/Performance Issue - INT4 RTN Flex-attention experienced a 5% performance |  | module: ao | AO |
| [3089](https://github.com/intel/torch-xpu-ops/issues/3089) | AssertionError: Torch not comp | open | jmamzax | P2 |  |  | bug_fix_stage5 | unknown |
| [3124](https://github.com/intel/torch-xpu-ops/issues/3124) | [TorchAO][Bug] ImportError: Re | open | None | P2 | Backend/Device Issue - Requires mslk >= 1.0.0 for XPU support with int4 quantiza |  |  | aten_ops |
| [3166](https://github.com/intel/torch-xpu-ops/issues/3166) | test_consistency_SparseCSR fai | open | yucai-intel | P2 |  |  | skipped, ut_upstream | aten_ops |


---

## 6. Duplicated Issues {#6-duplicated-issues}

Issues that share test cases with other issues.

| ID | Title | Owner | Reporter | Duplicated With | Priority | Root Cause | PR | Labels |
|---|-------|-------|----------|-----------------|---------|-----------|-----|--------|
| [1893](https://github.com/intel/torch-xpu-ops/issues/1893) | [upstream_ut] oneDNN accuracy  | chunhuanMeng | daisyden | 1951 | P2 |  |  | skipped, ut_upstream |
| [1951](https://github.com/intel/torch-xpu-ops/issues/1951) | Functionality issues in TestCo | AKloniecki | daisyden | 1893 | P2 |  |  | module: ut, skipped,.. |
| [1973](https://github.com/intel/torch-xpu-ops/issues/1973) | AssertionError: Scalars or Ten | gplutop7 | mengfei25 | 2837,2840 | P2 | Failure - Tensor values not equal or close in test_depthwise_conv_64bit_indexing_xpu |  | hw: PVC, module: ut,.. |
| [2006](https://github.com/intel/torch-xpu-ops/issues/2006) | work-item/workgroup issue in s | BartoszKokoszko | daisyden | 2257 | P2 | Backend/Device Issue - work-group size exceeds device limitations on XPU |  | module: ut, skipped |
| [2015](https://github.com/intel/torch-xpu-ops/issues/2015) | inf is returned by nn.Transfor | yucai-intel | daisyden | 2186,2529 | P2 | Dtype/Precision Issue - inf returned using float16 in TransformerEncoderLayer on |  | skipped |
| [2186](https://github.com/intel/torch-xpu-ops/issues/2186) | AssertionError: Mul tiheadAtte | daisyden | daisyden | 2015 | P2 | Failure - Mul tiheadAttention does not support NestedTensor outside of its fast path |  | dependency component.. |
| [2220](https://github.com/intel/torch-xpu-ops/issues/2220) | test/test_sparse_csr.py::TestS | None | wincent8 | 2246 | P2 | Backend/Device Issue - unknown SPIR-V extension 'SPV_INTEL_subgroup_matrix_multi |  | duplicate, module: d.. |
| [2230](https://github.com/intel/torch-xpu-ops/issues/2230) | test_sparse_csr.py::TestSparse | None | wincent8 | 2246,3175,3176 | P1 | Backend/Device Issue - inputs are not on the same GPU device |  | skipped |
| [2235](https://github.com/intel/torch-xpu-ops/issues/2235) | test/test_sparse_csr.py::TestS | None | wincent8 | 3047 | P2 | Backend/Device Issue - unexpected warning from non-optimal triton kernel paramet |  | skipped |
| [2238](https://github.com/intel/torch-xpu-ops/issues/2238) | Exception: Tensor-likes are no | BBBela | zxd1997066 | 3105 | P2 |  | [PR](https://github.com/intel/torch-xpu-ops/pull/2886) | skipped, bug_fix_sta.. |
| [2244](https://github.com/intel/torch-xpu-ops/issues/2244) | test/test_sparse_csr.py::TestS | jenniew | wincent8 | 3177 | P2 | Error - empty_sparse_compressed expects non-block layout but received SparseBsr |  | module: ut, skipped |
| [2246](https://github.com/intel/torch-xpu-ops/issues/2246) | torch/sparse/_triton_ops*.py n | None | wincent8 | 2220,2230 | P1 | Backend/Device Issue - Torch not compiled with CUDA enabled for Intel GPU suppor |  | skipped |
| [2253](https://github.com/intel/torch-xpu-ops/issues/2253) | the supported dtypes are not a | daisyden | daisyden | 2482 | P2 |  |  | duplicate, skipped, .. |
| [2257](https://github.com/intel/torch-xpu-ops/issues/2257) | Accuracy failures in test/xpu/ | pbielak | zxd1997066 | 2006 | P1 |  | [PR](https://github.com/intel/torch-xpu-ops/pull/2808) | skipped, bug_fix_sta.. |
| [2270](https://github.com/intel/torch-xpu-ops/issues/2270) | Backend Compatibility Error in | LuFinch | libohao1201 | 2442 | P2 | Flash Attention/Specific Ops Issue - test involves aten._flash_attention_forward |  | module: ut, skipped |
| [2285](https://github.com/intel/torch-xpu-ops/issues/2285) | Support efficient attention | chunhuanMeng | daisyden | 2358 | P2 | Backend/Device Issue - aten::_efficient_attention_forward not supported on CPU b |  | skipped |
| [2358](https://github.com/intel/torch-xpu-ops/issues/2358) | test/test_view_ops.py::TestOld | Silv3S | wincent8 | 2285 | P2 |  |  | Ready for merge, ut_.. |
| [2436](https://github.com/intel/torch-xpu-ops/issues/2436) | [upstream_ut]  AttributeError: | daisyden | daisyden | 2675 | P1 | Error - 'NoneType' object has no attribute 'clone' due to missing object handling |  | skipped, dependency .. |
| [2442](https://github.com/intel/torch-xpu-ops/issues/2442) | [Bug Skip]: NotImplementedErro | daisyden, LuFinch | CuiYifeng | 2270 | P2 | Flash Attention/Specific Ops Issue - aten::_flash_attention_forward not implemen |  | skipped, not_target |
| [2482](https://github.com/intel/torch-xpu-ops/issues/2482) | test_dtypes issue introduced b | daisyden | daisyden | 2253 | P2 | Dtype/Precision Issue - test_dtypes issue related to data types in conv_transpos |  | skipped |
| [2529](https://github.com/intel/torch-xpu-ops/issues/2529) | [upstream_ut]  AssertionError: | Silv3S | daisyden | 2015,3136 | P2 | Failure - test assertion failed with False is not true |  | skipped, port_from_s.. |
| [2530](https://github.com/intel/torch-xpu-ops/issues/2530) | Title: [upstream_ut]  Assertio | PatrykWilczewski | daisyden | 2817 | P2 | Failure - RuntimeError not raised as expected in test |  | skipped, bug_fix_sta.. |
| [2611](https://github.com/intel/torch-xpu-ops/issues/2611) | [upstream_ut]  AssertionError: | daisyden | daisyden | 2613 | P2 |  |  | dependency component.. |
| [2613](https://github.com/intel/torch-xpu-ops/issues/2613) | [upstream_ut]  AssertionError: | daisyden | daisyden | 2611 | P2 |  |  | dependency component.. |
| [2618](https://github.com/intel/torch-xpu-ops/issues/2618) | [Bug Skip]: [regression] Asser | jmamzax | kaileiyx | 3089 | P0 |  | [PR](https://github.com/numpy/numpy/pull/22525) | skipped, bug_fix_sta.. |
| [2675](https://github.com/intel/torch-xpu-ops/issues/2675) | [Bug Skip]: AttributeError: 'N | pponikox | kaileiyx | 2436 | P2 | Error - 'NoneType' object has no attribute 'clone' due to missing object reference |  | skipped, bug_fix_sta.. |
| [2817](https://github.com/intel/torch-xpu-ops/issues/2817) | Expected error message is diff | kdrozd-dev | Silv3S | 2530 | P2 | Failure - mismatch between expected and actual error message |  | skipped, bug_fix_sta.. |
| [2837](https://github.com/intel/torch-xpu-ops/issues/2837) | Accuracy issue for Muon optimi | Silv3S | kdrozd-dev | 1973 | P2 | Failure - Tensor-likes not close in Muon optimizer test |  | skipped, bug_fix_sta.. |
| [2840](https://github.com/intel/torch-xpu-ops/issues/2840) | Accuracy issue with 64 bit ind | SlawomirLaba, Silv3S | kdrozd-dev | 1973 | P2 | Dtype/Precision Issue - Tensor comparison failed due to small absolute differenc |  | skipped, bug_fix_sta.. |
| [2869](https://github.com/intel/torch-xpu-ops/issues/2869) | [Bug Skip]: New UT failure in  | None | RUIJIEZHONG66166 | 3160 | P2 | Skip/No Test Exists - Test is marked as skipped or not executed |  | skipped_windows |
| [2966](https://github.com/intel/torch-xpu-ops/issues/2966) | [Bug Skip]: [Regression]2026-3 | jmamzax | kaileiyx | 3114 | P0 | Timeout/Performance Issue - Example code timed out during test execution. |  | skipped, bug_fix_sta.. |
| [3047](https://github.com/intel/torch-xpu-ops/issues/3047) | [Bug Skip]: [Regression]UT fai | None | kaileiyx | 2235 | P0 | Failure - Torch not compiled with CUDA enabled assertion error |  | skipped |
| [3089](https://github.com/intel/torch-xpu-ops/issues/3089) | AssertionError: Torch not comp | jmamzax | jmamzax | 2618 | P2 |  |  | bug_fix_stage5 |
| [3105](https://github.com/intel/torch-xpu-ops/issues/3105) | Wrong results from oneDNN conv | BBBela | BBBela | 2238 | P2 | DNNL/OneDNN Issue - wrong results from oneDNN conv2d kernels due to data pointer |  | hw: PVC, module: ut,.. |
| [3114](https://github.com/intel/torch-xpu-ops/issues/3114) | [Bug Skip]: Failure skip on 20 | None | guangyey | 2966 | P2 | Skip/No Test Exists - test was skipped on 2026-3-21 |  | skipped, random |
| [3136](https://github.com/intel/torch-xpu-ops/issues/3136) | [upstream_ut]  AssertionError: | LuFinch | daisyden | 2529 | P2 |  |  | module: ut, skipped,.. |
| [3156](https://github.com/intel/torch-xpu-ops/issues/3156) | AssertionError: 'Assertion cur | kdrozd-dev | kdrozd-dev | 3184 | P2 | Failure - Assertion cur_target >= 0 && cur_target < n_classes failed in cross entropy loss test |  |  |
| [3160](https://github.com/intel/torch-xpu-ops/issues/3160) | compiler not found (Windows) | kdrozd-dev | kdrozd-dev | 2869 | P2 |  |  | os: Windows |
| [3175](https://github.com/intel/torch-xpu-ops/issues/3175) | [Bug Skip]: ValueError: sample | None | CuiYifeng | 2230 | P2 | Backend/Device Issue - inputs are not on the same XPU device |  | skipped |
| [3176](https://github.com/intel/torch-xpu-ops/issues/3176) | [Bug Skip]: ValueError: _scale | None | CuiYifeng | 2230 | P2 | Backend/Device Issue - inputs are not on the same XPU device |  | skipped |
| [3177](https://github.com/intel/torch-xpu-ops/issues/3177) | Accuracy gap of BF16/FP16 test | jenniew | CuiYifeng | 2244 | P2 | Dtype/Precision Issue - Accuracy gap in BF16/FP16 test |  | skipped |
| [3184](https://github.com/intel/torch-xpu-ops/issues/3184) | New failing UTs: test_cross_en | wpietka | BBBela | 3156 | P2 | Failure - test expects a specific condition to be true but it failed during execution. |  | module: ut, skipped |

---

## 7. Issues with Dependency {#7-issues-with-dependency}

Issues that have dependencies on other components.

| ID | Title | Owner | Priority | Root Cause | Dependency | Category | PR | Labels |
|---|-------|-------|---------|-----------|------------|----------|-----|--------|
| [1547](https://github.com/intel/torch-xpu-ops/issues/1547) | [distributed] NotImplemen | Chao1Han | P2 | Backend/Device Issue - XPU device does not support the 'symm_mem::fused_matmul_r | oneAPI | Distributed |  | module: distributed,.. |
| [1551](https://github.com/intel/torch-xpu-ops/issues/1551) | [distributed] NotImplemen | Chao1Han | P2 | Backend/Device Issue - XPU device does not support the 'symm_mem::fused_scaled_m | oneAPI | Distributed |  | module: distributed,.. |
| [1556](https://github.com/intel/torch-xpu-ops/issues/1556) | [distributed] NotImplemen | pkourdis | P2 | Distributed/Gloo Issue - Operator lacks a sharding strategy in distributed conte | oneDNN | Distributed |  | module: distributed,.. |
| [1912](https://github.com/intel/torch-xpu-ops/issues/1912) | Implement the torch.ops.a | liangan1 | P2 | Backend/Device Issue - Implementation required for XPU dequantization of CUDA in | oneDNN | TorchAO |  | dependency component.. |
| [2089](https://github.com/intel/torch-xpu-ops/issues/2089) | need an implementation th | guangyey | P2 | Backend/Device Issue - torch.xpu.is_available() initializes GPU context unintent | driver | Others |  | dependency component.. |
