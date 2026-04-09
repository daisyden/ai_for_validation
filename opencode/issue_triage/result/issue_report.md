# Torch XPU Ops Issue Report

**Repository:** [intel/torch-xpu-ops](https://github.com/intel/torch-xpu-ops)

**Issues Extracted:** April 5, 2026 (from GitHub issue data: `torch_xpu_ops_issues.json`)

**CI Data Sources:**
- Torch-XPU-OPS Nightly CI: XML files from `Inductor-XPU-UT-Data-*` + `Inductor-wheel-nightly-LTS2-XPU-E2E-Data-*` (commit: `a2d516a58c64f18b76880f3a77efbc02885d65af`)
- Stock PyTorch XPU CI: XML files from `test-default-*-linux.idc.xpu_*.zip` (run IDs: 69741866812, 69741866834, 69741866862, 69741866911, etc.)

Generated: 2026-04-08 23:01:09

---

## Index

1. [Summary](#1-summary)
2. [Statistics](#2-statistics)
   - [By Test Module](#by-test-module)
   - [By Module](#by-module)
   - [By Dependency](#by-dependency)
   - [By Action TBD](#by-action-tbd)
   - [By Category](#by-category-stats)
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
| Action Required | 349 |
| No Assignee | 3 |
| Duplicated Issues | 42 |
| With Dependency | 5 |
| Others | 18 |
| **Total** | 417 |

---

## 2. Statistics {#2-statistics}

### By Test Module {#by-test-module}

| Test Module | Count |
|-------------|-------|
| ut | 371 |
| e2e | 39 |
| build | 7 |

### By Module {#by-module}

| Module | Count |
|--------|-------|
| aten_ops | 303 |
| distributed | 39 |
| inductor | 30 |
| AO | 21 |
| unknown | 16 |
| profiling | 5 |
| low_precision | 3 |

### By Dependency {#by-dependency}

| Dependency | Count |
|------------|-------|
| oneAPI | 13 |
| driver | 10 |
| oneDNN | 8 |
| Triton | 5 |
| oneCCL | 1 |

### By Action TBD {#by-action-tbd}

| Action TBD | Count |
|------------|-------|
| Need reproduce steps (Only for bugs or performance issue) | 167 |
| Needs PyTorch Repo Changes (upstream) | 123 |
| Need more information - error logs and reproduction steps | 73 |
| Close fixed issue | 17 |
| Revisit the PR as case failed | 4 |
| add to skiplist | 4 |
| Verify the issue | 3 |

### By Category (Statistics) {#by-category-stats}

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

### By Priority {#by-priority}

| Priority | Count |
|----------|-------|
| P0 | 50 |
| P1 | 18 |
| P2 | 349 |

---

## 3. New Submitted Issues (Past Week) {#3-new-submitted-issues-past-week}

Issues created in the past 7 days (as of 2026-04-08).

| ID | Title | Status | Owner | Priority | Reason | Category | Root Cause | Labels | Module | Test Module |
|---|-------|--------|-------|---------|--------|----------|-----------|--------|--------|-------------|
| [3259](https://github.com/intel/torch-xpu-ops/issues/3259) | New failed test cases 2026-04-02 | open | SlawomirLaba | P2 | UT issue with few failures | Flash Attention / Transformer Related | Backend/Device Issue - XPU device initialization or compatibility failure | skipped | aten_ops | ut |
| [3258](https://github.com/intel/torch-xpu-ops/issues/3258) | Error in op: torch.ops.aten._scaled_dot_ | open | None | P2 | UT issue with few failures | Flash Attention / Transformer Related | 5 - Flash Attention/Specific Ops Issue - Error in _scaled_dot_product_fused_atte |  | aten_ops | ut |
| [3257](https://github.com/intel/torch-xpu-ops/issues/3257) | [Linux][E2E][Regression] Huggingface tes | open | None | P0 | Impacts real model/application | Others | Backend/Device Issue - Huggingface test models failed to run on XPU, indicating  |  | aten_ops | e2e |
| [3255](https://github.com/intel/torch-xpu-ops/issues/3255) | [Linux][PT2E][Regression] Some performan | open | None | P0 | Regression - passed before but failed now | TorchAO | Timeout/Performance Issue - performance tests failed due to regression in execut |  | aten_ops | e2e |

---

## 4. Action Required {#4-action-required}

### Reporter Actions {#reporter-actions}

#### Information Required {#information-required}

| ID | Title | Owner | Owner Transferred | Priority | Category | Root Cause | PR | Test Module |
|---|-------|-------|-------------------|---------|----------|-----------|-----|-------------|
| [1059](https://github.com/intel/torch-xpu-ops/issues/1059) | SYCL RT: Using recommended shortcut | CuiYifeng, jianyizh | fengyuan14 | P2 | Distributed | Backend/Device Issue - related to SYCL RT and kernel-specific max work group siz |  | ut |
| [1165](https://github.com/intel/torch-xpu-ops/issues/1165) | [CI] Add a test of PyTorch XPU with | RUIJIEZHONG66166 | dvrogozh | P0 | Flash Attention / Transformer Related | Skip/No Test Exists - No test was implemented or executed. |  | build |
| [1510](https://github.com/intel/torch-xpu-ops/issues/1510) | Some test cases will be hang on BMG | Stonepia, mengfei25 | mengfei25 | P2 | Others | Backend/Device Issue - test cases hang on BMG Ubuntu related to XPU backend |  | ut |
| [1587](https://github.com/intel/torch-xpu-ops/issues/1587) | Keep track on the latest CUDA op im | CuiYifeng, yucai-intel | toyxu | P2 | Others | Skip/No Test Exists - no test or error traceback provided |  | ut |
| [1594](https://github.com/intel/torch-xpu-ops/issues/1594) | Keep track on the building warning | CuiYifeng, chunhuanMeng | toyxu | P0 | Others | Others - building warning tracking issue |  | ut |
| [1645](https://github.com/intel/torch-xpu-ops/issues/1645) | [For Comparison] Save reference com | mengfei25 | mengfei25 | P2 | Others | Skip/No Test Exists - no test or error information provided |  | ut |
| [1678](https://github.com/intel/torch-xpu-ops/issues/1678) | missing op support for `model.share | None | jafraustro | P0 | Others | Memory/Shared Memory Issue - missing op support for `model.share_memory()` indic |  | ut |
| [1689](https://github.com/intel/torch-xpu-ops/issues/1689) | [For op Perf Comparison] Save refer | None | RUIJIEZHONG66166 | P2 | Others | Skip/No Test Exists - no test or error information provided |  | ut |
| [1722](https://github.com/intel/torch-xpu-ops/issues/1722) | Ask an API to query GPU type(iGPU/d | guangyey | xuhancn | P2 | Others | Mismatch - Request for an API to query GPU type (iGPU/dGPU) is missing or not properly implemented. |  | ut |
| [1729](https://github.com/intel/torch-xpu-ops/issues/1729) | Validation Check List | chuanqi129 | EikanWang | P2 | Others | Skip/No Test Exists - no test or error information provided |  | ut |
| [1745](https://github.com/intel/torch-xpu-ops/issues/1745) | torch.xpu.empty_cache() introduces  | guangyey | songhappy | P2 | Others | Memory/Shared Memory Issue - torch.xpu.empty_cache() introduces memory leak |  | ut |
| [1762](https://github.com/intel/torch-xpu-ops/issues/1762) | Add an ocloc AOT target compilation | chunhuanMeng | jingxu10 | P2 | PT2E | compilation-related task or issue. |  | ut |
| [1764](https://github.com/intel/torch-xpu-ops/issues/1764) | New kernels for concat | yucai-intel | jianyizh | P2 | Inductor / Compilation Related | Others - New kernels for concat, no specific error provided. |  | ut |
| [1856](https://github.com/intel/torch-xpu-ops/issues/1856) | channel last aten::hardswish_ will  | chunhuanMeng | jianyizh | P2 | Others | Memory/Shared Memory Issue - extra copy caused by channel last aten::hardswish_ |  | ut |
| [1962](https://github.com/intel/torch-xpu-ops/issues/1962) | [upstream_ut] segfault with test_fa | jenniew, mengfei25 | daisyden | P0 | Sparse Operations Related | Backend/Device Issue - segfault related to XPU device operation in test |  | ut |
| [1986](https://github.com/intel/torch-xpu-ops/issues/1986) | torch.xpu._sleep is missing, | guangyey | githubsgi | P2 | Others | Mismatch - torch.xpu._sleep is not implemented or available in the current setup. |  | ut |
| [1996](https://github.com/intel/torch-xpu-ops/issues/1996) | [TorchAO]  Memory Efficient Optimiz | arlesniak | liangan1 | P2 | TorchAO | Memory/Shared Memory Issue - related to memory efficient optimizers in TorchAO |  | ut |
| [2113](https://github.com/intel/torch-xpu-ops/issues/2113) | Update example for Distributed Data | songhappy | luoyu-intel | P2 | Distributed | Distributed/Gloo Issue - related to Distributed Data Parallel update example |  | ut |
| [2142](https://github.com/intel/torch-xpu-ops/issues/2142) | XPU max_memory_allocated have diffe | guangyey | jiqing-feng | P2 | Others | Memory/Shared Memory Issue - XPU and CUDA show different memory allocation outpu |  | ut |
| [2163](https://github.com/intel/torch-xpu-ops/issues/2163) | 3 distributed UT cases need to be s | githubsgi | libohao1201 | P2 | Distributed | Distributed/Gloo Issue - distributed UT cases need support from sac_estimator.py |  | ut |
| [2195](https://github.com/intel/torch-xpu-ops/issues/2195) | Tools in pti should get 100% functi | aostrowski-hbn | jianyizh | P2 | Others | Backend/Device Issue - functionality not working on BMG for PyTorch profiling |  | ut |
| [2200](https://github.com/intel/torch-xpu-ops/issues/2200) | support flash attention op on XPU d | ElaineBao | Zjq9409 | P2 | Flash Attention / Transformer Related | Flash Attention/Specific Ops Issue - request to support flash attention op on XP |  | ut |
| [2215](https://github.com/intel/torch-xpu-ops/issues/2215) | Find use case example for torch-xpu | dvrogozh | dvrogozh | P2 | Dtype / Precision Related | Others - Looking for use case example of torch-xpu-ops.lib in SYCL C++ extension |  | ut |
| [2229](https://github.com/intel/torch-xpu-ops/issues/2229) | test/test_sparse_csr.py::TestSparse | jenniew | wincent8 | P2 | Distributed | Backend/Device Issue - device mismatch between crow_indices and col_indices on X |  | ut |
| [2232](https://github.com/intel/torch-xpu-ops/issues/2232) | sdpa backward kernel is required to | None | xin3he | P2 | Flash Attention / Transformer Related | Memory/Shared Memory Issue - sdpa backward kernel needed to reduce memory usage |  | ut |
| [2250](https://github.com/intel/torch-xpu-ops/issues/2250) | Found mismatch when comparing the o | astachowiczhabana | daisyden | P2 | Flash Attention / Transformer Related | Mismatch - Mismatch in output of aten.view.default between FakeTensor and concrete Tensors. |  | ut |
| [2261](https://github.com/intel/torch-xpu-ops/issues/2261) | [xpu][profiler] Run with fork proce | moksiuc | chuanqi129 | P2 | Others | Backend/Device Issue - XPU profiler warning during fork process execution |  | ut |
| [2323](https://github.com/intel/torch-xpu-ops/issues/2323) | [TorchAO] MOE training enabling on  | riverliuintel | liangan1 | P2 | TorchAO | Backend/Device Issue - MOE training not enabled on XPU |  | ut |
| [2324](https://github.com/intel/torch-xpu-ops/issues/2324) | [TorchAO] FP8 conv support | Stonepia | liangan1 | P2 | TorchAO | Supported - FP8 conv is not supported yet in TorchAO |  | ut |
| [2325](https://github.com/intel/torch-xpu-ops/issues/2325) | [TorchAO] Float8 training support o | arlesniak, riverliuintel | liangan1 | P2 | TorchAO | Supported - Float8 training is not supported on XPU. |  | ut |
| [2326](https://github.com/intel/torch-xpu-ops/issues/2326) | [TorchAO] MX training  native PyTor | riverliuintel | liangan1 | P2 | TorchAO | Backend/Device Issue - Training on XPU with TorchAO and native PyTorch is not fu |  | ut |
| [2327](https://github.com/intel/torch-xpu-ops/issues/2327) | [TorchAO] benchmark enabling on XPU | None | liangan1 | P2 | TorchAO | Backend/Device Issue - XPU benchmark enabling issue in TorchAO |  | ut |
| [2333](https://github.com/intel/torch-xpu-ops/issues/2333) | [Don't merge] Collect the new passe | None | mengfei25 | P2 | Others | Skip/No Test Exists - test is empty or not applicable |  | ut |
| [2349](https://github.com/intel/torch-xpu-ops/issues/2349) | [oneAPI][backward compatibility] li | riverliuintel | dvrogozh | P2 | Others | Backend/Device Issue - missing library version for XPU backend compatibility |  | ut |
| [2390](https://github.com/intel/torch-xpu-ops/issues/2390) | SDPA in pytorch use different backe | LuFinch | jiqing-feng | P2 | Flash Attention / Transformer Related | Backend/Device Issue - SDPA uses different backend compared with IPEX on XPU |  | ut |
| [2400](https://github.com/intel/torch-xpu-ops/issues/2400) | [ut_upstream] tf32_on_and_off() nee | chunhuanMeng | daisyden | P2 | Others | Backend/Device Issue - XPU support required for tf32_on_and_off() test |  | ut |
| [2412](https://github.com/intel/torch-xpu-ops/issues/2412) | Some NestedTensor missing XPU suppo | yucai-intel | daisyden | P2 | Others | Backend/Device Issue - XPU support missing for NestedTensor operations |  | ut |
| [2465](https://github.com/intel/torch-xpu-ops/issues/2465) | [windows] ut hang | tadkrawiec | bjarzemb | P2 | Others | Timeout/Performance Issue - UT hang indicates a timeout or performance bottlenec |  | ut |
| [2467](https://github.com/intel/torch-xpu-ops/issues/2467) | Host may stuck when submit too many | jianyizh | jianyizh | P2 | Inductor / Compilation Related | Backend/Device Issue - Host stuck due to excessive kernel submissions on XPU |  | ut |
| [2469](https://github.com/intel/torch-xpu-ops/issues/2469) | test_matmul_cuda.py gaps | CuiYifeng, guangyey | daisyden | P2 | Others | Skip/No Test Exists - test_matmul_cuda.py gaps indicate missing or skipped tests |  | ut |
| [2471](https://github.com/intel/torch-xpu-ops/issues/2471) | test_cuda.py gaps | guangyey | daisyden | P2 | Others | Skip/No Test Exists - test_cuda.py gaps indicate missing or skipped tests. |  | ut |
| [2605](https://github.com/intel/torch-xpu-ops/issues/2605) | [int4][inductor] Add freezing patte | None | liangan1 | P2 | TorchAO | Inductor/Compilation Issue - Adding freezing pattern for fusing int4 mm kernel i |  | ut |
| [2640](https://github.com/intel/torch-xpu-ops/issues/2640) | random issue test_vjpvjp_index_redu | wpietka | daisyden | P2 | Distributed | Others - incomplete traceback and insufficient information to determine root cause |  | ut |
| [2649](https://github.com/intel/torch-xpu-ops/issues/2649) | [distributed][pipelining] test_sche | syedshahbaaz | zxd1997066 | P2 | Distributed | Distributed/Gloo Issue - test_schedule_multiproc.py is hanging during distribute |  | ut |
| [2700](https://github.com/intel/torch-xpu-ops/issues/2700) | [distributed] Hang issues with test | syedshahbaaz | madhumitha0102 | P2 | Distributed | Distributed/Gloo Issue - Hang issues with test_distributed_spawn.py suggest a pr |  | ut |
| [2816](https://github.com/intel/torch-xpu-ops/issues/2816) | torch.logcumsumexp incorrectly retu | Silv3S | Silv3S | P2 | Dtype / Precision Related | Dtype/Precision Issue - torch.logcumsumexp returns NaNs for complex64 input |  | ut |
| [2853](https://github.com/intel/torch-xpu-ops/issues/2853) | [upstream_ut] torch.ops.aten._flash | LuFinch | BBBela | P2 | Distributed | Flash Attention/Specific Ops Issue - _flash_attention_forward not supported on X |  | ut |
| [2899](https://github.com/intel/torch-xpu-ops/issues/2899) | Update nan_to_num XPU stub to use s | None | cleonard530 | P2 | Dtype / Precision Related | Backend/Device Issue - XPU stub uses incorrect type for nan_to_num parameters |  | ut |
| [2948](https://github.com/intel/torch-xpu-ops/issues/2948) | [AO] Benchmark enabling on XPU | None | liangan1 | P2 | Others | Backend/Device Issue - XPU benchmark enabling issue |  | ut |
| [2950](https://github.com/intel/torch-xpu-ops/issues/2950) | SYCL compilation flag -fsycl-id-que | BBBela | BBBela | P2 | Inductor / Compilation Related | Inductor/Compilation Issue - SYCL compilation flag not working as expected for T |  | ut |
| [3021](https://github.com/intel/torch-xpu-ops/issues/3021) | [distributed] all_to_all_single Com | zhangxiaoli73 | xiangyuT | P2 | Distributed | Distributed/Gloo Issue - all_to_all_single compatibility issue on B60/XCCL |  | ut |
| [3022](https://github.com/intel/torch-xpu-ops/issues/3022) | [distributed] batch_isend_irecv Com | zhangxiaoli73 | xiangyuT | P2 | Distributed | Distributed/Gloo Issue - batch_isend_irecv compatibility issue on B60/XCCL |  | ut |
| [3024](https://github.com/intel/torch-xpu-ops/issues/3024) | Enable clang-tidy checks | Silv3S | Silv3S | P2 | Others | Others - clang-tidy checks enablement is a code quality/linter configuration task, not a runtime or functional issue. |  | ut |
| [3048](https://github.com/intel/torch-xpu-ops/issues/3048) | Profiler result is not correct on B | aostrowski-hbn | jianyizh | P2 | Others | Backend/Device Issue - Profiler result discrepancy on B70 device. |  | ut |
| [3081](https://github.com/intel/torch-xpu-ops/issues/3081) | Sparse CSR gemm-like ops have not b | None | daisyden | P2 | Sparse Operations Related | Supported - Sparse CSR gemm-like operations are not supported yet. |  | ut |
| [3082](https://github.com/intel/torch-xpu-ops/issues/3082) | multithread support in distributed | None | daisyden | P2 | Distributed | Distributed/Gloo Issue - multithread support in distributed operations is affect |  | ut |
| [3084](https://github.com/intel/torch-xpu-ops/issues/3084) | torch.library.register_autocast doe | None | daisyden | P2 | Others | Backend/Device Issue - torch.library.register_autocast does not support XPU devi |  | ut |
| [3086](https://github.com/intel/torch-xpu-ops/issues/3086) | nvml support blocks some test cases | None | daisyden | P2 | Others | Backend/Device Issue - nvml support blocking test cases on XPU |  | ut |
| [3093](https://github.com/intel/torch-xpu-ops/issues/3093) | XPU does not support NestedTensor f | None | daisyden | P2 | Flash Attention / Transformer Related | Supported - XPU does not support NestedTensor for SDPA operations. |  | ut |
| [3096](https://github.com/intel/torch-xpu-ops/issues/3096) | VISIBLE_DEVICE support | None | daisyden | P2 | Others | Backend/Device Issue - VISIBLE_DEVICE support is related to device visibility an |  | ut |
| [3100](https://github.com/intel/torch-xpu-ops/issues/3100) | [distributed] /handler/dump_nccl_tr | None | zxd1997066 | P2 | Distributed | Distributed/Gloo Issue - related to NCCL logging and trace handling in distribut |  | ut |
| [3103](https://github.com/intel/torch-xpu-ops/issues/3103) | Tensor-likes are not equal for test | BBBela | BBBela | P2 | Dtype / Precision Related | Backend/Device Issue - XPU tensor-like comparison failure in test |  | ut |
| [3180](https://github.com/intel/torch-xpu-ops/issues/3180) | [E2E] Timm/Torchbench models got "e | None | libohao1201 | P0 | Others | Backend/Device Issue - eager_two_runs_differ on ARC XPU backend |  | ut |
| [3189](https://github.com/intel/torch-xpu-ops/issues/3189) | Task Tracker | guangyey | guangyey | P2 | Others | Skip/No Test Exists - no test or error details provided |  | ut |
| [3194](https://github.com/intel/torch-xpu-ops/issues/3194) | Incorrect strides in TestCommonXPU, | AKloniecki | AKloniecki | P2 | Distributed | Backend/Device Issue - Incorrect strides related to XPU device handling |  | ut |
| [3196](https://github.com/intel/torch-xpu-ops/issues/3196) | vitals is not supported, the cases  | libohao1201 | daisyden | P2 | Others | 10 - vitals feature is not supported, cases should be disabled |  | ut |
| [3216](https://github.com/intel/torch-xpu-ops/issues/3216) | [OPs] Some ops of XPU have non-dete | CuiYifeng | YangKai0616 | P2 | Others | Backend/Device Issue - XPU ops show non-determinism and inconsistency compared t |  | ut |
| [1171](https://github.com/intel/torch-xpu-ops/issues/1171) | LNL Windows got unexpected error me | xuhancn, chunhuanMeng | daisyden | P2 | Others | Backend/Device Issue - unexpected error on XPU for LNL Windows |  | ut |
| [1324](https://github.com/intel/torch-xpu-ops/issues/1324) | [Win] UR Error when OOM and break t | Stonepia | Stonepia | P2 | Others | Memory/Shared Memory Issue - Out of memory (OOM) leading to tensor context break |  | ut |
| [1548](https://github.com/intel/torch-xpu-ops/issues/1548) | [distributed] AssertionError: 'fuse | Chao1Han | PenghuiCheng | P2 | Distributed | Failure - 'fused_all_gather_matmul' not found in AOT ID list |  | ut |
| [1549](https://github.com/intel/torch-xpu-ops/issues/1549) | [distributed] AssertionError: 'fuse | Chao1Han | PenghuiCheng | P2 | Distributed | Distributed/Gloo Issue - 'fused_all_gather_scaled_matmul' not found in graph dur |  | ut |
| [1555](https://github.com/intel/torch-xpu-ops/issues/1555) | [distributed] RuntimeError: aten.ad | chuanqi129 | PenghuiCheng | P2 | Distributed | Distributed/Gloo Issue - mixed torch.Tensor and DTensor in distributed operation |  | ut |
| [1571](https://github.com/intel/torch-xpu-ops/issues/1571) | [distributed] ValueError: Cannot us | zhangxiaoli73 | daisyden | P2 | Distributed | Distributed/Gloo Issue - ReduceOp.PREMUL_SUM not supported with XCCL |  | ut |
| [1649](https://github.com/intel/torch-xpu-ops/issues/1649) | [cpp extension] Provide a clear err | dvrogozh | ZhaoqiongZ | P2 | Others | Backend/Device Issue - inconsistent oneAPI versions cause XPU backend errors |  | ut |
| [1661](https://github.com/intel/torch-xpu-ops/issues/1661) | [distributed] Accuracy gap in _comp | githubsgi | PenghuiCheng | P2 | Distributed | Distributed/Gloo Issue - Accuracy gap in _composable/fsdp on Xelink suggests a d |  | ut |
| [1727](https://github.com/intel/torch-xpu-ops/issues/1727) | [distributed] AttributeError: modul | guangyey | PenghuiCheng | P2 | Distributed | Backend/Device Issue - missing attribute '_sleep' in torch.xpu module |  | ut |
| [1749](https://github.com/intel/torch-xpu-ops/issues/1749) | transformers UT failure in XPU beca | LuFinch | sywangyi | P2 | Flash Attention / Transformer Related | Backend/Device Issue - XPU does not support backward or grad for SDPA operation |  | ut |
| [1784](https://github.com/intel/torch-xpu-ops/issues/1784) | [Performance] Torch XPU Profiler is | jfedorov | liangan1 | P2 | Others | 11 - Timeout/Performance Issue - Torch XPU Profiler is not reliable |  | ut |
| [1833](https://github.com/intel/torch-xpu-ops/issues/1833) | [PT2E] INT8 accuracy is lower than  | chunhuanMeng | mengfei25 | P2 | TorchAO | Dtype/Precision Issue - INT8 accuracy lower than FP32 due to precision reduction |  | ut |
| [1877](https://github.com/intel/torch-xpu-ops/issues/1877) | Torchbench model squeezenet1_1 and  | Silv3S | libohao1201 | P0 | Dtype / Precision Related | Backend/Device Issue - fail_accuracy on XPU for specific models |  | ut |
| [1969](https://github.com/intel/torch-xpu-ops/issues/1969) | torch._dynamo.exc.InternalTorchDyna | guangyey | shangerxin | P2 | PT2E | Error - cannot create weak reference to 'torch.Event' object |  | ut |
| [1970](https://github.com/intel/torch-xpu-ops/issues/1970) | torch._dynamo.exc.BackendCompilerFa | None | shangerxin | P2 | PT2E | Backend/Device Issue - CUDA not available on the system |  | ut |
| [2004](https://github.com/intel/torch-xpu-ops/issues/2004) | [distributed][shared_tensor] test\d | libohao1201 | libohao1201 | P2 | Distributed | Memory/Shared Memory Issue - error originated from shared memory connection in t |  | ut |
| [2022](https://github.com/intel/torch-xpu-ops/issues/2022) | [Windows] [CI] [UT] AssertionError: | None | RUIJIEZHONG66166 | P2 | Dtype / Precision Related | Failure - Tensor-likes are not close! |  | ut |
| [2024](https://github.com/intel/torch-xpu-ops/issues/2024) | AssertionError: Torch not compiled  | daisyden | mengfei25 | P2 | Distributed |  |  | ut |
| [2157](https://github.com/intel/torch-xpu-ops/issues/2157) | BMG d2h copy is very slow compare t | jianyizh, mengfei25 | jianyizh | P2 | Others | Timeout/Performance Issue - BMG d2h copy is significantly slower compared to PVC |  | ut |
| [2165](https://github.com/intel/torch-xpu-ops/issues/2165) | [distributed] test_device_mesh.py:: | jemitche1 | zxd1997066 | P2 | Distributed | Failure - test_flatten_mesh_3d encountered an assertion error |  | ut |
| [2169](https://github.com/intel/torch-xpu-ops/issues/2169) | Frame size comparison failed in tes | guangyey | daisyden | P2 | Distributed | Failure - Scalars are not equal in test comparison |  | ut |
| [2182](https://github.com/intel/torch-xpu-ops/issues/2182) | test_transform_bias_rescale_qkv_nes | PawelSwider2000 | wincent8 | P2 | Distributed | Failure - Scalars are not equal in test assertion |  | ut |
| [2201](https://github.com/intel/torch-xpu-ops/issues/2201) | [TorchAO][BMG] When using paged att | Stonepia | MingxuZh | P2 | TorchAO | Failure - assert vr is not None error encountered |  | ut |
| [2217](https://github.com/intel/torch-xpu-ops/issues/2217) | AO Performance issue track | Stonepia | liangan1 | P2 | Others | Timeout/Performance Issue - AO Performance issue track |  | ut |
| [2219](https://github.com/intel/torch-xpu-ops/issues/2219) | float8_e4m3fn precision overflow | CuiYifeng, yucai-intel | jiqing-feng | P2 | Dtype / Precision Related | Dtype/Precision Issue - float8_e4m3fn precision overflow |  | ut |
| [2239](https://github.com/intel/torch-xpu-ops/issues/2239) | Exception: could not create a primi | wpietka | zxd1997066 | P2 | Distributed | DNNL/OneDNN Issue - could not create a primitive descriptor for deconvolution fo |  | ut |
| [2240](https://github.com/intel/torch-xpu-ops/issues/2240) | RuntimeError: Trying to set a forwa | gplutop7 | zxd1997066 | P2 | Distributed | Error - forward gradient size mismatch with original Tensor |  | ut |
| [2245](https://github.com/intel/torch-xpu-ops/issues/2245) | oneDNN matmul received incorrect sh | CuiYifeng | wincent8 | P2 | Distributed | DNNL/OneDNN Issue - oneDNN matmul primitive descriptor creation failure |  | ut |
| [2263](https://github.com/intel/torch-xpu-ops/issues/2263) | [xpu][bug] XPU Trace event ends too | PawelSwider2000 | chuanqi129 | P2 | Others | Backend/Device Issue - XPU trace event timing discrepancy |  | ut |
| [2340](https://github.com/intel/torch-xpu-ops/issues/2340) | [distributed][_tools] AssertionErro | githubsgi | zxd1997066 | P2 | Distributed | Failure - Roofline estimation requires CUDA capabilities assertion failed |  | ut |
| [2389](https://github.com/intel/torch-xpu-ops/issues/2389) | [Bug Skip]: RuntimeError: Data corr | PatrykWilczewski | kaileiyx | P2 | Distributed | Error - Data corruption detected during test execution |  | ut |
| [2392](https://github.com/intel/torch-xpu-ops/issues/2392) | [Bug Skip]: torch.OutOfMemoryError: | xuhancn | RUIJIEZHONG66166 | P2 | Others | Memory/Shared Memory Issue - XPU out of memory error occurred |  | ut |
| [2404](https://github.com/intel/torch-xpu-ops/issues/2404) | [distributed][checkpoint] Assertion | None | zxd1997066 | P2 | Distributed | Failure - Booleans mismatch assertion error |  | ut |
| [2425](https://github.com/intel/torch-xpu-ops/issues/2425) | [upstream_ut]  RuntimeError: Expect | BBBela | daisyden | P2 | Dtype / Precision Related | Error - Nested tensor operation with non-nested tensor input |  | ut |
| [2434](https://github.com/intel/torch-xpu-ops/issues/2434) | [Bug Skip]: New failures 2025-11-28 | AKloniecki | mengfei25 | P2 | TorchAO | Skip/No Test Exists - test is marked as a bug skip or not implemented properly |  | ut |
| [2439](https://github.com/intel/torch-xpu-ops/issues/2439) | [oneDNN] TestDecompXPU.test_quick_a | libohao1201 | mengfei25 | P2 | Dtype / Precision Related | DNNL/OneDNN Issue - Test failure related to oneDNN decomposition on XPU with flo |  | ut |
| [2440](https://github.com/intel/torch-xpu-ops/issues/2440) | [For UT failures classify] Save ref | None | RUIJIEZHONG66166 | P2 | Others | Skip/No Test Exists - no test or error details provided |  | ut |
| [2444](https://github.com/intel/torch-xpu-ops/issues/2444) | [upstream_ut]  RuntimeError: UR bac | Silv3S | wincent8 | P2 | Dtype / Precision Related | Memory/Shared Memory Issue - UR backend returns out of resources error (40) indi |  | ut |
| [2446](https://github.com/intel/torch-xpu-ops/issues/2446) | [Bug Skip]: AssertionError: "Simula | None | kaileiyx | P2 | Distributed | Failure - mismatch between expected and actual error messages |  | ut |
| [2447](https://github.com/intel/torch-xpu-ops/issues/2447) | test_share_memory_xpu failure | None | daisyden | P2 | Others | Memory/Shared Memory Issue - failure in test_share_memory_xpu related to shared  |  | ut |
| [2463](https://github.com/intel/torch-xpu-ops/issues/2463) | [Bug Skip]: OSError: SYCL runtime i | xuhancn | RUIJIEZHONG66166 | P2 | Others | Backend/Device Issue - SYCL runtime not detected on XPU |  | ut |
| [2479](https://github.com/intel/torch-xpu-ops/issues/2479) | [Bug] torch.rand output different r | Stonepia, CuiYifeng | zufangzhu | P2 | Others | Backend/Device Issue - different output on BMG and PVC devices |  | ut |
| [2491](https://github.com/intel/torch-xpu-ops/issues/2491) | [upstream_ut]  AssertionError: Fals | PatrykWilczewski | libohao1201 | P2 | Dtype / Precision Related | Failure - test assertion failed with False is not true |  | ut |
| [2494](https://github.com/intel/torch-xpu-ops/issues/2494) | [upstream_ut]  AssertionError: Scal | PawelSwider2000 | libohao1201 | P2 | Dtype / Precision Related | Failure - Scalars are not equal assertion error in test |  | ut |
| [2510](https://github.com/intel/torch-xpu-ops/issues/2510) | [upstream_ut]  RuntimeError: Expect | PawelSwider2000 | libohao1201 | P2 | Dtype / Precision Related | Error - tensor size exceeds int32_t maximum limit |  | ut |
| [2513](https://github.com/intel/torch-xpu-ops/issues/2513) | [upstream_ut]  RuntimeError: _share | gplutop7 | libohao1201 | P2 | Others | Memory/Shared Memory Issue - _share_fd_ is not available on XPU, indicating a sh |  | ut |
| [2531](https://github.com/intel/torch-xpu-ops/issues/2531) | [upstream_ut]  AssertionError: Torc | daisyden | daisyden | P2 | Distributed |  |  | ut |
| [2532](https://github.com/intel/torch-xpu-ops/issues/2532) | Title: [upstream_ut]  AssertionErro | yucai-intel | daisyden | P2 | Distributed | Failure - wrong number of dimensions for int4 conversion op |  | ut |
| [2533](https://github.com/intel/torch-xpu-ops/issues/2533) | Title: [upstream_ut]  AttributeErro | astachowiczhabana | daisyden | P2 | Distributed | Mismatch - Test method 'test_qsoftmax' is missing in 'TestQuantizedOpsXPU' class. |  | ut |
| [2535](https://github.com/intel/torch-xpu-ops/issues/2535) | Title: [upstream_ut]  AttributeErro | Silv3S | daisyden | P2 | Distributed | Backend/Device Issue - missing XPU-specific attribute in torch._C for tunableop  |  | ut |
| [2536](https://github.com/intel/torch-xpu-ops/issues/2536) | Title: [upstream_ut]  AttributeErro | daisyden | daisyden | P2 | Distributed | Backend/Device Issue - missing attribute '_scatter' in torch._C for XPU device |  | ut |
| [2537](https://github.com/intel/torch-xpu-ops/issues/2537) | Title: [upstream_ut]  Failed: Unexp | PatrykWilczewski | daisyden | P2 | Others | Others - Test expects failure but passed unexpectedly, no specific error trace provided. |  | ut |
| [2538](https://github.com/intel/torch-xpu-ops/issues/2538) | Title: [upstream_ut]  RuntimeError: | Silv3S | daisyden | P2 | Distributed | DNNL/OneDNN Issue - Float8_e4m3fnuz is not supported in oneDNN. |  | ut |
| [2539](https://github.com/intel/torch-xpu-ops/issues/2539) | Title: [upstream_ut]  RuntimeError: | None | daisyden | P2 | Distributed | Backend/Device Issue - Attempt to instantiate dummy base class CUDAGraph on XPU |  | ut |
| [2541](https://github.com/intel/torch-xpu-ops/issues/2541) | Title: [upstream_ut]  RuntimeError: | yucai-intel | daisyden | P2 | Distributed | DNNL/OneDNN Issue - could not construct a memory descriptor using strides |  | ut |
| [2560](https://github.com/intel/torch-xpu-ops/issues/2560) | [UT] "RuntimeError: iter.device(arg | CuiYifeng | libohao1201 | P2 | Others | Backend/Device Issue - XPU device check failure in test |  | ut |
| [2562](https://github.com/intel/torch-xpu-ops/issues/2562) | Warning as Error | chunhuanMeng | EikanWang | P2 | Others | Others - warning treated as error but no traceback or specific error provided |  | ut |
| [2572](https://github.com/intel/torch-xpu-ops/issues/2572) | [TorchAO][UT] test/dtypes/test_affi | xiaowangintel | zxd1997066 | P0 | TorchAO | Failure - Tensor-likes are not close! |  | build |
| [2580](https://github.com/intel/torch-xpu-ops/issues/2580) | [TorchAO][UT] test/test_low_bit_opt | arlesniak | zxd1997066 | P0 | TorchAO | Failure - Tensor-likes are not close! |  | build |
| [2595](https://github.com/intel/torch-xpu-ops/issues/2595) | [Bug Skip]: Random crashed cases 20 | None | CuiYifeng | P0 | Sparse Operations Related | Skip/No Test Exists - Test was skipped due to random crashed cases. |  | ut |
| [2596](https://github.com/intel/torch-xpu-ops/issues/2596) | [TorchAO][BMG]Observed accuracy flu | None | LifengWang | P0 | TorchAO | Dtype/Precision Issue - accuracy fluctuations due to INT4 quantization methods ( |  | ut |
| [2597](https://github.com/intel/torch-xpu-ops/issues/2597) | [TorchAO][BMG] INT4 GPTQ shows wors | xiaowangintel | LifengWang | P2 | TorchAO | Timeout/Performance Issue - INT4 GPTQ shows worse performance compared with RTN  |  | ut |
| [2615](https://github.com/intel/torch-xpu-ops/issues/2615) | [Bug Skip]: New failures RuntimeErr | CuiYifeng | kaileiyx | P2 | Distributed | Dtype/Precision Issue - Unsupported dtype Half / torch.float16 |  | ut |
| [2630](https://github.com/intel/torch-xpu-ops/issues/2630) | Title: [upstream_ut]  AssertionErro | jmamzax | daisyden | P2 | Distributed | Failure - Scalars are not equal! |  | ut |
| [2639](https://github.com/intel/torch-xpu-ops/issues/2639) | test_to() failed during rnn isinsta | Silv3S | daisyden | P2 | Distributed | Failure - test_to() failed during rnn isinstance() check | [PR](https://github.com/intel/torch-xpu-ops/pull/3193) | ut |
| [2645](https://github.com/intel/torch-xpu-ops/issues/2645) | [Bug Skip]: [regression] RuntimeErr | CuiYifeng | kaileiyx | P0 | Inductor / Compilation Related | Backend/Device Issue - missing kernel for xpu in DispatchStub |  | ut |
| [2669](https://github.com/intel/torch-xpu-ops/issues/2669) | [upstream_ut]  AssertionError: Tens | tszulist-hbn | daisyden | P2 | Distributed | Failure - Tensor-likes are not close! in test_vmap_exhaustive_addmv_xpu_float32 |  | ut |
| [2676](https://github.com/intel/torch-xpu-ops/issues/2676) | Random failure in CI test | None | daisyden | P2 | Dtype / Precision Related | Others - Random failure with no traceback or specific error provided |  | ut |
| [2680](https://github.com/intel/torch-xpu-ops/issues/2680) | XPU Autocast does not support  fp32 | CuiYifeng | kaixuanliu | P2 | Dtype / Precision Related | Dtype/Precision Issue - XPU Autocast does not support fp32 dtypes |  | ut |
| [2686](https://github.com/intel/torch-xpu-ops/issues/2686) | [distributed] Accuracy issues with  | frost-intel | madhumitha0102 | P2 | Distributed | Distributed/Gloo Issue - Accuracy issues in distributed testing with test_distri |  | ut |
| [2689](https://github.com/intel/torch-xpu-ops/issues/2689) | [LNL][Windows] AssertionError: 'Ass | tadkrawiec | kaileiyx | P2 | Dtype / Precision Related | Failure - cur_target out of bounds assertion failed |  | ut |
| [2701](https://github.com/intel/torch-xpu-ops/issues/2701) | [distributed] Barrier Timeout Error | syedshahbaaz | madhumitha0102 | P2 | Distributed | Distributed/Gloo Issue - Barrier Timeout Error in distributed testing |  | ut |
| [2702](https://github.com/intel/torch-xpu-ops/issues/2702) | [distributed] RuntimeError: Work ra | syedshahbaaz | madhumitha0102 | P2 | Distributed | Timeout/Performance Issue - Work ran time out after 0 milliseconds in distribute |  | ut |
| [2707](https://github.com/intel/torch-xpu-ops/issues/2707) | [TorchAO][BMG] INT4 GPTQ failed due | xiaowangintel | LifengWang | P2 | TorchAO | Mismatch - INT4 GPTQ failed due to TorchAO API change. |  | ut |
| [2720](https://github.com/intel/torch-xpu-ops/issues/2720) | [upstream_ut] RuntimeError: false I | CuiYifeng | wincent8 | P2 | Distributed | Backend/Device Issue - missing kernel for xpu in DispatchStub |  | ut |
| [2729](https://github.com/intel/torch-xpu-ops/issues/2729) | [Bug Skip]: Random failures 2026WW0 | Silv3S | CuiYifeng | P2 | Sparse Operations Related | Skip/No Test Exists - test is marked as skipped due to random failures |  | ut |
| [2734](https://github.com/intel/torch-xpu-ops/issues/2734) | [TorchAO][BMG] INT4 RTN Flex-attent | Stonepia, hoshibara | LifengWang | P2 | TorchAO | Timeout/Performance Issue - 50% performance drop in INT4 RTN Flex-attention with | [PR](https://github.com/pytorch/pytorch/pull/160480, https://github.com/pytorch/pytorch/pull/172316) | ut |
| [2737](https://github.com/intel/torch-xpu-ops/issues/2737) | [distributed] AttributeError: modul | None | zxd1997066 | P2 | Distributed | Distributed/Gloo Issue - missing attribute '_gather' in distributed context |  | ut |
| [2738](https://github.com/intel/torch-xpu-ops/issues/2738) | [distributed] test_c10d_spawn_nccl. | jenniew | zxd1997066 | P2 | Distributed | Distributed/Gloo Issue - input tensor size mismatch in distributed context |  | ut |
| [2744](https://github.com/intel/torch-xpu-ops/issues/2744) | [Bug Skip]: extended test failures  | pbielak | daisyden | P2 | Others | Skip/No Test Exists - test was skipped due to changes in tolerance values causin |  | ut |
| [2751](https://github.com/intel/torch-xpu-ops/issues/2751) | [Bug Skip]: Random failures 2026WW0 | None | CuiYifeng | P2 | Sparse Operations Related | Skip/No Test Exists - test was skipped due to random failure标记 |  | ut |
| [2759](https://github.com/intel/torch-xpu-ops/issues/2759) | [Bug Skip]: New failed cases 2026-1 | AKloniecki | kaileiyx | P2 | Distributed | Failure - RuntimeError not raised as expected in test case |  | ut |
| [2766](https://github.com/intel/torch-xpu-ops/issues/2766) | MaxPool2d - investigate memory layo | BBBela | pbielak | P2 | Others | Memory/Shared Memory Issue - investigate memory layout performance for MaxPool2d |  | ut |
| [2767](https://github.com/intel/torch-xpu-ops/issues/2767) | [UT] test_control_flow_xpu.py got A | PatrykWilczewski | libohao1201 | P1 | PT2E | Failure - test_control_flow_xpu.py got AssertionError |  | ut |
| [2769](https://github.com/intel/torch-xpu-ops/issues/2769) | [oneDNN] New failed test cases with | LuFinch | mengfei25 | P2 | Others | DNNL/OneDNN Issue - failed test cases related to oneDNN with 3.11 compared to 3. |  | ut |
| [2777](https://github.com/intel/torch-xpu-ops/issues/2777) | [Bug Skip]: Random failures 2026WW0 | AKloniecki | CuiYifeng | P2 | Sparse Operations Related | Skip/No Test Exists - test is marked as a skip with no detailed error traceback  |  | ut |
| [2779](https://github.com/intel/torch-xpu-ops/issues/2779) | Accuracy failures in logspace op | PawelSwider2000 | PawelSwider2000 | P2 | Distributed | Dtype/Precision Issue - accuracy failures in logspace operation |  | ut |
| [2783](https://github.com/intel/torch-xpu-ops/issues/2783) | [Bug Skip]: Key "xpu" is missing fr | daisyden | CuiYifeng | P2 | Dtype / Precision Related | Skip/No Test Exists - Key "xpu" is missing, indicating the test is skipped or no |  | ut |
| [2795](https://github.com/intel/torch-xpu-ops/issues/2795) | Histc raises error with integer inp | CuiYifeng | YangKai0616 | P2 | Others | Dtype/Precision Issue - integer input causes error with deterministic algorithm  |  | ut |
| [2797](https://github.com/intel/torch-xpu-ops/issues/2797) | Copy error is not raise on test_dlp | None | shangerxin | P2 | Others | Others - Copy error not raised in test_dlpack.py test case |  | ut |
| [2801](https://github.com/intel/torch-xpu-ops/issues/2801) | to_dense() for Sparse CSR backend c | jenniew | jenniew | P2 | Sparse Operations Related | Error - source tensor shape mismatch during to_dense() for Sparse CSR backend |  | ut |
| [2811](https://github.com/intel/torch-xpu-ops/issues/2811) | [Bug Skip]: [Regression] failed cas | jmamzax | kaileiyx | P0 | Distributed | Failure - Expected and actual trace outputs do not match. |  | ut |
| [2815](https://github.com/intel/torch-xpu-ops/issues/2815) | RuntimeError: output with shape [2] | PawelSwider2000 | Silv3S | P2 | Others | Error - output shape mismatch during broadcasting |  | ut |
| [2823](https://github.com/intel/torch-xpu-ops/issues/2823) | [TorchAO][BMG] Llama-3.2-1B-Instruc | xiaowangintel, lchen2331 | LifengWang | P2 | TorchAO | Timeout/Performance Issue - 20% performance drop in next token generation with D |  | ut |
| [2845](https://github.com/intel/torch-xpu-ops/issues/2845) | [Bug Skip]:[UT] [Windows] failed ca | None | kaileiyx | P2 | Others | Skip/No Test Exists - test was skipped or does not exist |  | ut |
| [2852](https://github.com/intel/torch-xpu-ops/issues/2852) | [Bug Skip]: New UT failures in 0206 | None | chuanqi129 | P2 | Others | Skip/No Test Exists - test was skipped or not present |  | ut |
| [2858](https://github.com/intel/torch-xpu-ops/issues/2858) | [Bug Skip]: test_xpu new failures | None | RUIJIEZHONG66166 | P2 | Others | Skip/No Test Exists - test_xpu new failures marked as skipped or no test availab |  | ut |
| [2862](https://github.com/intel/torch-xpu-ops/issues/2862) | accuracy issue with test_float8_sca | tszulist-hbn | daisyden | P2 | Dtype / Precision Related | Dtype/Precision Issue - accuracy issue with float8 operations |  | ut |
| [2871](https://github.com/intel/torch-xpu-ops/issues/2871) | [distributed][fsdp] test_fsdp_overl | songhappy | zxd1997066 | P2 | Distributed | Failure - test_fsdp_overlap.py assertion failed with "False is not true" |  | ut |
| [2873](https://github.com/intel/torch-xpu-ops/issues/2873) | [Bug Skip]: test_repos.py contains  | PawelSwider2000 | shangerxin | P2 | PT2E | Skip/No Test Exists - test contains failed ops and is skipped |  | ut |
| [2879](https://github.com/intel/torch-xpu-ops/issues/2879) | RuntimeError: _share_fd_: only avai | Silv3S | Silv3S | P2 | Others | Backend/Device Issue - _share_fd_ is not available on XPU device |  | ut |
| [2914](https://github.com/intel/torch-xpu-ops/issues/2914) | Test case test/test_autograd.py::Te | None | shangerxin | P2 | Dtype / Precision Related | Failure - Tensor-likes are not close! |  | ut |
| [2920](https://github.com/intel/torch-xpu-ops/issues/2920) | Failing Test Cases in Nightly Wheel | Silv3S | BBBela | P2 | Others | Others - insufficient information to determine root cause |  | ut |
| [2921](https://github.com/intel/torch-xpu-ops/issues/2921) | [abs][complex64] - new failing test | AKloniecki | BBBela | P2 | Distributed | Dtype/Precision Issue - test failure related to complex64 data type and abs oper |  | ut |
| [2942](https://github.com/intel/torch-xpu-ops/issues/2942) | [Windows] Unit tests got Fatal pyth | xuhancn, Stonepia | mengfei25 | P2 | Others | Backend/Device Issue - Fatal Python error on Windows unit tests related to XPU b |  | ut |
| [2946](https://github.com/intel/torch-xpu-ops/issues/2946) | [Bug Skip]: Random failures 2026WW0 | None | CuiYifeng | P2 | TorchAO | Skip/No Test Exists - test is marked to be skipped with no valid test implementa |  | ut |
| [2965](https://github.com/intel/torch-xpu-ops/issues/2965) | [Bug Skip]: Random failures 2026WW1 | None | CuiYifeng | P2 | Sparse Operations Related | Skip/No Test Exists - test is marked as a skip with no valid test implementation |  | ut |
| [2968](https://github.com/intel/torch-xpu-ops/issues/2968) | [distributed] timeout issue in test | frost-intel | zxd1997066 | P2 | Distributed | Timeout/Performance Issue - test experienced a timeout in distributed execution  |  | ut |
| [2969](https://github.com/intel/torch-xpu-ops/issues/2969) | [distributed] AssertionError: Scala | frost-intel | zxd1997066 | P2 | Distributed | Failure - Scalars are not equal in test case |  | ut |
| [2971](https://github.com/intel/torch-xpu-ops/issues/2971) | [distributed] KeyError in test/dist | newtdms, frost-intel | zxd1997066 | P2 | Distributed | Distributed/Gloo Issue - KeyError in distributed test related to c10d_xccl |  | ut |
| [2972](https://github.com/intel/torch-xpu-ops/issues/2972) | [distributed] AssertionError: Value | newtdms | zxd1997066 | P2 | Distributed | Distributed/Gloo Issue - AssertionError in test/distributed/test_c10d_xccl.py re |  | ut |
| [2993](https://github.com/intel/torch-xpu-ops/issues/2993) | [Bug Skip]: Unexpected success of t | gplutop7 | CuiYifeng | P2 | Others | Skip/No Test Exists - test unexpectedly succeeded and should have been skipped |  | ut |
| [3000](https://github.com/intel/torch-xpu-ops/issues/3000) | [Bug Skip]: RuntimeError: _share_fd | gplutop7 | zxd1997066 | P2 | Others | Backend/Device Issue - _share_fd_ is not available on XPU device |  | ut |
| [3010](https://github.com/intel/torch-xpu-ops/issues/3010) | [distributed][tensor] test_random_o | jenniew | zxd1997066 | P2 | Distributed | Inductor/Compilation Issue - TorchRuntimeError during fake tensor call in Dynamo |  | ut |
| [3025](https://github.com/intel/torch-xpu-ops/issues/3025) | New failing test in Nightly Wheel t | None | BBBela | P2 | Distributed | Failure - Expected and actual decomposition outputs do not match. |  | ut |
| [3030](https://github.com/intel/torch-xpu-ops/issues/3030) | [Bug Skip] test/test_modules.py::Te | gplutop7 | shangerxin | P2 | Others | Skip/No Test Exists - test was skipped due to failure with no detailed error pro |  | ut |
| [3032](https://github.com/intel/torch-xpu-ops/issues/3032) | [TorchAO][UT] failures in test/prot | Stonepia | zxd1997066 | P0 | TorchAO | Others - insufficient information to determine root cause |  | build |
| [3033](https://github.com/intel/torch-xpu-ops/issues/3033) | [Bug Skip]: Softmax tolerance | chunhuanMeng | chunhuanMeng | P2 | Others | Skip/No Test Exists - test is skipped due to Softmax tolerance issue |  | ut |
| [3074](https://github.com/intel/torch-xpu-ops/issues/3074) | [Bug Skip] test_dlpack_exchange_api | AKloniecki | shangerxin | P2 | Others | Skip/No Test Exists - test is skipped expecting current_work_stream is not null |  | ut |
| [3076](https://github.com/intel/torch-xpu-ops/issues/3076) | [TorchAO][BMG] Llama-3.2-1B-Instruc | None | LifengWang | P2 | TorchAO | DNNL/OneDNN Issue - Performance drop with oneDNN 3.11.1 in Dynamic INT8 for Llam |  | ut |
| [3083](https://github.com/intel/torch-xpu-ops/issues/3083) | [Bug Skip]: Random failures 2026WW1 | None | CuiYifeng | P2 | Others | Skip/No Test Exists - test is marked to be skipped with no valid test to execute |  | ut |
| [3088](https://github.com/intel/torch-xpu-ops/issues/3088) | [TorchAO][BMG] INT4 RTN Flex-attent | Stonepia | LifengWang | P2 | TorchAO | Timeout/Performance Issue - INT4 RTN Flex-attention experienced a 5% performance |  | ut |
| [3101](https://github.com/intel/torch-xpu-ops/issues/3101) | [distributed] 'torch._C._distribute | None | zxd1997066 | P2 | Distributed | Distributed/Gloo Issue - ProcessGroupXCCL object lacks '_set_default_timeout' at |  | ut |
| [3102](https://github.com/intel/torch-xpu-ops/issues/3102) | [distributed] RuntimeError: Invalid | None | zxd1997066 | P2 | Distributed | Backend/Device Issue - invalid device string 'xpu:foo' indicates a problem with  |  | ut |
| [3121](https://github.com/intel/torch-xpu-ops/issues/3121) | [Bug Skip]: CUDA specific UT test_f | None | CuiYifeng | P1 | Dtype / Precision Related | Skip/No Test Exists - test is skipped or not applicable for XPU backend |  | ut |
| [3124](https://github.com/intel/torch-xpu-ops/issues/3124) | [TorchAO][Bug] ImportError: Require | None | FRAMEEE17 | P2 | TorchAO | Backend/Device Issue - Requires mslk >= 1.0.0 for XPU support with int4 quantiza |  | ut |
| [3139](https://github.com/intel/torch-xpu-ops/issues/3139) | [distributed][_composable] Assertio | Kanya-Mo | zxd1997066 | P2 | Distributed | Failure - Expects xpu:0 but got xpu:1 |  | ut |
| [3149](https://github.com/intel/torch-xpu-ops/issues/3149) | New failure in test_rms_norm_decomp | None | BBBela | P2 | Flash Attention / Transformer Related | Failure - RuntimeError not raised as expected in test |  | ut |
| [3174](https://github.com/intel/torch-xpu-ops/issues/3174) | [Bug Skip]: Accuracy failure of tes | None | CuiYifeng | P2 | Distributed | Failure - test assertion failed for Conv2d groups output comparison |  | ut |
| [3178](https://github.com/intel/torch-xpu-ops/issues/3178) | New failed test cases 2026-03-25 | pponikox | BBBela | P2 | Flash Attention / Transformer Related | Backend/Device Issue - XPU-specific test failure in vmap with SDPA operation |  | ut |
| [3195](https://github.com/intel/torch-xpu-ops/issues/3195) | test_sdpa_unbacked_no_dde_xpu crash | None | daisyden | P0 | Flash Attention / Transformer Related | Backend/Device Issue - test crashed on XPU backend execution |  | ut |
| [3206](https://github.com/intel/torch-xpu-ops/issues/3206) | [Bug Skip]: [new failures]RuntimeEr | tszulist-hbn | kaileiyx | P2 | Others | Backend/Device Issue - Tensors are on different devices (xpu:0 vs cpu) |  | ut |
| [3209](https://github.com/intel/torch-xpu-ops/issues/3209) | [Win][Build] There is Cyclic depend | Copilot | NeoZhangJianyu | P0 | Others | Backend/Device Issue - Cyclic dependencies during build with BUILD_SEPARATE_OPS= |  | build |
| [3224](https://github.com/intel/torch-xpu-ops/issues/3224) | [Win][Build] Building SYCL (Device) | chunhuanMeng | anmyachev | P0 | Inductor / Compilation Related | Backend/Device Issue - SYCL kernel build failure on Windows for XPU |  | build |
| [3227](https://github.com/intel/torch-xpu-ops/issues/3227) | torch xpu event has ~0.1ms latency, | guangyey | jianyizh | P2 | Others | Timeout/Performance Issue - high latency in XPU event (~0.1ms) indicates a perfo |  | ut |
| [3231](https://github.com/intel/torch-xpu-ops/issues/3231) | Dynamo failed to run FX node with f | None | daisyden | P2 | PT2E | Inductor/Compilation Issue - Dynamo failed to compile FX node with fake tensors  |  | ut |
| [3232](https://github.com/intel/torch-xpu-ops/issues/3232) | [distributed][tensor] AssertionErro | None | zxd1997066 | P2 | Distributed | Failure - AssertionError not raised for Placement (Shard(dim=2),) in test |  | ut |
| [3233](https://github.com/intel/torch-xpu-ops/issues/3233) | [distributed] RuntimeError: No back | None | zxd1997066 | P2 | Distributed | Distributed/Gloo Issue - No backend for the parent process group or its backend  |  | ut |
| [3236](https://github.com/intel/torch-xpu-ops/issues/3236) | AssertionError: 'def [28 chars]n un | jmamzax | jmamzax | P2 | Dtype / Precision Related | Failure - mismatch in expected IR code for XPU backend operations |  | ut |
| [3242](https://github.com/intel/torch-xpu-ops/issues/3242) | AssertionError: Torch not compiled  | None | zxd1997066 | P2 | Inductor / Compilation Related |  |  | ut |
| [3243](https://github.com/intel/torch-xpu-ops/issues/3243) | AssertionError: False is not true | pponikox | zxd1997066 | P2 | Dtype / Precision Related | Failure - assertion 'False is not true' failed in test |  | ut |
| [3258](https://github.com/intel/torch-xpu-ops/issues/3258) | Error in op: torch.ops.aten._scaled | None | bjarzemb | P2 | Flash Attention / Transformer Related | 5 - Flash Attention/Specific Ops Issue - Error in _scaled_dot_product_fused_atte |  | ut |
| [3259](https://github.com/intel/torch-xpu-ops/issues/3259) | New failed test cases 2026-04-02 | SlawomirLaba | Silv3S | P2 | Flash Attention / Transformer Related | Backend/Device Issue - XPU device initialization or compatibility failure |  | ut |

#### Close Fixed Issue {#close-fixed-issue}

| ID | Title | Owner | Owner Transferred | Priority | Category | Root Cause | PR | Test Module |
|---|-------|-------|-------------------|---------|----------|-----------|-----|-------------|
| [1624](https://github.com/intel/torch-xpu-ops/issues/1624) | [DONT CLOSE] Known UT Issue list | None | RUIJIEZHONG66166 | P2 | Dtype / Precision Related |  |  | ut |
| [2496](https://github.com/intel/torch-xpu-ops/issues/2496) | [upstream_ut]  Segmentation fault w | astachowiczhabana | libohao1201 | P0 | Dtype / Precision Related |  |  | ut |
| [2518](https://github.com/intel/torch-xpu-ops/issues/2518) | [upstream_ut]  TypeError: Creating  | astachowiczhabana | libohao1201 | P2 | Others |  |  | ut |
| [2592](https://github.com/intel/torch-xpu-ops/issues/2592) | [release/2.10] models got fail_accu | mengfei25 | mengfei25 | P0 | Dtype / Precision Related |  |  | e2e |
| [2619](https://github.com/intel/torch-xpu-ops/issues/2619) | [release/2.10] Some models inductor | jianyizh, weishi-deng | mengfei25 | P0 | Inductor / Compilation Related |  |  | e2e |
| [2953](https://github.com/intel/torch-xpu-ops/issues/2953) | [release/2.11][wsl] huggingface TrO | xuhancn | bjarzemb | P0 | Others |  |  | e2e |
| [2981](https://github.com/intel/torch-xpu-ops/issues/2981) | [release/2.11] T5 models performanc | jianyizh, weishi-deng | mengfei25 | P0 | Others |  |  | e2e |
| [3011](https://github.com/intel/torch-xpu-ops/issues/3011) | [upstream_ut] torch.OutOfMemoryErro | None | Silv3S | P2 | Others |  |  | ut |
| [3013](https://github.com/intel/torch-xpu-ops/issues/3013) | [upstream_ut] RuntimeError: Kernel  | None | Silv3S | P2 | Inductor / Compilation Related |  |  | ut |
| [3014](https://github.com/intel/torch-xpu-ops/issues/3014) | [upstream_ut] AssertionError: False | None | Silv3S | P2 | Dtype / Precision Related |  |  | ut |
| [3058](https://github.com/intel/torch-xpu-ops/issues/3058) | [E2E] hf_GPT2_large amp_fp16/amp_bf | weishi-deng | kaileiyx | P1 | Flash Attention / Transformer Related |  |  | e2e |
| [3106](https://github.com/intel/torch-xpu-ops/issues/3106) | Worker crashes when running TestDec | BBBela | BBBela | P0 | Others |  |  | ut |
| [3157](https://github.com/intel/torch-xpu-ops/issues/3157) | XPU out of memory. Tried to allocat | kdrozd-dev | kdrozd-dev | P2 | Others |  |  | ut |
| [3158](https://github.com/intel/torch-xpu-ops/issues/3158) | AttributeError: module 'triton.comp | tadkrawiec | kdrozd-dev | P2 | Inductor / Compilation Related |  |  | ut |
| [3161](https://github.com/intel/torch-xpu-ops/issues/3161) | Exception: Tensor-likes are not clo | tadkrawiec | kdrozd-dev | P2 | Dtype / Precision Related |  |  | ut |
| [3166](https://github.com/intel/torch-xpu-ops/issues/3166) | test_consistency_SparseCSR failures | yucai-intel | CuiYifeng | P2 | TorchAO |  |  | ut |

#### Enable Test {#enable-test}

| ID | Title | Owner | Owner Transferred | Priority | Category | Root Cause | PR | Test Module |
|---|-------|-------|-------------------|---------|----------|-----------|-----|-------------|

#### Add to Skiplist {#add-to-skiplist}

| ID | Title | Owner | Owner Transferred | Priority | Category | Root Cause | PR | Test Module |
|---|-------|-------|-------------------|---------|----------|-----------|-----|-------------|
| [2164](https://github.com/intel/torch-xpu-ops/issues/2164) | skip test_no_cuda_monkeypatch as it | daisyden | daisyden | P2 | Distributed |  |  | ut |
| [2309](https://github.com/intel/torch-xpu-ops/issues/2309) | unsupported ops with PYTORCH_ENABLE | daisyden | daisyden | P2 | TorchAO |  |  | ut |
| [2472](https://github.com/intel/torch-xpu-ops/issues/2472) | [upstream_ut]  NotImplementedError: | Silv3S | daisyden | P2 | PT2E |  |  | ut |
| [2508](https://github.com/intel/torch-xpu-ops/issues/2508) | TypedStorage / TypedTensors depreca | Silv3S | libohao1201 | P1 | TorchAO |  | [PR](https://github.com/intel/torch-xpu-ops/pull/3260) | ut |

#### Verify the Issue {#verify-the-issue}

| ID | Title | Owner | Owner Transferred | Priority | Category | Root Cause | PR | Test Module |
|---|-------|-------|-------------------|---------|----------|-----------|-----|-------------|
| [2331](https://github.com/intel/torch-xpu-ops/issues/2331) | [upstream_ut] AssertionError: Scala | hoshibara | daisyden | P2 | Dtype / Precision Related |  | [PR](https://github.com/pytorch/pytorch/pull/172314) | ut |
| [2694](https://github.com/intel/torch-xpu-ops/issues/2694) | Title: [upstream_ut]  AssertionErro | daisyden | daisyden | P2 | Distributed |  | [PR](https://github.com/pytorch/pytorch/pull/171773) | ut |
| [3007](https://github.com/intel/torch-xpu-ops/issues/3007) | AssertionError: Scalars are not equ | daisyden | daisyden | P2 | Flash Attention / Transformer Related |  | [PR](https://github.com/pytorch/pytorch/pull/178369) | e2e |

#### Need Reproduce Steps {#need-reproduce-steps}

| ID | Title | Owner | Owner Transferred | Priority | Category | Root Cause | PR | Test Module |
|---|-------|-------|-------------------|---------|----------|-----------|-----|-------------|

### Engineer Actions {#engineer-actions}

#### Needs PyTorch Repo Changes (upstream) {#needs-pytorch-repo-changes-upstream}

| ID | Title | Owner | Owner Transferred | Priority | Category | Root Cause | PR | Test Module |
|---|-------|-------|-------------------|---------|----------|-----------|-----|-------------|
| [489](https://github.com/intel/torch-xpu-ops/issues/489) | Moco NotImplementedError: xpu not s | weishi-deng | weishi-deng | P2 | Others | Backend/Device Issue - xpu not supported |  | e2e |
| [492](https://github.com/intel/torch-xpu-ops/issues/492) | Timm_efficientdet NotImplementedErr | weishi-deng | weishi-deng | P0 | Others | Backend/Device Issue - model code forces use of CUDA instead of XPU |  | e2e |
| [1159](https://github.com/intel/torch-xpu-ops/issues/1159) | [LNL Windows][Test by CD Nightly Wh | Stonepia | Stonepia | P0 | Flash Attention / Transformer Related | Dtype/Precision Issue - value cannot be converted to at::BFloat16 without overfl |  | e2e |
| [1505](https://github.com/intel/torch-xpu-ops/issues/1505) | [ARC-WSL-Ubuntu24.04] 15 Timm model | xuhancn, Stonepia | xuhancn, Stonepia | P0 | Dtype / Precision Related |  |  | e2e |
| [1519](https://github.com/intel/torch-xpu-ops/issues/1519) | [PVC][PT2.7][ABI=1][Torch-xpu-ops U | None | None | P2 | Others | Backend/Device Issue - coredump related to XPU operations in Torch-xpu-ops UT |  | ut |
| [1778](https://github.com/intel/torch-xpu-ops/issues/1778) | [Infra] Show known issues for accur | mengfei25 | mengfei25 | P1 | Dtype / Precision Related | Skip/No Test Exists - no test or error details provided |  | e2e |
| [1818](https://github.com/intel/torch-xpu-ops/issues/1818) | [BMG-Windows][PT2.8]Torch-xpu-ops U | kdrozd-dev | kdrozd-dev | P2 | Dtype / Precision Related | Backend/Device Issue - Accuracy issue on XPU for Torch-xpu-ops UT |  | ut |
| [1866](https://github.com/intel/torch-xpu-ops/issues/1866) | [release 2.8]Torchbench vision_mask | BartoszKokoszko | BartoszKokoszko | P0 | Dtype / Precision Related | Dtype/Precision Issue - amp_fp16 inference accuracy failure |  | e2e |
| [1894](https://github.com/intel/torch-xpu-ops/issues/1894) | [Linux][PT2E] performance test got  | jenniew | jenniew | P1 | TorchAO | precision-related failure in performance test |  | e2e |
| [1963](https://github.com/intel/torch-xpu-ops/issues/1963) | [upstream_ut] MetadataMismatchError | pbielak | pbielak | P2 | Others |  | [PR](https://github.com/pytorch/pytorch/pull/178277, https://github.com/pytorch/pytorch/pull/175965) | ut |
| [2055](https://github.com/intel/torch-xpu-ops/issues/2055) | New huggingface LLM models issues | jianyizh, mengfei25 | jianyizh, mengfei25 | P0 | Others | Others - insufficient information to determine root cause |  | e2e |
| [2058](https://github.com/intel/torch-xpu-ops/issues/2058) | [release/2.9] llama_v2_7b_16h amp i | jianyizh | jianyizh | P0 | Flash Attention / Transformer Related | device-specific backend problem. |  | e2e |
| [2128](https://github.com/intel/torch-xpu-ops/issues/2128) | [2.9][BMG-Windows][Torchbench] spee | chuanqi129 | chuanqi129 | P0 | Dtype / Precision Related | Backend/Device Issue - Exception Code 0xC0000005 indicates an access violation l |  | ut |
| [2132](https://github.com/intel/torch-xpu-ops/issues/2132) | [2.9][BMG-Windows][Torch-xpu-ops UT | pbielak | pbielak | P2 | Inductor / Compilation Related |  |  | ut |
| [2202](https://github.com/intel/torch-xpu-ops/issues/2202) | [TorchAO][BMG] The RTN performance  | Stonepia | Stonepia | P0 | TorchAO | Timeout/Performance Issue - RTN performance regression in next-token latency for |  | ut |
| [2214](https://github.com/intel/torch-xpu-ops/issues/2214) | test/test_sparse.py::TestSparseAnyX | jenniew | jenniew | P2 | Distributed |  |  | ut |
| [2234](https://github.com/intel/torch-xpu-ops/issues/2234) | [upstream_ut] AssertionError: Runti | Silv3S | Silv3S | P2 | Distributed |  |  | ut |
| [2248](https://github.com/intel/torch-xpu-ops/issues/2248) | [upstream_ut] test_cow failures | gplutop7 | gplutop7 | P2 | Distributed |  |  | ut |
| [2251](https://github.com/intel/torch-xpu-ops/issues/2251) | [upstream_ut] test_fake_autocase go | astachowiczhabana | astachowiczhabana | P2 | Dtype / Precision Related |  |  | ut |
| [2255](https://github.com/intel/torch-xpu-ops/issues/2255) | [upstream_ut] RuntimeError: Long is | daisyden | daisyden | P2 | Flash Attention / Transformer Related |  |  | ut |
| [2283](https://github.com/intel/torch-xpu-ops/issues/2283) | [upstream_ut] sparse._sampled_addmm | jenniew | jenniew | P1 | Distributed |  |  | ut |
| [2287](https://github.com/intel/torch-xpu-ops/issues/2287) | [upstream_ut] test_python_ref issue | yucai-intel | yucai-intel | P2 | Others |  |  | ut |
| [2295](https://github.com/intel/torch-xpu-ops/issues/2295) | [upstream_ut][xpu][test]nn/test_emb | yucai-intel | yucai-intel | P2 | Dtype / Precision Related |  |  | ut |
| [2301](https://github.com/intel/torch-xpu-ops/issues/2301) | [upstream_ut] dtypes not align with | daisyden | daisyden | P2 | Distributed |  |  | ut |
| [2329](https://github.com/intel/torch-xpu-ops/issues/2329) | [upstream_ut] feature missing: get_ | etaf | etaf | P2 | Others |  |  | ut |
| [2359](https://github.com/intel/torch-xpu-ops/issues/2359) | [upstream_ut] GradcheckError: Backw | BBBela | BBBela | P2 | Distributed |  |  | ut |
| [2512](https://github.com/intel/torch-xpu-ops/issues/2512) | [upstream_ut]  RuntimeError: _histc | chunhuanMeng | libohao1201 | P2 | Dtype / Precision Related | Backend/Device Issue - _histc_xpu lacks deterministic implementation on XPU back |  | ut |
| [2519](https://github.com/intel/torch-xpu-ops/issues/2519) | [upstream_ut]  TypeError: map2_ is  | Silv3S | libohao1201 | P2 | Others | Backend/Device Issue - map2_ is only implemented on CPU tensors, indicating lack |  | ut |
| [2554](https://github.com/intel/torch-xpu-ops/issues/2554) | [upstream_ut]  AssertionError: Asse | daisyden | daisyden | P2 | PT2E |  |  | ut |
| [2570](https://github.com/intel/torch-xpu-ops/issues/2570) | crash in sdpa. | LuFinch | sywangyi | P0 | Flash Attention / Transformer Related | Others - insufficient information to determine root cause |  | ut |
| [2578](https://github.com/intel/torch-xpu-ops/issues/2578) | [TorchAO][UT] test/quantization/tes | Stonepia | Stonepia | P0 | TorchAO |  |  | build |
| [2598](https://github.com/intel/torch-xpu-ops/issues/2598) | [TorchAO][BMG]The first token laten | Stonepia | Stonepia | P2 | TorchAO | Timeout/Performance Issue - First token latency drops significantly with change  |  | ut |
| [2609](https://github.com/intel/torch-xpu-ops/issues/2609) | [upstream_ut]  torch._inductor.exc. | daisyden | daisyden | P2 | PT2E |  | [PR](https://github.com/pytorch/pytorch/pull/171154) | ut |
| [2620](https://github.com/intel/torch-xpu-ops/issues/2620) | [upstream_ut]  AssertionError: dtyp | daisyden | daisyden | P2 | TorchAO |  |  | ut |
| [2650](https://github.com/intel/torch-xpu-ops/issues/2650) | [OOB Performance] The performance i | jianyizh | jianyizh | P0 | Inductor / Compilation Related | Inductor/Compilation Issue - Performance impact caused by TORCHINDUCTOR_ONLINE_S |  | e2e |
| [2654](https://github.com/intel/torch-xpu-ops/issues/2654) | [BMG][OOB] t5 inference performance | jianyizh | jianyizh | P0 | Dtype / Precision Related | Timeout/Performance Issue - inference performance drop |  | e2e |
| [2655](https://github.com/intel/torch-xpu-ops/issues/2655) | [BMG][OOB] hf_Reformer performance  | jianyizh | jianyizh | P0 | Others | Timeout/Performance Issue - hf_Reformer performance drop reported. |  | e2e |
| [2656](https://github.com/intel/torch-xpu-ops/issues/2656) | [release/2.10] models got fail_accu | None | None | P0 | Dtype / Precision Related | Backend/Device Issue - failure occurs specifically on BMG WSL2 XPU backend |  | ut |
| [2660](https://github.com/intel/torch-xpu-ops/issues/2660) | [release/2.10][Windows][BMG] New fa | tadkrawiec | tadkrawiec | P2 | Others | Others - insufficient information to determine root cause |  | ut |
| [2662](https://github.com/intel/torch-xpu-ops/issues/2662) | [release/2.10][Windows][BMG] New fa | tadkrawiec, kdrozd-dev | tadkrawiec, kdrozd-dev | P2 | Others | Backend/Device Issue - XPU related failure in test cases on Windows with BMG |  | ut |
| [2663](https://github.com/intel/torch-xpu-ops/issues/2663) | test_sparse_semi_structured.py gaps | None | None | P2 | Sparse Operations Related |  |  | ut |
| [2670](https://github.com/intel/torch-xpu-ops/issues/2670) | [upstream_ut]  RuntimeError: could  | tszulist-hbn | tszulist-hbn | P2 | Distributed |  |  | ut |
| [2693](https://github.com/intel/torch-xpu-ops/issues/2693) | Title: [upstream_ut]  AssertionErro | hoshibara | hoshibara | P2 | Distributed |  |  | ut |
| [2696](https://github.com/intel/torch-xpu-ops/issues/2696) | Title: [upstream_ut]  RuntimeError: | chunhuanMeng | chunhuanMeng | P2 | Flash Attention / Transformer Related |  |  | e2e |
| [2697](https://github.com/intel/torch-xpu-ops/issues/2697) | Title: [upstream_ut]  RuntimeError: | chunhuanMeng | chunhuanMeng | P2 | Flash Attention / Transformer Related |  |  | e2e |
| [2698](https://github.com/intel/torch-xpu-ops/issues/2698) | Title: [upstream_ut]  RuntimeError: | chunhuanMeng, LuFinch | chunhuanMeng, LuFinch | P2 | PT2E |  |  | ut |
| [2704](https://github.com/intel/torch-xpu-ops/issues/2704) | Title: [upstream_ut]  AssertionErro | kdrozd-dev | kdrozd-dev | P2 | Flash Attention / Transformer Related |  | [PR](https://github.com/pytorch/pytorch/pull/177636) | ut |
| [2712](https://github.com/intel/torch-xpu-ops/issues/2712) | [upstream_ut]  RuntimeError: Cannot | tszulist-hbn | tszulist-hbn | P2 | Others |  |  | ut |
| [2714](https://github.com/intel/torch-xpu-ops/issues/2714) | [upstream_ut]  AssertionError: Obje | Silv3S | Silv3S | P2 | Distributed |  |  | ut |
| [2715](https://github.com/intel/torch-xpu-ops/issues/2715) | [upstream_ut]  torch._dynamo.exc.Un | CuiYifeng | CuiYifeng | P2 | PT2E |  |  | ut |
| [2722](https://github.com/intel/torch-xpu-ops/issues/2722) | [Bug Skip]: NotImplementedError: Co | Silv3S | CuiYifeng | P2 | TorchAO | Backend/Device Issue - aten::flip not implemented for QuantizedXPU backend |  | ut |
| [2742](https://github.com/intel/torch-xpu-ops/issues/2742) | [Linux][PT2E] hf_Roberta_base model | chunhuanMeng | chunhuanMeng | P0 | Flash Attention / Transformer Related | Timeout/Performance Issue - hf_Roberta_base model performance failed for both AS |  | e2e |
| [2798](https://github.com/intel/torch-xpu-ops/issues/2798) | Test case  test/test_dlpack.py::Tes | None | None | P2 | Others |  |  | ut |
| [2800](https://github.com/intel/torch-xpu-ops/issues/2800) | AttributeError: 'torch._C._XpuDevic | guangyey | guangyey | P2 | Flash Attention / Transformer Related |  |  | ut |
| [2802](https://github.com/intel/torch-xpu-ops/issues/2802) | Three aten._scaled_dot_product_flas | LuFinch | LuFinch | P2 | Flash Attention / Transformer Related |  |  | ut |
| [2806](https://github.com/intel/torch-xpu-ops/issues/2806) | CompiledAOTI need XPU support | daisyden | daisyden | P2 | PT2E |  |  | ut |
| [2810](https://github.com/intel/torch-xpu-ops/issues/2810) | AssertionError: Object comparison f | daisyden | daisyden | P2 | Dtype / Precision Related |  |  | ut |
| [2888](https://github.com/intel/torch-xpu-ops/issues/2888) | torch._inductor.exc.InductorError:  | Stonepia | Stonepia | P2 | Inductor / Compilation Related |  |  | ut |
| [2891](https://github.com/intel/torch-xpu-ops/issues/2891) | RuntimeError: Expected to find "(26 | chunhuanMeng | chunhuanMeng | P2 | Others |  |  | e2e |
| [2907](https://github.com/intel/torch-xpu-ops/issues/2907) | [release/2.11] Models performance r | xuhancn | xuhancn | P0 | Others | Timeout/Performance Issue - models performance regression in testcases |  | ut |
| [2908](https://github.com/intel/torch-xpu-ops/issues/2908) | [release/2.11] Model fail_accuracy  | xuhancn | xuhancn | P1 | Dtype / Precision Related | Others - insufficient information to determine root cause |  | e2e |
| [2912](https://github.com/intel/torch-xpu-ops/issues/2912) | [release/2.11] UT extended 220 new  | None | None | P2 | Others | Others - insufficient information to determine root cause |  | ut |
| [2918](https://github.com/intel/torch-xpu-ops/issues/2918) | [XPU][upstream_ut][COW] Skip non-su | gplutop7 | gplutop7 | P2 | Others |  | [PR](https://github.com/pytorch/pytorch/pull/174670) | ut |
| [2919](https://github.com/intel/torch-xpu-ops/issues/2919) | [XPU][upstream_ut][COW] Fix materia | gplutop7 | gplutop7 | P2 | Others |  |  | ut |
| [2922](https://github.com/intel/torch-xpu-ops/issues/2922) | [release/2.11] UT inductor Assertio | tadkrawiec | tadkrawiec | P2 | Inductor / Compilation Related | Backend/Device Issue - pass_fds not supported on Windows |  | ut |
| [2924](https://github.com/intel/torch-xpu-ops/issues/2924) | [release/2.11] xcit_large_24_p8_224 | jianyizh, mengfei25 | jianyizh, mengfei25 | P1 | Dtype / Precision Related | Dtype/Precision Issue - amp_bf16 training accuracy failure |  | e2e |
| [2928](https://github.com/intel/torch-xpu-ops/issues/2928) | [release/2.11] pyhpc_turbulent_kine | jianyizh | jianyizh | P1 | Dtype / Precision Related | Dtype/Precision Issue - fp32 inference accuracy failure |  | e2e |
| [2929](https://github.com/intel/torch-xpu-ops/issues/2929) | [release/2.11] volo_d1_224 inferenc | jianyizh | jianyizh | P1 | Dtype / Precision Related | Backend/Device Issue - fail_to_run on XPU for volo_d1_224 inference |  | e2e |
| [2930](https://github.com/intel/torch-xpu-ops/issues/2930) | [release/2.11] UT skip test_binary_ | None | None | P2 | Others | Skip/No Test Exists - test is skipped due to RuntimeError |  | ut |
| [2932](https://github.com/intel/torch-xpu-ops/issues/2932) | [release/2.11] jx_nest_base and vol | jianyizh | jianyizh | P2 | Dtype / Precision Related | Failure - encountered AssertionError during training |  | e2e |
| [2935](https://github.com/intel/torch-xpu-ops/issues/2935) | [release/2.11][inductor] huggingfac | jianyizh | jianyizh | P0 | Inductor / Compilation Related | Inductor/Compilation Issue - performance regression in XLNetLMHeadModel with amp |  | e2e |
| [2938](https://github.com/intel/torch-xpu-ops/issues/2938) | [release/2.11] basic_gnn_gin and ba | jianyizh | jianyizh | P2 | Dtype / Precision Related | Timeout/Performance Issue - inference fp32 performance dropped ~25% |  | e2e |
| [2939](https://github.com/intel/torch-xpu-ops/issues/2939) | [release/2.11] gmlp_s16_224 inferen | jianyizh | jianyizh | P2 | Flash Attention / Transformer Related | Timeout/Performance Issue - inference amp performance dropped ~15% |  | e2e |
| [2940](https://github.com/intel/torch-xpu-ops/issues/2940) | [release/2.11] Models performance d | jianyizh, LuFinch | jianyizh, LuFinch | P0 | Others | Timeout/Performance Issue - Models performance dropped ~10% - 15% |  | e2e |
| [2952](https://github.com/intel/torch-xpu-ops/issues/2952) | [release/2.11][wsl] timm_models_acc | weishi-deng | weishi-deng | P0 | Dtype / Precision Related | Dtype/Precision Issue - bfloat16 accuracy failure in model training |  | ut |
| [2958](https://github.com/intel/torch-xpu-ops/issues/2958) | AssertionError of test_dtensor_basi | daisyden | daisyden | P2 | Inductor / Compilation Related |  |  | ut |
| [2960](https://github.com/intel/torch-xpu-ops/issues/2960) | [release/2.11] timm_models_xcit_lar | None | None | P0 | Dtype / Precision Related | Dtype/Precision Issue - float16 training accuracy test failure |  | ut |
| [2979](https://github.com/intel/torch-xpu-ops/issues/2979) | eca_halonext26ts got RuntimeError:  | None | None | P0 | Others | Backend/Device Issue - ZE_RESULT_ERROR_MODULE_BUILD_FAILURE indicates a problem  |  | e2e |
| [2984](https://github.com/intel/torch-xpu-ops/issues/2984) | [release/2.11] sebotnet33ts_256 fp3 | jianyizh, weishi-deng | jianyizh, weishi-deng | P1 | Dtype / Precision Related | Backend/Device Issue - XPU specific failure during fp32 training accuracy check |  | e2e |
| [2997](https://github.com/intel/torch-xpu-ops/issues/2997) | AssertionError of test_linear_and_c | etaf | etaf | P2 | Flash Attention / Transformer Related |  |  | ut |
| [2999](https://github.com/intel/torch-xpu-ops/issues/2999) | KeyError: 'eager_numerics.use_pytor | daisyden | daisyden | P2 | Others |  |  | ut |
| [3004](https://github.com/intel/torch-xpu-ops/issues/3004) | TypeError: _xpu_recordMemoryHistory | guangyey | guangyey | P2 | Others |  |  | ut |
| [3006](https://github.com/intel/torch-xpu-ops/issues/3006) | AssertionError: '.to(tl.float16)' u | CuiYifeng | CuiYifeng | P2 | PT2E |  |  | e2e |
| [3041](https://github.com/intel/torch-xpu-ops/issues/3041) | AssertionError: Expected len(flat_d | Silv3S | Silv3S | P2 | Dtype / Precision Related |  |  | ut |
| [3077](https://github.com/intel/torch-xpu-ops/issues/3077) | [Bug Skip] test_dlpack.py::TestTorc | AKloniecki | AKloniecki | P2 | Others |  |  | ut |
| [3094](https://github.com/intel/torch-xpu-ops/issues/3094) | XPUGraph tree support | None | None | P2 | Inductor / Compilation Related |  |  | ut |
| [3095](https://github.com/intel/torch-xpu-ops/issues/3095) | cutlass support blocks some unit te | None | None | P2 | Others |  |  | ut |
| [3126](https://github.com/intel/torch-xpu-ops/issues/3126) | [upstream_ut]  Two NestedTensor iss | LuFinch | LuFinch | P2 | Flash Attention / Transformer Related |  |  | ut |
| [3127](https://github.com/intel/torch-xpu-ops/issues/3127) | [upstream_ut]  AssertionError: Asse | LuFinch | LuFinch | P2 | Flash Attention / Transformer Related |  |  | ut |
| [3128](https://github.com/intel/torch-xpu-ops/issues/3128) | [upstream_ut]  AssertionError: Runt | LuFinch | LuFinch | P2 | Distributed |  |  | ut |
| [3129](https://github.com/intel/torch-xpu-ops/issues/3129) | [upstream_ut]  AssertionError: User | LuFinch | LuFinch | P2 | Flash Attention / Transformer Related |  |  | ut |
| [3130](https://github.com/intel/torch-xpu-ops/issues/3130) | [upstream_ut]  AssertionError: tens | LuFinch | LuFinch | P2 | Flash Attention / Transformer Related |  |  | ut |
| [3131](https://github.com/intel/torch-xpu-ops/issues/3131) | [upstream_ut]  NotImplementedError: | chunhuanMeng | chunhuanMeng | P2 | Flash Attention / Transformer Related |  |  | ut |
| [3132](https://github.com/intel/torch-xpu-ops/issues/3132) | [upstream_ut]  transfomers test rep | LuFinch | LuFinch | P2 | Flash Attention / Transformer Related |  |  | ut |
| [3133](https://github.com/intel/torch-xpu-ops/issues/3133) | [upstream_ut]  RuntimeError: scaled | LuFinch | LuFinch | P2 | Flash Attention / Transformer Related |  |  | ut |
| [3137](https://github.com/intel/torch-xpu-ops/issues/3137) | [upstream_ut]  RuntimeError: expect | LuFinch | LuFinch | P2 | Flash Attention / Transformer Related |  |  | ut |
| [3140](https://github.com/intel/torch-xpu-ops/issues/3140) | [upstream_ut]  RuntimeError: FlashA | LuFinch | LuFinch | P2 | Flash Attention / Transformer Related |  |  | ut |
| [3141](https://github.com/intel/torch-xpu-ops/issues/3141) | [upstream_ut]  RuntimeError: FlashA | LuFinch | LuFinch | P2 | Flash Attention / Transformer Related |  |  | ut |
| [3142](https://github.com/intel/torch-xpu-ops/issues/3142) | [upstream_ut]  RuntimeError: The sy | LuFinch | LuFinch | P2 | Flash Attention / Transformer Related |  |  | ut |
| [3143](https://github.com/intel/torch-xpu-ops/issues/3143) | NotImplementedError: The operator ' | LuFinch | LuFinch | P2 | Flash Attention / Transformer Related |  |  | ut |
| [3148](https://github.com/intel/torch-xpu-ops/issues/3148) | [Triton] Huggingface openai/whisper | None | None | P0 | Inductor / Compilation Related | Backend/Device Issue - XPU specific failure with Huggingface model accuracy |  | e2e |
| [3151](https://github.com/intel/torch-xpu-ops/issues/3151) | [Triton] Timm_models  rexnet_100 /  | None | None | P0 | Inductor / Compilation Related | Backend/Device Issue - XPU specific failure with Timm models in Triton. |  | e2e |
| [3163](https://github.com/intel/torch-xpu-ops/issues/3163) | [Bug Skip]: Object comparison faile | chunhuanMeng | chunhuanMeng | P2 | Distributed |  |  | ut |
| [3165](https://github.com/intel/torch-xpu-ops/issues/3165) | test_sparse_csr_xpu.py::TestSparseC | None | None | P2 | Sparse Operations Related |  |  | ut |
| [3167](https://github.com/intel/torch-xpu-ops/issues/3167) | NotImplementedError: Could not run  | tszulist-hbn | tszulist-hbn | P1 | Sparse Operations Related |  |  | ut |
| [3168](https://github.com/intel/torch-xpu-ops/issues/3168) | NotImplementedError: Could not run  | jenniew | jenniew | P1 | Sparse Operations Related |  |  | ut |
| [3169](https://github.com/intel/torch-xpu-ops/issues/3169) | NotImplementedError: Could not run  | jkosnox | jkosnox | P2 | Sparse Operations Related |  |  | ut |
| [3170](https://github.com/intel/torch-xpu-ops/issues/3170) | Unskip test_bmm_windows_error_xpu_f | jenniew | jenniew | P2 | Others |  |  | ut |
| [3187](https://github.com/intel/torch-xpu-ops/issues/3187) | PyTorch XPU gpu_cpp_wrapper fails w | CuiYifeng | CuiYifeng | P2 | Inductor / Compilation Related |  |  | ut |
| [3191](https://github.com/intel/torch-xpu-ops/issues/3191) | torch._inductor.exc.InductorError:  | EikanWang, Copilot | EikanWang, Copilot | P2 | Inductor / Compilation Related | Inductor/Compilation Issue - Assertion failure due to conflicting fallback and d |  | e2e |
| [3229](https://github.com/intel/torch-xpu-ops/issues/3229) | RuntimeError: No viable backend for | None | None | P2 | Flash Attention / Transformer Related |  |  | ut |
| [3238](https://github.com/intel/torch-xpu-ops/issues/3238) | The supported dtypes of _refs.stft  | CuiYifeng | CuiYifeng | P2 | Dtype / Precision Related |  |  | ut |
| [3247](https://github.com/intel/torch-xpu-ops/issues/3247) | NotImplementedError: "dot_xpu_mkl"  | Silv3S | Silv3S | P2 | Others |  |  | ut |
| [3255](https://github.com/intel/torch-xpu-ops/issues/3255) | [Linux][PT2E][Regression] Some perf | None | None | P0 | TorchAO | Timeout/Performance Issue - performance tests failed due to regression in execut |  | e2e |
| [3257](https://github.com/intel/torch-xpu-ops/issues/3257) | [Linux][E2E][Regression] Huggingfac | None | None | P0 | Others | Backend/Device Issue - Huggingface test models failed to run on XPU, indicating  |  | e2e |

#### Revisit the PR as Case Failed {#revisit-the-pr-as-case-failed}

| ID | Title | Owner | Owner Transferred | Priority | Category | Root Cause | PR | Test Module |
|---|-------|-------|-------------------|---------|----------|-----------|-----|-------------|
| [3246](https://github.com/intel/torch-xpu-ops/issues/3246) | AssertionError: Booleans mismatch:  | BartoszKokoszko | BartoszKokoszko | P2 | Distributed |  | [PR](https://github.com/intel/torch-xpu-ops/pull/3249) | ut |

---

## 5. By Category {#5-by-category}

#### Distributed (#distributed)

| ID | Title | Status | Owner | Priority | Root Cause | PR | Labels | Module | Test Module |
|---|-------|--------|-------|---------|-----------|-----|--------|--------|-------------|
| [1059](https://github.com/intel/torch-xpu-ops/issues/1059) | SYCL RT: Using recommended sho | open | CuiYifeng, jianyizh | P2 | Backend/Device Issue - related to SYCL RT and kernel-specific max work group siz |  | dependency component: oneAPI | aten_ops | ut |
| [1547](https://github.com/intel/torch-xpu-ops/issues/1547) | [distributed] NotImplementedEr | open | Chao1Han | P2 | Backend/Device Issue - XPU device does not support the 'symm_mem::fused_matmul_r |  | module: distributed, dependency component: oneAPI | distributed | ut |
| [1548](https://github.com/intel/torch-xpu-ops/issues/1548) | [distributed] AssertionError:  | open | Chao1Han | P2 | Failure - 'fused_all_gather_matmul' not found in AOT ID list |  | module: distributed, dependency component: oneAPI | distributed | ut |
| [1549](https://github.com/intel/torch-xpu-ops/issues/1549) | [distributed] AssertionError:  | open | Chao1Han | P2 | Distributed/Gloo Issue - 'fused_all_gather_scaled_matmul' not found in graph dur |  | module: distributed, dependency component: oneAPI | distributed | ut |
| [1551](https://github.com/intel/torch-xpu-ops/issues/1551) | [distributed] NotImplementedEr | open | Chao1Han | P2 | Backend/Device Issue - XPU device does not support the 'symm_mem::fused_scaled_m |  | module: distributed, dependency component: oneAPI | distributed | ut |
| [1555](https://github.com/intel/torch-xpu-ops/issues/1555) | [distributed] RuntimeError: at | open | chuanqi129 | P2 | Distributed/Gloo Issue - mixed torch.Tensor and DTensor in distributed operation |  | module: distributed, dependency component: oneDNN | distributed | ut |
| [1556](https://github.com/intel/torch-xpu-ops/issues/1556) | [distributed] NotImplementedEr | open | pkourdis | P2 | Distributed/Gloo Issue - Operator lacks a sharding strategy in distributed conte |  | module: distributed, dependency component: oneDNN | distributed | ut |
| [1571](https://github.com/intel/torch-xpu-ops/issues/1571) | [distributed] ValueError: Cann | open | zhangxiaoli73 | P2 | Distributed/Gloo Issue - ReduceOp.PREMUL_SUM not supported with XCCL |  | module: distributed | distributed | ut |
| [1661](https://github.com/intel/torch-xpu-ops/issues/1661) | [distributed] Accuracy gap in  | open | githubsgi | P2 | Distributed/Gloo Issue - Accuracy gap in _composable/fsdp on Xelink suggests a d |  | bug, module: distributed | distributed | ut |
| [1727](https://github.com/intel/torch-xpu-ops/issues/1727) | [distributed] AttributeError:  | open | guangyey | P2 | Backend/Device Issue - missing attribute '_sleep' in torch.xpu module |  | module: distributed, dependency component: oneAPI | distributed | ut |
| [1893](https://github.com/intel/torch-xpu-ops/issues/1893) | [upstream_ut] oneDNN accuracy  | open | chunhuanMeng | P2 |  |  | skipped, ut_upstream | aten_ops | ut |
| [1951](https://github.com/intel/torch-xpu-ops/issues/1951) | Functionality issues in TestCo | open | AKloniecki | P2 |  |  | module: ut, skipped, ut_upstream | aten_ops | ut |
| [1973](https://github.com/intel/torch-xpu-ops/issues/1973) | AssertionError: Scalars or Ten | open | gplutop7 | P2 | Failure - Tensor values not equal or close in test_depthwise_conv_64bit_indexing_xpu |  | hw: PVC, module: ut, skipped, bug_fix_stage4 | aten_ops | ut |
| [2004](https://github.com/intel/torch-xpu-ops/issues/2004) | [distributed][shared_tensor] t | open | libohao1201 | P2 | Memory/Shared Memory Issue - error originated from shared memory connection in t |  | bug, module: distributed | distributed | ut |
| [2024](https://github.com/intel/torch-xpu-ops/issues/2024) | AssertionError: Torch not comp | open | daisyden | P2 |  |  | module: ut, skipped | aten_ops | ut |
| [2113](https://github.com/intel/torch-xpu-ops/issues/2113) | Update example for Distributed | open | songhappy | P2 | Distributed/Gloo Issue - related to Distributed Data Parallel update example |  | module: distributed | distributed | ut |
| [2163](https://github.com/intel/torch-xpu-ops/issues/2163) | 3 distributed UT cases need to | open | githubsgi | P2 | Distributed/Gloo Issue - distributed UT cases need support from sac_estimator.py |  | module: distributed | distributed | ut |
| [2164](https://github.com/intel/torch-xpu-ops/issues/2164) | skip test_no_cuda_monkeypatch  | open | daisyden | P2 |  |  | wontfix, skipped | aten_ops | ut |
| [2165](https://github.com/intel/torch-xpu-ops/issues/2165) | [distributed] test_device_mesh | open | jemitche1 | P2 | Failure - test_flatten_mesh_3d encountered an assertion error |  | bug, module: distributed | distributed | ut |
| [2169](https://github.com/intel/torch-xpu-ops/issues/2169) | Frame size comparison failed i | open | guangyey | P2 | Failure - Scalars are not equal in test comparison |  | skipped | aten_ops | ut |
| [2182](https://github.com/intel/torch-xpu-ops/issues/2182) | test_transform_bias_rescale_qk | open | PawelSwider2000 | P2 | Failure - Scalars are not equal in test assertion |  | Accuracy, module: ut, skipped | aten_ops | ut |
| [2214](https://github.com/intel/torch-xpu-ops/issues/2214) | test/test_sparse.py::TestSpars | open | jenniew | P2 |  |  | skipped, ut_upstream | aten_ops | ut |
| [2229](https://github.com/intel/torch-xpu-ops/issues/2229) | test/test_sparse_csr.py::TestS | open | jenniew | P2 | Backend/Device Issue - device mismatch between crow_indices and col_indices on X |  | skipped | aten_ops | ut |
| [2230](https://github.com/intel/torch-xpu-ops/issues/2230) | test_sparse_csr.py::TestSparse | open | None | P1 | Backend/Device Issue - inputs are not on the same GPU device |  | skipped | aten_ops | ut |
| [2234](https://github.com/intel/torch-xpu-ops/issues/2234) | [upstream_ut] AssertionError:  | open | Silv3S | P2 |  |  | module: ut, skipped, ut_upstream | aten_ops | ut |
| [2235](https://github.com/intel/torch-xpu-ops/issues/2235) | test/test_sparse_csr.py::TestS | open | None | P2 | Backend/Device Issue - unexpected warning from non-optimal triton kernel paramet |  | skipped | aten_ops | ut |
| [2238](https://github.com/intel/torch-xpu-ops/issues/2238) | Exception: Tensor-likes are no | open | BBBela | P2 |  | [PR](https://github.com/intel/torch-xpu-ops/pull/2886) | skipped, bug_fix_stage3 | aten_ops | ut |
| [2239](https://github.com/intel/torch-xpu-ops/issues/2239) | Exception: could not create a  | open | wpietka | P2 | DNNL/OneDNN Issue - could not create a primitive descriptor for deconvolution fo |  | skipped, bug_fix_stage5 | aten_ops | ut |
| [2240](https://github.com/intel/torch-xpu-ops/issues/2240) | RuntimeError: Trying to set a  | open | gplutop7 | P2 | Error - forward gradient size mismatch with original Tensor |  | skipped, bug_fix_stage3 | aten_ops | ut |
| [2244](https://github.com/intel/torch-xpu-ops/issues/2244) | test/test_sparse_csr.py::TestS | open | jenniew | P2 | Error - empty_sparse_compressed expects non-block layout but received SparseBsr |  | module: ut, skipped | aten_ops | ut |
| [2245](https://github.com/intel/torch-xpu-ops/issues/2245) | oneDNN matmul received incorre | open | CuiYifeng | P2 | DNNL/OneDNN Issue - oneDNN matmul primitive descriptor creation failure |  | module: ut, skipped | aten_ops | ut |
| [2246](https://github.com/intel/torch-xpu-ops/issues/2246) | torch/sparse/_triton_ops*.py n | open | None | P1 | Backend/Device Issue - Torch not compiled with CUDA enabled for Intel GPU suppor |  | skipped | unknown | ut |
| [2248](https://github.com/intel/torch-xpu-ops/issues/2248) | [upstream_ut] test_cow failure | open | gplutop7 | P2 |  |  | skipped, bug_fix_stage3, ut_upstream | aten_ops | ut |
| [2257](https://github.com/intel/torch-xpu-ops/issues/2257) | Accuracy failures in test/xpu/ | open | pbielak | P1 |  | [PR](https://github.com/intel/torch-xpu-ops/pull/2808) | skipped, bug_fix_stage4 | aten_ops | ut |
| [2270](https://github.com/intel/torch-xpu-ops/issues/2270) | Backend Compatibility Error in | open | LuFinch | P2 | Flash Attention/Specific Ops Issue - test involves aten._flash_attention_forward |  | module: ut, skipped | aten_ops | ut |
| [2283](https://github.com/intel/torch-xpu-ops/issues/2283) | [upstream_ut] sparse._sampled_ | open | jenniew | P1 |  |  | skipped, ut_upstream | aten_ops | ut |
| [2301](https://github.com/intel/torch-xpu-ops/issues/2301) | [upstream_ut] dtypes not align | open | daisyden | P2 |  |  | skipped, ut_upstream | aten_ops | ut |
| [2340](https://github.com/intel/torch-xpu-ops/issues/2340) | [distributed][_tools] Assertio | open | githubsgi | P2 | Failure - Roofline estimation requires CUDA capabilities assertion failed |  | bug, duplicate, module: distributed | distributed | ut |
| [2359](https://github.com/intel/torch-xpu-ops/issues/2359) | [upstream_ut] GradcheckError:  | open | BBBela | P2 |  |  | skipped, ut_upstream | aten_ops | ut |
| [2389](https://github.com/intel/torch-xpu-ops/issues/2389) | [Bug Skip]: RuntimeError: Data | open | PatrykWilczewski | P2 | Error - Data corruption detected during test execution |  | skipped, bug_fix_stage4, random | aten_ops | ut |
| [2404](https://github.com/intel/torch-xpu-ops/issues/2404) | [distributed][checkpoint] Asse | open | None | P2 | Failure - Booleans mismatch assertion error |  | bug | aten_ops | ut |
| [2446](https://github.com/intel/torch-xpu-ops/issues/2446) | [Bug Skip]: AssertionError: "S | open | None | P2 | Failure - mismatch between expected and actual error messages |  | skipped, random | aten_ops | ut |
| [2529](https://github.com/intel/torch-xpu-ops/issues/2529) | [upstream_ut]  AssertionError: | open | Silv3S | P2 | Failure - test assertion failed with False is not true |  | skipped, port_from_skiplist | aten_ops | ut |
| [2530](https://github.com/intel/torch-xpu-ops/issues/2530) | Title: [upstream_ut]  Assertio | open | PatrykWilczewski | P2 | Failure - RuntimeError not raised as expected in test |  | skipped, bug_fix_stage5, port_from_skiplist | aten_ops | ut |
| [2531](https://github.com/intel/torch-xpu-ops/issues/2531) | [upstream_ut]  AssertionError: | open | daisyden | P2 |  |  | skipped, port_from_skiplist | unknown | ut |
| [2532](https://github.com/intel/torch-xpu-ops/issues/2532) | Title: [upstream_ut]  Assertio | open | yucai-intel | P2 | Failure - wrong number of dimensions for int4 conversion op |  | skipped, port_from_skiplist | aten_ops | ut |
| [2533](https://github.com/intel/torch-xpu-ops/issues/2533) | Title: [upstream_ut]  Attribut | open | astachowiczhabana | P2 | Mismatch - Test method 'test_qsoftmax' is missing in 'TestQuantizedOpsXPU' class. |  | skipped, port_from_skiplist | aten_ops | ut |
| [2535](https://github.com/intel/torch-xpu-ops/issues/2535) | Title: [upstream_ut]  Attribut | open | Silv3S | P2 | Backend/Device Issue - missing XPU-specific attribute in torch._C for tunableop  |  | skipped, port_from_skiplist | aten_ops | ut |
| [2536](https://github.com/intel/torch-xpu-ops/issues/2536) | Title: [upstream_ut]  Attribut | open | daisyden | P2 | Backend/Device Issue - missing attribute '_scatter' in torch._C for XPU device |  | skipped, port_from_skiplist, not_target | aten_ops | ut |
| [2538](https://github.com/intel/torch-xpu-ops/issues/2538) | Title: [upstream_ut]  RuntimeE | open | Silv3S | P2 | DNNL/OneDNN Issue - Float8_e4m3fnuz is not supported in oneDNN. |  | skipped, port_from_skiplist | aten_ops | ut |
| [2539](https://github.com/intel/torch-xpu-ops/issues/2539) | Title: [upstream_ut]  RuntimeE | open | None | P2 | Backend/Device Issue - Attempt to instantiate dummy base class CUDAGraph on XPU |  | skipped, port_from_skiplist | aten_ops | ut |
| [2541](https://github.com/intel/torch-xpu-ops/issues/2541) | Title: [upstream_ut]  RuntimeE | open | yucai-intel | P2 | DNNL/OneDNN Issue - could not construct a memory descriptor using strides |  | skipped, port_from_skiplist | aten_ops | ut |
| [2611](https://github.com/intel/torch-xpu-ops/issues/2611) | [upstream_ut]  AssertionError: | open | daisyden | P2 |  |  | dependency component: driver, module: inductor, skipped, ut_upstream | inductor | ut |
| [2613](https://github.com/intel/torch-xpu-ops/issues/2613) | [upstream_ut]  AssertionError: | open | daisyden | P2 |  |  | dependency component: driver, module: inductor, skipped, ut_upstream | inductor | ut |
| [2615](https://github.com/intel/torch-xpu-ops/issues/2615) | [Bug Skip]: New failures Runti | open | CuiYifeng | P2 | Dtype/Precision Issue - Unsupported dtype Half / torch.float16 |  | module: ut, skipped | aten_ops | ut |
| [2618](https://github.com/intel/torch-xpu-ops/issues/2618) | [Bug Skip]: [regression] Asser | open | jmamzax | P0 |  | [PR](https://github.com/numpy/numpy/pull/22525) | skipped, bug_fix_stage5 | unknown | ut |
| [2630](https://github.com/intel/torch-xpu-ops/issues/2630) | Title: [upstream_ut]  Assertio | open | jmamzax | P2 | Failure - Scalars are not equal! |  | skipped, port_from_skiplist | aten_ops | ut |
| [2639](https://github.com/intel/torch-xpu-ops/issues/2639) | test_to() failed during rnn is | open | Silv3S | P2 | Failure - test_to() failed during rnn isinstance() check | [PR](https://github.com/intel/torch-xpu-ops/pull/3193) | skipped | aten_ops | ut |
| [2640](https://github.com/intel/torch-xpu-ops/issues/2640) | random issue test_vjpvjp_index | open | wpietka | P2 | Others - incomplete traceback and insufficient information to determine root cause |  | skipped, random | aten_ops | ut |
| [2649](https://github.com/intel/torch-xpu-ops/issues/2649) | [distributed][pipelining] test | open | syedshahbaaz | P2 | Distributed/Gloo Issue - test_schedule_multiproc.py is hanging during distribute |  | bug, module: distributed | distributed | ut |
| [2659](https://github.com/intel/torch-xpu-ops/issues/2659) | [distributed] test_dist2.py Ru | open | Chao1Han | P2 | Distributed/Gloo Issue - Backend xccl does not implement getBackendOptions. |  | module: distributed | distributed | ut |
| [2669](https://github.com/intel/torch-xpu-ops/issues/2669) | [upstream_ut]  AssertionError: | open | tszulist-hbn | P2 | Failure - Tensor-likes are not close! in test_vmap_exhaustive_addmv_xpu_float32 |  | skipped | aten_ops | ut |
| [2670](https://github.com/intel/torch-xpu-ops/issues/2670) | [upstream_ut]  RuntimeError: c | open | tszulist-hbn | P2 |  |  | skipped, ut_upstream | aten_ops | ut |
| [2686](https://github.com/intel/torch-xpu-ops/issues/2686) | [distributed] Accuracy issues  | open | frost-intel | P2 | Distributed/Gloo Issue - Accuracy issues in distributed testing with test_distri |  | bug, module: distributed | distributed | ut |
| [2693](https://github.com/intel/torch-xpu-ops/issues/2693) | Title: [upstream_ut]  Assertio | open | hoshibara | P2 |  |  | module: inductor, skipped, ut_upstream | inductor | ut |
| [2694](https://github.com/intel/torch-xpu-ops/issues/2694) | Title: [upstream_ut]  Assertio | open | daisyden | P2 |  | [PR](https://github.com/pytorch/pytorch/pull/171773) | module: inductor, skipped, ut_upstream | inductor | ut |
| [2700](https://github.com/intel/torch-xpu-ops/issues/2700) | [distributed] Hang issues with | open | syedshahbaaz | P2 | Distributed/Gloo Issue - Hang issues with test_distributed_spawn.py suggest a pr |  | bug, module: distributed | distributed | ut |
| [2701](https://github.com/intel/torch-xpu-ops/issues/2701) | [distributed] Barrier Timeout  | open | syedshahbaaz | P2 | Distributed/Gloo Issue - Barrier Timeout Error in distributed testing |  | bug, module: distributed | distributed | ut |
| [2702](https://github.com/intel/torch-xpu-ops/issues/2702) | [distributed] RuntimeError: Wo | open | syedshahbaaz | P2 | Timeout/Performance Issue - Work ran time out after 0 milliseconds in distribute |  | bug, module: distributed | distributed | ut |
| [2714](https://github.com/intel/torch-xpu-ops/issues/2714) | [upstream_ut]  AssertionError: | open | Silv3S | P2 |  |  | skipped, ut_upstream | aten_ops | ut |
| [2720](https://github.com/intel/torch-xpu-ops/issues/2720) | [upstream_ut] RuntimeError: fa | open | CuiYifeng | P2 | Backend/Device Issue - missing kernel for xpu in DispatchStub |  | skipped | aten_ops | ut |
| [2737](https://github.com/intel/torch-xpu-ops/issues/2737) | [distributed] AttributeError:  | open | None | P2 | Distributed/Gloo Issue - missing attribute '_gather' in distributed context |  | bug, module: distributed | distributed | ut |
| [2738](https://github.com/intel/torch-xpu-ops/issues/2738) | [distributed] test_c10d_spawn_ | open | jenniew | P2 | Distributed/Gloo Issue - input tensor size mismatch in distributed context |  | bug, module: distributed | distributed | ut |
| [2759](https://github.com/intel/torch-xpu-ops/issues/2759) | [Bug Skip]: New failed cases 2 | open | AKloniecki | P2 | Failure - RuntimeError not raised as expected in test case |  | skipped | aten_ops | ut |
| [2779](https://github.com/intel/torch-xpu-ops/issues/2779) | Accuracy failures in logspace  | open | PawelSwider2000 | P2 | Dtype/Precision Issue - accuracy failures in logspace operation |  | module: ut, skipped, bug_fix_stage5 | aten_ops | ut |
| [2811](https://github.com/intel/torch-xpu-ops/issues/2811) | [Bug Skip]: [Regression] faile | open | jmamzax | P0 | Failure - Expected and actual trace outputs do not match. |  | skipped, bug_fix_stage5 | aten_ops | ut |
| [2837](https://github.com/intel/torch-xpu-ops/issues/2837) | Accuracy issue for Muon optimi | open | Silv3S | P2 | Failure - Tensor-likes not close in Muon optimizer test |  | skipped, bug_fix_stage5 | aten_ops | ut |
| [2840](https://github.com/intel/torch-xpu-ops/issues/2840) | Accuracy issue with 64 bit ind | open | SlawomirLaba, Silv3S | P2 | Dtype/Precision Issue - Tensor comparison failed due to small absolute differenc |  | skipped, bug_fix_stage5 | aten_ops | ut |
| [2853](https://github.com/intel/torch-xpu-ops/issues/2853) | [upstream_ut] torch.ops.aten._ | open | LuFinch | P2 | Flash Attention/Specific Ops Issue - _flash_attention_forward not supported on X |  | skipped | aten_ops | ut |
| [2871](https://github.com/intel/torch-xpu-ops/issues/2871) | [distributed][fsdp] test_fsdp_ | open | songhappy | P2 | Failure - test_fsdp_overlap.py assertion failed with "False is not true" |  | bug, module: distributed | distributed | ut |
| [2921](https://github.com/intel/torch-xpu-ops/issues/2921) | [abs][complex64] - new failing | open | AKloniecki | P2 | Dtype/Precision Issue - test failure related to complex64 data type and abs oper |  | skipped | aten_ops | ut |
| [2966](https://github.com/intel/torch-xpu-ops/issues/2966) | [Bug Skip]: [Regression]2026-3 | open | jmamzax | P0 | Timeout/Performance Issue - Example code timed out during test execution. |  | skipped, bug_fix_stage5, random | aten_ops | ut |
| [2967](https://github.com/intel/torch-xpu-ops/issues/2967) | [distributed] feature gaps in  | open | frost-intel | P2 | Distributed/Gloo Issue - feature gaps in distributed testing for XPU with test_c |  | bug, module: distributed | distributed | ut |
| [2968](https://github.com/intel/torch-xpu-ops/issues/2968) | [distributed] timeout issue in | open | frost-intel | P2 | Timeout/Performance Issue - test experienced a timeout in distributed execution  |  | bug, module: distributed | distributed | ut |
| [2969](https://github.com/intel/torch-xpu-ops/issues/2969) | [distributed] AssertionError:  | open | frost-intel | P2 | Failure - Scalars are not equal in test case |  | bug, module: distributed | distributed | ut |
| [2971](https://github.com/intel/torch-xpu-ops/issues/2971) | [distributed] KeyError in test | open | newtdms, frost-intel | P2 | Distributed/Gloo Issue - KeyError in distributed test related to c10d_xccl |  | bug, module: distributed | distributed | ut |
| [2972](https://github.com/intel/torch-xpu-ops/issues/2972) | [distributed] AssertionError:  | open | newtdms | P2 | Distributed/Gloo Issue - AssertionError in test/distributed/test_c10d_xccl.py re |  | bug, module: distributed | distributed | ut |
| [3010](https://github.com/intel/torch-xpu-ops/issues/3010) | [distributed][tensor] test_ran | open | jenniew | P2 | Inductor/Compilation Issue - TorchRuntimeError during fake tensor call in Dynamo |  | bug, module: distributed | distributed | ut |
| [3021](https://github.com/intel/torch-xpu-ops/issues/3021) | [distributed] all_to_all_singl | open | zhangxiaoli73 | P2 | Distributed/Gloo Issue - all_to_all_single compatibility issue on B60/XCCL |  | module: distributed | distributed | ut |
| [3022](https://github.com/intel/torch-xpu-ops/issues/3022) | [distributed] batch_isend_irec | open | zhangxiaoli73 | P2 | Distributed/Gloo Issue - batch_isend_irecv compatibility issue on B60/XCCL |  | module: distributed | distributed | ut |
| [3025](https://github.com/intel/torch-xpu-ops/issues/3025) | New failing test in Nightly Wh | open | None | P2 | Failure - Expected and actual decomposition outputs do not match. |  | skipped, random | aten_ops | ut |
| [3082](https://github.com/intel/torch-xpu-ops/issues/3082) | multithread support in distrib | open | None | P2 | Distributed/Gloo Issue - multithread support in distributed operations is affect |  | module: distributed, module: ut | distributed | ut |
| [3100](https://github.com/intel/torch-xpu-ops/issues/3100) | [distributed] /handler/dump_nc | open | None | P2 | Distributed/Gloo Issue - related to NCCL logging and trace handling in distribut |  | module: distributed | distributed | ut |
| [3101](https://github.com/intel/torch-xpu-ops/issues/3101) | [distributed] 'torch._C._distr | open | None | P2 | Distributed/Gloo Issue - ProcessGroupXCCL object lacks '_set_default_timeout' at |  | module: distributed | distributed | ut |
| [3102](https://github.com/intel/torch-xpu-ops/issues/3102) | [distributed] RuntimeError: In | open | None | P2 | Backend/Device Issue - invalid device string 'xpu:foo' indicates a problem with  |  | module: distributed | distributed | ut |
| [3105](https://github.com/intel/torch-xpu-ops/issues/3105) | Wrong results from oneDNN conv | open | BBBela | P2 | DNNL/OneDNN Issue - wrong results from oneDNN conv2d kernels due to data pointer |  | hw: PVC, module: ut, skipped | aten_ops | ut |
| [3128](https://github.com/intel/torch-xpu-ops/issues/3128) | [upstream_ut]  AssertionError: | open | LuFinch | P2 |  |  | module: ut, skipped, ut_upstream | aten_ops | ut |
| [3139](https://github.com/intel/torch-xpu-ops/issues/3139) | [distributed][_composable] Ass | open | Kanya-Mo | P2 | Failure - Expects xpu:0 but got xpu:1 |  | bug, module: distributed | distributed | ut |
| [3163](https://github.com/intel/torch-xpu-ops/issues/3163) | [Bug Skip]: Object comparison  | open | chunhuanMeng | P2 |  |  | skipped, ut_upstream | aten_ops | ut |
| [3174](https://github.com/intel/torch-xpu-ops/issues/3174) | [Bug Skip]: Accuracy failure o | open | None | P2 | Failure - test assertion failed for Conv2d groups output comparison |  | module: ut, skipped | aten_ops | ut |
| [3177](https://github.com/intel/torch-xpu-ops/issues/3177) | Accuracy gap of BF16/FP16 test | open | jenniew | P2 | Dtype/Precision Issue - Accuracy gap in BF16/FP16 test |  | skipped | aten_ops | ut |
| [3194](https://github.com/intel/torch-xpu-ops/issues/3194) | Incorrect strides in TestCommo | open | AKloniecki | P2 | Backend/Device Issue - Incorrect strides related to XPU device handling |  |  | aten_ops | ut |
| [3232](https://github.com/intel/torch-xpu-ops/issues/3232) | [distributed][tensor] Assertio | open | None | P2 | Failure - AssertionError not raised for Placement (Shard(dim=2),) in test |  | bug, module: distributed | distributed | ut |
| [3233](https://github.com/intel/torch-xpu-ops/issues/3233) | [distributed] RuntimeError: No | open | None | P2 | Distributed/Gloo Issue - No backend for the parent process group or its backend  |  | bug, module: distributed | distributed | ut |
| [3246](https://github.com/intel/torch-xpu-ops/issues/3246) | AssertionError: Booleans misma | open | BartoszKokoszko | P2 |  | [PR](https://github.com/intel/torch-xpu-ops/pull/3249) | skipped | aten_ops | ut |

#### Dtype / Precision Related (#dtype---precision-related)

| ID | Title | Status | Owner | Priority | Root Cause | PR | Labels | Module | Test Module |
|---|-------|--------|-------|---------|-----------|-----|--------|--------|-------------|
| [1505](https://github.com/intel/torch-xpu-ops/issues/1505) | [ARC-WSL-Ubuntu24.04] 15 Timm  | open | xuhancn, Stonepia | P0 |  |  | bug, E2E, client, os: Windows, module: inductor | inductor | e2e |
| [1624](https://github.com/intel/torch-xpu-ops/issues/1624) | [DONT CLOSE] Known UT Issue li | open | None | P2 |  |  | module: distributed, module: infra | distributed | ut |
| [1778](https://github.com/intel/torch-xpu-ops/issues/1778) | [Infra] Show known issues for  | open | mengfei25 | P1 | Skip/No Test Exists - no test or error details provided |  | E2E, Accuracy, skipped, module: infra | unknown | e2e |
| [1818](https://github.com/intel/torch-xpu-ops/issues/1818) | [BMG-Windows][PT2.8]Torch-xpu- | open | kdrozd-dev | P2 | Backend/Device Issue - Accuracy issue on XPU for Torch-xpu-ops UT |  | os: Windows, hw: BMG, bug_fix_stage5 | aten_ops | ut |
| [1866](https://github.com/intel/torch-xpu-ops/issues/1866) | [release 2.8]Torchbench vision | open | BartoszKokoszko | P0 | Dtype/Precision Issue - amp_fp16 inference accuracy failure |  | Accuracy, os: Windows, hw: BMG, bug_fix_stage5 | aten_ops | e2e |
| [1877](https://github.com/intel/torch-xpu-ops/issues/1877) | Torchbench model squeezenet1_1 | open | Silv3S | P0 | Backend/Device Issue - fail_accuracy on XPU for specific models |  | Accuracy, hw: BMG, hw: PVC, bug_fix_stage5 | aten_ops | ut |
| [2022](https://github.com/intel/torch-xpu-ops/issues/2022) | [Windows] [CI] [UT] AssertionE | open | None | P2 | Failure - Tensor-likes are not close! |  | os: Windows, module: ut, skipped_windows | aten_ops | ut |
| [2128](https://github.com/intel/torch-xpu-ops/issues/2128) | [2.9][BMG-Windows][Torchbench] | open | chuanqi129 | P0 | Backend/Device Issue - Exception Code 0xC0000005 indicates an access violation l |  |  | aten_ops | ut |
| [2215](https://github.com/intel/torch-xpu-ops/issues/2215) | Find use case example for torc | open | dvrogozh | P2 | Others - Looking for use case example of torch-xpu-ops.lib in SYCL C++ extension |  |  | aten_ops | ut |
| [2219](https://github.com/intel/torch-xpu-ops/issues/2219) | float8_e4m3fn precision overfl | open | CuiYifeng, yucai-intel | P2 | Dtype/Precision Issue - float8_e4m3fn precision overflow |  |  | aten_ops | ut |
| [2251](https://github.com/intel/torch-xpu-ops/issues/2251) | [upstream_ut] test_fake_autoca | open | astachowiczhabana | P2 |  |  | duplicate, module: ut, skipped, ut_upstream | aten_ops | ut |
| [2253](https://github.com/intel/torch-xpu-ops/issues/2253) | the supported dtypes are not a | open | daisyden | P2 |  |  | duplicate, skipped, ut_upstream | aten_ops | ut |
| [2295](https://github.com/intel/torch-xpu-ops/issues/2295) | [upstream_ut][xpu][test]nn/tes | open | yucai-intel | P2 |  |  | module: inductor, skipped, ut_upstream | inductor | ut |
| [2331](https://github.com/intel/torch-xpu-ops/issues/2331) | [upstream_ut] AssertionError:  | open | hoshibara | P2 |  | [PR](https://github.com/pytorch/pytorch/pull/172314) | dependency component: oneAPI, module: inductor, module: ut, ut_upstream | inductor | ut |
| [2376](https://github.com/intel/torch-xpu-ops/issues/2376) | [Bug Skip]: NotImplementedErro | open | CuiYifeng | P2 | Dtype/Precision Issue - "logaddexp_xpu" not implemented for 'Complex' dtype |  | module: ut, skipped | aten_ops | ut |
| [2425](https://github.com/intel/torch-xpu-ops/issues/2425) | [upstream_ut]  RuntimeError: E | open | BBBela | P2 | Error - Nested tensor operation with non-nested tensor input |  | skipped, bug_fix_stage4 | aten_ops | ut |
| [2439](https://github.com/intel/torch-xpu-ops/issues/2439) | [oneDNN] TestDecompXPU.test_qu | open | libohao1201 | P2 | DNNL/OneDNN Issue - Test failure related to oneDNN decomposition on XPU with flo |  | dependency component: oneDNN, module: ut | aten_ops | ut |
| [2444](https://github.com/intel/torch-xpu-ops/issues/2444) | [upstream_ut]  RuntimeError: U | open | Silv3S | P2 | Memory/Shared Memory Issue - UR backend returns out of resources error (40) indi |  | skipped | aten_ops | ut |
| [2482](https://github.com/intel/torch-xpu-ops/issues/2482) | test_dtypes issue introduced b | open | daisyden | P2 | Dtype/Precision Issue - test_dtypes issue related to data types in conv_transpos |  | skipped | aten_ops | ut |
| [2491](https://github.com/intel/torch-xpu-ops/issues/2491) | [upstream_ut]  AssertionError: | open | PatrykWilczewski | P2 | Failure - test assertion failed with False is not true |  | skipped, bug_fix_stage5 | aten_ops | ut |
| [2494](https://github.com/intel/torch-xpu-ops/issues/2494) | [upstream_ut]  AssertionError: | open | PawelSwider2000 | P2 | Failure - Scalars are not equal assertion error in test |  | skipped | aten_ops | ut |
| [2496](https://github.com/intel/torch-xpu-ops/issues/2496) | [upstream_ut]  Segmentation fa | open | astachowiczhabana | P0 |  |  | skipped | aten_ops | ut |
| [2510](https://github.com/intel/torch-xpu-ops/issues/2510) | [upstream_ut]  RuntimeError: E | open | PawelSwider2000 | P2 | Error - tensor size exceeds int32_t maximum limit |  | skipped | aten_ops | ut |
| [2512](https://github.com/intel/torch-xpu-ops/issues/2512) | [upstream_ut]  RuntimeError: _ | open | chunhuanMeng | P2 | Backend/Device Issue - _histc_xpu lacks deterministic implementation on XPU back |  | skipped | aten_ops | ut |
| [2592](https://github.com/intel/torch-xpu-ops/issues/2592) | [release/2.10] models got fail | open | mengfei25 | P0 |  |  | E2E, Accuracy, regression | aten_ops | e2e |
| [2654](https://github.com/intel/torch-xpu-ops/issues/2654) | [BMG][OOB] t5 inference perfor | open | jianyizh | P0 | Timeout/Performance Issue - inference performance drop |  | E2E, dtype: float16, triaged, performance, os: Ubuntu, hw: BMG, regression | aten_ops | e2e |
| [2656](https://github.com/intel/torch-xpu-ops/issues/2656) | [release/2.10] models got fail | open | None | P0 | Backend/Device Issue - failure occurs specifically on BMG WSL2 XPU backend |  | os: Windows | aten_ops | ut |
| [2676](https://github.com/intel/torch-xpu-ops/issues/2676) | Random failure in CI test | open | None | P2 | Others - Random failure with no traceback or specific error provided |  | skipped, random | unknown | ut |
| [2680](https://github.com/intel/torch-xpu-ops/issues/2680) | XPU Autocast does not support  | open | CuiYifeng | P2 | Dtype/Precision Issue - XPU Autocast does not support fp32 dtypes |  |  | aten_ops | ut |
| [2689](https://github.com/intel/torch-xpu-ops/issues/2689) | [LNL][Windows] AssertionError: | open | tadkrawiec | P2 | Failure - cur_target out of bounds assertion failed |  | os: Windows, module: ut | aten_ops | ut |
| [2783](https://github.com/intel/torch-xpu-ops/issues/2783) | [Bug Skip]: Key "xpu" is missi | open | daisyden | P2 | Skip/No Test Exists - Key "xpu" is missing, indicating the test is skipped or no |  | module: ut, skipped | aten_ops | ut |
| [2810](https://github.com/intel/torch-xpu-ops/issues/2810) | AssertionError: Object compari | open | daisyden | P2 |  |  | module: inductor, ut_upstream | inductor | ut |
| [2816](https://github.com/intel/torch-xpu-ops/issues/2816) | torch.logcumsumexp incorrectly | open | Silv3S | P2 | Dtype/Precision Issue - torch.logcumsumexp returns NaNs for complex64 input |  | Ready for merge, skipped, bug_fix_stage5 | aten_ops | ut |
| [2817](https://github.com/intel/torch-xpu-ops/issues/2817) | Expected error message is diff | open | kdrozd-dev | P2 | Failure - mismatch between expected and actual error message |  | skipped, bug_fix_stage5 | aten_ops | ut |
| [2862](https://github.com/intel/torch-xpu-ops/issues/2862) | accuracy issue with test_float | open | tszulist-hbn | P2 | Dtype/Precision Issue - accuracy issue with float8 operations |  |  | aten_ops | ut |
| [2899](https://github.com/intel/torch-xpu-ops/issues/2899) | Update nan_to_num XPU stub to  | open | None | P2 | Backend/Device Issue - XPU stub uses incorrect type for nan_to_num parameters |  |  | aten_ops | ut |
| [2908](https://github.com/intel/torch-xpu-ops/issues/2908) | [release/2.11] Model fail_accu | open | xuhancn | P1 | Others - insufficient information to determine root cause |  | E2E | aten_ops | e2e |
| [2914](https://github.com/intel/torch-xpu-ops/issues/2914) | Test case test/test_autograd.p | open | None | P2 | Failure - Tensor-likes are not close! |  |  | aten_ops | ut |
| [2924](https://github.com/intel/torch-xpu-ops/issues/2924) | [release/2.11] xcit_large_24_p | open | jianyizh, mengfei25 | P1 | Dtype/Precision Issue - amp_bf16 training accuracy failure |  | Accuracy | aten_ops | e2e |
| [2928](https://github.com/intel/torch-xpu-ops/issues/2928) | [release/2.11] pyhpc_turbulent | open | jianyizh | P1 | Dtype/Precision Issue - fp32 inference accuracy failure |  |  | aten_ops | e2e |
| [2929](https://github.com/intel/torch-xpu-ops/issues/2929) | [release/2.11] volo_d1_224 inf | open | jianyizh | P1 | Backend/Device Issue - fail_to_run on XPU for volo_d1_224 inference |  |  | aten_ops | e2e |
| [2932](https://github.com/intel/torch-xpu-ops/issues/2932) | [release/2.11] jx_nest_base an | open | jianyizh | P2 | Failure - encountered AssertionError during training |  |  | aten_ops | e2e |
| [2938](https://github.com/intel/torch-xpu-ops/issues/2938) | [release/2.11] basic_gnn_gin a | open | jianyizh | P2 | Timeout/Performance Issue - inference fp32 performance dropped ~25% |  | performance | aten_ops | e2e |
| [2952](https://github.com/intel/torch-xpu-ops/issues/2952) | [release/2.11][wsl] timm_model | open | weishi-deng | P0 | Dtype/Precision Issue - bfloat16 accuracy failure in model training |  | Accuracy, hw: BMG | aten_ops | ut |
| [2960](https://github.com/intel/torch-xpu-ops/issues/2960) | [release/2.11] timm_models_xci | open | None | P0 | Dtype/Precision Issue - float16 training accuracy test failure |  |  | aten_ops | ut |
| [2984](https://github.com/intel/torch-xpu-ops/issues/2984) | [release/2.11] sebotnet33ts_25 | open | jianyizh, weishi-deng | P1 | Backend/Device Issue - XPU specific failure during fp32 training accuracy check |  | os: Ubuntu, hw: BMG | aten_ops | e2e |
| [3014](https://github.com/intel/torch-xpu-ops/issues/3014) | [upstream_ut] AssertionError:  | open | None | P2 |  |  | bug_fix_stage5 | unknown | ut |
| [3041](https://github.com/intel/torch-xpu-ops/issues/3041) | AssertionError: Expected len(f | open | Silv3S | P2 |  |  | ut_upstream | aten_ops | ut |
| [3103](https://github.com/intel/torch-xpu-ops/issues/3103) | Tensor-likes are not equal for | open | BBBela | P2 | Backend/Device Issue - XPU tensor-like comparison failure in test |  | module: ut, skipped, random | aten_ops | ut |
| [3121](https://github.com/intel/torch-xpu-ops/issues/3121) | [Bug Skip]: CUDA specific UT t | open | None | P1 | Skip/No Test Exists - test is skipped or not applicable for XPU backend |  | skipped | aten_ops | ut |
| [3156](https://github.com/intel/torch-xpu-ops/issues/3156) | AssertionError: 'Assertion cur | open | kdrozd-dev | P2 | Failure - Assertion cur_target >= 0 && cur_target < n_classes failed in cross entropy loss test |  |  | aten_ops | ut |
| [3161](https://github.com/intel/torch-xpu-ops/issues/3161) | Exception: Tensor-likes are no | open | tadkrawiec | P2 |  |  | os: Windows | aten_ops | ut |
| [3184](https://github.com/intel/torch-xpu-ops/issues/3184) | New failing UTs: test_cross_en | open | wpietka | P2 | Failure - test expects a specific condition to be true but it failed during execution. |  | module: ut, skipped | aten_ops | ut |
| [3236](https://github.com/intel/torch-xpu-ops/issues/3236) | AssertionError: 'def [28 chars | open | jmamzax | P2 | Failure - mismatch in expected IR code for XPU backend operations |  | bug_fix_stage5 | aten_ops | ut |
| [3238](https://github.com/intel/torch-xpu-ops/issues/3238) | The supported dtypes of _refs. | open | CuiYifeng | P2 |  |  | ut_upstream | aten_ops | ut |
| [3243](https://github.com/intel/torch-xpu-ops/issues/3243) | AssertionError: False is not t | open | pponikox | P2 | Failure - assertion 'False is not true' failed in test |  | module: ut, skipped | aten_ops | ut |

#### Flash Attention / Transformer Related (#flash-attention---transformer-related)

| ID | Title | Status | Owner | Priority | Root Cause | PR | Labels | Module | Test Module |
|---|-------|--------|-------|---------|-----------|-----|--------|--------|-------------|
| [1159](https://github.com/intel/torch-xpu-ops/issues/1159) | [LNL Windows][Test by CD Night | open | Stonepia | P0 | Dtype/Precision Issue - value cannot be converted to at::BFloat16 without overfl |  | E2E, client, module: dependency bug, dependency: third_party packages | aten_ops | e2e |
| [1165](https://github.com/intel/torch-xpu-ops/issues/1165) | [CI] Add a test of PyTorch XPU | open | RUIJIEZHONG66166 | P0 | Skip/No Test Exists - No test was implemented or executed. |  | module: transformers | aten_ops | build |
| [1749](https://github.com/intel/torch-xpu-ops/issues/1749) | transformers UT failure in XPU | open | LuFinch | P2 | Backend/Device Issue - XPU does not support backward or grad for SDPA operation |  |  | aten_ops | ut |
| [2015](https://github.com/intel/torch-xpu-ops/issues/2015) | inf is returned by nn.Transfor | open | yucai-intel | P2 | Dtype/Precision Issue - inf returned using float16 in TransformerEncoderLayer on |  | skipped | aten_ops | ut |
| [2058](https://github.com/intel/torch-xpu-ops/issues/2058) | [release/2.9] llama_v2_7b_16h  | open | jianyizh | P0 | device-specific backend problem. |  | performance, regression, dependency component: community | aten_ops | e2e |
| [2186](https://github.com/intel/torch-xpu-ops/issues/2186) | AssertionError: Mul tiheadAtte | open | daisyden | P2 | Failure - Mul tiheadAttention does not support NestedTensor outside of its fast path |  | dependency component: oneDNN | aten_ops | ut |
| [2200](https://github.com/intel/torch-xpu-ops/issues/2200) | support flash attention op on  | open | ElaineBao | P2 | Flash Attention/Specific Ops Issue - request to support flash attention op on XP |  | dependency component: oneDNN | aten_ops | ut |
| [2232](https://github.com/intel/torch-xpu-ops/issues/2232) | sdpa backward kernel is requir | open | None | P2 | Memory/Shared Memory Issue - sdpa backward kernel needed to reduce memory usage |  |  | aten_ops | ut |
| [2250](https://github.com/intel/torch-xpu-ops/issues/2250) | Found mismatch when comparing  | open | astachowiczhabana | P2 | Mismatch - Mismatch in output of aten.view.default between FakeTensor and concrete Tensors. |  | skipped, bug_fix_stage3 | aten_ops | ut |
| [2255](https://github.com/intel/torch-xpu-ops/issues/2255) | [upstream_ut] RuntimeError: Lo | open | daisyden | P2 |  |  | skipped, ut_upstream | aten_ops | ut |
| [2285](https://github.com/intel/torch-xpu-ops/issues/2285) | Support efficient attention | open | chunhuanMeng | P2 | Backend/Device Issue - aten::_efficient_attention_forward not supported on CPU b |  | skipped | aten_ops | ut |
| [2390](https://github.com/intel/torch-xpu-ops/issues/2390) | SDPA in pytorch use different  | open | LuFinch | P2 | Backend/Device Issue - SDPA uses different backend compared with IPEX on XPU |  |  | aten_ops | ut |
| [2436](https://github.com/intel/torch-xpu-ops/issues/2436) | [upstream_ut]  AttributeError: | open | daisyden | P1 | Error - 'NoneType' object has no attribute 'clone' due to missing object handling |  | skipped, dependency component: community, random | aten_ops | ut |
| [2442](https://github.com/intel/torch-xpu-ops/issues/2442) | [Bug Skip]: NotImplementedErro | open | daisyden, LuFinch | P2 | Flash Attention/Specific Ops Issue - aten::_flash_attention_forward not implemen |  | skipped, not_target | aten_ops | ut |
| [2570](https://github.com/intel/torch-xpu-ops/issues/2570) | crash in sdpa. | open | LuFinch | P0 | Others - insufficient information to determine root cause |  | dependency component: oneDNN | aten_ops | ut |
| [2675](https://github.com/intel/torch-xpu-ops/issues/2675) | [Bug Skip]: AttributeError: 'N | open | pponikox | P2 | Error - 'NoneType' object has no attribute 'clone' due to missing object reference |  | skipped, bug_fix_stage5 | aten_ops | ut |
| [2696](https://github.com/intel/torch-xpu-ops/issues/2696) | Title: [upstream_ut]  RuntimeE | open | chunhuanMeng | P2 |  |  | module: inductor, skipped, ut_upstream | inductor | e2e |
| [2697](https://github.com/intel/torch-xpu-ops/issues/2697) | Title: [upstream_ut]  RuntimeE | open | chunhuanMeng | P2 |  |  | module: inductor, skipped, ut_upstream | inductor | e2e |
| [2704](https://github.com/intel/torch-xpu-ops/issues/2704) | Title: [upstream_ut]  Assertio | open | kdrozd-dev | P2 |  | [PR](https://github.com/pytorch/pytorch/pull/177636) | skipped, ut_upstream, bug_fix_stage5 | aten_ops | ut |
| [2742](https://github.com/intel/torch-xpu-ops/issues/2742) | [Linux][PT2E] hf_Roberta_base  | open | chunhuanMeng | P0 | Timeout/Performance Issue - hf_Roberta_base model performance failed for both AS |  |  | aten_ops | e2e |
| [2800](https://github.com/intel/torch-xpu-ops/issues/2800) | AttributeError: 'torch._C._Xpu | open | guangyey | P2 |  |  | dependency component: oneAPI, module: inductor, ut_upstream | inductor | ut |
| [2802](https://github.com/intel/torch-xpu-ops/issues/2802) | Three aten._scaled_dot_product | open | LuFinch | P2 |  |  | module: inductor, ut_upstream | inductor | ut |
| [2869](https://github.com/intel/torch-xpu-ops/issues/2869) | [Bug Skip]: New UT failure in  | open | None | P2 | Skip/No Test Exists - Test is marked as skipped or not executed |  | skipped_windows | aten_ops | ut |
| [2939](https://github.com/intel/torch-xpu-ops/issues/2939) | [release/2.11] gmlp_s16_224 in | open | jianyizh | P2 | Timeout/Performance Issue - inference amp performance dropped ~15% |  | performance | aten_ops | e2e |
| [2997](https://github.com/intel/torch-xpu-ops/issues/2997) | AssertionError of test_linear_ | open | etaf | P2 |  |  | module: inductor, ut_upstream | inductor | ut |
| [3007](https://github.com/intel/torch-xpu-ops/issues/3007) | AssertionError: Scalars are no | open | daisyden | P2 |  | [PR](https://github.com/pytorch/pytorch/pull/178369) | module: inductor, ut_upstream | inductor | e2e |
| [3047](https://github.com/intel/torch-xpu-ops/issues/3047) | [Bug Skip]: [Regression]UT fai | open | None | P0 | Failure - Torch not compiled with CUDA enabled assertion error |  | skipped | unknown | ut |
| [3058](https://github.com/intel/torch-xpu-ops/issues/3058) | [E2E] hf_GPT2_large amp_fp16/a | open | weishi-deng | P1 |  |  | E2E, hw: PVC | aten_ops | e2e |
| [3093](https://github.com/intel/torch-xpu-ops/issues/3093) | XPU does not support NestedTen | open | None | P2 | Supported - XPU does not support NestedTensor for SDPA operations. |  | module: ut | aten_ops | ut |
| [3114](https://github.com/intel/torch-xpu-ops/issues/3114) | [Bug Skip]: Failure skip on 20 | open | None | P2 | Skip/No Test Exists - test was skipped on 2026-3-21 |  | skipped, random | aten_ops | ut |
| [3126](https://github.com/intel/torch-xpu-ops/issues/3126) | [upstream_ut]  Two NestedTenso | open | LuFinch | P2 |  |  | module: ut, skipped, ut_upstream | aten_ops | ut |
| [3127](https://github.com/intel/torch-xpu-ops/issues/3127) | [upstream_ut]  AssertionError: | open | LuFinch | P2 |  |  | module: ut, skipped, ut_upstream | aten_ops | ut |
| [3129](https://github.com/intel/torch-xpu-ops/issues/3129) | [upstream_ut]  AssertionError: | open | LuFinch | P2 |  |  | module: ut, skipped, ut_upstream | aten_ops | ut |
| [3130](https://github.com/intel/torch-xpu-ops/issues/3130) | [upstream_ut]  AssertionError: | open | LuFinch | P2 |  |  | module: ut, skipped, ut_upstream | aten_ops | ut |
| [3131](https://github.com/intel/torch-xpu-ops/issues/3131) | [upstream_ut]  NotImplementedE | open | chunhuanMeng | P2 |  |  | module: ut, skipped, ut_upstream | aten_ops | ut |
| [3132](https://github.com/intel/torch-xpu-ops/issues/3132) | [upstream_ut]  transfomers tes | open | LuFinch | P2 |  |  | module: ut, skipped, ut_upstream | aten_ops | ut |
| [3133](https://github.com/intel/torch-xpu-ops/issues/3133) | [upstream_ut]  RuntimeError: s | open | LuFinch | P2 |  |  | module: ut, skipped, ut_upstream | aten_ops | ut |
| [3136](https://github.com/intel/torch-xpu-ops/issues/3136) | [upstream_ut]  AssertionError: | open | LuFinch | P2 |  |  | module: ut, skipped, ut_upstream | aten_ops | ut |
| [3137](https://github.com/intel/torch-xpu-ops/issues/3137) | [upstream_ut]  RuntimeError: e | open | LuFinch | P2 |  |  | module: ut, skipped, ut_upstream, random | aten_ops | ut |
| [3140](https://github.com/intel/torch-xpu-ops/issues/3140) | [upstream_ut]  RuntimeError: F | open | LuFinch | P2 |  |  | module: ut, skipped, ut_upstream | aten_ops | ut |
| [3141](https://github.com/intel/torch-xpu-ops/issues/3141) | [upstream_ut]  RuntimeError: F | open | LuFinch | P2 |  |  | module: ut, skipped, ut_upstream | aten_ops | ut |
| [3142](https://github.com/intel/torch-xpu-ops/issues/3142) | [upstream_ut]  RuntimeError: T | open | LuFinch | P2 |  |  | module: ut, skipped, ut_upstream | aten_ops | ut |
| [3143](https://github.com/intel/torch-xpu-ops/issues/3143) | NotImplementedError: The opera | open | LuFinch | P2 |  |  | module: ut, skipped, ut_upstream | aten_ops | ut |
| [3149](https://github.com/intel/torch-xpu-ops/issues/3149) | New failure in test_rms_norm_d | open | None | P2 | Failure - RuntimeError not raised as expected in test |  | module: ut, skipped | aten_ops | ut |
| [3176](https://github.com/intel/torch-xpu-ops/issues/3176) | [Bug Skip]: ValueError: _scale | open | None | P2 | Backend/Device Issue - inputs are not on the same XPU device |  | skipped | aten_ops | ut |
| [3178](https://github.com/intel/torch-xpu-ops/issues/3178) | New failed test cases 2026-03- | open | pponikox | P2 | Backend/Device Issue - XPU-specific test failure in vmap with SDPA operation |  | module: ut, skipped | aten_ops | ut |
| [3195](https://github.com/intel/torch-xpu-ops/issues/3195) | test_sdpa_unbacked_no_dde_xpu  | open | None | P0 | Backend/Device Issue - test crashed on XPU backend execution |  | skipped, random | aten_ops | ut |
| [3229](https://github.com/intel/torch-xpu-ops/issues/3229) | RuntimeError: No viable backen | open | None | P2 |  |  | skipped, ut_upstream | aten_ops | ut |
| [3258](https://github.com/intel/torch-xpu-ops/issues/3258) | Error in op: torch.ops.aten._s | open | None | P2 | 5 - Flash Attention/Specific Ops Issue - Error in _scaled_dot_product_fused_atte |  |  | aten_ops | ut |
| [3259](https://github.com/intel/torch-xpu-ops/issues/3259) | New failed test cases 2026-04- | open | SlawomirLaba | P2 | Backend/Device Issue - XPU device initialization or compatibility failure |  | skipped | aten_ops | ut |

#### Inductor / Compilation Related (#inductor---compilation-related)

| ID | Title | Status | Owner | Priority | Root Cause | PR | Labels | Module | Test Module |
|---|-------|--------|-------|---------|-----------|-----|--------|--------|-------------|
| [146](https://github.com/intel/torch-xpu-ops/issues/146) | Evaluate register spill in SYC | open | CuiYifeng, jianyizh, mengfei25 | P2 | Backend/Device Issue - register spill evaluation in SYCL kernel on XPU |  | enhancement | aten_ops | ut |
| [1764](https://github.com/intel/torch-xpu-ops/issues/1764) | New kernels for concat | open | yucai-intel | P2 | Others - New kernels for concat, no specific error provided. |  | performance, kernel_optimization, hw: BMG, module: op impl, benchmark | aten_ops | ut |
| [2132](https://github.com/intel/torch-xpu-ops/issues/2132) | [2.9][BMG-Windows][Torch-xpu-o | open | pbielak | P2 |  |  | module: ut | aten_ops | ut |
| [2140](https://github.com/intel/torch-xpu-ops/issues/2140) | Consider how to avoid copy in  | open | CuiYifeng | P2 | Memory/Shared Memory Issue - Avoiding copy in FFT kernels relates to memory hand |  | enhancement | aten_ops | ut |
| [2196](https://github.com/intel/torch-xpu-ops/issues/2196) | Fix DistributionElementwiseKer | open | None | P2 | Memory/Shared Memory Issue - register spill in DistributionElementwiseKernelFunc |  | enhancement | aten_ops | ut |
| [2467](https://github.com/intel/torch-xpu-ops/issues/2467) | Host may stuck when submit too | open | jianyizh | P2 | Backend/Device Issue - Host stuck due to excessive kernel submissions on XPU |  | dependency component: driver | aten_ops | ut |
| [2619](https://github.com/intel/torch-xpu-ops/issues/2619) | [release/2.10] Some models ind | open | jianyizh, weishi-deng | P0 |  |  | E2E, performance, regression | aten_ops | e2e |
| [2645](https://github.com/intel/torch-xpu-ops/issues/2645) | [Bug Skip]: [regression] Runti | open | CuiYifeng | P0 | Backend/Device Issue - missing kernel for xpu in DispatchStub |  | skipped | aten_ops | ut |
| [2650](https://github.com/intel/torch-xpu-ops/issues/2650) | [OOB Performance] The performa | open | jianyizh | P0 | Inductor/Compilation Issue - Performance impact caused by TORCHINDUCTOR_ONLINE_S |  | E2E, dtype: float16, triaged, performance, os: Ubuntu, hw: BMG, regression | aten_ops | e2e |
| [2888](https://github.com/intel/torch-xpu-ops/issues/2888) | torch._inductor.exc.InductorEr | open | Stonepia | P2 |  |  | module: inductor, ut_upstream | inductor | ut |
| [2922](https://github.com/intel/torch-xpu-ops/issues/2922) | [release/2.11] UT inductor Ass | open | tadkrawiec | P2 | Backend/Device Issue - pass_fds not supported on Windows |  | os: Windows | aten_ops | ut |
| [2935](https://github.com/intel/torch-xpu-ops/issues/2935) | [release/2.11][inductor] huggi | open | jianyizh | P0 | Inductor/Compilation Issue - performance regression in XLNetLMHeadModel with amp |  | performance | aten_ops | e2e |
| [2950](https://github.com/intel/torch-xpu-ops/issues/2950) | SYCL compilation flag -fsycl-i | open | BBBela | P2 | Inductor/Compilation Issue - SYCL compilation flag not working as expected for T |  |  | aten_ops | ut |
| [2958](https://github.com/intel/torch-xpu-ops/issues/2958) | AssertionError of test_dtensor | open | daisyden | P2 |  |  | module: inductor, ut_upstream | inductor | ut |
| [3013](https://github.com/intel/torch-xpu-ops/issues/3013) | [upstream_ut] RuntimeError: Ke | open | None | P2 |  |  | bug_fix_stage5 | unknown | ut |
| [3080](https://github.com/intel/torch-xpu-ops/issues/3080) | cudagraph tests blocked by fea | open | None | P2 | 10 - Feature Not Supported |  | module: ut | aten_ops | ut |
| [3094](https://github.com/intel/torch-xpu-ops/issues/3094) | XPUGraph tree support | open | None | P2 |  |  | module: inductor, ut_upstream | inductor | ut |
| [3148](https://github.com/intel/torch-xpu-ops/issues/3148) | [Triton] Huggingface openai/wh | open | None | P0 | Backend/Device Issue - XPU specific failure with Huggingface model accuracy |  | Accuracy, hw: BMG, hw: PVC, dependency component: Triton | aten_ops | e2e |
| [3150](https://github.com/intel/torch-xpu-ops/issues/3150) | [Task] Align XPU kernel's impl | open | guangyey | P2 | device-specific backend discrepancy. |  |  | aten_ops | ut |
| [3151](https://github.com/intel/torch-xpu-ops/issues/3151) | [Triton] Timm_models  rexnet_1 | open | None | P0 | Backend/Device Issue - XPU specific failure with Timm models in Triton. |  | Accuracy, hw: BMG, dependency component: Triton | aten_ops | e2e |
| [3158](https://github.com/intel/torch-xpu-ops/issues/3158) | AttributeError: module 'triton | open | tadkrawiec | P2 |  |  |  | aten_ops | ut |
| [3160](https://github.com/intel/torch-xpu-ops/issues/3160) | compiler not found (Windows) | open | kdrozd-dev | P2 |  |  | os: Windows | aten_ops | ut |
| [3187](https://github.com/intel/torch-xpu-ops/issues/3187) | PyTorch XPU gpu_cpp_wrapper fa | open | CuiYifeng | P2 |  |  | ut_upstream | aten_ops | ut |
| [3191](https://github.com/intel/torch-xpu-ops/issues/3191) | torch._inductor.exc.InductorEr | open | EikanWang, Copilot | P2 | Inductor/Compilation Issue - Assertion failure due to conflicting fallback and d |  | E2E, hw: PVC | aten_ops | e2e |
| [3224](https://github.com/intel/torch-xpu-ops/issues/3224) | [Win][Build] Building SYCL (De | open | chunhuanMeng | P0 | Backend/Device Issue - SYCL kernel build failure on Windows for XPU |  |  | aten_ops | build |
| [3242](https://github.com/intel/torch-xpu-ops/issues/3242) | AssertionError: Torch not comp | open | None | P2 |  |  | module: ut, skipped | aten_ops | ut |

#### Others (#others)

| ID | Title | Status | Owner | Priority | Root Cause | PR | Labels | Module | Test Module |
|---|-------|--------|-------|---------|-----------|-----|--------|--------|-------------|
| [208](https://github.com/intel/torch-xpu-ops/issues/208) | Abstract utility functions use | open | CuiYifeng | P2 | Others - abstract utility functions in ATen operator implementation |  | enhancement, module: op impl, long term | aten_ops | ut |
| [489](https://github.com/intel/torch-xpu-ops/issues/489) | Moco NotImplementedError: xpu  | open | weishi-deng | P2 | Backend/Device Issue - xpu not supported |  | E2E, Accuracy, module: torchbench, dtype: amp_bf16, dtype: amp_fp16, training, dtype: float16, dtype: float32, dtype: bfloat16 | aten_ops | e2e |
| [492](https://github.com/intel/torch-xpu-ops/issues/492) | Timm_efficientdet NotImplement | open | weishi-deng | P0 | Backend/Device Issue - model code forces use of CUDA instead of XPU |  | E2E, Accuracy, module: torchbench, dtype: amp_bf16, dtype: amp_fp16, training, dtype: float16, dtype: float32, dtype: bfloat16, triaged | aten_ops | e2e |
| [1171](https://github.com/intel/torch-xpu-ops/issues/1171) | LNL Windows got unexpected err | open | xuhancn, chunhuanMeng | P2 | Backend/Device Issue - unexpected error on XPU for LNL Windows |  | client, os: Windows, hw : LNL, hw: BMG, dependency component: driver | aten_ops | ut |
| [1324](https://github.com/intel/torch-xpu-ops/issues/1324) | [Win] UR Error when OOM and br | open | Stonepia | P2 | Memory/Shared Memory Issue - Out of memory (OOM) leading to tensor context break |  | client, os: Windows, module: dependency bug, dependency component: driver, dependency component: oneAPI | aten_ops | ut |
| [1510](https://github.com/intel/torch-xpu-ops/issues/1510) | Some test cases will be hang o | open | Stonepia, mengfei25 | P2 | Backend/Device Issue - test cases hang on BMG Ubuntu related to XPU backend |  | bug, client, os: Ubuntu, hw: BMG, dependency component: driver, module: ut | aten_ops | ut |
| [1519](https://github.com/intel/torch-xpu-ops/issues/1519) | [PVC][PT2.7][ABI=1][Torch-xpu- | open | None | P2 | Backend/Device Issue - coredump related to XPU operations in Torch-xpu-ops UT |  | dependency component: driver, module: ut | aten_ops | ut |
| [1574](https://github.com/intel/torch-xpu-ops/issues/1574) | The operator 'aten::_grouped_m | open | Stonepia | P2 | Backend/Device Issue - aten::_grouped_mm not implemented for XPU device |  | module: ao | AO | ut |
| [1587](https://github.com/intel/torch-xpu-ops/issues/1587) | Keep track on the latest CUDA  | open | CuiYifeng, yucai-intel | P2 | Skip/No Test Exists - no test or error traceback provided |  | kernel_optimization | aten_ops | ut |
| [1594](https://github.com/intel/torch-xpu-ops/issues/1594) | Keep track on the building war | open | CuiYifeng, chunhuanMeng | P0 | Others - building warning tracking issue |  | module: build | aten_ops | ut |
| [1645](https://github.com/intel/torch-xpu-ops/issues/1645) | [For Comparison] Save referenc | open | mengfei25 | P2 | Skip/No Test Exists - no test or error information provided |  | module: infra | inductor | ut |
| [1649](https://github.com/intel/torch-xpu-ops/issues/1649) | [cpp extension] Provide a clea | open | dvrogozh | P2 | Backend/Device Issue - inconsistent oneAPI versions cause XPU backend errors |  | dependency component: oneAPI, module: build | aten_ops | ut |
| [1678](https://github.com/intel/torch-xpu-ops/issues/1678) | missing op support for `model. | open | None | P0 | Memory/Shared Memory Issue - missing op support for `model.share_memory()` indic |  | bug_fix_stage3 | aten_ops | ut |
| [1689](https://github.com/intel/torch-xpu-ops/issues/1689) | [For op Perf Comparison] Save  | open | None | P2 | Skip/No Test Exists - no test or error information provided |  | module: infra | aten_ops | ut |
| [1722](https://github.com/intel/torch-xpu-ops/issues/1722) | Ask an API to query GPU type(i | open | guangyey | P2 | Mismatch - Request for an API to query GPU type (iGPU/dGPU) is missing or not properly implemented. |  | dependency component: oneAPI | aten_ops | ut |
| [1729](https://github.com/intel/torch-xpu-ops/issues/1729) | Validation Check List | open | chuanqi129 | P2 | Skip/No Test Exists - no test or error information provided |  | module: infra | aten_ops | ut |
| [1745](https://github.com/intel/torch-xpu-ops/issues/1745) | torch.xpu.empty_cache() introd | open | guangyey | P2 | Memory/Shared Memory Issue - torch.xpu.empty_cache() introduces memory leak |  | module: core | aten_ops | ut |
| [1784](https://github.com/intel/torch-xpu-ops/issues/1784) | [Performance] Torch XPU Profil | open | jfedorov | P2 | 11 - Timeout/Performance Issue - Torch XPU Profiler is not reliable |  | module: profiler | profiling | ut |
| [1856](https://github.com/intel/torch-xpu-ops/issues/1856) | channel last aten::hardswish_  | open | chunhuanMeng | P2 | Memory/Shared Memory Issue - extra copy caused by channel last aten::hardswish_ |  | performance, hw: BMG | aten_ops | ut |
| [1900](https://github.com/intel/torch-xpu-ops/issues/1900) | implement torch.linalg.qr xpu  | open | pbielak | P2 | Backend/Device Issue - XPU backend implementation missing for torch.linalg.qr |  | module: op impl, bug_fix_stage3 | aten_ops | ut |
| [1901](https://github.com/intel/torch-xpu-ops/issues/1901) | implement torch.linalg.svd xpu | open | CuiYifeng | P2 | Backend/Device Issue - XPU backend for torch.linalg.svd not implemented |  | module: op impl | aten_ops | ut |
| [1902](https://github.com/intel/torch-xpu-ops/issues/1902) | implement torch.linalg.pinv xp | open | mwiktor-intel | P2 | Backend/Device Issue - XPU backend for torch.linalg.pinv is not implemented |  | module: op impl, bug_fix_stage5 | aten_ops | ut |
| [1936](https://github.com/intel/torch-xpu-ops/issues/1936) | implement torch.linalg.cholesk | open | mwiktor-intel | P2 | Backend/Device Issue - XPU backend for torch.linalg.cholesky is not implemented |  | module: op impl, bug_fix_stage5 | aten_ops | ut |
| [1963](https://github.com/intel/torch-xpu-ops/issues/1963) | [upstream_ut] MetadataMismatch | open | pbielak | P2 |  | [PR](https://github.com/pytorch/pytorch/pull/178277, https://github.com/pytorch/pytorch/pull/175965) | module: ut, ut_upstream | aten_ops | ut |
| [1986](https://github.com/intel/torch-xpu-ops/issues/1986) | torch.xpu._sleep is missing, | open | guangyey | P2 | Mismatch - torch.xpu._sleep is not implemented or available in the current setup. |  | dependency component: oneAPI | aten_ops | ut |
| [2055](https://github.com/intel/torch-xpu-ops/issues/2055) | New huggingface LLM models iss | open | jianyizh, mengfei25 | P0 | Others - insufficient information to determine root cause |  | E2E, hw: PVC | aten_ops | e2e |
| [2063](https://github.com/intel/torch-xpu-ops/issues/2063) | Avoid using out-of-date term | open | CuiYifeng | P2 | Skip/No Test Exists - no test or error traceback provided |  | enhancement | aten_ops | ut |
| [2086](https://github.com/intel/torch-xpu-ops/issues/2086) | nd_item::barrier has been depr | open | dvrogozh | P2 | Backend/Device Issue - nd_item::barrier is deprecated on XPU backend. |  | enhancement | aten_ops | ut |
| [2089](https://github.com/intel/torch-xpu-ops/issues/2089) | need an implementation that wo | open | guangyey | P2 | Backend/Device Issue - torch.xpu.is_available() initializes GPU context unintent |  | dependency component: driver | aten_ops | ut |
| [2098](https://github.com/intel/torch-xpu-ops/issues/2098) | Upstream XPU functions in yaml | open | guangyey | P2 | Backend/Device Issue - XPU functions in yaml related to upstream backend issues |  | enhancement | aten_ops | ut |
| [2127](https://github.com/intel/torch-xpu-ops/issues/2127) | Path Coverage enhancement | open | CuiYifeng | P2 | Skip/No Test Exists - no test or error information provided |  | enhancement | aten_ops | ut |
| [2142](https://github.com/intel/torch-xpu-ops/issues/2142) | XPU max_memory_allocated have  | open | guangyey | P2 | Memory/Shared Memory Issue - XPU and CUDA show different memory allocation outpu |  | bug | aten_ops | ut |
| [2157](https://github.com/intel/torch-xpu-ops/issues/2157) | BMG d2h copy is very slow comp | open | jianyizh, mengfei25 | P2 | Timeout/Performance Issue - BMG d2h copy is significantly slower compared to PVC |  | performance, dependency component: driver | aten_ops | ut |
| [2195](https://github.com/intel/torch-xpu-ops/issues/2195) | Tools in pti should get 100% f | open | aostrowski-hbn | P2 | Backend/Device Issue - functionality not working on BMG for PyTorch profiling |  | module: profiler | profiling | ut |
| [2199](https://github.com/intel/torch-xpu-ops/issues/2199) | Fix reduction and norm registe | open | None | P2 | Memory/Shared Memory Issue - register spill in reduction and norm operations |  | enhancement | aten_ops | ut |
| [2217](https://github.com/intel/torch-xpu-ops/issues/2217) | AO Performance issue track | open | Stonepia | P2 | Timeout/Performance Issue - AO Performance issue track |  | module: ao | AO | ut |
| [2261](https://github.com/intel/torch-xpu-ops/issues/2261) | [xpu][profiler] Run with fork  | open | moksiuc | P2 | Backend/Device Issue - XPU profiler warning during fork process execution |  | dependency component: oneAPI, module: profiler | profiling | ut |
| [2263](https://github.com/intel/torch-xpu-ops/issues/2263) | [xpu][bug] XPU Trace event end | open | PawelSwider2000 | P2 | Backend/Device Issue - XPU trace event timing discrepancy |  | module: profiler | profiling | ut |
| [2287](https://github.com/intel/torch-xpu-ops/issues/2287) | [upstream_ut] test_python_ref  | open | yucai-intel | P2 |  |  | module: ut, ut_upstream | aten_ops | ut |
| [2329](https://github.com/intel/torch-xpu-ops/issues/2329) | [upstream_ut] feature missing: | open | etaf | P2 |  |  | duplicate, dependency component: Triton, module: inductor, ut_upstream | inductor | ut |
| [2333](https://github.com/intel/torch-xpu-ops/issues/2333) | [Don't merge] Collect the new  | open | None | P2 | Skip/No Test Exists - test is empty or not applicable |  | module: infra | aten_ops | ut |
| [2349](https://github.com/intel/torch-xpu-ops/issues/2349) | [oneAPI][backward compatibilit | open | riverliuintel | P2 | Backend/Device Issue - missing library version for XPU backend compatibility |  |  | aten_ops | ut |
| [2392](https://github.com/intel/torch-xpu-ops/issues/2392) | [Bug Skip]: torch.OutOfMemoryE | open | xuhancn | P2 | Memory/Shared Memory Issue - XPU out of memory error occurred |  | skipped_windows | aten_ops | ut |
| [2400](https://github.com/intel/torch-xpu-ops/issues/2400) | [ut_upstream] tf32_on_and_off( | open | chunhuanMeng | P2 | Backend/Device Issue - XPU support required for tf32_on_and_off() test |  |  | aten_ops | ut |
| [2412](https://github.com/intel/torch-xpu-ops/issues/2412) | Some NestedTensor missing XPU  | open | yucai-intel | P2 | Backend/Device Issue - XPU support missing for NestedTensor operations |  | module: ut | aten_ops | ut |
| [2440](https://github.com/intel/torch-xpu-ops/issues/2440) | [For UT failures classify] Sav | open | None | P2 | Skip/No Test Exists - no test or error details provided |  | module: infra | inductor | ut |
| [2447](https://github.com/intel/torch-xpu-ops/issues/2447) | test_share_memory_xpu failure | open | None | P2 | Memory/Shared Memory Issue - failure in test_share_memory_xpu related to shared  |  | skipped | aten_ops | ut |
| [2463](https://github.com/intel/torch-xpu-ops/issues/2463) | [Bug Skip]: OSError: SYCL runt | open | xuhancn | P2 | Backend/Device Issue - SYCL runtime not detected on XPU |  | skipped_windows | aten_ops | ut |
| [2465](https://github.com/intel/torch-xpu-ops/issues/2465) | [windows] ut hang | open | tadkrawiec | P2 | Timeout/Performance Issue - UT hang indicates a timeout or performance bottlenec |  | os: Windows | aten_ops | ut |
| [2469](https://github.com/intel/torch-xpu-ops/issues/2469) | test_matmul_cuda.py gaps | open | CuiYifeng, guangyey | P2 | Skip/No Test Exists - test_matmul_cuda.py gaps indicate missing or skipped tests |  |  | aten_ops | ut |
| [2471](https://github.com/intel/torch-xpu-ops/issues/2471) | test_cuda.py gaps | open | guangyey | P2 | Skip/No Test Exists - test_cuda.py gaps indicate missing or skipped tests. |  |  | aten_ops | ut |
| [2479](https://github.com/intel/torch-xpu-ops/issues/2479) | [Bug] torch.rand output differ | open | Stonepia, CuiYifeng | P2 | Backend/Device Issue - different output on BMG and PVC devices |  |  | aten_ops | ut |
| [2513](https://github.com/intel/torch-xpu-ops/issues/2513) | [upstream_ut]  RuntimeError: _ | open | gplutop7 | P2 | Memory/Shared Memory Issue - _share_fd_ is not available on XPU, indicating a sh |  | skipped | aten_ops | ut |
| [2518](https://github.com/intel/torch-xpu-ops/issues/2518) | [upstream_ut]  TypeError: Crea | open | astachowiczhabana | P2 |  |  | skipped | aten_ops | ut |
| [2519](https://github.com/intel/torch-xpu-ops/issues/2519) | [upstream_ut]  TypeError: map2 | open | Silv3S | P2 | Backend/Device Issue - map2_ is only implemented on CPU tensors, indicating lack |  | skipped | aten_ops | ut |
| [2537](https://github.com/intel/torch-xpu-ops/issues/2537) | Title: [upstream_ut]  Failed:  | open | PatrykWilczewski | P2 | Others - Test expects failure but passed unexpectedly, no specific error trace provided. |  | skipped, port_from_skiplist | aten_ops | ut |
| [2560](https://github.com/intel/torch-xpu-ops/issues/2560) | [UT] "RuntimeError: iter.devic | open | CuiYifeng | P2 | Backend/Device Issue - XPU device check failure in test |  | bug | aten_ops | ut |
| [2562](https://github.com/intel/torch-xpu-ops/issues/2562) | Warning as Error | open | chunhuanMeng | P2 | Others - warning treated as error but no traceback or specific error provided |  |  | aten_ops | ut |
| [2655](https://github.com/intel/torch-xpu-ops/issues/2655) | [BMG][OOB] hf_Reformer perform | open | jianyizh | P0 | Timeout/Performance Issue - hf_Reformer performance drop reported. |  | E2E, dtype: float16, triaged, performance, os: Ubuntu, hw: BMG, dependency component: Triton, regression | aten_ops | e2e |
| [2660](https://github.com/intel/torch-xpu-ops/issues/2660) | [release/2.10][Windows][BMG] N | open | tadkrawiec | P2 | Others - insufficient information to determine root cause |  | os: Windows, hw: BMG, module: ut | aten_ops | ut |
| [2662](https://github.com/intel/torch-xpu-ops/issues/2662) | [release/2.10][Windows][BMG] N | open | tadkrawiec, kdrozd-dev | P2 | Backend/Device Issue - XPU related failure in test cases on Windows with BMG |  | os: Windows, hw: BMG, module: ut | aten_ops | ut |
| [2712](https://github.com/intel/torch-xpu-ops/issues/2712) | [upstream_ut]  RuntimeError: C | open | tszulist-hbn | P2 |  |  | skipped, ut_upstream | aten_ops | ut |
| [2744](https://github.com/intel/torch-xpu-ops/issues/2744) | [Bug Skip]: extended test fail | open | pbielak | P2 | Skip/No Test Exists - test was skipped due to changes in tolerance values causin |  | skipped | aten_ops | ut |
| [2766](https://github.com/intel/torch-xpu-ops/issues/2766) | MaxPool2d - investigate memory | open | BBBela | P2 | Memory/Shared Memory Issue - investigate memory layout performance for MaxPool2d |  |  | aten_ops | ut |
| [2769](https://github.com/intel/torch-xpu-ops/issues/2769) | [oneDNN] New failed test cases | open | LuFinch | P2 | DNNL/OneDNN Issue - failed test cases related to oneDNN with 3.11 compared to 3. |  | hw: PVC, dependency component: oneDNN, module: ut | aten_ops | ut |
| [2795](https://github.com/intel/torch-xpu-ops/issues/2795) | Histc raises error with intege | open | CuiYifeng | P2 | Dtype/Precision Issue - integer input causes error with deterministic algorithm  |  |  | aten_ops | ut |
| [2797](https://github.com/intel/torch-xpu-ops/issues/2797) | Copy error is not raise on tes | open | None | P2 | Others - Copy error not raised in test_dlpack.py test case |  |  | aten_ops | ut |
| [2798](https://github.com/intel/torch-xpu-ops/issues/2798) | Test case  test/test_dlpack.py | open | None | P2 |  |  | ut_upstream | aten_ops | ut |
| [2815](https://github.com/intel/torch-xpu-ops/issues/2815) | RuntimeError: output with shap | open | PawelSwider2000 | P2 | Error - output shape mismatch during broadcasting |  | skipped, bug_fix_stage5 | aten_ops | ut |
| [2845](https://github.com/intel/torch-xpu-ops/issues/2845) | [Bug Skip]:[UT] [Windows] fail | open | None | P2 | Skip/No Test Exists - test was skipped or does not exist |  | skipped_windows | aten_ops | ut |
| [2852](https://github.com/intel/torch-xpu-ops/issues/2852) | [Bug Skip]: New UT failures in | open | None | P2 | Skip/No Test Exists - test was skipped or not present |  | skipped_windows | aten_ops | ut |
| [2858](https://github.com/intel/torch-xpu-ops/issues/2858) | [Bug Skip]: test_xpu new failu | open | None | P2 | Skip/No Test Exists - test_xpu new failures marked as skipped or no test availab |  | skipped_windows | unknown | ut |
| [2879](https://github.com/intel/torch-xpu-ops/issues/2879) | RuntimeError: _share_fd_: only | open | Silv3S | P2 | Backend/Device Issue - _share_fd_ is not available on XPU device |  | bug_fix_stage5 | aten_ops | ut |
| [2891](https://github.com/intel/torch-xpu-ops/issues/2891) | RuntimeError: Expected to find | open | chunhuanMeng | P2 |  |  | module: inductor, ut_upstream | inductor | e2e |
| [2907](https://github.com/intel/torch-xpu-ops/issues/2907) | [release/2.11] Models performa | open | xuhancn | P0 | Timeout/Performance Issue - models performance regression in testcases |  |  | aten_ops | ut |
| [2912](https://github.com/intel/torch-xpu-ops/issues/2912) | [release/2.11] UT extended 220 | open | None | P2 | Others - insufficient information to determine root cause |  | os: Windows, hw: BMG | aten_ops | ut |
| [2918](https://github.com/intel/torch-xpu-ops/issues/2918) | [XPU][upstream_ut][COW] Skip n | open | gplutop7 | P2 |  | [PR](https://github.com/pytorch/pytorch/pull/174670) | skipped, bug_fix_stage3, ut_upstream | aten_ops | ut |
| [2919](https://github.com/intel/torch-xpu-ops/issues/2919) | [XPU][upstream_ut][COW] Fix ma | open | gplutop7 | P2 |  |  | bug_fix_stage3, ut_upstream | aten_ops | ut |
| [2920](https://github.com/intel/torch-xpu-ops/issues/2920) | Failing Test Cases in Nightly  | open | Silv3S | P2 | Others - insufficient information to determine root cause |  | skipped | aten_ops | ut |
| [2930](https://github.com/intel/torch-xpu-ops/issues/2930) | [release/2.11] UT skip test_bi | open | None | P2 | Skip/No Test Exists - test is skipped due to RuntimeError |  |  | aten_ops | ut |
| [2940](https://github.com/intel/torch-xpu-ops/issues/2940) | [release/2.11] Models performa | open | jianyizh, LuFinch | P0 | Timeout/Performance Issue - Models performance dropped ~10% - 15% |  | performance | aten_ops | e2e |
| [2942](https://github.com/intel/torch-xpu-ops/issues/2942) | [Windows] Unit tests got Fatal | open | xuhancn, Stonepia | P2 | Backend/Device Issue - Fatal Python error on Windows unit tests related to XPU b |  | os: Windows | aten_ops | ut |
| [2948](https://github.com/intel/torch-xpu-ops/issues/2948) | [AO] Benchmark enabling on XPU | open | None | P2 | Backend/Device Issue - XPU benchmark enabling issue |  | module: ao, bug_fix_stage6 | AO | ut |
| [2953](https://github.com/intel/torch-xpu-ops/issues/2953) | [release/2.11][wsl] huggingfac | open | xuhancn | P0 |  |  |  | aten_ops | e2e |
| [2979](https://github.com/intel/torch-xpu-ops/issues/2979) | eca_halonext26ts got RuntimeEr | open | None | P0 | Backend/Device Issue - ZE_RESULT_ERROR_MODULE_BUILD_FAILURE indicates a problem  |  | hw: BMG, dependency component: driver | aten_ops | e2e |
| [2981](https://github.com/intel/torch-xpu-ops/issues/2981) | [release/2.11] T5 models perfo | open | jianyizh, weishi-deng | P0 |  |  | performance, os: Ubuntu, hw: BMG | aten_ops | e2e |
| [2993](https://github.com/intel/torch-xpu-ops/issues/2993) | [Bug Skip]: Unexpected success | open | gplutop7 | P2 | Skip/No Test Exists - test unexpectedly succeeded and should have been skipped |  | skipped | aten_ops | ut |
| [2999](https://github.com/intel/torch-xpu-ops/issues/2999) | KeyError: 'eager_numerics.use_ | open | daisyden | P2 |  |  | module: inductor, ut_upstream | inductor | ut |
| [3000](https://github.com/intel/torch-xpu-ops/issues/3000) | [Bug Skip]: RuntimeError: _sha | open | gplutop7 | P2 | Backend/Device Issue - _share_fd_ is not available on XPU device |  | skipped | aten_ops | ut |
| [3004](https://github.com/intel/torch-xpu-ops/issues/3004) | TypeError: _xpu_recordMemoryHi | open | guangyey | P2 |  |  | module: inductor, ut_upstream | inductor | ut |
| [3011](https://github.com/intel/torch-xpu-ops/issues/3011) | [upstream_ut] torch.OutOfMemor | open | None | P2 |  |  | bug_fix_stage5 | aten_ops | ut |
| [3024](https://github.com/intel/torch-xpu-ops/issues/3024) | Enable clang-tidy checks | open | Silv3S | P2 | Others - clang-tidy checks enablement is a code quality/linter configuration task, not a runtime or functional issue. |  | bug_fix_stage5 | aten_ops | ut |
| [3030](https://github.com/intel/torch-xpu-ops/issues/3030) | [Bug Skip] test/test_modules.p | open | gplutop7 | P2 | Skip/No Test Exists - test was skipped due to failure with no detailed error pro |  | skipped | aten_ops | ut |
| [3033](https://github.com/intel/torch-xpu-ops/issues/3033) | [Bug Skip]: Softmax tolerance | open | chunhuanMeng | P2 | Skip/No Test Exists - test is skipped due to Softmax tolerance issue |  | skipped, random | aten_ops | ut |
| [3048](https://github.com/intel/torch-xpu-ops/issues/3048) | Profiler result is not correct | open | aostrowski-hbn | P2 | Backend/Device Issue - Profiler result discrepancy on B70 device. |  | module: profiler | profiling | ut |
| [3060](https://github.com/intel/torch-xpu-ops/issues/3060) | Implement torch._scaled_groupe | open | Stonepia, liangan1 | P2 | Backend/Device Issue - Implementation required for XPU backend |  | module: quant | low_precision | ut |
| [3074](https://github.com/intel/torch-xpu-ops/issues/3074) | [Bug Skip] test_dlpack_exchang | open | AKloniecki | P2 | Skip/No Test Exists - test is skipped expecting current_work_stream is not null |  | skipped | aten_ops | ut |
| [3077](https://github.com/intel/torch-xpu-ops/issues/3077) | [Bug Skip] test_dlpack.py::Tes | open | AKloniecki | P2 |  |  | ut_upstream | aten_ops | ut |
| [3083](https://github.com/intel/torch-xpu-ops/issues/3083) | [Bug Skip]: Random failures 20 | open | None | P2 | Skip/No Test Exists - test is marked to be skipped with no valid test to execute |  | skipped, random | unknown | ut |
| [3084](https://github.com/intel/torch-xpu-ops/issues/3084) | torch.library.register_autocas | open | None | P2 | Backend/Device Issue - torch.library.register_autocast does not support XPU devi |  | module: ut | aten_ops | ut |
| [3086](https://github.com/intel/torch-xpu-ops/issues/3086) | nvml support blocks some test  | open | None | P2 | Backend/Device Issue - nvml support blocking test cases on XPU |  | module: ut | aten_ops | ut |
| [3095](https://github.com/intel/torch-xpu-ops/issues/3095) | cutlass support blocks some un | open | None | P2 |  |  | module: inductor, ut_upstream | inductor | ut |
| [3096](https://github.com/intel/torch-xpu-ops/issues/3096) | VISIBLE_DEVICE support | open | None | P2 | Backend/Device Issue - VISIBLE_DEVICE support is related to device visibility an |  | module: ut | aten_ops | ut |
| [3106](https://github.com/intel/torch-xpu-ops/issues/3106) | Worker crashes when running Te | open | BBBela | P0 |  |  | module: ut, skipped, bug_fix_stage4 | aten_ops | ut |
| [3157](https://github.com/intel/torch-xpu-ops/issues/3157) | XPU out of memory. Tried to al | open | kdrozd-dev | P2 |  |  |  | aten_ops | ut |
| [3170](https://github.com/intel/torch-xpu-ops/issues/3170) | Unskip test_bmm_windows_error_ | open | jenniew | P2 |  |  | skipped, ut_upstream | aten_ops | ut |
| [3180](https://github.com/intel/torch-xpu-ops/issues/3180) | [E2E] Timm/Torchbench models g | open | None | P0 | Backend/Device Issue - eager_two_runs_differ on ARC XPU backend |  |  | aten_ops | ut |
| [3189](https://github.com/intel/torch-xpu-ops/issues/3189) | Task Tracker | open | guangyey | P2 | Skip/No Test Exists - no test or error details provided |  |  | aten_ops | ut |
| [3196](https://github.com/intel/torch-xpu-ops/issues/3196) | vitals is not supported, the c | open | libohao1201 | P2 | 10 - vitals feature is not supported, cases should be disabled |  | skipped | aten_ops | ut |
| [3206](https://github.com/intel/torch-xpu-ops/issues/3206) | [Bug Skip]: [new failures]Runt | open | tszulist-hbn | P2 | Backend/Device Issue - Tensors are on different devices (xpu:0 vs cpu) |  | skipped | aten_ops | ut |
| [3209](https://github.com/intel/torch-xpu-ops/issues/3209) | [Win][Build] There is Cyclic d | open | Copilot | P0 | Backend/Device Issue - Cyclic dependencies during build with BUILD_SEPARATE_OPS= |  |  | aten_ops | build |
| [3216](https://github.com/intel/torch-xpu-ops/issues/3216) | [OPs] Some ops of XPU have non | open | CuiYifeng | P2 | Backend/Device Issue - XPU ops show non-determinism and inconsistency compared t |  |  | aten_ops | ut |
| [3227](https://github.com/intel/torch-xpu-ops/issues/3227) | torch xpu event has ~0.1ms lat | open | guangyey | P2 | Timeout/Performance Issue - high latency in XPU event (~0.1ms) indicates a perfo |  |  | aten_ops | ut |
| [3247](https://github.com/intel/torch-xpu-ops/issues/3247) | NotImplementedError: "dot_xpu_ | open | Silv3S | P2 |  |  | ut_upstream | aten_ops | ut |
| [3257](https://github.com/intel/torch-xpu-ops/issues/3257) | [Linux][E2E][Regression] Huggi | open | None | P0 | Backend/Device Issue - Huggingface test models failed to run on XPU, indicating  |  |  | aten_ops | e2e |

#### PT2E (#pt2e)

| ID | Title | Status | Owner | Priority | Root Cause | PR | Labels | Module | Test Module |
|---|-------|--------|-------|---------|-----------|-----|--------|--------|-------------|
| [1762](https://github.com/intel/torch-xpu-ops/issues/1762) | Add an ocloc AOT target compil | open | chunhuanMeng | P2 | compilation-related task or issue. |  | module: build | aten_ops | ut |
| [1969](https://github.com/intel/torch-xpu-ops/issues/1969) | torch._dynamo.exc.InternalTorc | open | guangyey | P2 | Error - cannot create weak reference to 'torch.Event' object |  | module: ut | aten_ops | ut |
| [1970](https://github.com/intel/torch-xpu-ops/issues/1970) | torch._dynamo.exc.BackendCompi | open | None | P2 | Backend/Device Issue - CUDA not available on the system |  | module: ut | aten_ops | ut |
| [2472](https://github.com/intel/torch-xpu-ops/issues/2472) | [upstream_ut]  NotImplementedE | open | Silv3S | P2 |  |  | wontfix, module: ut, skipped | aten_ops | ut |
| [2554](https://github.com/intel/torch-xpu-ops/issues/2554) | [upstream_ut]  AssertionError: | open | daisyden | P2 |  |  | module: inductor, skipped | inductor | ut |
| [2609](https://github.com/intel/torch-xpu-ops/issues/2609) | [upstream_ut]  torch._inductor | open | daisyden | P2 |  | [PR](https://github.com/pytorch/pytorch/pull/171154) | module: inductor, skipped, ut_upstream | inductor | ut |
| [2698](https://github.com/intel/torch-xpu-ops/issues/2698) | Title: [upstream_ut]  RuntimeE | open | chunhuanMeng, LuFinch | P2 |  |  | module: inductor, skipped, ut_upstream | inductor | ut |
| [2715](https://github.com/intel/torch-xpu-ops/issues/2715) | [upstream_ut]  torch._dynamo.e | open | CuiYifeng | P2 |  |  | skipped, ut_upstream | aten_ops | ut |
| [2767](https://github.com/intel/torch-xpu-ops/issues/2767) | [UT] test_control_flow_xpu.py  | open | PatrykWilczewski | P1 | Failure - test_control_flow_xpu.py got AssertionError |  | module: ut, skipped, bug_fix_stage5 | aten_ops | ut |
| [2806](https://github.com/intel/torch-xpu-ops/issues/2806) | CompiledAOTI need XPU support | open | daisyden | P2 |  |  | module: inductor, ut_upstream | inductor | ut |
| [2873](https://github.com/intel/torch-xpu-ops/issues/2873) | [Bug Skip]: test_repos.py cont | open | PawelSwider2000 | P2 | Skip/No Test Exists - test contains failed ops and is skipped |  | skipped | aten_ops | ut |
| [3006](https://github.com/intel/torch-xpu-ops/issues/3006) | AssertionError: '.to(tl.float1 | open | CuiYifeng | P2 |  |  | module: inductor, ut_upstream | inductor | e2e |
| [3231](https://github.com/intel/torch-xpu-ops/issues/3231) | Dynamo failed to run FX node w | open | None | P2 | Inductor/Compilation Issue - Dynamo failed to compile FX node with fake tensors  |  | module: ut, skipped | aten_ops | ut |

#### Sparse Operations Related (#sparse-operations-related)

| ID | Title | Status | Owner | Priority | Root Cause | PR | Labels | Module | Test Module |
|---|-------|--------|-------|---------|-----------|-----|--------|--------|-------------|
| [1962](https://github.com/intel/torch-xpu-ops/issues/1962) | [upstream_ut] segfault with te | open | jenniew, mengfei25 | P0 | Backend/Device Issue - segfault related to XPU device operation in test |  | dependency component: driver, module: ut, skipped | aten_ops | ut |
| [2595](https://github.com/intel/torch-xpu-ops/issues/2595) | [Bug Skip]: Random crashed cas | open | None | P0 | Skip/No Test Exists - Test was skipped due to random crashed cases. |  | skipped, random | aten_ops | ut |
| [2663](https://github.com/intel/torch-xpu-ops/issues/2663) | test_sparse_semi_structured.py | open | None | P2 |  |  | module: ut, ut_upstream | aten_ops | ut |
| [2729](https://github.com/intel/torch-xpu-ops/issues/2729) | [Bug Skip]: Random failures 20 | open | Silv3S | P2 | Skip/No Test Exists - test is marked as skipped due to random failures |  | skipped, bug_fix_stage5, random | unknown | ut |
| [2751](https://github.com/intel/torch-xpu-ops/issues/2751) | [Bug Skip]: Random failures 20 | open | None | P2 | Skip/No Test Exists - test was skipped due to random failure标记 |  | skipped, random | unknown | ut |
| [2777](https://github.com/intel/torch-xpu-ops/issues/2777) | [Bug Skip]: Random failures 20 | open | AKloniecki | P2 | Skip/No Test Exists - test is marked as a skip with no detailed error traceback  |  | skipped, random | unknown | ut |
| [2801](https://github.com/intel/torch-xpu-ops/issues/2801) | to_dense() for Sparse CSR back | open | jenniew | P2 | Error - source tensor shape mismatch during to_dense() for Sparse CSR backend |  |  | aten_ops | ut |
| [2965](https://github.com/intel/torch-xpu-ops/issues/2965) | [Bug Skip]: Random failures 20 | open | None | P2 | Skip/No Test Exists - test is marked as a skip with no valid test implementation |  | hw: PVC, skipped, random | unknown | ut |
| [3081](https://github.com/intel/torch-xpu-ops/issues/3081) | Sparse CSR gemm-like ops have  | open | None | P2 | Supported - Sparse CSR gemm-like operations are not supported yet. |  | module: ut | aten_ops | ut |
| [3165](https://github.com/intel/torch-xpu-ops/issues/3165) | test_sparse_csr_xpu.py::TestSp | open | None | P2 |  |  | skipped, ut_upstream | aten_ops | ut |
| [3167](https://github.com/intel/torch-xpu-ops/issues/3167) | NotImplementedError: Could not | open | tszulist-hbn | P1 |  |  | skipped, ut_upstream | aten_ops | ut |
| [3168](https://github.com/intel/torch-xpu-ops/issues/3168) | NotImplementedError: Could not | open | jenniew | P1 |  |  | skipped, ut_upstream | aten_ops | ut |
| [3169](https://github.com/intel/torch-xpu-ops/issues/3169) | NotImplementedError: Could not | open | jkosnox | P2 |  |  | skipped, ut_upstream | aten_ops | ut |
| [3175](https://github.com/intel/torch-xpu-ops/issues/3175) | [Bug Skip]: ValueError: sample | open | None | P2 | Backend/Device Issue - inputs are not on the same XPU device |  | skipped | aten_ops | ut |

#### TorchAO (#torchao)

| ID | Title | Status | Owner | Priority | Root Cause | PR | Labels | Module | Test Module |
|---|-------|--------|-------|---------|-----------|-----|--------|--------|-------------|
| [1833](https://github.com/intel/torch-xpu-ops/issues/1833) | [PT2E] INT8 accuracy is lower  | open | chunhuanMeng | P2 | Dtype/Precision Issue - INT8 accuracy lower than FP32 due to precision reduction |  | Accuracy, module: quant, dtype: int8 | low_precision | ut |
| [1894](https://github.com/intel/torch-xpu-ops/issues/1894) | [Linux][PT2E] performance test | open | jenniew | P1 | precision-related failure in performance test |  | module: quant | low_precision | e2e |
| [1912](https://github.com/intel/torch-xpu-ops/issues/1912) | Implement the torch.ops.aten._ | open | liangan1 | P2 | Backend/Device Issue - Implementation required for XPU dequantization of CUDA in |  | dependency component: oneDNN | aten_ops | ut |
| [1996](https://github.com/intel/torch-xpu-ops/issues/1996) | [TorchAO]  Memory Efficient Op | open | arlesniak | P2 | Memory/Shared Memory Issue - related to memory efficient optimizers in TorchAO |  | module: ao | AO | ut |
| [2006](https://github.com/intel/torch-xpu-ops/issues/2006) | work-item/workgroup issue in s | open | BartoszKokoszko | P2 | Backend/Device Issue - work-group size exceeds device limitations on XPU |  | module: ut, skipped | aten_ops | ut |
| [2201](https://github.com/intel/torch-xpu-ops/issues/2201) | [TorchAO][BMG] When using page | open | Stonepia | P2 | Failure - assert vr is not None error encountered |  | module: ao | AO | ut |
| [2202](https://github.com/intel/torch-xpu-ops/issues/2202) | [TorchAO][BMG] The RTN perform | open | Stonepia | P0 | Timeout/Performance Issue - RTN performance regression in next-token latency for |  | performance, regression, module: ao | AO | ut |
| [2207](https://github.com/intel/torch-xpu-ops/issues/2207) | Enable FP8/MXFP8 Ops with requ | open | CuiYifeng | P2 | Backend/Device Issue - FP8/MXFP8 Ops related to XPU and CUDA alignment |  | dtype: float8 | aten_ops | ut |
| [2220](https://github.com/intel/torch-xpu-ops/issues/2220) | test/test_sparse_csr.py::TestS | open | None | P2 | Backend/Device Issue - unknown SPIR-V extension 'SPV_INTEL_subgroup_matrix_multi |  | duplicate, module: dependency bug, dependency component: Triton, skipped | aten_ops | ut |
| [2309](https://github.com/intel/torch-xpu-ops/issues/2309) | unsupported ops with PYTORCH_E | open | daisyden | P2 |  |  | wontfix, module: op impl, skipped | aten_ops | ut |
| [2323](https://github.com/intel/torch-xpu-ops/issues/2323) | [TorchAO] MOE training enablin | open | riverliuintel | P2 | Backend/Device Issue - MOE training not enabled on XPU |  | module: ao | AO | ut |
| [2324](https://github.com/intel/torch-xpu-ops/issues/2324) | [TorchAO] FP8 conv support | open | Stonepia | P2 | Supported - FP8 conv is not supported yet in TorchAO |  | module: ao | AO | ut |
| [2325](https://github.com/intel/torch-xpu-ops/issues/2325) | [TorchAO] Float8 training supp | open | arlesniak, riverliuintel | P2 | Supported - Float8 training is not supported on XPU. |  | module: ao | AO | ut |
| [2326](https://github.com/intel/torch-xpu-ops/issues/2326) | [TorchAO] MX training  native  | open | riverliuintel | P2 | Backend/Device Issue - Training on XPU with TorchAO and native PyTorch is not fu |  | module: ao | AO | ut |
| [2327](https://github.com/intel/torch-xpu-ops/issues/2327) | [TorchAO] benchmark enabling o | open | None | P2 | Backend/Device Issue - XPU benchmark enabling issue in TorchAO |  |  | aten_ops | ut |
| [2358](https://github.com/intel/torch-xpu-ops/issues/2358) | test/test_view_ops.py::TestOld | open | Silv3S | P2 |  |  | Ready for merge, ut_upstream, bug_fix_stage5 | aten_ops | ut |
| [2434](https://github.com/intel/torch-xpu-ops/issues/2434) | [Bug Skip]: New failures 2025- | open | AKloniecki | P2 | Skip/No Test Exists - test is marked as a bug skip or not implemented properly |  | module: ut, skipped, bug_fix_stage4 | aten_ops | ut |
| [2508](https://github.com/intel/torch-xpu-ops/issues/2508) | TypedStorage / TypedTensors de | open | Silv3S | P1 |  | [PR](https://github.com/intel/torch-xpu-ops/pull/3260) | wontfix, skipped | aten_ops | ut |
| [2572](https://github.com/intel/torch-xpu-ops/issues/2572) | [TorchAO][UT] test/dtypes/test | open | xiaowangintel | P0 | Failure - Tensor-likes are not close! |  | module: ao | AO | build |
| [2578](https://github.com/intel/torch-xpu-ops/issues/2578) | [TorchAO][UT] test/quantizatio | open | Stonepia | P0 |  |  | module: ao, ut_upstream | AO | build |
| [2580](https://github.com/intel/torch-xpu-ops/issues/2580) | [TorchAO][UT] test/test_low_bi | open | arlesniak | P0 | Failure - Tensor-likes are not close! |  | module: ao | AO | build |
| [2596](https://github.com/intel/torch-xpu-ops/issues/2596) | [TorchAO][BMG]Observed accurac | open | None | P0 | Dtype/Precision Issue - accuracy fluctuations due to INT4 quantization methods ( |  | module: ao | AO | ut |
| [2597](https://github.com/intel/torch-xpu-ops/issues/2597) | [TorchAO][BMG] INT4 GPTQ shows | open | xiaowangintel | P2 | Timeout/Performance Issue - INT4 GPTQ shows worse performance compared with RTN  |  | module: ao | AO | ut |
| [2598](https://github.com/intel/torch-xpu-ops/issues/2598) | [TorchAO][BMG]The first token  | open | Stonepia | P2 | Timeout/Performance Issue - First token latency drops significantly with change  |  | module: ao | AO | ut |
| [2605](https://github.com/intel/torch-xpu-ops/issues/2605) | [int4][inductor] Add freezing  | open | None | P2 | Inductor/Compilation Issue - Adding freezing pattern for fusing int4 mm kernel i |  |  | aten_ops | ut |
| [2620](https://github.com/intel/torch-xpu-ops/issues/2620) | [upstream_ut]  AssertionError: | open | daisyden | P2 |  |  | module: inductor, skipped, ut_upstream | inductor | ut |
| [2707](https://github.com/intel/torch-xpu-ops/issues/2707) | [TorchAO][BMG] INT4 GPTQ faile | open | xiaowangintel | P2 | Mismatch - INT4 GPTQ failed due to TorchAO API change. |  | module: ao | AO | ut |
| [2722](https://github.com/intel/torch-xpu-ops/issues/2722) | [Bug Skip]: NotImplementedErro | open | Silv3S | P2 | Backend/Device Issue - aten::flip not implemented for QuantizedXPU backend |  | skipped, bug_fix_stage5 | aten_ops | ut |
| [2734](https://github.com/intel/torch-xpu-ops/issues/2734) | [TorchAO][BMG] INT4 RTN Flex-a | open | Stonepia, hoshibara | P2 | Timeout/Performance Issue - 50% performance drop in INT4 RTN Flex-attention with | [PR](https://github.com/pytorch/pytorch/pull/160480, https://github.com/pytorch/pytorch/pull/172316) | module: ao | AO | ut |
| [2823](https://github.com/intel/torch-xpu-ops/issues/2823) | [TorchAO][BMG] Llama-3.2-1B-In | open | xiaowangintel, lchen2331 | P2 | Timeout/Performance Issue - 20% performance drop in next token generation with D |  | module: ao | AO | ut |
| [2946](https://github.com/intel/torch-xpu-ops/issues/2946) | [Bug Skip]: Random failures 20 | open | None | P2 | Skip/No Test Exists - test is marked to be skipped with no valid test implementa |  | skipped, random | unknown | ut |
| [3032](https://github.com/intel/torch-xpu-ops/issues/3032) | [TorchAO][UT] failures in test | open | Stonepia | P0 | Others - insufficient information to determine root cause |  | module: ao | AO | build |
| [3076](https://github.com/intel/torch-xpu-ops/issues/3076) | [TorchAO][BMG] Llama-3.2-1B-In | open | None | P2 | DNNL/OneDNN Issue - Performance drop with oneDNN 3.11.1 in Dynamic INT8 for Llam |  |  | aten_ops | ut |
| [3088](https://github.com/intel/torch-xpu-ops/issues/3088) | [TorchAO][BMG] INT4 RTN Flex-a | open | Stonepia | P2 | Timeout/Performance Issue - INT4 RTN Flex-attention experienced a 5% performance |  | module: ao | AO | ut |
| [3089](https://github.com/intel/torch-xpu-ops/issues/3089) | AssertionError: Torch not comp | open | jmamzax | P2 |  |  | bug_fix_stage5 | unknown | ut |
| [3124](https://github.com/intel/torch-xpu-ops/issues/3124) | [TorchAO][Bug] ImportError: Re | open | None | P2 | Backend/Device Issue - Requires mslk >= 1.0.0 for XPU support with int4 quantiza |  |  | aten_ops | ut |
| [3166](https://github.com/intel/torch-xpu-ops/issues/3166) | test_consistency_SparseCSR fai | open | yucai-intel | P2 |  |  | skipped, ut_upstream | aten_ops | ut |
| [3255](https://github.com/intel/torch-xpu-ops/issues/3255) | [Linux][PT2E][Regression] Some | open | None | P0 | Timeout/Performance Issue - performance tests failed due to regression in execut |  |  | aten_ops | e2e |


---

## 6. Duplicated Issues {#6-duplicated-issues}

Issues that share test cases with other issues.

| ID | Title | Owner | Reporter | Duplicated With | Priority | Root Cause | PR | Labels | Module | Test Module |
|---|-------|-------|----------|-----------------|---------|-----------|-----|--------|--------|-------------|
| [1893](https://github.com/intel/torch-xpu-ops/issues/1893) | [upstream_ut] oneDNN accuracy  | chunhuanMeng | daisyden | 1951 | P2 |  |  | skipped, ut_upstream | aten_ops | ut |
| [1951](https://github.com/intel/torch-xpu-ops/issues/1951) | Functionality issues in TestCo | AKloniecki | daisyden | 1893 | P2 |  |  | module: ut, skipped, ut_upstream | aten_ops | ut |
| [1973](https://github.com/intel/torch-xpu-ops/issues/1973) | AssertionError: Scalars or Ten | gplutop7 | mengfei25 | 2837,2840 | P2 | Failure - Tensor values not equal or close in test_depthwise_conv_64bit_indexing_xpu |  | hw: PVC, module: ut, skipped, bug_fix_stage4 | aten_ops | ut |
| [2006](https://github.com/intel/torch-xpu-ops/issues/2006) | work-item/workgroup issue in s | BartoszKokoszko | daisyden | 2257 | P2 | Backend/Device Issue - work-group size exceeds device limitations on XPU |  | module: ut, skipped | aten_ops | ut |
| [2015](https://github.com/intel/torch-xpu-ops/issues/2015) | inf is returned by nn.Transfor | yucai-intel | daisyden | 2186,2529 | P2 | Dtype/Precision Issue - inf returned using float16 in TransformerEncoderLayer on |  | skipped | aten_ops | ut |
| [2186](https://github.com/intel/torch-xpu-ops/issues/2186) | AssertionError: Mul tiheadAtte | daisyden | daisyden | 2015 | P2 | Failure - Mul tiheadAttention does not support NestedTensor outside of its fast path |  | dependency component: oneDNN | aten_ops | ut |
| [2220](https://github.com/intel/torch-xpu-ops/issues/2220) | test/test_sparse_csr.py::TestS | None | wincent8 | 2246 | P2 | Backend/Device Issue - unknown SPIR-V extension 'SPV_INTEL_subgroup_matrix_multi |  | duplicate, module: dependency bug, dependency component: Triton, skipped | aten_ops | ut |
| [2230](https://github.com/intel/torch-xpu-ops/issues/2230) | test_sparse_csr.py::TestSparse | None | wincent8 | 2246,3175,3176 | P1 | Backend/Device Issue - inputs are not on the same GPU device |  | skipped | aten_ops | ut |
| [2235](https://github.com/intel/torch-xpu-ops/issues/2235) | test/test_sparse_csr.py::TestS | None | wincent8 | 3047 | P2 | Backend/Device Issue - unexpected warning from non-optimal triton kernel paramet |  | skipped | aten_ops | ut |
| [2238](https://github.com/intel/torch-xpu-ops/issues/2238) | Exception: Tensor-likes are no | BBBela | zxd1997066 | 3105 | P2 |  | [PR](https://github.com/intel/torch-xpu-ops/pull/2886) | skipped, bug_fix_stage3 | aten_ops | ut |
| [2244](https://github.com/intel/torch-xpu-ops/issues/2244) | test/test_sparse_csr.py::TestS | jenniew | wincent8 | 3177 | P2 | Error - empty_sparse_compressed expects non-block layout but received SparseBsr |  | module: ut, skipped | aten_ops | ut |
| [2246](https://github.com/intel/torch-xpu-ops/issues/2246) | torch/sparse/_triton_ops*.py n | None | wincent8 | 2220,2230 | P1 | Backend/Device Issue - Torch not compiled with CUDA enabled for Intel GPU suppor |  | skipped | unknown | ut |
| [2253](https://github.com/intel/torch-xpu-ops/issues/2253) | the supported dtypes are not a | daisyden | daisyden | 2482 | P2 |  |  | duplicate, skipped, ut_upstream | aten_ops | ut |
| [2257](https://github.com/intel/torch-xpu-ops/issues/2257) | Accuracy failures in test/xpu/ | pbielak | zxd1997066 | 2006 | P1 |  | [PR](https://github.com/intel/torch-xpu-ops/pull/2808) | skipped, bug_fix_stage4 | aten_ops | ut |
| [2270](https://github.com/intel/torch-xpu-ops/issues/2270) | Backend Compatibility Error in | LuFinch | libohao1201 | 2442 | P2 | Flash Attention/Specific Ops Issue - test involves aten._flash_attention_forward |  | module: ut, skipped | aten_ops | ut |
| [2285](https://github.com/intel/torch-xpu-ops/issues/2285) | Support efficient attention | chunhuanMeng | daisyden | 2358 | P2 | Backend/Device Issue - aten::_efficient_attention_forward not supported on CPU b |  | skipped | aten_ops | ut |
| [2358](https://github.com/intel/torch-xpu-ops/issues/2358) | test/test_view_ops.py::TestOld | Silv3S | wincent8 | 2285 | P2 |  |  | Ready for merge, ut_upstream, bug_fix_stage5 | aten_ops | ut |
| [2436](https://github.com/intel/torch-xpu-ops/issues/2436) | [upstream_ut]  AttributeError: | daisyden | daisyden | 2675 | P1 | Error - 'NoneType' object has no attribute 'clone' due to missing object handling |  | skipped, dependency component: community, random | aten_ops | ut |
| [2442](https://github.com/intel/torch-xpu-ops/issues/2442) | [Bug Skip]: NotImplementedErro | daisyden, LuFinch | CuiYifeng | 2270 | P2 | Flash Attention/Specific Ops Issue - aten::_flash_attention_forward not implemen |  | skipped, not_target | aten_ops | ut |
| [2482](https://github.com/intel/torch-xpu-ops/issues/2482) | test_dtypes issue introduced b | daisyden | daisyden | 2253 | P2 | Dtype/Precision Issue - test_dtypes issue related to data types in conv_transpos |  | skipped | aten_ops | ut |
| [2529](https://github.com/intel/torch-xpu-ops/issues/2529) | [upstream_ut]  AssertionError: | Silv3S | daisyden | 2015,3136 | P2 | Failure - test assertion failed with False is not true |  | skipped, port_from_skiplist | aten_ops | ut |
| [2530](https://github.com/intel/torch-xpu-ops/issues/2530) | Title: [upstream_ut]  Assertio | PatrykWilczewski | daisyden | 2817 | P2 | Failure - RuntimeError not raised as expected in test |  | skipped, bug_fix_stage5, port_from_skiplist | aten_ops | ut |
| [2611](https://github.com/intel/torch-xpu-ops/issues/2611) | [upstream_ut]  AssertionError: | daisyden | daisyden | 2613 | P2 |  |  | dependency component: driver, module: inductor, skipped, ut_upstream | inductor | ut |
| [2613](https://github.com/intel/torch-xpu-ops/issues/2613) | [upstream_ut]  AssertionError: | daisyden | daisyden | 2611 | P2 |  |  | dependency component: driver, module: inductor, skipped, ut_upstream | inductor | ut |
| [2618](https://github.com/intel/torch-xpu-ops/issues/2618) | [Bug Skip]: [regression] Asser | jmamzax | kaileiyx | 3089 | P0 |  | [PR](https://github.com/numpy/numpy/pull/22525) | skipped, bug_fix_stage5 | unknown | ut |
| [2675](https://github.com/intel/torch-xpu-ops/issues/2675) | [Bug Skip]: AttributeError: 'N | pponikox | kaileiyx | 2436 | P2 | Error - 'NoneType' object has no attribute 'clone' due to missing object reference |  | skipped, bug_fix_stage5 | aten_ops | ut |
| [2817](https://github.com/intel/torch-xpu-ops/issues/2817) | Expected error message is diff | kdrozd-dev | Silv3S | 2530 | P2 | Failure - mismatch between expected and actual error message |  | skipped, bug_fix_stage5 | aten_ops | ut |
| [2837](https://github.com/intel/torch-xpu-ops/issues/2837) | Accuracy issue for Muon optimi | Silv3S | kdrozd-dev | 1973 | P2 | Failure - Tensor-likes not close in Muon optimizer test |  | skipped, bug_fix_stage5 | aten_ops | ut |
| [2840](https://github.com/intel/torch-xpu-ops/issues/2840) | Accuracy issue with 64 bit ind | SlawomirLaba, Silv3S | kdrozd-dev | 1973 | P2 | Dtype/Precision Issue - Tensor comparison failed due to small absolute differenc |  | skipped, bug_fix_stage5 | aten_ops | ut |
| [2869](https://github.com/intel/torch-xpu-ops/issues/2869) | [Bug Skip]: New UT failure in  | None | RUIJIEZHONG66166 | 3160 | P2 | Skip/No Test Exists - Test is marked as skipped or not executed |  | skipped_windows | aten_ops | ut |
| [2966](https://github.com/intel/torch-xpu-ops/issues/2966) | [Bug Skip]: [Regression]2026-3 | jmamzax | kaileiyx | 3114 | P0 | Timeout/Performance Issue - Example code timed out during test execution. |  | skipped, bug_fix_stage5, random | aten_ops | ut |
| [3047](https://github.com/intel/torch-xpu-ops/issues/3047) | [Bug Skip]: [Regression]UT fai | None | kaileiyx | 2235 | P0 | Failure - Torch not compiled with CUDA enabled assertion error |  | skipped | unknown | ut |
| [3089](https://github.com/intel/torch-xpu-ops/issues/3089) | AssertionError: Torch not comp | jmamzax | jmamzax | 2618 | P2 |  |  | bug_fix_stage5 | unknown | ut |
| [3105](https://github.com/intel/torch-xpu-ops/issues/3105) | Wrong results from oneDNN conv | BBBela | BBBela | 2238 | P2 | DNNL/OneDNN Issue - wrong results from oneDNN conv2d kernels due to data pointer |  | hw: PVC, module: ut, skipped | aten_ops | ut |
| [3114](https://github.com/intel/torch-xpu-ops/issues/3114) | [Bug Skip]: Failure skip on 20 | None | guangyey | 2966 | P2 | Skip/No Test Exists - test was skipped on 2026-3-21 |  | skipped, random | aten_ops | ut |
| [3136](https://github.com/intel/torch-xpu-ops/issues/3136) | [upstream_ut]  AssertionError: | LuFinch | daisyden | 2529 | P2 |  |  | module: ut, skipped, ut_upstream | aten_ops | ut |
| [3156](https://github.com/intel/torch-xpu-ops/issues/3156) | AssertionError: 'Assertion cur | kdrozd-dev | kdrozd-dev | 3184 | P2 | Failure - Assertion cur_target >= 0 && cur_target < n_classes failed in cross entropy loss test |  |  | aten_ops | ut |
| [3160](https://github.com/intel/torch-xpu-ops/issues/3160) | compiler not found (Windows) | kdrozd-dev | kdrozd-dev | 2869 | P2 |  |  | os: Windows | aten_ops | ut |
| [3175](https://github.com/intel/torch-xpu-ops/issues/3175) | [Bug Skip]: ValueError: sample | None | CuiYifeng | 2230 | P2 | Backend/Device Issue - inputs are not on the same XPU device |  | skipped | aten_ops | ut |
| [3176](https://github.com/intel/torch-xpu-ops/issues/3176) | [Bug Skip]: ValueError: _scale | None | CuiYifeng | 2230 | P2 | Backend/Device Issue - inputs are not on the same XPU device |  | skipped | aten_ops | ut |
| [3177](https://github.com/intel/torch-xpu-ops/issues/3177) | Accuracy gap of BF16/FP16 test | jenniew | CuiYifeng | 2244 | P2 | Dtype/Precision Issue - Accuracy gap in BF16/FP16 test |  | skipped | aten_ops | ut |
| [3184](https://github.com/intel/torch-xpu-ops/issues/3184) | New failing UTs: test_cross_en | wpietka | BBBela | 3156 | P2 | Failure - test expects a specific condition to be true but it failed during execution. |  | module: ut, skipped | aten_ops | ut |

---

## 7. Issues with Dependency {#7-issues-with-dependency}

Issues that have dependencies on other components.

| ID | Title | Owner | Priority | Root Cause | Dependency | Category | PR | Labels | Test Module |
|---|-------|-------|---------|-----------|------------|----------|-----|--------|-------------|
| [1547](https://github.com/intel/torch-xpu-ops/issues/1547) | [distributed] NotImplemen | Chao1Han | P2 | Backend/Device Issue - XPU device does not support the 'symm_mem::fused_matmul_r | oneAPI | Distributed |  | module: distributed, dependency component: oneAPI | ut |
| [1551](https://github.com/intel/torch-xpu-ops/issues/1551) | [distributed] NotImplemen | Chao1Han | P2 | Backend/Device Issue - XPU device does not support the 'symm_mem::fused_scaled_m | oneAPI | Distributed |  | module: distributed, dependency component: oneAPI | ut |
| [1556](https://github.com/intel/torch-xpu-ops/issues/1556) | [distributed] NotImplemen | pkourdis | P2 | Distributed/Gloo Issue - Operator lacks a sharding strategy in distributed conte | oneDNN | Distributed |  | module: distributed, dependency component: oneDNN | ut |
| [1912](https://github.com/intel/torch-xpu-ops/issues/1912) | Implement the torch.ops.a | liangan1 | P2 | Backend/Device Issue - Implementation required for XPU dequantization of CUDA in | oneDNN | TorchAO |  | dependency component: oneDNN | ut |
| [2089](https://github.com/intel/torch-xpu-ops/issues/2089) | need an implementation th | guangyey | P2 | Backend/Device Issue - torch.xpu.is_available() initializes GPU context unintent | driver | Others |  | dependency component: driver | ut |
