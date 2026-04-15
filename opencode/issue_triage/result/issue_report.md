# Torch XPU Ops Issue Report

**Generated:** 2026-04-15 05:16:40
**Total Issues:** 384

---

## Index

- [1. Summary](#1-summary) - 384 issues
- [2. Action Required](#2-action-required)
    - [2.1.1. No specific action identified - needs investigation - Developer](#2.1-1-no-specific-action-identified---needs-in) - 68
    - [2.1.2. Bug/Perf issue awaiting reporter response - Developer](#2.1-2-bug-perf-issue-awaiting-reporter-respons) - 47
    - [2.1.3. Feature Requests - Developer](#2.1-3-feature-requests) - 27
    - [2.2.1. Needs Upstream Skip PR - Reporter](#2.2-1-needs-upstream-skip-pr) - 76
    - [2.2.2. add to skiplist - Reporter](#2.2-2-add-to-skiplist) - 5
    - [2.2.3. Close fixed issue - Reporter](#2.2-3-close-fixed-issue) - 4
    - [2.2.4. Verify the issue - Reporter](#2.2-4-verify-the-issue) - 3
    - [2.2.5. Awaiting response from reporter - Reporter](#2.2-5-awaiting-response-from-reporter) - 143
    - [2.2.6. E2E accuracy issue - Reporter](#2.2-6-e2e-accuracy-issue) - 11
- [3. Issues by Category](#3-issues-by-category) - 384 issues
- [4. Last Week Issues](#4-last-week-issues) - 5 issues
- [5. Stale Issues - No Update 2+ Weeks](#5-stale-issues) - 229 issues
- [6. Dependency Issues](#6-dependency-issues) - 384 issues
- [7. Duplicated Issues](#7-duplicated-issues) - 14 issues
- [8. Statistics](#8-statistics)


---

## <span id='1-summary'>1. Summary</span>

**Total Issues: 384**

### <span id='category-summary'>Issues by Category</span>

| Category | Count |
|----------|------:|
| Distributed | 37 |
| Dtype / Precision Related | 40 |
| Flash Attention / Transformer Related | 17 |
| Inductor / Compilation Related | 30 |
| Others | 214 |
| PT2E | 7 |
| Sparse Operations Related | 13 |
| TorchAO | 26 |


---

## <span id='2-action-required'>2. Action Required</span>

### <span id='action-required-developer'>2.1 Developer AR (Need Investigation by Action Reason)</span>

*Issues pending investigation, grouped by type of action needed - Developer*

#### <span id='2.1-1-no-specific-action-identified---needs-in'>2.1.1 No specific action identified - needs investigation - Developer</span> (68 issues)

| # | ID | Title | Priority | Priority Reason | Action Reason | Owner Transfer | Action TBD | Category | Test Module |
|---|------|------|----------|--------------------------------------------|----------|----------|------------|------------|-----------|
| 1 | 3306 | [distributed] no attribute '_reset_fr_recording_xccl' in... | P2 | UT issue with few failures | No specific action identified - needs investigation | None | Need Investigation | Distributed | ut |
| 2 | 3305 | [distributed] shrink operation support in... | P2 | UT issue with few failures | No specific action identified - needs investigation | None | Need Investigation | Distributed | ut |
| 3 | 3300 | [CI] When creating PR, several pull workflows are launched... | P2 | UT issue with few failures | No specific action identified - needs investigation | None | Need Investigation | Others | ut |
| 4 | 3266 | [RFC] Migrate XPU kernel math functions from std::/:: to... | P2 | UT issue with few failures | No specific action identified - needs investigation | None | Need Investigation | Others | ut |
| 5 | 3216 | [OPs] Some ops of XPU have non-determinism and are... | P2 | UT issue with few failures | No specific action identified - needs investigation | CuiYifeng | Need Investigation | Others | ut |
| 6 | 3196 | vitals is not supported, the cases should be disabled | P2 | UT issue with few failures | No specific action identified - needs investigation | libohao1201 | Need Investigation | Others | ut |
| 7 | 3194 | Incorrect strides in TestCommonXPU,test_out_addmv_xpu_float32 | P2 | UT issue with few failures | No specific action identified - needs investigation | AKloniecki | Need Investigation | Others | ut |
| 8 | 3180 | [E2E] Timm/Torchbench models got "eager_two_runs_differ" on ARC | P0 | Impacts customer custom model/application | No specific action identified - needs investigation | None | Need Investigation | Others | ut |
| 9 | 3103 | Tensor-likes are not equal for test_backward_nn_functional_con... | P2 | UT issue with few failures | No specific action identified - needs investigation | BBBela | Need Investigation | Dtype / Precision Related | ut |
| 10 | 3100 | [distributed] /handler/dump_nccl_trace_pickle and nccl_log... | P2 | UT issue with few failures | No specific action identified - needs investigation | None | Need Investigation | Distributed | ut |
| 11 | 3096 | VISIBLE_DEVICE support | P2 | UT issue with few failures | No specific action identified - needs investigation | None | Need Investigation | Others | ut |
| 12 | 3093 | XPU does not support NestedTensor for SDPA operations. | P2 | UT issue with few failures | No specific action identified - needs investigation | None | Need Investigation | Flash Attention / Transformer Related | ut |
| 13 | 3086 | nvml support blocks some test cases | P2 | UT issue with few failures | No specific action identified - needs investigation | None | Need Investigation | Others | ut |
| 14 | 3084 | torch.library.register_autocast does not support xpu | P2 | UT issue with few failures | No specific action identified - needs investigation | None | Need Investigation | Dtype / Precision Related | ut |
| 15 | 3082 | multithread support in distributed | P2 | UT issue with few failures | No specific action identified - needs investigation | None | Need Investigation | Distributed | ut |
| 16 | 3081 | Sparse CSR gemm-like ops have not been supported yet | P2 | UT issue with few failures | No specific action identified - needs investigation | tszulist-hbn | Need Investigation | Sparse Operations Related | ut |
| 17 | 3080 | cudagraph tests blocked by feature gap | P2 | UT issue with few failures | No specific action identified - needs investigation | None | Need Investigation | Others | ut |
| 18 | 3048 | Profiler result is not correct on B70 | P2 | UT issue with few failures | No specific action identified - needs investigation | aostrowski-hbn | Need Investigation | Others | ut |
| 19 | 3024 | Enable clang-tidy checks | P2 | UT issue with few failures | No specific action identified - needs investigation | Silv3S | Need Investigation | Others | ut |
| 20 | 3022 | [distributed] batch_isend_irecv Compatibility Issue on B60/XCCL | P2 | UT issue with few failures | No specific action identified - needs investigation | zhangxiaoli73 | Need Investigation | Distributed | ut |
| 21 | 2950 | SYCL compilation flag -fsycl-id-queries-fit-in-int does not... | P2 | UT issue with few failures | No specific action identified - needs investigation | BBBela | Need Investigation | Others | ut |
| 22 | 2948 | [AO] Benchmark enabling on XPU | P2 | UT issue with few failures | No specific action identified - needs investigation | None | Need Investigation | Others | ut |
| 23 | 2816 | torch.logcumsumexp incorrectly returns NaNs for complex64 input | P2 | UT issue with few failures | No specific action identified - needs investigation | Silv3S | Need Investigation | Others | ut |
| 24 | 2700 | [distributed] Hang issues with test_distributed_spawn.py | P2 | UT issue with few failures | No specific action identified - needs investigation | syedshahbaaz | Need Investigation | Distributed | ut |
| 25 | 2649 | [distributed][pipelining] test_schedule_multiproc.py hang issue | P2 | UT issue with few failures | No specific action identified - needs investigation | syedshahbaaz | Need Investigation | Distributed | ut |
| 26 | 2640 | random issue test_vjpvjp_index_reduce_prod_xpu_float32 | P2 | UT issue with few failures | No specific action identified - needs investigation | wpietka | Need Investigation | Dtype / Precision Related | ut |
| 27 | 2605 | [int4][inductor] Add freezing pattern for fusing int4 mm... | P2 | UT issue with few failures | No specific action identified - needs investigation | None | Need Investigation | TorchAO | ut |
| 28 | 2471 | test_cuda.py gaps | P2 | UT issue with few failures | No specific action identified - needs investigation | guangyey | Need Investigation | Others | ut |
| 29 | 2467 | Host may stuck when submit too many kernels when event recording | P2 | UT issue with few failures | No specific action identified - needs investigation | jianyizh | Need Investigation | Others | ut |
| 30 | 2465 | [windows] ut hang | P2 | UT issue with few failures | No specific action identified - needs investigation | tadkrawiec, mganczarenko | Need Investigation | Others | ut |
| 31 | 2349 | [oneAPI][backward compatibility] libur_loader.so.0: version... | P2 | UT issue with few failures | No specific action identified - needs investigation | riverliuintel | Need Investigation | Others | ut |
| 32 | 2327 | [TorchAO] benchmark enabling on XPU | P2 | UT issue with few failures | No specific action identified - needs investigation | None | Need Investigation | TorchAO | ut |
| 33 | 2326 | [TorchAO] MX training native PyTorch on XPU | P2 | UT issue with few failures | No specific action identified - needs investigation | riverliuintel | Need Investigation | TorchAO | ut |
| 34 | 2325 | [TorchAO] Float8 training support on XPU | P2 | UT issue with few failures | No specific action identified - needs investigation | arlesniak, riverliuintel | Need Investigation | TorchAO | ut |
| 35 | 2324 | [TorchAO] FP8 conv support | P2 | UT issue with few failures | No specific action identified - needs investigation | Stonepia | Need Investigation | TorchAO | ut |
| 36 | 2323 | [TorchAO] MOE training enabling on XPU | P2 | UT issue with few failures | No specific action identified - needs investigation | riverliuintel | Need Investigation | TorchAO | ut |
| 37 | 2261 | [xpu][profiler] Run with fork process has extra warning | P2 | UT issue with few failures | No specific action identified - needs investigation | moksiuc | Need Investigation | Others | ut |
| 38 | 2250 | Found mismatch when comparing the output of aten.view.default... | P2 | UT issue with few failures | No specific action identified - needs investigation | astachowiczhabana | Need Investigation | Others | ut |
| 39 | 2246 | torch/sparse/_triton_ops*.py need to be ported to enable for... | P2 | UT issue with few failures | No specific action identified - needs investigation | None | Need Investigation | Sparse Operations Related | ut |
| 40 | 2235 | test/test_sparse_csr.py::TestSparseCompressedTritonKernelsXPU:... | P2 | UT issue with few failures | No specific action identified - needs investigation | None | Need Investigation | Sparse Operations Related | ut |
| 41 | 2232 | sdpa backward kernel is required to reduce memory usage | P2 | UT issue with few failures | No specific action identified - needs investigation | None | Need Investigation | Flash Attention / Transformer Related | ut |
| 42 | 2229 | test/test_sparse_csr.py::TestSparseCompressedCPU::test_invalid... | P2 | UT issue with few failures | No specific action identified - needs investigation | jenniew | Need Investigation | Sparse Operations Related | ut |
| 43 | 2220 | test/test_sparse_csr.py::TestSparseCompressedTritonKernelsXPU:... | P2 | UT issue with few failures | No specific action identified - needs investigation | None | Need Investigation | Sparse Operations Related | ut |
| 44 | 2215 | Find use case example for torch-xpu-ops.lib in sycl cpp extension | P2 | UT issue with few failures | No specific action identified - needs investigation | dvrogozh | Need Investigation | Others | ut |
| 45 | 2207 | Enable FP8/MXFP8 Ops with requests and CUDA alignment | P2 | UT issue with few failures | No specific action identified - needs investigation | Stonepia, CuiYifeng, LuFinch | Need Investigation | TorchAO | ut |
| 46 | 2199 | Fix reduction and norm register spill | P2 | UT issue with few failures | No specific action identified - needs investigation | None | Need Investigation | Others | ut |
| 47 | 2196 | Fix DistributionElementwiseKernelFunctor register spill | P2 | UT issue with few failures | No specific action identified - needs investigation | None | Need Investigation | Others | ut |
| 48 | 2163 | 3 distributed UT cases need to be supported by -... | P2 | UT issue with few failures | No specific action identified - needs investigation | githubsgi | Need Investigation | Distributed | ut |
| 49 | 2142 | XPU max_memory_allocated have different output with CUDA | P2 | UT issue with few failures | No specific action identified - needs investigation | guangyey | Need Investigation | Others | ut |
| 50 | 2140 | Consider how to avoid copy in FFT kernels | P2 | UT issue with few failures | No specific action identified - needs investigation | CuiYifeng | Need Investigation | Others | ut |
| 51 | 2127 | Path Coverage enhancement | P2 | UT issue with few failures | No specific action identified - needs investigation | CuiYifeng | Need Investigation | Others | ut |
| 52 | 2113 | Update example for Distributed Data Parallel | P2 | UT issue with few failures | No specific action identified - needs investigation | songhappy | Need Investigation | Distributed | ut |
| 53 | 2086 | nd_item::barrier has been deprecated | P2 | UT issue with few failures | No specific action identified - needs investigation | dvrogozh | Need Investigation | Others | ut |
| 54 | 2063 | Avoid using out-of-date term | P2 | UT issue with few failures | No specific action identified - needs investigation | CuiYifeng | Need Investigation | Others | ut |
| 55 | 2015 | inf is returned by nn.TransformerEncoderLayer | P2 | UT issue with few failures | No specific action identified - needs investigation | yucai-intel | Need Investigation | Others | ut |
| 56 | 2006 | work-item/workgroup issue in softmax/unsampling/nonzero | P2 | UT issue with few failures | No specific action identified - needs investigation | BartoszKokoszko | Need Investigation | Others | ut |
| 57 | 1996 | [TorchAO] Memory Efficient Optimizers | P2 | UT issue with few failures | No specific action identified - needs investigation | None | Need Investigation | TorchAO | ut |
| 58 | 1986 | torch.xpu._sleep is missing, | P2 | UT issue with few failures | No specific action identified - needs investigation | guangyey | Need Investigation | Others | ut |
| 59 | 1856 | channel last aten::hardswish_ will call extra copy | P2 | UT issue with few failures | No specific action identified - needs investigation | chunhuanMeng | Need Investigation | Others | ut |
| 60 | 1762 | Add an ocloc AOT target compilation test in cmake | P2 | UT issue with few failures | No specific action identified - needs investigation | chunhuanMeng | Need Investigation | PT2E | ut |
| 61 | 1729 | Validation Check List | P2 | UT issue with few failures | No specific action identified - needs investigation | chuanqi129 | Need Investigation | Others | ut |
| 62 | 1722 | Ask an API to query GPU type(iGPU/dGPU). | P2 | UT issue with few failures | No specific action identified - needs investigation | guangyey | Need Investigation | Others | ut |
| 63 | 1689 | [For op Perf Comparison] Save reference comparison run id | P2 | UT issue with few failures | No specific action identified - needs investigation | None | Need Investigation | Others | ut |
| 64 | 1645 | [For Comparison] Save reference comparison run id | P2 | UT issue with few failures | No specific action identified - needs investigation | mengfei25 | Need Investigation | Others | ut |
| 65 | 1594 | Keep track on the building warning | P0 | Build crash - critical blocking issue | No specific action identified - needs investigation | CuiYifeng, chunhuanMeng | Need Investigation | Others | ut |
| 66 | 1587 | Keep track on the latest CUDA op impl | P2 | UT issue with few failures | No specific action identified - needs investigation | CuiYifeng, yucai-intel | Need Investigation | Others | ut |
| 67 | 1059 | SYCL RT: Using recommended shortcut API for kernel specific... | P2 | UT issue with few failures | No specific action identified - needs investigation | CuiYifeng, jianyizh | Need Investigation | Others | ut |
| 68 | 146 | Evaluate register spill in SYCL kernel | P2 | UT issue with few failures | No specific action identified - needs investigation | CuiYifeng, jianyizh, mengfei25 | Need Investigation | Others | ut |

#### <span id='2.1-2-bug-perf-issue-awaiting-reporter-respons'>2.1.2 Bug/Perf issue awaiting reporter response - Developer</span> (47 issues)

| # | ID | Title | Priority | Priority Reason | Action Reason | Owner Transfer | Action TBD | Category | Test Module |
|---|------|------|----------|--------------------------------------------|----------|----------|------------|------------|-----------|
| 1 | 3267 | New failed test cases 2026-04-06 | P2 | UT issue with few failures | Bug/Perf issue awaiting reporter response | zxd1997066 | Need Investigation | Others | ut |
| 2 | 3246 | AssertionError: Booleans mismatch: True is not False | P2 | UT issue with few failures | Bug/Perf issue awaiting reporter response | Silv3S | Need Investigation | Others | ut |
| 3 | 3243 | AssertionError: False is not true | P2 | UT issue with few failures | Bug/Perf issue awaiting reporter response | zxd1997066 | Need Investigation | Others | ut |
| 4 | 3242 | AssertionError: Torch not compiled with CUDA enabled | P2 | UT issue with few failures | Bug/Perf issue awaiting reporter response | zxd1997066 | Need Investigation | Others | ut |
| 5 | 3184 | New failing UTs: test_cross_entropy_loss_2d_out_of_bounds_clas... | P2 | UT issue with few failures | Bug/Perf issue awaiting reporter response | BBBela | Need Investigation | Others | ut |
| 6 | 3178 | New failed test cases 2026-03-25 | P2 | UT issue with few failures | Bug/Perf issue awaiting reporter response | BBBela | Need Investigation | Others | ut |
| 7 | 3177 | Accuracy gap of BF16/FP16 test_block_addmm | P2 | UT issue with few failures | Bug/Perf issue awaiting reporter response | CuiYifeng | Need Investigation | Dtype / Precision Related | ut |
| 8 | 3176 | [Bug Skip]: ValueError: _scaled_dot_product_attention(): all... | P2 | UT issue with few failures | Bug/Perf issue awaiting reporter response | CuiYifeng | Need Investigation | Flash Attention / Transformer Related | ut |
| 9 | 3175 | [Bug Skip]: ValueError: sampled_addmm(): all inputs are... | P2 | UT issue with few failures | Bug/Perf issue awaiting reporter response | CuiYifeng | Need Investigation | Others | ut |
| 10 | 3121 | [Bug Skip]: CUDA specific UT test_fft_half_and_chalf_not_power... | P2 | UT issue with few failures | Bug/Perf issue awaiting reporter response | CuiYifeng | Need Investigation | Others | ut |
| 11 | 3089 | AssertionError: Torch not compiled with CUDA enabled | P2 | UT issue with few failures | Bug/Perf issue awaiting reporter response | jmamzax | Need Investigation | Inductor / Compilation Related | ut |
| 12 | 3033 | [Bug Skip]: Softmax tolerance | P2 | UT issue with few failures | Bug/Perf issue awaiting reporter response | chunhuanMeng | Need Investigation | Others | ut |
| 13 | 3025 | New failing test in Nightly Wheel... | P2 | UT issue with few failures | Bug/Perf issue awaiting reporter response | BBBela | Need Investigation | Others | ut |
| 14 | 3000 | [Bug Skip]: RuntimeError: _share_fd_: only available on CPU... | P2 | UT issue with few failures | Bug/Perf issue awaiting reporter response | zxd1997066 | Need Investigation | Others | ut |
| 15 | 2965 | [Bug Skip]: Random failures 2026WW10 | P2 | UT issue with few failures | Bug/Perf issue awaiting reporter response | CuiYifeng | Need Investigation | Others | ut |
| 16 | 2921 | [abs][complex64] - new failing test cases caused by PyTorch... | P2 | UT issue with few failures | Bug/Perf issue awaiting reporter response | BBBela | Need Investigation | Others | ut |
| 17 | 2879 | RuntimeError: _share_fd_: only available on CPU | P2 | UT issue with few failures | Bug/Perf issue awaiting reporter response | Silv3S | Need Investigation | Others | ut |
| 18 | 2869 | [Bug Skip]: New UT failure in 0209 nightly windows. | P2 | UT issue with few failures | Bug/Perf issue awaiting reporter response | RUIJIEZHONG66166 | Need Investigation | Others | ut |
| 19 | 2840 | Accuracy issue with 64 bit indexing depthwise_conv | P2 | UT issue with few failures | Bug/Perf issue awaiting reporter response | kdrozd-dev | Need Investigation | Dtype / Precision Related | ut |
| 20 | 2837 | Accuracy issue for Muon optimizer | P2 | UT issue with few failures | Bug/Perf issue awaiting reporter response | kdrozd-dev | Need Investigation | Dtype / Precision Related | ut |
| 21 | 2817 | Expected error message is different than actual | P2 | UT issue with few failures | Bug/Perf issue awaiting reporter response | Silv3S | Need Investigation | Others | ut |
| 22 | 2815 | RuntimeError: output with shape [2] doesn't match the... | P2 | UT issue with few failures | Bug/Perf issue awaiting reporter response | Silv3S | Need Investigation | Others | ut |
| 23 | 2783 | [Bug Skip]: Key "xpu" is missing from dict "driver" in test_svd | P2 | UT issue with few failures | Bug/Perf issue awaiting reporter response | CuiYifeng | Need Investigation | Others | ut |
| 24 | 2759 | [Bug Skip]: New failed cases 2026-1-22 | P2 | UT issue with few failures | Bug/Perf issue awaiting reporter response | kaileiyx | Need Investigation | Others | ut |
| 25 | 2675 | [Bug Skip]: AttributeError: 'NoneType' object has no... | P2 | UT issue with few failures | Bug/Perf issue awaiting reporter response | kaileiyx | Need Investigation | Others | ut |
| 26 | 2669 | [upstream_ut] AssertionError: Tensor-likes are not close! in... | P2 | UT issue with few failures | Bug/Perf issue awaiting reporter response | daisyden | Need Investigation | Others | ut |
| 27 | 2639 | test_to() failed during rnn isinstance() check | P2 | UT issue with few failures | Bug/Perf issue awaiting reporter response | daisyden | Need Investigation | Others | ut |
| 28 | 2615 | [Bug Skip]: New failures RuntimeError: Unsupported dtype Half... | P2 | UT issue with few failures | Bug/Perf issue awaiting reporter response | kaileiyx | Need Investigation | Dtype / Precision Related | ut |
| 29 | 2595 | [Bug Skip]: Random crashed cases 2025-12-17 | P0 | Build crash - critical blocking issue | Bug/Perf issue awaiting reporter response | CuiYifeng | Need Investigation | Others | ut |
| 30 | 2537 | Title: [upstream_ut] Failed: Unexpected success | P2 | UT issue with few failures | Bug/Perf issue awaiting reporter response | daisyden | Need Investigation | Others | ut |
| 31 | 2536 | Title: [upstream_ut] AttributeError: module 'torch._C' has no... | P2 | UT issue with few failures | Bug/Perf issue awaiting reporter response | daisyden | Need Investigation | Others | ut |
| 32 | 2532 | Title: [upstream_ut] AssertionError: wrong number of... | P2 | UT issue with few failures | Bug/Perf issue awaiting reporter response | daisyden | Need Investigation | TorchAO | ut |
| 33 | 2531 | [upstream_ut] AssertionError: Torch not compiled with CUDA... | P2 | UT issue with few failures | Bug/Perf issue awaiting reporter response | daisyden | Need Investigation | Others | ut |
| 34 | 2530 | Title: [upstream_ut] AssertionError: RuntimeError not raised | P2 | UT issue with few failures | Bug/Perf issue awaiting reporter response | daisyden | Need Investigation | Others | ut |
| 35 | 2482 | test_dtypes issue introduced by pytorch test sample input updates | P2 | UT issue with few failures | Bug/Perf issue awaiting reporter response | daisyden | Need Investigation | Dtype / Precision Related | ut |
| 36 | 2446 | [Bug Skip]: AssertionError: "Simulate error" does not match... | P2 | UT issue with few failures | Bug/Perf issue awaiting reporter response | kaileiyx | Need Investigation | Others | ut |
| 37 | 2436 | [upstream_ut] AttributeError: 'NoneType' object has no... | P2 | UT issue with few failures | Bug/Perf issue awaiting reporter response | daisyden | Need Investigation | Others | ut |
| 38 | 2434 | [Bug Skip]: New failures 2025-11-28 | P2 | UT issue with few failures | Bug/Perf issue awaiting reporter response | mengfei25 | Need Investigation | Others | ut |
| 39 | 2425 | [upstream_ut] RuntimeError: Expected both self and other to... | P2 | UT issue with few failures | Bug/Perf issue awaiting reporter response | daisyden | Need Investigation | Others | ut |
| 40 | 2257 | Accuracy failures in test/xpu/test_unary_ufuncs_xpu.py | P2 | UT issue with few failures | Bug/Perf issue awaiting reporter response | zxd1997066 | Need Investigation | Dtype / Precision Related | ut |
| 41 | 2245 | oneDNN matmul received incorrect shape in... | P2 | UT issue with few failures | Bug/Perf issue awaiting reporter response | wincent8 | Need Investigation | Sparse Operations Related | ut |
| 42 | 2240 | RuntimeError: Trying to set a forward gradient that has a... | P2 | UT issue with few failures | Bug/Perf issue awaiting reporter response | zxd1997066 | Need Investigation | Others | ut |
| 43 | 2239 | Exception: could not create a primitive descriptor for the... | P2 | UT issue with few failures | Bug/Perf issue awaiting reporter response | zxd1997066 | Need Investigation | Others | ut |
| 44 | 2238 | Exception: Tensor-likes are not close! in... | P2 | UT issue with few failures | Bug/Perf issue awaiting reporter response | zxd1997066 | Need Investigation | Others | ut |
| 45 | 2230 | test_sparse_csr.py::TestSparseCompressedTritonKernelsXPU::test... | P2 | UT issue with few failures | Bug/Perf issue awaiting reporter response | wincent8 | Need Investigation | Sparse Operations Related | ut |
| 46 | 2186 | AssertionError: Mul tiheadAttention does not support... | P2 | UT issue with few failures | Bug/Perf issue awaiting reporter response | daisyden | Need Investigation | Others | ut |
| 47 | 2024 | AssertionError: Torch not compiled with CUDA enabled | P2 | UT issue with few failures | Bug/Perf issue awaiting reporter response | mengfei25 | Need Investigation | Inductor / Compilation Related | ut |

#### <span id='2.1-3-feature-requests'>2.1.3 Feature Requests - Developer</span> (27 issues)

| # | ID | Title | Priority | Priority Reason | Action Reason | Owner Transfer | Action TBD | Category | Test Module |
|---|------|------|----------|--------------------------------------------|----------|----------|------------|------------|-----------|
| 1 | 3189 | Task Tracker | P2 | UT issue with few failures | No specific action identified - needs investigation | guangyey | Need Investigation | Others | ut |
| 2 | 3150 | [Task] Align XPU kernel's implementation to stock PyTorch | P2 | UT issue with few failures | No specific action identified - needs investigation | guangyey | Need Investigation | Others | ut |
| 3 | 3060 | Implement torch._scaled_grouped_mm for xpu backend | P2 | UT issue with few failures | No specific action identified - needs investigation | Stonepia, liangan1 | Need Investigation | Others | ut |
| 4 | 3021 | [distributed] all_to_all_single Compatibility Issue on B60/XCCL | P2 | UT issue with few failures | No specific action identified - needs investigation | zhangxiaoli73 | Need Investigation | Distributed | ut |
| 5 | 3010 | [distributed][tensor] test_random_ops.py... | P2 | UT issue with few failures | Bug/Perf issue awaiting reporter response | zxd1997066 | Need Investigation | PT2E | ut |
| 6 | 2853 | [upstream_ut] torch.ops.aten._flash_attention_forward lack of... | P2 | UT issue with few failures | No specific action identified - needs investigation | LuFinch | Need Investigation | Flash Attention / Transformer Related | ut |
| 7 | 2779 | Accuracy failures in logspace op | P2 | UT issue with few failures | Bug/Perf issue awaiting reporter response | PawelSwider2000 | Need Investigation | Dtype / Precision Related | ut |
| 8 | 2722 | [Bug Skip]: NotImplementedError: Could not run 'aten::flip'... | P2 | UT issue with few failures | Bug/Perf issue awaiting reporter response | CuiYifeng | Need Investigation | TorchAO | ut |
| 9 | 2618 | [Bug Skip]: [regression] AssertionError: Scalars are not... | P0 | Regression - passed before but failed now | Bug/Perf issue awaiting reporter response | kaileiyx | Need Investigation | Others | ut |
| 10 | 2442 | [Bug Skip]: NotImplementedError: Could not run... | P2 | UT issue with few failures | Bug/Perf issue awaiting reporter response | CuiYifeng | Need Investigation | Flash Attention / Transformer Related | ut |
| 11 | 2412 | Some NestedTensor missing XPU support | P2 | UT issue with few failures | No specific action identified - needs investigation | yucai-intel | Need Investigation | Others | ut |
| 12 | 2400 | [ut_upstream] tf32_on_and_off() need xpu support | P2 | UT issue with few failures | No specific action identified - needs investigation | chunhuanMeng | Need Investigation | Others | ut |
| 13 | 2390 | SDPA in pytorch use different backend compared with ipex | P2 | UT issue with few failures | No specific action identified - needs investigation | LuFinch | Need Investigation | Flash Attention / Transformer Related | ut |
| 14 | 2376 | [Bug Skip]: NotImplementedError: "logaddexp_xpu" not... | P2 | UT issue with few failures | Bug/Perf issue awaiting reporter response | mengfei25 | Need Investigation | Others | ut |
| 15 | 2285 | Support efficient attention | P2 | UT issue with few failures | No specific action identified - needs investigation | chunhuanMeng | Need Investigation | Others | ut |
| 16 | 2200 | support flash attention op on XPU device | P2 | UT issue with few failures | No specific action identified - needs investigation | ElaineBao | Need Investigation | Flash Attention / Transformer Related | ut |
| 17 | 2098 | Upstream XPU functions in yaml | P2 | UT issue with few failures | No specific action identified - needs investigation | guangyey | Need Investigation | Others | ut |
| 18 | 2089 | need an implementation that won't initialize gpu context for... | P2 | UT issue with few failures | No specific action identified - needs investigation | guangyey | Need Investigation | Others | ut |
| 19 | 1936 | implement torch.linalg.cholesky xpu backend | P2 | UT issue with few failures | No specific action identified - needs investigation | mwiktor-intel | Need Investigation | Others | ut |
| 20 | 1912 | Implement the torch.ops.aten._weight_int4pack_mm for... | P2 | UT issue with few failures | No specific action identified - needs investigation | liangan1 | Need Investigation | TorchAO | ut |
| 21 | 1902 | implement torch.linalg.pinv xpu backend | P2 | UT issue with few failures | No specific action identified - needs investigation | mwiktor-intel | Need Investigation | Others | ut |
| 22 | 1901 | implement torch.linalg.svd xpu backend | P2 | UT issue with few failures | No specific action identified - needs investigation | CuiYifeng | Need Investigation | Others | ut |
| 23 | 1900 | implement torch.linalg.qr xpu backend | P2 | UT issue with few failures | No specific action identified - needs investigation | pbielak | Need Investigation | Others | ut |
| 24 | 1678 | missing op support for `model.share_memory()` | P0 | Impacts customer custom model/application | No specific action identified - needs investigation | None | Need Investigation | Others | ut |
| 25 | 1624 | [DONT CLOSE] Known UT Issue list | P2 | UT issue with few failures | No specific action identified - needs investigation | None | Need Investigation | Distributed | ut |
| 26 | 1574 | The operator 'aten::_grouped_mm' is not currently implemented... | P2 | UT issue with few failures | No specific action identified - needs investigation | Stonepia, LuFinch | Need Investigation | Others | ut |
| 27 | 208 | Abstract utility functions used in ATen operator implementation. | P2 | UT issue with few failures | No specific action identified - needs investigation | CuiYifeng | Need Investigation | Others | ut |

### <span id='action-required-reporter'>2.2 Reporter AR (Other Action TBD)</span>

*Action TBD values requiring reporter/community response*

#### <span id='2.2-1-needs-upstream-skip-pr'>2.2.1 Needs Upstream Skip PR - Reporter</span> (76 issues)

| # | ID | Title | Priority | Priority Reason | Action Reason | Owner Transfer | Action TBD | Category | Test Module |
|---|------|------|----------|--------------------------------------------|----------|----------|------------|------------|-----------|
| 1 | 3296 | accuracy gap of stft in float16 | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | None | Needs Upstream Skip PR | Dtype / Precision Related | ut |
| 2 | 3247 | NotImplementedError: "dot_xpu_mkl" not implemented for 'Long' | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | Silv3S | Needs Upstream Skip PR | Others | ut |
| 3 | 3238 | The supported dtypes of _refs.stft is not aligned to stft | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | daisyden | Needs Upstream Skip PR | Dtype / Precision Related | ut |
| 4 | 3229 | RuntimeError: No viable backend for... | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | tszulist-hbn | Needs Upstream Skip PR | Flash Attention / Transformer Related | ut |
| 5 | 3187 | PyTorch XPU gpu_cpp_wrapper fails with InductorError... | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | CuiYifeng | Needs Upstream Skip PR | Inductor / Compilation Related | ut |
| 6 | 3170 | Unskip test_bmm_windows_error_xpu_float64 | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | jenniew | Needs Upstream Skip PR | Others | ut |
| 7 | 3169 | NotImplementedError: Could not run 'aten::hspmm' with... | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | jkosnox | Needs Upstream Skip PR | Others | ut |
| 8 | 3167 | NotImplementedError: Could not run 'aten::triangular_solve.X'... | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | tszulist-hbn | Needs Upstream Skip PR | Others | ut |
| 9 | 3166 | test_consistency_SparseCSR failures | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | yucai-intel | Needs Upstream Skip PR | Sparse Operations Related | ut |
| 10 | 3165 | test_sparse_csr_xpu.py::TestSparseCompressedTritonKernelsXPU::... | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | None | Needs Upstream Skip PR | Others | ut |
| 11 | 3163 | [Bug Skip]: Object comparison failed: torch.int64 !=... | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | chunhuanMeng | Needs Upstream Skip PR | Dtype / Precision Related | ut |
| 12 | 3143 | NotImplementedError: The operator... | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | LuFinch | Needs Upstream Skip PR | Others | ut |
| 13 | 3142 | [upstream_ut] RuntimeError: The sycl_ext_oneapi_work_group_scr... | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | LuFinch | Needs Upstream Skip PR | Others | ut |
| 14 | 3141 | [upstream_ut] RuntimeError: FlashAttentionForwardXPU only... | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | LuFinch | Needs Upstream Skip PR | Flash Attention / Transformer Related | ut |
| 15 | 3140 | [upstream_ut] RuntimeError: FlashAttentionForwardXPU does not... | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | LuFinch | Needs Upstream Skip PR | Others | ut |
| 16 | 3137 | [upstream_ut] RuntimeError: expected scalar type Half but... | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | LuFinch | Needs Upstream Skip PR | Dtype / Precision Related | ut |
| 17 | 3136 | [upstream_ut] AssertionError: False is not true in... | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | LuFinch | Needs Upstream Skip PR | Flash Attention / Transformer Related | ut |
| 18 | 3133 | [upstream_ut] RuntimeError: scaled_dot_product_attention: If... | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | daisyden | Needs Upstream Skip PR | Flash Attention / Transformer Related | ut |
| 19 | 3132 | [upstream_ut] transfomers test reports RuntimeError: No... | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | LuFinch | Needs Upstream Skip PR | Others | ut |
| 20 | 3131 | [upstream_ut] NotImplementedError: The operator... | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | chunhuanMeng | Needs Upstream Skip PR | Others | ut |
| 21 | 3129 | [upstream_ut] AssertionError: UserWarning not triggered | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | daisyden | Needs Upstream Skip PR | Others | ut |
| 22 | 3128 | [upstream_ut] AssertionError: RuntimeError not raised by <lambda> | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | daisyden | Needs Upstream Skip PR | Others | ut |
| 23 | 3126 | [upstream_ut] Two NestedTensor issue with flash attention | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | daisyden | Needs Upstream Skip PR | Flash Attention / Transformer Related | ut |
| 24 | 3095 | cutlass support blocks some unit test cases | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | None | Needs Upstream Skip PR | Inductor / Compilation Related | ut |
| 25 | 3094 | XPUGraph tree support | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | None | Needs Upstream Skip PR | Inductor / Compilation Related | ut |
| 26 | 3077 | [Bug Skip] test_dlpack.py::TestTorchDlPackXPU::test_no_copy_xp... | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | AKloniecki | Needs Upstream Skip PR | Others | ut |
| 27 | 3041 | AssertionError: Expected len(flat_diff_results) > 0 in... | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | Silv3S | Needs Upstream Skip PR | Others | ut |
| 28 | 3007 | AssertionError: Scalars are not equal! with... | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | daisyden | Needs Upstream Skip PR | Flash Attention / Transformer Related | e2e |
| 29 | 3006 | AssertionError: '.to(tl.float16)' unexpectedly found in '# AOT ID | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | CuiYifeng | Needs Upstream Skip PR | PT2E | e2e |
| 30 | 3004 | TypeError: _xpu_recordMemoryHistory(): incompatible function... | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | guangyey | Needs Upstream Skip PR | Others | ut |
| 31 | 2999 | KeyError: 'eager_numerics.use_pytorch_libdevice' | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | daisyden | Needs Upstream Skip PR | Others | ut |
| 32 | 2997 | AssertionError of test_linear_and_cel_max_autotune | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | etaf | Needs Upstream Skip PR | Inductor / Compilation Related | ut |
| 33 | 2958 | AssertionError of test_dtensor_basic_compile | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | daisyden | Needs Upstream Skip PR | Inductor / Compilation Related | ut |
| 34 | 2919 | [XPU][upstream_ut][COW] Fix materialization in remaining... | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | gplutop7 | Needs Upstream Skip PR | Others | ut |
| 35 | 2918 | [XPU][upstream_ut][COW] Skip non-supported ops (jiterator +... | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | gplutop7 | Needs Upstream Skip PR | Others | ut |
| 36 | 2891 | RuntimeError: Expected to find "(262144, 0, 512, 1" but did... | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | chunhuanMeng | Needs Upstream Skip PR | Others | e2e |
| 37 | 2888 | torch._inductor.exc.InductorError: AssertionError:... | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | Stonepia | Needs Upstream Skip PR | Inductor / Compilation Related | ut |
| 38 | 2810 | AssertionError: Object comparison failed:... | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | daisyden | Needs Upstream Skip PR | Inductor / Compilation Related | ut |
| 39 | 2806 | CompiledAOTI need XPU support | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | daisyden | Needs Upstream Skip PR | Inductor / Compilation Related | ut |
| 40 | 2802 | Three aten._scaled_dot_product_flash_attention issues | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | LuFinch | Needs Upstream Skip PR | Flash Attention / Transformer Related | ut |
| 41 | 2800 | AttributeError: 'torch._C._XpuDeviceProperties' object has no... | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | guangyey | Needs Upstream Skip PR | Others | ut |
| 42 | 2798 | Test case test/test_dlpack.py::TestTorchDlPackCPU::test_numpy_... | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | None | Needs Upstream Skip PR | Others | ut |
| 43 | 2715 | [upstream_ut] torch._dynamo.exc.Unsupported: Attempted to... | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | CuiYifeng | Needs Upstream Skip PR | PT2E | ut |
| 44 | 2714 | [upstream_ut] AssertionError: Object comparison failed:... | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | Silv3S | Needs Upstream Skip PR | Others | ut |
| 45 | 2712 | [upstream_ut] RuntimeError: Cannot swap t2 because it has... | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | tszulist-hbn | Needs Upstream Skip PR | Others | ut |
| 46 | 2698 | Title: [upstream_ut] RuntimeError: FlashAttentionForwardXPU... | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | chunhuanMeng, LuFinch | Needs Upstream Skip PR | Others | ut |
| 47 | 2697 | Title: [upstream_ut] RuntimeError: Expected to find ", 0, "... | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | chunhuanMeng | Needs Upstream Skip PR | Others | e2e |
| 48 | 2694 | Title: [upstream_ut] AssertionError: Tensor-likes are not... | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | daisyden | Needs Upstream Skip PR | Inductor / Compilation Related | ut |
| 49 | 2693 | Title: [upstream_ut] AssertionError: Scalars are not equal! | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | hoshibara | Needs Upstream Skip PR | Inductor / Compilation Related | ut |
| 50 | 2670 | [upstream_ut] RuntimeError: could not create a primitive... | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | tszulist-hbn | Needs Upstream Skip PR | Others | ut |
| 51 | 2663 | test_sparse_semi_structured.py gaps | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | None | Needs Upstream Skip PR | Sparse Operations Related | ut |
| 52 | 2620 | [upstream_ut] AssertionError: dtype is needed to compute eps1... | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | daisyden | Needs Upstream Skip PR | Inductor / Compilation Related | ut |
| 53 | 2613 | [upstream_ut] AssertionError: Tensor-likes are not equal! in... | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | daisyden | Needs Upstream Skip PR | Inductor / Compilation Related | ut |
| 54 | 2611 | [upstream_ut] AssertionError: Tensor-likes are not equal! in... | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | daisyden | Needs Upstream Skip PR | Inductor / Compilation Related | ut |
| 55 | 2609 | [upstream_ut] torch._inductor.exc.InductorError:... | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | daisyden | Needs Upstream Skip PR | Inductor / Compilation Related | ut |
| 56 | 2578 | [TorchAO][UT] test/quantization/test_quant_api.py... | P0 | Build crash - critical blocking issue | Issue is upstream - needs skip PR upstream | Stonepia | Needs Upstream Skip PR | TorchAO | build |
| 57 | 2554 | [upstream_ut] AssertionError: AssertionError not raised | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | daisyden | Needs Upstream Skip PR | Inductor / Compilation Related | ut |
| 58 | 2359 | [upstream_ut] GradcheckError: Backward is not reentrant | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | BBBela | Needs Upstream Skip PR | Others | ut |
| 59 | 2358 | test/test_view_ops.py::TestOldViewOpsXPU::test_ravel_xpu meet... | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | Silv3S | Needs Upstream Skip PR | TorchAO | ut |
| 60 | 2331 | [upstream_ut] AssertionError: Scalars are not equal! with... | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | hoshibara | Needs Upstream Skip PR | Others | ut |
| 61 | 2329 | [upstream_ut] feature missing: get_device_tflops and... | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | etaf | Needs Upstream Skip PR | Inductor / Compilation Related | ut |
| 62 | 2301 | [upstream_ut] dtypes not align with OpInfo | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | daisyden | Needs Upstream Skip PR | Others | ut |
| 63 | 2295 | [upstream_ut][xpu][test]nn/test_embedding.py::TestEmbeddingNND... | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | yucai-intel | Needs Upstream Skip PR | Others | ut |
| 64 | 2287 | [upstream_ut] test_python_ref issues | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | yucai-intel | Needs Upstream Skip PR | Others | ut |
| 65 | 2283 | [upstream_ut] sparse._sampled_addmm is not supported | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | jenniew | Needs Upstream Skip PR | Sparse Operations Related | ut |
| 66 | 2263 | [xpu][bug] XPU Trace event ends too late! | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | PawelSwider2000 | Needs Upstream Skip PR | Others | ut |
| 67 | 2255 | [upstream_ut] RuntimeError: Long is not supported in oneDNN | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | daisyden | Needs Upstream Skip PR | Others | ut |
| 68 | 2253 | the supported dtypes are not align with cuda | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | daisyden | Needs Upstream Skip PR | Others | ut |
| 69 | 2251 | [upstream_ut] test_fake_autocase got Exception: Dtypes... | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | astachowiczhabana | Needs Upstream Skip PR | Dtype / Precision Related | ut |
| 70 | 2248 | [upstream_ut] test_cow failures | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | gplutop7 | Needs Upstream Skip PR | Others | ut |
| 71 | 2234 | [upstream_ut] AssertionError: RuntimeError not raised :... | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | Silv3S | Needs Upstream Skip PR | Others | ut |
| 72 | 2214 | test/test_sparse.py::TestSparseAnyXPU::test_gradcheck_mm... | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | jenniew | Needs Upstream Skip PR | Sparse Operations Related | ut |
| 73 | 1963 | [upstream_ut] MetadataMismatchError in TestFakeTensor of... | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | pbielak | Needs Upstream Skip PR | Others | ut |
| 74 | 1951 | Functionality issues in TestCommon.test_out. | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | AKloniecki | Needs Upstream Skip PR | Others | ut |
| 75 | 1893 | [upstream_ut] oneDNN accuracy issues in test_ops_xpu.py | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | chunhuanMeng | Needs Upstream Skip PR | Others | ut |
| 76 | 1505 | [ARC-WSL-Ubuntu24.04] 15 Timm models got fail_accuracy | P0 | Impacts customer custom model/application | Issue is upstream - needs skip PR upstream | None | Needs Upstream Skip PR | Inductor / Compilation Related | e2e |

#### <span id='2.2-2-add-to-skiplist'>2.2.2 add to skiplist - Reporter</span> (5 issues)

| # | ID | Title | Priority | Priority Reason | Action Reason | Owner Transfer | Action TBD | Category | Test Module |
|---|------|------|----------|--------------------------------------------|----------|----------|------------|------------|-----------|
| 1 | 3127 | [upstream_ut] AssertionError: AssertionError not raised | P2 | UT issue with few failures | Issue marked as not_target/wontfix - should be skipped for XPU enablement | daisyden | add to skiplist | Others | ut |
| 2 | 2508 | TypedStorage / TypedTensors deprecation | P2 | UT issue with few failures | Issue marked as not_target/wontfix - should be skipped for XPU enablement | libohao1201 | add to skiplist | Others | ut |
| 3 | 2472 | [upstream_ut] NotImplementedError: The operator... | P2 | UT issue with few failures | Issue marked as not_target/wontfix - should be skipped for XPU enablement | daisyden | add to skiplist | Others | ut |
| 4 | 2309 | unsupported ops with PYTORCH_ENABLE_XPU_FALLBACK unset | P2 | UT issue with few failures | Issue marked as not_target/wontfix - should be skipped for XPU enablement | daisyden | add to skiplist | Others | ut |
| 5 | 2164 | skip test_no_cuda_monkeypatch as it is cuda specific | P2 | UT issue with few failures | Issue marked as not_target/wontfix - should be skipped for XPU enablement | daisyden | add to skiplist | Others | ut |

#### <span id='2.2-3-close-fixed-issue'>2.2.3 Close fixed issue - Reporter</span> (4 issues)

| # | ID | Title | Priority | Priority Reason | Action Reason | Owner Transfer | Action TBD | Category | Test Module |
|---|------|------|----------|--------------------------------------------|----------|----------|------------|------------|-----------|
| 1 | 3174 | [Bug Skip]: Accuracy failure of test_Conv2d_groups_nobias | P2 | UT issue with few failures | All test cases passed on both XPU and stock - issue is resolved | CuiYifeng | Close fixed issue | Others | ut |
| 2 | 3160 | compiler not found (Windows) | P2 | UT issue with few failures | All test cases passed on both XPU and stock - issue is resolved | kdrozd-dev | Close fixed issue | Others | ut |
| 3 | 2518 | [upstream_ut] TypeError: Creating a Tensor subclass from a... | P2 | UT issue with few failures | All test cases passed on both XPU and stock - issue is resolved | libohao1201 | Close fixed issue | Others | ut |
| 4 | 2496 | [upstream_ut] Segmentation fault when running... | P0 | Build crash - critical blocking issue | All test cases passed on both XPU and stock - issue is resolved | libohao1201 | Close fixed issue | Others | ut |

#### <span id='2.2-4-verify-the-issue'>2.2.4 Verify the issue - Reporter</span> (3 issues)

| # | ID | Title | Priority | Priority Reason | Action Reason | Owner Transfer | Action TBD | Category | Test Module |
|---|------|------|----------|--------------------------------------------|----------|----------|------------|------------|-----------|
| 1 | 3286 | New failing test case after enabling tests from... | P2 | UT issue with few failures | PR closed but no failed tests - verify if issue still reproduces | BBBela | Verify the issue | Others | ut |
| 2 | 3284 | Optimize torch.nn.functional.one_hot | P2 | UT issue with few failures | PR closed but no failed tests - verify if issue still reproduces | xinyu-intel | Verify the issue | Others | ut |
| 3 | 3258 | huggingface accuracy inference Error in op:... | P2 | UT issue with few failures | PR closed but no failed tests - verify if issue still reproduces | bjarzemb | Verify the issue | Others | ut |

#### <span id='2.2-5-awaiting-response-from-reporter'>2.2.5 Awaiting response from reporter - Reporter</span> (143 issues)

| # | ID | Title | Priority | Priority Reason | Action Reason | Owner Transfer | Action TBD | Category | Test Module |
|---|------|------|----------|--------------------------------------------|----------|----------|------------|------------|-----------|
| 1 | 3280 | [Bug Skip]: New UT failure in 0406 nightly windows. | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | RUIJIEZHONG66166 | Awaiting response from reporter | Others | ut |
| 2 | 3270 | [distributed][tensor] RuntimeError: Invalid scaling... | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | zxd1997066 | Awaiting response from reporter | Others | ut |
| 3 | 3259 | New failed test cases 2026-04-02 | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | Silv3S | Awaiting response from reporter | Others | ut |
| 4 | 3233 | [distributed] RuntimeError: No backend for the parent process... | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | zxd1997066 | Awaiting response from reporter | Distributed | ut |
| 5 | 3232 | [distributed][tensor] AssertionError: AssertionError not... | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | zxd1997066 | Awaiting response from reporter | Distributed | ut |
| 6 | 3231 | Dynamo failed to run FX node with fake tensors: call_function... | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | daisyden | Awaiting response from reporter | PT2E | ut |
| 7 | 3227 | torch xpu event has ~0.1ms latency, which is too large | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | jianyizh | Awaiting response from reporter | Others | ut |
| 8 | 3224 | [Win][Build] Building SYCL (Device) object... | P0 | Build crash - critical blocking issue | Bug/Perf issue pending reporter response | anmyachev | Awaiting response from reporter | Others | build |
| 9 | 3209 | [Win][Build] There is Cyclic dependencies error when build... | P0 | Build crash - critical blocking issue | Bug/Perf issue pending reporter response | NeoZhangJianyu | Awaiting response from reporter | Others | build |
| 10 | 3195 | test_sdpa_unbacked_no_dde_xpu crashed | P0 | Build crash - critical blocking issue | Bug/Perf issue pending reporter response | daisyden | Awaiting response from reporter | Flash Attention / Transformer Related | ut |
| 11 | 3191 | torch._inductor.exc.InductorError: AssertionError: both a... | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | mengfei25 | Awaiting response from reporter | Inductor / Compilation Related | e2e |
| 12 | 3161 | Exception: Tensor-likes are not close! -... | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | kdrozd-dev | Awaiting response from reporter | Dtype / Precision Related | ut |
| 13 | 3158 | AttributeError: module 'triton.compiler' has no attribute... | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | kdrozd-dev | Awaiting response from reporter | Inductor / Compilation Related | ut |
| 14 | 3156 | AssertionError: 'Assertion cur_target >= 0 && cur_target <... | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | kdrozd-dev | Awaiting response from reporter | Others | ut |
| 15 | 3139 | [distributed][_composable] AssertionError: Expects xpu:0 but... | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | zxd1997066 | Awaiting response from reporter | Distributed | ut |
| 16 | 3124 | [TorchAO][Bug] ImportError: Requires mslk >= 1.0.0 when... | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | FRAMEEE17 | Awaiting response from reporter | TorchAO | ut |
| 17 | 3114 | [Bug Skip]: Failure skip on 2026-3-21 | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | guangyey | Awaiting response from reporter | Others | ut |
| 18 | 3106 | Worker crashes when running TestDecompXPU,test_quick_core_back... | P0 | Build crash - critical blocking issue | Bug/Perf issue pending reporter response | BBBela | Awaiting response from reporter | Others | ut |
| 19 | 3102 | [distributed] RuntimeError: Invalid device string: 'xpu:foo'... | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | zxd1997066 | Awaiting response from reporter | Others | ut |
| 20 | 3101 | [distributed] 'torch._C._distributed_c10d.ProcessGroupXCCL'... | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | zxd1997066 | Awaiting response from reporter | Distributed | ut |
| 21 | 3088 | [TorchAO][BMG] INT4 RTN Flex-attention got 5% performance drop | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | LifengWang | Awaiting response from reporter | TorchAO | ut |
| 22 | 3083 | [Bug Skip]: Random failures 2026WW12 | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | CuiYifeng | Awaiting response from reporter | Others | ut |
| 23 | 3076 | [TorchAO][BMG] Llama-3.2-1B-Instruct Dynamic INT8 got 10%... | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | LifengWang | Awaiting response from reporter | TorchAO | ut |
| 24 | 3074 | [Bug Skip] test_dlpack_exchange_api expect... | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | shangerxin | Awaiting response from reporter | Others | ut |
| 25 | 3032 | [TorchAO][UT] failures in test/prototype/safetensors/test_safe... | P0 | Build crash - critical blocking issue | Bug/Perf issue pending reporter response | zxd1997066 | Awaiting response from reporter | TorchAO | build |
| 26 | 3030 | [Bug Skip] test/test_modules.py::TestModuleXPU::test_cpu_gpu_p... | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | shangerxin | Awaiting response from reporter | Others | ut |
| 27 | 3014 | [upstream_ut] AssertionError: False is not true | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | Silv3S | Awaiting response from reporter | Others | ut |
| 28 | 3013 | [upstream_ut] RuntimeError: Kernel is incompatible with all... | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | Silv3S | Awaiting response from reporter | Others | ut |
| 29 | 3011 | [upstream_ut] torch.OutOfMemoryError: XPU out of memory | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | Silv3S | Awaiting response from reporter | Others | ut |
| 30 | 2993 | [Bug Skip]: Unexpected success of... | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | CuiYifeng | Awaiting response from reporter | Others | ut |
| 31 | 2981 | [release/2.11] T5 models performance dropped ~20% | P0 | Impacts customer custom model/application | Bug/Perf issue pending reporter response | mengfei25 | Awaiting response from reporter | Others | e2e |
| 32 | 2979 | eca_halonext26ts got RuntimeError:... | P0 | Build crash - critical blocking issue | Bug/Perf issue pending reporter response | mengfei25 | Awaiting response from reporter | Others | e2e |
| 33 | 2972 | [distributed] AssertionError: ValueError not raised in... | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | zxd1997066 | Awaiting response from reporter | Distributed | ut |
| 34 | 2969 | [distributed] AssertionError: Scalars are not equal! in... | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | zxd1997066 | Awaiting response from reporter | Distributed | ut |
| 35 | 2968 | [distributed] timeout issue in test/distributed/test_c10d_xccl.py | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | zxd1997066 | Awaiting response from reporter | Distributed | ut |
| 36 | 2966 | [Bug Skip]: [Regression]2026-3-2 ut failures | P0 | Regression - passed before but failed now | Bug/Perf issue pending reporter response | kaileiyx | Awaiting response from reporter | Others | ut |
| 37 | 2960 | [release/2.11] timm_models_xcit_large_24_p8_224_float16_traini... | P0 | Impacts customer custom model/application | Bug/Perf issue pending reporter response | shangerxin | Awaiting response from reporter | Dtype / Precision Related | ut |
| 38 | 2953 | [release/2.11][wsl] huggingface TrOCRForCausalLM and... | P2 | E2E benchmark model issue | Bug/Perf issue pending reporter response | bjarzemb | Awaiting response from reporter | Others | e2e |
| 39 | 2952 | [release/2.11][wsl] timm_models_accuracy_training_bfloat16... | P0 | Impacts customer custom model/application | Bug/Perf issue pending reporter response | bjarzemb | Awaiting response from reporter | Dtype / Precision Related | ut |
| 40 | 2946 | [Bug Skip]: Random failures 2026WW09 | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | CuiYifeng | Awaiting response from reporter | Others | ut |
| 41 | 2942 | [Windows] Unit tests got Fatal python error | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | mengfei25 | Awaiting response from reporter | Others | ut |
| 42 | 2939 | [release/2.11] gmlp_s16_224 inference amp performance dropped... | P2 | E2E performance issue | Bug/Perf issue pending reporter response | mengfei25 | Awaiting response from reporter | Others | e2e |
| 43 | 2938 | [release/2.11] basic_gnn_gin and basic_gnn_sage inference... | P2 | E2E performance issue | Bug/Perf issue pending reporter response | mengfei25 | Awaiting response from reporter | Others | e2e |
| 44 | 2935 | [release/2.11][inductor] huggingface amp_fp16 and float16... | P0 | Regression - passed before but failed now | Bug/Perf issue pending reporter response | agnottaski | Awaiting response from reporter | Inductor / Compilation Related | e2e |
| 45 | 2922 | [release/2.11] UT inductor AssertionError: pass_fds not... | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | bjarzemb | Awaiting response from reporter | Inductor / Compilation Related | ut |
| 46 | 2914 | Test case test/test_autograd.py::TestAutogradMultipleDispatchC... | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | shangerxin | Awaiting response from reporter | Others | ut |
| 47 | 2912 | [release/2.11] UT extended 220 new failures | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | bjarzemb | Awaiting response from reporter | Others | ut |
| 48 | 2907 | [release/2.11] Models performance regression for 5 testcases | P0 | Regression - passed before but failed now | Bug/Perf issue pending reporter response | bjarzemb | Awaiting response from reporter | Others | ut |
| 49 | 2873 | [Bug Skip]: test_repos.py contains several failed ops | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | shangerxin | Awaiting response from reporter | Others | ut |
| 50 | 2862 | accuracy issue with test_float8_scale_fast_accum_xpu | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | daisyden | Awaiting response from reporter | Dtype / Precision Related | ut |
| 51 | 2858 | [Bug Skip]: test_xpu new failures | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | RUIJIEZHONG66166 | Awaiting response from reporter | Others | ut |
| 52 | 2852 | [Bug Skip]: New UT failures in 0206 nightly on Windows | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | chuanqi129 | Awaiting response from reporter | Others | ut |
| 53 | 2845 | [Bug Skip]:[UT] [Windows] failed cases 2026-2-4 | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | kaileiyx | Awaiting response from reporter | Others | ut |
| 54 | 2823 | [TorchAO][BMG] Llama-3.2-1B-Instruct Dynamic INT8 got 20%... | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | LifengWang | Awaiting response from reporter | TorchAO | ut |
| 55 | 2811 | [Bug Skip]: [Regression] failed cases 2026-2-2 | P0 | Regression - passed before but failed now | Bug/Perf issue pending reporter response | kaileiyx | Awaiting response from reporter | Others | ut |
| 56 | 2801 | to_dense() for Sparse CSR backend cannot broadcast batch dim... | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | jenniew | Awaiting response from reporter | Sparse Operations Related | ut |
| 57 | 2795 | Histc raises error with integer input when deterministic... | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | YangKai0616 | Awaiting response from reporter | Others | ut |
| 58 | 2777 | [Bug Skip]: Random failures 2026WW05 | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | CuiYifeng | Awaiting response from reporter | Others | ut |
| 59 | 2769 | [oneDNN] New failed test cases with 3.11 compared with 3.10 | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | mengfei25 | Awaiting response from reporter | Others | ut |
| 60 | 2767 | [UT] test_control_flow_xpu.py got AssertionError | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | libohao1201 | Awaiting response from reporter | Others | ut |
| 61 | 2766 | MaxPool2d - investigate memory layout performance | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | pbielak | Awaiting response from reporter | Others | ut |
| 62 | 2751 | [Bug Skip]: Random failures 2026WW04 | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | CuiYifeng | Awaiting response from reporter | Others | ut |
| 63 | 2744 | [Bug Skip]: extended test failures when test_compare_cpu atol... | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | daisyden | Awaiting response from reporter | Others | ut |
| 64 | 2742 | [Linux][PT2E] hf_Roberta_base model performance ASYMM and... | P0 | Impacts customer custom model/application | Bug/Perf issue pending reporter response | kaileiyx | Awaiting response from reporter | PT2E | e2e |
| 65 | 2738 | [distributed] test_c10d_spawn_nccl.py ValueError: input... | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | zxd1997066 | Awaiting response from reporter | Distributed | ut |
| 66 | 2737 | [distributed] AttributeError: module 'torch._C' has no... | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | zxd1997066 | Awaiting response from reporter | Distributed | ut |
| 67 | 2729 | [Bug Skip]: Random failures 2026WW03 | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | CuiYifeng | Awaiting response from reporter | Others | ut |
| 68 | 2720 | [upstream_ut] RuntimeError: false INTERNAL ASSERT FAILED at... | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | wincent8 | Awaiting response from reporter | Others | ut |
| 69 | 2707 | [TorchAO][BMG] INT4 GPTQ failed due to TorchAO API change. | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | LifengWang | Awaiting response from reporter | TorchAO | ut |
| 70 | 2702 | [distributed] RuntimeError: Work ran time out after 0... | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | madhumitha0102 | Awaiting response from reporter | Distributed | ut |
| 71 | 2701 | [distributed] Barrier Timeout Error with... | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | madhumitha0102 | Awaiting response from reporter | Distributed | ut |
| 72 | 2689 | [LNL][Windows] AssertionError: 'Assertion `cur_target >= 0 &&... | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | kaileiyx | Awaiting response from reporter | Others | ut |
| 73 | 2686 | [distributed] Accuracy issues with test_distributed_spawn.py | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | madhumitha0102 | Awaiting response from reporter | Distributed | ut |
| 74 | 2680 | XPU Autocast does not support fp32 dtypes | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | kaixuanliu | Awaiting response from reporter | Dtype / Precision Related | ut |
| 75 | 2676 | Random failure in CI test | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | daisyden | Awaiting response from reporter | Others | ut |
| 76 | 2662 | [release/2.10][Windows][BMG] New failed test cases and 2.9... | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | mengfei25 | Awaiting response from reporter | Others | ut |
| 77 | 2660 | [release/2.10][Windows][BMG] New failed test cases | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | mengfei25 | Awaiting response from reporter | Others | ut |
| 78 | 2659 | [distributed] test_dist2.py RuntimeError: Backend xccl does... | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | zxd1997066 | Awaiting response from reporter | Distributed | ut |
| 79 | 2656 | [release/2.10] models got fail_accuracy on BMG WSL2 | P0 | Impacts customer custom model/application | Bug/Perf issue pending reporter response | libohao1201 | Awaiting response from reporter | Dtype / Precision Related | ut |
| 80 | 2655 | [BMG][OOB] hf_Reformer performance drop | P0 | Regression - passed before but failed now | Bug/Perf issue pending reporter response | jianyizh | Awaiting response from reporter | Dtype / Precision Related | e2e |
| 81 | 2654 | [BMG][OOB] t5 inference performance drop 2 | P0 | Regression - passed before but failed now | Bug/Perf issue pending reporter response | jianyizh | Awaiting response from reporter | Dtype / Precision Related | e2e |
| 82 | 2630 | Title: [upstream_ut] AssertionError: Scalars are not equal! | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | daisyden | Awaiting response from reporter | Others | ut |
| 83 | 2619 | [release/2.10] Some models inductor performance dropped ~ 10%... | P0 | Impacts customer custom model/application | Bug/Perf issue pending reporter response | mengfei25 | Awaiting response from reporter | Inductor / Compilation Related | e2e |
| 84 | 2598 | [TorchAO][BMG]The first token latency of... | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | LifengWang | Awaiting response from reporter | TorchAO | ut |
| 85 | 2597 | [TorchAO][BMG] INT4 GPTQ shows worse performance compared... | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | LifengWang | Awaiting response from reporter | TorchAO | ut |
| 86 | 2580 | [TorchAO][UT] test/test_low_bit_optim.py AssertionError:... | P0 | Build crash - critical blocking issue | Bug/Perf issue pending reporter response | zxd1997066 | Awaiting response from reporter | TorchAO | build |
| 87 | 2572 | [TorchAO][UT] test/dtypes/test_affine_quantized.py... | P0 | Build crash - critical blocking issue | Bug/Perf issue pending reporter response | zxd1997066 | Awaiting response from reporter | TorchAO | build |
| 88 | 2570 | crash in sdpa. | P0 | Build crash - critical blocking issue | Bug/Perf issue pending reporter response | sywangyi | Awaiting response from reporter | Flash Attention / Transformer Related | ut |
| 89 | 2562 | Warning as Error | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | EikanWang | Awaiting response from reporter | Others | ut |
| 90 | 2560 | [UT] "RuntimeError: iter.device(arg).is_xpu()" in... | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | libohao1201 | Awaiting response from reporter | Others | ut |
| 91 | 2541 | Title: [upstream_ut] RuntimeError: could not construct a... | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | daisyden | Awaiting response from reporter | Others | ut |
| 92 | 2539 | Title: [upstream_ut] RuntimeError: Tried to instantiate dummy... | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | daisyden | Awaiting response from reporter | Others | ut |
| 93 | 2535 | Title: [upstream_ut] AttributeError: module 'torch._C' has no... | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | daisyden | Awaiting response from reporter | Others | ut |
| 94 | 2533 | Title: [upstream_ut] AttributeError: 'TestQuantizedOpsXPU'... | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | daisyden | Awaiting response from reporter | TorchAO | ut |
| 95 | 2529 | [upstream_ut] AssertionError: False is not true | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | daisyden | Awaiting response from reporter | Others | ut |
| 96 | 2519 | [upstream_ut] TypeError: map2_ is only implemented on CPU tensors | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | libohao1201 | Awaiting response from reporter | Others | ut |
| 97 | 2513 | [upstream_ut] RuntimeError: _share_fd_: only available on CPU | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | libohao1201 | Awaiting response from reporter | Others | ut |
| 98 | 2512 | [upstream_ut] RuntimeError: _histc_xpu does not have a... | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | libohao1201 | Awaiting response from reporter | Others | ut |
| 99 | 2510 | [upstream_ut] RuntimeError: Expected output.numel() <=... | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | libohao1201 | Awaiting response from reporter | Others | ut |
| 100 | 2491 | [upstream_ut] AssertionError: False is not true | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | libohao1201 | Awaiting response from reporter | Others | ut |
| 101 | 2479 | [Bug] torch.rand output different result on bmg and pvc | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | zufangzhu | Awaiting response from reporter | Others | ut |
| 102 | 2463 | [Bug Skip]: OSError: SYCL runtime is not dected. | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | RUIJIEZHONG66166 | Awaiting response from reporter | Others | ut |
| 103 | 2444 | [upstream_ut] RuntimeError: UR backend failed. UR backend... | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | wincent8 | Awaiting response from reporter | Others | ut |
| 104 | 2439 | [oneDNN] TestDecompXPU.test_quick_addmv_xpu_float64 got fail... | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | mengfei25 | Awaiting response from reporter | Dtype / Precision Related | ut |
| 105 | 2404 | [distributed][checkpoint] AssertionError: Booleans mismatch:... | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | zxd1997066 | Awaiting response from reporter | Distributed | ut |
| 106 | 2392 | [Bug Skip]: torch.OutOfMemoryError: XPU out of memory | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | RUIJIEZHONG66166 | Awaiting response from reporter | Others | ut |
| 107 | 2389 | [Bug Skip]: RuntimeError: Data corruption detected | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | kaileiyx | Awaiting response from reporter | Others | ut |
| 108 | 2340 | [distributed][_tools] AssertionError: Roofline estimation... | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | zxd1997066 | Awaiting response from reporter | Distributed | ut |
| 109 | 2270 | Backend Compatibility Error in test/xpu/test_decomp.py | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | libohao1201 | Awaiting response from reporter | Others | ut |
| 110 | 2244 | test/test_sparse_csr.py::TestSparseCSRXPU::test_block_addmm... | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | wincent8 | Awaiting response from reporter | Sparse Operations Related | ut |
| 111 | 2219 | float8_e4m3fn precision overflow | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | jiqing-feng | Awaiting response from reporter | Dtype / Precision Related | ut |
| 112 | 2217 | AO Performance issue track | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | liangan1 | Awaiting response from reporter | Others | ut |
| 113 | 2201 | [TorchAO][BMG] When using paged attention backend, all cases... | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | MingxuZh | Awaiting response from reporter | TorchAO | ut |
| 114 | 2182 | test_transform_bias_rescale_qkv_nested_xpu_float32 failed... | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | wincent8 | Awaiting response from reporter | Dtype / Precision Related | ut |
| 115 | 2169 | Frame size comparison failed in test_size_comparison_no_recompile | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | daisyden | Awaiting response from reporter | Inductor / Compilation Related | ut |
| 116 | 2165 | [distributed] test_device_mesh.py::TestDeviceMeshGetItem::test... | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | zxd1997066 | Awaiting response from reporter | Distributed | ut |
| 117 | 2132 | [2.9][BMG-Windows][Torch-xpu-ops UT] 1 case failed with... | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | libohao1201 | Awaiting response from reporter | Inductor / Compilation Related | ut |
| 118 | 2128 | [2.9][BMG-Windows][Torchbench] speeach_transforer... | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | libohao1201 | Awaiting response from reporter | Dtype / Precision Related | ut |
| 119 | 2022 | [Windows] [CI] [UT] AssertionError: Tensor-likes are not close! | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | RUIJIEZHONG66166 | Awaiting response from reporter | Others | ut |
| 120 | 2004 | [distributed][shared_tensor] test\distributed\_shard\shared_te... | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | libohao1201 | Awaiting response from reporter | Distributed | ut |
| 121 | 1973 | AssertionError: Scalars or Tensor-likes are not equal or close! | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | mengfei25 | Awaiting response from reporter | Others | ut |
| 122 | 1970 | torch._dynamo.exc.BackendCompilerFailed: backend='inductor'... | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | shangerxin | Awaiting response from reporter | Inductor / Compilation Related | ut |
| 123 | 1969 | torch._dynamo.exc.InternalTorchDynamoError: TypeError: cannot... | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | shangerxin | Awaiting response from reporter | PT2E | ut |
| 124 | 1894 | [Linux][PT2E] performance test got failed, int8 ASYMM and... | P1 | E2E accuracy/functionality issue | Bug/Perf issue pending reporter response | kaileiyx | Awaiting response from reporter | TorchAO | e2e |
| 125 | 1877 | Torchbench model squeezenet1_1 and functorch_dp_cifar10 got... | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | libohao1201 | Awaiting response from reporter | Dtype / Precision Related | ut |
| 126 | 1818 | [BMG-Windows][PT2.8]Torch-xpu-ops UT got accuracy issue | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | libohao1201 | Awaiting response from reporter | Dtype / Precision Related | ut |
| 127 | 1784 | [Performance] Torch XPU Profiler is not reliable | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | liangan1 | Awaiting response from reporter | Others | ut |
| 128 | 1749 | transformers UT failure in XPU because SDPA check error... | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | sywangyi | Awaiting response from reporter | Flash Attention / Transformer Related | ut |
| 129 | 1727 | [distributed] AttributeError: module 'torch.xpu' has no... | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | PenghuiCheng | Awaiting response from reporter | Distributed | ut |
| 130 | 1661 | [distributed] Accuracy gap in _composable/fsdp on Xelink | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | PenghuiCheng | Awaiting response from reporter | Distributed | ut |
| 131 | 1649 | [cpp extension] Provide a clear error message when using... | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | ZhaoqiongZ | Awaiting response from reporter | Others | ut |
| 132 | 1571 | [distributed] ValueError: Cannot use ReduceOp.PREMUL_SUM with... | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | daisyden | Awaiting response from reporter | Distributed | ut |
| 133 | 1556 | [distributed] NotImplementedError: Operator... | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | PenghuiCheng | Awaiting response from reporter | Distributed | ut |
| 134 | 1555 | [distributed] RuntimeError: aten.add.Tensor: got mixed... | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | PenghuiCheng | Awaiting response from reporter | Distributed | ut |
| 135 | 1551 | [distributed] NotImplementedError: The operator... | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | PenghuiCheng | Awaiting response from reporter | Distributed | ut |
| 136 | 1549 | [distributed] AssertionError: 'fused_all_gather_scaled_matmul'... | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | PenghuiCheng | Awaiting response from reporter | Distributed | ut |
| 137 | 1548 | [distributed] AssertionError: 'fused_all_gather_matmul' not... | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | PenghuiCheng | Awaiting response from reporter | Distributed | ut |
| 138 | 1547 | [distributed] NotImplementedError: The operator... | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | PenghuiCheng | Awaiting response from reporter | Distributed | ut |
| 139 | 1324 | [Win] UR Error when OOM and break the tensor context | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | Stonepia | Awaiting response from reporter | Others | ut |
| 140 | 1171 | LNL Windows got unexpected error message | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | daisyden | Awaiting response from reporter | Others | ut |
| 141 | 1159 | [LNL Windows][Test by CD Nightly Wheels] hugging face model -... | P2 | E2E benchmark model issue | Bug/Perf issue pending reporter response | libohao1201 | Awaiting response from reporter | Dtype / Precision Related | e2e |
| 142 | 492 | Timm_efficientdet NotImplementedError: The original model... | P0 | Impacts customer custom model/application | Bug/Perf issue pending reporter response | mengfei25 | Awaiting response from reporter | Dtype / Precision Related | e2e |
| 143 | 489 | Moco NotImplementedError: xpu not supported | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | mengfei25 | Awaiting response from reporter | Dtype / Precision Related | e2e |

#### <span id='2.2-6-e2e-accuracy-issue'>2.2.6 E2E accuracy issue - Reporter</span> (11 issues)

| # | ID | Title | Priority | Priority Reason | Action Reason | Owner Transfer | Action TBD | Category | Test Module |
|---|------|------|----------|--------------------------------------------|----------|----------|------------|------------|-----------|
| 1 | 3290 | huggingface amp_fp16 inference accuracy openai/whisper-tiny... | P1 | E2E benchmark accuracy issue | E2E accuracy issue pending - needs upstream investigation | jianyizh, weishi-deng | E2E accuracy issue | Dtype / Precision Related | e2e |
| 2 | 3151 | [Triton] Timm_models rexnet_100 / fbnetv3_b /... | P0 | Impacts customer custom model/application | E2E accuracy issue pending - needs upstream investigation | None | E2E accuracy issue | Inductor / Compilation Related | e2e |
| 3 | 3148 | [Triton] Huggingface openai/whisper-tiny got fail_accuracy | P1 | E2E benchmark accuracy issue | E2E accuracy issue pending - needs upstream investigation | None | E2E accuracy issue | Inductor / Compilation Related | e2e |
| 4 | 3058 | [E2E] hf_GPT2_large amp_fp16/amp_bf16 training got fail_accuracy | P1 | E2E accuracy/functionality issue | E2E accuracy issue pending - needs upstream investigation | weishi-deng | E2E accuracy issue | Dtype / Precision Related | e2e |
| 5 | 2984 | [release/2.11] sebotnet33ts_256 fp32 training got fail_accuracy | P1 | E2E accuracy/functionality issue | E2E accuracy issue pending - needs upstream investigation | jianyizh, weishi-deng | E2E accuracy issue | Dtype / Precision Related | e2e |
| 6 | 2928 | [release/2.11] pyhpc_turbulent_kinetic_energy fp32 inference... | P1 | E2E accuracy/functionality issue | E2E accuracy issue pending - needs upstream investigation | jianyizh | E2E accuracy issue | Dtype / Precision Related | e2e |
| 7 | 2924 | [release/2.11] xcit_large_24_p8_224 amp_bf16 training got... | P1 | E2E accuracy/functionality issue | E2E accuracy issue pending - needs upstream investigation | jianyizh, mengfei25 | E2E accuracy issue | Inductor / Compilation Related | e2e |
| 8 | 2908 | [release/2.11] Model fail_accuracy for 5 testcases | P1 | E2E custom model accuracy/functionality issue | E2E accuracy issue pending - needs upstream investigation | xuhancn | E2E accuracy issue | Dtype / Precision Related | e2e |
| 9 | 2592 | [release/2.10] models got fail_accuracy | P0 | Impacts customer custom model/application | E2E accuracy issue pending - needs upstream investigation | mengfei25 | E2E accuracy issue | Dtype / Precision Related | e2e |
| 10 | 1866 | [release 2.8]Torchbench vision_maskrcnn (amp_b16/amp_fp16... | P1 | E2E accuracy/functionality issue | E2E accuracy issue pending - needs upstream investigation | BartoszKokoszko | E2E accuracy issue | Dtype / Precision Related | e2e |
| 11 | 1778 | [Infra] Show known issues for accuracy test | P1 | E2E accuracy/functionality issue | E2E accuracy issue pending - needs upstream investigation | mengfei25 | E2E accuracy issue | Dtype / Precision Related | e2e |


---

## <span id='3-issues-by-category'>3. Issues by Category</span>

### <span id='distributed'>Distributed</span> (37 issues)

| # | ID | Title | Priority | Priority Reason | Action Reason | Owner Transfer | Category | Test Module |
|---|------|------|----------|--------------------------------------------|----------|----------|------------|------------|
| 1 | 3306 | [distributed] no attribute '_reset_fr_recording_xccl' in... | P2 | UT issue with few failures | No specific action identified - needs investigation | None | Distributed | ut |
| 2 | 3305 | [distributed] shrink operation support in... | P2 | UT issue with few failures | No specific action identified - needs investigation | None | Distributed | ut |
| 3 | 3233 | [distributed] RuntimeError: No backend for the parent process... | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | zxd1997066 | Distributed | ut |
| 4 | 3232 | [distributed][tensor] AssertionError: AssertionError not... | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | zxd1997066 | Distributed | ut |
| 5 | 3139 | [distributed][_composable] AssertionError: Expects xpu:0 but... | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | zxd1997066 | Distributed | ut |
| 6 | 3101 | [distributed] 'torch._C._distributed_c10d.ProcessGroupXCCL'... | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | zxd1997066 | Distributed | ut |
| 7 | 3100 | [distributed] /handler/dump_nccl_trace_pickle and nccl_log... | P2 | UT issue with few failures | No specific action identified - needs investigation | None | Distributed | ut |
| 8 | 3082 | multithread support in distributed | P2 | UT issue with few failures | No specific action identified - needs investigation | None | Distributed | ut |
| 9 | 3022 | [distributed] batch_isend_irecv Compatibility Issue on B60/XCCL | P2 | UT issue with few failures | No specific action identified - needs investigation | zhangxiaoli73 | Distributed | ut |
| 10 | 3021 | [distributed] all_to_all_single Compatibility Issue on B60/XCCL | P2 | UT issue with few failures | No specific action identified - needs investigation | zhangxiaoli73 | Distributed | ut |
| 11 | 2972 | [distributed] AssertionError: ValueError not raised in... | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | zxd1997066 | Distributed | ut |
| 12 | 2969 | [distributed] AssertionError: Scalars are not equal! in... | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | zxd1997066 | Distributed | ut |
| 13 | 2968 | [distributed] timeout issue in test/distributed/test_c10d_xccl.py | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | zxd1997066 | Distributed | ut |
| 14 | 2738 | [distributed] test_c10d_spawn_nccl.py ValueError: input... | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | zxd1997066 | Distributed | ut |
| 15 | 2737 | [distributed] AttributeError: module 'torch._C' has no... | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | zxd1997066 | Distributed | ut |
| 16 | 2702 | [distributed] RuntimeError: Work ran time out after 0... | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | madhumitha0102 | Distributed | ut |
| 17 | 2701 | [distributed] Barrier Timeout Error with... | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | madhumitha0102 | Distributed | ut |
| 18 | 2700 | [distributed] Hang issues with test_distributed_spawn.py | P2 | UT issue with few failures | No specific action identified - needs investigation | syedshahbaaz | Distributed | ut |
| 19 | 2686 | [distributed] Accuracy issues with test_distributed_spawn.py | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | madhumitha0102 | Distributed | ut |
| 20 | 2659 | [distributed] test_dist2.py RuntimeError: Backend xccl does... | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | zxd1997066 | Distributed | ut |
| 21 | 2649 | [distributed][pipelining] test_schedule_multiproc.py hang issue | P2 | UT issue with few failures | No specific action identified - needs investigation | syedshahbaaz | Distributed | ut |
| 22 | 2404 | [distributed][checkpoint] AssertionError: Booleans mismatch:... | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | zxd1997066 | Distributed | ut |
| 23 | 2340 | [distributed][_tools] AssertionError: Roofline estimation... | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | zxd1997066 | Distributed | ut |
| 24 | 2165 | [distributed] test_device_mesh.py::TestDeviceMeshGetItem::test... | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | zxd1997066 | Distributed | ut |
| 25 | 2163 | 3 distributed UT cases need to be supported by -... | P2 | UT issue with few failures | No specific action identified - needs investigation | githubsgi | Distributed | ut |
| 26 | 2113 | Update example for Distributed Data Parallel | P2 | UT issue with few failures | No specific action identified - needs investigation | songhappy | Distributed | ut |
| 27 | 2004 | [distributed][shared_tensor] test\distributed\_shard\shared_te... | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | libohao1201 | Distributed | ut |
| 28 | 1727 | [distributed] AttributeError: module 'torch.xpu' has no... | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | PenghuiCheng | Distributed | ut |
| 29 | 1661 | [distributed] Accuracy gap in _composable/fsdp on Xelink | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | PenghuiCheng | Distributed | ut |
| 30 | 1624 | [DONT CLOSE] Known UT Issue list | P2 | UT issue with few failures | No specific action identified - needs investigation | None | Distributed | ut |
| 31 | 1571 | [distributed] ValueError: Cannot use ReduceOp.PREMUL_SUM with... | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | daisyden | Distributed | ut |
| 32 | 1556 | [distributed] NotImplementedError: Operator... | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | PenghuiCheng | Distributed | ut |
| 33 | 1555 | [distributed] RuntimeError: aten.add.Tensor: got mixed... | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | PenghuiCheng | Distributed | ut |
| 34 | 1551 | [distributed] NotImplementedError: The operator... | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | PenghuiCheng | Distributed | ut |
| 35 | 1549 | [distributed] AssertionError: 'fused_all_gather_scaled_matmul'... | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | PenghuiCheng | Distributed | ut |
| 36 | 1548 | [distributed] AssertionError: 'fused_all_gather_matmul' not... | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | PenghuiCheng | Distributed | ut |
| 37 | 1547 | [distributed] NotImplementedError: The operator... | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | PenghuiCheng | Distributed | ut |

### <span id='torchao'>TorchAO</span> (26 issues)

| # | ID | Title | Priority | Priority Reason | Action Reason | Owner Transfer | Category | Test Module |
|---|------|------|----------|--------------------------------------------|----------|----------|------------|------------|
| 1 | 3124 | [TorchAO][Bug] ImportError: Requires mslk >= 1.0.0 when... | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | FRAMEEE17 | TorchAO | ut |
| 2 | 3088 | [TorchAO][BMG] INT4 RTN Flex-attention got 5% performance drop | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | LifengWang | TorchAO | ut |
| 3 | 3076 | [TorchAO][BMG] Llama-3.2-1B-Instruct Dynamic INT8 got 10%... | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | LifengWang | TorchAO | ut |
| 4 | 3032 | [TorchAO][UT] failures in test/prototype/safetensors/test_safe... | P0 | Build crash - critical blocking issue | Bug/Perf issue pending reporter response | zxd1997066 | TorchAO | build |
| 5 | 2823 | [TorchAO][BMG] Llama-3.2-1B-Instruct Dynamic INT8 got 20%... | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | LifengWang | TorchAO | ut |
| 6 | 2722 | [Bug Skip]: NotImplementedError: Could not run 'aten::flip'... | P2 | UT issue with few failures | Bug/Perf issue awaiting reporter response | CuiYifeng | TorchAO | ut |
| 7 | 2707 | [TorchAO][BMG] INT4 GPTQ failed due to TorchAO API change. | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | LifengWang | TorchAO | ut |
| 8 | 2605 | [int4][inductor] Add freezing pattern for fusing int4 mm... | P2 | UT issue with few failures | No specific action identified - needs investigation | None | TorchAO | ut |
| 9 | 2598 | [TorchAO][BMG]The first token latency of... | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | LifengWang | TorchAO | ut |
| 10 | 2597 | [TorchAO][BMG] INT4 GPTQ shows worse performance compared... | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | LifengWang | TorchAO | ut |
| 11 | 2580 | [TorchAO][UT] test/test_low_bit_optim.py AssertionError:... | P0 | Build crash - critical blocking issue | Bug/Perf issue pending reporter response | zxd1997066 | TorchAO | build |
| 12 | 2578 | [TorchAO][UT] test/quantization/test_quant_api.py... | P0 | Build crash - critical blocking issue | Issue is upstream - needs skip PR upstream | Stonepia | TorchAO | build |
| 13 | 2572 | [TorchAO][UT] test/dtypes/test_affine_quantized.py... | P0 | Build crash - critical blocking issue | Bug/Perf issue pending reporter response | zxd1997066 | TorchAO | build |
| 14 | 2533 | Title: [upstream_ut] AttributeError: 'TestQuantizedOpsXPU'... | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | daisyden | TorchAO | ut |
| 15 | 2532 | Title: [upstream_ut] AssertionError: wrong number of... | P2 | UT issue with few failures | Bug/Perf issue awaiting reporter response | daisyden | TorchAO | ut |
| 16 | 2358 | test/test_view_ops.py::TestOldViewOpsXPU::test_ravel_xpu meet... | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | Silv3S | TorchAO | ut |
| 17 | 2327 | [TorchAO] benchmark enabling on XPU | P2 | UT issue with few failures | No specific action identified - needs investigation | None | TorchAO | ut |
| 18 | 2326 | [TorchAO] MX training native PyTorch on XPU | P2 | UT issue with few failures | No specific action identified - needs investigation | riverliuintel | TorchAO | ut |
| 19 | 2325 | [TorchAO] Float8 training support on XPU | P2 | UT issue with few failures | No specific action identified - needs investigation | arlesniak, riverliuintel | TorchAO | ut |
| 20 | 2324 | [TorchAO] FP8 conv support | P2 | UT issue with few failures | No specific action identified - needs investigation | Stonepia | TorchAO | ut |
| 21 | 2323 | [TorchAO] MOE training enabling on XPU | P2 | UT issue with few failures | No specific action identified - needs investigation | riverliuintel | TorchAO | ut |
| 22 | 2207 | Enable FP8/MXFP8 Ops with requests and CUDA alignment | P2 | UT issue with few failures | No specific action identified - needs investigation | Stonepia, CuiYifeng, LuFinch | TorchAO | ut |
| 23 | 2201 | [TorchAO][BMG] When using paged attention backend, all cases... | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | MingxuZh | TorchAO | ut |
| 24 | 1996 | [TorchAO] Memory Efficient Optimizers | P2 | UT issue with few failures | No specific action identified - needs investigation | None | TorchAO | ut |
| 25 | 1912 | Implement the torch.ops.aten._weight_int4pack_mm for... | P2 | UT issue with few failures | No specific action identified - needs investigation | liangan1 | TorchAO | ut |
| 26 | 1894 | [Linux][PT2E] performance test got failed, int8 ASYMM and... | P1 | E2E accuracy/functionality issue | Bug/Perf issue pending reporter response | kaileiyx | TorchAO | e2e |

### <span id='pt2e'>PT2E</span> (7 issues)

| # | ID | Title | Priority | Priority Reason | Action Reason | Owner Transfer | Category | Test Module |
|---|------|------|----------|--------------------------------------------|----------|----------|------------|------------|
| 1 | 3231 | Dynamo failed to run FX node with fake tensors: call_function... | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | daisyden | PT2E | ut |
| 2 | 3010 | [distributed][tensor] test_random_ops.py... | P2 | UT issue with few failures | Bug/Perf issue awaiting reporter response | zxd1997066 | PT2E | ut |
| 3 | 3006 | AssertionError: '.to(tl.float16)' unexpectedly found in '# AOT ID | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | CuiYifeng | PT2E | e2e |
| 4 | 2742 | [Linux][PT2E] hf_Roberta_base model performance ASYMM and... | P0 | Impacts customer custom model/application | Bug/Perf issue pending reporter response | kaileiyx | PT2E | e2e |
| 5 | 2715 | [upstream_ut] torch._dynamo.exc.Unsupported: Attempted to... | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | CuiYifeng | PT2E | ut |
| 6 | 1969 | torch._dynamo.exc.InternalTorchDynamoError: TypeError: cannot... | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | shangerxin | PT2E | ut |
| 7 | 1762 | Add an ocloc AOT target compilation test in cmake | P2 | UT issue with few failures | No specific action identified - needs investigation | chunhuanMeng | PT2E | ut |

### <span id='flash-attention---transformer-related'>Flash Attention / Transformer Related</span> (17 issues)

| # | ID | Title | Priority | Priority Reason | Action Reason | Owner Transfer | Category | Test Module |
|---|------|------|----------|--------------------------------------------|----------|----------|------------|------------|
| 1 | 3229 | RuntimeError: No viable backend for... | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | tszulist-hbn | Flash Attention / Transformer Related | ut |
| 2 | 3195 | test_sdpa_unbacked_no_dde_xpu crashed | P0 | Build crash - critical blocking issue | Bug/Perf issue pending reporter response | daisyden | Flash Attention / Transformer Related | ut |
| 3 | 3176 | [Bug Skip]: ValueError: _scaled_dot_product_attention(): all... | P2 | UT issue with few failures | Bug/Perf issue awaiting reporter response | CuiYifeng | Flash Attention / Transformer Related | ut |
| 4 | 3141 | [upstream_ut] RuntimeError: FlashAttentionForwardXPU only... | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | LuFinch | Flash Attention / Transformer Related | ut |
| 5 | 3136 | [upstream_ut] AssertionError: False is not true in... | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | LuFinch | Flash Attention / Transformer Related | ut |
| 6 | 3133 | [upstream_ut] RuntimeError: scaled_dot_product_attention: If... | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | daisyden | Flash Attention / Transformer Related | ut |
| 7 | 3126 | [upstream_ut] Two NestedTensor issue with flash attention | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | daisyden | Flash Attention / Transformer Related | ut |
| 8 | 3093 | XPU does not support NestedTensor for SDPA operations. | P2 | UT issue with few failures | No specific action identified - needs investigation | None | Flash Attention / Transformer Related | ut |
| 9 | 3007 | AssertionError: Scalars are not equal! with... | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | daisyden | Flash Attention / Transformer Related | e2e |
| 10 | 2853 | [upstream_ut] torch.ops.aten._flash_attention_forward lack of... | P2 | UT issue with few failures | No specific action identified - needs investigation | LuFinch | Flash Attention / Transformer Related | ut |
| 11 | 2802 | Three aten._scaled_dot_product_flash_attention issues | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | LuFinch | Flash Attention / Transformer Related | ut |
| 12 | 2570 | crash in sdpa. | P0 | Build crash - critical blocking issue | Bug/Perf issue pending reporter response | sywangyi | Flash Attention / Transformer Related | ut |
| 13 | 2442 | [Bug Skip]: NotImplementedError: Could not run... | P2 | UT issue with few failures | Bug/Perf issue awaiting reporter response | CuiYifeng | Flash Attention / Transformer Related | ut |
| 14 | 2390 | SDPA in pytorch use different backend compared with ipex | P2 | UT issue with few failures | No specific action identified - needs investigation | LuFinch | Flash Attention / Transformer Related | ut |
| 15 | 2232 | sdpa backward kernel is required to reduce memory usage | P2 | UT issue with few failures | No specific action identified - needs investigation | None | Flash Attention / Transformer Related | ut |
| 16 | 2200 | support flash attention op on XPU device | P2 | UT issue with few failures | No specific action identified - needs investigation | ElaineBao | Flash Attention / Transformer Related | ut |
| 17 | 1749 | transformers UT failure in XPU because SDPA check error... | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | sywangyi | Flash Attention / Transformer Related | ut |

### <span id='sparse-operations-related'>Sparse Operations Related</span> (13 issues)

| # | ID | Title | Priority | Priority Reason | Action Reason | Owner Transfer | Category | Test Module |
|---|------|------|----------|--------------------------------------------|----------|----------|------------|------------|
| 1 | 3166 | test_consistency_SparseCSR failures | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | yucai-intel | Sparse Operations Related | ut |
| 2 | 3081 | Sparse CSR gemm-like ops have not been supported yet | P2 | UT issue with few failures | No specific action identified - needs investigation | tszulist-hbn | Sparse Operations Related | ut |
| 3 | 2801 | to_dense() for Sparse CSR backend cannot broadcast batch dim... | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | jenniew | Sparse Operations Related | ut |
| 4 | 2663 | test_sparse_semi_structured.py gaps | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | None | Sparse Operations Related | ut |
| 5 | 2283 | [upstream_ut] sparse._sampled_addmm is not supported | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | jenniew | Sparse Operations Related | ut |
| 6 | 2246 | torch/sparse/_triton_ops*.py need to be ported to enable for... | P2 | UT issue with few failures | No specific action identified - needs investigation | None | Sparse Operations Related | ut |
| 7 | 2245 | oneDNN matmul received incorrect shape in... | P2 | UT issue with few failures | Bug/Perf issue awaiting reporter response | wincent8 | Sparse Operations Related | ut |
| 8 | 2244 | test/test_sparse_csr.py::TestSparseCSRXPU::test_block_addmm... | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | wincent8 | Sparse Operations Related | ut |
| 9 | 2235 | test/test_sparse_csr.py::TestSparseCompressedTritonKernelsXPU:... | P2 | UT issue with few failures | No specific action identified - needs investigation | None | Sparse Operations Related | ut |
| 10 | 2230 | test_sparse_csr.py::TestSparseCompressedTritonKernelsXPU::test... | P2 | UT issue with few failures | Bug/Perf issue awaiting reporter response | wincent8 | Sparse Operations Related | ut |
| 11 | 2229 | test/test_sparse_csr.py::TestSparseCompressedCPU::test_invalid... | P2 | UT issue with few failures | No specific action identified - needs investigation | jenniew | Sparse Operations Related | ut |
| 12 | 2220 | test/test_sparse_csr.py::TestSparseCompressedTritonKernelsXPU:... | P2 | UT issue with few failures | No specific action identified - needs investigation | None | Sparse Operations Related | ut |
| 13 | 2214 | test/test_sparse.py::TestSparseAnyXPU::test_gradcheck_mm... | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | jenniew | Sparse Operations Related | ut |

### <span id='inductor---compilation-related'>Inductor / Compilation Related</span> (30 issues)

| # | ID | Title | Priority | Priority Reason | Action Reason | Owner Transfer | Category | Test Module |
|---|------|------|----------|--------------------------------------------|----------|----------|------------|------------|
| 1 | 3191 | torch._inductor.exc.InductorError: AssertionError: both a... | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | mengfei25 | Inductor / Compilation Related | e2e |
| 2 | 3187 | PyTorch XPU gpu_cpp_wrapper fails with InductorError... | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | CuiYifeng | Inductor / Compilation Related | ut |
| 3 | 3158 | AttributeError: module 'triton.compiler' has no attribute... | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | kdrozd-dev | Inductor / Compilation Related | ut |
| 4 | 3151 | [Triton] Timm_models rexnet_100 / fbnetv3_b /... | P0 | Impacts customer custom model/application | E2E accuracy issue pending - needs upstream investigation | None | Inductor / Compilation Related | e2e |
| 5 | 3148 | [Triton] Huggingface openai/whisper-tiny got fail_accuracy | P1 | E2E benchmark accuracy issue | E2E accuracy issue pending - needs upstream investigation | None | Inductor / Compilation Related | e2e |
| 6 | 3095 | cutlass support blocks some unit test cases | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | None | Inductor / Compilation Related | ut |
| 7 | 3094 | XPUGraph tree support | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | None | Inductor / Compilation Related | ut |
| 8 | 3089 | AssertionError: Torch not compiled with CUDA enabled | P2 | UT issue with few failures | Bug/Perf issue awaiting reporter response | jmamzax | Inductor / Compilation Related | ut |
| 9 | 2997 | AssertionError of test_linear_and_cel_max_autotune | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | etaf | Inductor / Compilation Related | ut |
| 10 | 2958 | AssertionError of test_dtensor_basic_compile | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | daisyden | Inductor / Compilation Related | ut |
| 11 | 2935 | [release/2.11][inductor] huggingface amp_fp16 and float16... | P0 | Regression - passed before but failed now | Bug/Perf issue pending reporter response | agnottaski | Inductor / Compilation Related | e2e |
| 12 | 2924 | [release/2.11] xcit_large_24_p8_224 amp_bf16 training got... | P1 | E2E accuracy/functionality issue | E2E accuracy issue pending - needs upstream investigation | jianyizh, mengfei25 | Inductor / Compilation Related | e2e |
| 13 | 2922 | [release/2.11] UT inductor AssertionError: pass_fds not... | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | bjarzemb | Inductor / Compilation Related | ut |
| 14 | 2888 | torch._inductor.exc.InductorError: AssertionError:... | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | Stonepia | Inductor / Compilation Related | ut |
| 15 | 2810 | AssertionError: Object comparison failed:... | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | daisyden | Inductor / Compilation Related | ut |
| 16 | 2806 | CompiledAOTI need XPU support | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | daisyden | Inductor / Compilation Related | ut |
| 17 | 2694 | Title: [upstream_ut] AssertionError: Tensor-likes are not... | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | daisyden | Inductor / Compilation Related | ut |
| 18 | 2693 | Title: [upstream_ut] AssertionError: Scalars are not equal! | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | hoshibara | Inductor / Compilation Related | ut |
| 19 | 2620 | [upstream_ut] AssertionError: dtype is needed to compute eps1... | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | daisyden | Inductor / Compilation Related | ut |
| 20 | 2619 | [release/2.10] Some models inductor performance dropped ~ 10%... | P0 | Impacts customer custom model/application | Bug/Perf issue pending reporter response | mengfei25 | Inductor / Compilation Related | e2e |
| 21 | 2613 | [upstream_ut] AssertionError: Tensor-likes are not equal! in... | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | daisyden | Inductor / Compilation Related | ut |
| 22 | 2611 | [upstream_ut] AssertionError: Tensor-likes are not equal! in... | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | daisyden | Inductor / Compilation Related | ut |
| 23 | 2609 | [upstream_ut] torch._inductor.exc.InductorError:... | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | daisyden | Inductor / Compilation Related | ut |
| 24 | 2554 | [upstream_ut] AssertionError: AssertionError not raised | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | daisyden | Inductor / Compilation Related | ut |
| 25 | 2329 | [upstream_ut] feature missing: get_device_tflops and... | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | etaf | Inductor / Compilation Related | ut |
| 26 | 2169 | Frame size comparison failed in test_size_comparison_no_recompile | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | daisyden | Inductor / Compilation Related | ut |
| 27 | 2132 | [2.9][BMG-Windows][Torch-xpu-ops UT] 1 case failed with... | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | libohao1201 | Inductor / Compilation Related | ut |
| 28 | 2024 | AssertionError: Torch not compiled with CUDA enabled | P2 | UT issue with few failures | Bug/Perf issue awaiting reporter response | mengfei25 | Inductor / Compilation Related | ut |
| 29 | 1970 | torch._dynamo.exc.BackendCompilerFailed: backend='inductor'... | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | shangerxin | Inductor / Compilation Related | ut |
| 30 | 1505 | [ARC-WSL-Ubuntu24.04] 15 Timm models got fail_accuracy | P0 | Impacts customer custom model/application | Issue is upstream - needs skip PR upstream | None | Inductor / Compilation Related | e2e |

### <span id='others'>Others</span> (214 issues)

| # | ID | Title | Priority | Priority Reason | Action Reason | Owner Transfer | Category | Test Module |
|---|------|------|----------|--------------------------------------------|----------|----------|------------|------------|
| 1 | 3300 | [CI] When creating PR, several pull workflows are launched... | P2 | UT issue with few failures | No specific action identified - needs investigation | None | Others | ut |
| 2 | 3286 | New failing test case after enabling tests from... | P2 | UT issue with few failures | PR closed but no failed tests - verify if issue still reproduces | BBBela | Others | ut |
| 3 | 3284 | Optimize torch.nn.functional.one_hot | P2 | UT issue with few failures | PR closed but no failed tests - verify if issue still reproduces | xinyu-intel | Others | ut |
| 4 | 3280 | [Bug Skip]: New UT failure in 0406 nightly windows. | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | RUIJIEZHONG66166 | Others | ut |
| 5 | 3270 | [distributed][tensor] RuntimeError: Invalid scaling... | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | zxd1997066 | Others | ut |
| 6 | 3267 | New failed test cases 2026-04-06 | P2 | UT issue with few failures | Bug/Perf issue awaiting reporter response | zxd1997066 | Others | ut |
| 7 | 3266 | [RFC] Migrate XPU kernel math functions from std::/:: to... | P2 | UT issue with few failures | No specific action identified - needs investigation | None | Others | ut |
| 8 | 3259 | New failed test cases 2026-04-02 | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | Silv3S | Others | ut |
| 9 | 3258 | huggingface accuracy inference Error in op:... | P2 | UT issue with few failures | PR closed but no failed tests - verify if issue still reproduces | bjarzemb | Others | ut |
| 10 | 3247 | NotImplementedError: "dot_xpu_mkl" not implemented for 'Long' | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | Silv3S | Others | ut |
| 11 | 3246 | AssertionError: Booleans mismatch: True is not False | P2 | UT issue with few failures | Bug/Perf issue awaiting reporter response | Silv3S | Others | ut |
| 12 | 3243 | AssertionError: False is not true | P2 | UT issue with few failures | Bug/Perf issue awaiting reporter response | zxd1997066 | Others | ut |
| 13 | 3242 | AssertionError: Torch not compiled with CUDA enabled | P2 | UT issue with few failures | Bug/Perf issue awaiting reporter response | zxd1997066 | Others | ut |
| 14 | 3227 | torch xpu event has ~0.1ms latency, which is too large | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | jianyizh | Others | ut |
| 15 | 3224 | [Win][Build] Building SYCL (Device) object... | P0 | Build crash - critical blocking issue | Bug/Perf issue pending reporter response | anmyachev | Others | build |
| 16 | 3216 | [OPs] Some ops of XPU have non-determinism and are... | P2 | UT issue with few failures | No specific action identified - needs investigation | CuiYifeng | Others | ut |
| 17 | 3209 | [Win][Build] There is Cyclic dependencies error when build... | P0 | Build crash - critical blocking issue | Bug/Perf issue pending reporter response | NeoZhangJianyu | Others | build |
| 18 | 3196 | vitals is not supported, the cases should be disabled | P2 | UT issue with few failures | No specific action identified - needs investigation | libohao1201 | Others | ut |
| 19 | 3194 | Incorrect strides in TestCommonXPU,test_out_addmv_xpu_float32 | P2 | UT issue with few failures | No specific action identified - needs investigation | AKloniecki | Others | ut |
| 20 | 3189 | Task Tracker | P2 | UT issue with few failures | No specific action identified - needs investigation | guangyey | Others | ut |
| 21 | 3184 | New failing UTs: test_cross_entropy_loss_2d_out_of_bounds_clas... | P2 | UT issue with few failures | Bug/Perf issue awaiting reporter response | BBBela | Others | ut |
| 22 | 3180 | [E2E] Timm/Torchbench models got "eager_two_runs_differ" on ARC | P0 | Impacts customer custom model/application | No specific action identified - needs investigation | None | Others | ut |
| 23 | 3178 | New failed test cases 2026-03-25 | P2 | UT issue with few failures | Bug/Perf issue awaiting reporter response | BBBela | Others | ut |
| 24 | 3175 | [Bug Skip]: ValueError: sampled_addmm(): all inputs are... | P2 | UT issue with few failures | Bug/Perf issue awaiting reporter response | CuiYifeng | Others | ut |
| 25 | 3174 | [Bug Skip]: Accuracy failure of test_Conv2d_groups_nobias | P2 | UT issue with few failures | All test cases passed on both XPU and stock - issue is resolved | CuiYifeng | Others | ut |
| 26 | 3170 | Unskip test_bmm_windows_error_xpu_float64 | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | jenniew | Others | ut |
| 27 | 3169 | NotImplementedError: Could not run 'aten::hspmm' with... | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | jkosnox | Others | ut |
| 28 | 3167 | NotImplementedError: Could not run 'aten::triangular_solve.X'... | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | tszulist-hbn | Others | ut |
| 29 | 3165 | test_sparse_csr_xpu.py::TestSparseCompressedTritonKernelsXPU::... | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | None | Others | ut |
| 30 | 3160 | compiler not found (Windows) | P2 | UT issue with few failures | All test cases passed on both XPU and stock - issue is resolved | kdrozd-dev | Others | ut |
| 31 | 3156 | AssertionError: 'Assertion cur_target >= 0 && cur_target <... | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | kdrozd-dev | Others | ut |
| 32 | 3150 | [Task] Align XPU kernel's implementation to stock PyTorch | P2 | UT issue with few failures | No specific action identified - needs investigation | guangyey | Others | ut |
| 33 | 3143 | NotImplementedError: The operator... | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | LuFinch | Others | ut |
| 34 | 3142 | [upstream_ut] RuntimeError: The sycl_ext_oneapi_work_group_scr... | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | LuFinch | Others | ut |
| 35 | 3140 | [upstream_ut] RuntimeError: FlashAttentionForwardXPU does not... | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | LuFinch | Others | ut |
| 36 | 3132 | [upstream_ut] transfomers test reports RuntimeError: No... | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | LuFinch | Others | ut |
| 37 | 3131 | [upstream_ut] NotImplementedError: The operator... | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | chunhuanMeng | Others | ut |
| 38 | 3129 | [upstream_ut] AssertionError: UserWarning not triggered | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | daisyden | Others | ut |
| 39 | 3128 | [upstream_ut] AssertionError: RuntimeError not raised by <lambda> | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | daisyden | Others | ut |
| 40 | 3127 | [upstream_ut] AssertionError: AssertionError not raised | P2 | UT issue with few failures | Issue marked as not_target/wontfix - should be skipped for XPU enablement | daisyden | Others | ut |
| 41 | 3121 | [Bug Skip]: CUDA specific UT test_fft_half_and_chalf_not_power... | P2 | UT issue with few failures | Bug/Perf issue awaiting reporter response | CuiYifeng | Others | ut |
| 42 | 3114 | [Bug Skip]: Failure skip on 2026-3-21 | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | guangyey | Others | ut |
| 43 | 3106 | Worker crashes when running TestDecompXPU,test_quick_core_back... | P0 | Build crash - critical blocking issue | Bug/Perf issue pending reporter response | BBBela | Others | ut |
| 44 | 3102 | [distributed] RuntimeError: Invalid device string: 'xpu:foo'... | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | zxd1997066 | Others | ut |
| 45 | 3096 | VISIBLE_DEVICE support | P2 | UT issue with few failures | No specific action identified - needs investigation | None | Others | ut |
| 46 | 3086 | nvml support blocks some test cases | P2 | UT issue with few failures | No specific action identified - needs investigation | None | Others | ut |
| 47 | 3083 | [Bug Skip]: Random failures 2026WW12 | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | CuiYifeng | Others | ut |
| 48 | 3080 | cudagraph tests blocked by feature gap | P2 | UT issue with few failures | No specific action identified - needs investigation | None | Others | ut |
| 49 | 3077 | [Bug Skip] test_dlpack.py::TestTorchDlPackXPU::test_no_copy_xp... | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | AKloniecki | Others | ut |
| 50 | 3074 | [Bug Skip] test_dlpack_exchange_api expect... | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | shangerxin | Others | ut |
| 51 | 3060 | Implement torch._scaled_grouped_mm for xpu backend | P2 | UT issue with few failures | No specific action identified - needs investigation | Stonepia, liangan1 | Others | ut |
| 52 | 3048 | Profiler result is not correct on B70 | P2 | UT issue with few failures | No specific action identified - needs investigation | aostrowski-hbn | Others | ut |
| 53 | 3041 | AssertionError: Expected len(flat_diff_results) > 0 in... | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | Silv3S | Others | ut |
| 54 | 3033 | [Bug Skip]: Softmax tolerance | P2 | UT issue with few failures | Bug/Perf issue awaiting reporter response | chunhuanMeng | Others | ut |
| 55 | 3030 | [Bug Skip] test/test_modules.py::TestModuleXPU::test_cpu_gpu_p... | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | shangerxin | Others | ut |
| 56 | 3025 | New failing test in Nightly Wheel... | P2 | UT issue with few failures | Bug/Perf issue awaiting reporter response | BBBela | Others | ut |
| 57 | 3024 | Enable clang-tidy checks | P2 | UT issue with few failures | No specific action identified - needs investigation | Silv3S | Others | ut |
| 58 | 3014 | [upstream_ut] AssertionError: False is not true | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | Silv3S | Others | ut |
| 59 | 3013 | [upstream_ut] RuntimeError: Kernel is incompatible with all... | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | Silv3S | Others | ut |
| 60 | 3011 | [upstream_ut] torch.OutOfMemoryError: XPU out of memory | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | Silv3S | Others | ut |
| 61 | 3004 | TypeError: _xpu_recordMemoryHistory(): incompatible function... | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | guangyey | Others | ut |
| 62 | 3000 | [Bug Skip]: RuntimeError: _share_fd_: only available on CPU... | P2 | UT issue with few failures | Bug/Perf issue awaiting reporter response | zxd1997066 | Others | ut |
| 63 | 2999 | KeyError: 'eager_numerics.use_pytorch_libdevice' | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | daisyden | Others | ut |
| 64 | 2993 | [Bug Skip]: Unexpected success of... | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | CuiYifeng | Others | ut |
| 65 | 2981 | [release/2.11] T5 models performance dropped ~20% | P0 | Impacts customer custom model/application | Bug/Perf issue pending reporter response | mengfei25 | Others | e2e |
| 66 | 2979 | eca_halonext26ts got RuntimeError:... | P0 | Build crash - critical blocking issue | Bug/Perf issue pending reporter response | mengfei25 | Others | e2e |
| 67 | 2966 | [Bug Skip]: [Regression]2026-3-2 ut failures | P0 | Regression - passed before but failed now | Bug/Perf issue pending reporter response | kaileiyx | Others | ut |
| 68 | 2965 | [Bug Skip]: Random failures 2026WW10 | P2 | UT issue with few failures | Bug/Perf issue awaiting reporter response | CuiYifeng | Others | ut |
| 69 | 2953 | [release/2.11][wsl] huggingface TrOCRForCausalLM and... | P2 | E2E benchmark model issue | Bug/Perf issue pending reporter response | bjarzemb | Others | e2e |
| 70 | 2950 | SYCL compilation flag -fsycl-id-queries-fit-in-int does not... | P2 | UT issue with few failures | No specific action identified - needs investigation | BBBela | Others | ut |
| 71 | 2948 | [AO] Benchmark enabling on XPU | P2 | UT issue with few failures | No specific action identified - needs investigation | None | Others | ut |
| 72 | 2946 | [Bug Skip]: Random failures 2026WW09 | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | CuiYifeng | Others | ut |
| 73 | 2942 | [Windows] Unit tests got Fatal python error | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | mengfei25 | Others | ut |
| 74 | 2939 | [release/2.11] gmlp_s16_224 inference amp performance dropped... | P2 | E2E performance issue | Bug/Perf issue pending reporter response | mengfei25 | Others | e2e |
| 75 | 2938 | [release/2.11] basic_gnn_gin and basic_gnn_sage inference... | P2 | E2E performance issue | Bug/Perf issue pending reporter response | mengfei25 | Others | e2e |
| 76 | 2921 | [abs][complex64] - new failing test cases caused by PyTorch... | P2 | UT issue with few failures | Bug/Perf issue awaiting reporter response | BBBela | Others | ut |
| 77 | 2919 | [XPU][upstream_ut][COW] Fix materialization in remaining... | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | gplutop7 | Others | ut |
| 78 | 2918 | [XPU][upstream_ut][COW] Skip non-supported ops (jiterator +... | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | gplutop7 | Others | ut |
| 79 | 2914 | Test case test/test_autograd.py::TestAutogradMultipleDispatchC... | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | shangerxin | Others | ut |
| 80 | 2912 | [release/2.11] UT extended 220 new failures | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | bjarzemb | Others | ut |
| 81 | 2907 | [release/2.11] Models performance regression for 5 testcases | P0 | Regression - passed before but failed now | Bug/Perf issue pending reporter response | bjarzemb | Others | ut |
| 82 | 2891 | RuntimeError: Expected to find "(262144, 0, 512, 1" but did... | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | chunhuanMeng | Others | e2e |
| 83 | 2879 | RuntimeError: _share_fd_: only available on CPU | P2 | UT issue with few failures | Bug/Perf issue awaiting reporter response | Silv3S | Others | ut |
| 84 | 2873 | [Bug Skip]: test_repos.py contains several failed ops | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | shangerxin | Others | ut |
| 85 | 2869 | [Bug Skip]: New UT failure in 0209 nightly windows. | P2 | UT issue with few failures | Bug/Perf issue awaiting reporter response | RUIJIEZHONG66166 | Others | ut |
| 86 | 2858 | [Bug Skip]: test_xpu new failures | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | RUIJIEZHONG66166 | Others | ut |
| 87 | 2852 | [Bug Skip]: New UT failures in 0206 nightly on Windows | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | chuanqi129 | Others | ut |
| 88 | 2845 | [Bug Skip]:[UT] [Windows] failed cases 2026-2-4 | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | kaileiyx | Others | ut |
| 89 | 2817 | Expected error message is different than actual | P2 | UT issue with few failures | Bug/Perf issue awaiting reporter response | Silv3S | Others | ut |
| 90 | 2816 | torch.logcumsumexp incorrectly returns NaNs for complex64 input | P2 | UT issue with few failures | No specific action identified - needs investigation | Silv3S | Others | ut |
| 91 | 2815 | RuntimeError: output with shape [2] doesn't match the... | P2 | UT issue with few failures | Bug/Perf issue awaiting reporter response | Silv3S | Others | ut |
| 92 | 2811 | [Bug Skip]: [Regression] failed cases 2026-2-2 | P0 | Regression - passed before but failed now | Bug/Perf issue pending reporter response | kaileiyx | Others | ut |
| 93 | 2800 | AttributeError: 'torch._C._XpuDeviceProperties' object has no... | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | guangyey | Others | ut |
| 94 | 2798 | Test case test/test_dlpack.py::TestTorchDlPackCPU::test_numpy_... | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | None | Others | ut |
| 95 | 2795 | Histc raises error with integer input when deterministic... | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | YangKai0616 | Others | ut |
| 96 | 2783 | [Bug Skip]: Key "xpu" is missing from dict "driver" in test_svd | P2 | UT issue with few failures | Bug/Perf issue awaiting reporter response | CuiYifeng | Others | ut |
| 97 | 2777 | [Bug Skip]: Random failures 2026WW05 | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | CuiYifeng | Others | ut |
| 98 | 2769 | [oneDNN] New failed test cases with 3.11 compared with 3.10 | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | mengfei25 | Others | ut |
| 99 | 2767 | [UT] test_control_flow_xpu.py got AssertionError | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | libohao1201 | Others | ut |
| 100 | 2766 | MaxPool2d - investigate memory layout performance | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | pbielak | Others | ut |
| 101 | 2759 | [Bug Skip]: New failed cases 2026-1-22 | P2 | UT issue with few failures | Bug/Perf issue awaiting reporter response | kaileiyx | Others | ut |
| 102 | 2751 | [Bug Skip]: Random failures 2026WW04 | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | CuiYifeng | Others | ut |
| 103 | 2744 | [Bug Skip]: extended test failures when test_compare_cpu atol... | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | daisyden | Others | ut |
| 104 | 2729 | [Bug Skip]: Random failures 2026WW03 | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | CuiYifeng | Others | ut |
| 105 | 2720 | [upstream_ut] RuntimeError: false INTERNAL ASSERT FAILED at... | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | wincent8 | Others | ut |
| 106 | 2714 | [upstream_ut] AssertionError: Object comparison failed:... | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | Silv3S | Others | ut |
| 107 | 2712 | [upstream_ut] RuntimeError: Cannot swap t2 because it has... | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | tszulist-hbn | Others | ut |
| 108 | 2698 | Title: [upstream_ut] RuntimeError: FlashAttentionForwardXPU... | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | chunhuanMeng, LuFinch | Others | ut |
| 109 | 2697 | Title: [upstream_ut] RuntimeError: Expected to find ", 0, "... | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | chunhuanMeng | Others | e2e |
| 110 | 2689 | [LNL][Windows] AssertionError: 'Assertion `cur_target >= 0 &&... | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | kaileiyx | Others | ut |
| 111 | 2676 | Random failure in CI test | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | daisyden | Others | ut |
| 112 | 2675 | [Bug Skip]: AttributeError: 'NoneType' object has no... | P2 | UT issue with few failures | Bug/Perf issue awaiting reporter response | kaileiyx | Others | ut |
| 113 | 2670 | [upstream_ut] RuntimeError: could not create a primitive... | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | tszulist-hbn | Others | ut |
| 114 | 2669 | [upstream_ut] AssertionError: Tensor-likes are not close! in... | P2 | UT issue with few failures | Bug/Perf issue awaiting reporter response | daisyden | Others | ut |
| 115 | 2662 | [release/2.10][Windows][BMG] New failed test cases and 2.9... | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | mengfei25 | Others | ut |
| 116 | 2660 | [release/2.10][Windows][BMG] New failed test cases | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | mengfei25 | Others | ut |
| 117 | 2639 | test_to() failed during rnn isinstance() check | P2 | UT issue with few failures | Bug/Perf issue awaiting reporter response | daisyden | Others | ut |
| 118 | 2630 | Title: [upstream_ut] AssertionError: Scalars are not equal! | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | daisyden | Others | ut |
| 119 | 2618 | [Bug Skip]: [regression] AssertionError: Scalars are not... | P0 | Regression - passed before but failed now | Bug/Perf issue awaiting reporter response | kaileiyx | Others | ut |
| 120 | 2595 | [Bug Skip]: Random crashed cases 2025-12-17 | P0 | Build crash - critical blocking issue | Bug/Perf issue awaiting reporter response | CuiYifeng | Others | ut |
| 121 | 2562 | Warning as Error | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | EikanWang | Others | ut |
| 122 | 2560 | [UT] "RuntimeError: iter.device(arg).is_xpu()" in... | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | libohao1201 | Others | ut |
| 123 | 2541 | Title: [upstream_ut] RuntimeError: could not construct a... | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | daisyden | Others | ut |
| 124 | 2539 | Title: [upstream_ut] RuntimeError: Tried to instantiate dummy... | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | daisyden | Others | ut |
| 125 | 2537 | Title: [upstream_ut] Failed: Unexpected success | P2 | UT issue with few failures | Bug/Perf issue awaiting reporter response | daisyden | Others | ut |
| 126 | 2536 | Title: [upstream_ut] AttributeError: module 'torch._C' has no... | P2 | UT issue with few failures | Bug/Perf issue awaiting reporter response | daisyden | Others | ut |
| 127 | 2535 | Title: [upstream_ut] AttributeError: module 'torch._C' has no... | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | daisyden | Others | ut |
| 128 | 2531 | [upstream_ut] AssertionError: Torch not compiled with CUDA... | P2 | UT issue with few failures | Bug/Perf issue awaiting reporter response | daisyden | Others | ut |
| 129 | 2530 | Title: [upstream_ut] AssertionError: RuntimeError not raised | P2 | UT issue with few failures | Bug/Perf issue awaiting reporter response | daisyden | Others | ut |
| 130 | 2529 | [upstream_ut] AssertionError: False is not true | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | daisyden | Others | ut |
| 131 | 2519 | [upstream_ut] TypeError: map2_ is only implemented on CPU tensors | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | libohao1201 | Others | ut |
| 132 | 2518 | [upstream_ut] TypeError: Creating a Tensor subclass from a... | P2 | UT issue with few failures | All test cases passed on both XPU and stock - issue is resolved | libohao1201 | Others | ut |
| 133 | 2513 | [upstream_ut] RuntimeError: _share_fd_: only available on CPU | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | libohao1201 | Others | ut |
| 134 | 2512 | [upstream_ut] RuntimeError: _histc_xpu does not have a... | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | libohao1201 | Others | ut |
| 135 | 2510 | [upstream_ut] RuntimeError: Expected output.numel() <=... | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | libohao1201 | Others | ut |
| 136 | 2508 | TypedStorage / TypedTensors deprecation | P2 | UT issue with few failures | Issue marked as not_target/wontfix - should be skipped for XPU enablement | libohao1201 | Others | ut |
| 137 | 2496 | [upstream_ut] Segmentation fault when running... | P0 | Build crash - critical blocking issue | All test cases passed on both XPU and stock - issue is resolved | libohao1201 | Others | ut |
| 138 | 2491 | [upstream_ut] AssertionError: False is not true | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | libohao1201 | Others | ut |
| 139 | 2479 | [Bug] torch.rand output different result on bmg and pvc | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | zufangzhu | Others | ut |
| 140 | 2472 | [upstream_ut] NotImplementedError: The operator... | P2 | UT issue with few failures | Issue marked as not_target/wontfix - should be skipped for XPU enablement | daisyden | Others | ut |
| 141 | 2471 | test_cuda.py gaps | P2 | UT issue with few failures | No specific action identified - needs investigation | guangyey | Others | ut |
| 142 | 2467 | Host may stuck when submit too many kernels when event recording | P2 | UT issue with few failures | No specific action identified - needs investigation | jianyizh | Others | ut |
| 143 | 2465 | [windows] ut hang | P2 | UT issue with few failures | No specific action identified - needs investigation | tadkrawiec, mganczarenko | Others | ut |
| 144 | 2463 | [Bug Skip]: OSError: SYCL runtime is not dected. | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | RUIJIEZHONG66166 | Others | ut |
| 145 | 2446 | [Bug Skip]: AssertionError: "Simulate error" does not match... | P2 | UT issue with few failures | Bug/Perf issue awaiting reporter response | kaileiyx | Others | ut |
| 146 | 2444 | [upstream_ut] RuntimeError: UR backend failed. UR backend... | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | wincent8 | Others | ut |
| 147 | 2436 | [upstream_ut] AttributeError: 'NoneType' object has no... | P2 | UT issue with few failures | Bug/Perf issue awaiting reporter response | daisyden | Others | ut |
| 148 | 2434 | [Bug Skip]: New failures 2025-11-28 | P2 | UT issue with few failures | Bug/Perf issue awaiting reporter response | mengfei25 | Others | ut |
| 149 | 2425 | [upstream_ut] RuntimeError: Expected both self and other to... | P2 | UT issue with few failures | Bug/Perf issue awaiting reporter response | daisyden | Others | ut |
| 150 | 2412 | Some NestedTensor missing XPU support | P2 | UT issue with few failures | No specific action identified - needs investigation | yucai-intel | Others | ut |
| 151 | 2400 | [ut_upstream] tf32_on_and_off() need xpu support | P2 | UT issue with few failures | No specific action identified - needs investigation | chunhuanMeng | Others | ut |
| 152 | 2392 | [Bug Skip]: torch.OutOfMemoryError: XPU out of memory | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | RUIJIEZHONG66166 | Others | ut |
| 153 | 2389 | [Bug Skip]: RuntimeError: Data corruption detected | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | kaileiyx | Others | ut |
| 154 | 2376 | [Bug Skip]: NotImplementedError: "logaddexp_xpu" not... | P2 | UT issue with few failures | Bug/Perf issue awaiting reporter response | mengfei25 | Others | ut |
| 155 | 2359 | [upstream_ut] GradcheckError: Backward is not reentrant | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | BBBela | Others | ut |
| 156 | 2349 | [oneAPI][backward compatibility] libur_loader.so.0: version... | P2 | UT issue with few failures | No specific action identified - needs investigation | riverliuintel | Others | ut |
| 157 | 2331 | [upstream_ut] AssertionError: Scalars are not equal! with... | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | hoshibara | Others | ut |
| 158 | 2309 | unsupported ops with PYTORCH_ENABLE_XPU_FALLBACK unset | P2 | UT issue with few failures | Issue marked as not_target/wontfix - should be skipped for XPU enablement | daisyden | Others | ut |
| 159 | 2301 | [upstream_ut] dtypes not align with OpInfo | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | daisyden | Others | ut |
| 160 | 2295 | [upstream_ut][xpu][test]nn/test_embedding.py::TestEmbeddingNND... | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | yucai-intel | Others | ut |
| 161 | 2287 | [upstream_ut] test_python_ref issues | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | yucai-intel | Others | ut |
| 162 | 2285 | Support efficient attention | P2 | UT issue with few failures | No specific action identified - needs investigation | chunhuanMeng | Others | ut |
| 163 | 2270 | Backend Compatibility Error in test/xpu/test_decomp.py | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | libohao1201 | Others | ut |
| 164 | 2263 | [xpu][bug] XPU Trace event ends too late! | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | PawelSwider2000 | Others | ut |
| 165 | 2261 | [xpu][profiler] Run with fork process has extra warning | P2 | UT issue with few failures | No specific action identified - needs investigation | moksiuc | Others | ut |
| 166 | 2255 | [upstream_ut] RuntimeError: Long is not supported in oneDNN | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | daisyden | Others | ut |
| 167 | 2253 | the supported dtypes are not align with cuda | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | daisyden | Others | ut |
| 168 | 2250 | Found mismatch when comparing the output of aten.view.default... | P2 | UT issue with few failures | No specific action identified - needs investigation | astachowiczhabana | Others | ut |
| 169 | 2248 | [upstream_ut] test_cow failures | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | gplutop7 | Others | ut |
| 170 | 2240 | RuntimeError: Trying to set a forward gradient that has a... | P2 | UT issue with few failures | Bug/Perf issue awaiting reporter response | zxd1997066 | Others | ut |
| 171 | 2239 | Exception: could not create a primitive descriptor for the... | P2 | UT issue with few failures | Bug/Perf issue awaiting reporter response | zxd1997066 | Others | ut |
| 172 | 2238 | Exception: Tensor-likes are not close! in... | P2 | UT issue with few failures | Bug/Perf issue awaiting reporter response | zxd1997066 | Others | ut |
| 173 | 2234 | [upstream_ut] AssertionError: RuntimeError not raised :... | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | Silv3S | Others | ut |
| 174 | 2217 | AO Performance issue track | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | liangan1 | Others | ut |
| 175 | 2215 | Find use case example for torch-xpu-ops.lib in sycl cpp extension | P2 | UT issue with few failures | No specific action identified - needs investigation | dvrogozh | Others | ut |
| 176 | 2199 | Fix reduction and norm register spill | P2 | UT issue with few failures | No specific action identified - needs investigation | None | Others | ut |
| 177 | 2196 | Fix DistributionElementwiseKernelFunctor register spill | P2 | UT issue with few failures | No specific action identified - needs investigation | None | Others | ut |
| 178 | 2186 | AssertionError: Mul tiheadAttention does not support... | P2 | UT issue with few failures | Bug/Perf issue awaiting reporter response | daisyden | Others | ut |
| 179 | 2164 | skip test_no_cuda_monkeypatch as it is cuda specific | P2 | UT issue with few failures | Issue marked as not_target/wontfix - should be skipped for XPU enablement | daisyden | Others | ut |
| 180 | 2142 | XPU max_memory_allocated have different output with CUDA | P2 | UT issue with few failures | No specific action identified - needs investigation | guangyey | Others | ut |
| 181 | 2140 | Consider how to avoid copy in FFT kernels | P2 | UT issue with few failures | No specific action identified - needs investigation | CuiYifeng | Others | ut |
| 182 | 2127 | Path Coverage enhancement | P2 | UT issue with few failures | No specific action identified - needs investigation | CuiYifeng | Others | ut |
| 183 | 2098 | Upstream XPU functions in yaml | P2 | UT issue with few failures | No specific action identified - needs investigation | guangyey | Others | ut |
| 184 | 2089 | need an implementation that won't initialize gpu context for... | P2 | UT issue with few failures | No specific action identified - needs investigation | guangyey | Others | ut |
| 185 | 2086 | nd_item::barrier has been deprecated | P2 | UT issue with few failures | No specific action identified - needs investigation | dvrogozh | Others | ut |
| 186 | 2063 | Avoid using out-of-date term | P2 | UT issue with few failures | No specific action identified - needs investigation | CuiYifeng | Others | ut |
| 187 | 2022 | [Windows] [CI] [UT] AssertionError: Tensor-likes are not close! | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | RUIJIEZHONG66166 | Others | ut |
| 188 | 2015 | inf is returned by nn.TransformerEncoderLayer | P2 | UT issue with few failures | No specific action identified - needs investigation | yucai-intel | Others | ut |
| 189 | 2006 | work-item/workgroup issue in softmax/unsampling/nonzero | P2 | UT issue with few failures | No specific action identified - needs investigation | BartoszKokoszko | Others | ut |
| 190 | 1986 | torch.xpu._sleep is missing, | P2 | UT issue with few failures | No specific action identified - needs investigation | guangyey | Others | ut |
| 191 | 1973 | AssertionError: Scalars or Tensor-likes are not equal or close! | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | mengfei25 | Others | ut |
| 192 | 1963 | [upstream_ut] MetadataMismatchError in TestFakeTensor of... | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | pbielak | Others | ut |
| 193 | 1951 | Functionality issues in TestCommon.test_out. | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | AKloniecki | Others | ut |
| 194 | 1936 | implement torch.linalg.cholesky xpu backend | P2 | UT issue with few failures | No specific action identified - needs investigation | mwiktor-intel | Others | ut |
| 195 | 1902 | implement torch.linalg.pinv xpu backend | P2 | UT issue with few failures | No specific action identified - needs investigation | mwiktor-intel | Others | ut |
| 196 | 1901 | implement torch.linalg.svd xpu backend | P2 | UT issue with few failures | No specific action identified - needs investigation | CuiYifeng | Others | ut |
| 197 | 1900 | implement torch.linalg.qr xpu backend | P2 | UT issue with few failures | No specific action identified - needs investigation | pbielak | Others | ut |
| 198 | 1893 | [upstream_ut] oneDNN accuracy issues in test_ops_xpu.py | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | chunhuanMeng | Others | ut |
| 199 | 1856 | channel last aten::hardswish_ will call extra copy | P2 | UT issue with few failures | No specific action identified - needs investigation | chunhuanMeng | Others | ut |
| 200 | 1784 | [Performance] Torch XPU Profiler is not reliable | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | liangan1 | Others | ut |
| 201 | 1729 | Validation Check List | P2 | UT issue with few failures | No specific action identified - needs investigation | chuanqi129 | Others | ut |
| 202 | 1722 | Ask an API to query GPU type(iGPU/dGPU). | P2 | UT issue with few failures | No specific action identified - needs investigation | guangyey | Others | ut |
| 203 | 1689 | [For op Perf Comparison] Save reference comparison run id | P2 | UT issue with few failures | No specific action identified - needs investigation | None | Others | ut |
| 204 | 1678 | missing op support for `model.share_memory()` | P0 | Impacts customer custom model/application | No specific action identified - needs investigation | None | Others | ut |
| 205 | 1649 | [cpp extension] Provide a clear error message when using... | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | ZhaoqiongZ | Others | ut |
| 206 | 1645 | [For Comparison] Save reference comparison run id | P2 | UT issue with few failures | No specific action identified - needs investigation | mengfei25 | Others | ut |
| 207 | 1594 | Keep track on the building warning | P0 | Build crash - critical blocking issue | No specific action identified - needs investigation | CuiYifeng, chunhuanMeng | Others | ut |
| 208 | 1587 | Keep track on the latest CUDA op impl | P2 | UT issue with few failures | No specific action identified - needs investigation | CuiYifeng, yucai-intel | Others | ut |
| 209 | 1574 | The operator 'aten::_grouped_mm' is not currently implemented... | P2 | UT issue with few failures | No specific action identified - needs investigation | Stonepia, LuFinch | Others | ut |
| 210 | 1324 | [Win] UR Error when OOM and break the tensor context | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | Stonepia | Others | ut |
| 211 | 1171 | LNL Windows got unexpected error message | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | daisyden | Others | ut |
| 212 | 1059 | SYCL RT: Using recommended shortcut API for kernel specific... | P2 | UT issue with few failures | No specific action identified - needs investigation | CuiYifeng, jianyizh | Others | ut |
| 213 | 208 | Abstract utility functions used in ATen operator implementation. | P2 | UT issue with few failures | No specific action identified - needs investigation | CuiYifeng | Others | ut |
| 214 | 146 | Evaluate register spill in SYCL kernel | P2 | UT issue with few failures | No specific action identified - needs investigation | CuiYifeng, jianyizh, mengfei25 | Others | ut |

### <span id='dtype---precision-related'>Dtype / Precision Related</span> (40 issues)

| # | ID | Title | Priority | Priority Reason | Action Reason | Owner Transfer | Category | Test Module |
|---|------|------|----------|--------------------------------------------|----------|----------|------------|------------|
| 1 | 3296 | accuracy gap of stft in float16 | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | None | Dtype / Precision Related | ut |
| 2 | 3290 | huggingface amp_fp16 inference accuracy openai/whisper-tiny... | P1 | E2E benchmark accuracy issue | E2E accuracy issue pending - needs upstream investigation | jianyizh, weishi-deng | Dtype / Precision Related | e2e |
| 3 | 3238 | The supported dtypes of _refs.stft is not aligned to stft | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | daisyden | Dtype / Precision Related | ut |
| 4 | 3177 | Accuracy gap of BF16/FP16 test_block_addmm | P2 | UT issue with few failures | Bug/Perf issue awaiting reporter response | CuiYifeng | Dtype / Precision Related | ut |
| 5 | 3163 | [Bug Skip]: Object comparison failed: torch.int64 !=... | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | chunhuanMeng | Dtype / Precision Related | ut |
| 6 | 3161 | Exception: Tensor-likes are not close! -... | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | kdrozd-dev | Dtype / Precision Related | ut |
| 7 | 3137 | [upstream_ut] RuntimeError: expected scalar type Half but... | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | LuFinch | Dtype / Precision Related | ut |
| 8 | 3103 | Tensor-likes are not equal for test_backward_nn_functional_con... | P2 | UT issue with few failures | No specific action identified - needs investigation | BBBela | Dtype / Precision Related | ut |
| 9 | 3084 | torch.library.register_autocast does not support xpu | P2 | UT issue with few failures | No specific action identified - needs investigation | None | Dtype / Precision Related | ut |
| 10 | 3058 | [E2E] hf_GPT2_large amp_fp16/amp_bf16 training got fail_accuracy | P1 | E2E accuracy/functionality issue | E2E accuracy issue pending - needs upstream investigation | weishi-deng | Dtype / Precision Related | e2e |
| 11 | 2984 | [release/2.11] sebotnet33ts_256 fp32 training got fail_accuracy | P1 | E2E accuracy/functionality issue | E2E accuracy issue pending - needs upstream investigation | jianyizh, weishi-deng | Dtype / Precision Related | e2e |
| 12 | 2960 | [release/2.11] timm_models_xcit_large_24_p8_224_float16_traini... | P0 | Impacts customer custom model/application | Bug/Perf issue pending reporter response | shangerxin | Dtype / Precision Related | ut |
| 13 | 2952 | [release/2.11][wsl] timm_models_accuracy_training_bfloat16... | P0 | Impacts customer custom model/application | Bug/Perf issue pending reporter response | bjarzemb | Dtype / Precision Related | ut |
| 14 | 2928 | [release/2.11] pyhpc_turbulent_kinetic_energy fp32 inference... | P1 | E2E accuracy/functionality issue | E2E accuracy issue pending - needs upstream investigation | jianyizh | Dtype / Precision Related | e2e |
| 15 | 2908 | [release/2.11] Model fail_accuracy for 5 testcases | P1 | E2E custom model accuracy/functionality issue | E2E accuracy issue pending - needs upstream investigation | xuhancn | Dtype / Precision Related | e2e |
| 16 | 2862 | accuracy issue with test_float8_scale_fast_accum_xpu | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | daisyden | Dtype / Precision Related | ut |
| 17 | 2840 | Accuracy issue with 64 bit indexing depthwise_conv | P2 | UT issue with few failures | Bug/Perf issue awaiting reporter response | kdrozd-dev | Dtype / Precision Related | ut |
| 18 | 2837 | Accuracy issue for Muon optimizer | P2 | UT issue with few failures | Bug/Perf issue awaiting reporter response | kdrozd-dev | Dtype / Precision Related | ut |
| 19 | 2779 | Accuracy failures in logspace op | P2 | UT issue with few failures | Bug/Perf issue awaiting reporter response | PawelSwider2000 | Dtype / Precision Related | ut |
| 20 | 2680 | XPU Autocast does not support fp32 dtypes | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | kaixuanliu | Dtype / Precision Related | ut |
| 21 | 2656 | [release/2.10] models got fail_accuracy on BMG WSL2 | P0 | Impacts customer custom model/application | Bug/Perf issue pending reporter response | libohao1201 | Dtype / Precision Related | ut |
| 22 | 2655 | [BMG][OOB] hf_Reformer performance drop | P0 | Regression - passed before but failed now | Bug/Perf issue pending reporter response | jianyizh | Dtype / Precision Related | e2e |
| 23 | 2654 | [BMG][OOB] t5 inference performance drop 2 | P0 | Regression - passed before but failed now | Bug/Perf issue pending reporter response | jianyizh | Dtype / Precision Related | e2e |
| 24 | 2640 | random issue test_vjpvjp_index_reduce_prod_xpu_float32 | P2 | UT issue with few failures | No specific action identified - needs investigation | wpietka | Dtype / Precision Related | ut |
| 25 | 2615 | [Bug Skip]: New failures RuntimeError: Unsupported dtype Half... | P2 | UT issue with few failures | Bug/Perf issue awaiting reporter response | kaileiyx | Dtype / Precision Related | ut |
| 26 | 2592 | [release/2.10] models got fail_accuracy | P0 | Impacts customer custom model/application | E2E accuracy issue pending - needs upstream investigation | mengfei25 | Dtype / Precision Related | e2e |
| 27 | 2482 | test_dtypes issue introduced by pytorch test sample input updates | P2 | UT issue with few failures | Bug/Perf issue awaiting reporter response | daisyden | Dtype / Precision Related | ut |
| 28 | 2439 | [oneDNN] TestDecompXPU.test_quick_addmv_xpu_float64 got fail... | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | mengfei25 | Dtype / Precision Related | ut |
| 29 | 2257 | Accuracy failures in test/xpu/test_unary_ufuncs_xpu.py | P2 | UT issue with few failures | Bug/Perf issue awaiting reporter response | zxd1997066 | Dtype / Precision Related | ut |
| 30 | 2251 | [upstream_ut] test_fake_autocase got Exception: Dtypes... | P2 | UT issue with few failures | Issue is upstream - needs skip PR upstream | astachowiczhabana | Dtype / Precision Related | ut |
| 31 | 2219 | float8_e4m3fn precision overflow | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | jiqing-feng | Dtype / Precision Related | ut |
| 32 | 2182 | test_transform_bias_rescale_qkv_nested_xpu_float32 failed... | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | wincent8 | Dtype / Precision Related | ut |
| 33 | 2128 | [2.9][BMG-Windows][Torchbench] speeach_transforer... | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | libohao1201 | Dtype / Precision Related | ut |
| 34 | 1877 | Torchbench model squeezenet1_1 and functorch_dp_cifar10 got... | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | libohao1201 | Dtype / Precision Related | ut |
| 35 | 1866 | [release 2.8]Torchbench vision_maskrcnn (amp_b16/amp_fp16... | P1 | E2E accuracy/functionality issue | E2E accuracy issue pending - needs upstream investigation | BartoszKokoszko | Dtype / Precision Related | e2e |
| 36 | 1818 | [BMG-Windows][PT2.8]Torch-xpu-ops UT got accuracy issue | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | libohao1201 | Dtype / Precision Related | ut |
| 37 | 1778 | [Infra] Show known issues for accuracy test | P1 | E2E accuracy/functionality issue | E2E accuracy issue pending - needs upstream investigation | mengfei25 | Dtype / Precision Related | e2e |
| 38 | 1159 | [LNL Windows][Test by CD Nightly Wheels] hugging face model -... | P2 | E2E benchmark model issue | Bug/Perf issue pending reporter response | libohao1201 | Dtype / Precision Related | e2e |
| 39 | 492 | Timm_efficientdet NotImplementedError: The original model... | P0 | Impacts customer custom model/application | Bug/Perf issue pending reporter response | mengfei25 | Dtype / Precision Related | e2e |
| 40 | 489 | Moco NotImplementedError: xpu not supported | P2 | UT issue with few failures | Bug/Perf issue pending reporter response | mengfei25 | Dtype / Precision Related | e2e |


---

## <span id='4-last-week-issues'>4. Last Week Issues</span>

**Issues reported in last 7 days: 5**

| # | ID | Title | Priority | Action Reason | Category | Created Time |
|---|------|------|----------|--------------------------------------------|----------|--------------|
| 1 | 3306 | [distributed] no attribute '_reset_fr_recording_xccl' in... | N | No specific action identified - needs investigation | Distributed | 2026-04-10 |
| 2 | 3305 | [distributed] shrink operation support in... | N | No specific action identified - needs investigation | Distributed | 2026-04-10 |
| 3 | 3300 | [CI] When creating PR, several pull workflows are launched... | N | No specific action identified - needs investigation | Others | 2026-04-10 |
| 4 | 3296 | accuracy gap of stft in float16 | 5 | Issue is upstream - needs skip PR upstream | Dtype / Precision Related | 2026-04-10 |
| 5 | 3290 | huggingface amp_fp16 inference accuracy openai/whisper-tiny... | 7 | E2E accuracy issue pending - needs upstream investigation | Dtype / Precision Related | 2026-04-09 |

## <span id='4-stale-issues'>4. Stale Issues - No Update 2+ Weeks</span>

**Issues without update for 2+ weeks (excluding closed): 229**

| # | ID | Title | Priority | Action Reason | Category | Updated Time | Days Since Update |
|---|------|------|----------|--------------------------------------------|----------|---------------|----------------|
| 1 | 1729 | Validation Check List | N | No specific action identified - needs investigation | Others | 2025-06-11 | 308 |
| 2 | 2199 | Fix reduction and norm register spill | N | No specific action identified - needs investigation | Others | 2025-10-22 | 175 |
| 3 | 2163 | 3 distributed UT cases need to be supported by -... | N | No specific action identified - needs investigation | Distributed | 2025-11-18 | 148 |
| 4 | 1762 | Add an ocloc AOT target compilation test in cmake | N | No specific action identified - needs investigation | PT2E | 2025-11-27 | 139 |
| 5 | 2349 | [oneAPI][backward compatibility] libur_loader.so.0: version... | N | No specific action identified - needs investigation | Others | 2025-12-10 | 126 |
| 6 | 1678 | missing op support for `model.share_memory()` | N | No specific action identified - needs investigation | Others | 2025-12-12 | 124 |
| 7 | 2165 | [distributed] test_device_mesh.py::TestDeviceMeshGetItem::test... | 9 | Bug/Perf issue pending reporter response | Distributed | 2025-12-17 | 119 |
| 8 | 2340 | [distributed][_tools] AssertionError: Roofline estimation... | 9 | Bug/Perf issue pending reporter response | Distributed | 2025-12-17 | 119 |
| 9 | 2113 | Update example for Distributed Data Parallel | N | No specific action identified - needs investigation | Distributed | 2025-12-17 | 119 |
| 10 | 2609 | [upstream_ut] torch._inductor.exc.InductorError:... | 5 | Issue is upstream - needs skip PR upstream | Inductor / Compilation Related | 2025-12-29 | 107 |
| 11 | 2620 | [upstream_ut] AssertionError: dtype is needed to compute eps1... | 5 | Issue is upstream - needs skip PR upstream | Inductor / Compilation Related | 2025-12-29 | 107 |
| 12 | 2611 | [upstream_ut] AssertionError: Tensor-likes are not equal! in... | 5 | Issue is upstream - needs skip PR upstream | Inductor / Compilation Related | 2025-12-29 | 107 |
| 13 | 2613 | [upstream_ut] AssertionError: Tensor-likes are not equal! in... | 5 | Issue is upstream - needs skip PR upstream | Inductor / Compilation Related | 2025-12-29 | 107 |
| 14 | 2004 | [distributed][shared_tensor] test\distributed\_shard\shared_te... | 9 | Bug/Perf issue pending reporter response | Distributed | 2026-01-05 | 100 |
| 15 | 2659 | [distributed] test_dist2.py RuntimeError: Backend xccl does... | 9 | Bug/Perf issue pending reporter response | Distributed | 2026-01-06 | 99 |
| 16 | 2697 | Title: [upstream_ut] RuntimeError: Expected to find ", 0, "... | 5 | Issue is upstream - needs skip PR upstream | Others | 2026-01-07 | 98 |
| 17 | 2693 | Title: [upstream_ut] AssertionError: Scalars are not equal! | 5 | Issue is upstream - needs skip PR upstream | Inductor / Compilation Related | 2026-01-09 | 96 |
| 18 | 2737 | [distributed] AttributeError: module 'torch._C' has no... | 9 | Bug/Perf issue pending reporter response | Distributed | 2026-01-13 | 92 |
| 19 | 2742 | [Linux][PT2E] hf_Roberta_base model performance ASYMM and... | 9 | Bug/Perf issue pending reporter response | PT2E | 2026-01-15 | 90 |
| 20 | 1689 | [For op Perf Comparison] Save reference comparison run id | N | No specific action identified - needs investigation | Others | 2026-01-16 | 89 |
| 21 | 2142 | XPU max_memory_allocated have different output with CUDA | N | No specific action identified - needs investigation | Others | 2026-01-28 | 77 |
| 22 | 2200 | support flash attention op on XPU device | N | No specific action identified - needs investigation | Flash Attention / Transformer Related | 2026-01-28 | 77 |
| 23 | 2232 | sdpa backward kernel is required to reduce memory usage | N | No specific action identified - needs investigation | Flash Attention / Transformer Related | 2026-01-28 | 77 |
| 24 | 2270 | Backend Compatibility Error in test/xpu/test_decomp.py | 9 | Bug/Perf issue pending reporter response | Others | 2026-01-28 | 77 |
| 25 | 2783 | [Bug Skip]: Key "xpu" is missing from dict "driver" in test_svd | N | Bug/Perf issue awaiting reporter response | Others | 2026-01-28 | 77 |
| 26 | 2694 | Title: [upstream_ut] AssertionError: Tensor-likes are not... | 5 | Issue is upstream - needs skip PR upstream | Inductor / Compilation Related | 2026-01-29 | 76 |
| 27 | 2531 | [upstream_ut] AssertionError: Torch not compiled with CUDA... | N | Bug/Perf issue awaiting reporter response | Others | 2026-02-04 | 70 |
| 28 | 2326 | [TorchAO] MX training native PyTorch on XPU | N | No specific action identified - needs investigation | TorchAO | 2026-02-19 | 55 |
| 29 | 2325 | [TorchAO] Float8 training support on XPU | N | No specific action identified - needs investigation | TorchAO | 2026-02-19 | 55 |
| 30 | 2240 | RuntimeError: Trying to set a forward gradient that has a... | N | Bug/Perf issue awaiting reporter response | Others | 2026-02-25 | 49 |
| 31 | 2948 | [AO] Benchmark enabling on XPU | N | No specific action identified - needs investigation | Others | 2026-02-27 | 47 |
| 32 | 2950 | SYCL compilation flag -fsycl-id-queries-fit-in-int does not... | N | No specific action identified - needs investigation | Others | 2026-02-27 | 47 |
| 33 | 2935 | [release/2.11][inductor] huggingface amp_fp16 and float16... | 9 | Bug/Perf issue pending reporter response | Inductor / Compilation Related | 2026-02-27 | 47 |
| 34 | 2939 | [release/2.11] gmlp_s16_224 inference amp performance dropped... | 9 | Bug/Perf issue pending reporter response | Others | 2026-02-28 | 46 |
| 35 | 1551 | [distributed] NotImplementedError: The operator... | 9 | Bug/Perf issue pending reporter response | Distributed | 2026-02-28 | 46 |
| 36 | 1547 | [distributed] NotImplementedError: The operator... | 9 | Bug/Perf issue pending reporter response | Distributed | 2026-02-28 | 46 |
| 37 | 1548 | [distributed] AssertionError: 'fused_all_gather_matmul' not... | 9 | Bug/Perf issue pending reporter response | Distributed | 2026-02-28 | 46 |
| 38 | 1549 | [distributed] AssertionError: 'fused_all_gather_scaled_matmul'... | 9 | Bug/Perf issue pending reporter response | Distributed | 2026-02-28 | 46 |
| 39 | 1661 | [distributed] Accuracy gap in _composable/fsdp on Xelink | 9 | Bug/Perf issue pending reporter response | Distributed | 2026-02-28 | 46 |
| 40 | 1727 | [distributed] AttributeError: module 'torch.xpu' has no... | 9 | Bug/Perf issue pending reporter response | Distributed | 2026-02-28 | 46 |
| 41 | 1556 | [distributed] NotImplementedError: Operator... | 9 | Bug/Perf issue pending reporter response | Distributed | 2026-02-28 | 46 |
| 42 | 2463 | [Bug Skip]: OSError: SYCL runtime is not dected. | 9 | Bug/Perf issue pending reporter response | Others | 2026-03-02 | 44 |
| 43 | 1571 | [distributed] ValueError: Cannot use ReduceOp.PREMUL_SUM with... | 9 | Bug/Perf issue pending reporter response | Distributed | 2026-03-02 | 44 |
| 44 | 2261 | [xpu][profiler] Run with fork process has extra warning | N | No specific action identified - needs investigation | Others | 2026-03-02 | 44 |
| 45 | 1856 | channel last aten::hardswish_ will call extra copy | N | No specific action identified - needs investigation | Others | 2026-03-02 | 44 |
| 46 | 2248 | [upstream_ut] test_cow failures | 5 | Issue is upstream - needs skip PR upstream | Others | 2026-03-02 | 44 |
| 47 | 2250 | Found mismatch when comparing the output of aten.view.default... | N | No specific action identified - needs investigation | Others | 2026-03-02 | 44 |
| 48 | 2358 | test/test_view_ops.py::TestOldViewOpsXPU::test_ravel_xpu meet... | 5 | Issue is upstream - needs skip PR upstream | TorchAO | 2026-03-02 | 44 |
| 49 | 2285 | Support efficient attention | N | No specific action identified - needs investigation | Others | 2026-03-02 | 44 |
| 50 | 2482 | test_dtypes issue introduced by pytorch test sample input updates | N | Bug/Perf issue awaiting reporter response | Dtype / Precision Related | 2026-03-02 | 44 |
| 51 | 2400 | [ut_upstream] tf32_on_and_off() need xpu support | N | No specific action identified - needs investigation | Others | 2026-03-02 | 44 |
| 52 | 2253 | the supported dtypes are not align with cuda | 5 | Issue is upstream - needs skip PR upstream | Others | 2026-03-02 | 44 |
| 53 | 2255 | [upstream_ut] RuntimeError: Long is not supported in oneDNN | 5 | Issue is upstream - needs skip PR upstream | Others | 2026-03-02 | 44 |
| 54 | 2425 | [upstream_ut] RuntimeError: Expected both self and other to... | N | Bug/Perf issue awaiting reporter response | Others | 2026-03-02 | 44 |
| 55 | 2015 | inf is returned by nn.TransformerEncoderLayer | N | No specific action identified - needs investigation | Others | 2026-03-02 | 44 |
| 56 | 2301 | [upstream_ut] dtypes not align with OpInfo | 5 | Issue is upstream - needs skip PR upstream | Others | 2026-03-02 | 44 |
| 57 | 1555 | [distributed] RuntimeError: aten.add.Tensor: got mixed... | 9 | Bug/Perf issue pending reporter response | Distributed | 2026-03-02 | 44 |
| 58 | 2329 | [upstream_ut] feature missing: get_device_tflops and... | 5 | Issue is upstream - needs skip PR upstream | Inductor / Compilation Related | 2026-03-04 | 42 |
| 59 | 2979 | eca_halonext26ts got RuntimeError:... | 9 | Bug/Perf issue pending reporter response | Others | 2026-03-04 | 42 |
| 60 | 2412 | Some NestedTensor missing XPU support | N | No specific action identified - needs investigation | Others | 2026-03-04 | 42 |
| 61 | 2390 | SDPA in pytorch use different backend compared with ipex | N | No specific action identified - needs investigation | Flash Attention / Transformer Related | 2026-03-04 | 42 |
| 62 | 2331 | [upstream_ut] AssertionError: Scalars are not equal! with... | 5 | Issue is upstream - needs skip PR upstream | Others | 2026-03-04 | 42 |
| 63 | 2722 | [Bug Skip]: NotImplementedError: Could not run 'aten::flip'... | N | Bug/Perf issue awaiting reporter response | TorchAO | 2026-03-04 | 42 |
| 64 | 2491 | [upstream_ut] AssertionError: False is not true | 9 | Bug/Perf issue pending reporter response | Others | 2026-03-05 | 41 |
| 65 | 2907 | [release/2.11] Models performance regression for 5 testcases | 9 | Bug/Perf issue pending reporter response | Others | 2026-03-06 | 40 |
| 66 | 2392 | [Bug Skip]: torch.OutOfMemoryError: XPU out of memory | 9 | Bug/Perf issue pending reporter response | Others | 2026-03-06 | 40 |
| 67 | 2239 | Exception: could not create a primitive descriptor for the... | N | Bug/Perf issue awaiting reporter response | Others | 2026-03-06 | 40 |
| 68 | 2997 | AssertionError of test_linear_and_cel_max_autotune | 5 | Issue is upstream - needs skip PR upstream | Inductor / Compilation Related | 2026-03-06 | 40 |
| 69 | 3004 | TypeError: _xpu_recordMemoryHistory(): incompatible function... | 5 | Issue is upstream - needs skip PR upstream | Others | 2026-03-09 | 37 |
| 70 | 3013 | [upstream_ut] RuntimeError: Kernel is incompatible with all... | 9 | Bug/Perf issue pending reporter response | Others | 2026-03-09 | 37 |
| 71 | 3014 | [upstream_ut] AssertionError: False is not true | 9 | Bug/Perf issue pending reporter response | Others | 2026-03-09 | 37 |
| 72 | 3011 | [upstream_ut] torch.OutOfMemoryError: XPU out of memory | 9 | Bug/Perf issue pending reporter response | Others | 2026-03-09 | 37 |
| 73 | 2541 | Title: [upstream_ut] RuntimeError: could not construct a... | 9 | Bug/Perf issue pending reporter response | Others | 2026-03-09 | 37 |
| 74 | 2823 | [TorchAO][BMG] Llama-3.2-1B-Instruct Dynamic INT8 got 20%... | 9 | Bug/Perf issue pending reporter response | TorchAO | 2026-03-10 | 36 |
| 75 | 2729 | [Bug Skip]: Random failures 2026WW03 | 9 | Bug/Perf issue pending reporter response | Others | 2026-03-10 | 36 |
| 76 | 2777 | [Bug Skip]: Random failures 2026WW05 | 9 | Bug/Perf issue pending reporter response | Others | 2026-03-10 | 36 |
| 77 | 2767 | [UT] test_control_flow_xpu.py got AssertionError | 9 | Bug/Perf issue pending reporter response | Others | 2026-03-11 | 35 |
| 78 | 3032 | [TorchAO][UT] failures in test/prototype/safetensors/test_safe... | 9 | Bug/Perf issue pending reporter response | TorchAO | 2026-03-11 | 35 |
| 79 | 2229 | test/test_sparse_csr.py::TestSparseCompressedCPU::test_invalid... | N | No specific action identified - needs investigation | Sparse Operations Related | 2026-03-12 | 34 |
| 80 | 2530 | Title: [upstream_ut] AssertionError: RuntimeError not raised | N | Bug/Perf issue awaiting reporter response | Others | 2026-03-13 | 33 |
| 81 | 2640 | random issue test_vjpvjp_index_reduce_prod_xpu_float32 | N | No specific action identified - needs investigation | Dtype / Precision Related | 2026-03-16 | 30 |
| 82 | 2702 | [distributed] RuntimeError: Work ran time out after 0... | 9 | Bug/Perf issue pending reporter response | Distributed | 2026-03-16 | 30 |
| 83 | 2686 | [distributed] Accuracy issues with test_distributed_spawn.py | 9 | Bug/Perf issue pending reporter response | Distributed | 2026-03-16 | 30 |
| 84 | 1749 | transformers UT failure in XPU because SDPA check error... | 9 | Bug/Perf issue pending reporter response | Flash Attention / Transformer Related | 2026-03-17 | 29 |
| 85 | 3060 | Implement torch._scaled_grouped_mm for xpu backend | N | No specific action identified - needs investigation | Others | 2026-03-17 | 29 |
| 86 | 3033 | [Bug Skip]: Softmax tolerance | N | Bug/Perf issue awaiting reporter response | Others | 2026-03-17 | 29 |
| 87 | 2186 | AssertionError: Mul tiheadAttention does not support... | N | Bug/Perf issue awaiting reporter response | Others | 2026-03-17 | 29 |
| 88 | 3000 | [Bug Skip]: RuntimeError: _share_fd_: only available on CPU... | N | Bug/Perf issue awaiting reporter response | Others | 2026-03-17 | 29 |
| 89 | 2439 | [oneDNN] TestDecompXPU.test_quick_addmv_xpu_float64 got fail... | 9 | Bug/Perf issue pending reporter response | Dtype / Precision Related | 2026-03-18 | 28 |
| 90 | 2086 | nd_item::barrier has been deprecated | N | No specific action identified - needs investigation | Others | 2026-03-18 | 28 |
| 91 | 2215 | Find use case example for torch-xpu-ops.lib in sycl cpp extension | N | No specific action identified - needs investigation | Others | 2026-03-18 | 28 |
| 92 | 2098 | Upstream XPU functions in yaml | N | No specific action identified - needs investigation | Others | 2026-03-18 | 28 |
| 93 | 2128 | [2.9][BMG-Windows][Torchbench] speeach_transforer... | 9 | Bug/Perf issue pending reporter response | Dtype / Precision Related | 2026-03-18 | 28 |
| 94 | 2560 | [UT] "RuntimeError: iter.device(arg).is_xpu()" in... | 9 | Bug/Perf issue pending reporter response | Others | 2026-03-18 | 28 |
| 95 | 3076 | [TorchAO][BMG] Llama-3.2-1B-Instruct Dynamic INT8 got 10%... | 9 | Bug/Perf issue pending reporter response | TorchAO | 2026-03-18 | 28 |
| 96 | 1902 | implement torch.linalg.pinv xpu backend | N | No specific action identified - needs investigation | Others | 2026-03-18 | 28 |
| 97 | 2376 | [Bug Skip]: NotImplementedError: "logaddexp_xpu" not... | N | Bug/Perf issue awaiting reporter response | Others | 2026-03-19 | 27 |
| 98 | 2324 | [TorchAO] FP8 conv support | N | No specific action identified - needs investigation | TorchAO | 2026-03-19 | 27 |
| 99 | 2244 | test/test_sparse_csr.py::TestSparseCSRXPU::test_block_addmm... | 9 | Bug/Perf issue pending reporter response | Sparse Operations Related | 2026-03-19 | 27 |
| 100 | 2214 | test/test_sparse.py::TestSparseAnyXPU::test_gradcheck_mm... | 5 | Issue is upstream - needs skip PR upstream | Sparse Operations Related | 2026-03-19 | 27 |
| 101 | 2309 | unsupported ops with PYTORCH_ENABLE_XPU_FALLBACK unset | 1 | Issue marked as not_target/wontfix - should be skipped for XPU enablement | Others | 2026-03-19 | 27 |
| 102 | 2512 | [upstream_ut] RuntimeError: _histc_xpu does not have a... | 9 | Bug/Perf issue pending reporter response | Others | 2026-03-19 | 27 |
| 103 | 3030 | [Bug Skip] test/test_modules.py::TestModuleXPU::test_cpu_gpu_p... | 9 | Bug/Perf issue pending reporter response | Others | 2026-03-19 | 27 |
| 104 | 3089 | AssertionError: Torch not compiled with CUDA enabled | N | Bug/Perf issue awaiting reporter response | Inductor / Compilation Related | 2026-03-19 | 27 |
| 105 | 3086 | nvml support blocks some test cases | N | No specific action identified - needs investigation | Others | 2026-03-20 | 26 |
| 106 | 3084 | torch.library.register_autocast does not support xpu | N | No specific action identified - needs investigation | Dtype / Precision Related | 2026-03-20 | 26 |
| 107 | 3082 | multithread support in distributed | N | No specific action identified - needs investigation | Distributed | 2026-03-20 | 26 |
| 108 | 3080 | cudagraph tests blocked by feature gap | N | No specific action identified - needs investigation | Others | 2026-03-20 | 26 |
| 109 | 3100 | [distributed] /handler/dump_nccl_trace_pickle and nccl_log... | N | No specific action identified - needs investigation | Distributed | 2026-03-20 | 26 |
| 110 | 3101 | [distributed] 'torch._C._distributed_c10d.ProcessGroupXCCL'... | 9 | Bug/Perf issue pending reporter response | Distributed | 2026-03-20 | 26 |
| 111 | 3102 | [distributed] RuntimeError: Invalid device string: 'xpu:foo'... | 9 | Bug/Perf issue pending reporter response | Others | 2026-03-20 | 26 |
| 112 | 2471 | test_cuda.py gaps | N | No specific action identified - needs investigation | Others | 2026-03-20 | 26 |
| 113 | 2816 | torch.logcumsumexp incorrectly returns NaNs for complex64 input | N | No specific action identified - needs investigation | Others | 2026-03-20 | 26 |
| 114 | 2817 | Expected error message is different than actual | N | Bug/Perf issue awaiting reporter response | Others | 2026-03-20 | 26 |
| 115 | 2815 | RuntimeError: output with shape [2] doesn't match the... | N | Bug/Perf issue awaiting reporter response | Others | 2026-03-20 | 26 |
| 116 | 3077 | [Bug Skip] test_dlpack.py::TestTorchDlPackXPU::test_no_copy_xp... | 5 | Issue is upstream - needs skip PR upstream | Others | 2026-03-20 | 26 |
| 117 | 3074 | [Bug Skip] test_dlpack_exchange_api expect... | 9 | Bug/Perf issue pending reporter response | Others | 2026-03-20 | 26 |
| 118 | 1951 | Functionality issues in TestCommon.test_out. | 5 | Issue is upstream - needs skip PR upstream | Others | 2026-03-20 | 26 |
| 119 | 2251 | [upstream_ut] test_fake_autocase got Exception: Dtypes... | 5 | Issue is upstream - needs skip PR upstream | Dtype / Precision Related | 2026-03-20 | 26 |
| 120 | 2518 | [upstream_ut] TypeError: Creating a Tensor subclass from a... | 2 | All test cases passed on both XPU and stock - issue is resolved | Others | 2026-03-20 | 26 |
| 121 | 2496 | [upstream_ut] Segmentation fault when running... | 2 | All test cases passed on both XPU and stock - issue is resolved | Others | 2026-03-20 | 26 |
| 122 | 2533 | Title: [upstream_ut] AttributeError: 'TestQuantizedOpsXPU'... | 9 | Bug/Perf issue pending reporter response | TorchAO | 2026-03-20 | 26 |
| 123 | 2993 | [Bug Skip]: Unexpected success of... | 9 | Bug/Perf issue pending reporter response | Others | 2026-03-20 | 26 |
| 124 | 2513 | [upstream_ut] RuntimeError: _share_fd_: only available on CPU | 9 | Bug/Perf issue pending reporter response | Others | 2026-03-20 | 26 |
| 125 | 2630 | Title: [upstream_ut] AssertionError: Scalars are not equal! | 9 | Bug/Perf issue pending reporter response | Others | 2026-03-20 | 26 |
| 126 | 2873 | [Bug Skip]: test_repos.py contains several failed ops | 9 | Bug/Perf issue pending reporter response | Others | 2026-03-20 | 26 |
| 127 | 2712 | [upstream_ut] RuntimeError: Cannot swap t2 because it has... | 5 | Issue is upstream - needs skip PR upstream | Others | 2026-03-20 | 26 |
| 128 | 3041 | AssertionError: Expected len(flat_diff_results) > 0 in... | 5 | Issue is upstream - needs skip PR upstream | Others | 2026-03-20 | 26 |
| 129 | 3025 | New failing test in Nightly Wheel... | N | Bug/Perf issue awaiting reporter response | Others | 2026-03-21 | 25 |
| 130 | 2022 | [Windows] [CI] [UT] AssertionError: Tensor-likes are not close! | 9 | Bug/Perf issue pending reporter response | Others | 2026-03-23 | 23 |
| 131 | 2779 | Accuracy failures in logspace op | N | Bug/Perf issue awaiting reporter response | Dtype / Precision Related | 2026-03-23 | 23 |
| 132 | 2246 | torch/sparse/_triton_ops*.py need to be ported to enable for... | N | No specific action identified - needs investigation | Sparse Operations Related | 2026-03-23 | 23 |
| 133 | 2235 | test/test_sparse_csr.py::TestSparseCompressedTritonKernelsXPU:... | N | No specific action identified - needs investigation | Sparse Operations Related | 2026-03-23 | 23 |
| 134 | 2230 | test_sparse_csr.py::TestSparseCompressedTritonKernelsXPU::test... | N | Bug/Perf issue awaiting reporter response | Sparse Operations Related | 2026-03-23 | 23 |
| 135 | 2220 | test/test_sparse_csr.py::TestSparseCompressedTritonKernelsXPU:... | N | No specific action identified - needs investigation | Sparse Operations Related | 2026-03-23 | 23 |
| 136 | 2663 | test_sparse_semi_structured.py gaps | 5 | Issue is upstream - needs skip PR upstream | Sparse Operations Related | 2026-03-23 | 23 |
| 137 | 3121 | [Bug Skip]: CUDA specific UT test_fft_half_and_chalf_not_power... | N | Bug/Perf issue awaiting reporter response | Others | 2026-03-24 | 22 |
| 138 | 3124 | [TorchAO][Bug] ImportError: Requires mslk >= 1.0.0 when... | 9 | Bug/Perf issue pending reporter response | TorchAO | 2026-03-24 | 22 |
| 139 | 3131 | [upstream_ut] NotImplementedError: The operator... | 5 | Issue is upstream - needs skip PR upstream | Others | 2026-03-24 | 22 |
| 140 | 3141 | [upstream_ut] RuntimeError: FlashAttentionForwardXPU only... | 5 | Issue is upstream - needs skip PR upstream | Flash Attention / Transformer Related | 2026-03-24 | 22 |
| 141 | 3140 | [upstream_ut] RuntimeError: FlashAttentionForwardXPU does not... | 5 | Issue is upstream - needs skip PR upstream | Others | 2026-03-24 | 22 |
| 142 | 1970 | torch._dynamo.exc.BackendCompilerFailed: backend='inductor'... | 9 | Bug/Perf issue pending reporter response | Inductor / Compilation Related | 2026-03-24 | 22 |
| 143 | 2554 | [upstream_ut] AssertionError: AssertionError not raised | 5 | Issue is upstream - needs skip PR upstream | Inductor / Compilation Related | 2026-03-24 | 22 |
| 144 | 3150 | [Task] Align XPU kernel's implementation to stock PyTorch | N | No specific action identified - needs investigation | Others | 2026-03-24 | 22 |
| 145 | 3021 | [distributed] all_to_all_single Compatibility Issue on B60/XCCL | N | No specific action identified - needs investigation | Distributed | 2026-03-24 | 22 |
| 146 | 3022 | [distributed] batch_isend_irecv Compatibility Issue on B60/XCCL | N | No specific action identified - needs investigation | Distributed | 2026-03-24 | 22 |
| 147 | 2662 | [release/2.10][Windows][BMG] New failed test cases and 2.9... | 9 | Bug/Perf issue pending reporter response | Others | 2026-03-24 | 22 |
| 148 | 2245 | oneDNN matmul received incorrect shape in... | N | Bug/Perf issue awaiting reporter response | Sparse Operations Related | 2026-03-24 | 22 |
| 149 | 1963 | [upstream_ut] MetadataMismatchError in TestFakeTensor of... | 5 | Issue is upstream - needs skip PR upstream | Others | 2026-03-24 | 22 |
| 150 | 3166 | test_consistency_SparseCSR failures | 5 | Issue is upstream - needs skip PR upstream | Sparse Operations Related | 2026-03-24 | 22 |
| 151 | 3165 | test_sparse_csr_xpu.py::TestSparseCompressedTritonKernelsXPU::... | 5 | Issue is upstream - needs skip PR upstream | Others | 2026-03-24 | 22 |
| 152 | 2798 | Test case test/test_dlpack.py::TestTorchDlPackCPU::test_numpy_... | 5 | Issue is upstream - needs skip PR upstream | Others | 2026-03-25 | 21 |
| 153 | 3163 | [Bug Skip]: Object comparison failed: torch.int64 !=... | 5 | Issue is upstream - needs skip PR upstream | Dtype / Precision Related | 2026-03-25 | 21 |
| 154 | 2562 | Warning as Error | 9 | Bug/Perf issue pending reporter response | Others | 2026-03-25 | 21 |
| 155 | 2578 | [TorchAO][UT] test/quantization/test_quant_api.py... | 5 | Issue is upstream - needs skip PR upstream | TorchAO | 2026-03-25 | 21 |
| 156 | 2654 | [BMG][OOB] t5 inference performance drop 2 | 9 | Bug/Perf issue pending reporter response | Dtype / Precision Related | 2026-03-25 | 21 |
| 157 | 2912 | [release/2.11] UT extended 220 new failures | 9 | Bug/Perf issue pending reporter response | Others | 2026-03-25 | 21 |
| 158 | 2921 | [abs][complex64] - new failing test cases caused by PyTorch... | N | Bug/Perf issue awaiting reporter response | Others | 2026-03-25 | 21 |
| 159 | 2914 | Test case test/test_autograd.py::TestAutogradMultipleDispatchC... | 9 | Bug/Perf issue pending reporter response | Others | 2026-03-25 | 21 |
| 160 | 2888 | torch._inductor.exc.InductorError: AssertionError:... | 5 | Issue is upstream - needs skip PR upstream | Inductor / Compilation Related | 2026-03-25 | 21 |
| 161 | 2869 | [Bug Skip]: New UT failure in 0209 nightly windows. | N | Bug/Perf issue awaiting reporter response | Others | 2026-03-25 | 21 |
| 162 | 2852 | [Bug Skip]: New UT failures in 0206 nightly on Windows | 9 | Bug/Perf issue pending reporter response | Others | 2026-03-25 | 21 |
| 163 | 2845 | [Bug Skip]:[UT] [Windows] failed cases 2026-2-4 | 9 | Bug/Perf issue pending reporter response | Others | 2026-03-25 | 21 |
| 164 | 2759 | [Bug Skip]: New failed cases 2026-1-22 | N | Bug/Perf issue awaiting reporter response | Others | 2026-03-25 | 21 |
| 165 | 2580 | [TorchAO][UT] test/test_low_bit_optim.py AssertionError:... | 9 | Bug/Perf issue pending reporter response | TorchAO | 2026-03-25 | 21 |
| 166 | 2928 | [release/2.11] pyhpc_turbulent_kinetic_energy fp32 inference... | 7 | E2E accuracy issue pending - needs upstream investigation | Dtype / Precision Related | 2026-03-25 | 21 |
| 167 | 2327 | [TorchAO] benchmark enabling on XPU | N | No specific action identified - needs investigation | TorchAO | 2026-03-25 | 21 |
| 168 | 3175 | [Bug Skip]: ValueError: sampled_addmm(): all inputs are... | N | Bug/Perf issue awaiting reporter response | Others | 2026-03-25 | 21 |
| 169 | 3176 | [Bug Skip]: ValueError: _scaled_dot_product_attention(): all... | N | Bug/Perf issue awaiting reporter response | Flash Attention / Transformer Related | 2026-03-25 | 21 |
| 170 | 2164 | skip test_no_cuda_monkeypatch as it is cuda specific | 1 | Issue marked as not_target/wontfix - should be skipped for XPU enablement | Others | 2026-03-25 | 21 |
| 171 | 3170 | Unskip test_bmm_windows_error_xpu_float64 | 5 | Issue is upstream - needs skip PR upstream | Others | 2026-03-25 | 21 |
| 172 | 2891 | RuntimeError: Expected to find "(262144, 0, 512, 1" but did... | 5 | Issue is upstream - needs skip PR upstream | Others | 2026-03-25 | 21 |
| 173 | 3180 | [E2E] Timm/Torchbench models got "eager_two_runs_differ" on ARC | N | No specific action identified - needs investigation | Others | 2026-03-25 | 21 |
| 174 | 2999 | KeyError: 'eager_numerics.use_pytorch_libdevice' | 5 | Issue is upstream - needs skip PR upstream | Others | 2026-03-25 | 21 |
| 175 | 2537 | Title: [upstream_ut] Failed: Unexpected success | N | Bug/Perf issue awaiting reporter response | Others | 2026-03-25 | 21 |
| 176 | 2958 | AssertionError of test_dtensor_basic_compile | 5 | Issue is upstream - needs skip PR upstream | Inductor / Compilation Related | 2026-03-25 | 21 |
| 177 | 3169 | NotImplementedError: Could not run 'aten::hspmm' with... | 5 | Issue is upstream - needs skip PR upstream | Others | 2026-03-25 | 21 |
| 178 | 2806 | CompiledAOTI need XPU support | 5 | Issue is upstream - needs skip PR upstream | Inductor / Compilation Related | 2026-03-25 | 21 |
| 179 | 3184 | New failing UTs: test_cross_entropy_loss_2d_out_of_bounds_clas... | N | Bug/Perf issue awaiting reporter response | Others | 2026-03-25 | 21 |
| 180 | 3148 | [Triton] Huggingface openai/whisper-tiny got fail_accuracy | 7 | E2E accuracy issue pending - needs upstream investigation | Inductor / Compilation Related | 2026-03-26 | 20 |
| 181 | 2715 | [upstream_ut] torch._dynamo.exc.Unsupported: Attempted to... | 5 | Issue is upstream - needs skip PR upstream | PT2E | 2026-03-26 | 20 |
| 182 | 2006 | work-item/workgroup issue in softmax/unsampling/nonzero | N | No specific action identified - needs investigation | Others | 2026-03-26 | 20 |
| 183 | 1778 | [Infra] Show known issues for accuracy test | 7 | E2E accuracy issue pending - needs upstream investigation | Dtype / Precision Related | 2026-03-26 | 20 |
| 184 | 2639 | test_to() failed during rnn isinstance() check | N | Bug/Perf issue awaiting reporter response | Others | 2026-03-26 | 20 |
| 185 | 2592 | [release/2.10] models got fail_accuracy | 7 | E2E accuracy issue pending - needs upstream investigation | Dtype / Precision Related | 2026-03-27 | 19 |
| 186 | 3195 | test_sdpa_unbacked_no_dde_xpu crashed | 9 | Bug/Perf issue pending reporter response | Flash Attention / Transformer Related | 2026-03-27 | 19 |
| 187 | 2837 | Accuracy issue for Muon optimizer | N | Bug/Perf issue awaiting reporter response | Dtype / Precision Related | 2026-03-27 | 19 |
| 188 | 2359 | [upstream_ut] GradcheckError: Backward is not reentrant | 5 | Issue is upstream - needs skip PR upstream | Others | 2026-03-27 | 19 |
| 189 | 2404 | [distributed][checkpoint] AssertionError: Booleans mismatch:... | 9 | Bug/Perf issue pending reporter response | Distributed | 2026-03-27 | 19 |
| 190 | 2323 | [TorchAO] MOE training enabling on XPU | N | No specific action identified - needs investigation | TorchAO | 2026-03-29 | 17 |
| 191 | 2295 | [upstream_ut][xpu][test]nn/test_embedding.py::TestEmbeddingNND... | 5 | Issue is upstream - needs skip PR upstream | Others | 2026-03-30 | 16 |
| 192 | 3189 | Task Tracker | N | No specific action identified - needs investigation | Others | 2026-03-30 | 16 |
| 193 | 2287 | [upstream_ut] test_python_ref issues | 5 | Issue is upstream - needs skip PR upstream | Others | 2026-03-30 | 16 |
| 194 | 3187 | PyTorch XPU gpu_cpp_wrapper fails with InductorError... | 5 | Issue is upstream - needs skip PR upstream | Inductor / Compilation Related | 2026-03-30 | 16 |
| 195 | 3209 | [Win][Build] There is Cyclic dependencies error when build... | 9 | Bug/Perf issue pending reporter response | Others | 2026-03-30 | 16 |
| 196 | 2389 | [Bug Skip]: RuntimeError: Data corruption detected | 9 | Bug/Perf issue pending reporter response | Others | 2026-03-30 | 16 |
| 197 | 3191 | torch._inductor.exc.InductorError: AssertionError: both a... | 9 | Bug/Perf issue pending reporter response | Inductor / Compilation Related | 2026-03-30 | 16 |
| 198 | 3160 | compiler not found (Windows) | 2 | All test cases passed on both XPU and stock - issue is resolved | Others | 2026-03-30 | 16 |
| 199 | 3083 | [Bug Skip]: Random failures 2026WW12 | 9 | Bug/Perf issue pending reporter response | Others | 2026-03-30 | 16 |
| 200 | 2132 | [2.9][BMG-Windows][Torch-xpu-ops UT] 1 case failed with... | 9 | Bug/Perf issue pending reporter response | Inductor / Compilation Related | 2026-03-30 | 16 |
| 201 | 2701 | [distributed] Barrier Timeout Error with... | 9 | Bug/Perf issue pending reporter response | Distributed | 2026-03-30 | 16 |
| 202 | 2700 | [distributed] Hang issues with test_distributed_spawn.py | N | No specific action identified - needs investigation | Distributed | 2026-03-30 | 16 |
| 203 | 3139 | [distributed][_composable] AssertionError: Expects xpu:0 but... | 9 | Bug/Perf issue pending reporter response | Distributed | 2026-03-30 | 16 |
| 204 | 2972 | [distributed] AssertionError: ValueError not raised in... | 9 | Bug/Perf issue pending reporter response | Distributed | 2026-03-30 | 16 |
| 205 | 2969 | [distributed] AssertionError: Scalars are not equal! in... | 9 | Bug/Perf issue pending reporter response | Distributed | 2026-03-30 | 16 |
| 206 | 2968 | [distributed] timeout issue in test/distributed/test_c10d_xccl.py | 9 | Bug/Perf issue pending reporter response | Distributed | 2026-03-30 | 16 |
| 207 | 2738 | [distributed] test_c10d_spawn_nccl.py ValueError: input... | 9 | Bug/Perf issue pending reporter response | Distributed | 2026-03-30 | 16 |
| 208 | 2597 | [TorchAO][BMG] INT4 GPTQ shows worse performance compared... | 9 | Bug/Perf issue pending reporter response | TorchAO | 2026-03-31 | 15 |
| 209 | 3231 | Dynamo failed to run FX node with fake tensors: call_function... | 9 | Bug/Perf issue pending reporter response | PT2E | 2026-03-31 | 15 |
| 210 | 1818 | [BMG-Windows][PT2.8]Torch-xpu-ops UT got accuracy issue | 9 | Bug/Perf issue pending reporter response | Dtype / Precision Related | 2026-03-31 | 15 |
| 211 | 1866 | [release 2.8]Torchbench vision_maskrcnn (amp_b16/amp_fp16... | 7 | E2E accuracy issue pending - needs upstream investigation | Dtype / Precision Related | 2026-03-31 | 15 |
| 212 | 1877 | Torchbench model squeezenet1_1 and functorch_dp_cifar10 got... | 9 | Bug/Perf issue pending reporter response | Dtype / Precision Related | 2026-03-31 | 15 |
| 213 | 2766 | MaxPool2d - investigate memory layout performance | 9 | Bug/Perf issue pending reporter response | Others | 2026-03-31 | 15 |
| 214 | 2942 | [Windows] Unit tests got Fatal python error | 9 | Bug/Perf issue pending reporter response | Others | 2026-04-01 | 14 |
| 215 | 1324 | [Win] UR Error when OOM and break the tensor context | 9 | Bug/Perf issue pending reporter response | Others | 2026-04-01 | 14 |
| 216 | 1059 | SYCL RT: Using recommended shortcut API for kernel specific... | N | No specific action identified - needs investigation | Others | 2026-04-01 | 14 |
| 217 | 1649 | [cpp extension] Provide a clear error message when using... | 9 | Bug/Perf issue pending reporter response | Others | 2026-04-01 | 14 |
| 218 | 1986 | torch.xpu._sleep is missing, | N | No specific action identified - needs investigation | Others | 2026-04-01 | 14 |
| 219 | 2800 | AttributeError: 'torch._C._XpuDeviceProperties' object has no... | 5 | Issue is upstream - needs skip PR upstream | Others | 2026-04-01 | 14 |
| 220 | 3007 | AssertionError: Scalars are not equal! with... | 5 | Issue is upstream - needs skip PR upstream | Flash Attention / Transformer Related | 2026-04-01 | 14 |
| 221 | 1912 | Implement the torch.ops.aten._weight_int4pack_mm for... | N | No specific action identified - needs investigation | TorchAO | 2026-04-01 | 14 |
| 222 | 2966 | [Bug Skip]: [Regression]2026-3-2 ut failures | 9 | Bug/Perf issue pending reporter response | Others | 2026-04-01 | 14 |
| 223 | 2862 | accuracy issue with test_float8_scale_fast_accum_xpu | 9 | Bug/Perf issue pending reporter response | Dtype / Precision Related | 2026-04-01 | 14 |
| 224 | 2669 | [upstream_ut] AssertionError: Tensor-likes are not close! in... | N | Bug/Perf issue awaiting reporter response | Others | 2026-04-01 | 14 |
| 225 | 3106 | Worker crashes when running TestDecompXPU,test_quick_core_back... | 9 | Bug/Perf issue pending reporter response | Others | 2026-04-01 | 14 |
| 226 | 1973 | AssertionError: Scalars or Tensor-likes are not equal or close! | 9 | Bug/Perf issue pending reporter response | Others | 2026-04-01 | 14 |
| 227 | 2529 | [upstream_ut] AssertionError: False is not true | 9 | Bug/Perf issue pending reporter response | Others | 2026-04-01 | 14 |
| 228 | 3196 | vitals is not supported, the cases should be disabled | N | No specific action identified - needs investigation | Others | 2026-04-01 | 14 |
| 229 | 2536 | Title: [upstream_ut] AttributeError: module 'torch._C' has no... | N | Bug/Perf issue awaiting reporter response | Others | 2026-04-01 | 14 |

## <span id='6-dependency-issues'>6. Dependency Issues</span>

**Issues with dependencies: 384**

| # | ID | Title | Priority | Dependency | Category |
|---|------|------|----------|--------------------------------------------|----------|
| 1 | 3306 | [distributed] no attribute '_reset_fr_recording_xccl' in... | N | None | Distributed |
| 2 | 3305 | [distributed] shrink operation support in... | N | None | Distributed |
| 3 | 3300 | [CI] When creating PR, several pull workflows are launched... | N | None | Others |
| 4 | 3296 | accuracy gap of stft in float16 | 5 | None | Dtype / Precision Related |
| 5 | 3290 | huggingface amp_fp16 inference accuracy openai/whisper-tiny... | 7 | None | Dtype / Precision Related |
| 6 | 3286 | New failing test case after enabling tests from... | 3 | None | Others |
| 7 | 3284 | Optimize torch.nn.functional.one_hot | 3 | None | Others |
| 8 | 3280 | [Bug Skip]: New UT failure in 0406 nightly windows. | 9 | None | Others |
| 9 | 3270 | [distributed][tensor] RuntimeError: Invalid scaling... | 9 | None | Others |
| 10 | 3267 | New failed test cases 2026-04-06 | N | None | Others |
| 11 | 3266 | [RFC] Migrate XPU kernel math functions from std::/:: to... | N | None | Others |
| 12 | 3259 | New failed test cases 2026-04-02 | 9 | None | Others |
| 13 | 3258 | huggingface accuracy inference Error in op:... | 3 | None | Others |
| 14 | 3247 | NotImplementedError: "dot_xpu_mkl" not implemented for 'Long' | 5 | None | Others |
| 15 | 3246 | AssertionError: Booleans mismatch: True is not False | N | None | Others |
| 16 | 3243 | AssertionError: False is not true | N | None | Others |
| 17 | 3242 | AssertionError: Torch not compiled with CUDA enabled | N | None | Others |
| 18 | 3238 | The supported dtypes of _refs.stft is not aligned to stft | 5 | None | Dtype / Precision Related |
| 19 | 3233 | [distributed] RuntimeError: No backend for the parent process... | 9 | None | Distributed |
| 20 | 3232 | [distributed][tensor] AssertionError: AssertionError not... | 9 | None | Distributed |
| 21 | 3231 | Dynamo failed to run FX node with fake tensors: call_function... | 9 | None | PT2E |
| 22 | 3229 | RuntimeError: No viable backend for... | 5 | None | Flash Attention / Transformer Related |
| 23 | 3227 | torch xpu event has ~0.1ms latency, which is too large | 9 | None | Others |
| 24 | 3224 | [Win][Build] Building SYCL (Device) object... | 9 | None | Others |
| 25 | 3216 | [OPs] Some ops of XPU have non-determinism and are... | N | None | Others |
| 26 | 3209 | [Win][Build] There is Cyclic dependencies error when build... | 9 | None | Others |
| 27 | 3196 | vitals is not supported, the cases should be disabled | N | None | Others |
| 28 | 3195 | test_sdpa_unbacked_no_dde_xpu crashed | 9 | None | Flash Attention / Transformer Related |
| 29 | 3194 | Incorrect strides in TestCommonXPU,test_out_addmv_xpu_float32 | N | None | Others |
| 30 | 3191 | torch._inductor.exc.InductorError: AssertionError: both a... | 9 | None | Inductor / Compilation Related |
| 31 | 3189 | Task Tracker | N | None | Others |
| 32 | 3187 | PyTorch XPU gpu_cpp_wrapper fails with InductorError... | 5 | None | Inductor / Compilation Related |
| 33 | 3184 | New failing UTs: test_cross_entropy_loss_2d_out_of_bounds_clas... | N | None | Others |
| 34 | 3180 | [E2E] Timm/Torchbench models got "eager_two_runs_differ" on ARC | N | None | Others |
| 35 | 3178 | New failed test cases 2026-03-25 | N | None | Others |
| 36 | 3177 | Accuracy gap of BF16/FP16 test_block_addmm | N | None | Dtype / Precision Related |
| 37 | 3176 | [Bug Skip]: ValueError: _scaled_dot_product_attention(): all... | N | None | Flash Attention / Transformer Related |
| 38 | 3175 | [Bug Skip]: ValueError: sampled_addmm(): all inputs are... | N | None | Others |
| 39 | 3174 | [Bug Skip]: Accuracy failure of test_Conv2d_groups_nobias | 2 | None | Others |
| 40 | 3170 | Unskip test_bmm_windows_error_xpu_float64 | 5 | None | Others |
| 41 | 3169 | NotImplementedError: Could not run 'aten::hspmm' with... | 5 | None | Others |
| 42 | 3167 | NotImplementedError: Could not run 'aten::triangular_solve.X'... | 5 | None | Others |
| 43 | 3166 | test_consistency_SparseCSR failures | 5 | None | Sparse Operations Related |
| 44 | 3165 | test_sparse_csr_xpu.py::TestSparseCompressedTritonKernelsXPU::... | 5 | None | Others |
| 45 | 3163 | [Bug Skip]: Object comparison failed: torch.int64 !=... | 5 | None | Dtype / Precision Related |
| 46 | 3161 | Exception: Tensor-likes are not close! -... | 9 | None | Dtype / Precision Related |
| 47 | 3160 | compiler not found (Windows) | 2 | None | Others |
| 48 | 3158 | AttributeError: module 'triton.compiler' has no attribute... | 9 | None | Inductor / Compilation Related |
| 49 | 3156 | AssertionError: 'Assertion cur_target >= 0 && cur_target <... | 9 | None | Others |
| 50 | 3151 | [Triton] Timm_models rexnet_100 / fbnetv3_b /... | 7 | Triton | Inductor / Compilation Related |
| 51 | 3150 | [Task] Align XPU kernel's implementation to stock PyTorch | N | None | Others |
| 52 | 3148 | [Triton] Huggingface openai/whisper-tiny got fail_accuracy | 7 | Triton | Inductor / Compilation Related |
| 53 | 3143 | NotImplementedError: The operator... | 5 | None | Others |
| 54 | 3142 | [upstream_ut] RuntimeError: The sycl_ext_oneapi_work_group_scr... | 5 | None | Others |
| 55 | 3141 | [upstream_ut] RuntimeError: FlashAttentionForwardXPU only... | 5 | None | Flash Attention / Transformer Related |
| 56 | 3140 | [upstream_ut] RuntimeError: FlashAttentionForwardXPU does not... | 5 | None | Others |
| 57 | 3139 | [distributed][_composable] AssertionError: Expects xpu:0 but... | 9 | None | Distributed |
| 58 | 3137 | [upstream_ut] RuntimeError: expected scalar type Half but... | 5 | None | Dtype / Precision Related |
| 59 | 3136 | [upstream_ut] AssertionError: False is not true in... | 5 | None | Flash Attention / Transformer Related |
| 60 | 3133 | [upstream_ut] RuntimeError: scaled_dot_product_attention: If... | 5 | None | Flash Attention / Transformer Related |
| 61 | 3132 | [upstream_ut] transfomers test reports RuntimeError: No... | 5 | None | Others |
| 62 | 3131 | [upstream_ut] NotImplementedError: The operator... | 5 | None | Others |
| 63 | 3129 | [upstream_ut] AssertionError: UserWarning not triggered | 5 | None | Others |
| 64 | 3128 | [upstream_ut] AssertionError: RuntimeError not raised by <lambda> | 5 | None | Others |
| 65 | 3127 | [upstream_ut] AssertionError: AssertionError not raised | 1 | None | Others |
| 66 | 3126 | [upstream_ut] Two NestedTensor issue with flash attention | 5 | None | Flash Attention / Transformer Related |
| 67 | 3124 | [TorchAO][Bug] ImportError: Requires mslk >= 1.0.0 when... | 9 | None | TorchAO |
| 68 | 3121 | [Bug Skip]: CUDA specific UT test_fft_half_and_chalf_not_power... | N | None | Others |
| 69 | 3114 | [Bug Skip]: Failure skip on 2026-3-21 | 9 | None | Others |
| 70 | 3106 | Worker crashes when running TestDecompXPU,test_quick_core_back... | 9 | None | Others |
| 71 | 3103 | Tensor-likes are not equal for test_backward_nn_functional_con... | N | None | Dtype / Precision Related |
| 72 | 3102 | [distributed] RuntimeError: Invalid device string: 'xpu:foo'... | 9 | None | Others |
| 73 | 3101 | [distributed] 'torch._C._distributed_c10d.ProcessGroupXCCL'... | 9 | None | Distributed |
| 74 | 3100 | [distributed] /handler/dump_nccl_trace_pickle and nccl_log... | N | None | Distributed |
| 75 | 3096 | VISIBLE_DEVICE support | N | None | Others |
| 76 | 3095 | cutlass support blocks some unit test cases | 5 | None | Inductor / Compilation Related |
| 77 | 3094 | XPUGraph tree support | 5 | None | Inductor / Compilation Related |
| 78 | 3093 | XPU does not support NestedTensor for SDPA operations. | N | None | Flash Attention / Transformer Related |
| 79 | 3089 | AssertionError: Torch not compiled with CUDA enabled | N | None | Inductor / Compilation Related |
| 80 | 3088 | [TorchAO][BMG] INT4 RTN Flex-attention got 5% performance drop | 9 | None | TorchAO |
| 81 | 3086 | nvml support blocks some test cases | N | None | Others |
| 82 | 3084 | torch.library.register_autocast does not support xpu | N | None | Dtype / Precision Related |
| 83 | 3083 | [Bug Skip]: Random failures 2026WW12 | 9 | None | Others |
| 84 | 3082 | multithread support in distributed | N | None | Distributed |
| 85 | 3081 | Sparse CSR gemm-like ops have not been supported yet | N | None | Sparse Operations Related |
| 86 | 3080 | cudagraph tests blocked by feature gap | N | None | Others |
| 87 | 3077 | [Bug Skip] test_dlpack.py::TestTorchDlPackXPU::test_no_copy_xp... | 5 | None | Others |
| 88 | 3076 | [TorchAO][BMG] Llama-3.2-1B-Instruct Dynamic INT8 got 10%... | 9 | None | TorchAO |
| 89 | 3074 | [Bug Skip] test_dlpack_exchange_api expect... | 9 | None | Others |
| 90 | 3060 | Implement torch._scaled_grouped_mm for xpu backend | N | None | Others |
| 91 | 3058 | [E2E] hf_GPT2_large amp_fp16/amp_bf16 training got fail_accuracy | 7 | None | Dtype / Precision Related |
| 92 | 3048 | Profiler result is not correct on B70 | N | None | Others |
| 93 | 3041 | AssertionError: Expected len(flat_diff_results) > 0 in... | 5 | None | Others |
| 94 | 3033 | [Bug Skip]: Softmax tolerance | N | None | Others |
| 95 | 3032 | [TorchAO][UT] failures in test/prototype/safetensors/test_safe... | 9 | None | TorchAO |
| 96 | 3030 | [Bug Skip] test/test_modules.py::TestModuleXPU::test_cpu_gpu_p... | 9 | None | Others |
| 97 | 3025 | New failing test in Nightly Wheel... | N | None | Others |
| 98 | 3024 | Enable clang-tidy checks | N | None | Others |
| 99 | 3022 | [distributed] batch_isend_irecv Compatibility Issue on B60/XCCL | N | None | Distributed |
| 100 | 3021 | [distributed] all_to_all_single Compatibility Issue on B60/XCCL | N | None | Distributed |
| 101 | 3014 | [upstream_ut] AssertionError: False is not true | 9 | None | Others |
| 102 | 3013 | [upstream_ut] RuntimeError: Kernel is incompatible with all... | 9 | None | Others |
| 103 | 3011 | [upstream_ut] torch.OutOfMemoryError: XPU out of memory | 9 | None | Others |
| 104 | 3010 | [distributed][tensor] test_random_ops.py... | N | None | PT2E |
| 105 | 3007 | AssertionError: Scalars are not equal! with... | 5 | None | Flash Attention / Transformer Related |
| 106 | 3006 | AssertionError: '.to(tl.float16)' unexpectedly found in '# AOT ID | 5 | None | PT2E |
| 107 | 3004 | TypeError: _xpu_recordMemoryHistory(): incompatible function... | 5 | None | Others |
| 108 | 3000 | [Bug Skip]: RuntimeError: _share_fd_: only available on CPU... | N | None | Others |
| 109 | 2999 | KeyError: 'eager_numerics.use_pytorch_libdevice' | 5 | None | Others |
| 110 | 2997 | AssertionError of test_linear_and_cel_max_autotune | 5 | None | Inductor / Compilation Related |
| 111 | 2993 | [Bug Skip]: Unexpected success of... | 9 | None | Others |
| 112 | 2984 | [release/2.11] sebotnet33ts_256 fp32 training got fail_accuracy | 7 | None | Dtype / Precision Related |
| 113 | 2981 | [release/2.11] T5 models performance dropped ~20% | 9 | None | Others |
| 114 | 2979 | eca_halonext26ts got RuntimeError:... | 9 | driver | Others |
| 115 | 2972 | [distributed] AssertionError: ValueError not raised in... | 9 | None | Distributed |
| 116 | 2969 | [distributed] AssertionError: Scalars are not equal! in... | 9 | None | Distributed |
| 117 | 2968 | [distributed] timeout issue in test/distributed/test_c10d_xccl.py | 9 | None | Distributed |
| 118 | 2966 | [Bug Skip]: [Regression]2026-3-2 ut failures | 9 | None | Others |
| 119 | 2965 | [Bug Skip]: Random failures 2026WW10 | N | None | Others |
| 120 | 2960 | [release/2.11] timm_models_xcit_large_24_p8_224_float16_traini... | 9 | None | Dtype / Precision Related |
| 121 | 2958 | AssertionError of test_dtensor_basic_compile | 5 | None | Inductor / Compilation Related |
| 122 | 2953 | [release/2.11][wsl] huggingface TrOCRForCausalLM and... | 9 | None | Others |
| 123 | 2952 | [release/2.11][wsl] timm_models_accuracy_training_bfloat16... | 9 | None | Dtype / Precision Related |
| 124 | 2950 | SYCL compilation flag -fsycl-id-queries-fit-in-int does not... | N | None | Others |
| 125 | 2948 | [AO] Benchmark enabling on XPU | N | None | Others |
| 126 | 2946 | [Bug Skip]: Random failures 2026WW09 | 9 | None | Others |
| 127 | 2942 | [Windows] Unit tests got Fatal python error | 9 | None | Others |
| 128 | 2939 | [release/2.11] gmlp_s16_224 inference amp performance dropped... | 9 | None | Others |
| 129 | 2938 | [release/2.11] basic_gnn_gin and basic_gnn_sage inference... | 9 | None | Others |
| 130 | 2935 | [release/2.11][inductor] huggingface amp_fp16 and float16... | 9 | None | Inductor / Compilation Related |
| 131 | 2928 | [release/2.11] pyhpc_turbulent_kinetic_energy fp32 inference... | 7 | None | Dtype / Precision Related |
| 132 | 2924 | [release/2.11] xcit_large_24_p8_224 amp_bf16 training got... | 7 | Triton | Inductor / Compilation Related |
| 133 | 2922 | [release/2.11] UT inductor AssertionError: pass_fds not... | 9 | None | Inductor / Compilation Related |
| 134 | 2921 | [abs][complex64] - new failing test cases caused by PyTorch... | N | None | Others |
| 135 | 2919 | [XPU][upstream_ut][COW] Fix materialization in remaining... | 5 | None | Others |
| 136 | 2918 | [XPU][upstream_ut][COW] Skip non-supported ops (jiterator +... | 5 | None | Others |
| 137 | 2914 | Test case test/test_autograd.py::TestAutogradMultipleDispatchC... | 9 | None | Others |
| 138 | 2912 | [release/2.11] UT extended 220 new failures | 9 | None | Others |
| 139 | 2908 | [release/2.11] Model fail_accuracy for 5 testcases | 7 | None | Dtype / Precision Related |
| 140 | 2907 | [release/2.11] Models performance regression for 5 testcases | 9 | None | Others |
| 141 | 2891 | RuntimeError: Expected to find "(262144, 0, 512, 1" but did... | 5 | None | Others |
| 142 | 2888 | torch._inductor.exc.InductorError: AssertionError:... | 5 | None | Inductor / Compilation Related |
| 143 | 2879 | RuntimeError: _share_fd_: only available on CPU | N | None | Others |
| 144 | 2873 | [Bug Skip]: test_repos.py contains several failed ops | 9 | None | Others |
| 145 | 2869 | [Bug Skip]: New UT failure in 0209 nightly windows. | N | None | Others |
| 146 | 2862 | accuracy issue with test_float8_scale_fast_accum_xpu | 9 | None | Dtype / Precision Related |
| 147 | 2858 | [Bug Skip]: test_xpu new failures | 9 | None | Others |
| 148 | 2853 | [upstream_ut] torch.ops.aten._flash_attention_forward lack of... | N | None | Flash Attention / Transformer Related |
| 149 | 2852 | [Bug Skip]: New UT failures in 0206 nightly on Windows | 9 | None | Others |
| 150 | 2845 | [Bug Skip]:[UT] [Windows] failed cases 2026-2-4 | 9 | None | Others |
| 151 | 2840 | Accuracy issue with 64 bit indexing depthwise_conv | N | oneDNN | Dtype / Precision Related |
| 152 | 2837 | Accuracy issue for Muon optimizer | N | None | Dtype / Precision Related |
| 153 | 2823 | [TorchAO][BMG] Llama-3.2-1B-Instruct Dynamic INT8 got 20%... | 9 | None | TorchAO |
| 154 | 2817 | Expected error message is different than actual | N | None | Others |
| 155 | 2816 | torch.logcumsumexp incorrectly returns NaNs for complex64 input | N | None | Others |
| 156 | 2815 | RuntimeError: output with shape [2] doesn't match the... | N | None | Others |
| 157 | 2811 | [Bug Skip]: [Regression] failed cases 2026-2-2 | 9 | None | Others |
| 158 | 2810 | AssertionError: Object comparison failed:... | 5 | None | Inductor / Compilation Related |
| 159 | 2806 | CompiledAOTI need XPU support | 5 | None | Inductor / Compilation Related |
| 160 | 2802 | Three aten._scaled_dot_product_flash_attention issues | 5 | None | Flash Attention / Transformer Related |
| 161 | 2801 | to_dense() for Sparse CSR backend cannot broadcast batch dim... | 9 | None | Sparse Operations Related |
| 162 | 2800 | AttributeError: 'torch._C._XpuDeviceProperties' object has no... | 5 | oneAPI | Others |
| 163 | 2798 | Test case test/test_dlpack.py::TestTorchDlPackCPU::test_numpy_... | 5 | None | Others |
| 164 | 2795 | Histc raises error with integer input when deterministic... | 9 | None | Others |
| 165 | 2783 | [Bug Skip]: Key "xpu" is missing from dict "driver" in test_svd | N | None | Others |
| 166 | 2779 | Accuracy failures in logspace op | N | None | Dtype / Precision Related |
| 167 | 2777 | [Bug Skip]: Random failures 2026WW05 | 9 | None | Others |
| 168 | 2769 | [oneDNN] New failed test cases with 3.11 compared with 3.10 | 9 | oneDNN | Others |
| 169 | 2767 | [UT] test_control_flow_xpu.py got AssertionError | 9 | None | Others |
| 170 | 2766 | MaxPool2d - investigate memory layout performance | 9 | None | Others |
| 171 | 2759 | [Bug Skip]: New failed cases 2026-1-22 | N | None | Others |
| 172 | 2751 | [Bug Skip]: Random failures 2026WW04 | 9 | None | Others |
| 173 | 2744 | [Bug Skip]: extended test failures when test_compare_cpu atol... | 9 | None | Others |
| 174 | 2742 | [Linux][PT2E] hf_Roberta_base model performance ASYMM and... | 9 | None | PT2E |
| 175 | 2738 | [distributed] test_c10d_spawn_nccl.py ValueError: input... | 9 | None | Distributed |
| 176 | 2737 | [distributed] AttributeError: module 'torch._C' has no... | 9 | None | Distributed |
| 177 | 2729 | [Bug Skip]: Random failures 2026WW03 | 9 | None | Others |
| 178 | 2722 | [Bug Skip]: NotImplementedError: Could not run 'aten::flip'... | N | None | TorchAO |
| 179 | 2720 | [upstream_ut] RuntimeError: false INTERNAL ASSERT FAILED at... | 9 | None | Others |
| 180 | 2715 | [upstream_ut] torch._dynamo.exc.Unsupported: Attempted to... | 5 | None | PT2E |
| 181 | 2714 | [upstream_ut] AssertionError: Object comparison failed:... | 5 | None | Others |
| 182 | 2712 | [upstream_ut] RuntimeError: Cannot swap t2 because it has... | 5 | None | Others |
| 183 | 2707 | [TorchAO][BMG] INT4 GPTQ failed due to TorchAO API change. | 9 | None | TorchAO |
| 184 | 2702 | [distributed] RuntimeError: Work ran time out after 0... | 9 | None | Distributed |
| 185 | 2701 | [distributed] Barrier Timeout Error with... | 9 | None | Distributed |
| 186 | 2700 | [distributed] Hang issues with test_distributed_spawn.py | N | None | Distributed |
| 187 | 2698 | Title: [upstream_ut] RuntimeError: FlashAttentionForwardXPU... | 5 | None | Others |
| 188 | 2697 | Title: [upstream_ut] RuntimeError: Expected to find ", 0, "... | 5 | None | Others |
| 189 | 2694 | Title: [upstream_ut] AssertionError: Tensor-likes are not... | 5 | None | Inductor / Compilation Related |
| 190 | 2693 | Title: [upstream_ut] AssertionError: Scalars are not equal! | 5 | None | Inductor / Compilation Related |
| 191 | 2689 | [LNL][Windows] AssertionError: 'Assertion `cur_target >= 0 &&... | 9 | None | Others |
| 192 | 2686 | [distributed] Accuracy issues with test_distributed_spawn.py | 9 | None | Distributed |
| 193 | 2680 | XPU Autocast does not support fp32 dtypes | 9 | None | Dtype / Precision Related |
| 194 | 2676 | Random failure in CI test | 9 | None | Others |
| 195 | 2675 | [Bug Skip]: AttributeError: 'NoneType' object has no... | N | None | Others |
| 196 | 2670 | [upstream_ut] RuntimeError: could not create a primitive... | 5 | None | Others |
| 197 | 2669 | [upstream_ut] AssertionError: Tensor-likes are not close! in... | N | None | Others |
| 198 | 2663 | test_sparse_semi_structured.py gaps | 5 | None | Sparse Operations Related |
| 199 | 2662 | [release/2.10][Windows][BMG] New failed test cases and 2.9... | 9 | None | Others |
| 200 | 2660 | [release/2.10][Windows][BMG] New failed test cases | 9 | None | Others |
| 201 | 2659 | [distributed] test_dist2.py RuntimeError: Backend xccl does... | 9 | None | Distributed |
| 202 | 2656 | [release/2.10] models got fail_accuracy on BMG WSL2 | 9 | None | Dtype / Precision Related |
| 203 | 2655 | [BMG][OOB] hf_Reformer performance drop | 9 | Triton | Dtype / Precision Related |
| 204 | 2654 | [BMG][OOB] t5 inference performance drop 2 | 9 | None | Dtype / Precision Related |
| 205 | 2649 | [distributed][pipelining] test_schedule_multiproc.py hang issue | N | None | Distributed |
| 206 | 2640 | random issue test_vjpvjp_index_reduce_prod_xpu_float32 | N | None | Dtype / Precision Related |
| 207 | 2639 | test_to() failed during rnn isinstance() check | N | None | Others |
| 208 | 2630 | Title: [upstream_ut] AssertionError: Scalars are not equal! | 9 | None | Others |
| 209 | 2620 | [upstream_ut] AssertionError: dtype is needed to compute eps1... | 5 | None | Inductor / Compilation Related |
| 210 | 2619 | [release/2.10] Some models inductor performance dropped ~ 10%... | 9 | None | Inductor / Compilation Related |
| 211 | 2618 | [Bug Skip]: [regression] AssertionError: Scalars are not... | N | None | Others |
| 212 | 2615 | [Bug Skip]: New failures RuntimeError: Unsupported dtype Half... | N | None | Dtype / Precision Related |
| 213 | 2613 | [upstream_ut] AssertionError: Tensor-likes are not equal! in... | 5 | driver | Inductor / Compilation Related |
| 214 | 2611 | [upstream_ut] AssertionError: Tensor-likes are not equal! in... | 5 | driver | Inductor / Compilation Related |
| 215 | 2609 | [upstream_ut] torch._inductor.exc.InductorError:... | 5 | None | Inductor / Compilation Related |
| 216 | 2605 | [int4][inductor] Add freezing pattern for fusing int4 mm... | N | None | TorchAO |
| 217 | 2598 | [TorchAO][BMG]The first token latency of... | 9 | None | TorchAO |
| 218 | 2597 | [TorchAO][BMG] INT4 GPTQ shows worse performance compared... | 9 | None | TorchAO |
| 219 | 2595 | [Bug Skip]: Random crashed cases 2025-12-17 | N | None | Others |
| 220 | 2592 | [release/2.10] models got fail_accuracy | 7 | None | Dtype / Precision Related |
| 221 | 2580 | [TorchAO][UT] test/test_low_bit_optim.py AssertionError:... | 9 | None | TorchAO |
| 222 | 2578 | [TorchAO][UT] test/quantization/test_quant_api.py... | 5 | None | TorchAO |
| 223 | 2572 | [TorchAO][UT] test/dtypes/test_affine_quantized.py... | 9 | None | TorchAO |
| 224 | 2570 | crash in sdpa. | 9 | oneDNN | Flash Attention / Transformer Related |
| 225 | 2562 | Warning as Error | 9 | None | Others |
| 226 | 2560 | [UT] "RuntimeError: iter.device(arg).is_xpu()" in... | 9 | None | Others |
| 227 | 2554 | [upstream_ut] AssertionError: AssertionError not raised | 5 | None | Inductor / Compilation Related |
| 228 | 2541 | Title: [upstream_ut] RuntimeError: could not construct a... | 9 | None | Others |
| 229 | 2539 | Title: [upstream_ut] RuntimeError: Tried to instantiate dummy... | 9 | None | Others |
| 230 | 2537 | Title: [upstream_ut] Failed: Unexpected success | N | None | Others |
| 231 | 2536 | Title: [upstream_ut] AttributeError: module 'torch._C' has no... | N | None | Others |
| 232 | 2535 | Title: [upstream_ut] AttributeError: module 'torch._C' has no... | 9 | None | Others |
| 233 | 2533 | Title: [upstream_ut] AttributeError: 'TestQuantizedOpsXPU'... | 9 | None | TorchAO |
| 234 | 2532 | Title: [upstream_ut] AssertionError: wrong number of... | N | None | TorchAO |
| 235 | 2531 | [upstream_ut] AssertionError: Torch not compiled with CUDA... | N | None | Others |
| 236 | 2530 | Title: [upstream_ut] AssertionError: RuntimeError not raised | N | None | Others |
| 237 | 2529 | [upstream_ut] AssertionError: False is not true | 9 | None | Others |
| 238 | 2519 | [upstream_ut] TypeError: map2_ is only implemented on CPU tensors | 9 | None | Others |
| 239 | 2518 | [upstream_ut] TypeError: Creating a Tensor subclass from a... | 2 | None | Others |
| 240 | 2513 | [upstream_ut] RuntimeError: _share_fd_: only available on CPU | 9 | None | Others |
| 241 | 2512 | [upstream_ut] RuntimeError: _histc_xpu does not have a... | 9 | None | Others |
| 242 | 2510 | [upstream_ut] RuntimeError: Expected output.numel() <=... | 9 | None | Others |
| 243 | 2508 | TypedStorage / TypedTensors deprecation | 1 | None | Others |
| 244 | 2496 | [upstream_ut] Segmentation fault when running... | 2 | None | Others |
| 245 | 2491 | [upstream_ut] AssertionError: False is not true | 9 | None | Others |
| 246 | 2482 | test_dtypes issue introduced by pytorch test sample input updates | N | None | Dtype / Precision Related |
| 247 | 2479 | [Bug] torch.rand output different result on bmg and pvc | 9 | None | Others |
| 248 | 2472 | [upstream_ut] NotImplementedError: The operator... | 1 | None | Others |
| 249 | 2471 | test_cuda.py gaps | N | None | Others |
| 250 | 2467 | Host may stuck when submit too many kernels when event recording | N | driver | Others |
| 251 | 2465 | [windows] ut hang | N | None | Others |
| 252 | 2463 | [Bug Skip]: OSError: SYCL runtime is not dected. | 9 | None | Others |
| 253 | 2446 | [Bug Skip]: AssertionError: "Simulate error" does not match... | N | None | Others |
| 254 | 2444 | [upstream_ut] RuntimeError: UR backend failed. UR backend... | 9 | None | Others |
| 255 | 2442 | [Bug Skip]: NotImplementedError: Could not run... | N | None | Flash Attention / Transformer Related |
| 256 | 2439 | [oneDNN] TestDecompXPU.test_quick_addmv_xpu_float64 got fail... | 9 | oneDNN | Dtype / Precision Related |
| 257 | 2436 | [upstream_ut] AttributeError: 'NoneType' object has no... | N | None | Others |
| 258 | 2434 | [Bug Skip]: New failures 2025-11-28 | N | None | Others |
| 259 | 2425 | [upstream_ut] RuntimeError: Expected both self and other to... | N | None | Others |
| 260 | 2412 | Some NestedTensor missing XPU support | N | None | Others |
| 261 | 2404 | [distributed][checkpoint] AssertionError: Booleans mismatch:... | 9 | None | Distributed |
| 262 | 2400 | [ut_upstream] tf32_on_and_off() need xpu support | N | None | Others |
| 263 | 2392 | [Bug Skip]: torch.OutOfMemoryError: XPU out of memory | 9 | None | Others |
| 264 | 2390 | SDPA in pytorch use different backend compared with ipex | N | None | Flash Attention / Transformer Related |
| 265 | 2389 | [Bug Skip]: RuntimeError: Data corruption detected | 9 | None | Others |
| 266 | 2376 | [Bug Skip]: NotImplementedError: "logaddexp_xpu" not... | N | None | Others |
| 267 | 2359 | [upstream_ut] GradcheckError: Backward is not reentrant | 5 | None | Others |
| 268 | 2358 | test/test_view_ops.py::TestOldViewOpsXPU::test_ravel_xpu meet... | 5 | None | TorchAO |
| 269 | 2349 | [oneAPI][backward compatibility] libur_loader.so.0: version... | N | None | Others |
| 270 | 2340 | [distributed][_tools] AssertionError: Roofline estimation... | 9 | None | Distributed |
| 271 | 2331 | [upstream_ut] AssertionError: Scalars are not equal! with... | 5 | oneAPI | Others |
| 272 | 2329 | [upstream_ut] feature missing: get_device_tflops and... | 5 | Triton | Inductor / Compilation Related |
| 273 | 2327 | [TorchAO] benchmark enabling on XPU | N | None | TorchAO |
| 274 | 2326 | [TorchAO] MX training native PyTorch on XPU | N | None | TorchAO |
| 275 | 2325 | [TorchAO] Float8 training support on XPU | N | None | TorchAO |
| 276 | 2324 | [TorchAO] FP8 conv support | N | None | TorchAO |
| 277 | 2323 | [TorchAO] MOE training enabling on XPU | N | None | TorchAO |
| 278 | 2309 | unsupported ops with PYTORCH_ENABLE_XPU_FALLBACK unset | 1 | None | Others |
| 279 | 2301 | [upstream_ut] dtypes not align with OpInfo | 5 | None | Others |
| 280 | 2295 | [upstream_ut][xpu][test]nn/test_embedding.py::TestEmbeddingNND... | 5 | None | Others |
| 281 | 2287 | [upstream_ut] test_python_ref issues | 5 | None | Others |
| 282 | 2285 | Support efficient attention | N | None | Others |
| 283 | 2283 | [upstream_ut] sparse._sampled_addmm is not supported | 5 | None | Sparse Operations Related |
| 284 | 2270 | Backend Compatibility Error in test/xpu/test_decomp.py | 9 | None | Others |
| 285 | 2263 | [xpu][bug] XPU Trace event ends too late! | 5 | None | Others |
| 286 | 2261 | [xpu][profiler] Run with fork process has extra warning | N | oneAPI | Others |
| 287 | 2257 | Accuracy failures in test/xpu/test_unary_ufuncs_xpu.py | N | None | Dtype / Precision Related |
| 288 | 2255 | [upstream_ut] RuntimeError: Long is not supported in oneDNN | 5 | None | Others |
| 289 | 2253 | the supported dtypes are not align with cuda | 5 | None | Others |
| 290 | 2251 | [upstream_ut] test_fake_autocase got Exception: Dtypes... | 5 | None | Dtype / Precision Related |
| 291 | 2250 | Found mismatch when comparing the output of aten.view.default... | N | None | Others |
| 292 | 2248 | [upstream_ut] test_cow failures | 5 | None | Others |
| 293 | 2246 | torch/sparse/_triton_ops*.py need to be ported to enable for... | N | None | Sparse Operations Related |
| 294 | 2245 | oneDNN matmul received incorrect shape in... | N | None | Sparse Operations Related |
| 295 | 2244 | test/test_sparse_csr.py::TestSparseCSRXPU::test_block_addmm... | 9 | None | Sparse Operations Related |
| 296 | 2240 | RuntimeError: Trying to set a forward gradient that has a... | N | None | Others |
| 297 | 2239 | Exception: could not create a primitive descriptor for the... | N | None | Others |
| 298 | 2238 | Exception: Tensor-likes are not close! in... | N | None | Others |
| 299 | 2235 | test/test_sparse_csr.py::TestSparseCompressedTritonKernelsXPU:... | N | None | Sparse Operations Related |
| 300 | 2234 | [upstream_ut] AssertionError: RuntimeError not raised :... | 5 | None | Others |
| 301 | 2232 | sdpa backward kernel is required to reduce memory usage | N | None | Flash Attention / Transformer Related |
| 302 | 2230 | test_sparse_csr.py::TestSparseCompressedTritonKernelsXPU::test... | N | None | Sparse Operations Related |
| 303 | 2229 | test/test_sparse_csr.py::TestSparseCompressedCPU::test_invalid... | N | None | Sparse Operations Related |
| 304 | 2220 | test/test_sparse_csr.py::TestSparseCompressedTritonKernelsXPU:... | N | Triton | Sparse Operations Related |
| 305 | 2219 | float8_e4m3fn precision overflow | 9 | None | Dtype / Precision Related |
| 306 | 2217 | AO Performance issue track | 9 | None | Others |
| 307 | 2215 | Find use case example for torch-xpu-ops.lib in sycl cpp extension | N | None | Others |
| 308 | 2214 | test/test_sparse.py::TestSparseAnyXPU::test_gradcheck_mm... | 5 | None | Sparse Operations Related |
| 309 | 2207 | Enable FP8/MXFP8 Ops with requests and CUDA alignment | N | None | TorchAO |
| 310 | 2201 | [TorchAO][BMG] When using paged attention backend, all cases... | 9 | None | TorchAO |
| 311 | 2200 | support flash attention op on XPU device | N | oneDNN | Flash Attention / Transformer Related |
| 312 | 2199 | Fix reduction and norm register spill | N | None | Others |
| 313 | 2196 | Fix DistributionElementwiseKernelFunctor register spill | N | None | Others |
| 314 | 2186 | AssertionError: Mul tiheadAttention does not support... | N | oneDNN | Others |
| 315 | 2182 | test_transform_bias_rescale_qkv_nested_xpu_float32 failed... | 9 | None | Dtype / Precision Related |
| 316 | 2169 | Frame size comparison failed in test_size_comparison_no_recompile | 9 | None | Inductor / Compilation Related |
| 317 | 2165 | [distributed] test_device_mesh.py::TestDeviceMeshGetItem::test... | 9 | None | Distributed |
| 318 | 2164 | skip test_no_cuda_monkeypatch as it is cuda specific | 1 | None | Others |
| 319 | 2163 | 3 distributed UT cases need to be supported by -... | N | None | Distributed |
| 320 | 2142 | XPU max_memory_allocated have different output with CUDA | N | None | Others |
| 321 | 2140 | Consider how to avoid copy in FFT kernels | N | None | Others |
| 322 | 2132 | [2.9][BMG-Windows][Torch-xpu-ops UT] 1 case failed with... | 9 | None | Inductor / Compilation Related |
| 323 | 2128 | [2.9][BMG-Windows][Torchbench] speeach_transforer... | 9 | None | Dtype / Precision Related |
| 324 | 2127 | Path Coverage enhancement | N | None | Others |
| 325 | 2113 | Update example for Distributed Data Parallel | N | None | Distributed |
| 326 | 2098 | Upstream XPU functions in yaml | N | None | Others |
| 327 | 2089 | need an implementation that won't initialize gpu context for... | N | driver | Others |
| 328 | 2086 | nd_item::barrier has been deprecated | N | None | Others |
| 329 | 2063 | Avoid using out-of-date term | N | None | Others |
| 330 | 2024 | AssertionError: Torch not compiled with CUDA enabled | N | None | Inductor / Compilation Related |
| 331 | 2022 | [Windows] [CI] [UT] AssertionError: Tensor-likes are not close! | 9 | None | Others |
| 332 | 2015 | inf is returned by nn.TransformerEncoderLayer | N | None | Others |
| 333 | 2006 | work-item/workgroup issue in softmax/unsampling/nonzero | N | None | Others |
| 334 | 2004 | [distributed][shared_tensor] test\distributed\_shard\shared_te... | 9 | None | Distributed |
| 335 | 1996 | [TorchAO] Memory Efficient Optimizers | N | None | TorchAO |
| 336 | 1986 | torch.xpu._sleep is missing, | N | oneAPI | Others |
| 337 | 1973 | AssertionError: Scalars or Tensor-likes are not equal or close! | 9 | None | Others |
| 338 | 1970 | torch._dynamo.exc.BackendCompilerFailed: backend='inductor'... | 9 | None | Inductor / Compilation Related |
| 339 | 1969 | torch._dynamo.exc.InternalTorchDynamoError: TypeError: cannot... | 9 | None | PT2E |
| 340 | 1963 | [upstream_ut] MetadataMismatchError in TestFakeTensor of... | 5 | None | Others |
| 341 | 1951 | Functionality issues in TestCommon.test_out. | 5 | None | Others |
| 342 | 1936 | implement torch.linalg.cholesky xpu backend | N | None | Others |
| 343 | 1912 | Implement the torch.ops.aten._weight_int4pack_mm for... | N | oneDNN | TorchAO |
| 344 | 1902 | implement torch.linalg.pinv xpu backend | N | None | Others |
| 345 | 1901 | implement torch.linalg.svd xpu backend | N | None | Others |
| 346 | 1900 | implement torch.linalg.qr xpu backend | N | None | Others |
| 347 | 1894 | [Linux][PT2E] performance test got failed, int8 ASYMM and... | 9 | None | TorchAO |
| 348 | 1893 | [upstream_ut] oneDNN accuracy issues in test_ops_xpu.py | 5 | None | Others |
| 349 | 1877 | Torchbench model squeezenet1_1 and functorch_dp_cifar10 got... | 9 | None | Dtype / Precision Related |
| 350 | 1866 | [release 2.8]Torchbench vision_maskrcnn (amp_b16/amp_fp16... | 7 | None | Dtype / Precision Related |
| 351 | 1856 | channel last aten::hardswish_ will call extra copy | N | None | Others |
| 352 | 1818 | [BMG-Windows][PT2.8]Torch-xpu-ops UT got accuracy issue | 9 | None | Dtype / Precision Related |
| 353 | 1784 | [Performance] Torch XPU Profiler is not reliable | 9 | None | Others |
| 354 | 1778 | [Infra] Show known issues for accuracy test | 7 | None | Dtype / Precision Related |
| 355 | 1762 | Add an ocloc AOT target compilation test in cmake | N | None | PT2E |
| 356 | 1749 | transformers UT failure in XPU because SDPA check error... | 9 | None | Flash Attention / Transformer Related |
| 357 | 1729 | Validation Check List | N | None | Others |
| 358 | 1727 | [distributed] AttributeError: module 'torch.xpu' has no... | 9 | oneAPI | Distributed |
| 359 | 1722 | Ask an API to query GPU type(iGPU/dGPU). | N | oneAPI | Others |
| 360 | 1689 | [For op Perf Comparison] Save reference comparison run id | N | None | Others |
| 361 | 1678 | missing op support for `model.share_memory()` | N | None | Others |
| 362 | 1661 | [distributed] Accuracy gap in _composable/fsdp on Xelink | 9 | None | Distributed |
| 363 | 1649 | [cpp extension] Provide a clear error message when using... | 9 | oneAPI | Others |
| 364 | 1645 | [For Comparison] Save reference comparison run id | N | None | Others |
| 365 | 1624 | [DONT CLOSE] Known UT Issue list | N | oneCCL | Distributed |
| 366 | 1594 | Keep track on the building warning | N | None | Others |
| 367 | 1587 | Keep track on the latest CUDA op impl | N | None | Others |
| 368 | 1574 | The operator 'aten::_grouped_mm' is not currently implemented... | N | None | Others |
| 369 | 1571 | [distributed] ValueError: Cannot use ReduceOp.PREMUL_SUM with... | 9 | None | Distributed |
| 370 | 1556 | [distributed] NotImplementedError: Operator... | 9 | oneDNN | Distributed |
| 371 | 1555 | [distributed] RuntimeError: aten.add.Tensor: got mixed... | 9 | oneDNN | Distributed |
| 372 | 1551 | [distributed] NotImplementedError: The operator... | 9 | oneAPI | Distributed |
| 373 | 1549 | [distributed] AssertionError: 'fused_all_gather_scaled_matmul'... | 9 | oneAPI | Distributed |
| 374 | 1548 | [distributed] AssertionError: 'fused_all_gather_matmul' not... | 9 | oneAPI | Distributed |
| 375 | 1547 | [distributed] NotImplementedError: The operator... | 9 | oneAPI | Distributed |
| 376 | 1505 | [ARC-WSL-Ubuntu24.04] 15 Timm models got fail_accuracy | 5 | None | Inductor / Compilation Related |
| 377 | 1324 | [Win] UR Error when OOM and break the tensor context | 9 | oneAPI | Others |
| 378 | 1171 | LNL Windows got unexpected error message | 9 | driver | Others |
| 379 | 1159 | [LNL Windows][Test by CD Nightly Wheels] hugging face model -... | 9 | None | Dtype / Precision Related |
| 380 | 1059 | SYCL RT: Using recommended shortcut API for kernel specific... | N | oneAPI | Others |
| 381 | 492 | Timm_efficientdet NotImplementedError: The original model... | 9 | None | Dtype / Precision Related |
| 382 | 489 | Moco NotImplementedError: xpu not supported | 9 | None | Dtype / Precision Related |
| 383 | 208 | Abstract utility functions used in ATen operator implementation. | N | None | Others |
| 384 | 146 | Evaluate register spill in SYCL kernel | N | None | Others |

## <span id='7-duplicated-issues'>7. Duplicated Issues</span>

**Issues marked as duplicated: 14**

| # | ID | Title | Priority | Duplicated Issue |
|---|------|------|----------|--------------------------------------------|
| 1 | 3286 | New failing test case after enabling tests from... | 3 | 2715 |
| 2 | 2873 | [Bug Skip]: test_repos.py contains several failed ops | 9 | 2714,2714 |
| 3 | 2853 | [upstream_ut] torch.ops.aten._flash_attention_forward lack of... | N | 2285 |
| 4 | 2715 | [upstream_ut] torch._dynamo.exc.Unsupported: Attempted to... | 5 | 3286 |
| 5 | 2714 | [upstream_ut] AssertionError: Object comparison failed:... | 5 | 2873 |
| 6 | 2536 | Title: [upstream_ut] AttributeError: module 'torch._C' has no... | N | 2508 |
| 7 | 2508 | TypedStorage / TypedTensors deprecation | 1 | 2536 |
| 8 | 2444 | [upstream_ut] RuntimeError: UR backend failed. UR backend... | 9 | 2024 |
| 9 | 2301 | [upstream_ut] dtypes not align with OpInfo | 5 | 2255 |
| 10 | 2285 | Support efficient attention | N | 2853 |
| 11 | 2255 | [upstream_ut] RuntimeError: Long is not supported in oneDNN | 5 | 2301 |
| 12 | 2024 | AssertionError: Torch not compiled with CUDA enabled | N | 2444 |
| 13 | 2015 | inf is returned by nn.TransformerEncoderLayer | N | 2006 |
| 14 | 2006 | work-item/workgroup issue in softmax/unsampling/nonzero | N | 2015 |

## <span id='8-statistics'>8. Statistics</span>

### Action TBD Distribution

| Action TBD | Count |
|------------|------:|
| Awaiting response from reporter | 143 |
| Close fixed issue | 4 |
| E2E accuracy issue | 11 |
| Need Investigation | 142 |
| Needs Upstream Skip PR | 76 |
| Verify the issue | 3 |
| add to skiplist | 5 |

### Category Distribution

| Category | Count |
|----------|------:|
| Distributed | 37 |
| Dtype / Precision Related | 40 |
| Flash Attention / Transformer Related | 17 |
| Inductor / Compilation Related | 30 |
| Others | 214 |
| PT2E | 7 |
| Sparse Operations Related | 13 |
| TorchAO | 26 |

### Test Module Distribution

| Test Module | Count |
|-------------|------:|
| build | 6 |
| e2e | 31 |
| ut | 347 |
