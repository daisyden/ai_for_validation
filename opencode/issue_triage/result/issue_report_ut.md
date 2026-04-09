# Torch XPU Ops UT Issue Report (Custom Filtered)

**Repository:** [intel/torch-xpu-ops](https://github.com/intel/torch-xpu-ops)

**Report Type:** UT (Unit Test) Issues - Custom Filtered List

**Generated:** 2026-04-08 23:59:54

---

## Index

1. [Summary](#user-content-1-summary)
2. [Statistics](#user-content-2-statistics)
   - [By Test Module](#user-content-by-test-module)
   - [By Module](#user-content-by-module)
   - [By Dependency](#user-content-by-dependency)
   - [By Action TBD](#user-content-by-action-tbd)
   - [By Category](#user-content-by-category-stats)
   - [By Priority](#user-content-by-priority)
3. [New Submitted Issues (Past Week)](#user-content-3-new-submitted-issues-past-week)
4. [Action Required](#user-content-4-action-required)
   - [Reporter Actions](#user-content-reporter-actions)
     - [Information Required](#user-content-information-required)
     - [Close Fixed Issue](#user-content-close-fixed-issue)
     - [Enable Test](#user-content-enable-test)
     - [Add to Skiplist](#user-content-add-to-skiplist)
     - [Verify the Issue](#user-content-verify-the-issue)
     - [Need Reproduce Steps](#user-content-need-reproduce-steps)
   - [Engineer Actions](#user-content-engineer-actions)
     - [Needs PyTorch Repo Changes (upstream)](#user-content-needs-pytorch-repo-changes-upstream)
     - [Revisit the PR as Case Failed](#user-content-revisit-the-pr-as-case-failed)
5. [By Category](#user-content-5-by-category)
6. [Duplicated Issues](#user-content-6-duplicated-issues)
7. [Issues with Dependency](#user-content-7-issues-with-dependency)

---

## 1. Summary

| Category | Count |
|----------|-------|
| Action Required | 61 |
| No Assignee | 0 |
| Duplicated Issues | 13 |
| With Dependency | 0 |
| Others | 1 |
| **Total** | 75 |

---

## 2. Statistics

### By Test Module

| Test Module | Count |
|-------------|-------|
| ut | 70 |
| e2e | 4 |
| build | 1 |

### By Module

| Module | Count |
|--------|-------|
| aten_ops | 51 |
| inductor | 22 |
| AO | 1 |
| unknown | 1 |

### By Dependency

| Dependency | Count |
|------------|-------|
| driver | 3 |
| oneAPI | 1 |
| Triton | 1 |

### By Action TBD

| Action TBD | Count |
|------------|-------|
| Needs PyTorch Repo Changes (upstream) | 48 |
| Need reproduce steps (Only for bugs or performance issue) | 16 |
| Need more information - error logs and reproduction steps | 5 |
| Verify the issue | 2 |
| add to skiplist | 2 |
| Close fixed issue | 1 |

### By Category (Statistics)

| Category | Count |
|----------|-------|
| Distributed | 24 |
| Flash Attention / Transformer Related | 22 |
| Dtype / Precision Related | 8 |
| Others | 7 |
| PT2E | 6 |
| TorchAO | 4 |
| Inductor / Compilation Related | 3 |
| Sparse Operations Related | 1 |

### By Priority

| Priority | Count |
|----------|-------|
| P0 | 2 |
| P1 | 2 |
| P2 | 71 |

---

## 3. New Submitted Issues (Past Week)

Issues created in the past 7 days (as of 2026-04-08).

| ID | Title | Status | Owner | Priority | Reason | Category | Root Cause | Labels | Module | Test Module |
|---|-------|--------|-------|---------|--------|----------|-----------|--------|--------|-------------|

---

## 4. Action Required

### Reporter Actions

#### Information Required

| ID | Title | Owner | Owner Transferred | Priority | Category | Root Cause | PR | Test Module |
|---|-------|-------|-------------------|---------|----------|-----------|-----|-------------|
| [1962](https://github.com/intel/torch-xpu-ops/issues/1962) | [upstream_ut] segfault with test_fa | jenniew, mengfei25 | daisyden | P0 | Sparse Operations Related | Backend/Device Issue - segfault related to XPU device operation in test |  | ut |
| [2229](https://github.com/intel/torch-xpu-ops/issues/2229) | test/test_sparse_csr.py::TestSparse | jenniew | wincent8 | P2 | Distributed | Backend/Device Issue - device mismatch between crow_indices and col_indices on X |  | ut |
| [2853](https://github.com/intel/torch-xpu-ops/issues/2853) | [upstream_ut] torch.ops.aten._flash | LuFinch | BBBela | P2 | Distributed | Flash Attention/Specific Ops Issue - _flash_attention_forward not supported on X |  | ut |
| [2024](https://github.com/intel/torch-xpu-ops/issues/2024) | AssertionError: Torch not compiled  | daisyden | mengfei25 | P2 | Distributed |  |  | ut |
| [2169](https://github.com/intel/torch-xpu-ops/issues/2169) | Frame size comparison failed in tes | guangyey | daisyden | P2 | Distributed | Failure - Scalars are not equal in test comparison |  | ut |
| [2245](https://github.com/intel/torch-xpu-ops/issues/2245) | oneDNN matmul received incorrect sh | CuiYifeng | wincent8 | P2 | Distributed | DNNL/OneDNN Issue - oneDNN matmul primitive descriptor creation failure |  | ut |
| [2531](https://github.com/intel/torch-xpu-ops/issues/2531) | [upstream_ut]  AssertionError: Torc | daisyden | daisyden | P2 | Distributed |  |  | ut |
| [2532](https://github.com/intel/torch-xpu-ops/issues/2532) | Title: [upstream_ut]  AssertionErro | yucai-intel | daisyden | P2 | Distributed | Failure - wrong number of dimensions for int4 conversion op |  | ut |
| [2536](https://github.com/intel/torch-xpu-ops/issues/2536) | Title: [upstream_ut]  AttributeErro | daisyden | daisyden | P2 | Distributed | Backend/Device Issue - missing attribute '_scatter' in torch._C for XPU device |  | ut |
| [2541](https://github.com/intel/torch-xpu-ops/issues/2541) | Title: [upstream_ut]  RuntimeError: | yucai-intel | daisyden | P2 | Distributed | DNNL/OneDNN Issue - could not construct a memory descriptor using strides |  | ut |
| [2615](https://github.com/intel/torch-xpu-ops/issues/2615) | [Bug Skip]: New failures RuntimeErr | CuiYifeng | kaileiyx | P2 | Distributed | Dtype/Precision Issue - Unsupported dtype Half / torch.float16 |  | ut |
| [2720](https://github.com/intel/torch-xpu-ops/issues/2720) | [upstream_ut] RuntimeError: false I | CuiYifeng | wincent8 | P2 | Distributed | Backend/Device Issue - missing kernel for xpu in DispatchStub |  | ut |
| [2783](https://github.com/intel/torch-xpu-ops/issues/2783) | [Bug Skip]: Key "xpu" is missing fr | daisyden | CuiYifeng | P2 | Dtype / Precision Related | Skip/No Test Exists - Key "xpu" is missing, indicating the test is skipped or no |  | ut |
| [3033](https://github.com/intel/torch-xpu-ops/issues/3033) | [Bug Skip]: Softmax tolerance | chunhuanMeng | chunhuanMeng | P2 | Others | Skip/No Test Exists - test is skipped due to Softmax tolerance issue |  | ut |

#### Close Fixed Issue

| ID | Title | Owner | Owner Transferred | Priority | Category | Root Cause | PR | Test Module |
|---|-------|-------|-------------------|---------|----------|-----------|-----|-------------|
| [3166](https://github.com/intel/torch-xpu-ops/issues/3166) | test_consistency_SparseCSR failures | yucai-intel | CuiYifeng | P2 | TorchAO |  |  | ut |

#### Enable Test

| ID | Title | Owner | Owner Transferred | Priority | Category | Root Cause | PR | Test Module |
|---|-------|-------|-------------------|---------|----------|-----------|-----|-------------|

#### Add to Skiplist

| ID | Title | Owner | Owner Transferred | Priority | Category | Root Cause | PR | Test Module |
|---|-------|-------|-------------------|---------|----------|-----------|-----|-------------|
| [2164](https://github.com/intel/torch-xpu-ops/issues/2164) | skip test_no_cuda_monkeypatch as it | daisyden | daisyden | P2 | Distributed |  |  | ut |
| [2309](https://github.com/intel/torch-xpu-ops/issues/2309) | unsupported ops with PYTORCH_ENABLE | daisyden | daisyden | P2 | TorchAO |  |  | ut |

#### Verify the Issue

| ID | Title | Owner | Owner Transferred | Priority | Category | Root Cause | PR | Test Module |
|---|-------|-------|-------------------|---------|----------|-----------|-----|-------------|
| [2694](https://github.com/intel/torch-xpu-ops/issues/2694) | Title: [upstream_ut]  AssertionErro | daisyden | daisyden | P2 | Distributed |  | [PR](https://github.com/pytorch/pytorch/pull/171773) | ut |
| [3007](https://github.com/intel/torch-xpu-ops/issues/3007) | AssertionError: Scalars are not equ | daisyden | daisyden | P2 | Flash Attention / Transformer Related |  | [PR](https://github.com/pytorch/pytorch/pull/178369) | e2e |

#### Need Reproduce Steps

| ID | Title | Owner | Owner Transferred | Priority | Category | Root Cause | PR | Test Module |
|---|-------|-------|-------------------|---------|----------|-----------|-----|-------------|

### Engineer Actions

#### Needs PyTorch Repo Changes (upstream)

| ID | Title | Owner | Owner Transferred | Priority | Category | Root Cause | PR | Test Module |
|---|-------|-------|-------------------|---------|----------|-----------|-----|-------------|
| [2214](https://github.com/intel/torch-xpu-ops/issues/2214) | test/test_sparse.py::TestSparseAnyX | jenniew | jenniew | P2 | Distributed |  |  | ut |
| [2255](https://github.com/intel/torch-xpu-ops/issues/2255) | [upstream_ut] RuntimeError: Long is | daisyden | daisyden | P2 | Flash Attention / Transformer Related |  |  | ut |
| [2283](https://github.com/intel/torch-xpu-ops/issues/2283) | [upstream_ut] sparse._sampled_addmm | jenniew | jenniew | P1 | Distributed |  |  | ut |
| [2287](https://github.com/intel/torch-xpu-ops/issues/2287) | [upstream_ut] test_python_ref issue | yucai-intel | yucai-intel | P2 | Others |  |  | ut |
| [2295](https://github.com/intel/torch-xpu-ops/issues/2295) | [upstream_ut][xpu][test]nn/test_emb | yucai-intel | yucai-intel | P2 | Dtype / Precision Related |  |  | ut |
| [2301](https://github.com/intel/torch-xpu-ops/issues/2301) | [upstream_ut] dtypes not align with | daisyden | daisyden | P2 | Distributed |  |  | ut |
| [2329](https://github.com/intel/torch-xpu-ops/issues/2329) | [upstream_ut] feature missing: get_ | etaf | etaf | P2 | Others |  |  | ut |
| [2512](https://github.com/intel/torch-xpu-ops/issues/2512) | [upstream_ut]  RuntimeError: _histc | chunhuanMeng | libohao1201 | P2 | Dtype / Precision Related | Backend/Device Issue - _histc_xpu lacks deterministic implementation on XPU back |  | ut |
| [2554](https://github.com/intel/torch-xpu-ops/issues/2554) | [upstream_ut]  AssertionError: Asse | daisyden | daisyden | P2 | PT2E |  |  | ut |
| [2578](https://github.com/intel/torch-xpu-ops/issues/2578) | [TorchAO][UT] test/quantization/tes | Stonepia | Stonepia | P0 | TorchAO |  |  | build |
| [2609](https://github.com/intel/torch-xpu-ops/issues/2609) | [upstream_ut]  torch._inductor.exc. | daisyden | daisyden | P2 | PT2E |  | [PR](https://github.com/pytorch/pytorch/pull/171154) | ut |
| [2620](https://github.com/intel/torch-xpu-ops/issues/2620) | [upstream_ut]  AssertionError: dtyp | daisyden | daisyden | P2 | TorchAO |  |  | ut |
| [2697](https://github.com/intel/torch-xpu-ops/issues/2697) | Title: [upstream_ut]  RuntimeError: | chunhuanMeng | chunhuanMeng | P2 | Flash Attention / Transformer Related |  |  | e2e |
| [2698](https://github.com/intel/torch-xpu-ops/issues/2698) | Title: [upstream_ut]  RuntimeError: | chunhuanMeng, LuFinch | chunhuanMeng, LuFinch | P2 | PT2E |  |  | ut |
| [2715](https://github.com/intel/torch-xpu-ops/issues/2715) | [upstream_ut]  torch._dynamo.exc.Un | CuiYifeng | CuiYifeng | P2 | PT2E |  |  | ut |
| [2800](https://github.com/intel/torch-xpu-ops/issues/2800) | AttributeError: 'torch._C._XpuDevic | guangyey | guangyey | P2 | Flash Attention / Transformer Related |  |  | ut |
| [2802](https://github.com/intel/torch-xpu-ops/issues/2802) | Three aten._scaled_dot_product_flas | LuFinch | LuFinch | P2 | Flash Attention / Transformer Related |  |  | ut |
| [2806](https://github.com/intel/torch-xpu-ops/issues/2806) | CompiledAOTI need XPU support | daisyden | daisyden | P2 | PT2E |  |  | ut |
| [2810](https://github.com/intel/torch-xpu-ops/issues/2810) | AssertionError: Object comparison f | daisyden | daisyden | P2 | Dtype / Precision Related |  |  | ut |
| [2888](https://github.com/intel/torch-xpu-ops/issues/2888) | torch._inductor.exc.InductorError:  | Stonepia | Stonepia | P2 | Inductor / Compilation Related |  |  | ut |
| [2891](https://github.com/intel/torch-xpu-ops/issues/2891) | RuntimeError: Expected to find "(26 | chunhuanMeng | chunhuanMeng | P2 | Others |  |  | e2e |
| [2958](https://github.com/intel/torch-xpu-ops/issues/2958) | AssertionError of test_dtensor_basi | daisyden | daisyden | P2 | Inductor / Compilation Related |  |  | ut |
| [2997](https://github.com/intel/torch-xpu-ops/issues/2997) | AssertionError of test_linear_and_c | etaf | etaf | P2 | Flash Attention / Transformer Related |  |  | ut |
| [2999](https://github.com/intel/torch-xpu-ops/issues/2999) | KeyError: 'eager_numerics.use_pytor | daisyden | daisyden | P2 | Others |  |  | ut |
| [3004](https://github.com/intel/torch-xpu-ops/issues/3004) | TypeError: _xpu_recordMemoryHistory | guangyey | guangyey | P2 | Others |  |  | ut |
| [3006](https://github.com/intel/torch-xpu-ops/issues/3006) | AssertionError: '.to(tl.float16)' u | CuiYifeng | CuiYifeng | P2 | PT2E |  |  | e2e |
| [3126](https://github.com/intel/torch-xpu-ops/issues/3126) | [upstream_ut]  Two NestedTensor iss | LuFinch | LuFinch | P2 | Flash Attention / Transformer Related |  |  | ut |
| [3127](https://github.com/intel/torch-xpu-ops/issues/3127) | [upstream_ut]  AssertionError: Asse | LuFinch | LuFinch | P2 | Flash Attention / Transformer Related |  |  | ut |
| [3128](https://github.com/intel/torch-xpu-ops/issues/3128) | [upstream_ut]  AssertionError: Runt | LuFinch | LuFinch | P2 | Distributed |  |  | ut |
| [3129](https://github.com/intel/torch-xpu-ops/issues/3129) | [upstream_ut]  AssertionError: User | LuFinch | LuFinch | P2 | Flash Attention / Transformer Related |  |  | ut |
| [3131](https://github.com/intel/torch-xpu-ops/issues/3131) | [upstream_ut]  NotImplementedError: | chunhuanMeng | chunhuanMeng | P2 | Flash Attention / Transformer Related |  |  | ut |
| [3132](https://github.com/intel/torch-xpu-ops/issues/3132) | [upstream_ut]  transfomers test rep | LuFinch | LuFinch | P2 | Flash Attention / Transformer Related |  |  | ut |
| [3133](https://github.com/intel/torch-xpu-ops/issues/3133) | [upstream_ut]  RuntimeError: scaled | LuFinch | LuFinch | P2 | Flash Attention / Transformer Related |  |  | ut |
| [3137](https://github.com/intel/torch-xpu-ops/issues/3137) | [upstream_ut]  RuntimeError: expect | LuFinch | LuFinch | P2 | Flash Attention / Transformer Related |  |  | ut |
| [3140](https://github.com/intel/torch-xpu-ops/issues/3140) | [upstream_ut]  RuntimeError: FlashA | LuFinch | LuFinch | P2 | Flash Attention / Transformer Related |  |  | ut |
| [3141](https://github.com/intel/torch-xpu-ops/issues/3141) | [upstream_ut]  RuntimeError: FlashA | LuFinch | LuFinch | P2 | Flash Attention / Transformer Related |  |  | ut |
| [3142](https://github.com/intel/torch-xpu-ops/issues/3142) | [upstream_ut]  RuntimeError: The sy | LuFinch | LuFinch | P2 | Flash Attention / Transformer Related |  |  | ut |
| [3143](https://github.com/intel/torch-xpu-ops/issues/3143) | NotImplementedError: The operator ' | LuFinch | LuFinch | P2 | Flash Attention / Transformer Related |  |  | ut |
| [3163](https://github.com/intel/torch-xpu-ops/issues/3163) | [Bug Skip]: Object comparison faile | chunhuanMeng | chunhuanMeng | P2 | Distributed |  |  | ut |
| [3170](https://github.com/intel/torch-xpu-ops/issues/3170) | Unskip test_bmm_windows_error_xpu_f | jenniew | jenniew | P2 | Others |  |  | ut |
| [3187](https://github.com/intel/torch-xpu-ops/issues/3187) | PyTorch XPU gpu_cpp_wrapper fails w | CuiYifeng | CuiYifeng | P2 | Inductor / Compilation Related |  |  | ut |
| [3238](https://github.com/intel/torch-xpu-ops/issues/3238) | The supported dtypes of _refs.stft  | CuiYifeng | CuiYifeng | P2 | Dtype / Precision Related |  |  | ut |

#### Revisit the PR as Case Failed

| ID | Title | Owner | Owner Transferred | Priority | Category | Root Cause | PR | Test Module |
|---|-------|-------|-------------------|---------|----------|-----------|-----|-------------|

---

## 5. By Category

#### Distributed

| ID | Title | Status | Owner | Priority | Root Cause | PR | Labels | Module | Test Module |
|---|-------|--------|-------|---------|-----------|-----|--------|--------|-------------|
| [1893](https://github.com/intel/torch-xpu-ops/issues/1893) | [upstream_ut] oneDNN accuracy  | open | chunhuanMeng | P2 |  |  | skipped, ut_upstream | aten_ops | ut |
| [2024](https://github.com/intel/torch-xpu-ops/issues/2024) | AssertionError: Torch not comp | open | daisyden | P2 |  |  | module: ut, skipped | aten_ops | ut |
| [2164](https://github.com/intel/torch-xpu-ops/issues/2164) | skip test_no_cuda_monkeypatch  | open | daisyden | P2 |  |  | wontfix, skipped | aten_ops | ut |
| [2169](https://github.com/intel/torch-xpu-ops/issues/2169) | Frame size comparison failed i | open | guangyey | P2 | Failure - Scalars are not equal in test comparison |  | skipped | aten_ops | ut |
| [2214](https://github.com/intel/torch-xpu-ops/issues/2214) | test/test_sparse.py::TestSpars | open | jenniew | P2 |  |  | skipped, ut_upstream | aten_ops | ut |
| [2229](https://github.com/intel/torch-xpu-ops/issues/2229) | test/test_sparse_csr.py::TestS | open | jenniew | P2 | Backend/Device Issue - device mismatch between crow_indices and col_indices on X |  | skipped | aten_ops | ut |
| [2244](https://github.com/intel/torch-xpu-ops/issues/2244) | test/test_sparse_csr.py::TestS | open | jenniew | P2 | Error - empty_sparse_compressed expects non-block layout but received SparseBsr |  | module: ut, skipped | aten_ops | ut |
| [2245](https://github.com/intel/torch-xpu-ops/issues/2245) | oneDNN matmul received incorre | open | CuiYifeng | P2 | DNNL/OneDNN Issue - oneDNN matmul primitive descriptor creation failure |  | module: ut, skipped | aten_ops | ut |
| [2270](https://github.com/intel/torch-xpu-ops/issues/2270) | Backend Compatibility Error in | open | LuFinch | P2 | Flash Attention/Specific Ops Issue - test involves aten._flash_attention_forward |  | module: ut, skipped | aten_ops | ut |
| [2283](https://github.com/intel/torch-xpu-ops/issues/2283) | [upstream_ut] sparse._sampled_ | open | jenniew | P1 |  |  | skipped, ut_upstream | aten_ops | ut |
| [2301](https://github.com/intel/torch-xpu-ops/issues/2301) | [upstream_ut] dtypes not align | open | daisyden | P2 |  |  | skipped, ut_upstream | aten_ops | ut |
| [2531](https://github.com/intel/torch-xpu-ops/issues/2531) | [upstream_ut]  AssertionError: | open | daisyden | P2 |  |  | skipped, port_from_skiplist | unknown | ut |
| [2532](https://github.com/intel/torch-xpu-ops/issues/2532) | Title: [upstream_ut]  Assertio | open | yucai-intel | P2 | Failure - wrong number of dimensions for int4 conversion op |  | skipped, port_from_skiplist | aten_ops | ut |
| [2536](https://github.com/intel/torch-xpu-ops/issues/2536) | Title: [upstream_ut]  Attribut | open | daisyden | P2 | Backend/Device Issue - missing attribute '_scatter' in torch._C for XPU device |  | skipped, port_from_skiplist, not_target | aten_ops | ut |
| [2541](https://github.com/intel/torch-xpu-ops/issues/2541) | Title: [upstream_ut]  RuntimeE | open | yucai-intel | P2 | DNNL/OneDNN Issue - could not construct a memory descriptor using strides |  | skipped, port_from_skiplist | aten_ops | ut |
| [2611](https://github.com/intel/torch-xpu-ops/issues/2611) | [upstream_ut]  AssertionError: | open | daisyden | P2 |  |  | dependency component: driver, module: inductor, skipped, ut_upstream | inductor | ut |
| [2613](https://github.com/intel/torch-xpu-ops/issues/2613) | [upstream_ut]  AssertionError: | open | daisyden | P2 |  |  | dependency component: driver, module: inductor, skipped, ut_upstream | inductor | ut |
| [2615](https://github.com/intel/torch-xpu-ops/issues/2615) | [Bug Skip]: New failures Runti | open | CuiYifeng | P2 | Dtype/Precision Issue - Unsupported dtype Half / torch.float16 |  | module: ut, skipped | aten_ops | ut |
| [2694](https://github.com/intel/torch-xpu-ops/issues/2694) | Title: [upstream_ut]  Assertio | open | daisyden | P2 |  | [PR](https://github.com/pytorch/pytorch/pull/171773) | module: inductor, skipped, ut_upstream | inductor | ut |
| [2720](https://github.com/intel/torch-xpu-ops/issues/2720) | [upstream_ut] RuntimeError: fa | open | CuiYifeng | P2 | Backend/Device Issue - missing kernel for xpu in DispatchStub |  | skipped | aten_ops | ut |
| [2853](https://github.com/intel/torch-xpu-ops/issues/2853) | [upstream_ut] torch.ops.aten._ | open | LuFinch | P2 | Flash Attention/Specific Ops Issue - _flash_attention_forward not supported on X |  | skipped | aten_ops | ut |
| [3128](https://github.com/intel/torch-xpu-ops/issues/3128) | [upstream_ut]  AssertionError: | open | LuFinch | P2 |  |  | module: ut, skipped, ut_upstream | aten_ops | ut |
| [3163](https://github.com/intel/torch-xpu-ops/issues/3163) | [Bug Skip]: Object comparison  | open | chunhuanMeng | P2 |  |  | skipped, ut_upstream | aten_ops | ut |
| [3177](https://github.com/intel/torch-xpu-ops/issues/3177) | Accuracy gap of BF16/FP16 test | open | jenniew | P2 | Dtype/Precision Issue - Accuracy gap in BF16/FP16 test |  | skipped | aten_ops | ut |

#### Dtype / Precision Related

| ID | Title | Status | Owner | Priority | Root Cause | PR | Labels | Module | Test Module |
|---|-------|--------|-------|---------|-----------|-----|--------|--------|-------------|
| [2253](https://github.com/intel/torch-xpu-ops/issues/2253) | the supported dtypes are not a | open | daisyden | P2 |  |  | duplicate, skipped, ut_upstream | aten_ops | ut |
| [2295](https://github.com/intel/torch-xpu-ops/issues/2295) | [upstream_ut][xpu][test]nn/tes | open | yucai-intel | P2 |  |  | module: inductor, skipped, ut_upstream | inductor | ut |
| [2376](https://github.com/intel/torch-xpu-ops/issues/2376) | [Bug Skip]: NotImplementedErro | open | CuiYifeng | P2 | Dtype/Precision Issue - "logaddexp_xpu" not implemented for 'Complex' dtype |  | module: ut, skipped | aten_ops | ut |
| [2482](https://github.com/intel/torch-xpu-ops/issues/2482) | test_dtypes issue introduced b | open | daisyden | P2 | Dtype/Precision Issue - test_dtypes issue related to data types in conv_transpos |  | skipped | aten_ops | ut |
| [2512](https://github.com/intel/torch-xpu-ops/issues/2512) | [upstream_ut]  RuntimeError: _ | open | chunhuanMeng | P2 | Backend/Device Issue - _histc_xpu lacks deterministic implementation on XPU back |  | skipped | aten_ops | ut |
| [2783](https://github.com/intel/torch-xpu-ops/issues/2783) | [Bug Skip]: Key "xpu" is missi | open | daisyden | P2 | Skip/No Test Exists - Key "xpu" is missing, indicating the test is skipped or no |  | module: ut, skipped | aten_ops | ut |
| [2810](https://github.com/intel/torch-xpu-ops/issues/2810) | AssertionError: Object compari | open | daisyden | P2 |  |  | module: inductor, ut_upstream | inductor | ut |
| [3238](https://github.com/intel/torch-xpu-ops/issues/3238) | The supported dtypes of _refs. | open | CuiYifeng | P2 |  |  | ut_upstream | aten_ops | ut |

#### Flash Attention / Transformer Related

| ID | Title | Status | Owner | Priority | Root Cause | PR | Labels | Module | Test Module |
|---|-------|--------|-------|---------|-----------|-----|--------|--------|-------------|
| [2015](https://github.com/intel/torch-xpu-ops/issues/2015) | inf is returned by nn.Transfor | open | yucai-intel | P2 | Dtype/Precision Issue - inf returned using float16 in TransformerEncoderLayer on |  | skipped | aten_ops | ut |
| [2255](https://github.com/intel/torch-xpu-ops/issues/2255) | [upstream_ut] RuntimeError: Lo | open | daisyden | P2 |  |  | skipped, ut_upstream | aten_ops | ut |
| [2285](https://github.com/intel/torch-xpu-ops/issues/2285) | Support efficient attention | open | chunhuanMeng | P2 | Backend/Device Issue - aten::_efficient_attention_forward not supported on CPU b |  | skipped | aten_ops | ut |
| [2436](https://github.com/intel/torch-xpu-ops/issues/2436) | [upstream_ut]  AttributeError: | open | daisyden | P1 | Error - 'NoneType' object has no attribute 'clone' due to missing object handling |  | skipped, dependency component: community, random | aten_ops | ut |
| [2442](https://github.com/intel/torch-xpu-ops/issues/2442) | [Bug Skip]: NotImplementedErro | open | daisyden, LuFinch | P2 | Flash Attention/Specific Ops Issue - aten::_flash_attention_forward not implemen |  | skipped, not_target | aten_ops | ut |
| [2697](https://github.com/intel/torch-xpu-ops/issues/2697) | Title: [upstream_ut]  RuntimeE | open | chunhuanMeng | P2 |  |  | module: inductor, skipped, ut_upstream | inductor | e2e |
| [2800](https://github.com/intel/torch-xpu-ops/issues/2800) | AttributeError: 'torch._C._Xpu | open | guangyey | P2 |  |  | dependency component: oneAPI, module: inductor, ut_upstream | inductor | ut |
| [2802](https://github.com/intel/torch-xpu-ops/issues/2802) | Three aten._scaled_dot_product | open | LuFinch | P2 |  |  | module: inductor, ut_upstream | inductor | ut |
| [2997](https://github.com/intel/torch-xpu-ops/issues/2997) | AssertionError of test_linear_ | open | etaf | P2 |  |  | module: inductor, ut_upstream | inductor | ut |
| [3007](https://github.com/intel/torch-xpu-ops/issues/3007) | AssertionError: Scalars are no | open | daisyden | P2 |  | [PR](https://github.com/pytorch/pytorch/pull/178369) | module: inductor, ut_upstream | inductor | e2e |
| [3126](https://github.com/intel/torch-xpu-ops/issues/3126) | [upstream_ut]  Two NestedTenso | open | LuFinch | P2 |  |  | module: ut, skipped, ut_upstream | aten_ops | ut |
| [3127](https://github.com/intel/torch-xpu-ops/issues/3127) | [upstream_ut]  AssertionError: | open | LuFinch | P2 |  |  | module: ut, skipped, ut_upstream | aten_ops | ut |
| [3129](https://github.com/intel/torch-xpu-ops/issues/3129) | [upstream_ut]  AssertionError: | open | LuFinch | P2 |  |  | module: ut, skipped, ut_upstream | aten_ops | ut |
| [3131](https://github.com/intel/torch-xpu-ops/issues/3131) | [upstream_ut]  NotImplementedE | open | chunhuanMeng | P2 |  |  | module: ut, skipped, ut_upstream | aten_ops | ut |
| [3132](https://github.com/intel/torch-xpu-ops/issues/3132) | [upstream_ut]  transfomers tes | open | LuFinch | P2 |  |  | module: ut, skipped, ut_upstream | aten_ops | ut |
| [3133](https://github.com/intel/torch-xpu-ops/issues/3133) | [upstream_ut]  RuntimeError: s | open | LuFinch | P2 |  |  | module: ut, skipped, ut_upstream | aten_ops | ut |
| [3136](https://github.com/intel/torch-xpu-ops/issues/3136) | [upstream_ut]  AssertionError: | open | LuFinch | P2 |  |  | module: ut, skipped, ut_upstream | aten_ops | ut |
| [3137](https://github.com/intel/torch-xpu-ops/issues/3137) | [upstream_ut]  RuntimeError: e | open | LuFinch | P2 |  |  | module: ut, skipped, ut_upstream, random | aten_ops | ut |
| [3140](https://github.com/intel/torch-xpu-ops/issues/3140) | [upstream_ut]  RuntimeError: F | open | LuFinch | P2 |  |  | module: ut, skipped, ut_upstream | aten_ops | ut |
| [3141](https://github.com/intel/torch-xpu-ops/issues/3141) | [upstream_ut]  RuntimeError: F | open | LuFinch | P2 |  |  | module: ut, skipped, ut_upstream | aten_ops | ut |
| [3142](https://github.com/intel/torch-xpu-ops/issues/3142) | [upstream_ut]  RuntimeError: T | open | LuFinch | P2 |  |  | module: ut, skipped, ut_upstream | aten_ops | ut |
| [3143](https://github.com/intel/torch-xpu-ops/issues/3143) | NotImplementedError: The opera | open | LuFinch | P2 |  |  | module: ut, skipped, ut_upstream | aten_ops | ut |

#### Inductor / Compilation Related

| ID | Title | Status | Owner | Priority | Root Cause | PR | Labels | Module | Test Module |
|---|-------|--------|-------|---------|-----------|-----|--------|--------|-------------|
| [2888](https://github.com/intel/torch-xpu-ops/issues/2888) | torch._inductor.exc.InductorEr | open | Stonepia | P2 |  |  | module: inductor, ut_upstream | inductor | ut |
| [2958](https://github.com/intel/torch-xpu-ops/issues/2958) | AssertionError of test_dtensor | open | daisyden | P2 |  |  | module: inductor, ut_upstream | inductor | ut |
| [3187](https://github.com/intel/torch-xpu-ops/issues/3187) | PyTorch XPU gpu_cpp_wrapper fa | open | CuiYifeng | P2 |  |  | ut_upstream | aten_ops | ut |

#### Others

| ID | Title | Status | Owner | Priority | Root Cause | PR | Labels | Module | Test Module |
|---|-------|--------|-------|---------|-----------|-----|--------|--------|-------------|
| [2287](https://github.com/intel/torch-xpu-ops/issues/2287) | [upstream_ut] test_python_ref  | open | yucai-intel | P2 |  |  | module: ut, ut_upstream | aten_ops | ut |
| [2329](https://github.com/intel/torch-xpu-ops/issues/2329) | [upstream_ut] feature missing: | open | etaf | P2 |  |  | duplicate, dependency component: Triton, module: inductor, ut_upstream | inductor | ut |
| [2891](https://github.com/intel/torch-xpu-ops/issues/2891) | RuntimeError: Expected to find | open | chunhuanMeng | P2 |  |  | module: inductor, ut_upstream | inductor | e2e |
| [2999](https://github.com/intel/torch-xpu-ops/issues/2999) | KeyError: 'eager_numerics.use_ | open | daisyden | P2 |  |  | module: inductor, ut_upstream | inductor | ut |
| [3004](https://github.com/intel/torch-xpu-ops/issues/3004) | TypeError: _xpu_recordMemoryHi | open | guangyey | P2 |  |  | module: inductor, ut_upstream | inductor | ut |
| [3033](https://github.com/intel/torch-xpu-ops/issues/3033) | [Bug Skip]: Softmax tolerance | open | chunhuanMeng | P2 | Skip/No Test Exists - test is skipped due to Softmax tolerance issue |  | skipped, random | aten_ops | ut |
| [3170](https://github.com/intel/torch-xpu-ops/issues/3170) | Unskip test_bmm_windows_error_ | open | jenniew | P2 |  |  | skipped, ut_upstream | aten_ops | ut |

#### PT2E

| ID | Title | Status | Owner | Priority | Root Cause | PR | Labels | Module | Test Module |
|---|-------|--------|-------|---------|-----------|-----|--------|--------|-------------|
| [2554](https://github.com/intel/torch-xpu-ops/issues/2554) | [upstream_ut]  AssertionError: | open | daisyden | P2 |  |  | module: inductor, skipped | inductor | ut |
| [2609](https://github.com/intel/torch-xpu-ops/issues/2609) | [upstream_ut]  torch._inductor | open | daisyden | P2 |  | [PR](https://github.com/pytorch/pytorch/pull/171154) | module: inductor, skipped, ut_upstream | inductor | ut |
| [2698](https://github.com/intel/torch-xpu-ops/issues/2698) | Title: [upstream_ut]  RuntimeE | open | chunhuanMeng, LuFinch | P2 |  |  | module: inductor, skipped, ut_upstream | inductor | ut |
| [2715](https://github.com/intel/torch-xpu-ops/issues/2715) | [upstream_ut]  torch._dynamo.e | open | CuiYifeng | P2 |  |  | skipped, ut_upstream | aten_ops | ut |
| [2806](https://github.com/intel/torch-xpu-ops/issues/2806) | CompiledAOTI need XPU support | open | daisyden | P2 |  |  | module: inductor, ut_upstream | inductor | ut |
| [3006](https://github.com/intel/torch-xpu-ops/issues/3006) | AssertionError: '.to(tl.float1 | open | CuiYifeng | P2 |  |  | module: inductor, ut_upstream | inductor | e2e |

#### Sparse Operations Related

| ID | Title | Status | Owner | Priority | Root Cause | PR | Labels | Module | Test Module |
|---|-------|--------|-------|---------|-----------|-----|--------|--------|-------------|
| [1962](https://github.com/intel/torch-xpu-ops/issues/1962) | [upstream_ut] segfault with te | open | jenniew, mengfei25 | P0 | Backend/Device Issue - segfault related to XPU device operation in test |  | dependency component: driver, module: ut, skipped | aten_ops | ut |

#### TorchAO

| ID | Title | Status | Owner | Priority | Root Cause | PR | Labels | Module | Test Module |
|---|-------|--------|-------|---------|-----------|-----|--------|--------|-------------|
| [2309](https://github.com/intel/torch-xpu-ops/issues/2309) | unsupported ops with PYTORCH_E | open | daisyden | P2 |  |  | wontfix, module: op impl, skipped | aten_ops | ut |
| [2578](https://github.com/intel/torch-xpu-ops/issues/2578) | [TorchAO][UT] test/quantizatio | open | Stonepia | P0 |  |  | module: ao, ut_upstream | AO | build |
| [2620](https://github.com/intel/torch-xpu-ops/issues/2620) | [upstream_ut]  AssertionError: | open | daisyden | P2 |  |  | module: inductor, skipped, ut_upstream | inductor | ut |
| [3166](https://github.com/intel/torch-xpu-ops/issues/3166) | test_consistency_SparseCSR fai | open | yucai-intel | P2 |  |  | skipped, ut_upstream | aten_ops | ut |


---

## 6. Duplicated Issues

Issues that share test cases with other issues.

| ID | Title | Owner | Reporter | Duplicated With | Priority | Root Cause | PR | Labels | Module | Test Module |
|---|-------|-------|----------|-----------------|---------|-----------|-----|--------|--------|-------------|
| [1893](https://github.com/intel/torch-xpu-ops/issues/1893) | [upstream_ut] oneDNN accuracy  | chunhuanMeng | daisyden | 1951 | P2 |  |  | skipped, ut_upstream | aten_ops | ut |
| [2015](https://github.com/intel/torch-xpu-ops/issues/2015) | inf is returned by nn.Transfor | yucai-intel | daisyden | 2186,2529 | P2 | Dtype/Precision Issue - inf returned using float16 in TransformerEncoderLayer on |  | skipped | aten_ops | ut |
| [2244](https://github.com/intel/torch-xpu-ops/issues/2244) | test/test_sparse_csr.py::TestS | jenniew | wincent8 | 3177 | P2 | Error - empty_sparse_compressed expects non-block layout but received SparseBsr |  | module: ut, skipped | aten_ops | ut |
| [2253](https://github.com/intel/torch-xpu-ops/issues/2253) | the supported dtypes are not a | daisyden | daisyden | 2482 | P2 |  |  | duplicate, skipped, ut_upstream | aten_ops | ut |
| [2270](https://github.com/intel/torch-xpu-ops/issues/2270) | Backend Compatibility Error in | LuFinch | libohao1201 | 2442 | P2 | Flash Attention/Specific Ops Issue - test involves aten._flash_attention_forward |  | module: ut, skipped | aten_ops | ut |
| [2285](https://github.com/intel/torch-xpu-ops/issues/2285) | Support efficient attention | chunhuanMeng | daisyden | 2358 | P2 | Backend/Device Issue - aten::_efficient_attention_forward not supported on CPU b |  | skipped | aten_ops | ut |
| [2436](https://github.com/intel/torch-xpu-ops/issues/2436) | [upstream_ut]  AttributeError: | daisyden | daisyden | 2675 | P1 | Error - 'NoneType' object has no attribute 'clone' due to missing object handling |  | skipped, dependency component: community, random | aten_ops | ut |
| [2442](https://github.com/intel/torch-xpu-ops/issues/2442) | [Bug Skip]: NotImplementedErro | daisyden, LuFinch | CuiYifeng | 2270 | P2 | Flash Attention/Specific Ops Issue - aten::_flash_attention_forward not implemen |  | skipped, not_target | aten_ops | ut |
| [2482](https://github.com/intel/torch-xpu-ops/issues/2482) | test_dtypes issue introduced b | daisyden | daisyden | 2253 | P2 | Dtype/Precision Issue - test_dtypes issue related to data types in conv_transpos |  | skipped | aten_ops | ut |
| [2611](https://github.com/intel/torch-xpu-ops/issues/2611) | [upstream_ut]  AssertionError: | daisyden | daisyden | 2613 | P2 |  |  | dependency component: driver, module: inductor, skipped, ut_upstream | inductor | ut |
| [2613](https://github.com/intel/torch-xpu-ops/issues/2613) | [upstream_ut]  AssertionError: | daisyden | daisyden | 2611 | P2 |  |  | dependency component: driver, module: inductor, skipped, ut_upstream | inductor | ut |
| [3136](https://github.com/intel/torch-xpu-ops/issues/3136) | [upstream_ut]  AssertionError: | LuFinch | daisyden | 2529 | P2 |  |  | module: ut, skipped, ut_upstream | aten_ops | ut |
| [3177](https://github.com/intel/torch-xpu-ops/issues/3177) | Accuracy gap of BF16/FP16 test | jenniew | CuiYifeng | 2244 | P2 | Dtype/Precision Issue - Accuracy gap in BF16/FP16 test |  | skipped | aten_ops | ut |

---

## 7. Issues with Dependency

Issues that have dependencies on other components.

| ID | Title | Owner | Priority | Root Cause | Dependency | Category | PR | Labels | Test Module |
|---|-------|-------|---------|-----------|------------|----------|-----|--------|-------------|
