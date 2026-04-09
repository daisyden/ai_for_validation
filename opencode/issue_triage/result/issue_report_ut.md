# Torch XPU Ops UT Issue Report (Custom Filtered)

**Repository:** [intel/torch-xpu-ops](https://github.com/intel/torch-xpu-ops)

**Report Type:** UT (Unit Test) Issues - Custom Filtered List

**Generated:** 2026-04-08 22:56:17

---

## 1. Summary {#1-summary}

| Total Issues | 75 |
|-------------|-------------|

| Count by Action TBD | |
|---------------------|---|
| Needs PyTorch Repo Changes (upstream) | 48 |
| Need reproduce steps (Only for bugs or performance issue) | 16 |
| Need more information - error logs and reproduction steps | 5 |
| add to skiplist | 2 |
| Verify the issue | 2 |
| Close fixed issue | 1 |

| Count by Category | |
|-------------------|---|
| Distributed | 24 |
| Flash Attention / Transformer Related | 22 |
| Dtype / Precision Related | 8 |
| Others | 7 |
| PT2E | 6 |
| TorchAO | 4 |
| Inductor / Compilation Related | 3 |
| Sparse Operations Related | 1 |

| Count by Priority | |
|-------------------|---|
| P0 | 2 |
| P1 | 2 |
| P2 | 71 |

---

## 2. Detailed Issue List

| ID | Title | Owner | Priority | Category | Root Cause | Labels | Module |
|---|-------|-------|---------|----------|-----------|--------|--------|
| [1893](https://github.com/intel/torch-xpu-ops/issues/1893) | [upstream_ut] oneDNN accuracy issues in  | chunhuanMeng | P2 | Distributed |  | skipped, ut_upstream | aten_ops |
| [1962](https://github.com/intel/torch-xpu-ops/issues/1962) | [upstream_ut] segfault with test_fake_cr | jenniew, mengfei25 | P0 | Sparse Operations Related | Backend/Device Issue - segfault related to XPU device operation in test | dependency component.. | aten_ops |
| [2015](https://github.com/intel/torch-xpu-ops/issues/2015) | inf is returned by nn.TransformerEncoder | yucai-intel | P2 | Flash Attention / Transformer Related | Dtype/Precision Issue - inf returned using float16 in TransformerEncoderLayer on | skipped | aten_ops |
| [2024](https://github.com/intel/torch-xpu-ops/issues/2024) | AssertionError: Torch not compiled with  | daisyden | P2 | Distributed |  | module: ut, skipped | aten_ops |
| [2164](https://github.com/intel/torch-xpu-ops/issues/2164) | skip test_no_cuda_monkeypatch as it is c | daisyden | P2 | Distributed |  | wontfix, skipped | aten_ops |
| [2169](https://github.com/intel/torch-xpu-ops/issues/2169) | Frame size comparison failed in test_siz | guangyey | P2 | Distributed | Failure - Scalars are not equal in test comparison | skipped | aten_ops |
| [2214](https://github.com/intel/torch-xpu-ops/issues/2214) | test/test_sparse.py::TestSparseAnyXPU::t | jenniew | P2 | Distributed |  | skipped, ut_upstream | aten_ops |
| [2229](https://github.com/intel/torch-xpu-ops/issues/2229) | test/test_sparse_csr.py::TestSparseCompr | jenniew | P2 | Distributed | Backend/Device Issue - device mismatch between crow_indices and col_indices on X | skipped | aten_ops |
| [2244](https://github.com/intel/torch-xpu-ops/issues/2244) | test/test_sparse_csr.py::TestSparseCSRXP | jenniew | P2 | Distributed | Error - empty_sparse_compressed expects non-block layout but received SparseBsr | module: ut, skipped | aten_ops |
| [2245](https://github.com/intel/torch-xpu-ops/issues/2245) | oneDNN matmul received incorrect shape i | CuiYifeng | P2 | Distributed | DNNL/OneDNN Issue - oneDNN matmul primitive descriptor creation failure | module: ut, skipped | aten_ops |
| [2253](https://github.com/intel/torch-xpu-ops/issues/2253) | the supported dtypes are not align with  | daisyden | P2 | Dtype / Precision Related |  | duplicate, skipped, .. | aten_ops |
| [2255](https://github.com/intel/torch-xpu-ops/issues/2255) | [upstream_ut] RuntimeError: Long is not  | daisyden | P2 | Flash Attention / Transformer Related |  | skipped, ut_upstream | aten_ops |
| [2270](https://github.com/intel/torch-xpu-ops/issues/2270) | Backend Compatibility Error in test/xpu/ | LuFinch | P2 | Distributed | Flash Attention/Specific Ops Issue - test involves aten._flash_attention_forward | module: ut, skipped | aten_ops |
| [2283](https://github.com/intel/torch-xpu-ops/issues/2283) | [upstream_ut] sparse._sampled_addmm is n | jenniew | P1 | Distributed |  | skipped, ut_upstream | aten_ops |
| [2285](https://github.com/intel/torch-xpu-ops/issues/2285) | Support efficient attention | chunhuanMeng | P2 | Flash Attention / Transformer Related | Backend/Device Issue - aten::_efficient_attention_forward not supported on CPU b | skipped | aten_ops |
| [2287](https://github.com/intel/torch-xpu-ops/issues/2287) | [upstream_ut] test_python_ref issues | yucai-intel | P2 | Others |  | module: ut, ut_upstr.. | aten_ops |
| [2295](https://github.com/intel/torch-xpu-ops/issues/2295) | [upstream_ut][xpu][test]nn/test_embeddin | yucai-intel | P2 | Dtype / Precision Related |  | module: inductor, sk.. | inductor |
| [2301](https://github.com/intel/torch-xpu-ops/issues/2301) | [upstream_ut] dtypes not align with OpIn | daisyden | P2 | Distributed |  | skipped, ut_upstream | aten_ops |
| [2309](https://github.com/intel/torch-xpu-ops/issues/2309) | unsupported ops with PYTORCH_ENABLE_XPU_ | daisyden | P2 | TorchAO |  | wontfix, module: op .. | aten_ops |
| [2329](https://github.com/intel/torch-xpu-ops/issues/2329) | [upstream_ut] feature missing: get_devic | etaf | P2 | Others |  | duplicate, dependenc.. | inductor |
| [2376](https://github.com/intel/torch-xpu-ops/issues/2376) | [Bug Skip]: NotImplementedError: "logadd | CuiYifeng | P2 | Dtype / Precision Related | Dtype/Precision Issue - "logaddexp_xpu" not implemented for 'Complex' dtype | module: ut, skipped | aten_ops |
| [2436](https://github.com/intel/torch-xpu-ops/issues/2436) | [upstream_ut]  AttributeError: 'NoneType | daisyden | P1 | Flash Attention / Transformer Related | Error - 'NoneType' object has no attribute 'clone' due to missing object handling | skipped, dependency .. | aten_ops |
| [2442](https://github.com/intel/torch-xpu-ops/issues/2442) | [Bug Skip]: NotImplementedError: Could n | daisyden, LuFinch | P2 | Flash Attention / Transformer Related | Flash Attention/Specific Ops Issue - aten::_flash_attention_forward not implemen | skipped, not_target | aten_ops |
| [2482](https://github.com/intel/torch-xpu-ops/issues/2482) | test_dtypes issue introduced by pytorch  | daisyden | P2 | Dtype / Precision Related | Dtype/Precision Issue - test_dtypes issue related to data types in conv_transpos | skipped | aten_ops |
| [2512](https://github.com/intel/torch-xpu-ops/issues/2512) | [upstream_ut]  RuntimeError: _histc_xpu  | chunhuanMeng | P2 | Dtype / Precision Related | Backend/Device Issue - _histc_xpu lacks deterministic implementation on XPU back | skipped | aten_ops |
| [2531](https://github.com/intel/torch-xpu-ops/issues/2531) | [upstream_ut]  AssertionError: Torch not | daisyden | P2 | Distributed |  | skipped, port_from_s.. | unknown |
| [2532](https://github.com/intel/torch-xpu-ops/issues/2532) | Title: [upstream_ut]  AssertionError: wr | yucai-intel | P2 | Distributed | Failure - wrong number of dimensions for int4 conversion op | skipped, port_from_s.. | aten_ops |
| [2536](https://github.com/intel/torch-xpu-ops/issues/2536) | Title: [upstream_ut]  AttributeError: mo | daisyden | P2 | Distributed | Backend/Device Issue - missing attribute '_scatter' in torch._C for XPU device | skipped, port_from_s.. | aten_ops |
| [2541](https://github.com/intel/torch-xpu-ops/issues/2541) | Title: [upstream_ut]  RuntimeError: coul | yucai-intel | P2 | Distributed | DNNL/OneDNN Issue - could not construct a memory descriptor using strides | skipped, port_from_s.. | aten_ops |
| [2554](https://github.com/intel/torch-xpu-ops/issues/2554) | [upstream_ut]  AssertionError: Assertion | daisyden | P2 | PT2E |  | module: inductor, sk.. | inductor |
| [2578](https://github.com/intel/torch-xpu-ops/issues/2578) | [TorchAO][UT] test/quantization/test_qua | Stonepia | P0 | TorchAO |  | module: ao, ut_upstr.. | AO |
| [2609](https://github.com/intel/torch-xpu-ops/issues/2609) | [upstream_ut]  torch._inductor.exc.Induc | daisyden | P2 | PT2E |  | module: inductor, sk.. | inductor |
| [2611](https://github.com/intel/torch-xpu-ops/issues/2611) | [upstream_ut]  AssertionError: Tensor-li | daisyden | P2 | Distributed |  | dependency component.. | inductor |
| [2613](https://github.com/intel/torch-xpu-ops/issues/2613) | [upstream_ut]  AssertionError: Tensor-li | daisyden | P2 | Distributed |  | dependency component.. | inductor |
| [2615](https://github.com/intel/torch-xpu-ops/issues/2615) | [Bug Skip]: New failures RuntimeError: U | CuiYifeng | P2 | Distributed | Dtype/Precision Issue - Unsupported dtype Half / torch.float16 | module: ut, skipped | aten_ops |
| [2620](https://github.com/intel/torch-xpu-ops/issues/2620) | [upstream_ut]  AssertionError: dtype is  | daisyden | P2 | TorchAO |  | module: inductor, sk.. | inductor |
| [2694](https://github.com/intel/torch-xpu-ops/issues/2694) | Title: [upstream_ut]  AssertionError: Te | daisyden | P2 | Distributed |  | module: inductor, sk.. | inductor |
| [2697](https://github.com/intel/torch-xpu-ops/issues/2697) | Title: [upstream_ut]  RuntimeError: Expe | chunhuanMeng | P2 | Flash Attention / Transformer Related |  | module: inductor, sk.. | inductor |
| [2698](https://github.com/intel/torch-xpu-ops/issues/2698) | Title: [upstream_ut]  RuntimeError: Flas | chunhuanMeng, LuFinch | P2 | PT2E |  | module: inductor, sk.. | inductor |
| [2715](https://github.com/intel/torch-xpu-ops/issues/2715) | [upstream_ut]  torch._dynamo.exc.Unsuppo | CuiYifeng | P2 | PT2E |  | skipped, ut_upstream | aten_ops |
| [2720](https://github.com/intel/torch-xpu-ops/issues/2720) | [upstream_ut] RuntimeError: false INTERN | CuiYifeng | P2 | Distributed | Backend/Device Issue - missing kernel for xpu in DispatchStub | skipped | aten_ops |
| [2783](https://github.com/intel/torch-xpu-ops/issues/2783) | [Bug Skip]: Key "xpu" is missing from di | daisyden | P2 | Dtype / Precision Related | Skip/No Test Exists - Key "xpu" is missing, indicating the test is skipped or no | module: ut, skipped | aten_ops |
| [2800](https://github.com/intel/torch-xpu-ops/issues/2800) | AttributeError: 'torch._C._XpuDeviceProp | guangyey | P2 | Flash Attention / Transformer Related |  | dependency component.. | inductor |
| [2802](https://github.com/intel/torch-xpu-ops/issues/2802) | Three aten._scaled_dot_product_flash_att | LuFinch | P2 | Flash Attention / Transformer Related |  | module: inductor, ut.. | inductor |
| [2806](https://github.com/intel/torch-xpu-ops/issues/2806) | CompiledAOTI need XPU support | daisyden | P2 | PT2E |  | module: inductor, ut.. | inductor |
| [2810](https://github.com/intel/torch-xpu-ops/issues/2810) | AssertionError: Object comparison failed | daisyden | P2 | Dtype / Precision Related |  | module: inductor, ut.. | inductor |
| [2853](https://github.com/intel/torch-xpu-ops/issues/2853) | [upstream_ut] torch.ops.aten._flash_atte | LuFinch | P2 | Distributed | Flash Attention/Specific Ops Issue - _flash_attention_forward not supported on X | skipped | aten_ops |
| [2888](https://github.com/intel/torch-xpu-ops/issues/2888) | torch._inductor.exc.InductorError: Asser | Stonepia | P2 | Inductor / Compilation Related |  | module: inductor, ut.. | inductor |
| [2891](https://github.com/intel/torch-xpu-ops/issues/2891) | RuntimeError: Expected to find "(262144, | chunhuanMeng | P2 | Others |  | module: inductor, ut.. | inductor |
| [2958](https://github.com/intel/torch-xpu-ops/issues/2958) | AssertionError of test_dtensor_basic_com | daisyden | P2 | Inductor / Compilation Related |  | module: inductor, ut.. | inductor |
| [2997](https://github.com/intel/torch-xpu-ops/issues/2997) | AssertionError of test_linear_and_cel_ma | etaf | P2 | Flash Attention / Transformer Related |  | module: inductor, ut.. | inductor |
| [2999](https://github.com/intel/torch-xpu-ops/issues/2999) | KeyError: 'eager_numerics.use_pytorch_li | daisyden | P2 | Others |  | module: inductor, ut.. | inductor |
| [3004](https://github.com/intel/torch-xpu-ops/issues/3004) | TypeError: _xpu_recordMemoryHistory(): i | guangyey | P2 | Others |  | module: inductor, ut.. | inductor |
| [3006](https://github.com/intel/torch-xpu-ops/issues/3006) | AssertionError: '.to(tl.float16)' unexpe | CuiYifeng | P2 | PT2E |  | module: inductor, ut.. | inductor |
| [3007](https://github.com/intel/torch-xpu-ops/issues/3007) | AssertionError: Scalars are not equal! w | daisyden | P2 | Flash Attention / Transformer Related |  | module: inductor, ut.. | inductor |
| [3033](https://github.com/intel/torch-xpu-ops/issues/3033) | [Bug Skip]: Softmax tolerance | chunhuanMeng | P2 | Others | Skip/No Test Exists - test is skipped due to Softmax tolerance issue | skipped, random | aten_ops |
| [3126](https://github.com/intel/torch-xpu-ops/issues/3126) | [upstream_ut]  Two NestedTensor issue wi | LuFinch | P2 | Flash Attention / Transformer Related |  | module: ut, skipped,.. | aten_ops |
| [3127](https://github.com/intel/torch-xpu-ops/issues/3127) | [upstream_ut]  AssertionError: Assertion | LuFinch | P2 | Flash Attention / Transformer Related |  | module: ut, skipped,.. | aten_ops |
| [3128](https://github.com/intel/torch-xpu-ops/issues/3128) | [upstream_ut]  AssertionError: RuntimeEr | LuFinch | P2 | Distributed |  | module: ut, skipped,.. | aten_ops |
| [3129](https://github.com/intel/torch-xpu-ops/issues/3129) | [upstream_ut]  AssertionError: UserWarni | LuFinch | P2 | Flash Attention / Transformer Related |  | module: ut, skipped,.. | aten_ops |
| [3131](https://github.com/intel/torch-xpu-ops/issues/3131) | [upstream_ut]  NotImplementedError: The  | chunhuanMeng | P2 | Flash Attention / Transformer Related |  | module: ut, skipped,.. | aten_ops |
| [3132](https://github.com/intel/torch-xpu-ops/issues/3132) | [upstream_ut]  transfomers test reports  | LuFinch | P2 | Flash Attention / Transformer Related |  | module: ut, skipped,.. | aten_ops |
| [3133](https://github.com/intel/torch-xpu-ops/issues/3133) | [upstream_ut]  RuntimeError: scaled_dot_ | LuFinch | P2 | Flash Attention / Transformer Related |  | module: ut, skipped,.. | aten_ops |
| [3136](https://github.com/intel/torch-xpu-ops/issues/3136) | [upstream_ut]  AssertionError: False is  | LuFinch | P2 | Flash Attention / Transformer Related |  | module: ut, skipped,.. | aten_ops |
| [3137](https://github.com/intel/torch-xpu-ops/issues/3137) | [upstream_ut]  RuntimeError: expected sc | LuFinch | P2 | Flash Attention / Transformer Related |  | module: ut, skipped,.. | aten_ops |
| [3140](https://github.com/intel/torch-xpu-ops/issues/3140) | [upstream_ut]  RuntimeError: FlashAttent | LuFinch | P2 | Flash Attention / Transformer Related |  | module: ut, skipped,.. | aten_ops |
| [3141](https://github.com/intel/torch-xpu-ops/issues/3141) | [upstream_ut]  RuntimeError: FlashAttent | LuFinch | P2 | Flash Attention / Transformer Related |  | module: ut, skipped,.. | aten_ops |
| [3142](https://github.com/intel/torch-xpu-ops/issues/3142) | [upstream_ut]  RuntimeError: The sycl_ex | LuFinch | P2 | Flash Attention / Transformer Related |  | module: ut, skipped,.. | aten_ops |
| [3143](https://github.com/intel/torch-xpu-ops/issues/3143) | NotImplementedError: The operator 'aten: | LuFinch | P2 | Flash Attention / Transformer Related |  | module: ut, skipped,.. | aten_ops |
| [3163](https://github.com/intel/torch-xpu-ops/issues/3163) | [Bug Skip]: Object comparison failed: to | chunhuanMeng | P2 | Distributed |  | skipped, ut_upstream | aten_ops |
| [3166](https://github.com/intel/torch-xpu-ops/issues/3166) | test_consistency_SparseCSR failures | yucai-intel | P2 | TorchAO |  | skipped, ut_upstream | aten_ops |
| [3170](https://github.com/intel/torch-xpu-ops/issues/3170) | Unskip test_bmm_windows_error_xpu_float6 | jenniew | P2 | Others |  | skipped, ut_upstream | aten_ops |
| [3177](https://github.com/intel/torch-xpu-ops/issues/3177) | Accuracy gap of BF16/FP16 test_block_add | jenniew | P2 | Distributed | Dtype/Precision Issue - Accuracy gap in BF16/FP16 test | skipped | aten_ops |
| [3187](https://github.com/intel/torch-xpu-ops/issues/3187) | PyTorch XPU gpu_cpp_wrapper fails with I | CuiYifeng | P2 | Inductor / Compilation Related |  | ut_upstream | aten_ops |
| [3238](https://github.com/intel/torch-xpu-ops/issues/3238) | The supported dtypes of _refs.stft is no | CuiYifeng | P2 | Dtype / Precision Related |  | ut_upstream | aten_ops |

---

## 3. By Category {#3-by-category}

### Distributed (#distributed) - 24 issues

- [1893](https://github.com/intel/torch-xpu-ops/issues/1893): [upstream_ut] oneDNN accuracy issues in  | Owner: chunhuanMeng | Priority: P2 | Action: Needs PyTorch Repo Changes (upstream)
- [2024](https://github.com/intel/torch-xpu-ops/issues/2024): AssertionError: Torch not compiled with  | Owner: daisyden | Priority: P2 | Action: Need reproduce steps (Only for bugs or performance issue)
- [2164](https://github.com/intel/torch-xpu-ops/issues/2164): skip test_no_cuda_monkeypatch as it is c | Owner: daisyden | Priority: P2 | Action: add to skiplist
- [2169](https://github.com/intel/torch-xpu-ops/issues/2169): Frame size comparison failed in test_siz | Owner: guangyey | Priority: P2 | Action: Need reproduce steps (Only for bugs or performance issue)
- [2214](https://github.com/intel/torch-xpu-ops/issues/2214): test/test_sparse.py::TestSparseAnyXPU::t | Owner: jenniew | Priority: P2 | Action: Needs PyTorch Repo Changes (upstream)
- [2229](https://github.com/intel/torch-xpu-ops/issues/2229): test/test_sparse_csr.py::TestSparseCompr | Owner: jenniew | Priority: P2 | Action: Need more information - error logs and reproduction steps
- [2244](https://github.com/intel/torch-xpu-ops/issues/2244): test/test_sparse_csr.py::TestSparseCSRXP | Owner: jenniew | Priority: P2 | Action: Need reproduce steps (Only for bugs or performance issue)
- [2245](https://github.com/intel/torch-xpu-ops/issues/2245): oneDNN matmul received incorrect shape i | Owner: CuiYifeng | Priority: P2 | Action: Need reproduce steps (Only for bugs or performance issue)
- [2270](https://github.com/intel/torch-xpu-ops/issues/2270): Backend Compatibility Error in test/xpu/ | Owner: LuFinch | Priority: P2 | Action: Need reproduce steps (Only for bugs or performance issue)
- [2283](https://github.com/intel/torch-xpu-ops/issues/2283): [upstream_ut] sparse._sampled_addmm is n | Owner: jenniew | Priority: P1 | Action: Needs PyTorch Repo Changes (upstream)
- [2301](https://github.com/intel/torch-xpu-ops/issues/2301): [upstream_ut] dtypes not align with OpIn | Owner: daisyden | Priority: P2 | Action: Needs PyTorch Repo Changes (upstream)
- [2531](https://github.com/intel/torch-xpu-ops/issues/2531): [upstream_ut]  AssertionError: Torch not | Owner: daisyden | Priority: P2 | Action: Need reproduce steps (Only for bugs or performance issue)
- [2532](https://github.com/intel/torch-xpu-ops/issues/2532): Title: [upstream_ut]  AssertionError: wr | Owner: yucai-intel | Priority: P2 | Action: Need reproduce steps (Only for bugs or performance issue)
- [2536](https://github.com/intel/torch-xpu-ops/issues/2536): Title: [upstream_ut]  AttributeError: mo | Owner: daisyden | Priority: P2 | Action: Need reproduce steps (Only for bugs or performance issue)
- [2541](https://github.com/intel/torch-xpu-ops/issues/2541): Title: [upstream_ut]  RuntimeError: coul | Owner: yucai-intel | Priority: P2 | Action: Need reproduce steps (Only for bugs or performance issue)
- [2611](https://github.com/intel/torch-xpu-ops/issues/2611): [upstream_ut]  AssertionError: Tensor-li | Owner: daisyden | Priority: P2 | Action: Needs PyTorch Repo Changes (upstream)
- [2613](https://github.com/intel/torch-xpu-ops/issues/2613): [upstream_ut]  AssertionError: Tensor-li | Owner: daisyden | Priority: P2 | Action: Needs PyTorch Repo Changes (upstream)
- [2615](https://github.com/intel/torch-xpu-ops/issues/2615): [Bug Skip]: New failures RuntimeError: U | Owner: CuiYifeng | Priority: P2 | Action: Need reproduce steps (Only for bugs or performance issue)
- [2694](https://github.com/intel/torch-xpu-ops/issues/2694): Title: [upstream_ut]  AssertionError: Te | Owner: daisyden | Priority: P2 | Action: Verify the issue
- [2720](https://github.com/intel/torch-xpu-ops/issues/2720): [upstream_ut] RuntimeError: false INTERN | Owner: CuiYifeng | Priority: P2 | Action: Need reproduce steps (Only for bugs or performance issue)
- [2853](https://github.com/intel/torch-xpu-ops/issues/2853): [upstream_ut] torch.ops.aten._flash_atte | Owner: LuFinch | Priority: P2 | Action: Need more information - error logs and reproduction steps
- [3128](https://github.com/intel/torch-xpu-ops/issues/3128): [upstream_ut]  AssertionError: RuntimeEr | Owner: LuFinch | Priority: P2 | Action: Needs PyTorch Repo Changes (upstream)
- [3163](https://github.com/intel/torch-xpu-ops/issues/3163): [Bug Skip]: Object comparison failed: to | Owner: chunhuanMeng | Priority: P2 | Action: Needs PyTorch Repo Changes (upstream)
- [3177](https://github.com/intel/torch-xpu-ops/issues/3177): Accuracy gap of BF16/FP16 test_block_add | Owner: jenniew | Priority: P2 | Action: Need reproduce steps (Only for bugs or performance issue)

### Dtype / Precision Related (#dtype---precision-related) - 8 issues

- [2253](https://github.com/intel/torch-xpu-ops/issues/2253): the supported dtypes are not align with  | Owner: daisyden | Priority: P2 | Action: Needs PyTorch Repo Changes (upstream)
- [2295](https://github.com/intel/torch-xpu-ops/issues/2295): [upstream_ut][xpu][test]nn/test_embeddin | Owner: yucai-intel | Priority: P2 | Action: Needs PyTorch Repo Changes (upstream)
- [2376](https://github.com/intel/torch-xpu-ops/issues/2376): [Bug Skip]: NotImplementedError: "logadd | Owner: CuiYifeng | Priority: P2 | Action: 
- [2482](https://github.com/intel/torch-xpu-ops/issues/2482): test_dtypes issue introduced by pytorch  | Owner: daisyden | Priority: P2 | Action: Need reproduce steps (Only for bugs or performance issue)
- [2512](https://github.com/intel/torch-xpu-ops/issues/2512): [upstream_ut]  RuntimeError: _histc_xpu  | Owner: chunhuanMeng | Priority: P2 | Action: Needs PyTorch Repo Changes (upstream)
- [2783](https://github.com/intel/torch-xpu-ops/issues/2783): [Bug Skip]: Key "xpu" is missing from di | Owner: daisyden | Priority: P2 | Action: Need reproduce steps (Only for bugs or performance issue)
- [2810](https://github.com/intel/torch-xpu-ops/issues/2810): AssertionError: Object comparison failed | Owner: daisyden | Priority: P2 | Action: Needs PyTorch Repo Changes (upstream)
- [3238](https://github.com/intel/torch-xpu-ops/issues/3238): The supported dtypes of _refs.stft is no | Owner: CuiYifeng | Priority: P2 | Action: Needs PyTorch Repo Changes (upstream)

### Flash Attention / Transformer Related (#flash-attention---transformer-related) - 22 issues

- [2015](https://github.com/intel/torch-xpu-ops/issues/2015): inf is returned by nn.TransformerEncoder | Owner: yucai-intel | Priority: P2 | Action: Need more information - error logs and reproduction steps
- [2255](https://github.com/intel/torch-xpu-ops/issues/2255): [upstream_ut] RuntimeError: Long is not  | Owner: daisyden | Priority: P2 | Action: Needs PyTorch Repo Changes (upstream)
- [2285](https://github.com/intel/torch-xpu-ops/issues/2285): Support efficient attention | Owner: chunhuanMeng | Priority: P2 | Action: Need more information - error logs and reproduction steps
- [2436](https://github.com/intel/torch-xpu-ops/issues/2436): [upstream_ut]  AttributeError: 'NoneType | Owner: daisyden | Priority: P1 | Action: Need reproduce steps (Only for bugs or performance issue)
- [2442](https://github.com/intel/torch-xpu-ops/issues/2442): [Bug Skip]: NotImplementedError: Could n | Owner: daisyden, LuFinch | Priority: P2 | Action: Needs PyTorch Repo Changes (upstream)
- [2697](https://github.com/intel/torch-xpu-ops/issues/2697): Title: [upstream_ut]  RuntimeError: Expe | Owner: chunhuanMeng | Priority: P2 | Action: Needs PyTorch Repo Changes (upstream)
- [2800](https://github.com/intel/torch-xpu-ops/issues/2800): AttributeError: 'torch._C._XpuDeviceProp | Owner: guangyey | Priority: P2 | Action: Needs PyTorch Repo Changes (upstream)
- [2802](https://github.com/intel/torch-xpu-ops/issues/2802): Three aten._scaled_dot_product_flash_att | Owner: LuFinch | Priority: P2 | Action: Needs PyTorch Repo Changes (upstream)
- [2997](https://github.com/intel/torch-xpu-ops/issues/2997): AssertionError of test_linear_and_cel_ma | Owner: etaf | Priority: P2 | Action: Needs PyTorch Repo Changes (upstream)
- [3007](https://github.com/intel/torch-xpu-ops/issues/3007): AssertionError: Scalars are not equal! w | Owner: daisyden | Priority: P2 | Action: Verify the issue
- [3126](https://github.com/intel/torch-xpu-ops/issues/3126): [upstream_ut]  Two NestedTensor issue wi | Owner: LuFinch | Priority: P2 | Action: Needs PyTorch Repo Changes (upstream)
- [3127](https://github.com/intel/torch-xpu-ops/issues/3127): [upstream_ut]  AssertionError: Assertion | Owner: LuFinch | Priority: P2 | Action: Needs PyTorch Repo Changes (upstream)
- [3129](https://github.com/intel/torch-xpu-ops/issues/3129): [upstream_ut]  AssertionError: UserWarni | Owner: LuFinch | Priority: P2 | Action: Needs PyTorch Repo Changes (upstream)
- [3131](https://github.com/intel/torch-xpu-ops/issues/3131): [upstream_ut]  NotImplementedError: The  | Owner: chunhuanMeng | Priority: P2 | Action: Needs PyTorch Repo Changes (upstream)
- [3132](https://github.com/intel/torch-xpu-ops/issues/3132): [upstream_ut]  transfomers test reports  | Owner: LuFinch | Priority: P2 | Action: Needs PyTorch Repo Changes (upstream)
- [3133](https://github.com/intel/torch-xpu-ops/issues/3133): [upstream_ut]  RuntimeError: scaled_dot_ | Owner: LuFinch | Priority: P2 | Action: Needs PyTorch Repo Changes (upstream)
- [3136](https://github.com/intel/torch-xpu-ops/issues/3136): [upstream_ut]  AssertionError: False is  | Owner: LuFinch | Priority: P2 | Action: Needs PyTorch Repo Changes (upstream)
- [3137](https://github.com/intel/torch-xpu-ops/issues/3137): [upstream_ut]  RuntimeError: expected sc | Owner: LuFinch | Priority: P2 | Action: Needs PyTorch Repo Changes (upstream)
- [3140](https://github.com/intel/torch-xpu-ops/issues/3140): [upstream_ut]  RuntimeError: FlashAttent | Owner: LuFinch | Priority: P2 | Action: Needs PyTorch Repo Changes (upstream)
- [3141](https://github.com/intel/torch-xpu-ops/issues/3141): [upstream_ut]  RuntimeError: FlashAttent | Owner: LuFinch | Priority: P2 | Action: Needs PyTorch Repo Changes (upstream)
- [3142](https://github.com/intel/torch-xpu-ops/issues/3142): [upstream_ut]  RuntimeError: The sycl_ex | Owner: LuFinch | Priority: P2 | Action: Needs PyTorch Repo Changes (upstream)
- [3143](https://github.com/intel/torch-xpu-ops/issues/3143): NotImplementedError: The operator 'aten: | Owner: LuFinch | Priority: P2 | Action: Needs PyTorch Repo Changes (upstream)

### Inductor / Compilation Related (#inductor---compilation-related) - 3 issues

- [2888](https://github.com/intel/torch-xpu-ops/issues/2888): torch._inductor.exc.InductorError: Asser | Owner: Stonepia | Priority: P2 | Action: Needs PyTorch Repo Changes (upstream)
- [2958](https://github.com/intel/torch-xpu-ops/issues/2958): AssertionError of test_dtensor_basic_com | Owner: daisyden | Priority: P2 | Action: Needs PyTorch Repo Changes (upstream)
- [3187](https://github.com/intel/torch-xpu-ops/issues/3187): PyTorch XPU gpu_cpp_wrapper fails with I | Owner: CuiYifeng | Priority: P2 | Action: Needs PyTorch Repo Changes (upstream)

### Others (#others) - 7 issues

- [2287](https://github.com/intel/torch-xpu-ops/issues/2287): [upstream_ut] test_python_ref issues | Owner: yucai-intel | Priority: P2 | Action: Needs PyTorch Repo Changes (upstream)
- [2329](https://github.com/intel/torch-xpu-ops/issues/2329): [upstream_ut] feature missing: get_devic | Owner: etaf | Priority: P2 | Action: Needs PyTorch Repo Changes (upstream)
- [2891](https://github.com/intel/torch-xpu-ops/issues/2891): RuntimeError: Expected to find "(262144, | Owner: chunhuanMeng | Priority: P2 | Action: Needs PyTorch Repo Changes (upstream)
- [2999](https://github.com/intel/torch-xpu-ops/issues/2999): KeyError: 'eager_numerics.use_pytorch_li | Owner: daisyden | Priority: P2 | Action: Needs PyTorch Repo Changes (upstream)
- [3004](https://github.com/intel/torch-xpu-ops/issues/3004): TypeError: _xpu_recordMemoryHistory(): i | Owner: guangyey | Priority: P2 | Action: Needs PyTorch Repo Changes (upstream)
- [3033](https://github.com/intel/torch-xpu-ops/issues/3033): [Bug Skip]: Softmax tolerance | Owner: chunhuanMeng | Priority: P2 | Action: Need reproduce steps (Only for bugs or performance issue)
- [3170](https://github.com/intel/torch-xpu-ops/issues/3170): Unskip test_bmm_windows_error_xpu_float6 | Owner: jenniew | Priority: P2 | Action: Needs PyTorch Repo Changes (upstream)

### PT2E (#pt2e) - 6 issues

- [2554](https://github.com/intel/torch-xpu-ops/issues/2554): [upstream_ut]  AssertionError: Assertion | Owner: daisyden | Priority: P2 | Action: Needs PyTorch Repo Changes (upstream)
- [2609](https://github.com/intel/torch-xpu-ops/issues/2609): [upstream_ut]  torch._inductor.exc.Induc | Owner: daisyden | Priority: P2 | Action: Needs PyTorch Repo Changes (upstream)
- [2698](https://github.com/intel/torch-xpu-ops/issues/2698): Title: [upstream_ut]  RuntimeError: Flas | Owner: chunhuanMeng, LuFinch | Priority: P2 | Action: Needs PyTorch Repo Changes (upstream)
- [2715](https://github.com/intel/torch-xpu-ops/issues/2715): [upstream_ut]  torch._dynamo.exc.Unsuppo | Owner: CuiYifeng | Priority: P2 | Action: Needs PyTorch Repo Changes (upstream)
- [2806](https://github.com/intel/torch-xpu-ops/issues/2806): CompiledAOTI need XPU support | Owner: daisyden | Priority: P2 | Action: Needs PyTorch Repo Changes (upstream)
- [3006](https://github.com/intel/torch-xpu-ops/issues/3006): AssertionError: '.to(tl.float16)' unexpe | Owner: CuiYifeng | Priority: P2 | Action: Needs PyTorch Repo Changes (upstream)

### Sparse Operations Related (#sparse-operations-related) - 1 issues

- [1962](https://github.com/intel/torch-xpu-ops/issues/1962): [upstream_ut] segfault with test_fake_cr | Owner: jenniew, mengfei25 | Priority: P0 | Action: Need more information - error logs and reproduction steps

### TorchAO (#torchao) - 4 issues

- [2309](https://github.com/intel/torch-xpu-ops/issues/2309): unsupported ops with PYTORCH_ENABLE_XPU_ | Owner: daisyden | Priority: P2 | Action: add to skiplist
- [2578](https://github.com/intel/torch-xpu-ops/issues/2578): [TorchAO][UT] test/quantization/test_qua | Owner: Stonepia | Priority: P0 | Action: Needs PyTorch Repo Changes (upstream)
- [2620](https://github.com/intel/torch-xpu-ops/issues/2620): [upstream_ut]  AssertionError: dtype is  | Owner: daisyden | Priority: P2 | Action: Needs PyTorch Repo Changes (upstream)
- [3166](https://github.com/intel/torch-xpu-ops/issues/3166): test_consistency_SparseCSR failures | Owner: yucai-intel | Priority: P2 | Action: Close fixed issue

