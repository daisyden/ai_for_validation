# Torch XPU Ops Issue Report

Generated: 2026-04-06 20:56:49

## Summary

| Category | Count |
|----------|-------|
| Action Required | 161 |
| No Assignee | 44 |
| Duplicated Issues | 42 |
| With Dependency | 19 |
| Others | 151 |
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
| Needs PyTorch Repo Changes (upstream) | 75 |
| Dtype / Precision Related | 32 |
| Flash Attention / Transformer Related | 23 |
| Sparse Operations Related | 17 |
| Close fixed issue | 17 |
| Inductor / Compilation Related | 13 |
| Enable test | 5 |
| add to skiplist | 4 |
| Verify the issue | 3 |
| Revisit the PR as case failed | 3 |

### Other Stats

| Category | Count |
|----------|-------|
| Not Assigned | 44 |
| Duplicated Issues | 42 |
| Others | 151 |

---

## 1. Action Required

Issues that need action based on test results analysis.


### 1.1 Close fixed issue

| ID | Title | Owner | Owner Transferred | TBD | Module | Test Module |
|---|-------|-------|-------------------|-----|--------|-------------|
| 1624 | [DONT CLOSE] Known UT Issue list | None | RUIJIEZHONG66166 | Close fixed issue | distributed | ut |
| 2496 | [upstream_ut]  Segmentation fault when runnin | astachowiczhabana | libohao1201 | Close fixed issue | aten_ops | ut |
| 2518 | [upstream_ut]  TypeError: Creating a Tensor s | astachowiczhabana | libohao1201 | Close fixed issue | aten_ops | ut |
| 2592 | [release/2.10] models got fail_accuracy | mengfei25 | mengfei25 | Close fixed issue | aten_ops | e2e |
| 2619 | [release/2.10] Some models inductor performan | jianyizh, weishi-deng | mengfei25 | Close fixed issue | aten_ops | e2e |
| 2953 | [release/2.11][wsl] huggingface TrOCRForCausa | xuhancn | bjarzemb | Close fixed issue | aten_ops | e2e |
| 2981 | [release/2.11] T5 models performance dropped  | jianyizh, weishi-deng | mengfei25 | Close fixed issue | aten_ops | e2e |
| 3011 | [upstream_ut] torch.OutOfMemoryError: XPU out | None | Silv3S | Close fixed issue | aten_ops | ut |
| 3013 | [upstream_ut] RuntimeError: Kernel is incompa | None | Silv3S | Close fixed issue | unknown | ut |
| 3014 | [upstream_ut] AssertionError: False is not tr | None | Silv3S | Close fixed issue | unknown | ut |
| 3058 | [E2E] hf_GPT2_large amp_fp16/amp_bf16  traini | weishi-deng | kaileiyx | Close fixed issue | aten_ops | e2e |
| 3106 | Worker crashes when running TestDecompXPU,tes | BBBela | BBBela | Close fixed issue | aten_ops | ut |
| 3157 | XPU out of memory. Tried to allocate 32.00 Gi | kdrozd-dev | kdrozd-dev | Close fixed issue | aten_ops | ut |
| 3158 | AttributeError: module 'triton.compiler' has  | tadkrawiec | kdrozd-dev | Close fixed issue | aten_ops | ut |
| 3161 | Exception: Tensor-likes are not close! - test | tadkrawiec | kdrozd-dev | Close fixed issue | aten_ops | ut |
| 3166 | test_consistency_SparseCSR failures | yucai-intel | CuiYifeng | Close fixed issue | aten_ops | ut |

### 1.2 Dtype / Precision Related

| ID | Title | Owner | Owner Transferred | TBD | Module | Test Module |
|---|-------|-------|-------------------|-----|--------|-------------|
| 1661 | [distributed] Accuracy gap in _composable/fsd | githubsgi | githubsgi | Dtype / Precision Related | distributed | ut |
| 1778 | [Infra] Show known issues for accuracy test | mengfei25 | mengfei25 | Dtype / Precision Related | unknown | e2e |
| 1818 | [BMG-Windows][PT2.8]Torch-xpu-ops UT got accu | kdrozd-dev | kdrozd-dev | Dtype / Precision Related | aten_ops | ut |
| 1833 | [PT2E] INT8 accuracy is lower than FP32 | chunhuanMeng | chunhuanMeng | Dtype / Precision Related | low_precision | ut |
| 1866 | [release 2.8]Torchbench vision_maskrcnn (amp_ | BartoszKokoszko | BartoszKokoszko | Dtype / Precision Related | aten_ops | e2e |
| 1877 | Torchbench model squeezenet1_1 and functorch_ | Silv3S | Silv3S | Dtype / Precision Related | aten_ops | ut |
| 2128 | [2.9][BMG-Windows][Torchbench] speeach_transf | chuanqi129 | chuanqi129 | Dtype / Precision Related | aten_ops | ut |
| 2219 | float8_e4m3fn precision overflow | CuiYifeng, yucai-intel | CuiYifeng, yucai-intel | Dtype / Precision Related | aten_ops | ut |
| 2376 | [Bug Skip]: NotImplementedError: "logaddexp_x | CuiYifeng | CuiYifeng | Dtype / Precision Related | aten_ops | ut |
| 2439 | [oneDNN] TestDecompXPU.test_quick_addmv_xpu_f | libohao1201 | libohao1201 | Dtype / Precision Related | aten_ops | ut |
| 2444 | [upstream_ut]  RuntimeError: UR backend faile | Silv3S | Silv3S | Dtype / Precision Related | aten_ops | ut |
| 2572 | [TorchAO][UT] test/dtypes/test_affine_quantiz | xiaowangintel | xiaowangintel | Dtype / Precision Related | AO | build |
| 2596 | [TorchAO][BMG]Observed accuracy fluctuations  | None | None | Dtype / Precision Related | AO | ut |
| 2615 | [Bug Skip]: New failures RuntimeError: Unsupp | CuiYifeng | CuiYifeng | Dtype / Precision Related | aten_ops | ut |
| 2656 | [release/2.10] models got fail_accuracy on BM | None | None | Dtype / Precision Related | aten_ops | ut |
| 2680 | XPU Autocast does not support  fp32 dtypes | CuiYifeng | CuiYifeng | Dtype / Precision Related | aten_ops | ut |
| 2686 | [distributed] Accuracy issues with test_distr | frost-intel | frost-intel | Dtype / Precision Related | distributed | ut |
| 2779 | Accuracy failures in logspace op | PawelSwider2000 | PawelSwider2000 | Dtype / Precision Related | aten_ops | ut |
| 2862 | accuracy issue with test_float8_scale_fast_ac | tszulist-hbn | tszulist-hbn | Dtype / Precision Related | aten_ops | ut |
| 2908 | [release/2.11] Model fail_accuracy for 5 test | xuhancn | xuhancn | Dtype / Precision Related | aten_ops | e2e |
| 2924 | [release/2.11] xcit_large_24_p8_224 amp_bf16  | jianyizh, mengfei25 | jianyizh, mengfei25 | Dtype / Precision Related | aten_ops | e2e |
| 2928 | [release/2.11] pyhpc_turbulent_kinetic_energy | jianyizh | jianyizh | Dtype / Precision Related | aten_ops | e2e |
| 2952 | [release/2.11][wsl] timm_models_accuracy_trai | weishi-deng | weishi-deng | Dtype / Precision Related | aten_ops | ut |
| 2960 | [release/2.11] timm_models_xcit_large_24_p8_2 | None | None | Dtype / Precision Related | aten_ops | ut |
| 2984 | [release/2.11] sebotnet33ts_256 fp32 training | jianyizh, weishi-deng | jianyizh, weishi-deng | Dtype / Precision Related | aten_ops | e2e |
| 3148 | [Triton] Huggingface openai/whisper-tiny got  | None | None | Dtype / Precision Related | aten_ops | e2e |
| 3151 | [Triton] Timm_models  rexnet_100 / fbnetv3_b  | None | None | Dtype / Precision Related | aten_ops | e2e |
| 3174 | [Bug Skip]: Accuracy failure of test_Conv2d_g | None | None | Dtype / Precision Related | aten_ops | ut |

### 1.3 Enable test

| ID | Title | Owner | Owner Transferred | TBD | Module | Test Module |
|---|-------|-------|-------------------|-----|--------|-------------|
| 2024 | AssertionError: Torch not compiled with CUDA  | daisyden | daisyden | Enable test | aten_ops | ut |
| 2132 | [2.9][BMG-Windows][Torch-xpu-ops UT] 1 case f | pbielak | daisyden | Enable test | aten_ops | ut |
| 2531 | [upstream_ut]  AssertionError: Torch not comp | daisyden | daisyden | Enable test | unknown | ut |
| 3242 | AssertionError: Torch not compiled with CUDA  | None | daisyden | Enable test | aten_ops | ut |

### 1.4 Flash Attention / Transformer Related

| ID | Title | Owner | Owner Transferred | TBD | Module | Test Module |
|---|-------|-------|-------------------|-----|--------|-------------|
| 1165 | [CI] Add a test of PyTorch XPU with Huggingfa | RUIJIEZHONG66166 | RUIJIEZHONG66166 | Flash Attention / Transformer Related | aten_ops | build |
| 1556 | [distributed] NotImplementedError: Operator a | pkourdis | pkourdis | Flash Attention / Transformer Related | distributed | ut |
| 1749 | transformers UT failure in XPU because SDPA c | LuFinch | LuFinch | Flash Attention / Transformer Related | aten_ops | ut |
| 2200 | support flash attention op on XPU device | ElaineBao | ElaineBao | Flash Attention / Transformer Related | aten_ops | ut |
| 2232 | sdpa backward kernel is required to reduce me | None | None | Flash Attention / Transformer Related | aten_ops | ut |
| 2390 | SDPA in pytorch use different backend compare | LuFinch | LuFinch | Flash Attention / Transformer Related | aten_ops | ut |
| 2570 | crash in sdpa. | LuFinch | LuFinch | Flash Attention / Transformer Related | aten_ops | ut |
| 2853 | [upstream_ut] torch.ops.aten._flash_attention | LuFinch | LuFinch | Flash Attention / Transformer Related | aten_ops | ut |
| 3093 | XPU does not support NestedTensor for SDPA op | None | None | Flash Attention / Transformer Related | aten_ops | ut |
| 3178 | New failed test cases 2026-03-25 | pponikox | pponikox | Flash Attention / Transformer Related | aten_ops | ut |
| 3195 | test_sdpa_unbacked_no_dde_xpu crashed | None | None | Flash Attention / Transformer Related | aten_ops | ut |
| 3231 | Dynamo failed to run FX node with fake tensor | None | None | Flash Attention / Transformer Related | aten_ops | ut |
| 3258 | Error in op: torch.ops.aten._scaled_dot_produ | None | None | Flash Attention / Transformer Related | aten_ops | ut |
| 3259 | New failed test cases 2026-04-02 | SlawomirLaba | SlawomirLaba | Flash Attention / Transformer Related | aten_ops | ut |

### 1.5 Inductor / Compilation Related

| ID | Title | Owner | Owner Transferred | TBD | Module | Test Module |
|---|-------|-------|-------------------|-----|--------|-------------|
| 1548 | [distributed] AssertionError: 'fused_all_gath | Chao1Han | Chao1Han | Inductor / Compilation Related | distributed | ut |
| 1762 | Add an ocloc AOT target compilation test in c | chunhuanMeng | chunhuanMeng | Inductor / Compilation Related | aten_ops | ut |
| 1970 | torch._dynamo.exc.BackendCompilerFailed: back | None | None | Inductor / Compilation Related | aten_ops | ut |
| 2169 | Frame size comparison failed in test_size_com | guangyey | guangyey | Inductor / Compilation Related | aten_ops | ut |
| 2532 | Title: [upstream_ut]  AssertionError: wrong n | yucai-intel | yucai-intel | Inductor / Compilation Related | aten_ops | ut |
| 2605 | [int4][inductor] Add freezing pattern for fus | None | None | Inductor / Compilation Related | aten_ops | ut |
| 2650 | [OOB Performance] The performance impact caus | jianyizh | jianyizh | Inductor / Compilation Related | aten_ops | e2e |
| 2767 | [UT] test_control_flow_xpu.py got AssertionEr | PatrykWilczewski | PatrykWilczewski | Inductor / Compilation Related | aten_ops | ut |
| 2873 | [Bug Skip]: test_repos.py contains several fa | PawelSwider2000 | PawelSwider2000 | Inductor / Compilation Related | aten_ops | ut |
| 2922 | [release/2.11] UT inductor AssertionError: pa | tadkrawiec | tadkrawiec | Inductor / Compilation Related | aten_ops | ut |
| 2935 | [release/2.11][inductor] huggingface amp_fp16 | jianyizh | jianyizh | Inductor / Compilation Related | aten_ops | e2e |
| 3191 | torch._inductor.exc.InductorError: AssertionE | EikanWang, Copilot | EikanWang, Copilot | Inductor / Compilation Related | aten_ops | e2e |

### 1.6 Needs PyTorch Repo Changes (upstream)

| ID | Title | Owner | Owner Transferred | TBD | Module | Test Module |
|---|-------|-------|-------------------|-----|--------|-------------|
| 1505 | [ARC-WSL-Ubuntu24.04] 15 Timm models got fail | xuhancn, Stonepia | xuhancn, Stonepia | Needs PyTorch Repo Changes (upstream) | inductor | e2e |
| 1963 | [upstream_ut] MetadataMismatchError in TestFa | pbielak | pbielak | Needs PyTorch Repo Changes (upstream) | aten_ops | ut |
| 2214 | test/test_sparse.py::TestSparseAnyXPU::test_g | jenniew | jenniew | Needs PyTorch Repo Changes (upstream) | aten_ops | ut |
| 2234 | [upstream_ut] AssertionError: RuntimeError no | Silv3S | Silv3S | Needs PyTorch Repo Changes (upstream) | aten_ops | ut |
| 2248 | [upstream_ut] test_cow failures | gplutop7 | gplutop7 | Needs PyTorch Repo Changes (upstream) | aten_ops | ut |
| 2251 | [upstream_ut] test_fake_autocase got Exceptio | astachowiczhabana | astachowiczhabana | Needs PyTorch Repo Changes (upstream) | aten_ops | ut |
| 2255 | [upstream_ut] RuntimeError: Long is not suppo | daisyden | daisyden | Needs PyTorch Repo Changes (upstream) | aten_ops | ut |
| 2283 | [upstream_ut] sparse._sampled_addmm is not su | jenniew | jenniew | Needs PyTorch Repo Changes (upstream) | aten_ops | ut |
| 2287 | [upstream_ut] test_python_ref issues | yucai-intel | yucai-intel | Needs PyTorch Repo Changes (upstream) | aten_ops | ut |
| 2295 | [upstream_ut][xpu][test]nn/test_embedding.py: | yucai-intel | yucai-intel | Needs PyTorch Repo Changes (upstream) | inductor | ut |
| 2301 | [upstream_ut] dtypes not align with OpInfo | daisyden | daisyden | Needs PyTorch Repo Changes (upstream) | aten_ops | ut |
| 2329 | [upstream_ut] feature missing: get_device_tfl | etaf | etaf | Needs PyTorch Repo Changes (upstream) | inductor | ut |
| 2359 | [upstream_ut] GradcheckError: Backward is not | BBBela | BBBela | Needs PyTorch Repo Changes (upstream) | aten_ops | ut |
| 2554 | [upstream_ut]  AssertionError: AssertionError | daisyden | daisyden | Needs PyTorch Repo Changes (upstream) | inductor | ut |
| 2578 | [TorchAO][UT] test/quantization/test_quant_ap | Stonepia | Stonepia | Needs PyTorch Repo Changes (upstream) | AO | build |
| 2609 | [upstream_ut]  torch._inductor.exc.InductorEr | daisyden | daisyden | Needs PyTorch Repo Changes (upstream) | inductor | ut |
| 2620 | [upstream_ut]  AssertionError: dtype is neede | daisyden | daisyden | Needs PyTorch Repo Changes (upstream) | inductor | ut |
| 2663 | test_sparse_semi_structured.py gaps | None | None | Needs PyTorch Repo Changes (upstream) | aten_ops | ut |
| 2670 | [upstream_ut]  RuntimeError: could not create | tszulist-hbn | tszulist-hbn | Needs PyTorch Repo Changes (upstream) | aten_ops | ut |
| 2693 | Title: [upstream_ut]  AssertionError: Scalars | hoshibara | hoshibara | Needs PyTorch Repo Changes (upstream) | inductor | ut |
| 2696 | Title: [upstream_ut]  RuntimeError: Expected  | chunhuanMeng | chunhuanMeng | Needs PyTorch Repo Changes (upstream) | inductor | e2e |
| 2697 | Title: [upstream_ut]  RuntimeError: Expected  | chunhuanMeng | chunhuanMeng | Needs PyTorch Repo Changes (upstream) | inductor | e2e |
| 2698 | Title: [upstream_ut]  RuntimeError: FlashAtte | chunhuanMeng, LuFinch | chunhuanMeng, LuFinch | Needs PyTorch Repo Changes (upstream) | inductor | ut |
| 2704 | Title: [upstream_ut]  AssertionError: Asserti | kdrozd-dev | kdrozd-dev | Needs PyTorch Repo Changes (upstream) | aten_ops | ut |
| 2712 | [upstream_ut]  RuntimeError: Cannot swap t2 b | tszulist-hbn | tszulist-hbn | Needs PyTorch Repo Changes (upstream) | aten_ops | ut |
| 2714 | [upstream_ut]  AssertionError: Object compari | Silv3S | Silv3S | Needs PyTorch Repo Changes (upstream) | aten_ops | ut |
| 2715 | [upstream_ut]  torch._dynamo.exc.Unsupported: | CuiYifeng | CuiYifeng | Needs PyTorch Repo Changes (upstream) | aten_ops | ut |
| 2798 | Test case  test/test_dlpack.py::TestTorchDlPa | None | None | Needs PyTorch Repo Changes (upstream) | aten_ops | ut |
| 2800 | AttributeError: 'torch._C._XpuDevicePropertie | guangyey | guangyey | Needs PyTorch Repo Changes (upstream) | inductor | ut |
| 2802 | Three aten._scaled_dot_product_flash_attentio | LuFinch | LuFinch | Needs PyTorch Repo Changes (upstream) | inductor | ut |
| 2806 | CompiledAOTI need XPU support | daisyden | daisyden | Needs PyTorch Repo Changes (upstream) | inductor | ut |
| 2810 | AssertionError: Object comparison failed: Dec | daisyden | daisyden | Needs PyTorch Repo Changes (upstream) | inductor | ut |
| 2888 | torch._inductor.exc.InductorError: AssertionE | Stonepia | Stonepia | Needs PyTorch Repo Changes (upstream) | inductor | ut |
| 2891 | RuntimeError: Expected to find "(262144, 0, 5 | chunhuanMeng | chunhuanMeng | Needs PyTorch Repo Changes (upstream) | inductor | e2e |
| 2918 | [XPU][upstream_ut][COW] Skip non-supported op | gplutop7 | gplutop7 | Needs PyTorch Repo Changes (upstream) | aten_ops | ut |
| 2919 | [XPU][upstream_ut][COW] Fix materialization i | gplutop7 | gplutop7 | Needs PyTorch Repo Changes (upstream) | aten_ops | ut |
| 2958 | AssertionError of test_dtensor_basic_compile | daisyden | daisyden | Needs PyTorch Repo Changes (upstream) | inductor | ut |
| 2997 | AssertionError of test_linear_and_cel_max_aut | etaf | etaf | Needs PyTorch Repo Changes (upstream) | inductor | ut |
| 2999 | KeyError: 'eager_numerics.use_pytorch_libdevi | daisyden | daisyden | Needs PyTorch Repo Changes (upstream) | inductor | ut |
| 3004 | TypeError: _xpu_recordMemoryHistory(): incomp | guangyey | guangyey | Needs PyTorch Repo Changes (upstream) | inductor | ut |
| 3006 | AssertionError: '.to(tl.float16)' unexpectedl | CuiYifeng | CuiYifeng | Needs PyTorch Repo Changes (upstream) | inductor | e2e |
| 3041 | AssertionError: Expected len(flat_diff_result | Silv3S | Silv3S | Needs PyTorch Repo Changes (upstream) | aten_ops | ut |
| 3077 | [Bug Skip] test_dlpack.py::TestTorchDlPackXPU | AKloniecki | AKloniecki | Needs PyTorch Repo Changes (upstream) | aten_ops | ut |
| 3094 | XPUGraph tree support | None | None | Needs PyTorch Repo Changes (upstream) | inductor | ut |
| 3095 | cutlass support blocks some unit test cases | None | None | Needs PyTorch Repo Changes (upstream) | inductor | ut |
| 3126 | [upstream_ut]  Two NestedTensor issue with fl | LuFinch | LuFinch | Needs PyTorch Repo Changes (upstream) | aten_ops | ut |
| 3127 | [upstream_ut]  AssertionError: AssertionError | LuFinch | LuFinch | Needs PyTorch Repo Changes (upstream) | aten_ops | ut |
| 3128 | [upstream_ut]  AssertionError: RuntimeError n | LuFinch | LuFinch | Needs PyTorch Repo Changes (upstream) | aten_ops | ut |
| 3129 | [upstream_ut]  AssertionError: UserWarning no | LuFinch | LuFinch | Needs PyTorch Repo Changes (upstream) | aten_ops | ut |
| 3130 | [upstream_ut]  AssertionError: tensor(True, d | LuFinch | LuFinch | Needs PyTorch Repo Changes (upstream) | aten_ops | ut |
| 3131 | [upstream_ut]  NotImplementedError: The opera | chunhuanMeng | chunhuanMeng | Needs PyTorch Repo Changes (upstream) | aten_ops | ut |
| 3132 | [upstream_ut]  transfomers test reports Runti | LuFinch | LuFinch | Needs PyTorch Repo Changes (upstream) | aten_ops | ut |
| 3133 | [upstream_ut]  RuntimeError: scaled_dot_produ | LuFinch | LuFinch | Needs PyTorch Repo Changes (upstream) | aten_ops | ut |
| 3137 | [upstream_ut]  RuntimeError: expected scalar  | LuFinch | LuFinch | Needs PyTorch Repo Changes (upstream) | aten_ops | ut |
| 3140 | [upstream_ut]  RuntimeError: FlashAttentionFo | LuFinch | LuFinch | Needs PyTorch Repo Changes (upstream) | aten_ops | ut |
| 3141 | [upstream_ut]  RuntimeError: FlashAttentionFo | LuFinch | LuFinch | Needs PyTorch Repo Changes (upstream) | aten_ops | ut |
| 3142 | [upstream_ut]  RuntimeError: The sycl_ext_one | LuFinch | LuFinch | Needs PyTorch Repo Changes (upstream) | aten_ops | ut |
| 3143 | NotImplementedError: The operator 'aten::_sca | LuFinch | LuFinch | Needs PyTorch Repo Changes (upstream) | aten_ops | ut |
| 3163 | [Bug Skip]: Object comparison failed: torch.i | chunhuanMeng | chunhuanMeng | Needs PyTorch Repo Changes (upstream) | aten_ops | ut |
| 3165 | test_sparse_csr_xpu.py::TestSparseCompressedT | None | None | Needs PyTorch Repo Changes (upstream) | aten_ops | ut |
| 3167 | NotImplementedError: Could not run 'aten::tri | tszulist-hbn | tszulist-hbn | Needs PyTorch Repo Changes (upstream) | aten_ops | ut |
| 3168 | NotImplementedError: Could not run 'aten::add | jenniew | jenniew | Needs PyTorch Repo Changes (upstream) | aten_ops | ut |
| 3169 | NotImplementedError: Could not run 'aten::hsp | jkosnox | jkosnox | Needs PyTorch Repo Changes (upstream) | aten_ops | ut |
| 3170 | Unskip test_bmm_windows_error_xpu_float64 | jenniew | jenniew | Needs PyTorch Repo Changes (upstream) | aten_ops | ut |
| 3187 | PyTorch XPU gpu_cpp_wrapper fails with Induct | CuiYifeng | CuiYifeng | Needs PyTorch Repo Changes (upstream) | aten_ops | ut |
| 3229 | RuntimeError: No viable backend for scaled_do | None | None | Needs PyTorch Repo Changes (upstream) | aten_ops | ut |
| 3238 | The supported dtypes of _refs.stft is not ali | CuiYifeng | CuiYifeng | Needs PyTorch Repo Changes (upstream) | aten_ops | ut |
| 3247 | NotImplementedError: "dot_xpu_mkl" not implem | Silv3S | Silv3S | Needs PyTorch Repo Changes (upstream) | aten_ops | ut |

### 1.7 Sparse Operations Related

| ID | Title | Owner | Owner Transferred | TBD | Module | Test Module |
|---|-------|-------|-------------------|-----|--------|-------------|
| 1962 | [upstream_ut] segfault with test_fake_crossre | jenniew, mengfei25 | jenniew, mengfei25 | Sparse Operations Related | aten_ops | ut |
| 2229 | test/test_sparse_csr.py::TestSparseCompressed | jenniew | jenniew | Sparse Operations Related | aten_ops | ut |
| 2245 | oneDNN matmul received incorrect shape in tes | CuiYifeng | CuiYifeng | Sparse Operations Related | aten_ops | ut |
| 2595 | [Bug Skip]: Random crashed cases 2025-12-17 | None | None | Sparse Operations Related | aten_ops | ut |
| 2729 | [Bug Skip]: Random failures 2026WW03 | Silv3S | Silv3S | Sparse Operations Related | unknown | ut |
| 2751 | [Bug Skip]: Random failures 2026WW04 | None | None | Sparse Operations Related | unknown | ut |
| 2777 | [Bug Skip]: Random failures 2026WW05 | AKloniecki | AKloniecki | Sparse Operations Related | unknown | ut |
| 2801 | to_dense() for Sparse CSR backend cannot broa | jenniew | jenniew | Sparse Operations Related | aten_ops | ut |
| 2921 | [abs][complex64] - new failing test cases cau | AKloniecki | AKloniecki | Sparse Operations Related | aten_ops | ut |
| 2946 | [Bug Skip]: Random failures 2026WW09 | None | None | Sparse Operations Related | unknown | ut |
| 2965 | [Bug Skip]: Random failures 2026WW10 | None | None | Sparse Operations Related | unknown | ut |
| 3081 | Sparse CSR gemm-like ops have not been suppor | None | None | Sparse Operations Related | aten_ops | ut |

### 1.8 Verify the issue

| ID | Title | Owner | Owner Transferred | TBD | Module | Test Module |
|---|-------|-------|-------------------|-----|--------|-------------|
| 2331 | [upstream_ut] AssertionError: Scalars are not | hoshibara | daisyden | Verify the issue | inductor | ut |
| 2694 | Title: [upstream_ut]  AssertionError: Tensor- | daisyden | daisyden | Verify the issue | inductor | ut |
| 3007 | AssertionError: Scalars are not equal! with t | daisyden | daisyden | Verify the issue | inductor | e2e |

### 1.9 add to skiplist

| ID | Title | Owner | Owner Transferred | TBD | Module | Test Module |
|---|-------|-------|-------------------|-----|--------|-------------|
| 2164 | skip test_no_cuda_monkeypatch as it is cuda s | daisyden | daisyden | add to skiplist | aten_ops | ut |
| 2309 | unsupported ops with PYTORCH_ENABLE_XPU_FALLB | daisyden | daisyden | add to skiplist | aten_ops | ut |
| 2472 | [upstream_ut]  NotImplementedError: The opera | Silv3S | daisyden | add to skiplist | aten_ops | ut |
| 2508 | TypedStorage / TypedTensors deprecation | Silv3S | libohao1201 | add to skiplist | aten_ops | ut |

### Issues without Assignee

| ID | Title | Owner | Owner Transferred | TBD | Module | Test Module |
|---|-------|-------|-------------------|-----|--------|-------------|
| 3257 | [Linux][E2E][Regression] Huggingface test mod | None | chuanqi | assign owner | aten_ops | e2e |
| 3255 | [Linux][PT2E][Regression] Some performance te | None | chuanqi | assign owner | aten_ops | e2e |
| 3233 | [distributed] RuntimeError: No backend for th | None | chuanqi | assign owner | distributed | ut |
| 3232 | [distributed][tensor] AssertionError: Asserti | None | chuanqi | assign owner | distributed | ut |
| 3180 | [E2E] Timm/Torchbench models got "eager_two_r | None | chuanqi | assign owner | aten_ops | ut |
| 3149 | New failure in test_rms_norm_decomp_accelerat | None | chuanqi | assign owner | aten_ops | ut |
| 3124 | [TorchAO][Bug] ImportError: Requires mslk >=  | None | chuanqi | assign owner | aten_ops | ut |
| 3121 | [Bug Skip]: CUDA specific UT test_fft_half_an | None | chuanqi | assign owner | aten_ops | ut |
| 3102 | [distributed] RuntimeError: Invalid device st | None | chuanqi | assign owner | distributed | ut |
| 3101 | [distributed] 'torch._C._distributed_c10d.Pro | None | chuanqi | assign owner | distributed | ut |
| 3100 | [distributed] /handler/dump_nccl_trace_pickle | None | chuanqi | assign owner | distributed | ut |
| 3096 | VISIBLE_DEVICE support | None | chuanqi | assign owner | aten_ops | ut |
| 3086 | nvml support blocks some test cases | None | chuanqi | assign owner | aten_ops | ut |
| 3084 | torch.library.register_autocast does not supp | None | chuanqi | assign owner | aten_ops | ut |
| 3083 | [Bug Skip]: Random failures 2026WW12 | None | chuanqi | assign owner | unknown | ut |
| 3082 | multithread support in distributed | None | chuanqi | assign owner | distributed | ut |
| 3080 | cudagraph tests blocked by feature gap | None | chuanqi | assign owner | aten_ops | ut |
| 3076 | [TorchAO][BMG] Llama-3.2-1B-Instruct Dynamic  | None | chuanqi | assign owner | aten_ops | ut |
| 3025 | New failing test in Nightly Wheel test_decomp | None | chuanqi | assign owner | aten_ops | ut |
| 2979 | eca_halonext26ts got RuntimeError: ZE_RESULT_ | None | chuanqi | assign owner | aten_ops | e2e |
| 2948 | [AO] Benchmark enabling on XPU | None | chuanqi | assign owner | AO | ut |
| 2930 | [release/2.11] UT skip test_binary_ufuncs_xpu | None | chuanqi | assign owner | aten_ops | ut |
| 2914 | Test case test/test_autograd.py::TestAutograd | None | chuanqi | assign owner | aten_ops | ut |
| 2912 | [release/2.11] UT extended 220 new failures | None | chuanqi | assign owner | aten_ops | ut |
| 2899 | Update nan_to_num XPU stub to use std::option | None | chuanqi | assign owner | aten_ops | ut |
| 2858 | [Bug Skip]: test_xpu new failures | None | chuanqi | assign owner | unknown | ut |
| 2852 | [Bug Skip]: New UT failures in 0206 nightly o | None | chuanqi | assign owner | aten_ops | ut |
| 2845 | [Bug Skip]:[UT] [Windows] failed cases 2026-2 | None | chuanqi | assign owner | aten_ops | ut |
| 2797 | Copy error is not raise on test_dlpack.py::te | None | chuanqi | assign owner | aten_ops | ut |
| 2737 | [distributed] AttributeError: module 'torch._ | None | chuanqi | assign owner | distributed | ut |
| 2676 | Random failure in CI test | None | chuanqi | assign owner | unknown | ut |
| 2539 | Title: [upstream_ut]  RuntimeError: Tried to  | None | chuanqi | assign owner | aten_ops | ut |
| 2447 | test_share_memory_xpu failure | None | chuanqi | assign owner | aten_ops | ut |
| 2446 | [Bug Skip]: AssertionError: "Simulate error"  | None | chuanqi | assign owner | aten_ops | ut |
| 2440 | [For UT failures classify] Save reference com | None | chuanqi | assign owner | inductor | ut |
| 2404 | [distributed][checkpoint] AssertionError: Boo | None | chuanqi | assign owner | aten_ops | ut |
| 2333 | [Don't merge] Collect the new passed cases in | None | chuanqi | assign owner | aten_ops | ut |
| 2327 | [TorchAO] benchmark enabling on XPU | None | chuanqi | assign owner | aten_ops | ut |
| 2199 | Fix reduction and norm register spill | None | chuanqi | assign owner | aten_ops | ut |
| 2196 | Fix DistributionElementwiseKernelFunctor regi | None | chuanqi | assign owner | aten_ops | ut |
| 2022 | [Windows] [CI] [UT] AssertionError: Tensor-li | None | chuanqi | assign owner | aten_ops | ut |
| 1689 | [For op Perf Comparison] Save reference compa | None | chuanqi | assign owner | aten_ops | ut |
| 1678 | missing op support for `model.share_memory()` | None | chuanqi | assign owner | aten_ops | ut |
| 1519 | [PVC][PT2.7][ABI=1][Torch-xpu-ops UT][ww10] 2 | None | chuanqi | assign owner | aten_ops | ut |

---

## 2. Duplicated Issues

Issues that share test cases with other issues.

| ID | Title | Owner | Reporter | Duplicated With | Module | Test Module |
|---|-------|-------|----------|-----------------|--------|-------------|
| 1893 | [upstream_ut] oneDNN accuracy issues in  | chunhuanMeng | daisyden | 1951 | aten_ops | ut |
| 1951 | Functionality issues in TestCommon.test_ | AKloniecki | daisyden | 1893 | aten_ops | ut |
| 1973 | AssertionError: Scalars or Tensor-likes  | gplutop7 | mengfei25 | 2837,2840 | aten_ops | ut |
| 2006 | work-item/workgroup issue in softmax/uns | BartoszKokoszko | daisyden | 2257 | aten_ops | ut |
| 2015 | inf is returned by nn.TransformerEncoder | yucai-intel | daisyden | 2186,2529 | aten_ops | ut |
| 2186 | AssertionError: Mul tiheadAttention does | daisyden | daisyden | 2015 | aten_ops | ut |
| 2220 | test/test_sparse_csr.py::TestSparseCompr | None | wincent8 | 2246 | aten_ops | ut |
| 2230 | test_sparse_csr.py::TestSparseCompressed | None | wincent8 | 2246,3175,3176 | aten_ops | ut |
| 2235 | test/test_sparse_csr.py::TestSparseCompr | None | wincent8 | 3047 | aten_ops | ut |
| 2238 | Exception: Tensor-likes are not close! i | BBBela | zxd1997066 | 3105 | aten_ops | ut |
| 2244 | test/test_sparse_csr.py::TestSparseCSRXP | jenniew | wincent8 | 3177 | aten_ops | ut |
| 2246 | torch/sparse/_triton_ops*.py need to be  | None | wincent8 | 2220,2230 | unknown | ut |
| 2253 | the supported dtypes are not align with  | daisyden | daisyden | 2482 | aten_ops | ut |
| 2257 | Accuracy failures in test/xpu/test_unary | pbielak | zxd1997066 | 2006 | aten_ops | ut |
| 2270 | Backend Compatibility Error in test/xpu/ | LuFinch | libohao1201 | 2442 | aten_ops | ut |
| 2285 | Support efficient attention | chunhuanMeng | daisyden | 2358 | aten_ops | ut |
| 2358 | test/test_view_ops.py::TestOldViewOpsXPU | Silv3S | wincent8 | 2285 | aten_ops | ut |
| 2436 | [upstream_ut]  AttributeError: 'NoneType | daisyden | daisyden | 2675 | aten_ops | ut |
| 2442 | [Bug Skip]: NotImplementedError: Could n | daisyden, LuFinch | CuiYifeng | 2270 | aten_ops | ut |
| 2482 | test_dtypes issue introduced by pytorch  | daisyden | daisyden | 2253 | aten_ops | ut |
| 2529 | [upstream_ut]  AssertionError: False is  | Silv3S | daisyden | 2015,3136 | aten_ops | ut |
| 2530 | Title: [upstream_ut]  AssertionError: Ru | PatrykWilczewski | daisyden | 2817 | aten_ops | ut |
| 2611 | [upstream_ut]  AssertionError: Tensor-li | daisyden | daisyden | 2613 | inductor | ut |
| 2613 | [upstream_ut]  AssertionError: Tensor-li | daisyden | daisyden | 2611 | inductor | ut |
| 2618 | [Bug Skip]: [regression] AssertionError: | jmamzax | kaileiyx | 3089 | unknown | ut |
| 2675 | [Bug Skip]: AttributeError: 'NoneType' o | pponikox | kaileiyx | 2436 | aten_ops | ut |
| 2817 | Expected error message is different than | kdrozd-dev | Silv3S | 2530 | aten_ops | ut |
| 2837 | Accuracy issue for Muon optimizer | Silv3S | kdrozd-dev | 1973 | aten_ops | ut |
| 2840 | Accuracy issue with 64 bit indexing dept | SlawomirLaba, Silv3S | kdrozd-dev | 1973 | aten_ops | ut |
| 2869 | [Bug Skip]: New UT failure in 0209 night | None | RUIJIEZHONG66166 | 3160 | aten_ops | ut |
| 2966 | [Bug Skip]: [Regression]2026-3-2 ut fail | jmamzax | kaileiyx | 3114 | aten_ops | ut |
| 3047 | [Bug Skip]: [Regression]UT failures 2026 | None | kaileiyx | 2235 | unknown | ut |
| 3089 | AssertionError: Torch not compiled with  | jmamzax | jmamzax | 2618 | unknown | ut |
| 3105 | Wrong results from oneDNN conv2d kernels | BBBela | BBBela | 2238 | aten_ops | ut |
| 3114 | [Bug Skip]: Failure skip on 2026-3-21 | None | guangyey | 2966 | aten_ops | ut |
| 3136 | [upstream_ut]  AssertionError: False is  | LuFinch | daisyden | 2529 | aten_ops | ut |
| 3156 | AssertionError: 'Assertion cur_target >= | kdrozd-dev | kdrozd-dev | 3184 | aten_ops | ut |
| 3160 | compiler not found (Windows) | kdrozd-dev | kdrozd-dev | 2869 | aten_ops | ut |
| 3175 | [Bug Skip]: ValueError: sampled_addmm(): | None | CuiYifeng | 2230 | aten_ops | ut |
| 3176 | [Bug Skip]: ValueError: _scaled_dot_prod | None | CuiYifeng | 2230 | aten_ops | ut |
| 3177 | Accuracy gap of BF16/FP16 test_block_add | jenniew | CuiYifeng | 2244 | aten_ops | ut |
| 3184 | New failing UTs: test_cross_entropy_loss | wpietka | BBBela | 3156 | aten_ops | ut |

---

## 3. Issues with Dependency

Issues that have dependencies on other components (non-empty).

| ID | Title | Owner | Type | Module | Test Module | Dependency | Labels |
|---|-------|------|------|--------|-------------|------------|--------|
| 1059 | SYCL RT: Using recommended shortcut | CuiYifeng, jianyizh | unknown | aten_ops | ut | oneAPI | dependency component: one |
| 1171 | LNL Windows got unexpected error me | xuhancn, chunhuanMeng | functionality bug | aten_ops | ut | driver | client, os: Windows, hw : |
| 1324 | [Win] UR Error when OOM and break t | Stonepia | functionality bug | aten_ops | ut | oneAPI | client, os: Windows, modu |
| 1510 | Some test cases will be hang on BMG | Stonepia, mengfei25 | functionality bug | aten_ops | ut | driver | bug, client, os: Ubuntu,  |
| 1547 | [distributed] NotImplementedError:  | Chao1Han | feature request | distributed | ut | oneAPI | module: distributed, depe |
| 1549 | [distributed] AssertionError: 'fuse | Chao1Han | functionality bug | distributed | ut | oneAPI | module: distributed, depe |
| 1551 | [distributed] NotImplementedError:  | Chao1Han | feature request | distributed | ut | oneAPI | module: distributed, depe |
| 1555 | [distributed] RuntimeError: aten.ad | chuanqi129 | functionality bug | distributed | ut | oneDNN | module: distributed, depe |
| 1649 | [cpp extension] Provide a clear err | dvrogozh | feature request | aten_ops | ut | oneAPI | dependency component: one |
| 1722 | Ask an API to query GPU type(iGPU/d | guangyey | unknown | aten_ops | ut | oneAPI | dependency component: one |
| 1727 | [distributed] AttributeError: modul | guangyey | functionality bug | distributed | ut | oneAPI | module: distributed, depe |
| 1912 | Implement the torch.ops.aten._weigh | liangan1 | feature request | aten_ops | ut | oneDNN | dependency component: one |
| 1986 | torch.xpu._sleep is missing, | guangyey | functionality bug | aten_ops | ut | oneAPI | dependency component: one |
| 2089 | need an implementation that won't i | guangyey | feature request | aten_ops | ut | driver | dependency component: dri |
| 2157 | BMG d2h copy is very slow compare t | jianyizh, mengfei25 | functionality bug | aten_ops | ut | driver | performance, dependency c |
| 2261 | [xpu][profiler] Run with fork proce | moksiuc | functionality bug | profiling | ut | oneAPI | dependency component: one |
| 2467 | Host may stuck when submit too many | jianyizh | functionality bug | aten_ops | ut | driver | dependency component: dri |
| 2655 | [BMG][OOB] hf_Reformer performance  | jianyizh | performance issue | aten_ops | e2e | Triton | E2E, dtype: float16, tria |
| 2769 | [oneDNN] New failed test cases with | LuFinch | functionality bug | aten_ops | ut | oneDNN | hw: PVC, dependency compo |

---

## 4. Others

Issues that don't fall into the categories above.

| ID | Title | Owner | Reporter | Labels | Module | Test Module |
|---|-------|-------|----------|--------|--------|-------------|
| 3246 | AssertionError: Booleans mismatch: True  | BartoszKokoszko | Silv3S | skipped | aten_ops | ut |
| 3243 | AssertionError: False is not true | pponikox | zxd1997066 | module: ut, skipped | aten_ops | ut |
| 3236 | AssertionError: 'def [28 chars]n unbind  | jmamzax | jmamzax | bug_fix_stage5 | aten_ops | ut |
| 3227 | torch xpu event has ~0.1ms latency, whic | guangyey | jianyizh |  | aten_ops | ut |
| 3224 | [Win][Build] Building SYCL (Device) obje | chunhuanMeng | anmyachev |  | aten_ops | build |
| 3216 | [OPs] Some ops of XPU have non-determini | CuiYifeng | YangKai0616 |  | aten_ops | ut |
| 3209 | [Win][Build] There is Cyclic dependencie | Copilot | NeoZhangJianyu |  | aten_ops | build |
| 3206 | [Bug Skip]: [new failures]RuntimeError:  | tszulist-hbn | kaileiyx | skipped | aten_ops | ut |
| 3196 | vitals is not supported, the cases shoul | libohao1201 | daisyden | skipped | aten_ops | ut |
| 3194 | Incorrect strides in TestCommonXPU,test_ | AKloniecki | AKloniecki |  | aten_ops | ut |
| 3189 | Task Tracker | guangyey | guangyey |  | aten_ops | ut |
| 3150 | [Task] Align XPU kernel's implementation | guangyey | guangyey |  | aten_ops | ut |
| 3139 | [distributed][_composable] AssertionErro | Kanya-Mo | zxd1997066 | bug, module: distributed | distributed | ut |
| 3103 | Tensor-likes are not equal for test_back | BBBela | BBBela | module: ut, skipped, rand | aten_ops | ut |
| 3088 | [TorchAO][BMG] INT4 RTN Flex-attention g | Stonepia | LifengWang | module: ao | AO | ut |
| 3074 | [Bug Skip] test_dlpack_exchange_api expe | AKloniecki | shangerxin | skipped | aten_ops | ut |
| 3060 | Implement torch._scaled_grouped_mm for x | Stonepia, liangan1 | kgajdamo | module: quant | low_precision | ut |
| 3048 | Profiler result is not correct on B70 | aostrowski-hbn | jianyizh | module: profiler | profiling | ut |
| 3033 | [Bug Skip]: Softmax tolerance | chunhuanMeng | chunhuanMeng | skipped, random | aten_ops | ut |
| 3032 | [TorchAO][UT] failures in test/prototype | Stonepia | zxd1997066 | module: ao | AO | build |
| 3030 | [Bug Skip] test/test_modules.py::TestMod | gplutop7 | shangerxin | skipped | aten_ops | ut |
| 3024 | Enable clang-tidy checks | Silv3S | Silv3S | bug_fix_stage5 | aten_ops | ut |
| 3022 | [distributed] batch_isend_irecv Compatib | zhangxiaoli73 | xiangyuT | module: distributed | distributed | ut |
| 3021 | [distributed] all_to_all_single Compatib | zhangxiaoli73 | xiangyuT | module: distributed | distributed | ut |
| 3010 | [distributed][tensor] test_random_ops.py | jenniew | zxd1997066 | bug, module: distributed | distributed | ut |
| 3000 | [Bug Skip]: RuntimeError: _share_fd_: on | gplutop7 | zxd1997066 | skipped | aten_ops | ut |
| 2993 | [Bug Skip]: Unexpected success of test_c | gplutop7 | CuiYifeng | skipped | aten_ops | ut |
| 2972 | [distributed] AssertionError: ValueError | newtdms | zxd1997066 | bug, module: distributed | distributed | ut |
| 2971 | [distributed] KeyError in test/distribut | newtdms, frost-intel | zxd1997066 | bug, module: distributed | distributed | ut |
| 2969 | [distributed] AssertionError: Scalars ar | frost-intel | zxd1997066 | bug, module: distributed | distributed | ut |
| 2968 | [distributed] timeout issue in test/dist | frost-intel | zxd1997066 | bug, module: distributed | distributed | ut |
| 2967 | [distributed] feature gaps in test/distr | frost-intel | zxd1997066 | bug, module: distributed | distributed | ut |
| 2950 | SYCL compilation flag -fsycl-id-queries- | BBBela | BBBela |  | aten_ops | ut |
| 2942 | [Windows] Unit tests got Fatal python er | xuhancn, Stonepia | mengfei25 | os: Windows | aten_ops | ut |
| 2940 | [release/2.11] Models performance droppe | jianyizh, LuFinch | mengfei25 | performance | aten_ops | e2e |
| 2939 | [release/2.11] gmlp_s16_224 inference am | jianyizh | mengfei25 | performance | aten_ops | e2e |
| 2938 | [release/2.11] basic_gnn_gin and basic_g | jianyizh | mengfei25 | performance | aten_ops | e2e |
| 2932 | [release/2.11] jx_nest_base and volo_d1_ | jianyizh | mengfei25 |  | aten_ops | e2e |
| 2929 | [release/2.11] volo_d1_224 inference got | jianyizh | mengfei25 |  | aten_ops | e2e |
| 2920 | Failing Test Cases in Nightly Wheel [202 | Silv3S | BBBela | skipped | aten_ops | ut |
| 2907 | [release/2.11] Models performance regres | xuhancn | bjarzemb |  | aten_ops | ut |
| 2879 | RuntimeError: _share_fd_: only available | Silv3S | Silv3S | bug_fix_stage5 | aten_ops | ut |
| 2871 | [distributed][fsdp] test_fsdp_overlap.py | songhappy | zxd1997066 | bug, module: distributed | distributed | ut |
| 2823 | [TorchAO][BMG] Llama-3.2-1B-Instruct Dyn | xiaowangintel, lchen2331 | LifengWang | module: ao | AO | ut |
| 2816 | torch.logcumsumexp incorrectly returns N | Silv3S | Silv3S | Ready for merge, skipped, | aten_ops | ut |
| 2815 | RuntimeError: output with shape [2] does | PawelSwider2000 | Silv3S | skipped, bug_fix_stage5 | aten_ops | ut |
| 2811 | [Bug Skip]: [Regression] failed cases 20 | jmamzax | kaileiyx | skipped, bug_fix_stage5 | aten_ops | ut |
| 2795 | Histc raises error with integer input wh | CuiYifeng | YangKai0616 |  | aten_ops | ut |
| 2783 | [Bug Skip]: Key "xpu" is missing from di | daisyden | CuiYifeng | module: ut, skipped | aten_ops | ut |
| 2766 | MaxPool2d - investigate memory layout pe | BBBela | pbielak |  | aten_ops | ut |

*... and 101 more issues*
