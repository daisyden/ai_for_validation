# Torch XPU Ops Issue Report

**Repository:** [intel/torch-xpu-ops](https://github.com/intel/torch-xpu-ops)

**Issues Extracted:** April 5, 2026 (from GitHub issue data: `torch_xpu_ops_issues.json`)

**CI Data Sources:**
- Torch-XPU-OPS Nightly CI: XML files from `Inductor-XPU-UT-Data-*` + `Inductor-wheel-nightly-LTS2-XPU-E2E-Data-*` (commit: `a2d516a58c64f18b76880f3a77efbc02885d65af`)
- Stock PyTorch XPU CI: XML files from `test-default-*-linux.idc.xpu_*.zip` (run IDs: 69741866812, 69741866834, 69741866862, 69741866911, etc.)

Generated: 2026-04-08 08:16:41

## Summary

| Category | Count |
|----------|-------|
| Action Required | 96 |
| No Assignee | 62 |
| Duplicated Issues | 42 |
| With Dependency | 25 |
| Others | 192 |
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
| Close fixed issue | 17 |
| Enable test | 5 |
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

### Other Stats

| Category | Count |
|----------|-------|
| Not Assigned | 62 |
| Duplicated Issues | 42 |
| Others | 192 |

---

## 1. Action Required

Issues that need action based on test results analysis.


### 1.1 Close fixed issue

| ID | Title | Owner | Owner Transferred | TBD | Priority | Reason | Category | PR | Module | Test Module |
|---|-------|-------|-------------------|-----|---------|--------|----------|-----|--------|-------------|
| [1624](https://github.com/intel/torch-xpu-ops/issues/1624) | [DONT CLOSE] Known UT Issue list | None | RUIJIEZHONG66166 | Close fixed issue | P2 | UT issue with few failures | Dtype / Precision Related |  | distributed | ut |
| [2496](https://github.com/intel/torch-xpu-ops/issues/2496) | [upstream_ut]  Segmentation fault w | astachowiczhabana | libohao1201 | Close fixed issue | P0 | Build crash - critical blocking issue | Dtype / Precision Related |  | aten_ops | ut |
| [2518](https://github.com/intel/torch-xpu-ops/issues/2518) | [upstream_ut]  TypeError: Creating  | astachowiczhabana | libohao1201 | Close fixed issue | P2 | UT issue with few failures | Others |  | aten_ops | ut |
| [2592](https://github.com/intel/torch-xpu-ops/issues/2592) | [release/2.10] models got fail_accu | mengfei25 | mengfei25 | Close fixed issue | P0 | Impacts real model/application | Dtype / Precision Related |  | aten_ops | e2e |
| [2619](https://github.com/intel/torch-xpu-ops/issues/2619) | [release/2.10] Some models inductor | jianyizh, weishi-deng | mengfei25 | Close fixed issue | P0 | Impacts real model/application | Inductor / Compilation Related |  | aten_ops | e2e |
| [2953](https://github.com/intel/torch-xpu-ops/issues/2953) | [release/2.11][wsl] huggingface TrO | xuhancn | bjarzemb | Close fixed issue | P0 | Impacts real model/application | Others |  | aten_ops | e2e |
| [2981](https://github.com/intel/torch-xpu-ops/issues/2981) | [release/2.11] T5 models performanc | jianyizh, weishi-deng | mengfei25 | Close fixed issue | P0 | Impacts real model/application | Others |  | aten_ops | e2e |
| [3011](https://github.com/intel/torch-xpu-ops/issues/3011) | [upstream_ut] torch.OutOfMemoryErro | None | Silv3S | Close fixed issue | P2 | UT issue with few failures | Others |  | aten_ops | ut |
| [3013](https://github.com/intel/torch-xpu-ops/issues/3013) | [upstream_ut] RuntimeError: Kernel  | None | Silv3S | Close fixed issue | P2 | UT issue with few failures | Inductor / Compilation Related |  | unknown | ut |
| [3014](https://github.com/intel/torch-xpu-ops/issues/3014) | [upstream_ut] AssertionError: False | None | Silv3S | Close fixed issue | P2 | UT issue with few failures | Dtype / Precision Related |  | unknown | ut |
| [3058](https://github.com/intel/torch-xpu-ops/issues/3058) | [E2E] hf_GPT2_large amp_fp16/amp_bf | weishi-deng | kaileiyx | Close fixed issue | P1 | E2E benchmark accuracy/functionality issue | Flash Attention / Transformer Related |  | aten_ops | e2e |
| [3106](https://github.com/intel/torch-xpu-ops/issues/3106) | Worker crashes when running TestDec | BBBela | BBBela | Close fixed issue | P0 | Build crash - critical blocking issue | Others |  | aten_ops | ut |
| [3157](https://github.com/intel/torch-xpu-ops/issues/3157) | XPU out of memory. Tried to allocat | kdrozd-dev | kdrozd-dev | Close fixed issue | P2 | UT issue with few failures | Others |  | aten_ops | ut |
| [3158](https://github.com/intel/torch-xpu-ops/issues/3158) | AttributeError: module 'triton.comp | tadkrawiec | kdrozd-dev | Close fixed issue | P2 | UT issue with few failures | Inductor / Compilation Related |  | aten_ops | ut |
| [3161](https://github.com/intel/torch-xpu-ops/issues/3161) | Exception: Tensor-likes are not clo | tadkrawiec | kdrozd-dev | Close fixed issue | P2 | UT issue with few failures | Dtype / Precision Related |  | aten_ops | ut |
| [3166](https://github.com/intel/torch-xpu-ops/issues/3166) | test_consistency_SparseCSR failures | yucai-intel | CuiYifeng | Close fixed issue | P2 | UT issue with few failures | TorchAO |  | aten_ops | ut |

### 1.2 Enable test

| ID | Title | Owner | Owner Transferred | TBD | Priority | Reason | Category | PR | Module | Test Module |
|---|-------|-------|-------------------|-----|---------|--------|----------|-----|--------|-------------|
| [2024](https://github.com/intel/torch-xpu-ops/issues/2024) | AssertionError: Torch not compiled  | daisyden | daisyden | Enable test | P2 | UT issue with few failures | Distributed |  | aten_ops | ut |
| [2132](https://github.com/intel/torch-xpu-ops/issues/2132) | [2.9][BMG-Windows][Torch-xpu-ops UT | pbielak | daisyden | Enable test | P2 | UT issue with few failures | Inductor / Compilation Related |  | aten_ops | ut |
| [2531](https://github.com/intel/torch-xpu-ops/issues/2531) | [upstream_ut]  AssertionError: Torc | daisyden | daisyden | Enable test | P2 | UT issue with few failures | Distributed |  | unknown | ut |
| [3242](https://github.com/intel/torch-xpu-ops/issues/3242) | AssertionError: Torch not compiled  | None | daisyden | Enable test | P2 | UT issue with few failures | Inductor / Compilation Related |  | aten_ops | ut |

### 1.3 Needs PyTorch Repo Changes (upstream)

| ID | Title | Owner | Owner Transferred | TBD | Priority | Reason | Category | PR | Module | Test Module |
|---|-------|-------|-------------------|-----|---------|--------|----------|-----|--------|-------------|
| [1505](https://github.com/intel/torch-xpu-ops/issues/1505) | [ARC-WSL-Ubuntu24.04] 15 Timm model | xuhancn, Stonepia | xuhancn, Stonepia | Needs PyTorch Repo Changes (upstream) | P0 | Impacts real model/application | Dtype / Precision Related |  | inductor | e2e |
| [1963](https://github.com/intel/torch-xpu-ops/issues/1963) | [upstream_ut] MetadataMismatchError | pbielak | pbielak | Needs PyTorch Repo Changes (upstream) | P2 | UT issue with few failures | Others | [PR](https://github.com/pytorch/pytorch/pull/178277, https://github.com/pytorch/pytorch/pull/175965) | aten_ops | ut |
| [2214](https://github.com/intel/torch-xpu-ops/issues/2214) | test/test_sparse.py::TestSparseAnyX | jenniew | jenniew | Needs PyTorch Repo Changes (upstream) | P2 | UT issue with few failures | Distributed |  | aten_ops | ut |
| [2234](https://github.com/intel/torch-xpu-ops/issues/2234) | [upstream_ut] AssertionError: Runti | Silv3S | Silv3S | Needs PyTorch Repo Changes (upstream) | P2 | UT issue with few failures | Distributed |  | aten_ops | ut |
| [2248](https://github.com/intel/torch-xpu-ops/issues/2248) | [upstream_ut] test_cow failures | gplutop7 | gplutop7 | Needs PyTorch Repo Changes (upstream) | P2 | UT issue with few failures | Distributed |  | aten_ops | ut |
| [2251](https://github.com/intel/torch-xpu-ops/issues/2251) | [upstream_ut] test_fake_autocase go | astachowiczhabana | astachowiczhabana | Needs PyTorch Repo Changes (upstream) | P2 | UT issue with few failures | Dtype / Precision Related |  | aten_ops | ut |
| [2255](https://github.com/intel/torch-xpu-ops/issues/2255) | [upstream_ut] RuntimeError: Long is | daisyden | daisyden | Needs PyTorch Repo Changes (upstream) | P2 | UT issue with few failures | Flash Attention / Transformer Related |  | aten_ops | ut |
| [2283](https://github.com/intel/torch-xpu-ops/issues/2283) | [upstream_ut] sparse._sampled_addmm | jenniew | jenniew | Needs PyTorch Repo Changes (upstream) | P1 | UT with 21 failed test cases | Distributed |  | aten_ops | ut |
| [2287](https://github.com/intel/torch-xpu-ops/issues/2287) | [upstream_ut] test_python_ref issue | yucai-intel | yucai-intel | Needs PyTorch Repo Changes (upstream) | P2 | UT issue with few failures | Others |  | aten_ops | ut |
| [2295](https://github.com/intel/torch-xpu-ops/issues/2295) | [upstream_ut][xpu][test]nn/test_emb | yucai-intel | yucai-intel | Needs PyTorch Repo Changes (upstream) | P2 | UT issue with few failures | Dtype / Precision Related |  | inductor | ut |
| [2301](https://github.com/intel/torch-xpu-ops/issues/2301) | [upstream_ut] dtypes not align with | daisyden | daisyden | Needs PyTorch Repo Changes (upstream) | P2 | UT issue with few failures | Distributed |  | aten_ops | ut |
| [2329](https://github.com/intel/torch-xpu-ops/issues/2329) | [upstream_ut] feature missing: get_ | etaf | etaf | Needs PyTorch Repo Changes (upstream) | P2 | UT issue with few failures | Others |  | inductor | ut |
| [2359](https://github.com/intel/torch-xpu-ops/issues/2359) | [upstream_ut] GradcheckError: Backw | BBBela | BBBela | Needs PyTorch Repo Changes (upstream) | P2 | UT issue with few failures | Distributed |  | aten_ops | ut |
| [2554](https://github.com/intel/torch-xpu-ops/issues/2554) | [upstream_ut]  AssertionError: Asse | daisyden | daisyden | Needs PyTorch Repo Changes (upstream) | P2 | UT issue with few failures | PT2E |  | inductor | ut |
| [2578](https://github.com/intel/torch-xpu-ops/issues/2578) | [TorchAO][UT] test/quantization/tes | Stonepia | Stonepia | Needs PyTorch Repo Changes (upstream) | P0 | Build crash - critical blocking issue | TorchAO |  | AO | build |
| [2609](https://github.com/intel/torch-xpu-ops/issues/2609) | [upstream_ut]  torch._inductor.exc. | daisyden | daisyden | Needs PyTorch Repo Changes (upstream) | P2 | UT issue with few failures | PT2E | [PR](https://github.com/pytorch/pytorch/pull/171154) | inductor | ut |
| [2620](https://github.com/intel/torch-xpu-ops/issues/2620) | [upstream_ut]  AssertionError: dtyp | daisyden | daisyden | Needs PyTorch Repo Changes (upstream) | P2 | UT issue with few failures | TorchAO |  | inductor | ut |
| [2663](https://github.com/intel/torch-xpu-ops/issues/2663) | test_sparse_semi_structured.py gaps | None | None | Needs PyTorch Repo Changes (upstream) | P2 | UT issue with few failures | Sparse Operations Related |  | aten_ops | ut |
| [2670](https://github.com/intel/torch-xpu-ops/issues/2670) | [upstream_ut]  RuntimeError: could  | tszulist-hbn | tszulist-hbn | Needs PyTorch Repo Changes (upstream) | P2 | UT issue with few failures | Distributed |  | aten_ops | ut |
| [2693](https://github.com/intel/torch-xpu-ops/issues/2693) | Title: [upstream_ut]  AssertionErro | hoshibara | hoshibara | Needs PyTorch Repo Changes (upstream) | P2 | UT issue with few failures | Distributed |  | inductor | ut |
| [2696](https://github.com/intel/torch-xpu-ops/issues/2696) | Title: [upstream_ut]  RuntimeError: | chunhuanMeng | chunhuanMeng | Needs PyTorch Repo Changes (upstream) | P2 | UT issue with few failures | Flash Attention / Transformer Related |  | inductor | e2e |
| [2697](https://github.com/intel/torch-xpu-ops/issues/2697) | Title: [upstream_ut]  RuntimeError: | chunhuanMeng | chunhuanMeng | Needs PyTorch Repo Changes (upstream) | P2 | UT issue with few failures | Flash Attention / Transformer Related |  | inductor | e2e |
| [2698](https://github.com/intel/torch-xpu-ops/issues/2698) | Title: [upstream_ut]  RuntimeError: | chunhuanMeng, LuFinch | chunhuanMeng, LuFinch | Needs PyTorch Repo Changes (upstream) | P2 | UT issue with few failures | PT2E |  | inductor | ut |
| [2704](https://github.com/intel/torch-xpu-ops/issues/2704) | Title: [upstream_ut]  AssertionErro | kdrozd-dev | kdrozd-dev | Needs PyTorch Repo Changes (upstream) | P2 | UT issue with few failures | Flash Attention / Transformer Related | [PR](https://github.com/pytorch/pytorch/pull/177636) | aten_ops | ut |
| [2712](https://github.com/intel/torch-xpu-ops/issues/2712) | [upstream_ut]  RuntimeError: Cannot | tszulist-hbn | tszulist-hbn | Needs PyTorch Repo Changes (upstream) | P2 | UT issue with few failures | Others |  | aten_ops | ut |
| [2714](https://github.com/intel/torch-xpu-ops/issues/2714) | [upstream_ut]  AssertionError: Obje | Silv3S | Silv3S | Needs PyTorch Repo Changes (upstream) | P2 | UT issue with few failures | Distributed |  | aten_ops | ut |
| [2715](https://github.com/intel/torch-xpu-ops/issues/2715) | [upstream_ut]  torch._dynamo.exc.Un | CuiYifeng | CuiYifeng | Needs PyTorch Repo Changes (upstream) | P2 | UT issue with few failures | PT2E |  | aten_ops | ut |
| [2798](https://github.com/intel/torch-xpu-ops/issues/2798) | Test case  test/test_dlpack.py::Tes | None | None | Needs PyTorch Repo Changes (upstream) | P2 | UT issue with few failures | Others |  | aten_ops | ut |
| [2800](https://github.com/intel/torch-xpu-ops/issues/2800) | AttributeError: 'torch._C._XpuDevic | guangyey | guangyey | Needs PyTorch Repo Changes (upstream) | P2 | UT issue with few failures | Flash Attention / Transformer Related |  | inductor | ut |
| [2802](https://github.com/intel/torch-xpu-ops/issues/2802) | Three aten._scaled_dot_product_flas | LuFinch | LuFinch | Needs PyTorch Repo Changes (upstream) | P2 | UT issue with few failures | Flash Attention / Transformer Related |  | inductor | ut |
| [2806](https://github.com/intel/torch-xpu-ops/issues/2806) | CompiledAOTI need XPU support | daisyden | daisyden | Needs PyTorch Repo Changes (upstream) | P2 | UT issue with few failures | PT2E |  | inductor | ut |
| [2810](https://github.com/intel/torch-xpu-ops/issues/2810) | AssertionError: Object comparison f | daisyden | daisyden | Needs PyTorch Repo Changes (upstream) | P2 | UT issue with few failures | Dtype / Precision Related |  | inductor | ut |
| [2888](https://github.com/intel/torch-xpu-ops/issues/2888) | torch._inductor.exc.InductorError:  | Stonepia | Stonepia | Needs PyTorch Repo Changes (upstream) | P2 | UT issue with few failures | Inductor / Compilation Related |  | inductor | ut |
| [2891](https://github.com/intel/torch-xpu-ops/issues/2891) | RuntimeError: Expected to find "(26 | chunhuanMeng | chunhuanMeng | Needs PyTorch Repo Changes (upstream) | P2 | UT issue with few failures | Others |  | inductor | e2e |
| [2918](https://github.com/intel/torch-xpu-ops/issues/2918) | [XPU][upstream_ut][COW] Skip non-su | gplutop7 | gplutop7 | Needs PyTorch Repo Changes (upstream) | P2 | UT issue with few failures | Others | [PR](https://github.com/pytorch/pytorch/pull/174670) | aten_ops | ut |
| [2919](https://github.com/intel/torch-xpu-ops/issues/2919) | [XPU][upstream_ut][COW] Fix materia | gplutop7 | gplutop7 | Needs PyTorch Repo Changes (upstream) | P2 | UT issue with few failures | Others |  | aten_ops | ut |
| [2958](https://github.com/intel/torch-xpu-ops/issues/2958) | AssertionError of test_dtensor_basi | daisyden | daisyden | Needs PyTorch Repo Changes (upstream) | P2 | UT issue with few failures | Inductor / Compilation Related |  | inductor | ut |
| [2997](https://github.com/intel/torch-xpu-ops/issues/2997) | AssertionError of test_linear_and_c | etaf | etaf | Needs PyTorch Repo Changes (upstream) | P2 | UT issue with few failures | Flash Attention / Transformer Related |  | inductor | ut |
| [2999](https://github.com/intel/torch-xpu-ops/issues/2999) | KeyError: 'eager_numerics.use_pytor | daisyden | daisyden | Needs PyTorch Repo Changes (upstream) | P2 | UT issue with few failures | Others |  | inductor | ut |
| [3004](https://github.com/intel/torch-xpu-ops/issues/3004) | TypeError: _xpu_recordMemoryHistory | guangyey | guangyey | Needs PyTorch Repo Changes (upstream) | P2 | UT issue with few failures | Others |  | inductor | ut |
| [3006](https://github.com/intel/torch-xpu-ops/issues/3006) | AssertionError: '.to(tl.float16)' u | CuiYifeng | CuiYifeng | Needs PyTorch Repo Changes (upstream) | P2 | UT issue with few failures | PT2E |  | inductor | e2e |
| [3041](https://github.com/intel/torch-xpu-ops/issues/3041) | AssertionError: Expected len(flat_d | Silv3S | Silv3S | Needs PyTorch Repo Changes (upstream) | P2 | UT issue with few failures | Dtype / Precision Related |  | aten_ops | ut |
| [3077](https://github.com/intel/torch-xpu-ops/issues/3077) | [Bug Skip] test_dlpack.py::TestTorc | AKloniecki | AKloniecki | Needs PyTorch Repo Changes (upstream) | P2 | UT issue with few failures | Others |  | aten_ops | ut |
| [3094](https://github.com/intel/torch-xpu-ops/issues/3094) | XPUGraph tree support | None | None | Needs PyTorch Repo Changes (upstream) | P2 | UT issue with few failures | Inductor / Compilation Related |  | inductor | ut |
| [3095](https://github.com/intel/torch-xpu-ops/issues/3095) | cutlass support blocks some unit te | None | None | Needs PyTorch Repo Changes (upstream) | P2 | UT issue with few failures | Others |  | inductor | ut |
| [3126](https://github.com/intel/torch-xpu-ops/issues/3126) | [upstream_ut]  Two NestedTensor iss | LuFinch | LuFinch | Needs PyTorch Repo Changes (upstream) | P2 | UT issue with few failures | Flash Attention / Transformer Related |  | aten_ops | ut |
| [3127](https://github.com/intel/torch-xpu-ops/issues/3127) | [upstream_ut]  AssertionError: Asse | LuFinch | LuFinch | Needs PyTorch Repo Changes (upstream) | P2 | UT issue with few failures | Flash Attention / Transformer Related |  | aten_ops | ut |
| [3128](https://github.com/intel/torch-xpu-ops/issues/3128) | [upstream_ut]  AssertionError: Runt | LuFinch | LuFinch | Needs PyTorch Repo Changes (upstream) | P2 | UT issue with few failures | Distributed |  | aten_ops | ut |
| [3129](https://github.com/intel/torch-xpu-ops/issues/3129) | [upstream_ut]  AssertionError: User | LuFinch | LuFinch | Needs PyTorch Repo Changes (upstream) | P2 | UT issue with few failures | Flash Attention / Transformer Related |  | aten_ops | ut |
| [3130](https://github.com/intel/torch-xpu-ops/issues/3130) | [upstream_ut]  AssertionError: tens | LuFinch | LuFinch | Needs PyTorch Repo Changes (upstream) | P2 | UT issue with few failures | Flash Attention / Transformer Related |  | aten_ops | ut |
| [3131](https://github.com/intel/torch-xpu-ops/issues/3131) | [upstream_ut]  NotImplementedError: | chunhuanMeng | chunhuanMeng | Needs PyTorch Repo Changes (upstream) | P2 | UT issue with few failures | Flash Attention / Transformer Related |  | aten_ops | ut |
| [3132](https://github.com/intel/torch-xpu-ops/issues/3132) | [upstream_ut]  transfomers test rep | LuFinch | LuFinch | Needs PyTorch Repo Changes (upstream) | P2 | UT issue with few failures | Flash Attention / Transformer Related |  | aten_ops | ut |
| [3133](https://github.com/intel/torch-xpu-ops/issues/3133) | [upstream_ut]  RuntimeError: scaled | LuFinch | LuFinch | Needs PyTorch Repo Changes (upstream) | P2 | UT issue with few failures | Flash Attention / Transformer Related |  | aten_ops | ut |
| [3137](https://github.com/intel/torch-xpu-ops/issues/3137) | [upstream_ut]  RuntimeError: expect | LuFinch | LuFinch | Needs PyTorch Repo Changes (upstream) | P2 | UT issue with few failures | Flash Attention / Transformer Related |  | aten_ops | ut |
| [3140](https://github.com/intel/torch-xpu-ops/issues/3140) | [upstream_ut]  RuntimeError: FlashA | LuFinch | LuFinch | Needs PyTorch Repo Changes (upstream) | P2 | UT issue with few failures | Flash Attention / Transformer Related |  | aten_ops | ut |
| [3141](https://github.com/intel/torch-xpu-ops/issues/3141) | [upstream_ut]  RuntimeError: FlashA | LuFinch | LuFinch | Needs PyTorch Repo Changes (upstream) | P2 | UT issue with few failures | Flash Attention / Transformer Related |  | aten_ops | ut |
| [3142](https://github.com/intel/torch-xpu-ops/issues/3142) | [upstream_ut]  RuntimeError: The sy | LuFinch | LuFinch | Needs PyTorch Repo Changes (upstream) | P2 | UT issue with few failures | Flash Attention / Transformer Related |  | aten_ops | ut |
| [3143](https://github.com/intel/torch-xpu-ops/issues/3143) | NotImplementedError: The operator ' | LuFinch | LuFinch | Needs PyTorch Repo Changes (upstream) | P2 | UT issue with few failures | Flash Attention / Transformer Related |  | aten_ops | ut |
| [3163](https://github.com/intel/torch-xpu-ops/issues/3163) | [Bug Skip]: Object comparison faile | chunhuanMeng | chunhuanMeng | Needs PyTorch Repo Changes (upstream) | P2 | UT issue with few failures | Distributed |  | aten_ops | ut |
| [3165](https://github.com/intel/torch-xpu-ops/issues/3165) | test_sparse_csr_xpu.py::TestSparseC | None | None | Needs PyTorch Repo Changes (upstream) | P2 | UT issue with few failures | Sparse Operations Related |  | aten_ops | ut |
| [3167](https://github.com/intel/torch-xpu-ops/issues/3167) | NotImplementedError: Could not run  | tszulist-hbn | tszulist-hbn | Needs PyTorch Repo Changes (upstream) | P1 | UT with 68 failed test cases | Sparse Operations Related |  | aten_ops | ut |
| [3168](https://github.com/intel/torch-xpu-ops/issues/3168) | NotImplementedError: Could not run  | jenniew | jenniew | Needs PyTorch Repo Changes (upstream) | P1 | UT with 40 failed test cases | Sparse Operations Related |  | aten_ops | ut |
| [3169](https://github.com/intel/torch-xpu-ops/issues/3169) | NotImplementedError: Could not run  | jkosnox | jkosnox | Needs PyTorch Repo Changes (upstream) | P2 | UT issue with few failures | Sparse Operations Related |  | aten_ops | ut |
| [3170](https://github.com/intel/torch-xpu-ops/issues/3170) | Unskip test_bmm_windows_error_xpu_f | jenniew | jenniew | Needs PyTorch Repo Changes (upstream) | P2 | UT issue with few failures | Others |  | aten_ops | ut |
| [3187](https://github.com/intel/torch-xpu-ops/issues/3187) | PyTorch XPU gpu_cpp_wrapper fails w | CuiYifeng | CuiYifeng | Needs PyTorch Repo Changes (upstream) | P2 | UT issue with few failures | Inductor / Compilation Related |  | aten_ops | ut |
| [3229](https://github.com/intel/torch-xpu-ops/issues/3229) | RuntimeError: No viable backend for | None | None | Needs PyTorch Repo Changes (upstream) | P2 | UT issue with few failures | Flash Attention / Transformer Related |  | aten_ops | ut |
| [3238](https://github.com/intel/torch-xpu-ops/issues/3238) | The supported dtypes of _refs.stft  | CuiYifeng | CuiYifeng | Needs PyTorch Repo Changes (upstream) | P2 | UT issue with few failures | Dtype / Precision Related |  | aten_ops | ut |
| [3247](https://github.com/intel/torch-xpu-ops/issues/3247) | NotImplementedError: "dot_xpu_mkl"  | Silv3S | Silv3S | Needs PyTorch Repo Changes (upstream) | P2 | UT issue with few failures | Others |  | aten_ops | ut |

### 1.4 Revisit the PR as case failed

| ID | Title | Owner | Owner Transferred | TBD | Priority | Reason | Category | PR | Module | Test Module |
|---|-------|-------|-------------------|-----|---------|--------|----------|-----|--------|-------------|
| [3246](https://github.com/intel/torch-xpu-ops/issues/3246) | AssertionError: Booleans mismatch:  | BartoszKokoszko | BartoszKokoszko | Revisit the PR as case failed | P2 | UT issue with few failures | Distributed | [PR](https://github.com/intel/torch-xpu-ops/pull/3249) | aten_ops | ut |

### 1.5 Verify the issue

| ID | Title | Owner | Owner Transferred | TBD | Priority | Reason | Category | PR | Module | Test Module |
|---|-------|-------|-------------------|-----|---------|--------|----------|-----|--------|-------------|
| [2331](https://github.com/intel/torch-xpu-ops/issues/2331) | [upstream_ut] AssertionError: Scala | hoshibara | daisyden | Verify the issue | P2 | UT issue with few failures | Dtype / Precision Related | [PR](https://github.com/pytorch/pytorch/pull/172314) | inductor | ut |
| [2694](https://github.com/intel/torch-xpu-ops/issues/2694) | Title: [upstream_ut]  AssertionErro | daisyden | daisyden | Verify the issue | P2 | UT issue with few failures | Distributed | [PR](https://github.com/pytorch/pytorch/pull/171773) | inductor | ut |
| [3007](https://github.com/intel/torch-xpu-ops/issues/3007) | AssertionError: Scalars are not equ | daisyden | daisyden | Verify the issue | P2 | UT issue with few failures | Flash Attention / Transformer Related | [PR](https://github.com/pytorch/pytorch/pull/178369) | inductor | e2e |

### 1.6 add to skiplist

| ID | Title | Owner | Owner Transferred | TBD | Priority | Reason | Category | PR | Module | Test Module |
|---|-------|-------|-------------------|-----|---------|--------|----------|-----|--------|-------------|
| [2164](https://github.com/intel/torch-xpu-ops/issues/2164) | skip test_no_cuda_monkeypatch as it | daisyden | daisyden | add to skiplist | P2 | UT issue with few failures | Distributed |  | aten_ops | ut |
| [2309](https://github.com/intel/torch-xpu-ops/issues/2309) | unsupported ops with PYTORCH_ENABLE | daisyden | daisyden | add to skiplist | P2 | UT issue with few failures | TorchAO |  | aten_ops | ut |
| [2472](https://github.com/intel/torch-xpu-ops/issues/2472) | [upstream_ut]  NotImplementedError: | Silv3S | daisyden | add to skiplist | P2 | UT issue with few failures | PT2E |  | aten_ops | ut |
| [2508](https://github.com/intel/torch-xpu-ops/issues/2508) | TypedStorage / TypedTensors depreca | Silv3S | libohao1201 | add to skiplist | P1 | UT with 27 failed test cases | TorchAO | [PR](https://github.com/intel/torch-xpu-ops/pull/3260) | aten_ops | ut |

### Issues without Assignee

| ID | Title | Owner | Owner Transferred | TBD | Priority | Reason | Category | PR | Module | Test Module |
|---|-------|-------|-------------------|-----|---------|--------|----------|-----|--------|-------------|
| [3258](https://github.com/intel/torch-xpu-ops/issues/3258) | Error in op: torch.ops.aten._scaled | None | chuanqi | assign owner | P2 | UT issue with few failures | Flash Attention / Transformer Related |  | aten_ops | ut |
| [3257](https://github.com/intel/torch-xpu-ops/issues/3257) | [Linux][E2E][Regression] Huggingfac | None | chuanqi | assign owner | P0 | Impacts real model/application | Others |  | aten_ops | e2e |
| [3255](https://github.com/intel/torch-xpu-ops/issues/3255) | [Linux][PT2E][Regression] Some perf | None | chuanqi | assign owner | P0 | Regression - passed before but failed now | TorchAO |  | aten_ops | e2e |
| [3233](https://github.com/intel/torch-xpu-ops/issues/3233) | [distributed] RuntimeError: No back | None | chuanqi | assign owner | P2 | UT issue with few failures | Distributed |  | distributed | ut |
| [3232](https://github.com/intel/torch-xpu-ops/issues/3232) | [distributed][tensor] AssertionErro | None | chuanqi | assign owner | P2 | UT issue with few failures | Distributed |  | distributed | ut |
| [3231](https://github.com/intel/torch-xpu-ops/issues/3231) | Dynamo failed to run FX node with f | None | chuanqi | assign owner | P2 | UT issue with few failures | PT2E |  | aten_ops | ut |
| [3195](https://github.com/intel/torch-xpu-ops/issues/3195) | test_sdpa_unbacked_no_dde_xpu crash | None | chuanqi | assign owner | P0 | Build crash - critical blocking issue | Flash Attention / Transformer Related |  | aten_ops | ut |
| [3180](https://github.com/intel/torch-xpu-ops/issues/3180) | [E2E] Timm/Torchbench models got "e | None | chuanqi | assign owner | P0 | Impacts real model/application | Others |  | aten_ops | ut |
| [3174](https://github.com/intel/torch-xpu-ops/issues/3174) | [Bug Skip]: Accuracy failure of tes | None | chuanqi | assign owner | P2 | UT issue with few failures | Distributed |  | aten_ops | ut |
| [3151](https://github.com/intel/torch-xpu-ops/issues/3151) | [Triton] Timm_models  rexnet_100 /  | None | chuanqi | assign owner | P0 | Impacts real model/application | Inductor / Compilation Related |  | aten_ops | e2e |
| [3149](https://github.com/intel/torch-xpu-ops/issues/3149) | New failure in test_rms_norm_decomp | None | chuanqi | assign owner | P2 | UT issue with few failures | Flash Attention / Transformer Related |  | aten_ops | ut |
| [3148](https://github.com/intel/torch-xpu-ops/issues/3148) | [Triton] Huggingface openai/whisper | None | chuanqi | assign owner | P0 | Impacts real model/application | Inductor / Compilation Related |  | aten_ops | e2e |
| [3124](https://github.com/intel/torch-xpu-ops/issues/3124) | [TorchAO][Bug] ImportError: Require | None | chuanqi | assign owner | P2 | UT issue with few failures | TorchAO |  | aten_ops | ut |
| [3121](https://github.com/intel/torch-xpu-ops/issues/3121) | [Bug Skip]: CUDA specific UT test_f | None | chuanqi | assign owner | P1 | UT with 60 failed test cases | Dtype / Precision Related |  | aten_ops | ut |
| [3102](https://github.com/intel/torch-xpu-ops/issues/3102) | [distributed] RuntimeError: Invalid | None | chuanqi | assign owner | P2 | UT issue with few failures | Distributed |  | distributed | ut |
| [3101](https://github.com/intel/torch-xpu-ops/issues/3101) | [distributed] 'torch._C._distribute | None | chuanqi | assign owner | P2 | UT issue with few failures | Distributed |  | distributed | ut |
| [3100](https://github.com/intel/torch-xpu-ops/issues/3100) | [distributed] /handler/dump_nccl_tr | None | chuanqi | assign owner | P2 | UT issue with few failures | Distributed |  | distributed | ut |
| [3096](https://github.com/intel/torch-xpu-ops/issues/3096) | VISIBLE_DEVICE support | None | chuanqi | assign owner | P2 | UT issue with few failures | Others |  | aten_ops | ut |
| [3093](https://github.com/intel/torch-xpu-ops/issues/3093) | XPU does not support NestedTensor f | None | chuanqi | assign owner | P2 | UT issue with few failures | Flash Attention / Transformer Related |  | aten_ops | ut |
| [3086](https://github.com/intel/torch-xpu-ops/issues/3086) | nvml support blocks some test cases | None | chuanqi | assign owner | P2 | UT issue with few failures | Others |  | aten_ops | ut |
| [3084](https://github.com/intel/torch-xpu-ops/issues/3084) | torch.library.register_autocast doe | None | chuanqi | assign owner | P2 | UT issue with few failures | Others |  | aten_ops | ut |
| [3083](https://github.com/intel/torch-xpu-ops/issues/3083) | [Bug Skip]: Random failures 2026WW1 | None | chuanqi | assign owner | P2 | UT issue with few failures | Others |  | unknown | ut |
| [3082](https://github.com/intel/torch-xpu-ops/issues/3082) | multithread support in distributed | None | chuanqi | assign owner | P2 | UT issue with few failures | Distributed |  | distributed | ut |
| [3081](https://github.com/intel/torch-xpu-ops/issues/3081) | Sparse CSR gemm-like ops have not b | None | chuanqi | assign owner | P2 | UT issue with few failures | Sparse Operations Related |  | aten_ops | ut |
| [3080](https://github.com/intel/torch-xpu-ops/issues/3080) | cudagraph tests blocked by feature  | None | chuanqi | assign owner | P2 | UT issue with few failures | Inductor / Compilation Related |  | aten_ops | ut |
| [3076](https://github.com/intel/torch-xpu-ops/issues/3076) | [TorchAO][BMG] Llama-3.2-1B-Instruc | None | chuanqi | assign owner | P2 | UT issue with few failures | TorchAO |  | aten_ops | ut |
| [3025](https://github.com/intel/torch-xpu-ops/issues/3025) | New failing test in Nightly Wheel t | None | chuanqi | assign owner | P2 | UT issue with few failures | Distributed |  | aten_ops | ut |
| [2979](https://github.com/intel/torch-xpu-ops/issues/2979) | eca_halonext26ts got RuntimeError:  | None | chuanqi | assign owner | P0 | Build crash - critical blocking issue | Others |  | aten_ops | e2e |
| [2965](https://github.com/intel/torch-xpu-ops/issues/2965) | [Bug Skip]: Random failures 2026WW1 | None | chuanqi | assign owner | P2 | UT issue with few failures | Sparse Operations Related |  | unknown | ut |
| [2960](https://github.com/intel/torch-xpu-ops/issues/2960) | [release/2.11] timm_models_xcit_lar | None | chuanqi | assign owner | P0 | Impacts real model/application | Dtype / Precision Related |  | aten_ops | ut |
| [2948](https://github.com/intel/torch-xpu-ops/issues/2948) | [AO] Benchmark enabling on XPU | None | chuanqi | assign owner | P2 | UT issue with few failures | Others |  | AO | ut |
| [2946](https://github.com/intel/torch-xpu-ops/issues/2946) | [Bug Skip]: Random failures 2026WW0 | None | chuanqi | assign owner | P2 | UT issue with few failures | TorchAO |  | unknown | ut |
| [2930](https://github.com/intel/torch-xpu-ops/issues/2930) | [release/2.11] UT skip test_binary_ | None | chuanqi | assign owner | P2 | UT issue with few failures | Others |  | aten_ops | ut |
| [2914](https://github.com/intel/torch-xpu-ops/issues/2914) | Test case test/test_autograd.py::Te | None | chuanqi | assign owner | P2 | UT issue with few failures | Dtype / Precision Related |  | aten_ops | ut |
| [2912](https://github.com/intel/torch-xpu-ops/issues/2912) | [release/2.11] UT extended 220 new  | None | chuanqi | assign owner | P2 | UT issue with few failures | Others |  | aten_ops | ut |
| [2899](https://github.com/intel/torch-xpu-ops/issues/2899) | Update nan_to_num XPU stub to use s | None | chuanqi | assign owner | P2 | UT issue with few failures | Dtype / Precision Related |  | aten_ops | ut |
| [2858](https://github.com/intel/torch-xpu-ops/issues/2858) | [Bug Skip]: test_xpu new failures | None | chuanqi | assign owner | P2 | UT issue with few failures | Others |  | unknown | ut |
| [2852](https://github.com/intel/torch-xpu-ops/issues/2852) | [Bug Skip]: New UT failures in 0206 | None | chuanqi | assign owner | P2 | UT issue with few failures | Others |  | aten_ops | ut |
| [2845](https://github.com/intel/torch-xpu-ops/issues/2845) | [Bug Skip]:[UT] [Windows] failed ca | None | chuanqi | assign owner | P2 | UT issue with few failures | Others |  | aten_ops | ut |
| [2797](https://github.com/intel/torch-xpu-ops/issues/2797) | Copy error is not raise on test_dlp | None | chuanqi | assign owner | P2 | UT issue with few failures | Others |  | aten_ops | ut |
| [2751](https://github.com/intel/torch-xpu-ops/issues/2751) | [Bug Skip]: Random failures 2026WW0 | None | chuanqi | assign owner | P2 | UT issue with few failures | Sparse Operations Related |  | unknown | ut |
| [2737](https://github.com/intel/torch-xpu-ops/issues/2737) | [distributed] AttributeError: modul | None | chuanqi | assign owner | P2 | UT issue with few failures | Distributed |  | distributed | ut |
| [2676](https://github.com/intel/torch-xpu-ops/issues/2676) | Random failure in CI test | None | chuanqi | assign owner | P2 | UT issue with few failures | Dtype / Precision Related |  | unknown | ut |
| [2656](https://github.com/intel/torch-xpu-ops/issues/2656) | [release/2.10] models got fail_accu | None | chuanqi | assign owner | P0 | Impacts real model/application | Dtype / Precision Related |  | aten_ops | ut |
| [2605](https://github.com/intel/torch-xpu-ops/issues/2605) | [int4][inductor] Add freezing patte | None | chuanqi | assign owner | P2 | UT issue with few failures | TorchAO |  | aten_ops | ut |
| [2596](https://github.com/intel/torch-xpu-ops/issues/2596) | [TorchAO][BMG]Observed accuracy flu | None | chuanqi | assign owner | P0 | Impacts real model/application | TorchAO |  | AO | ut |
| [2595](https://github.com/intel/torch-xpu-ops/issues/2595) | [Bug Skip]: Random crashed cases 20 | None | chuanqi | assign owner | P0 | Build crash - critical blocking issue | Sparse Operations Related |  | aten_ops | ut |
| [2539](https://github.com/intel/torch-xpu-ops/issues/2539) | Title: [upstream_ut]  RuntimeError: | None | chuanqi | assign owner | P2 | UT issue with few failures | Distributed |  | aten_ops | ut |
| [2447](https://github.com/intel/torch-xpu-ops/issues/2447) | test_share_memory_xpu failure | None | chuanqi | assign owner | P2 | UT issue with few failures | Others |  | aten_ops | ut |
| [2446](https://github.com/intel/torch-xpu-ops/issues/2446) | [Bug Skip]: AssertionError: "Simula | None | chuanqi | assign owner | P2 | UT issue with few failures | Distributed |  | aten_ops | ut |

*... and 12 more issues*

---

## 2. Duplicated Issues

Issues that share test cases with other issues.

| ID | Title | Owner | Reporter | Duplicated With | Priority | Reason | Category | PR | Module | Test Module |
|---|-------|-------|----------|-----------------|---------|--------|----------|-----|--------|-------------|
| [1893](https://github.com/intel/torch-xpu-ops/issues/1893) | [upstream_ut] oneDNN accuracy  | chunhuanMeng | daisyden | 1951 | P2 | UT issue with few failures | Distributed |  | aten_ops | ut |
| [1951](https://github.com/intel/torch-xpu-ops/issues/1951) | Functionality issues in TestCo | AKloniecki | daisyden | 1893 | P2 | UT issue with few failures | Distributed |  | aten_ops | ut |
| [1973](https://github.com/intel/torch-xpu-ops/issues/1973) | AssertionError: Scalars or Ten | gplutop7 | mengfei25 | 2837,2840 | P2 | UT issue with few failures | Distributed |  | aten_ops | ut |
| [2006](https://github.com/intel/torch-xpu-ops/issues/2006) | work-item/workgroup issue in s | BartoszKokoszko | daisyden | 2257 | P2 | UT issue with few failures | TorchAO |  | aten_ops | ut |
| [2015](https://github.com/intel/torch-xpu-ops/issues/2015) | inf is returned by nn.Transfor | yucai-intel | daisyden | 2186,2529 | P2 | UT issue with few failures | Flash Attention / Transformer Related |  | aten_ops | ut |
| [2186](https://github.com/intel/torch-xpu-ops/issues/2186) | AssertionError: Mul tiheadAtte | daisyden | daisyden | 2015 | P2 | UT issue with few failures | Flash Attention / Transformer Related |  | aten_ops | ut |
| [2220](https://github.com/intel/torch-xpu-ops/issues/2220) | test/test_sparse_csr.py::TestS | None | wincent8 | 2246 | P2 | UT issue with few failures | TorchAO |  | aten_ops | ut |
| [2230](https://github.com/intel/torch-xpu-ops/issues/2230) | test_sparse_csr.py::TestSparse | None | wincent8 | 2246,3175,3176 | P1 | UT with 28 failed test cases | Distributed |  | aten_ops | ut |
| [2235](https://github.com/intel/torch-xpu-ops/issues/2235) | test/test_sparse_csr.py::TestS | None | wincent8 | 3047 | P2 | UT issue with few failures | Distributed |  | aten_ops | ut |
| [2238](https://github.com/intel/torch-xpu-ops/issues/2238) | Exception: Tensor-likes are no | BBBela | zxd1997066 | 3105 | P2 | UT issue with few failures | Distributed | [PR](https://github.com/intel/torch-xpu-ops/pull/2886) | aten_ops | ut |
| [2244](https://github.com/intel/torch-xpu-ops/issues/2244) | test/test_sparse_csr.py::TestS | jenniew | wincent8 | 3177 | P2 | UT issue with few failures | Distributed |  | aten_ops | ut |
| [2246](https://github.com/intel/torch-xpu-ops/issues/2246) | torch/sparse/_triton_ops*.py n | None | wincent8 | 2220,2230 | P1 | UT with 33 failed test cases | Distributed |  | unknown | ut |
| [2253](https://github.com/intel/torch-xpu-ops/issues/2253) | the supported dtypes are not a | daisyden | daisyden | 2482 | P2 | UT issue with few failures | Dtype / Precision Related |  | aten_ops | ut |
| [2257](https://github.com/intel/torch-xpu-ops/issues/2257) | Accuracy failures in test/xpu/ | pbielak | zxd1997066 | 2006 | P1 | UT with 40 failed test cases | Distributed | [PR](https://github.com/intel/torch-xpu-ops/pull/2808) | aten_ops | ut |
| [2270](https://github.com/intel/torch-xpu-ops/issues/2270) | Backend Compatibility Error in | LuFinch | libohao1201 | 2442 | P2 | UT issue with few failures | Distributed |  | aten_ops | ut |
| [2285](https://github.com/intel/torch-xpu-ops/issues/2285) | Support efficient attention | chunhuanMeng | daisyden | 2358 | P2 | UT issue with few failures | Flash Attention / Transformer Related |  | aten_ops | ut |
| [2358](https://github.com/intel/torch-xpu-ops/issues/2358) | test/test_view_ops.py::TestOld | Silv3S | wincent8 | 2285 | P2 | UT issue with few failures | TorchAO |  | aten_ops | ut |
| [2436](https://github.com/intel/torch-xpu-ops/issues/2436) | [upstream_ut]  AttributeError: | daisyden | daisyden | 2675 | P1 | UT with 51 failed test cases | Flash Attention / Transformer Related |  | aten_ops | ut |
| [2442](https://github.com/intel/torch-xpu-ops/issues/2442) | [Bug Skip]: NotImplementedErro | daisyden, LuFinch | CuiYifeng | 2270 | P2 | UT issue with few failures | Flash Attention / Transformer Related |  | aten_ops | ut |
| [2482](https://github.com/intel/torch-xpu-ops/issues/2482) | test_dtypes issue introduced b | daisyden | daisyden | 2253 | P2 | UT issue with few failures | Dtype / Precision Related |  | aten_ops | ut |
| [2529](https://github.com/intel/torch-xpu-ops/issues/2529) | [upstream_ut]  AssertionError: | Silv3S | daisyden | 2015,3136 | P2 | UT issue with few failures | Distributed |  | aten_ops | ut |
| [2530](https://github.com/intel/torch-xpu-ops/issues/2530) | Title: [upstream_ut]  Assertio | PatrykWilczewski | daisyden | 2817 | P2 | UT issue with few failures | Distributed |  | aten_ops | ut |
| [2611](https://github.com/intel/torch-xpu-ops/issues/2611) | [upstream_ut]  AssertionError: | daisyden | daisyden | 2613 | P2 | UT issue with few failures | Distributed |  | inductor | ut |
| [2613](https://github.com/intel/torch-xpu-ops/issues/2613) | [upstream_ut]  AssertionError: | daisyden | daisyden | 2611 | P2 | UT issue with few failures | Distributed |  | inductor | ut |
| [2618](https://github.com/intel/torch-xpu-ops/issues/2618) | [Bug Skip]: [regression] Asser | jmamzax | kaileiyx | 3089 | P0 | Regression - passed before but failed now | Distributed | [PR](https://github.com/numpy/numpy/pull/22525) | unknown | ut |
| [2675](https://github.com/intel/torch-xpu-ops/issues/2675) | [Bug Skip]: AttributeError: 'N | pponikox | kaileiyx | 2436 | P2 | UT issue with few failures | Flash Attention / Transformer Related |  | aten_ops | ut |
| [2817](https://github.com/intel/torch-xpu-ops/issues/2817) | Expected error message is diff | kdrozd-dev | Silv3S | 2530 | P2 | UT issue with few failures | Dtype / Precision Related |  | aten_ops | ut |
| [2837](https://github.com/intel/torch-xpu-ops/issues/2837) | Accuracy issue for Muon optimi | Silv3S | kdrozd-dev | 1973 | P2 | UT issue with few failures | Distributed |  | aten_ops | ut |
| [2840](https://github.com/intel/torch-xpu-ops/issues/2840) | Accuracy issue with 64 bit ind | SlawomirLaba, Silv3S | kdrozd-dev | 1973 | P2 | UT issue with few failures | Distributed |  | aten_ops | ut |
| [2869](https://github.com/intel/torch-xpu-ops/issues/2869) | [Bug Skip]: New UT failure in  | None | RUIJIEZHONG66166 | 3160 | P2 | UT issue with few failures | Flash Attention / Transformer Related |  | aten_ops | ut |
| [2966](https://github.com/intel/torch-xpu-ops/issues/2966) | [Bug Skip]: [Regression]2026-3 | jmamzax | kaileiyx | 3114 | P0 | Regression - passed before but failed now | Distributed |  | aten_ops | ut |
| [3047](https://github.com/intel/torch-xpu-ops/issues/3047) | [Bug Skip]: [Regression]UT fai | None | kaileiyx | 2235 | P0 | Regression - passed before but failed now | Flash Attention / Transformer Related |  | unknown | ut |
| [3089](https://github.com/intel/torch-xpu-ops/issues/3089) | AssertionError: Torch not comp | jmamzax | jmamzax | 2618 | P2 | UT issue with few failures | TorchAO |  | unknown | ut |
| [3105](https://github.com/intel/torch-xpu-ops/issues/3105) | Wrong results from oneDNN conv | BBBela | BBBela | 2238 | P2 | UT issue with few failures | Distributed |  | aten_ops | ut |
| [3114](https://github.com/intel/torch-xpu-ops/issues/3114) | [Bug Skip]: Failure skip on 20 | None | guangyey | 2966 | P2 | UT issue with few failures | Flash Attention / Transformer Related |  | aten_ops | ut |
| [3136](https://github.com/intel/torch-xpu-ops/issues/3136) | [upstream_ut]  AssertionError: | LuFinch | daisyden | 2529 | P2 | UT issue with few failures | Flash Attention / Transformer Related |  | aten_ops | ut |
| [3156](https://github.com/intel/torch-xpu-ops/issues/3156) | AssertionError: 'Assertion cur | kdrozd-dev | kdrozd-dev | 3184 | P2 | UT issue with few failures | Dtype / Precision Related |  | aten_ops | ut |
| [3160](https://github.com/intel/torch-xpu-ops/issues/3160) | compiler not found (Windows) | kdrozd-dev | kdrozd-dev | 2869 | P2 | UT issue with few failures | Inductor / Compilation Related |  | aten_ops | ut |
| [3175](https://github.com/intel/torch-xpu-ops/issues/3175) | [Bug Skip]: ValueError: sample | None | CuiYifeng | 2230 | P2 | UT issue with few failures | Sparse Operations Related |  | aten_ops | ut |
| [3176](https://github.com/intel/torch-xpu-ops/issues/3176) | [Bug Skip]: ValueError: _scale | None | CuiYifeng | 2230 | P2 | UT issue with few failures | Flash Attention / Transformer Related |  | aten_ops | ut |
| [3177](https://github.com/intel/torch-xpu-ops/issues/3177) | Accuracy gap of BF16/FP16 test | jenniew | CuiYifeng | 2244 | P2 | UT issue with few failures | Distributed |  | aten_ops | ut |
| [3184](https://github.com/intel/torch-xpu-ops/issues/3184) | New failing UTs: test_cross_en | wpietka | BBBela | 3156 | P2 | UT issue with few failures | Dtype / Precision Related |  | aten_ops | ut |

---

## 3. Issues with Dependency

Issues that have dependencies on other components (non-empty).

| ID | Title | Owner | Type | Priority | Reason | Category | Dependency | PR | Labels |
|---|-------|------|------|---------|--------|----------|------------|-----|--------|
| [1059](https://github.com/intel/torch-xpu-ops/issues/1059) | SYCL RT: Using recommende | CuiYifeng, jianyizh | unknown | P2 | UT issue with few failures | Distributed | oneAPI |  | dependency component: oneAPI |
| [1171](https://github.com/intel/torch-xpu-ops/issues/1171) | LNL Windows got unexpecte | xuhancn, chunhuanMeng | functionality bug | P2 | UT issue with few failures | Others | driver |  | client, os: Windows, hw : LNL, hw: BMG, dependency component: driver |
| [1324](https://github.com/intel/torch-xpu-ops/issues/1324) | [Win] UR Error when OOM a | Stonepia | functionality bug | P2 | UT issue with few failures | Others | oneAPI |  | client, os: Windows, module: dependency bug, dependency component: driver, dependency component: oneAPI |
| [1510](https://github.com/intel/torch-xpu-ops/issues/1510) | Some test cases will be h | Stonepia, mengfei25 | functionality bug | P2 | UT issue with few failures | Others | driver |  | bug, client, os: Ubuntu, hw: BMG, dependency component: driver, module: ut |
| [1547](https://github.com/intel/torch-xpu-ops/issues/1547) | [distributed] NotImplemen | Chao1Han | feature request | P2 | UT issue with few failures | Distributed | oneAPI |  | module: distributed, dependency component: oneAPI |
| [1548](https://github.com/intel/torch-xpu-ops/issues/1548) | [distributed] AssertionEr | Chao1Han | functionality bug | P2 | UT issue with few failures | Distributed | oneAPI |  | module: distributed, dependency component: oneAPI |
| [1549](https://github.com/intel/torch-xpu-ops/issues/1549) | [distributed] AssertionEr | Chao1Han | functionality bug | P2 | UT issue with few failures | Distributed | oneAPI |  | module: distributed, dependency component: oneAPI |
| [1551](https://github.com/intel/torch-xpu-ops/issues/1551) | [distributed] NotImplemen | Chao1Han | feature request | P2 | UT issue with few failures | Distributed | oneAPI |  | module: distributed, dependency component: oneAPI |
| [1555](https://github.com/intel/torch-xpu-ops/issues/1555) | [distributed] RuntimeErro | chuanqi129 | functionality bug | P2 | UT issue with few failures | Distributed | oneDNN |  | module: distributed, dependency component: oneDNN |
| [1556](https://github.com/intel/torch-xpu-ops/issues/1556) | [distributed] NotImplemen | pkourdis | feature request | P2 | UT issue with few failures | Distributed | oneDNN |  | module: distributed, dependency component: oneDNN |
| [1649](https://github.com/intel/torch-xpu-ops/issues/1649) | [cpp extension] Provide a | dvrogozh | feature request | P2 | UT issue with few failures | Others | oneAPI |  | dependency component: oneAPI, module: build |
| [1722](https://github.com/intel/torch-xpu-ops/issues/1722) | Ask an API to query GPU t | guangyey | unknown | P2 | UT issue with few failures | Others | oneAPI |  | dependency component: oneAPI |
| [1727](https://github.com/intel/torch-xpu-ops/issues/1727) | [distributed] AttributeEr | guangyey | functionality bug | P2 | UT issue with few failures | Distributed | oneAPI |  | module: distributed, dependency component: oneAPI |
| [1912](https://github.com/intel/torch-xpu-ops/issues/1912) | Implement the torch.ops.a | liangan1 | feature request | P2 | UT issue with few failures | TorchAO | oneDNN |  | dependency component: oneDNN |
| [1962](https://github.com/intel/torch-xpu-ops/issues/1962) | [upstream_ut] segfault wi | jenniew, mengfei25 | functionality bug | P0 | Build crash - critical blocking issue | Sparse Operations Related | driver |  | dependency component: driver, module: ut, skipped |
| [1986](https://github.com/intel/torch-xpu-ops/issues/1986) | torch.xpu._sleep is missi | guangyey | functionality bug | P2 | UT issue with few failures | Others | oneAPI |  | dependency component: oneAPI |
| [2089](https://github.com/intel/torch-xpu-ops/issues/2089) | need an implementation th | guangyey | feature request | P2 | UT issue with few failures | Others | driver |  | dependency component: driver |
| [2157](https://github.com/intel/torch-xpu-ops/issues/2157) | BMG d2h copy is very slow | jianyizh, mengfei25 | functionality bug | P2 | UT issue with few failures | Others | driver |  | performance, dependency component: driver |
| [2200](https://github.com/intel/torch-xpu-ops/issues/2200) | support flash attention o | ElaineBao | feature request | P2 | UT issue with few failures | Flash Attention / Transformer Related | oneDNN |  | dependency component: oneDNN |
| [2261](https://github.com/intel/torch-xpu-ops/issues/2261) | [xpu][profiler] Run with  | moksiuc | functionality bug | P2 | UT issue with few failures | Others | oneAPI |  | dependency component: oneAPI, module: profiler |
| [2439](https://github.com/intel/torch-xpu-ops/issues/2439) | [oneDNN] TestDecompXPU.te | libohao1201 | functionality bug | P2 | UT issue with few failures | Dtype / Precision Related | oneDNN |  | dependency component: oneDNN, module: ut |
| [2467](https://github.com/intel/torch-xpu-ops/issues/2467) | Host may stuck when submi | jianyizh | functionality bug | P2 | UT issue with few failures | Inductor / Compilation Related | driver |  | dependency component: driver |
| [2570](https://github.com/intel/torch-xpu-ops/issues/2570) | crash in sdpa. | LuFinch | functionality bug | P0 | Build crash - critical blocking issue | Flash Attention / Transformer Related | oneDNN |  | dependency component: oneDNN |
| [2655](https://github.com/intel/torch-xpu-ops/issues/2655) | [BMG][OOB] hf_Reformer pe | jianyizh | performance issue | P0 | Regression - passed before but failed now | Others | Triton |  | E2E, dtype: float16, triaged, performance, os: Ubuntu, hw: BMG, dependency component: Triton, regression |
| [2769](https://github.com/intel/torch-xpu-ops/issues/2769) | [oneDNN] New failed test  | LuFinch | functionality bug | P2 | UT issue with few failures | Others | oneDNN |  | hw: PVC, dependency component: oneDNN, module: ut |

---

## 4. Others

Issues that don't fall into the categories above.

| ID | Title | Owner | Priority | Reason | Category | Labels | PR | Module | Test Module |
|---|-------|-------|---------|--------|----------|--------|-----|--------|-------------|
| [3259](https://github.com/intel/torch-xpu-ops/issues/3259) | New failed test cases 2026-04- | SlawomirLaba | P2 | UT issue with few failures | Flash Attention / Transformer Related | skipped |  | aten_ops | ut |
| [3243](https://github.com/intel/torch-xpu-ops/issues/3243) | AssertionError: False is not t | pponikox | P2 | UT issue with few failures | Dtype / Precision Related | module: ut, skipped |  | aten_ops | ut |
| [3236](https://github.com/intel/torch-xpu-ops/issues/3236) | AssertionError: 'def [28 chars | jmamzax | P2 | UT issue with few failures | Dtype / Precision Related | bug_fix_stage5 |  | aten_ops | ut |
| [3227](https://github.com/intel/torch-xpu-ops/issues/3227) | torch xpu event has ~0.1ms lat | guangyey | P2 | UT issue with few failures | Others |  |  | aten_ops | ut |
| [3224](https://github.com/intel/torch-xpu-ops/issues/3224) | [Win][Build] Building SYCL (De | chunhuanMeng | P0 | Build crash - critical blocking issue | Inductor / Compilation Related |  |  | aten_ops | build |
| [3216](https://github.com/intel/torch-xpu-ops/issues/3216) | [OPs] Some ops of XPU have non | CuiYifeng | P2 | UT issue with few failures | Others |  |  | aten_ops | ut |
| [3209](https://github.com/intel/torch-xpu-ops/issues/3209) | [Win][Build] There is Cyclic d | Copilot | P0 | Build crash - critical blocking issue | Others |  |  | aten_ops | build |
| [3206](https://github.com/intel/torch-xpu-ops/issues/3206) | [Bug Skip]: [new failures]Runt | tszulist-hbn | P2 | UT issue with few failures | Others | skipped |  | aten_ops | ut |
| [3196](https://github.com/intel/torch-xpu-ops/issues/3196) | vitals is not supported, the c | libohao1201 | P2 | UT issue with few failures | Others | skipped |  | aten_ops | ut |
| [3194](https://github.com/intel/torch-xpu-ops/issues/3194) | Incorrect strides in TestCommo | AKloniecki | P2 | UT issue with few failures | Distributed |  |  | aten_ops | ut |
| [3191](https://github.com/intel/torch-xpu-ops/issues/3191) | torch._inductor.exc.InductorEr | EikanWang, Copilot | P2 | UT issue with few failures | Inductor / Compilation Related | E2E, hw: PVC |  | aten_ops | e2e |
| [3189](https://github.com/intel/torch-xpu-ops/issues/3189) | Task Tracker | guangyey | P2 | UT issue with few failures | Others |  |  | aten_ops | ut |
| [3178](https://github.com/intel/torch-xpu-ops/issues/3178) | New failed test cases 2026-03- | pponikox | P2 | UT issue with few failures | Flash Attention / Transformer Related | module: ut, skipped |  | aten_ops | ut |
| [3150](https://github.com/intel/torch-xpu-ops/issues/3150) | [Task] Align XPU kernel's impl | guangyey | P2 | UT issue with few failures | Inductor / Compilation Related |  |  | aten_ops | ut |
| [3139](https://github.com/intel/torch-xpu-ops/issues/3139) | [distributed][_composable] Ass | Kanya-Mo | P2 | UT issue with few failures | Distributed | bug, module: distributed |  | distributed | ut |
| [3103](https://github.com/intel/torch-xpu-ops/issues/3103) | Tensor-likes are not equal for | BBBela | P2 | UT issue with few failures | Dtype / Precision Related | module: ut, skipped, random |  | aten_ops | ut |
| [3088](https://github.com/intel/torch-xpu-ops/issues/3088) | [TorchAO][BMG] INT4 RTN Flex-a | Stonepia | P2 | UT issue with few failures | TorchAO | module: ao |  | AO | ut |
| [3074](https://github.com/intel/torch-xpu-ops/issues/3074) | [Bug Skip] test_dlpack_exchang | AKloniecki | P2 | UT issue with few failures | Others | skipped |  | aten_ops | ut |
| [3060](https://github.com/intel/torch-xpu-ops/issues/3060) | Implement torch._scaled_groupe | Stonepia, liangan1 | P2 | UT issue with few failures | Others | module: quant |  | low_precision | ut |
| [3048](https://github.com/intel/torch-xpu-ops/issues/3048) | Profiler result is not correct | aostrowski-hbn | P2 | UT issue with few failures | Others | module: profiler |  | profiling | ut |
| [3033](https://github.com/intel/torch-xpu-ops/issues/3033) | [Bug Skip]: Softmax tolerance | chunhuanMeng | P2 | UT issue with few failures | Others | skipped, random |  | aten_ops | ut |
| [3032](https://github.com/intel/torch-xpu-ops/issues/3032) | [TorchAO][UT] failures in test | Stonepia | P0 | Build crash - critical blocking issue | TorchAO | module: ao |  | AO | build |
| [3030](https://github.com/intel/torch-xpu-ops/issues/3030) | [Bug Skip] test/test_modules.p | gplutop7 | P2 | UT issue with few failures | Others | skipped |  | aten_ops | ut |
| [3024](https://github.com/intel/torch-xpu-ops/issues/3024) | Enable clang-tidy checks | Silv3S | P2 | UT issue with few failures | Others | bug_fix_stage5 |  | aten_ops | ut |
| [3022](https://github.com/intel/torch-xpu-ops/issues/3022) | [distributed] batch_isend_irec | zhangxiaoli73 | P2 | UT issue with few failures | Distributed | module: distributed |  | distributed | ut |
| [3021](https://github.com/intel/torch-xpu-ops/issues/3021) | [distributed] all_to_all_singl | zhangxiaoli73 | P2 | UT issue with few failures | Distributed | module: distributed |  | distributed | ut |
| [3010](https://github.com/intel/torch-xpu-ops/issues/3010) | [distributed][tensor] test_ran | jenniew | P2 | UT issue with few failures | Distributed | bug, module: distributed |  | distributed | ut |
| [3000](https://github.com/intel/torch-xpu-ops/issues/3000) | [Bug Skip]: RuntimeError: _sha | gplutop7 | P2 | UT issue with few failures | Others | skipped |  | aten_ops | ut |
| [2993](https://github.com/intel/torch-xpu-ops/issues/2993) | [Bug Skip]: Unexpected success | gplutop7 | P2 | UT issue with few failures | Others | skipped |  | aten_ops | ut |
| [2984](https://github.com/intel/torch-xpu-ops/issues/2984) | [release/2.11] sebotnet33ts_25 | jianyizh, weishi-deng | P1 | E2E benchmark accuracy/functionality issue | Dtype / Precision Related | os: Ubuntu, hw: BMG |  | aten_ops | e2e |
| [2972](https://github.com/intel/torch-xpu-ops/issues/2972) | [distributed] AssertionError:  | newtdms | P2 | UT issue with few failures | Distributed | bug, module: distributed |  | distributed | ut |
| [2971](https://github.com/intel/torch-xpu-ops/issues/2971) | [distributed] KeyError in test | newtdms, frost-intel | P2 | UT issue with few failures | Distributed | bug, module: distributed |  | distributed | ut |
| [2969](https://github.com/intel/torch-xpu-ops/issues/2969) | [distributed] AssertionError:  | frost-intel | P2 | UT issue with few failures | Distributed | bug, module: distributed |  | distributed | ut |
| [2968](https://github.com/intel/torch-xpu-ops/issues/2968) | [distributed] timeout issue in | frost-intel | P2 | UT issue with few failures | Distributed | bug, module: distributed |  | distributed | ut |
| [2967](https://github.com/intel/torch-xpu-ops/issues/2967) | [distributed] feature gaps in  | frost-intel | P2 | UT issue with few failures | Distributed | bug, module: distributed |  | distributed | ut |
| [2952](https://github.com/intel/torch-xpu-ops/issues/2952) | [release/2.11][wsl] timm_model | weishi-deng | P0 | Impacts real model/application | Dtype / Precision Related | Accuracy, hw: BMG |  | aten_ops | ut |
| [2950](https://github.com/intel/torch-xpu-ops/issues/2950) | SYCL compilation flag -fsycl-i | BBBela | P2 | UT issue with few failures | Inductor / Compilation Related |  |  | aten_ops | ut |
| [2942](https://github.com/intel/torch-xpu-ops/issues/2942) | [Windows] Unit tests got Fatal | xuhancn, Stonepia | P2 | UT issue with few failures | Others | os: Windows |  | aten_ops | ut |
| [2940](https://github.com/intel/torch-xpu-ops/issues/2940) | [release/2.11] Models performa | jianyizh, LuFinch | P0 | Impacts real model/application | Others | performance |  | aten_ops | e2e |
| [2939](https://github.com/intel/torch-xpu-ops/issues/2939) | [release/2.11] gmlp_s16_224 in | jianyizh | P2 | E2E benchmark performance issue | Flash Attention / Transformer Related | performance |  | aten_ops | e2e |
| [2938](https://github.com/intel/torch-xpu-ops/issues/2938) | [release/2.11] basic_gnn_gin a | jianyizh | P2 | E2E benchmark performance issue | Dtype / Precision Related | performance |  | aten_ops | e2e |
| [2935](https://github.com/intel/torch-xpu-ops/issues/2935) | [release/2.11][inductor] huggi | jianyizh | P0 | Impacts real model/application | Inductor / Compilation Related | performance |  | aten_ops | e2e |
| [2932](https://github.com/intel/torch-xpu-ops/issues/2932) | [release/2.11] jx_nest_base an | jianyizh | P2 | E2E benchmark performance issue | Dtype / Precision Related |  |  | aten_ops | e2e |
| [2929](https://github.com/intel/torch-xpu-ops/issues/2929) | [release/2.11] volo_d1_224 inf | jianyizh | P1 | E2E benchmark accuracy/functionality issue | Dtype / Precision Related |  |  | aten_ops | e2e |
| [2928](https://github.com/intel/torch-xpu-ops/issues/2928) | [release/2.11] pyhpc_turbulent | jianyizh | P1 | E2E benchmark accuracy/functionality issue | Dtype / Precision Related |  |  | aten_ops | e2e |
| [2924](https://github.com/intel/torch-xpu-ops/issues/2924) | [release/2.11] xcit_large_24_p | jianyizh, mengfei25 | P1 | E2E benchmark accuracy/functionality issue | Dtype / Precision Related | Accuracy |  | aten_ops | e2e |
| [2922](https://github.com/intel/torch-xpu-ops/issues/2922) | [release/2.11] UT inductor Ass | tadkrawiec | P2 | UT issue with few failures | Inductor / Compilation Related | os: Windows |  | aten_ops | ut |
| [2921](https://github.com/intel/torch-xpu-ops/issues/2921) | [abs][complex64] - new failing | AKloniecki | P2 | UT issue with few failures | Distributed | skipped |  | aten_ops | ut |
| [2920](https://github.com/intel/torch-xpu-ops/issues/2920) | Failing Test Cases in Nightly  | Silv3S | P2 | UT issue with few failures | Others | skipped |  | aten_ops | ut |
| [2908](https://github.com/intel/torch-xpu-ops/issues/2908) | [release/2.11] Model fail_accu | xuhancn | P1 | E2E benchmark accuracy/functionality issue | Dtype / Precision Related | E2E |  | aten_ops | e2e |
| [2907](https://github.com/intel/torch-xpu-ops/issues/2907) | [release/2.11] Models performa | xuhancn | P0 | Regression - passed before but failed now | Others |  |  | aten_ops | ut |
| [2879](https://github.com/intel/torch-xpu-ops/issues/2879) | RuntimeError: _share_fd_: only | Silv3S | P2 | UT issue with few failures | Others | bug_fix_stage5 |  | aten_ops | ut |
| [2873](https://github.com/intel/torch-xpu-ops/issues/2873) | [Bug Skip]: test_repos.py cont | PawelSwider2000 | P2 | UT issue with few failures | PT2E | skipped |  | aten_ops | ut |
| [2871](https://github.com/intel/torch-xpu-ops/issues/2871) | [distributed][fsdp] test_fsdp_ | songhappy | P2 | UT issue with few failures | Distributed | bug, module: distributed |  | distributed | ut |
| [2862](https://github.com/intel/torch-xpu-ops/issues/2862) | accuracy issue with test_float | tszulist-hbn | P2 | UT issue with few failures | Dtype / Precision Related |  |  | aten_ops | ut |
| [2853](https://github.com/intel/torch-xpu-ops/issues/2853) | [upstream_ut] torch.ops.aten._ | LuFinch | P2 | UT issue with few failures | Distributed | skipped |  | aten_ops | ut |
| [2823](https://github.com/intel/torch-xpu-ops/issues/2823) | [TorchAO][BMG] Llama-3.2-1B-In | xiaowangintel, lchen2331 | P2 | UT issue with few failures | TorchAO | module: ao |  | AO | ut |
| [2816](https://github.com/intel/torch-xpu-ops/issues/2816) | torch.logcumsumexp incorrectly | Silv3S | P2 | UT issue with few failures | Dtype / Precision Related | Ready for merge, skipped, bug_fix_stage5 |  | aten_ops | ut |
| [2815](https://github.com/intel/torch-xpu-ops/issues/2815) | RuntimeError: output with shap | PawelSwider2000 | P2 | UT issue with few failures | Others | skipped, bug_fix_stage5 |  | aten_ops | ut |
| [2811](https://github.com/intel/torch-xpu-ops/issues/2811) | [Bug Skip]: [Regression] faile | jmamzax | P0 | Regression - passed before but failed now | Distributed | skipped, bug_fix_stage5 |  | aten_ops | ut |
| [2801](https://github.com/intel/torch-xpu-ops/issues/2801) | to_dense() for Sparse CSR back | jenniew | P2 | UT issue with few failures | Sparse Operations Related |  |  | aten_ops | ut |
| [2795](https://github.com/intel/torch-xpu-ops/issues/2795) | Histc raises error with intege | CuiYifeng | P2 | UT issue with few failures | Others |  |  | aten_ops | ut |
| [2783](https://github.com/intel/torch-xpu-ops/issues/2783) | [Bug Skip]: Key "xpu" is missi | daisyden | P2 | UT issue with few failures | Dtype / Precision Related | module: ut, skipped |  | aten_ops | ut |
| [2779](https://github.com/intel/torch-xpu-ops/issues/2779) | Accuracy failures in logspace  | PawelSwider2000 | P2 | UT issue with few failures | Distributed | module: ut, skipped, bug_fix_stage5 |  | aten_ops | ut |
| [2777](https://github.com/intel/torch-xpu-ops/issues/2777) | [Bug Skip]: Random failures 20 | AKloniecki | P2 | UT issue with few failures | Sparse Operations Related | skipped, random |  | unknown | ut |
| [2767](https://github.com/intel/torch-xpu-ops/issues/2767) | [UT] test_control_flow_xpu.py  | PatrykWilczewski | P1 | UT with 21 failed test cases | PT2E | module: ut, skipped, bug_fix_stage5 |  | aten_ops | ut |
| [2766](https://github.com/intel/torch-xpu-ops/issues/2766) | MaxPool2d - investigate memory | BBBela | P2 | UT issue with few failures | Others |  |  | aten_ops | ut |
| [2759](https://github.com/intel/torch-xpu-ops/issues/2759) | [Bug Skip]: New failed cases 2 | AKloniecki | P2 | UT issue with few failures | Distributed | skipped |  | aten_ops | ut |
| [2744](https://github.com/intel/torch-xpu-ops/issues/2744) | [Bug Skip]: extended test fail | pbielak | P2 | UT issue with few failures | Others | skipped |  | aten_ops | ut |
| [2742](https://github.com/intel/torch-xpu-ops/issues/2742) | [Linux][PT2E] hf_Roberta_base  | chunhuanMeng | P0 | Impacts real model/application | Flash Attention / Transformer Related |  |  | aten_ops | e2e |
| [2738](https://github.com/intel/torch-xpu-ops/issues/2738) | [distributed] test_c10d_spawn_ | jenniew | P2 | UT issue with few failures | Distributed | bug, module: distributed |  | distributed | ut |
| [2734](https://github.com/intel/torch-xpu-ops/issues/2734) | [TorchAO][BMG] INT4 RTN Flex-a | Stonepia, hoshibara | P2 | UT issue with few failures | TorchAO | module: ao | [PR](https://github.com/pytorch/pytorch/pull/160480, https://github.com/pytorch/pytorch/pull/172316) | AO | ut |
| [2729](https://github.com/intel/torch-xpu-ops/issues/2729) | [Bug Skip]: Random failures 20 | Silv3S | P2 | UT issue with few failures | Sparse Operations Related | skipped, bug_fix_stage5, random |  | unknown | ut |
| [2722](https://github.com/intel/torch-xpu-ops/issues/2722) | [Bug Skip]: NotImplementedErro | Silv3S | P2 | UT issue with few failures | TorchAO | skipped, bug_fix_stage5 |  | aten_ops | ut |
| [2720](https://github.com/intel/torch-xpu-ops/issues/2720) | [upstream_ut] RuntimeError: fa | CuiYifeng | P2 | UT issue with few failures | Distributed | skipped |  | aten_ops | ut |
| [2707](https://github.com/intel/torch-xpu-ops/issues/2707) | [TorchAO][BMG] INT4 GPTQ faile | xiaowangintel | P2 | UT issue with few failures | TorchAO | module: ao |  | AO | ut |
| [2702](https://github.com/intel/torch-xpu-ops/issues/2702) | [distributed] RuntimeError: Wo | syedshahbaaz | P2 | UT issue with few failures | Distributed | bug, module: distributed |  | distributed | ut |
| [2701](https://github.com/intel/torch-xpu-ops/issues/2701) | [distributed] Barrier Timeout  | syedshahbaaz | P2 | UT issue with few failures | Distributed | bug, module: distributed |  | distributed | ut |
| [2700](https://github.com/intel/torch-xpu-ops/issues/2700) | [distributed] Hang issues with | syedshahbaaz | P2 | UT issue with few failures | Distributed | bug, module: distributed |  | distributed | ut |
| [2689](https://github.com/intel/torch-xpu-ops/issues/2689) | [LNL][Windows] AssertionError: | tadkrawiec | P2 | UT issue with few failures | Dtype / Precision Related | os: Windows, module: ut |  | aten_ops | ut |
| [2686](https://github.com/intel/torch-xpu-ops/issues/2686) | [distributed] Accuracy issues  | frost-intel | P2 | UT issue with few failures | Distributed | bug, module: distributed |  | distributed | ut |
| [2680](https://github.com/intel/torch-xpu-ops/issues/2680) | XPU Autocast does not support  | CuiYifeng | P2 | UT issue with few failures | Dtype / Precision Related |  |  | aten_ops | ut |
| [2669](https://github.com/intel/torch-xpu-ops/issues/2669) | [upstream_ut]  AssertionError: | tszulist-hbn | P2 | UT issue with few failures | Distributed | skipped |  | aten_ops | ut |
| [2662](https://github.com/intel/torch-xpu-ops/issues/2662) | [release/2.10][Windows][BMG] N | tadkrawiec, kdrozd-dev | P2 | UT issue with few failures | Others | os: Windows, hw: BMG, module: ut |  | aten_ops | ut |
| [2660](https://github.com/intel/torch-xpu-ops/issues/2660) | [release/2.10][Windows][BMG] N | tadkrawiec | P2 | UT issue with few failures | Others | os: Windows, hw: BMG, module: ut |  | aten_ops | ut |
| [2659](https://github.com/intel/torch-xpu-ops/issues/2659) | [distributed] test_dist2.py Ru | Chao1Han | P2 | UT issue with few failures | Distributed | module: distributed |  | distributed | ut |
| [2654](https://github.com/intel/torch-xpu-ops/issues/2654) | [BMG][OOB] t5 inference perfor | jianyizh | P0 | Regression - passed before but failed now | Dtype / Precision Related | E2E, dtype: float16, triaged, performance, os: Ubuntu, hw: BMG, regression |  | aten_ops | e2e |
| [2650](https://github.com/intel/torch-xpu-ops/issues/2650) | [OOB Performance] The performa | jianyizh | P0 | Regression - passed before but failed now | Inductor / Compilation Related | E2E, dtype: float16, triaged, performance, os: Ubuntu, hw: BMG, regression |  | aten_ops | e2e |
| [2649](https://github.com/intel/torch-xpu-ops/issues/2649) | [distributed][pipelining] test | syedshahbaaz | P2 | UT issue with few failures | Distributed | bug, module: distributed |  | distributed | ut |
| [2645](https://github.com/intel/torch-xpu-ops/issues/2645) | [Bug Skip]: [regression] Runti | CuiYifeng | P0 | Regression - passed before but failed now | Inductor / Compilation Related | skipped |  | aten_ops | ut |
| [2640](https://github.com/intel/torch-xpu-ops/issues/2640) | random issue test_vjpvjp_index | wpietka | P2 | UT issue with few failures | Distributed | skipped, random |  | aten_ops | ut |
| [2639](https://github.com/intel/torch-xpu-ops/issues/2639) | test_to() failed during rnn is | Silv3S | P2 | UT issue with few failures | Distributed | skipped | [PR](https://github.com/intel/torch-xpu-ops/pull/3193) | aten_ops | ut |
| [2630](https://github.com/intel/torch-xpu-ops/issues/2630) | Title: [upstream_ut]  Assertio | jmamzax | P2 | UT issue with few failures | Distributed | skipped, port_from_skiplist |  | aten_ops | ut |
| [2615](https://github.com/intel/torch-xpu-ops/issues/2615) | [Bug Skip]: New failures Runti | CuiYifeng | P2 | UT issue with few failures | Distributed | module: ut, skipped |  | aten_ops | ut |
| [2598](https://github.com/intel/torch-xpu-ops/issues/2598) | [TorchAO][BMG]The first token  | Stonepia | P2 | UT issue with few failures | TorchAO | module: ao |  | AO | ut |
| [2597](https://github.com/intel/torch-xpu-ops/issues/2597) | [TorchAO][BMG] INT4 GPTQ shows | xiaowangintel | P2 | UT issue with few failures | TorchAO | module: ao |  | AO | ut |
| [2580](https://github.com/intel/torch-xpu-ops/issues/2580) | [TorchAO][UT] test/test_low_bi | arlesniak | P0 | Build crash - critical blocking issue | TorchAO | module: ao |  | AO | build |
| [2572](https://github.com/intel/torch-xpu-ops/issues/2572) | [TorchAO][UT] test/dtypes/test | xiaowangintel | P0 | Build crash - critical blocking issue | TorchAO | module: ao |  | AO | build |
| [2562](https://github.com/intel/torch-xpu-ops/issues/2562) | Warning as Error | chunhuanMeng | P2 | UT issue with few failures | Others |  |  | aten_ops | ut |
| [2560](https://github.com/intel/torch-xpu-ops/issues/2560) | [UT] "RuntimeError: iter.devic | CuiYifeng | P2 | UT issue with few failures | Others | bug |  | aten_ops | ut |
| [2541](https://github.com/intel/torch-xpu-ops/issues/2541) | Title: [upstream_ut]  RuntimeE | yucai-intel | P2 | UT issue with few failures | Distributed | skipped, port_from_skiplist |  | aten_ops | ut |
| [2538](https://github.com/intel/torch-xpu-ops/issues/2538) | Title: [upstream_ut]  RuntimeE | Silv3S | P2 | UT issue with few failures | Distributed | skipped, port_from_skiplist |  | aten_ops | ut |
| [2537](https://github.com/intel/torch-xpu-ops/issues/2537) | Title: [upstream_ut]  Failed:  | PatrykWilczewski | P2 | UT issue with few failures | Others | skipped, port_from_skiplist |  | aten_ops | ut |
| [2536](https://github.com/intel/torch-xpu-ops/issues/2536) | Title: [upstream_ut]  Attribut | daisyden | P2 | UT issue with few failures | Distributed | skipped, port_from_skiplist, not_target |  | aten_ops | ut |
| [2535](https://github.com/intel/torch-xpu-ops/issues/2535) | Title: [upstream_ut]  Attribut | Silv3S | P2 | UT issue with few failures | Distributed | skipped, port_from_skiplist |  | aten_ops | ut |
| [2533](https://github.com/intel/torch-xpu-ops/issues/2533) | Title: [upstream_ut]  Attribut | astachowiczhabana | P2 | UT issue with few failures | Distributed | skipped, port_from_skiplist |  | aten_ops | ut |
| [2532](https://github.com/intel/torch-xpu-ops/issues/2532) | Title: [upstream_ut]  Assertio | yucai-intel | P2 | UT issue with few failures | Distributed | skipped, port_from_skiplist |  | aten_ops | ut |
| [2519](https://github.com/intel/torch-xpu-ops/issues/2519) | [upstream_ut]  TypeError: map2 | Silv3S | P2 | UT issue with few failures | Others | skipped |  | aten_ops | ut |
| [2513](https://github.com/intel/torch-xpu-ops/issues/2513) | [upstream_ut]  RuntimeError: _ | gplutop7 | P2 | UT issue with few failures | Others | skipped |  | aten_ops | ut |
| [2512](https://github.com/intel/torch-xpu-ops/issues/2512) | [upstream_ut]  RuntimeError: _ | chunhuanMeng | P2 | UT issue with few failures | Dtype / Precision Related | skipped |  | aten_ops | ut |
| [2510](https://github.com/intel/torch-xpu-ops/issues/2510) | [upstream_ut]  RuntimeError: E | PawelSwider2000 | P2 | UT issue with few failures | Dtype / Precision Related | skipped |  | aten_ops | ut |
| [2494](https://github.com/intel/torch-xpu-ops/issues/2494) | [upstream_ut]  AssertionError: | PawelSwider2000 | P2 | UT issue with few failures | Dtype / Precision Related | skipped |  | aten_ops | ut |
| [2491](https://github.com/intel/torch-xpu-ops/issues/2491) | [upstream_ut]  AssertionError: | PatrykWilczewski | P2 | UT issue with few failures | Dtype / Precision Related | skipped, bug_fix_stage5 |  | aten_ops | ut |
| [2479](https://github.com/intel/torch-xpu-ops/issues/2479) | [Bug] torch.rand output differ | Stonepia, CuiYifeng | P2 | UT issue with few failures | Others |  |  | aten_ops | ut |
| [2471](https://github.com/intel/torch-xpu-ops/issues/2471) | test_cuda.py gaps | guangyey | P2 | UT issue with few failures | Others |  |  | aten_ops | ut |
| [2469](https://github.com/intel/torch-xpu-ops/issues/2469) | test_matmul_cuda.py gaps | CuiYifeng, guangyey | P2 | UT issue with few failures | Others |  |  | aten_ops | ut |
| [2465](https://github.com/intel/torch-xpu-ops/issues/2465) | [windows] ut hang | tadkrawiec | P2 | UT issue with few failures | Others | os: Windows |  | aten_ops | ut |
| [2463](https://github.com/intel/torch-xpu-ops/issues/2463) | [Bug Skip]: OSError: SYCL runt | xuhancn | P2 | UT issue with few failures | Others | skipped_windows |  | aten_ops | ut |
| [2444](https://github.com/intel/torch-xpu-ops/issues/2444) | [upstream_ut]  RuntimeError: U | Silv3S | P2 | UT issue with few failures | Dtype / Precision Related | skipped |  | aten_ops | ut |
| [2434](https://github.com/intel/torch-xpu-ops/issues/2434) | [Bug Skip]: New failures 2025- | AKloniecki | P2 | UT issue with few failures | TorchAO | module: ut, skipped, bug_fix_stage4 |  | aten_ops | ut |
| [2425](https://github.com/intel/torch-xpu-ops/issues/2425) | [upstream_ut]  RuntimeError: E | BBBela | P2 | UT issue with few failures | Dtype / Precision Related | skipped, bug_fix_stage4 |  | aten_ops | ut |
| [2412](https://github.com/intel/torch-xpu-ops/issues/2412) | Some NestedTensor missing XPU  | yucai-intel | P2 | UT issue with few failures | Others | module: ut |  | aten_ops | ut |
| [2400](https://github.com/intel/torch-xpu-ops/issues/2400) | [ut_upstream] tf32_on_and_off( | chunhuanMeng | P2 | UT issue with few failures | Others |  |  | aten_ops | ut |
| [2392](https://github.com/intel/torch-xpu-ops/issues/2392) | [Bug Skip]: torch.OutOfMemoryE | xuhancn | P2 | UT issue with few failures | Others | skipped_windows |  | aten_ops | ut |
| [2390](https://github.com/intel/torch-xpu-ops/issues/2390) | SDPA in pytorch use different  | LuFinch | P2 | UT issue with few failures | Flash Attention / Transformer Related |  |  | aten_ops | ut |
| [2389](https://github.com/intel/torch-xpu-ops/issues/2389) | [Bug Skip]: RuntimeError: Data | PatrykWilczewski | P2 | UT issue with few failures | Distributed | skipped, bug_fix_stage4, random |  | aten_ops | ut |
| [2376](https://github.com/intel/torch-xpu-ops/issues/2376) | [Bug Skip]: NotImplementedErro | CuiYifeng | P2 | UT issue with few failures | Dtype / Precision Related | module: ut, skipped |  | aten_ops | ut |
| [2349](https://github.com/intel/torch-xpu-ops/issues/2349) | [oneAPI][backward compatibilit | riverliuintel | P2 | UT issue with few failures | Others |  |  | aten_ops | ut |
| [2340](https://github.com/intel/torch-xpu-ops/issues/2340) | [distributed][_tools] Assertio | githubsgi | P2 | UT issue with few failures | Distributed | bug, duplicate, module: distributed |  | distributed | ut |
| [2326](https://github.com/intel/torch-xpu-ops/issues/2326) | [TorchAO] MX training  native  | riverliuintel | P2 | UT issue with few failures | TorchAO | module: ao |  | AO | ut |
| [2325](https://github.com/intel/torch-xpu-ops/issues/2325) | [TorchAO] Float8 training supp | arlesniak, riverliuintel | P2 | UT issue with few failures | TorchAO | module: ao |  | AO | ut |
| [2324](https://github.com/intel/torch-xpu-ops/issues/2324) | [TorchAO] FP8 conv support | Stonepia | P2 | UT issue with few failures | TorchAO | module: ao |  | AO | ut |
| [2323](https://github.com/intel/torch-xpu-ops/issues/2323) | [TorchAO] MOE training enablin | riverliuintel | P2 | UT issue with few failures | TorchAO | module: ao |  | AO | ut |
| [2263](https://github.com/intel/torch-xpu-ops/issues/2263) | [xpu][bug] XPU Trace event end | PawelSwider2000 | P2 | UT issue with few failures | Others | module: profiler |  | profiling | ut |
| [2250](https://github.com/intel/torch-xpu-ops/issues/2250) | Found mismatch when comparing  | astachowiczhabana | P2 | UT issue with few failures | Flash Attention / Transformer Related | skipped, bug_fix_stage3 |  | aten_ops | ut |
| [2245](https://github.com/intel/torch-xpu-ops/issues/2245) | oneDNN matmul received incorre | CuiYifeng | P2 | UT issue with few failures | Distributed | module: ut, skipped |  | aten_ops | ut |
| [2240](https://github.com/intel/torch-xpu-ops/issues/2240) | RuntimeError: Trying to set a  | gplutop7 | P2 | UT issue with few failures | Distributed | skipped, bug_fix_stage3 |  | aten_ops | ut |
| [2239](https://github.com/intel/torch-xpu-ops/issues/2239) | Exception: could not create a  | wpietka | P2 | UT issue with few failures | Distributed | skipped, bug_fix_stage5 |  | aten_ops | ut |
| [2229](https://github.com/intel/torch-xpu-ops/issues/2229) | test/test_sparse_csr.py::TestS | jenniew | P2 | UT issue with few failures | Distributed | skipped |  | aten_ops | ut |
| [2219](https://github.com/intel/torch-xpu-ops/issues/2219) | float8_e4m3fn precision overfl | CuiYifeng, yucai-intel | P2 | UT issue with few failures | Dtype / Precision Related |  |  | aten_ops | ut |
| [2217](https://github.com/intel/torch-xpu-ops/issues/2217) | AO Performance issue track | Stonepia | P2 | UT issue with few failures | Others | module: ao |  | AO | ut |
| [2215](https://github.com/intel/torch-xpu-ops/issues/2215) | Find use case example for torc | dvrogozh | P2 | UT issue with few failures | Dtype / Precision Related |  |  | aten_ops | ut |
| [2207](https://github.com/intel/torch-xpu-ops/issues/2207) | Enable FP8/MXFP8 Ops with requ | CuiYifeng | P2 | UT issue with few failures | TorchAO | dtype: float8 |  | aten_ops | ut |
| [2202](https://github.com/intel/torch-xpu-ops/issues/2202) | [TorchAO][BMG] The RTN perform | Stonepia | P0 | Regression - passed before but failed now | TorchAO | performance, regression, module: ao |  | AO | ut |
| [2201](https://github.com/intel/torch-xpu-ops/issues/2201) | [TorchAO][BMG] When using page | Stonepia | P2 | UT issue with few failures | TorchAO | module: ao |  | AO | ut |
| [2195](https://github.com/intel/torch-xpu-ops/issues/2195) | Tools in pti should get 100% f | aostrowski-hbn | P2 | UT issue with few failures | Others | module: profiler |  | profiling | ut |
| [2182](https://github.com/intel/torch-xpu-ops/issues/2182) | test_transform_bias_rescale_qk | PawelSwider2000 | P2 | UT issue with few failures | Distributed | Accuracy, module: ut, skipped |  | aten_ops | ut |
| [2169](https://github.com/intel/torch-xpu-ops/issues/2169) | Frame size comparison failed i | guangyey | P2 | UT issue with few failures | Distributed | skipped |  | aten_ops | ut |
| [2165](https://github.com/intel/torch-xpu-ops/issues/2165) | [distributed] test_device_mesh | jemitche1 | P2 | UT issue with few failures | Distributed | bug, module: distributed |  | distributed | ut |
| [2163](https://github.com/intel/torch-xpu-ops/issues/2163) | 3 distributed UT cases need to | githubsgi | P2 | UT issue with few failures | Distributed | module: distributed |  | distributed | ut |
| [2142](https://github.com/intel/torch-xpu-ops/issues/2142) | XPU max_memory_allocated have  | guangyey | P2 | UT issue with few failures | Others | bug |  | aten_ops | ut |
| [2140](https://github.com/intel/torch-xpu-ops/issues/2140) | Consider how to avoid copy in  | CuiYifeng | P2 | UT issue with few failures | Inductor / Compilation Related | enhancement |  | aten_ops | ut |
| [2128](https://github.com/intel/torch-xpu-ops/issues/2128) | [2.9][BMG-Windows][Torchbench] | chuanqi129 | P0 | Impacts real model/application | Dtype / Precision Related |  |  | aten_ops | ut |
| [2127](https://github.com/intel/torch-xpu-ops/issues/2127) | Path Coverage enhancement | CuiYifeng | P2 | UT issue with few failures | Others | enhancement |  | aten_ops | ut |
| [2113](https://github.com/intel/torch-xpu-ops/issues/2113) | Update example for Distributed | songhappy | P2 | UT issue with few failures | Distributed | module: distributed |  | distributed | ut |
| [2098](https://github.com/intel/torch-xpu-ops/issues/2098) | Upstream XPU functions in yaml | guangyey | P2 | UT issue with few failures | Others | enhancement |  | aten_ops | ut |
| [2086](https://github.com/intel/torch-xpu-ops/issues/2086) | nd_item::barrier has been depr | dvrogozh | P2 | UT issue with few failures | Others | enhancement |  | aten_ops | ut |
| [2063](https://github.com/intel/torch-xpu-ops/issues/2063) | Avoid using out-of-date term | CuiYifeng | P2 | UT issue with few failures | Others | enhancement |  | aten_ops | ut |
| [2058](https://github.com/intel/torch-xpu-ops/issues/2058) | [release/2.9] llama_v2_7b_16h  | jianyizh | P0 | Build crash - critical blocking issue | Flash Attention / Transformer Related | performance, regression, dependency component: community |  | aten_ops | e2e |
| [2055](https://github.com/intel/torch-xpu-ops/issues/2055) | New huggingface LLM models iss | jianyizh, mengfei25 | P0 | Impacts real model/application | Others | E2E, hw: PVC |  | aten_ops | e2e |
| [2004](https://github.com/intel/torch-xpu-ops/issues/2004) | [distributed][shared_tensor] t | libohao1201 | P2 | UT issue with few failures | Distributed | bug, module: distributed |  | distributed | ut |
| [1996](https://github.com/intel/torch-xpu-ops/issues/1996) | [TorchAO]  Memory Efficient Op | arlesniak | P2 | UT issue with few failures | TorchAO | module: ao |  | AO | ut |
| [1969](https://github.com/intel/torch-xpu-ops/issues/1969) | torch._dynamo.exc.InternalTorc | guangyey | P2 | UT issue with few failures | PT2E | module: ut |  | aten_ops | ut |
| [1936](https://github.com/intel/torch-xpu-ops/issues/1936) | implement torch.linalg.cholesk | mwiktor-intel | P2 | UT issue with few failures | Others | module: op impl, bug_fix_stage5 |  | aten_ops | ut |
| [1902](https://github.com/intel/torch-xpu-ops/issues/1902) | implement torch.linalg.pinv xp | mwiktor-intel | P2 | UT issue with few failures | Others | module: op impl, bug_fix_stage5 |  | aten_ops | ut |
| [1901](https://github.com/intel/torch-xpu-ops/issues/1901) | implement torch.linalg.svd xpu | CuiYifeng | P2 | UT issue with few failures | Others | module: op impl |  | aten_ops | ut |
| [1900](https://github.com/intel/torch-xpu-ops/issues/1900) | implement torch.linalg.qr xpu  | pbielak | P2 | UT issue with few failures | Others | module: op impl, bug_fix_stage3 |  | aten_ops | ut |
| [1894](https://github.com/intel/torch-xpu-ops/issues/1894) | [Linux][PT2E] performance test | jenniew | P1 | E2E benchmark accuracy/functionality issue | TorchAO | module: quant |  | low_precision | e2e |
| [1877](https://github.com/intel/torch-xpu-ops/issues/1877) | Torchbench model squeezenet1_1 | Silv3S | P0 | Impacts real model/application | Dtype / Precision Related | Accuracy, hw: BMG, hw: PVC, bug_fix_stage5 |  | aten_ops | ut |
| [1866](https://github.com/intel/torch-xpu-ops/issues/1866) | [release 2.8]Torchbench vision | BartoszKokoszko | P0 | Impacts real model/application | Dtype / Precision Related | Accuracy, os: Windows, hw: BMG, bug_fix_stage5 |  | aten_ops | e2e |
| [1856](https://github.com/intel/torch-xpu-ops/issues/1856) | channel last aten::hardswish_  | chunhuanMeng | P2 | UT issue with few failures | Others | performance, hw: BMG |  | aten_ops | ut |
| [1833](https://github.com/intel/torch-xpu-ops/issues/1833) | [PT2E] INT8 accuracy is lower  | chunhuanMeng | P2 | UT issue with few failures | TorchAO | Accuracy, module: quant, dtype: int8 |  | low_precision | ut |
| [1818](https://github.com/intel/torch-xpu-ops/issues/1818) | [BMG-Windows][PT2.8]Torch-xpu- | kdrozd-dev | P2 | UT issue with few failures | Dtype / Precision Related | os: Windows, hw: BMG, bug_fix_stage5 |  | aten_ops | ut |
| [1784](https://github.com/intel/torch-xpu-ops/issues/1784) | [Performance] Torch XPU Profil | jfedorov | P2 | UT issue with few failures | Others | module: profiler |  | profiling | ut |
| [1778](https://github.com/intel/torch-xpu-ops/issues/1778) | [Infra] Show known issues for  | mengfei25 | P1 | E2E benchmark accuracy/functionality issue | Dtype / Precision Related | E2E, Accuracy, skipped, module: infra |  | unknown | e2e |
| [1764](https://github.com/intel/torch-xpu-ops/issues/1764) | New kernels for concat | yucai-intel | P2 | UT issue with few failures | Inductor / Compilation Related | performance, kernel_optimization, hw: BMG, module: op impl, benchmark |  | aten_ops | ut |
| [1762](https://github.com/intel/torch-xpu-ops/issues/1762) | Add an ocloc AOT target compil | chunhuanMeng | P2 | UT issue with few failures | PT2E | module: build |  | aten_ops | ut |
| [1749](https://github.com/intel/torch-xpu-ops/issues/1749) | transformers UT failure in XPU | LuFinch | P2 | UT issue with few failures | Flash Attention / Transformer Related |  |  | aten_ops | ut |
| [1745](https://github.com/intel/torch-xpu-ops/issues/1745) | torch.xpu.empty_cache() introd | guangyey | P2 | UT issue with few failures | Others | module: core |  | aten_ops | ut |
| [1729](https://github.com/intel/torch-xpu-ops/issues/1729) | Validation Check List | chuanqi129 | P2 | UT issue with few failures | Others | module: infra |  | aten_ops | ut |
| [1661](https://github.com/intel/torch-xpu-ops/issues/1661) | [distributed] Accuracy gap in  | githubsgi | P2 | UT issue with few failures | Distributed | bug, module: distributed |  | distributed | ut |
| [1645](https://github.com/intel/torch-xpu-ops/issues/1645) | [For Comparison] Save referenc | mengfei25 | P2 | UT issue with few failures | Others | module: infra |  | inductor | ut |
| [1594](https://github.com/intel/torch-xpu-ops/issues/1594) | Keep track on the building war | CuiYifeng, chunhuanMeng | P0 | Build crash - critical blocking issue | Others | module: build |  | aten_ops | ut |
| [1587](https://github.com/intel/torch-xpu-ops/issues/1587) | Keep track on the latest CUDA  | CuiYifeng, yucai-intel | P2 | UT issue with few failures | Others | kernel_optimization |  | aten_ops | ut |
| [1574](https://github.com/intel/torch-xpu-ops/issues/1574) | The operator 'aten::_grouped_m | Stonepia | P2 | UT issue with few failures | Others | module: ao |  | AO | ut |
| [1571](https://github.com/intel/torch-xpu-ops/issues/1571) | [distributed] ValueError: Cann | zhangxiaoli73 | P2 | UT issue with few failures | Distributed | module: distributed |  | distributed | ut |
| [1165](https://github.com/intel/torch-xpu-ops/issues/1165) | [CI] Add a test of PyTorch XPU | RUIJIEZHONG66166 | P0 | Build crash - critical blocking issue | Flash Attention / Transformer Related | module: transformers |  | aten_ops | build |
| [1159](https://github.com/intel/torch-xpu-ops/issues/1159) | [LNL Windows][Test by CD Night | Stonepia | P0 | Impacts real model/application | Flash Attention / Transformer Related | E2E, client, module: dependency bug, dependency: third_party packages |  | aten_ops | e2e |
| [492](https://github.com/intel/torch-xpu-ops/issues/492) | Timm_efficientdet NotImplement | weishi-deng | P0 | Impacts real model/application | Others | E2E, Accuracy, module: torchbench, dtype: amp_bf16, dtype: amp_fp16, training, dtype: float16, dtype: float32, dtype: bfloat16, triaged |  | aten_ops | e2e |
| [489](https://github.com/intel/torch-xpu-ops/issues/489) | Moco NotImplementedError: xpu  | weishi-deng | P2 | UT issue with few failures | Others | E2E, Accuracy, module: torchbench, dtype: amp_bf16, dtype: amp_fp16, training, dtype: float16, dtype: float32, dtype: bfloat16 |  | aten_ops | e2e |
| [208](https://github.com/intel/torch-xpu-ops/issues/208) | Abstract utility functions use | CuiYifeng | P2 | UT issue with few failures | Others | enhancement, module: op impl, long term |  | aten_ops | ut |
| [146](https://github.com/intel/torch-xpu-ops/issues/146) | Evaluate register spill in SYC | CuiYifeng, jianyizh, mengfei25 | P2 | UT issue with few failures | Inductor / Compilation Related | enhancement |  | aten_ops | ut |

---

## 5. Recent Issues (Last 10 Days)

Issues created in the last 10 days (as of 2026-04-08).

| ID | Title | Status | Owner | Priority | Reason | Category | Labels | Module | Test Module |
|---|-------|--------|-------|---------|--------|----------|--------|--------|-------------|
| [3259](https://github.com/intel/torch-xpu-ops/issues/3259) | New failed test cases 2026-04-02 | open | SlawomirLaba | P2 | UT issue with few failures | Flash Attention / Transformer Related | skipped | aten_ops | ut |
| [3258](https://github.com/intel/torch-xpu-ops/issues/3258) | Error in op: torch.ops.aten._scaled_dot_ | open | None | P2 | UT issue with few failures | Flash Attention / Transformer Related |  | aten_ops | ut |
| [3257](https://github.com/intel/torch-xpu-ops/issues/3257) | [Linux][E2E][Regression] Huggingface tes | open | None | P0 | Impacts real model/application | Others |  | aten_ops | e2e |
| [3255](https://github.com/intel/torch-xpu-ops/issues/3255) | [Linux][PT2E][Regression] Some performan | open | None | P0 | Regression - passed before but failed now | TorchAO |  | aten_ops | e2e |
| [3247](https://github.com/intel/torch-xpu-ops/issues/3247) | NotImplementedError: "dot_xpu_mkl" not i | open | Silv3S | P2 | UT issue with few failures | Others | ut_upstream | aten_ops | ut |
| [3246](https://github.com/intel/torch-xpu-ops/issues/3246) | AssertionError: Booleans mismatch: True  | open | BartoszKokoszko | P2 | UT issue with few failures | Distributed | skipped | aten_ops | ut |
| [3243](https://github.com/intel/torch-xpu-ops/issues/3243) | AssertionError: False is not true | open | pponikox | P2 | UT issue with few failures | Dtype / Precision Related | module: ut, skipped | aten_ops | ut |
| [3242](https://github.com/intel/torch-xpu-ops/issues/3242) | AssertionError: Torch not compiled with  | open | None | P2 | UT issue with few failures | Inductor / Compilation Related | module: ut, skipped | aten_ops | ut |
| [3238](https://github.com/intel/torch-xpu-ops/issues/3238) | The supported dtypes of _refs.stft is no | open | CuiYifeng | P2 | UT issue with few failures | Dtype / Precision Related | ut_upstream | aten_ops | ut |
| [3236](https://github.com/intel/torch-xpu-ops/issues/3236) | AssertionError: 'def [28 chars]n unbind  | open | jmamzax | P2 | UT issue with few failures | Dtype / Precision Related | bug_fix_stage5 | aten_ops | ut |
| [3233](https://github.com/intel/torch-xpu-ops/issues/3233) | [distributed] RuntimeError: No backend f | open | None | P2 | UT issue with few failures | Distributed | bug, module: distributed | distributed | ut |
| [3232](https://github.com/intel/torch-xpu-ops/issues/3232) | [distributed][tensor] AssertionError: As | open | None | P2 | UT issue with few failures | Distributed | bug, module: distributed | distributed | ut |
| [3231](https://github.com/intel/torch-xpu-ops/issues/3231) | Dynamo failed to run FX node with fake t | open | None | P2 | UT issue with few failures | PT2E | module: ut, skipped | aten_ops | ut |
| [3229](https://github.com/intel/torch-xpu-ops/issues/3229) | RuntimeError: No viable backend for scal | open | None | P2 | UT issue with few failures | Flash Attention / Transformer Related | skipped, ut_upstream | aten_ops | ut |
| [3227](https://github.com/intel/torch-xpu-ops/issues/3227) | torch xpu event has ~0.1ms latency, whic | open | guangyey | P2 | UT issue with few failures | Others |  | aten_ops | ut |
| [3224](https://github.com/intel/torch-xpu-ops/issues/3224) | [Win][Build] Building SYCL (Device) obje | open | chunhuanMeng | P0 | Build crash - critical blocking issue | Inductor / Compilation Related |  | aten_ops | build |
| [3216](https://github.com/intel/torch-xpu-ops/issues/3216) | [OPs] Some ops of XPU have non-determini | open | CuiYifeng | P2 | UT issue with few failures | Others |  | aten_ops | ut |
| [3209](https://github.com/intel/torch-xpu-ops/issues/3209) | [Win][Build] There is Cyclic dependencie | open | Copilot | P0 | Build crash - critical blocking issue | Others |  | aten_ops | build |
| [3206](https://github.com/intel/torch-xpu-ops/issues/3206) | [Bug Skip]: [new failures]RuntimeError:  | open | tszulist-hbn | P2 | UT issue with few failures | Others | skipped | aten_ops | ut |
