# Torch XPU Ops Issue Report

**Repository:** [intel/torch-xpu-ops](https://github.com/intel/torch-xpu-ops)

**Issues Extracted:** April 5, 2026 (from GitHub issue data: `torch_xpu_ops_issues.json`)

**CI Data Sources:**
- Torch-XPU-OPS Nightly CI: XML files from `Inductor-XPU-UT-Data-*` + `Inductor-wheel-nightly-LTS2-XPU-E2E-Data-*` (commit: `a2d516a58c64f18b76880f3a77efbc02885d65af`)
- Stock PyTorch XPU CI: XML files from `test-default-*-linux.idc.xpu_*.zip` (run IDs: 69741866812, 69741866834, 69741866862, 69741866911, etc.)

Generated: 2026-04-07 20:56:14

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

### By Priority

| Priority | Count |
|----------|-------|
| P0 | 50 |
| P1 | 18 |
| P2 | 349 |

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

| ID | Title | Owner | Owner Transferred | TBD | Priority | Reason | PR | Module | Test Module |
|---|-------|-------|-------------------|-----|---------|--------|-----|--------|-------------|
| [1624](https://github.com/intel/torch-xpu-ops/issues/1624) | [DONT CLOSE] Known UT Issue list | None | RUIJIEZHONG66166 | Close fixed issue | P2 | UT issue with few failures |  | distributed | ut |
| [2496](https://github.com/intel/torch-xpu-ops/issues/2496) | [upstream_ut]  Segmentation fault when r | astachowiczhabana | libohao1201 | Close fixed issue | P0 | Build crash - critical blocking issue |  | aten_ops | ut |
| [2518](https://github.com/intel/torch-xpu-ops/issues/2518) | [upstream_ut]  TypeError: Creating a Ten | astachowiczhabana | libohao1201 | Close fixed issue | P2 | UT issue with few failures |  | aten_ops | ut |
| [2592](https://github.com/intel/torch-xpu-ops/issues/2592) | [release/2.10] models got fail_accuracy | mengfei25 | mengfei25 | Close fixed issue | P0 | Impacts real model/application |  | aten_ops | e2e |
| [2619](https://github.com/intel/torch-xpu-ops/issues/2619) | [release/2.10] Some models inductor perf | jianyizh, weishi-deng | mengfei25 | Close fixed issue | P0 | Impacts real model/application |  | aten_ops | e2e |
| [2953](https://github.com/intel/torch-xpu-ops/issues/2953) | [release/2.11][wsl] huggingface TrOCRFor | xuhancn | bjarzemb | Close fixed issue | P0 | Impacts real model/application |  | aten_ops | e2e |
| [2981](https://github.com/intel/torch-xpu-ops/issues/2981) | [release/2.11] T5 models performance dro | jianyizh, weishi-deng | mengfei25 | Close fixed issue | P0 | Impacts real model/application |  | aten_ops | e2e |
| [3011](https://github.com/intel/torch-xpu-ops/issues/3011) | [upstream_ut] torch.OutOfMemoryError: XP | None | Silv3S | Close fixed issue | P2 | UT issue with few failures |  | aten_ops | ut |
| [3013](https://github.com/intel/torch-xpu-ops/issues/3013) | [upstream_ut] RuntimeError: Kernel is in | None | Silv3S | Close fixed issue | P2 | UT issue with few failures |  | unknown | ut |
| [3014](https://github.com/intel/torch-xpu-ops/issues/3014) | [upstream_ut] AssertionError: False is n | None | Silv3S | Close fixed issue | P2 | UT issue with few failures |  | unknown | ut |
| [3058](https://github.com/intel/torch-xpu-ops/issues/3058) | [E2E] hf_GPT2_large amp_fp16/amp_bf16  t | weishi-deng | kaileiyx | Close fixed issue | P1 | E2E benchmark accuracy/functionality issue |  | aten_ops | e2e |
| [3106](https://github.com/intel/torch-xpu-ops/issues/3106) | Worker crashes when running TestDecompXP | BBBela | BBBela | Close fixed issue | P0 | Build crash - critical blocking issue |  | aten_ops | ut |
| [3157](https://github.com/intel/torch-xpu-ops/issues/3157) | XPU out of memory. Tried to allocate 32. | kdrozd-dev | kdrozd-dev | Close fixed issue | P2 | UT issue with few failures |  | aten_ops | ut |
| [3158](https://github.com/intel/torch-xpu-ops/issues/3158) | AttributeError: module 'triton.compiler' | tadkrawiec | kdrozd-dev | Close fixed issue | P2 | UT issue with few failures |  | aten_ops | ut |
| [3161](https://github.com/intel/torch-xpu-ops/issues/3161) | Exception: Tensor-likes are not close! - | tadkrawiec | kdrozd-dev | Close fixed issue | P2 | UT issue with few failures |  | aten_ops | ut |
| [3166](https://github.com/intel/torch-xpu-ops/issues/3166) | test_consistency_SparseCSR failures | yucai-intel | CuiYifeng | Close fixed issue | P2 | UT issue with few failures |  | aten_ops | ut |

### 1.2 Dtype / Precision Related

| ID | Title | Owner | Owner Transferred | TBD | Priority | Reason | PR | Module | Test Module |
|---|-------|-------|-------------------|-----|---------|--------|-----|--------|-------------|
| [1661](https://github.com/intel/torch-xpu-ops/issues/1661) | [distributed] Accuracy gap in _composabl | githubsgi | githubsgi | Dtype / Precision Related | P2 | UT issue with few failures |  | distributed | ut |
| [1778](https://github.com/intel/torch-xpu-ops/issues/1778) | [Infra] Show known issues for accuracy t | mengfei25 | mengfei25 | Dtype / Precision Related | P1 | E2E benchmark accuracy/functionality issue |  | unknown | e2e |
| [1818](https://github.com/intel/torch-xpu-ops/issues/1818) | [BMG-Windows][PT2.8]Torch-xpu-ops UT got | kdrozd-dev | kdrozd-dev | Dtype / Precision Related | P2 | UT issue with few failures |  | aten_ops | ut |
| [1833](https://github.com/intel/torch-xpu-ops/issues/1833) | [PT2E] INT8 accuracy is lower than FP32 | chunhuanMeng | chunhuanMeng | Dtype / Precision Related | P2 | UT issue with few failures |  | low_precision | ut |
| [1866](https://github.com/intel/torch-xpu-ops/issues/1866) | [release 2.8]Torchbench vision_maskrcnn  | BartoszKokoszko | BartoszKokoszko | Dtype / Precision Related | P0 | Impacts real model/application |  | aten_ops | e2e |
| [1877](https://github.com/intel/torch-xpu-ops/issues/1877) | Torchbench model squeezenet1_1 and funct | Silv3S | Silv3S | Dtype / Precision Related | P0 | Impacts real model/application |  | aten_ops | ut |
| [2128](https://github.com/intel/torch-xpu-ops/issues/2128) | [2.9][BMG-Windows][Torchbench] speeach_t | chuanqi129 | chuanqi129 | Dtype / Precision Related | P0 | Impacts real model/application |  | aten_ops | ut |
| [2219](https://github.com/intel/torch-xpu-ops/issues/2219) | float8_e4m3fn precision overflow | CuiYifeng, yucai-intel | CuiYifeng, yucai-intel | Dtype / Precision Related | P2 | UT issue with few failures |  | aten_ops | ut |
| [2376](https://github.com/intel/torch-xpu-ops/issues/2376) | [Bug Skip]: NotImplementedError: "logadd | CuiYifeng | CuiYifeng | Dtype / Precision Related | P2 | UT issue with few failures |  | aten_ops | ut |
| [2439](https://github.com/intel/torch-xpu-ops/issues/2439) | [oneDNN] TestDecompXPU.test_quick_addmv_ | libohao1201 | libohao1201 | Dtype / Precision Related | P2 | UT issue with few failures |  | aten_ops | ut |
| [2444](https://github.com/intel/torch-xpu-ops/issues/2444) | [upstream_ut]  RuntimeError: UR backend  | Silv3S | Silv3S | Dtype / Precision Related | P2 | UT issue with few failures |  | aten_ops | ut |
| [2572](https://github.com/intel/torch-xpu-ops/issues/2572) | [TorchAO][UT] test/dtypes/test_affine_qu | xiaowangintel | xiaowangintel | Dtype / Precision Related | P0 | Build crash - critical blocking issue |  | AO | build |
| [2596](https://github.com/intel/torch-xpu-ops/issues/2596) | [TorchAO][BMG]Observed accuracy fluctuat | None | None | Dtype / Precision Related | P0 | Impacts real model/application |  | AO | ut |
| [2615](https://github.com/intel/torch-xpu-ops/issues/2615) | [Bug Skip]: New failures RuntimeError: U | CuiYifeng | CuiYifeng | Dtype / Precision Related | P2 | UT issue with few failures |  | aten_ops | ut |
| [2656](https://github.com/intel/torch-xpu-ops/issues/2656) | [release/2.10] models got fail_accuracy  | None | None | Dtype / Precision Related | P0 | Impacts real model/application |  | aten_ops | ut |
| [2680](https://github.com/intel/torch-xpu-ops/issues/2680) | XPU Autocast does not support  fp32 dtyp | CuiYifeng | CuiYifeng | Dtype / Precision Related | P2 | UT issue with few failures |  | aten_ops | ut |
| [2686](https://github.com/intel/torch-xpu-ops/issues/2686) | [distributed] Accuracy issues with test_ | frost-intel | frost-intel | Dtype / Precision Related | P2 | UT issue with few failures |  | distributed | ut |
| [2779](https://github.com/intel/torch-xpu-ops/issues/2779) | Accuracy failures in logspace op | PawelSwider2000 | PawelSwider2000 | Dtype / Precision Related | P2 | UT issue with few failures |  | aten_ops | ut |
| [2862](https://github.com/intel/torch-xpu-ops/issues/2862) | accuracy issue with test_float8_scale_fa | tszulist-hbn | tszulist-hbn | Dtype / Precision Related | P2 | UT issue with few failures |  | aten_ops | ut |
| [2908](https://github.com/intel/torch-xpu-ops/issues/2908) | [release/2.11] Model fail_accuracy for 5 | xuhancn | xuhancn | Dtype / Precision Related | P1 | E2E benchmark accuracy/functionality issue |  | aten_ops | e2e |
| [2924](https://github.com/intel/torch-xpu-ops/issues/2924) | [release/2.11] xcit_large_24_p8_224 amp_ | jianyizh, mengfei25 | jianyizh, mengfei25 | Dtype / Precision Related | P1 | E2E benchmark accuracy/functionality issue |  | aten_ops | e2e |
| [2928](https://github.com/intel/torch-xpu-ops/issues/2928) | [release/2.11] pyhpc_turbulent_kinetic_e | jianyizh | jianyizh | Dtype / Precision Related | P1 | E2E benchmark accuracy/functionality issue |  | aten_ops | e2e |
| [2952](https://github.com/intel/torch-xpu-ops/issues/2952) | [release/2.11][wsl] timm_models_accuracy | weishi-deng | weishi-deng | Dtype / Precision Related | P0 | Impacts real model/application |  | aten_ops | ut |
| [2960](https://github.com/intel/torch-xpu-ops/issues/2960) | [release/2.11] timm_models_xcit_large_24 | None | None | Dtype / Precision Related | P0 | Impacts real model/application |  | aten_ops | ut |
| [2984](https://github.com/intel/torch-xpu-ops/issues/2984) | [release/2.11] sebotnet33ts_256 fp32 tra | jianyizh, weishi-deng | jianyizh, weishi-deng | Dtype / Precision Related | P1 | E2E benchmark accuracy/functionality issue |  | aten_ops | e2e |
| [3148](https://github.com/intel/torch-xpu-ops/issues/3148) | [Triton] Huggingface openai/whisper-tiny | None | None | Dtype / Precision Related | P0 | Impacts real model/application |  | aten_ops | e2e |
| [3151](https://github.com/intel/torch-xpu-ops/issues/3151) | [Triton] Timm_models  rexnet_100 / fbnet | None | None | Dtype / Precision Related | P0 | Impacts real model/application |  | aten_ops | e2e |
| [3174](https://github.com/intel/torch-xpu-ops/issues/3174) | [Bug Skip]: Accuracy failure of test_Con | None | None | Dtype / Precision Related | P2 | UT issue with few failures |  | aten_ops | ut |

### 1.3 Enable test

| ID | Title | Owner | Owner Transferred | TBD | Priority | Reason | PR | Module | Test Module |
|---|-------|-------|-------------------|-----|---------|--------|-----|--------|-------------|
| [2024](https://github.com/intel/torch-xpu-ops/issues/2024) | AssertionError: Torch not compiled with  | daisyden | daisyden | Enable test | P2 | UT issue with few failures |  | aten_ops | ut |
| [2132](https://github.com/intel/torch-xpu-ops/issues/2132) | [2.9][BMG-Windows][Torch-xpu-ops UT] 1 c | pbielak | daisyden | Enable test | P2 | UT issue with few failures |  | aten_ops | ut |
| [2531](https://github.com/intel/torch-xpu-ops/issues/2531) | [upstream_ut]  AssertionError: Torch not | daisyden | daisyden | Enable test | P2 | UT issue with few failures |  | unknown | ut |
| [3242](https://github.com/intel/torch-xpu-ops/issues/3242) | AssertionError: Torch not compiled with  | None | daisyden | Enable test | P2 | UT issue with few failures |  | aten_ops | ut |

### 1.4 Flash Attention / Transformer Related

| ID | Title | Owner | Owner Transferred | TBD | Priority | Reason | PR | Module | Test Module |
|---|-------|-------|-------------------|-----|---------|--------|-----|--------|-------------|
| [1165](https://github.com/intel/torch-xpu-ops/issues/1165) | [CI] Add a test of PyTorch XPU with Hugg | RUIJIEZHONG66166 | RUIJIEZHONG66166 | Flash Attention / Transformer Related | P0 | Build crash - critical blocking issue |  | aten_ops | build |
| [1556](https://github.com/intel/torch-xpu-ops/issues/1556) | [distributed] NotImplementedError: Opera | pkourdis | pkourdis | Flash Attention / Transformer Related | P2 | UT issue with few failures |  | distributed | ut |
| [1749](https://github.com/intel/torch-xpu-ops/issues/1749) | transformers UT failure in XPU because S | LuFinch | LuFinch | Flash Attention / Transformer Related | P2 | UT issue with few failures |  | aten_ops | ut |
| [2200](https://github.com/intel/torch-xpu-ops/issues/2200) | support flash attention op on XPU device | ElaineBao | ElaineBao | Flash Attention / Transformer Related | P2 | UT issue with few failures |  | aten_ops | ut |
| [2232](https://github.com/intel/torch-xpu-ops/issues/2232) | sdpa backward kernel is required to redu | None | None | Flash Attention / Transformer Related | P2 | UT issue with few failures |  | aten_ops | ut |
| [2390](https://github.com/intel/torch-xpu-ops/issues/2390) | SDPA in pytorch use different backend co | LuFinch | LuFinch | Flash Attention / Transformer Related | P2 | UT issue with few failures |  | aten_ops | ut |
| [2570](https://github.com/intel/torch-xpu-ops/issues/2570) | crash in sdpa. | LuFinch | LuFinch | Flash Attention / Transformer Related | P0 | Build crash - critical blocking issue |  | aten_ops | ut |
| [2853](https://github.com/intel/torch-xpu-ops/issues/2853) | [upstream_ut] torch.ops.aten._flash_atte | LuFinch | LuFinch | Flash Attention / Transformer Related | P2 | UT issue with few failures |  | aten_ops | ut |
| [3093](https://github.com/intel/torch-xpu-ops/issues/3093) | XPU does not support NestedTensor for SD | None | None | Flash Attention / Transformer Related | P2 | UT issue with few failures |  | aten_ops | ut |
| [3178](https://github.com/intel/torch-xpu-ops/issues/3178) | New failed test cases 2026-03-25 | pponikox | pponikox | Flash Attention / Transformer Related | P2 | UT issue with few failures |  | aten_ops | ut |
| [3195](https://github.com/intel/torch-xpu-ops/issues/3195) | test_sdpa_unbacked_no_dde_xpu crashed | None | None | Flash Attention / Transformer Related | P0 | Build crash - critical blocking issue |  | aten_ops | ut |
| [3231](https://github.com/intel/torch-xpu-ops/issues/3231) | Dynamo failed to run FX node with fake t | None | None | Flash Attention / Transformer Related | P2 | UT issue with few failures |  | aten_ops | ut |
| [3258](https://github.com/intel/torch-xpu-ops/issues/3258) | Error in op: torch.ops.aten._scaled_dot_ | None | None | Flash Attention / Transformer Related | P2 | UT issue with few failures |  | aten_ops | ut |
| [3259](https://github.com/intel/torch-xpu-ops/issues/3259) | New failed test cases 2026-04-02 | SlawomirLaba | SlawomirLaba | Flash Attention / Transformer Related | P2 | UT issue with few failures |  | aten_ops | ut |

### 1.5 Inductor / Compilation Related

| ID | Title | Owner | Owner Transferred | TBD | Priority | Reason | PR | Module | Test Module |
|---|-------|-------|-------------------|-----|---------|--------|-----|--------|-------------|
| [1548](https://github.com/intel/torch-xpu-ops/issues/1548) | [distributed] AssertionError: 'fused_all | Chao1Han | Chao1Han | Inductor / Compilation Related | P2 | UT issue with few failures |  | distributed | ut |
| [1762](https://github.com/intel/torch-xpu-ops/issues/1762) | Add an ocloc AOT target compilation test | chunhuanMeng | chunhuanMeng | Inductor / Compilation Related | P2 | UT issue with few failures |  | aten_ops | ut |
| [1970](https://github.com/intel/torch-xpu-ops/issues/1970) | torch._dynamo.exc.BackendCompilerFailed: | None | None | Inductor / Compilation Related | P2 | UT issue with few failures |  | aten_ops | ut |
| [2169](https://github.com/intel/torch-xpu-ops/issues/2169) | Frame size comparison failed in test_siz | guangyey | guangyey | Inductor / Compilation Related | P2 | UT issue with few failures |  | aten_ops | ut |
| [2532](https://github.com/intel/torch-xpu-ops/issues/2532) | Title: [upstream_ut]  AssertionError: wr | yucai-intel | yucai-intel | Inductor / Compilation Related | P2 | UT issue with few failures |  | aten_ops | ut |
| [2605](https://github.com/intel/torch-xpu-ops/issues/2605) | [int4][inductor] Add freezing pattern fo | None | None | Inductor / Compilation Related | P2 | UT issue with few failures |  | aten_ops | ut |
| [2650](https://github.com/intel/torch-xpu-ops/issues/2650) | [OOB Performance] The performance impact | jianyizh | jianyizh | Inductor / Compilation Related | P0 | Regression - passed before but failed now |  | aten_ops | e2e |
| [2767](https://github.com/intel/torch-xpu-ops/issues/2767) | [UT] test_control_flow_xpu.py got Assert | PatrykWilczewski | PatrykWilczewski | Inductor / Compilation Related | P1 | UT with 21 failed test cases |  | aten_ops | ut |
| [2873](https://github.com/intel/torch-xpu-ops/issues/2873) | [Bug Skip]: test_repos.py contains sever | PawelSwider2000 | PawelSwider2000 | Inductor / Compilation Related | P2 | UT issue with few failures |  | aten_ops | ut |
| [2922](https://github.com/intel/torch-xpu-ops/issues/2922) | [release/2.11] UT inductor AssertionErro | tadkrawiec | tadkrawiec | Inductor / Compilation Related | P2 | UT issue with few failures |  | aten_ops | ut |
| [2935](https://github.com/intel/torch-xpu-ops/issues/2935) | [release/2.11][inductor] huggingface amp | jianyizh | jianyizh | Inductor / Compilation Related | P0 | Impacts real model/application |  | aten_ops | e2e |
| [3191](https://github.com/intel/torch-xpu-ops/issues/3191) | torch._inductor.exc.InductorError: Asser | EikanWang, Copilot | EikanWang, Copilot | Inductor / Compilation Related | P2 | UT issue with few failures |  | aten_ops | e2e |

### 1.6 Needs PyTorch Repo Changes (upstream)

| ID | Title | Owner | Owner Transferred | TBD | Priority | Reason | PR | Module | Test Module |
|---|-------|-------|-------------------|-----|---------|--------|-----|--------|-------------|
| [1505](https://github.com/intel/torch-xpu-ops/issues/1505) | [ARC-WSL-Ubuntu24.04] 15 Timm models got | xuhancn, Stonepia | xuhancn, Stonepia | Needs PyTorch Repo Changes (upstream) | P0 | Impacts real model/application |  | inductor | e2e |
| [1963](https://github.com/intel/torch-xpu-ops/issues/1963) | [upstream_ut] MetadataMismatchError in T | pbielak | pbielak | Needs PyTorch Repo Changes (upstream) | P2 | UT issue with few failures | [PR](https://github.com/pytorch/pytorch/pull/178277, https://github.com/pytorch/pytorch/pull/175965) | aten_ops | ut |
| [2214](https://github.com/intel/torch-xpu-ops/issues/2214) | test/test_sparse.py::TestSparseAnyXPU::t | jenniew | jenniew | Needs PyTorch Repo Changes (upstream) | P2 | UT issue with few failures |  | aten_ops | ut |
| [2234](https://github.com/intel/torch-xpu-ops/issues/2234) | [upstream_ut] AssertionError: RuntimeErr | Silv3S | Silv3S | Needs PyTorch Repo Changes (upstream) | P2 | UT issue with few failures |  | aten_ops | ut |
| [2248](https://github.com/intel/torch-xpu-ops/issues/2248) | [upstream_ut] test_cow failures | gplutop7 | gplutop7 | Needs PyTorch Repo Changes (upstream) | P2 | UT issue with few failures |  | aten_ops | ut |
| [2251](https://github.com/intel/torch-xpu-ops/issues/2251) | [upstream_ut] test_fake_autocase got Exc | astachowiczhabana | astachowiczhabana | Needs PyTorch Repo Changes (upstream) | P2 | UT issue with few failures |  | aten_ops | ut |
| [2255](https://github.com/intel/torch-xpu-ops/issues/2255) | [upstream_ut] RuntimeError: Long is not  | daisyden | daisyden | Needs PyTorch Repo Changes (upstream) | P2 | UT issue with few failures |  | aten_ops | ut |
| [2283](https://github.com/intel/torch-xpu-ops/issues/2283) | [upstream_ut] sparse._sampled_addmm is n | jenniew | jenniew | Needs PyTorch Repo Changes (upstream) | P1 | UT with 21 failed test cases |  | aten_ops | ut |
| [2287](https://github.com/intel/torch-xpu-ops/issues/2287) | [upstream_ut] test_python_ref issues | yucai-intel | yucai-intel | Needs PyTorch Repo Changes (upstream) | P2 | UT issue with few failures |  | aten_ops | ut |
| [2295](https://github.com/intel/torch-xpu-ops/issues/2295) | [upstream_ut][xpu][test]nn/test_embeddin | yucai-intel | yucai-intel | Needs PyTorch Repo Changes (upstream) | P2 | UT issue with few failures |  | inductor | ut |
| [2301](https://github.com/intel/torch-xpu-ops/issues/2301) | [upstream_ut] dtypes not align with OpIn | daisyden | daisyden | Needs PyTorch Repo Changes (upstream) | P2 | UT issue with few failures |  | aten_ops | ut |
| [2329](https://github.com/intel/torch-xpu-ops/issues/2329) | [upstream_ut] feature missing: get_devic | etaf | etaf | Needs PyTorch Repo Changes (upstream) | P2 | UT issue with few failures |  | inductor | ut |
| [2359](https://github.com/intel/torch-xpu-ops/issues/2359) | [upstream_ut] GradcheckError: Backward i | BBBela | BBBela | Needs PyTorch Repo Changes (upstream) | P2 | UT issue with few failures |  | aten_ops | ut |
| [2554](https://github.com/intel/torch-xpu-ops/issues/2554) | [upstream_ut]  AssertionError: Assertion | daisyden | daisyden | Needs PyTorch Repo Changes (upstream) | P2 | UT issue with few failures |  | inductor | ut |
| [2578](https://github.com/intel/torch-xpu-ops/issues/2578) | [TorchAO][UT] test/quantization/test_qua | Stonepia | Stonepia | Needs PyTorch Repo Changes (upstream) | P0 | Build crash - critical blocking issue |  | AO | build |
| [2609](https://github.com/intel/torch-xpu-ops/issues/2609) | [upstream_ut]  torch._inductor.exc.Induc | daisyden | daisyden | Needs PyTorch Repo Changes (upstream) | P2 | UT issue with few failures | [PR](https://github.com/pytorch/pytorch/pull/171154) | inductor | ut |
| [2620](https://github.com/intel/torch-xpu-ops/issues/2620) | [upstream_ut]  AssertionError: dtype is  | daisyden | daisyden | Needs PyTorch Repo Changes (upstream) | P2 | UT issue with few failures |  | inductor | ut |
| [2663](https://github.com/intel/torch-xpu-ops/issues/2663) | test_sparse_semi_structured.py gaps | None | None | Needs PyTorch Repo Changes (upstream) | P2 | UT issue with few failures |  | aten_ops | ut |
| [2670](https://github.com/intel/torch-xpu-ops/issues/2670) | [upstream_ut]  RuntimeError: could not c | tszulist-hbn | tszulist-hbn | Needs PyTorch Repo Changes (upstream) | P2 | UT issue with few failures |  | aten_ops | ut |
| [2693](https://github.com/intel/torch-xpu-ops/issues/2693) | Title: [upstream_ut]  AssertionError: Sc | hoshibara | hoshibara | Needs PyTorch Repo Changes (upstream) | P2 | UT issue with few failures |  | inductor | ut |
| [2696](https://github.com/intel/torch-xpu-ops/issues/2696) | Title: [upstream_ut]  RuntimeError: Expe | chunhuanMeng | chunhuanMeng | Needs PyTorch Repo Changes (upstream) | P2 | UT issue with few failures |  | inductor | e2e |
| [2697](https://github.com/intel/torch-xpu-ops/issues/2697) | Title: [upstream_ut]  RuntimeError: Expe | chunhuanMeng | chunhuanMeng | Needs PyTorch Repo Changes (upstream) | P2 | UT issue with few failures |  | inductor | e2e |
| [2698](https://github.com/intel/torch-xpu-ops/issues/2698) | Title: [upstream_ut]  RuntimeError: Flas | chunhuanMeng, LuFinch | chunhuanMeng, LuFinch | Needs PyTorch Repo Changes (upstream) | P2 | UT issue with few failures |  | inductor | ut |
| [2704](https://github.com/intel/torch-xpu-ops/issues/2704) | Title: [upstream_ut]  AssertionError: As | kdrozd-dev | kdrozd-dev | Needs PyTorch Repo Changes (upstream) | P2 | UT issue with few failures | [PR](https://github.com/pytorch/pytorch/pull/177636) | aten_ops | ut |
| [2712](https://github.com/intel/torch-xpu-ops/issues/2712) | [upstream_ut]  RuntimeError: Cannot swap | tszulist-hbn | tszulist-hbn | Needs PyTorch Repo Changes (upstream) | P2 | UT issue with few failures |  | aten_ops | ut |
| [2714](https://github.com/intel/torch-xpu-ops/issues/2714) | [upstream_ut]  AssertionError: Object co | Silv3S | Silv3S | Needs PyTorch Repo Changes (upstream) | P2 | UT issue with few failures |  | aten_ops | ut |
| [2715](https://github.com/intel/torch-xpu-ops/issues/2715) | [upstream_ut]  torch._dynamo.exc.Unsuppo | CuiYifeng | CuiYifeng | Needs PyTorch Repo Changes (upstream) | P2 | UT issue with few failures |  | aten_ops | ut |
| [2798](https://github.com/intel/torch-xpu-ops/issues/2798) | Test case  test/test_dlpack.py::TestTorc | None | None | Needs PyTorch Repo Changes (upstream) | P2 | UT issue with few failures |  | aten_ops | ut |
| [2800](https://github.com/intel/torch-xpu-ops/issues/2800) | AttributeError: 'torch._C._XpuDeviceProp | guangyey | guangyey | Needs PyTorch Repo Changes (upstream) | P2 | UT issue with few failures |  | inductor | ut |
| [2802](https://github.com/intel/torch-xpu-ops/issues/2802) | Three aten._scaled_dot_product_flash_att | LuFinch | LuFinch | Needs PyTorch Repo Changes (upstream) | P2 | UT issue with few failures |  | inductor | ut |
| [2806](https://github.com/intel/torch-xpu-ops/issues/2806) | CompiledAOTI need XPU support | daisyden | daisyden | Needs PyTorch Repo Changes (upstream) | P2 | UT issue with few failures |  | inductor | ut |
| [2810](https://github.com/intel/torch-xpu-ops/issues/2810) | AssertionError: Object comparison failed | daisyden | daisyden | Needs PyTorch Repo Changes (upstream) | P2 | UT issue with few failures |  | inductor | ut |
| [2888](https://github.com/intel/torch-xpu-ops/issues/2888) | torch._inductor.exc.InductorError: Asser | Stonepia | Stonepia | Needs PyTorch Repo Changes (upstream) | P2 | UT issue with few failures |  | inductor | ut |
| [2891](https://github.com/intel/torch-xpu-ops/issues/2891) | RuntimeError: Expected to find "(262144, | chunhuanMeng | chunhuanMeng | Needs PyTorch Repo Changes (upstream) | P2 | UT issue with few failures |  | inductor | e2e |
| [2918](https://github.com/intel/torch-xpu-ops/issues/2918) | [XPU][upstream_ut][COW] Skip non-support | gplutop7 | gplutop7 | Needs PyTorch Repo Changes (upstream) | P2 | UT issue with few failures | [PR](https://github.com/pytorch/pytorch/pull/174670) | aten_ops | ut |
| [2919](https://github.com/intel/torch-xpu-ops/issues/2919) | [XPU][upstream_ut][COW] Fix materializat | gplutop7 | gplutop7 | Needs PyTorch Repo Changes (upstream) | P2 | UT issue with few failures |  | aten_ops | ut |
| [2958](https://github.com/intel/torch-xpu-ops/issues/2958) | AssertionError of test_dtensor_basic_com | daisyden | daisyden | Needs PyTorch Repo Changes (upstream) | P2 | UT issue with few failures |  | inductor | ut |
| [2997](https://github.com/intel/torch-xpu-ops/issues/2997) | AssertionError of test_linear_and_cel_ma | etaf | etaf | Needs PyTorch Repo Changes (upstream) | P2 | UT issue with few failures |  | inductor | ut |
| [2999](https://github.com/intel/torch-xpu-ops/issues/2999) | KeyError: 'eager_numerics.use_pytorch_li | daisyden | daisyden | Needs PyTorch Repo Changes (upstream) | P2 | UT issue with few failures |  | inductor | ut |
| [3004](https://github.com/intel/torch-xpu-ops/issues/3004) | TypeError: _xpu_recordMemoryHistory(): i | guangyey | guangyey | Needs PyTorch Repo Changes (upstream) | P2 | UT issue with few failures |  | inductor | ut |
| [3006](https://github.com/intel/torch-xpu-ops/issues/3006) | AssertionError: '.to(tl.float16)' unexpe | CuiYifeng | CuiYifeng | Needs PyTorch Repo Changes (upstream) | P2 | UT issue with few failures |  | inductor | e2e |
| [3041](https://github.com/intel/torch-xpu-ops/issues/3041) | AssertionError: Expected len(flat_diff_r | Silv3S | Silv3S | Needs PyTorch Repo Changes (upstream) | P2 | UT issue with few failures |  | aten_ops | ut |
| [3077](https://github.com/intel/torch-xpu-ops/issues/3077) | [Bug Skip] test_dlpack.py::TestTorchDlPa | AKloniecki | AKloniecki | Needs PyTorch Repo Changes (upstream) | P2 | UT issue with few failures |  | aten_ops | ut |
| [3094](https://github.com/intel/torch-xpu-ops/issues/3094) | XPUGraph tree support | None | None | Needs PyTorch Repo Changes (upstream) | P2 | UT issue with few failures |  | inductor | ut |
| [3095](https://github.com/intel/torch-xpu-ops/issues/3095) | cutlass support blocks some unit test ca | None | None | Needs PyTorch Repo Changes (upstream) | P2 | UT issue with few failures |  | inductor | ut |
| [3126](https://github.com/intel/torch-xpu-ops/issues/3126) | [upstream_ut]  Two NestedTensor issue wi | LuFinch | LuFinch | Needs PyTorch Repo Changes (upstream) | P2 | UT issue with few failures |  | aten_ops | ut |
| [3127](https://github.com/intel/torch-xpu-ops/issues/3127) | [upstream_ut]  AssertionError: Assertion | LuFinch | LuFinch | Needs PyTorch Repo Changes (upstream) | P2 | UT issue with few failures |  | aten_ops | ut |
| [3128](https://github.com/intel/torch-xpu-ops/issues/3128) | [upstream_ut]  AssertionError: RuntimeEr | LuFinch | LuFinch | Needs PyTorch Repo Changes (upstream) | P2 | UT issue with few failures |  | aten_ops | ut |
| [3129](https://github.com/intel/torch-xpu-ops/issues/3129) | [upstream_ut]  AssertionError: UserWarni | LuFinch | LuFinch | Needs PyTorch Repo Changes (upstream) | P2 | UT issue with few failures |  | aten_ops | ut |
| [3130](https://github.com/intel/torch-xpu-ops/issues/3130) | [upstream_ut]  AssertionError: tensor(Tr | LuFinch | LuFinch | Needs PyTorch Repo Changes (upstream) | P2 | UT issue with few failures |  | aten_ops | ut |
| [3131](https://github.com/intel/torch-xpu-ops/issues/3131) | [upstream_ut]  NotImplementedError: The  | chunhuanMeng | chunhuanMeng | Needs PyTorch Repo Changes (upstream) | P2 | UT issue with few failures |  | aten_ops | ut |
| [3132](https://github.com/intel/torch-xpu-ops/issues/3132) | [upstream_ut]  transfomers test reports  | LuFinch | LuFinch | Needs PyTorch Repo Changes (upstream) | P2 | UT issue with few failures |  | aten_ops | ut |
| [3133](https://github.com/intel/torch-xpu-ops/issues/3133) | [upstream_ut]  RuntimeError: scaled_dot_ | LuFinch | LuFinch | Needs PyTorch Repo Changes (upstream) | P2 | UT issue with few failures |  | aten_ops | ut |
| [3137](https://github.com/intel/torch-xpu-ops/issues/3137) | [upstream_ut]  RuntimeError: expected sc | LuFinch | LuFinch | Needs PyTorch Repo Changes (upstream) | P2 | UT issue with few failures |  | aten_ops | ut |
| [3140](https://github.com/intel/torch-xpu-ops/issues/3140) | [upstream_ut]  RuntimeError: FlashAttent | LuFinch | LuFinch | Needs PyTorch Repo Changes (upstream) | P2 | UT issue with few failures |  | aten_ops | ut |
| [3141](https://github.com/intel/torch-xpu-ops/issues/3141) | [upstream_ut]  RuntimeError: FlashAttent | LuFinch | LuFinch | Needs PyTorch Repo Changes (upstream) | P2 | UT issue with few failures |  | aten_ops | ut |
| [3142](https://github.com/intel/torch-xpu-ops/issues/3142) | [upstream_ut]  RuntimeError: The sycl_ex | LuFinch | LuFinch | Needs PyTorch Repo Changes (upstream) | P2 | UT issue with few failures |  | aten_ops | ut |
| [3143](https://github.com/intel/torch-xpu-ops/issues/3143) | NotImplementedError: The operator 'aten: | LuFinch | LuFinch | Needs PyTorch Repo Changes (upstream) | P2 | UT issue with few failures |  | aten_ops | ut |
| [3163](https://github.com/intel/torch-xpu-ops/issues/3163) | [Bug Skip]: Object comparison failed: to | chunhuanMeng | chunhuanMeng | Needs PyTorch Repo Changes (upstream) | P2 | UT issue with few failures |  | aten_ops | ut |
| [3165](https://github.com/intel/torch-xpu-ops/issues/3165) | test_sparse_csr_xpu.py::TestSparseCompre | None | None | Needs PyTorch Repo Changes (upstream) | P2 | UT issue with few failures |  | aten_ops | ut |
| [3167](https://github.com/intel/torch-xpu-ops/issues/3167) | NotImplementedError: Could not run 'aten | tszulist-hbn | tszulist-hbn | Needs PyTorch Repo Changes (upstream) | P1 | UT with 68 failed test cases |  | aten_ops | ut |
| [3168](https://github.com/intel/torch-xpu-ops/issues/3168) | NotImplementedError: Could not run 'aten | jenniew | jenniew | Needs PyTorch Repo Changes (upstream) | P1 | UT with 40 failed test cases |  | aten_ops | ut |
| [3169](https://github.com/intel/torch-xpu-ops/issues/3169) | NotImplementedError: Could not run 'aten | jkosnox | jkosnox | Needs PyTorch Repo Changes (upstream) | P2 | UT issue with few failures |  | aten_ops | ut |
| [3170](https://github.com/intel/torch-xpu-ops/issues/3170) | Unskip test_bmm_windows_error_xpu_float6 | jenniew | jenniew | Needs PyTorch Repo Changes (upstream) | P2 | UT issue with few failures |  | aten_ops | ut |
| [3187](https://github.com/intel/torch-xpu-ops/issues/3187) | PyTorch XPU gpu_cpp_wrapper fails with I | CuiYifeng | CuiYifeng | Needs PyTorch Repo Changes (upstream) | P2 | UT issue with few failures |  | aten_ops | ut |
| [3229](https://github.com/intel/torch-xpu-ops/issues/3229) | RuntimeError: No viable backend for scal | None | None | Needs PyTorch Repo Changes (upstream) | P2 | UT issue with few failures |  | aten_ops | ut |
| [3238](https://github.com/intel/torch-xpu-ops/issues/3238) | The supported dtypes of _refs.stft is no | CuiYifeng | CuiYifeng | Needs PyTorch Repo Changes (upstream) | P2 | UT issue with few failures |  | aten_ops | ut |
| [3247](https://github.com/intel/torch-xpu-ops/issues/3247) | NotImplementedError: "dot_xpu_mkl" not i | Silv3S | Silv3S | Needs PyTorch Repo Changes (upstream) | P2 | UT issue with few failures |  | aten_ops | ut |

### 1.7 Sparse Operations Related

| ID | Title | Owner | Owner Transferred | TBD | Priority | Reason | PR | Module | Test Module |
|---|-------|-------|-------------------|-----|---------|--------|-----|--------|-------------|
| [1962](https://github.com/intel/torch-xpu-ops/issues/1962) | [upstream_ut] segfault with test_fake_cr | jenniew, mengfei25 | jenniew, mengfei25 | Sparse Operations Related | P0 | Build crash - critical blocking issue |  | aten_ops | ut |
| [2229](https://github.com/intel/torch-xpu-ops/issues/2229) | test/test_sparse_csr.py::TestSparseCompr | jenniew | jenniew | Sparse Operations Related | P2 | UT issue with few failures |  | aten_ops | ut |
| [2245](https://github.com/intel/torch-xpu-ops/issues/2245) | oneDNN matmul received incorrect shape i | CuiYifeng | CuiYifeng | Sparse Operations Related | P2 | UT issue with few failures |  | aten_ops | ut |
| [2595](https://github.com/intel/torch-xpu-ops/issues/2595) | [Bug Skip]: Random crashed cases 2025-12 | None | None | Sparse Operations Related | P0 | Build crash - critical blocking issue |  | aten_ops | ut |
| [2729](https://github.com/intel/torch-xpu-ops/issues/2729) | [Bug Skip]: Random failures 2026WW03 | Silv3S | Silv3S | Sparse Operations Related | P2 | UT issue with few failures |  | unknown | ut |
| [2751](https://github.com/intel/torch-xpu-ops/issues/2751) | [Bug Skip]: Random failures 2026WW04 | None | None | Sparse Operations Related | P2 | UT issue with few failures |  | unknown | ut |
| [2777](https://github.com/intel/torch-xpu-ops/issues/2777) | [Bug Skip]: Random failures 2026WW05 | AKloniecki | AKloniecki | Sparse Operations Related | P2 | UT issue with few failures |  | unknown | ut |
| [2801](https://github.com/intel/torch-xpu-ops/issues/2801) | to_dense() for Sparse CSR backend cannot | jenniew | jenniew | Sparse Operations Related | P2 | UT issue with few failures |  | aten_ops | ut |
| [2921](https://github.com/intel/torch-xpu-ops/issues/2921) | [abs][complex64] - new failing test case | AKloniecki | AKloniecki | Sparse Operations Related | P2 | UT issue with few failures |  | aten_ops | ut |
| [2946](https://github.com/intel/torch-xpu-ops/issues/2946) | [Bug Skip]: Random failures 2026WW09 | None | None | Sparse Operations Related | P2 | UT issue with few failures |  | unknown | ut |
| [2965](https://github.com/intel/torch-xpu-ops/issues/2965) | [Bug Skip]: Random failures 2026WW10 | None | None | Sparse Operations Related | P2 | UT issue with few failures |  | unknown | ut |
| [3081](https://github.com/intel/torch-xpu-ops/issues/3081) | Sparse CSR gemm-like ops have not been s | None | None | Sparse Operations Related | P2 | UT issue with few failures |  | aten_ops | ut |

### 1.8 Verify the issue

| ID | Title | Owner | Owner Transferred | TBD | Priority | Reason | PR | Module | Test Module |
|---|-------|-------|-------------------|-----|---------|--------|-----|--------|-------------|
| [2331](https://github.com/intel/torch-xpu-ops/issues/2331) | [upstream_ut] AssertionError: Scalars ar | hoshibara | daisyden | Verify the issue | P2 | UT issue with few failures | [PR](https://github.com/pytorch/pytorch/pull/172314) | inductor | ut |
| [2694](https://github.com/intel/torch-xpu-ops/issues/2694) | Title: [upstream_ut]  AssertionError: Te | daisyden | daisyden | Verify the issue | P2 | UT issue with few failures | [PR](https://github.com/pytorch/pytorch/pull/171773) | inductor | ut |
| [3007](https://github.com/intel/torch-xpu-ops/issues/3007) | AssertionError: Scalars are not equal! w | daisyden | daisyden | Verify the issue | P2 | UT issue with few failures | [PR](https://github.com/pytorch/pytorch/pull/178369) | inductor | e2e |

### 1.9 add to skiplist

| ID | Title | Owner | Owner Transferred | TBD | Priority | Reason | PR | Module | Test Module |
|---|-------|-------|-------------------|-----|---------|--------|-----|--------|-------------|
| [2164](https://github.com/intel/torch-xpu-ops/issues/2164) | skip test_no_cuda_monkeypatch as it is c | daisyden | daisyden | add to skiplist | P2 | UT issue with few failures |  | aten_ops | ut |
| [2309](https://github.com/intel/torch-xpu-ops/issues/2309) | unsupported ops with PYTORCH_ENABLE_XPU_ | daisyden | daisyden | add to skiplist | P2 | UT issue with few failures |  | aten_ops | ut |
| [2472](https://github.com/intel/torch-xpu-ops/issues/2472) | [upstream_ut]  NotImplementedError: The  | Silv3S | daisyden | add to skiplist | P2 | UT issue with few failures |  | aten_ops | ut |
| [2508](https://github.com/intel/torch-xpu-ops/issues/2508) | TypedStorage / TypedTensors deprecation | Silv3S | libohao1201 | add to skiplist | P1 | UT with 27 failed test cases | [PR](https://github.com/intel/torch-xpu-ops/pull/3260) | aten_ops | ut |

### Issues without Assignee

| ID | Title | Owner | Owner Transferred | TBD | Priority | Reason | PR | Module | Test Module |
|---|-------|-------|-------------------|-----|---------|--------|-----|--------|-------------|
| [3257](https://github.com/intel/torch-xpu-ops/issues/3257) | [Linux][E2E][Regression] Huggingface tes | None | chuanqi | assign owner | P0 | Impacts real model/application |  | aten_ops | e2e |
| [3255](https://github.com/intel/torch-xpu-ops/issues/3255) | [Linux][PT2E][Regression] Some performan | None | chuanqi | assign owner | P0 | Regression - passed before but failed now |  | aten_ops | e2e |
| [3233](https://github.com/intel/torch-xpu-ops/issues/3233) | [distributed] RuntimeError: No backend f | None | chuanqi | assign owner | P2 | UT issue with few failures |  | distributed | ut |
| [3232](https://github.com/intel/torch-xpu-ops/issues/3232) | [distributed][tensor] AssertionError: As | None | chuanqi | assign owner | P2 | UT issue with few failures |  | distributed | ut |
| [3180](https://github.com/intel/torch-xpu-ops/issues/3180) | [E2E] Timm/Torchbench models got "eager_ | None | chuanqi | assign owner | P0 | Impacts real model/application |  | aten_ops | ut |
| [3149](https://github.com/intel/torch-xpu-ops/issues/3149) | New failure in test_rms_norm_decomp_acce | None | chuanqi | assign owner | P2 | UT issue with few failures |  | aten_ops | ut |
| [3124](https://github.com/intel/torch-xpu-ops/issues/3124) | [TorchAO][Bug] ImportError: Requires msl | None | chuanqi | assign owner | P2 | UT issue with few failures |  | aten_ops | ut |
| [3121](https://github.com/intel/torch-xpu-ops/issues/3121) | [Bug Skip]: CUDA specific UT test_fft_ha | None | chuanqi | assign owner | P1 | UT with 60 failed test cases |  | aten_ops | ut |
| [3102](https://github.com/intel/torch-xpu-ops/issues/3102) | [distributed] RuntimeError: Invalid devi | None | chuanqi | assign owner | P2 | UT issue with few failures |  | distributed | ut |
| [3101](https://github.com/intel/torch-xpu-ops/issues/3101) | [distributed] 'torch._C._distributed_c10 | None | chuanqi | assign owner | P2 | UT issue with few failures |  | distributed | ut |
| [3100](https://github.com/intel/torch-xpu-ops/issues/3100) | [distributed] /handler/dump_nccl_trace_p | None | chuanqi | assign owner | P2 | UT issue with few failures |  | distributed | ut |
| [3096](https://github.com/intel/torch-xpu-ops/issues/3096) | VISIBLE_DEVICE support | None | chuanqi | assign owner | P2 | UT issue with few failures |  | aten_ops | ut |
| [3086](https://github.com/intel/torch-xpu-ops/issues/3086) | nvml support blocks some test cases | None | chuanqi | assign owner | P2 | UT issue with few failures |  | aten_ops | ut |
| [3084](https://github.com/intel/torch-xpu-ops/issues/3084) | torch.library.register_autocast does not | None | chuanqi | assign owner | P2 | UT issue with few failures |  | aten_ops | ut |
| [3083](https://github.com/intel/torch-xpu-ops/issues/3083) | [Bug Skip]: Random failures 2026WW12 | None | chuanqi | assign owner | P2 | UT issue with few failures |  | unknown | ut |
| [3082](https://github.com/intel/torch-xpu-ops/issues/3082) | multithread support in distributed | None | chuanqi | assign owner | P2 | UT issue with few failures |  | distributed | ut |
| [3080](https://github.com/intel/torch-xpu-ops/issues/3080) | cudagraph tests blocked by feature gap | None | chuanqi | assign owner | P2 | UT issue with few failures |  | aten_ops | ut |
| [3076](https://github.com/intel/torch-xpu-ops/issues/3076) | [TorchAO][BMG] Llama-3.2-1B-Instruct Dyn | None | chuanqi | assign owner | P2 | UT issue with few failures |  | aten_ops | ut |
| [3025](https://github.com/intel/torch-xpu-ops/issues/3025) | New failing test in Nightly Wheel test_d | None | chuanqi | assign owner | P2 | UT issue with few failures |  | aten_ops | ut |
| [2979](https://github.com/intel/torch-xpu-ops/issues/2979) | eca_halonext26ts got RuntimeError: ZE_RE | None | chuanqi | assign owner | P0 | Build crash - critical blocking issue |  | aten_ops | e2e |
| [2948](https://github.com/intel/torch-xpu-ops/issues/2948) | [AO] Benchmark enabling on XPU | None | chuanqi | assign owner | P2 | UT issue with few failures |  | AO | ut |
| [2930](https://github.com/intel/torch-xpu-ops/issues/2930) | [release/2.11] UT skip test_binary_ufunc | None | chuanqi | assign owner | P2 | UT issue with few failures |  | aten_ops | ut |
| [2914](https://github.com/intel/torch-xpu-ops/issues/2914) | Test case test/test_autograd.py::TestAut | None | chuanqi | assign owner | P2 | UT issue with few failures |  | aten_ops | ut |
| [2912](https://github.com/intel/torch-xpu-ops/issues/2912) | [release/2.11] UT extended 220 new failu | None | chuanqi | assign owner | P2 | UT issue with few failures |  | aten_ops | ut |
| [2899](https://github.com/intel/torch-xpu-ops/issues/2899) | Update nan_to_num XPU stub to use std::o | None | chuanqi | assign owner | P2 | UT issue with few failures |  | aten_ops | ut |
| [2858](https://github.com/intel/torch-xpu-ops/issues/2858) | [Bug Skip]: test_xpu new failures | None | chuanqi | assign owner | P2 | UT issue with few failures |  | unknown | ut |
| [2852](https://github.com/intel/torch-xpu-ops/issues/2852) | [Bug Skip]: New UT failures in 0206 nigh | None | chuanqi | assign owner | P2 | UT issue with few failures |  | aten_ops | ut |
| [2845](https://github.com/intel/torch-xpu-ops/issues/2845) | [Bug Skip]:[UT] [Windows] failed cases 2 | None | chuanqi | assign owner | P2 | UT issue with few failures |  | aten_ops | ut |
| [2797](https://github.com/intel/torch-xpu-ops/issues/2797) | Copy error is not raise on test_dlpack.p | None | chuanqi | assign owner | P2 | UT issue with few failures |  | aten_ops | ut |
| [2737](https://github.com/intel/torch-xpu-ops/issues/2737) | [distributed] AttributeError: module 'to | None | chuanqi | assign owner | P2 | UT issue with few failures |  | distributed | ut |
| [2676](https://github.com/intel/torch-xpu-ops/issues/2676) | Random failure in CI test | None | chuanqi | assign owner | P2 | UT issue with few failures |  | unknown | ut |
| [2539](https://github.com/intel/torch-xpu-ops/issues/2539) | Title: [upstream_ut]  RuntimeError: Trie | None | chuanqi | assign owner | P2 | UT issue with few failures |  | aten_ops | ut |
| [2447](https://github.com/intel/torch-xpu-ops/issues/2447) | test_share_memory_xpu failure | None | chuanqi | assign owner | P2 | UT issue with few failures |  | aten_ops | ut |
| [2446](https://github.com/intel/torch-xpu-ops/issues/2446) | [Bug Skip]: AssertionError: "Simulate er | None | chuanqi | assign owner | P2 | UT issue with few failures |  | aten_ops | ut |
| [2440](https://github.com/intel/torch-xpu-ops/issues/2440) | [For UT failures classify] Save referenc | None | chuanqi | assign owner | P2 | UT issue with few failures |  | inductor | ut |
| [2404](https://github.com/intel/torch-xpu-ops/issues/2404) | [distributed][checkpoint] AssertionError | None | chuanqi | assign owner | P2 | UT issue with few failures |  | aten_ops | ut |
| [2333](https://github.com/intel/torch-xpu-ops/issues/2333) | [Don't merge] Collect the new passed cas | None | chuanqi | assign owner | P2 | UT issue with few failures |  | aten_ops | ut |
| [2327](https://github.com/intel/torch-xpu-ops/issues/2327) | [TorchAO] benchmark enabling on XPU | None | chuanqi | assign owner | P2 | UT issue with few failures |  | aten_ops | ut |
| [2199](https://github.com/intel/torch-xpu-ops/issues/2199) | Fix reduction and norm register spill | None | chuanqi | assign owner | P2 | UT issue with few failures |  | aten_ops | ut |
| [2196](https://github.com/intel/torch-xpu-ops/issues/2196) | Fix DistributionElementwiseKernelFunctor | None | chuanqi | assign owner | P2 | UT issue with few failures |  | aten_ops | ut |
| [2022](https://github.com/intel/torch-xpu-ops/issues/2022) | [Windows] [CI] [UT] AssertionError: Tens | None | chuanqi | assign owner | P2 | UT issue with few failures |  | aten_ops | ut |
| [1689](https://github.com/intel/torch-xpu-ops/issues/1689) | [For op Perf Comparison] Save reference  | None | chuanqi | assign owner | P2 | UT issue with few failures |  | aten_ops | ut |
| [1678](https://github.com/intel/torch-xpu-ops/issues/1678) | missing op support for `model.share_memo | None | chuanqi | assign owner | P0 | Impacts real model/application |  | aten_ops | ut |
| [1519](https://github.com/intel/torch-xpu-ops/issues/1519) | [PVC][PT2.7][ABI=1][Torch-xpu-ops UT][ww | None | chuanqi | assign owner | P2 | UT issue with few failures |  | aten_ops | ut |

---

## 2. Duplicated Issues

Issues that share test cases with other issues.

| ID | Title | Owner | Reporter | Duplicated With | Priority | Reason | PR | Module | Test Module |
|---|-------|-------|----------|-----------------|---------|--------|-----|--------|-------------|
| [1893](https://github.com/intel/torch-xpu-ops/issues/1893) | [upstream_ut] oneDNN accuracy issue | chunhuanMeng | daisyden | 1951 | P2 | UT issue with few failures |  | aten_ops | ut |
| [1951](https://github.com/intel/torch-xpu-ops/issues/1951) | Functionality issues in TestCommon. | AKloniecki | daisyden | 1893 | P2 | UT issue with few failures |  | aten_ops | ut |
| [1973](https://github.com/intel/torch-xpu-ops/issues/1973) | AssertionError: Scalars or Tensor-l | gplutop7 | mengfei25 | 2837,2840 | P2 | UT issue with few failures |  | aten_ops | ut |
| [2006](https://github.com/intel/torch-xpu-ops/issues/2006) | work-item/workgroup issue in softma | BartoszKokoszko | daisyden | 2257 | P2 | UT issue with few failures |  | aten_ops | ut |
| [2015](https://github.com/intel/torch-xpu-ops/issues/2015) | inf is returned by nn.TransformerEn | yucai-intel | daisyden | 2186,2529 | P2 | UT issue with few failures |  | aten_ops | ut |
| [2186](https://github.com/intel/torch-xpu-ops/issues/2186) | AssertionError: Mul tiheadAttention | daisyden | daisyden | 2015 | P2 | UT issue with few failures |  | aten_ops | ut |
| [2220](https://github.com/intel/torch-xpu-ops/issues/2220) | test/test_sparse_csr.py::TestSparse | None | wincent8 | 2246 | P2 | UT issue with few failures |  | aten_ops | ut |
| [2230](https://github.com/intel/torch-xpu-ops/issues/2230) | test_sparse_csr.py::TestSparseCompr | None | wincent8 | 2246,3175,3176 | P1 | UT with 28 failed test cases |  | aten_ops | ut |
| [2235](https://github.com/intel/torch-xpu-ops/issues/2235) | test/test_sparse_csr.py::TestSparse | None | wincent8 | 3047 | P2 | UT issue with few failures |  | aten_ops | ut |
| [2238](https://github.com/intel/torch-xpu-ops/issues/2238) | Exception: Tensor-likes are not clo | BBBela | zxd1997066 | 3105 | P2 | UT issue with few failures | [PR](https://github.com/intel/torch-xpu-ops/pull/2886) | aten_ops | ut |
| [2244](https://github.com/intel/torch-xpu-ops/issues/2244) | test/test_sparse_csr.py::TestSparse | jenniew | wincent8 | 3177 | P2 | UT issue with few failures |  | aten_ops | ut |
| [2246](https://github.com/intel/torch-xpu-ops/issues/2246) | torch/sparse/_triton_ops*.py need t | None | wincent8 | 2220,2230 | P1 | UT with 33 failed test cases |  | unknown | ut |
| [2253](https://github.com/intel/torch-xpu-ops/issues/2253) | the supported dtypes are not align  | daisyden | daisyden | 2482 | P2 | UT issue with few failures |  | aten_ops | ut |
| [2257](https://github.com/intel/torch-xpu-ops/issues/2257) | Accuracy failures in test/xpu/test_ | pbielak | zxd1997066 | 2006 | P1 | UT with 40 failed test cases | [PR](https://github.com/intel/torch-xpu-ops/pull/2808) | aten_ops | ut |
| [2270](https://github.com/intel/torch-xpu-ops/issues/2270) | Backend Compatibility Error in test | LuFinch | libohao1201 | 2442 | P2 | UT issue with few failures |  | aten_ops | ut |
| [2285](https://github.com/intel/torch-xpu-ops/issues/2285) | Support efficient attention | chunhuanMeng | daisyden | 2358 | P2 | UT issue with few failures |  | aten_ops | ut |
| [2358](https://github.com/intel/torch-xpu-ops/issues/2358) | test/test_view_ops.py::TestOldViewO | Silv3S | wincent8 | 2285 | P2 | UT issue with few failures |  | aten_ops | ut |
| [2436](https://github.com/intel/torch-xpu-ops/issues/2436) | [upstream_ut]  AttributeError: 'Non | daisyden | daisyden | 2675 | P1 | UT with 51 failed test cases |  | aten_ops | ut |
| [2442](https://github.com/intel/torch-xpu-ops/issues/2442) | [Bug Skip]: NotImplementedError: Co | daisyden, LuFinch | CuiYifeng | 2270 | P2 | UT issue with few failures |  | aten_ops | ut |
| [2482](https://github.com/intel/torch-xpu-ops/issues/2482) | test_dtypes issue introduced by pyt | daisyden | daisyden | 2253 | P2 | UT issue with few failures |  | aten_ops | ut |
| [2529](https://github.com/intel/torch-xpu-ops/issues/2529) | [upstream_ut]  AssertionError: Fals | Silv3S | daisyden | 2015,3136 | P2 | UT issue with few failures |  | aten_ops | ut |
| [2530](https://github.com/intel/torch-xpu-ops/issues/2530) | Title: [upstream_ut]  AssertionErro | PatrykWilczewski | daisyden | 2817 | P2 | UT issue with few failures |  | aten_ops | ut |
| [2611](https://github.com/intel/torch-xpu-ops/issues/2611) | [upstream_ut]  AssertionError: Tens | daisyden | daisyden | 2613 | P2 | UT issue with few failures |  | inductor | ut |
| [2613](https://github.com/intel/torch-xpu-ops/issues/2613) | [upstream_ut]  AssertionError: Tens | daisyden | daisyden | 2611 | P2 | UT issue with few failures |  | inductor | ut |
| [2618](https://github.com/intel/torch-xpu-ops/issues/2618) | [Bug Skip]: [regression] AssertionE | jmamzax | kaileiyx | 3089 | P0 | Regression - passed before but failed now | [PR](https://github.com/numpy/numpy/pull/22525) | unknown | ut |
| [2675](https://github.com/intel/torch-xpu-ops/issues/2675) | [Bug Skip]: AttributeError: 'NoneTy | pponikox | kaileiyx | 2436 | P2 | UT issue with few failures |  | aten_ops | ut |
| [2817](https://github.com/intel/torch-xpu-ops/issues/2817) | Expected error message is different | kdrozd-dev | Silv3S | 2530 | P2 | UT issue with few failures |  | aten_ops | ut |
| [2837](https://github.com/intel/torch-xpu-ops/issues/2837) | Accuracy issue for Muon optimizer | Silv3S | kdrozd-dev | 1973 | P2 | UT issue with few failures |  | aten_ops | ut |
| [2840](https://github.com/intel/torch-xpu-ops/issues/2840) | Accuracy issue with 64 bit indexing | SlawomirLaba, Silv3S | kdrozd-dev | 1973 | P2 | UT issue with few failures |  | aten_ops | ut |
| [2869](https://github.com/intel/torch-xpu-ops/issues/2869) | [Bug Skip]: New UT failure in 0209  | None | RUIJIEZHONG66166 | 3160 | P2 | UT issue with few failures |  | aten_ops | ut |
| [2966](https://github.com/intel/torch-xpu-ops/issues/2966) | [Bug Skip]: [Regression]2026-3-2 ut | jmamzax | kaileiyx | 3114 | P0 | Regression - passed before but failed now |  | aten_ops | ut |
| [3047](https://github.com/intel/torch-xpu-ops/issues/3047) | [Bug Skip]: [Regression]UT failures | None | kaileiyx | 2235 | P0 | Regression - passed before but failed now |  | unknown | ut |
| [3089](https://github.com/intel/torch-xpu-ops/issues/3089) | AssertionError: Torch not compiled  | jmamzax | jmamzax | 2618 | P2 | UT issue with few failures |  | unknown | ut |
| [3105](https://github.com/intel/torch-xpu-ops/issues/3105) | Wrong results from oneDNN conv2d ke | BBBela | BBBela | 2238 | P2 | UT issue with few failures |  | aten_ops | ut |
| [3114](https://github.com/intel/torch-xpu-ops/issues/3114) | [Bug Skip]: Failure skip on 2026-3- | None | guangyey | 2966 | P2 | UT issue with few failures |  | aten_ops | ut |
| [3136](https://github.com/intel/torch-xpu-ops/issues/3136) | [upstream_ut]  AssertionError: Fals | LuFinch | daisyden | 2529 | P2 | UT issue with few failures |  | aten_ops | ut |
| [3156](https://github.com/intel/torch-xpu-ops/issues/3156) | AssertionError: 'Assertion cur_targ | kdrozd-dev | kdrozd-dev | 3184 | P2 | UT issue with few failures |  | aten_ops | ut |
| [3160](https://github.com/intel/torch-xpu-ops/issues/3160) | compiler not found (Windows) | kdrozd-dev | kdrozd-dev | 2869 | P2 | UT issue with few failures |  | aten_ops | ut |
| [3175](https://github.com/intel/torch-xpu-ops/issues/3175) | [Bug Skip]: ValueError: sampled_add | None | CuiYifeng | 2230 | P2 | UT issue with few failures |  | aten_ops | ut |
| [3176](https://github.com/intel/torch-xpu-ops/issues/3176) | [Bug Skip]: ValueError: _scaled_dot | None | CuiYifeng | 2230 | P2 | UT issue with few failures |  | aten_ops | ut |
| [3177](https://github.com/intel/torch-xpu-ops/issues/3177) | Accuracy gap of BF16/FP16 test_bloc | jenniew | CuiYifeng | 2244 | P2 | UT issue with few failures |  | aten_ops | ut |
| [3184](https://github.com/intel/torch-xpu-ops/issues/3184) | New failing UTs: test_cross_entropy | wpietka | BBBela | 3156 | P2 | UT issue with few failures |  | aten_ops | ut |

---

## 3. Issues with Dependency

Issues that have dependencies on other components (non-empty).

| ID | Title | Owner | Type | Priority | Reason | Module | Test Module | Dependency | PR | Labels |
|---|-------|------|------|---------|--------|--------|-------------|------------|-----|--------|
| [1059](https://github.com/intel/torch-xpu-ops/issues/1059) | SYCL RT: Using recommended sho | CuiYifeng, jianyizh | unknown | P2 | UT issue with few failures | aten_ops | ut | oneAPI |  | dependency component: oneAPI |
| [1059](https://github.com/intel/torch-xpu-ops/issues/1059) | SYCL RT: Using recommended sho | CuiYifeng, jianyizh | unknown | aten_ops | ut | oneAPI |  | dependency component: oneAPI |
| [1171](https://github.com/intel/torch-xpu-ops/issues/1171) | LNL Windows got unexpected err | xuhancn, chunhuanMeng | functionality bug | P2 | UT issue with few failures | aten_ops | ut | driver |  | client, os: Windows, hw : LNL, hw: BMG, dependency component: driver |
| [1171](https://github.com/intel/torch-xpu-ops/issues/1171) | LNL Windows got unexpected err | xuhancn, chunhuanMeng | functionality bug | aten_ops | ut | driver |  | client, os: Windows, hw : LNL, hw: BMG, dependency component: driver |
| [1324](https://github.com/intel/torch-xpu-ops/issues/1324) | [Win] UR Error when OOM and br | Stonepia | functionality bug | P2 | UT issue with few failures | aten_ops | ut | oneAPI |  | client, os: Windows, module: dependency bug, dependency component: driver, dependency component: oneAPI |
| [1324](https://github.com/intel/torch-xpu-ops/issues/1324) | [Win] UR Error when OOM and br | Stonepia | functionality bug | aten_ops | ut | oneAPI |  | client, os: Windows, module: dependency bug, dependency component: driver, dependency component: oneAPI |
| [1510](https://github.com/intel/torch-xpu-ops/issues/1510) | Some test cases will be hang o | Stonepia, mengfei25 | functionality bug | P2 | UT issue with few failures | aten_ops | ut | driver |  | bug, client, os: Ubuntu, hw: BMG, dependency component: driver, module: ut |
| [1510](https://github.com/intel/torch-xpu-ops/issues/1510) | Some test cases will be hang o | Stonepia, mengfei25 | functionality bug | aten_ops | ut | driver |  | bug, client, os: Ubuntu, hw: BMG, dependency component: driver, module: ut |
| [1547](https://github.com/intel/torch-xpu-ops/issues/1547) | [distributed] NotImplementedEr | Chao1Han | feature request | P2 | UT issue with few failures | distributed | ut | oneAPI |  | module: distributed, dependency component: oneAPI |
| [1547](https://github.com/intel/torch-xpu-ops/issues/1547) | [distributed] NotImplementedEr | Chao1Han | feature request | distributed | ut | oneAPI |  | module: distributed, dependency component: oneAPI |
| [1549](https://github.com/intel/torch-xpu-ops/issues/1549) | [distributed] AssertionError:  | Chao1Han | functionality bug | P2 | UT issue with few failures | distributed | ut | oneAPI |  | module: distributed, dependency component: oneAPI |
| [1549](https://github.com/intel/torch-xpu-ops/issues/1549) | [distributed] AssertionError:  | Chao1Han | functionality bug | distributed | ut | oneAPI |  | module: distributed, dependency component: oneAPI |
| [1551](https://github.com/intel/torch-xpu-ops/issues/1551) | [distributed] NotImplementedEr | Chao1Han | feature request | P2 | UT issue with few failures | distributed | ut | oneAPI |  | module: distributed, dependency component: oneAPI |
| [1551](https://github.com/intel/torch-xpu-ops/issues/1551) | [distributed] NotImplementedEr | Chao1Han | feature request | distributed | ut | oneAPI |  | module: distributed, dependency component: oneAPI |
| [1555](https://github.com/intel/torch-xpu-ops/issues/1555) | [distributed] RuntimeError: at | chuanqi129 | functionality bug | P2 | UT issue with few failures | distributed | ut | oneDNN |  | module: distributed, dependency component: oneDNN |
| [1555](https://github.com/intel/torch-xpu-ops/issues/1555) | [distributed] RuntimeError: at | chuanqi129 | functionality bug | distributed | ut | oneDNN |  | module: distributed, dependency component: oneDNN |
| [1649](https://github.com/intel/torch-xpu-ops/issues/1649) | [cpp extension] Provide a clea | dvrogozh | feature request | P2 | UT issue with few failures | aten_ops | ut | oneAPI |  | dependency component: oneAPI, module: build |
| [1649](https://github.com/intel/torch-xpu-ops/issues/1649) | [cpp extension] Provide a clea | dvrogozh | feature request | aten_ops | ut | oneAPI |  | dependency component: oneAPI, module: build |
| [1722](https://github.com/intel/torch-xpu-ops/issues/1722) | Ask an API to query GPU type(i | guangyey | unknown | P2 | UT issue with few failures | aten_ops | ut | oneAPI |  | dependency component: oneAPI |
| [1722](https://github.com/intel/torch-xpu-ops/issues/1722) | Ask an API to query GPU type(i | guangyey | unknown | aten_ops | ut | oneAPI |  | dependency component: oneAPI |
| [1727](https://github.com/intel/torch-xpu-ops/issues/1727) | [distributed] AttributeError:  | guangyey | functionality bug | P2 | UT issue with few failures | distributed | ut | oneAPI |  | module: distributed, dependency component: oneAPI |
| [1727](https://github.com/intel/torch-xpu-ops/issues/1727) | [distributed] AttributeError:  | guangyey | functionality bug | distributed | ut | oneAPI |  | module: distributed, dependency component: oneAPI |
| [1912](https://github.com/intel/torch-xpu-ops/issues/1912) | Implement the torch.ops.aten._ | liangan1 | feature request | P2 | UT issue with few failures | aten_ops | ut | oneDNN |  | dependency component: oneDNN |
| [1912](https://github.com/intel/torch-xpu-ops/issues/1912) | Implement the torch.ops.aten._ | liangan1 | feature request | aten_ops | ut | oneDNN |  | dependency component: oneDNN |
| [1986](https://github.com/intel/torch-xpu-ops/issues/1986) | torch.xpu._sleep is missing, | guangyey | functionality bug | P2 | UT issue with few failures | aten_ops | ut | oneAPI |  | dependency component: oneAPI |
| [1986](https://github.com/intel/torch-xpu-ops/issues/1986) | torch.xpu._sleep is missing, | guangyey | functionality bug | aten_ops | ut | oneAPI |  | dependency component: oneAPI |
| [2089](https://github.com/intel/torch-xpu-ops/issues/2089) | need an implementation that wo | guangyey | feature request | P2 | UT issue with few failures | aten_ops | ut | driver |  | dependency component: driver |
| [2089](https://github.com/intel/torch-xpu-ops/issues/2089) | need an implementation that wo | guangyey | feature request | aten_ops | ut | driver |  | dependency component: driver |
| [2157](https://github.com/intel/torch-xpu-ops/issues/2157) | BMG d2h copy is very slow comp | jianyizh, mengfei25 | functionality bug | P2 | UT issue with few failures | aten_ops | ut | driver |  | performance, dependency component: driver |
| [2157](https://github.com/intel/torch-xpu-ops/issues/2157) | BMG d2h copy is very slow comp | jianyizh, mengfei25 | functionality bug | aten_ops | ut | driver |  | performance, dependency component: driver |
| [2261](https://github.com/intel/torch-xpu-ops/issues/2261) | [xpu][profiler] Run with fork  | moksiuc | functionality bug | P2 | UT issue with few failures | profiling | ut | oneAPI |  | dependency component: oneAPI, module: profiler |
| [2261](https://github.com/intel/torch-xpu-ops/issues/2261) | [xpu][profiler] Run with fork  | moksiuc | functionality bug | profiling | ut | oneAPI |  | dependency component: oneAPI, module: profiler |
| [2467](https://github.com/intel/torch-xpu-ops/issues/2467) | Host may stuck when submit too | jianyizh | functionality bug | P2 | UT issue with few failures | aten_ops | ut | driver |  | dependency component: driver |
| [2467](https://github.com/intel/torch-xpu-ops/issues/2467) | Host may stuck when submit too | jianyizh | functionality bug | aten_ops | ut | driver |  | dependency component: driver |
| [2655](https://github.com/intel/torch-xpu-ops/issues/2655) | [BMG][OOB] hf_Reformer perform | jianyizh | performance issue | P0 | Regression - passed before but failed now | aten_ops | e2e | Triton |  | E2E, dtype: float16, triaged, performance, os: Ubuntu, hw: BMG, dependency component: Triton, regression |
| [2655](https://github.com/intel/torch-xpu-ops/issues/2655) | [BMG][OOB] hf_Reformer perform | jianyizh | performance issue | aten_ops | e2e | Triton |  | E2E, dtype: float16, triaged, performance, os: Ubuntu, hw: BMG, dependency component: Triton, regression |
| [2769](https://github.com/intel/torch-xpu-ops/issues/2769) | [oneDNN] New failed test cases | LuFinch | functionality bug | P2 | UT issue with few failures | aten_ops | ut | oneDNN |  | hw: PVC, dependency component: oneDNN, module: ut |
| [2769](https://github.com/intel/torch-xpu-ops/issues/2769) | [oneDNN] New failed test cases | LuFinch | functionality bug | aten_ops | ut | oneDNN |  | hw: PVC, dependency component: oneDNN, module: ut |

---

## 4. Others

Issues that don't fall into the categories above.

| ID | Title | Owner | Reporter | Priority | Reason | Labels | PR | Module | Test Module |
|---|-------|-------|----------|---------|--------|--------|-----|--------|-------------|
| [3246](https://github.com/intel/torch-xpu-ops/issues/3246) | AssertionError: Booleans mismatch:  | BartoszKokoszko | Silv3S | P2 | UT issue with few failures | skipped | [PR](https://github.com/intel/torch-xpu-ops/pull/3249) | aten_ops | ut |
| [3243](https://github.com/intel/torch-xpu-ops/issues/3243) | AssertionError: False is not true | pponikox | zxd1997066 | P2 | UT issue with few failures | module: ut, skipped |  | aten_ops | ut |
| [3236](https://github.com/intel/torch-xpu-ops/issues/3236) | AssertionError: 'def [28 chars]n un | jmamzax | jmamzax | P2 | UT issue with few failures | bug_fix_stage5 |  | aten_ops | ut |
| [3227](https://github.com/intel/torch-xpu-ops/issues/3227) | torch xpu event has ~0.1ms latency, | guangyey | jianyizh | P2 | UT issue with few failures |  |  | aten_ops | ut |
| [3224](https://github.com/intel/torch-xpu-ops/issues/3224) | [Win][Build] Building SYCL (Device) | chunhuanMeng | anmyachev | P0 | Build crash - critical blocking issue |  |  | aten_ops | build |
| [3216](https://github.com/intel/torch-xpu-ops/issues/3216) | [OPs] Some ops of XPU have non-dete | CuiYifeng | YangKai0616 | P2 | UT issue with few failures |  |  | aten_ops | ut |
| [3209](https://github.com/intel/torch-xpu-ops/issues/3209) | [Win][Build] There is Cyclic depend | Copilot | NeoZhangJianyu | P0 | Build crash - critical blocking issue |  |  | aten_ops | build |
| [3206](https://github.com/intel/torch-xpu-ops/issues/3206) | [Bug Skip]: [new failures]RuntimeEr | tszulist-hbn | kaileiyx | P2 | UT issue with few failures | skipped |  | aten_ops | ut |
| [3196](https://github.com/intel/torch-xpu-ops/issues/3196) | vitals is not supported, the cases  | libohao1201 | daisyden | P2 | UT issue with few failures | skipped |  | aten_ops | ut |
| [3194](https://github.com/intel/torch-xpu-ops/issues/3194) | Incorrect strides in TestCommonXPU, | AKloniecki | AKloniecki | P2 | UT issue with few failures |  |  | aten_ops | ut |
| [3189](https://github.com/intel/torch-xpu-ops/issues/3189) | Task Tracker | guangyey | guangyey | P2 | UT issue with few failures |  |  | aten_ops | ut |
| [3150](https://github.com/intel/torch-xpu-ops/issues/3150) | [Task] Align XPU kernel's implement | guangyey | guangyey | P2 | UT issue with few failures |  |  | aten_ops | ut |
| [3139](https://github.com/intel/torch-xpu-ops/issues/3139) | [distributed][_composable] Assertio | Kanya-Mo | zxd1997066 | P2 | UT issue with few failures | bug, module: distributed |  | distributed | ut |
| [3103](https://github.com/intel/torch-xpu-ops/issues/3103) | Tensor-likes are not equal for test | BBBela | BBBela | P2 | UT issue with few failures | module: ut, skipped, random |  | aten_ops | ut |
| [3088](https://github.com/intel/torch-xpu-ops/issues/3088) | [TorchAO][BMG] INT4 RTN Flex-attent | Stonepia | LifengWang | P2 | UT issue with few failures | module: ao |  | AO | ut |
| [3074](https://github.com/intel/torch-xpu-ops/issues/3074) | [Bug Skip] test_dlpack_exchange_api | AKloniecki | shangerxin | P2 | UT issue with few failures | skipped |  | aten_ops | ut |
| [3060](https://github.com/intel/torch-xpu-ops/issues/3060) | Implement torch._scaled_grouped_mm  | Stonepia, liangan1 | kgajdamo | P2 | UT issue with few failures | module: quant |  | low_precision | ut |
| [3048](https://github.com/intel/torch-xpu-ops/issues/3048) | Profiler result is not correct on B | aostrowski-hbn | jianyizh | P2 | UT issue with few failures | module: profiler |  | profiling | ut |
| [3033](https://github.com/intel/torch-xpu-ops/issues/3033) | [Bug Skip]: Softmax tolerance | chunhuanMeng | chunhuanMeng | P2 | UT issue with few failures | skipped, random |  | aten_ops | ut |
| [3032](https://github.com/intel/torch-xpu-ops/issues/3032) | [TorchAO][UT] failures in test/prot | Stonepia | zxd1997066 | P0 | Build crash - critical blocking issue | module: ao |  | AO | build |
| [3030](https://github.com/intel/torch-xpu-ops/issues/3030) | [Bug Skip] test/test_modules.py::Te | gplutop7 | shangerxin | P2 | UT issue with few failures | skipped |  | aten_ops | ut |
| [3024](https://github.com/intel/torch-xpu-ops/issues/3024) | Enable clang-tidy checks | Silv3S | Silv3S | P2 | UT issue with few failures | bug_fix_stage5 |  | aten_ops | ut |
| [3022](https://github.com/intel/torch-xpu-ops/issues/3022) | [distributed] batch_isend_irecv Com | zhangxiaoli73 | xiangyuT | P2 | UT issue with few failures | module: distributed |  | distributed | ut |
| [3021](https://github.com/intel/torch-xpu-ops/issues/3021) | [distributed] all_to_all_single Com | zhangxiaoli73 | xiangyuT | P2 | UT issue with few failures | module: distributed |  | distributed | ut |
| [3010](https://github.com/intel/torch-xpu-ops/issues/3010) | [distributed][tensor] test_random_o | jenniew | zxd1997066 | P2 | UT issue with few failures | bug, module: distributed |  | distributed | ut |
| [3000](https://github.com/intel/torch-xpu-ops/issues/3000) | [Bug Skip]: RuntimeError: _share_fd | gplutop7 | zxd1997066 | P2 | UT issue with few failures | skipped |  | aten_ops | ut |
| [2993](https://github.com/intel/torch-xpu-ops/issues/2993) | [Bug Skip]: Unexpected success of t | gplutop7 | CuiYifeng | P2 | UT issue with few failures | skipped |  | aten_ops | ut |
| [2972](https://github.com/intel/torch-xpu-ops/issues/2972) | [distributed] AssertionError: Value | newtdms | zxd1997066 | P2 | UT issue with few failures | bug, module: distributed |  | distributed | ut |
| [2971](https://github.com/intel/torch-xpu-ops/issues/2971) | [distributed] KeyError in test/dist | newtdms, frost-intel | zxd1997066 | P2 | UT issue with few failures | bug, module: distributed |  | distributed | ut |
| [2969](https://github.com/intel/torch-xpu-ops/issues/2969) | [distributed] AssertionError: Scala | frost-intel | zxd1997066 | P2 | UT issue with few failures | bug, module: distributed |  | distributed | ut |
| [2968](https://github.com/intel/torch-xpu-ops/issues/2968) | [distributed] timeout issue in test | frost-intel | zxd1997066 | P2 | UT issue with few failures | bug, module: distributed |  | distributed | ut |
| [2967](https://github.com/intel/torch-xpu-ops/issues/2967) | [distributed] feature gaps in test/ | frost-intel | zxd1997066 | P2 | UT issue with few failures | bug, module: distributed |  | distributed | ut |
| [2950](https://github.com/intel/torch-xpu-ops/issues/2950) | SYCL compilation flag -fsycl-id-que | BBBela | BBBela | P2 | UT issue with few failures |  |  | aten_ops | ut |
| [2942](https://github.com/intel/torch-xpu-ops/issues/2942) | [Windows] Unit tests got Fatal pyth | xuhancn, Stonepia | mengfei25 | P2 | UT issue with few failures | os: Windows |  | aten_ops | ut |
| [2940](https://github.com/intel/torch-xpu-ops/issues/2940) | [release/2.11] Models performance d | jianyizh, LuFinch | mengfei25 | P0 | Impacts real model/application | performance |  | aten_ops | e2e |
| [2939](https://github.com/intel/torch-xpu-ops/issues/2939) | [release/2.11] gmlp_s16_224 inferen | jianyizh | mengfei25 | P2 | E2E benchmark performance issue | performance |  | aten_ops | e2e |
| [2938](https://github.com/intel/torch-xpu-ops/issues/2938) | [release/2.11] basic_gnn_gin and ba | jianyizh | mengfei25 | P2 | E2E benchmark performance issue | performance |  | aten_ops | e2e |
| [2932](https://github.com/intel/torch-xpu-ops/issues/2932) | [release/2.11] jx_nest_base and vol | jianyizh | mengfei25 | P2 | E2E benchmark performance issue |  |  | aten_ops | e2e |
| [2929](https://github.com/intel/torch-xpu-ops/issues/2929) | [release/2.11] volo_d1_224 inferenc | jianyizh | mengfei25 | P1 | E2E benchmark accuracy/functionality issue |  |  | aten_ops | e2e |
| [2920](https://github.com/intel/torch-xpu-ops/issues/2920) | Failing Test Cases in Nightly Wheel | Silv3S | BBBela | P2 | UT issue with few failures | skipped |  | aten_ops | ut |
| [2907](https://github.com/intel/torch-xpu-ops/issues/2907) | [release/2.11] Models performance r | xuhancn | bjarzemb | P0 | Regression - passed before but failed now |  |  | aten_ops | ut |
| [2879](https://github.com/intel/torch-xpu-ops/issues/2879) | RuntimeError: _share_fd_: only avai | Silv3S | Silv3S | P2 | UT issue with few failures | bug_fix_stage5 |  | aten_ops | ut |
| [2871](https://github.com/intel/torch-xpu-ops/issues/2871) | [distributed][fsdp] test_fsdp_overl | songhappy | zxd1997066 | P2 | UT issue with few failures | bug, module: distributed |  | distributed | ut |
| [2823](https://github.com/intel/torch-xpu-ops/issues/2823) | [TorchAO][BMG] Llama-3.2-1B-Instruc | xiaowangintel, lchen2331 | LifengWang | P2 | UT issue with few failures | module: ao |  | AO | ut |
| [2816](https://github.com/intel/torch-xpu-ops/issues/2816) | torch.logcumsumexp incorrectly retu | Silv3S | Silv3S | P2 | UT issue with few failures | Ready for merge, skipped, bug_fix_stage5 |  | aten_ops | ut |
| [2815](https://github.com/intel/torch-xpu-ops/issues/2815) | RuntimeError: output with shape [2] | PawelSwider2000 | Silv3S | P2 | UT issue with few failures | skipped, bug_fix_stage5 |  | aten_ops | ut |
| [2811](https://github.com/intel/torch-xpu-ops/issues/2811) | [Bug Skip]: [Regression] failed cas | jmamzax | kaileiyx | P0 | Regression - passed before but failed now | skipped, bug_fix_stage5 |  | aten_ops | ut |
| [2795](https://github.com/intel/torch-xpu-ops/issues/2795) | Histc raises error with integer inp | CuiYifeng | YangKai0616 | P2 | UT issue with few failures |  |  | aten_ops | ut |
| [2783](https://github.com/intel/torch-xpu-ops/issues/2783) | [Bug Skip]: Key "xpu" is missing fr | daisyden | CuiYifeng | P2 | UT issue with few failures | module: ut, skipped |  | aten_ops | ut |
| [2766](https://github.com/intel/torch-xpu-ops/issues/2766) | MaxPool2d - investigate memory layo | BBBela | pbielak | P2 | UT issue with few failures |  |  | aten_ops | ut |
| [2759](https://github.com/intel/torch-xpu-ops/issues/2759) | [Bug Skip]: New failed cases 2026-1 | AKloniecki | kaileiyx | P2 | UT issue with few failures | skipped |  | aten_ops | ut |
| [2744](https://github.com/intel/torch-xpu-ops/issues/2744) | [Bug Skip]: extended test failures  | pbielak | daisyden | P2 | UT issue with few failures | skipped |  | aten_ops | ut |
| [2742](https://github.com/intel/torch-xpu-ops/issues/2742) | [Linux][PT2E] hf_Roberta_base model | chunhuanMeng | kaileiyx | P0 | Impacts real model/application |  |  | aten_ops | e2e |
| [2738](https://github.com/intel/torch-xpu-ops/issues/2738) | [distributed] test_c10d_spawn_nccl. | jenniew | zxd1997066 | P2 | UT issue with few failures | bug, module: distributed |  | distributed | ut |
| [2734](https://github.com/intel/torch-xpu-ops/issues/2734) | [TorchAO][BMG] INT4 RTN Flex-attent | Stonepia, hoshibara | LifengWang | P2 | UT issue with few failures | module: ao | [PR](https://github.com/pytorch/pytorch/pull/160480, https://github.com/pytorch/pytorch/pull/172316) | AO | ut |
| [2722](https://github.com/intel/torch-xpu-ops/issues/2722) | [Bug Skip]: NotImplementedError: Co | Silv3S | CuiYifeng | P2 | UT issue with few failures | skipped, bug_fix_stage5 |  | aten_ops | ut |
| [2720](https://github.com/intel/torch-xpu-ops/issues/2720) | [upstream_ut] RuntimeError: false I | CuiYifeng | wincent8 | P2 | UT issue with few failures | skipped |  | aten_ops | ut |
| [2707](https://github.com/intel/torch-xpu-ops/issues/2707) | [TorchAO][BMG] INT4 GPTQ failed due | xiaowangintel | LifengWang | P2 | UT issue with few failures | module: ao |  | AO | ut |
| [2702](https://github.com/intel/torch-xpu-ops/issues/2702) | [distributed] RuntimeError: Work ra | syedshahbaaz | madhumitha0102 | P2 | UT issue with few failures | bug, module: distributed |  | distributed | ut |
| [2701](https://github.com/intel/torch-xpu-ops/issues/2701) | [distributed] Barrier Timeout Error | syedshahbaaz | madhumitha0102 | P2 | UT issue with few failures | bug, module: distributed |  | distributed | ut |
| [2700](https://github.com/intel/torch-xpu-ops/issues/2700) | [distributed] Hang issues with test | syedshahbaaz | madhumitha0102 | P2 | UT issue with few failures | bug, module: distributed |  | distributed | ut |
| [2689](https://github.com/intel/torch-xpu-ops/issues/2689) | [LNL][Windows] AssertionError: 'Ass | tadkrawiec | kaileiyx | P2 | UT issue with few failures | os: Windows, module: ut |  | aten_ops | ut |
| [2669](https://github.com/intel/torch-xpu-ops/issues/2669) | [upstream_ut]  AssertionError: Tens | tszulist-hbn | daisyden | P2 | UT issue with few failures | skipped |  | aten_ops | ut |
| [2662](https://github.com/intel/torch-xpu-ops/issues/2662) | [release/2.10][Windows][BMG] New fa | tadkrawiec, kdrozd-dev | mengfei25 | P2 | UT issue with few failures | os: Windows, hw: BMG, module: ut |  | aten_ops | ut |
| [2660](https://github.com/intel/torch-xpu-ops/issues/2660) | [release/2.10][Windows][BMG] New fa | tadkrawiec | mengfei25 | P2 | UT issue with few failures | os: Windows, hw: BMG, module: ut |  | aten_ops | ut |
| [2659](https://github.com/intel/torch-xpu-ops/issues/2659) | [distributed] test_dist2.py Runtime | Chao1Han | zxd1997066 | P2 | UT issue with few failures | module: distributed |  | distributed | ut |
| [2654](https://github.com/intel/torch-xpu-ops/issues/2654) | [BMG][OOB] t5 inference performance | jianyizh | jianyizh | P0 | Regression - passed before but failed now | E2E, dtype: float16, triaged, performance, os: Ubuntu, hw: BMG, regression |  | aten_ops | e2e |
| [2649](https://github.com/intel/torch-xpu-ops/issues/2649) | [distributed][pipelining] test_sche | syedshahbaaz | zxd1997066 | P2 | UT issue with few failures | bug, module: distributed |  | distributed | ut |
| [2645](https://github.com/intel/torch-xpu-ops/issues/2645) | [Bug Skip]: [regression] RuntimeErr | CuiYifeng | kaileiyx | P0 | Regression - passed before but failed now | skipped |  | aten_ops | ut |
| [2640](https://github.com/intel/torch-xpu-ops/issues/2640) | random issue test_vjpvjp_index_redu | wpietka | daisyden | P2 | UT issue with few failures | skipped, random |  | aten_ops | ut |
| [2639](https://github.com/intel/torch-xpu-ops/issues/2639) | test_to() failed during rnn isinsta | Silv3S | daisyden | P2 | UT issue with few failures | skipped | [PR](https://github.com/intel/torch-xpu-ops/pull/3193) | aten_ops | ut |
| [2630](https://github.com/intel/torch-xpu-ops/issues/2630) | Title: [upstream_ut]  AssertionErro | jmamzax | daisyden | P2 | UT issue with few failures | skipped, port_from_skiplist |  | aten_ops | ut |
| [2598](https://github.com/intel/torch-xpu-ops/issues/2598) | [TorchAO][BMG]The first token laten | Stonepia | LifengWang | P2 | UT issue with few failures | module: ao |  | AO | ut |
| [2597](https://github.com/intel/torch-xpu-ops/issues/2597) | [TorchAO][BMG] INT4 GPTQ shows wors | xiaowangintel | LifengWang | P2 | UT issue with few failures | module: ao |  | AO | ut |
| [2580](https://github.com/intel/torch-xpu-ops/issues/2580) | [TorchAO][UT] test/test_low_bit_opt | arlesniak | zxd1997066 | P0 | Build crash - critical blocking issue | module: ao |  | AO | build |
| [2562](https://github.com/intel/torch-xpu-ops/issues/2562) | Warning as Error | chunhuanMeng | EikanWang | P2 | UT issue with few failures |  |  | aten_ops | ut |
| [2560](https://github.com/intel/torch-xpu-ops/issues/2560) | [UT] "RuntimeError: iter.device(arg | CuiYifeng | libohao1201 | P2 | UT issue with few failures | bug |  | aten_ops | ut |
| [2541](https://github.com/intel/torch-xpu-ops/issues/2541) | Title: [upstream_ut]  RuntimeError: | yucai-intel | daisyden | P2 | UT issue with few failures | skipped, port_from_skiplist |  | aten_ops | ut |
| [2538](https://github.com/intel/torch-xpu-ops/issues/2538) | Title: [upstream_ut]  RuntimeError: | Silv3S | daisyden | P2 | UT issue with few failures | skipped, port_from_skiplist |  | aten_ops | ut |
| [2537](https://github.com/intel/torch-xpu-ops/issues/2537) | Title: [upstream_ut]  Failed: Unexp | PatrykWilczewski | daisyden | P2 | UT issue with few failures | skipped, port_from_skiplist |  | aten_ops | ut |
| [2536](https://github.com/intel/torch-xpu-ops/issues/2536) | Title: [upstream_ut]  AttributeErro | daisyden | daisyden | P2 | UT issue with few failures | skipped, port_from_skiplist, not_target |  | aten_ops | ut |
| [2535](https://github.com/intel/torch-xpu-ops/issues/2535) | Title: [upstream_ut]  AttributeErro | Silv3S | daisyden | P2 | UT issue with few failures | skipped, port_from_skiplist |  | aten_ops | ut |
| [2533](https://github.com/intel/torch-xpu-ops/issues/2533) | Title: [upstream_ut]  AttributeErro | astachowiczhabana | daisyden | P2 | UT issue with few failures | skipped, port_from_skiplist |  | aten_ops | ut |
| [2519](https://github.com/intel/torch-xpu-ops/issues/2519) | [upstream_ut]  TypeError: map2_ is  | Silv3S | libohao1201 | P2 | UT issue with few failures | skipped |  | aten_ops | ut |
| [2513](https://github.com/intel/torch-xpu-ops/issues/2513) | [upstream_ut]  RuntimeError: _share | gplutop7 | libohao1201 | P2 | UT issue with few failures | skipped |  | aten_ops | ut |
| [2512](https://github.com/intel/torch-xpu-ops/issues/2512) | [upstream_ut]  RuntimeError: _histc | chunhuanMeng | libohao1201 | P2 | UT issue with few failures | skipped |  | aten_ops | ut |
| [2510](https://github.com/intel/torch-xpu-ops/issues/2510) | [upstream_ut]  RuntimeError: Expect | PawelSwider2000 | libohao1201 | P2 | UT issue with few failures | skipped |  | aten_ops | ut |
| [2494](https://github.com/intel/torch-xpu-ops/issues/2494) | [upstream_ut]  AssertionError: Scal | PawelSwider2000 | libohao1201 | P2 | UT issue with few failures | skipped |  | aten_ops | ut |
| [2491](https://github.com/intel/torch-xpu-ops/issues/2491) | [upstream_ut]  AssertionError: Fals | PatrykWilczewski | libohao1201 | P2 | UT issue with few failures | skipped, bug_fix_stage5 |  | aten_ops | ut |
| [2479](https://github.com/intel/torch-xpu-ops/issues/2479) | [Bug] torch.rand output different r | Stonepia, CuiYifeng | zufangzhu | P2 | UT issue with few failures |  |  | aten_ops | ut |
| [2471](https://github.com/intel/torch-xpu-ops/issues/2471) | test_cuda.py gaps | guangyey | daisyden | P2 | UT issue with few failures |  |  | aten_ops | ut |
| [2469](https://github.com/intel/torch-xpu-ops/issues/2469) | test_matmul_cuda.py gaps | CuiYifeng, guangyey | daisyden | P2 | UT issue with few failures |  |  | aten_ops | ut |
| [2465](https://github.com/intel/torch-xpu-ops/issues/2465) | [windows] ut hang | tadkrawiec | bjarzemb | P2 | UT issue with few failures | os: Windows |  | aten_ops | ut |
| [2463](https://github.com/intel/torch-xpu-ops/issues/2463) | [Bug Skip]: OSError: SYCL runtime i | xuhancn | RUIJIEZHONG66166 | P2 | UT issue with few failures | skipped_windows |  | aten_ops | ut |
| [2434](https://github.com/intel/torch-xpu-ops/issues/2434) | [Bug Skip]: New failures 2025-11-28 | AKloniecki | mengfei25 | P2 | UT issue with few failures | module: ut, skipped, bug_fix_stage4 |  | aten_ops | ut |
| [2425](https://github.com/intel/torch-xpu-ops/issues/2425) | [upstream_ut]  RuntimeError: Expect | BBBela | daisyden | P2 | UT issue with few failures | skipped, bug_fix_stage4 |  | aten_ops | ut |
| [2412](https://github.com/intel/torch-xpu-ops/issues/2412) | Some NestedTensor missing XPU suppo | yucai-intel | daisyden | P2 | UT issue with few failures | module: ut |  | aten_ops | ut |
| [2400](https://github.com/intel/torch-xpu-ops/issues/2400) | [ut_upstream] tf32_on_and_off() nee | chunhuanMeng | daisyden | P2 | UT issue with few failures |  |  | aten_ops | ut |
| [2392](https://github.com/intel/torch-xpu-ops/issues/2392) | [Bug Skip]: torch.OutOfMemoryError: | xuhancn | RUIJIEZHONG66166 | P2 | UT issue with few failures | skipped_windows |  | aten_ops | ut |
| [2389](https://github.com/intel/torch-xpu-ops/issues/2389) | [Bug Skip]: RuntimeError: Data corr | PatrykWilczewski | kaileiyx | P2 | UT issue with few failures | skipped, bug_fix_stage4, random |  | aten_ops | ut |
| [2349](https://github.com/intel/torch-xpu-ops/issues/2349) | [oneAPI][backward compatibility] li | riverliuintel | dvrogozh | P2 | UT issue with few failures |  |  | aten_ops | ut |
| [2340](https://github.com/intel/torch-xpu-ops/issues/2340) | [distributed][_tools] AssertionErro | githubsgi | zxd1997066 | P2 | UT issue with few failures | bug, duplicate, module: distributed |  | distributed | ut |
| [2326](https://github.com/intel/torch-xpu-ops/issues/2326) | [TorchAO] MX training  native PyTor | riverliuintel | liangan1 | P2 | UT issue with few failures | module: ao |  | AO | ut |
| [2325](https://github.com/intel/torch-xpu-ops/issues/2325) | [TorchAO] Float8 training support o | arlesniak, riverliuintel | liangan1 | P2 | UT issue with few failures | module: ao |  | AO | ut |
| [2324](https://github.com/intel/torch-xpu-ops/issues/2324) | [TorchAO] FP8 conv support | Stonepia | liangan1 | P2 | UT issue with few failures | module: ao |  | AO | ut |
| [2323](https://github.com/intel/torch-xpu-ops/issues/2323) | [TorchAO] MOE training enabling on  | riverliuintel | liangan1 | P2 | UT issue with few failures | module: ao |  | AO | ut |
| [2263](https://github.com/intel/torch-xpu-ops/issues/2263) | [xpu][bug] XPU Trace event ends too | PawelSwider2000 | chuanqi129 | P2 | UT issue with few failures | module: profiler |  | profiling | ut |
| [2250](https://github.com/intel/torch-xpu-ops/issues/2250) | Found mismatch when comparing the o | astachowiczhabana | daisyden | P2 | UT issue with few failures | skipped, bug_fix_stage3 |  | aten_ops | ut |
| [2240](https://github.com/intel/torch-xpu-ops/issues/2240) | RuntimeError: Trying to set a forwa | gplutop7 | zxd1997066 | P2 | UT issue with few failures | skipped, bug_fix_stage3 |  | aten_ops | ut |
| [2239](https://github.com/intel/torch-xpu-ops/issues/2239) | Exception: could not create a primi | wpietka | zxd1997066 | P2 | UT issue with few failures | skipped, bug_fix_stage5 |  | aten_ops | ut |
| [2217](https://github.com/intel/torch-xpu-ops/issues/2217) | AO Performance issue track | Stonepia | liangan1 | P2 | UT issue with few failures | module: ao |  | AO | ut |
| [2215](https://github.com/intel/torch-xpu-ops/issues/2215) | Find use case example for torch-xpu | dvrogozh | dvrogozh | P2 | UT issue with few failures |  |  | aten_ops | ut |
| [2207](https://github.com/intel/torch-xpu-ops/issues/2207) | Enable FP8/MXFP8 Ops with requests  | CuiYifeng | CuiYifeng | P2 | UT issue with few failures | dtype: float8 |  | aten_ops | ut |
| [2202](https://github.com/intel/torch-xpu-ops/issues/2202) | [TorchAO][BMG] The RTN performance  | Stonepia | MingxuZh | P0 | Regression - passed before but failed now | performance, regression, module: ao |  | AO | ut |
| [2201](https://github.com/intel/torch-xpu-ops/issues/2201) | [TorchAO][BMG] When using paged att | Stonepia | MingxuZh | P2 | UT issue with few failures | module: ao |  | AO | ut |
| [2195](https://github.com/intel/torch-xpu-ops/issues/2195) | Tools in pti should get 100% functi | aostrowski-hbn | jianyizh | P2 | UT issue with few failures | module: profiler |  | profiling | ut |
| [2182](https://github.com/intel/torch-xpu-ops/issues/2182) | test_transform_bias_rescale_qkv_nes | PawelSwider2000 | wincent8 | P2 | UT issue with few failures | Accuracy, module: ut, skipped |  | aten_ops | ut |
| [2165](https://github.com/intel/torch-xpu-ops/issues/2165) | [distributed] test_device_mesh.py:: | jemitche1 | zxd1997066 | P2 | UT issue with few failures | bug, module: distributed |  | distributed | ut |
| [2163](https://github.com/intel/torch-xpu-ops/issues/2163) | 3 distributed UT cases need to be s | githubsgi | libohao1201 | P2 | UT issue with few failures | module: distributed |  | distributed | ut |
| [2142](https://github.com/intel/torch-xpu-ops/issues/2142) | XPU max_memory_allocated have diffe | guangyey | jiqing-feng | P2 | UT issue with few failures | bug |  | aten_ops | ut |
| [2140](https://github.com/intel/torch-xpu-ops/issues/2140) | Consider how to avoid copy in FFT k | CuiYifeng | CuiYifeng | P2 | UT issue with few failures | enhancement |  | aten_ops | ut |
| [2127](https://github.com/intel/torch-xpu-ops/issues/2127) | Path Coverage enhancement | CuiYifeng | CuiYifeng | P2 | UT issue with few failures | enhancement |  | aten_ops | ut |
| [2113](https://github.com/intel/torch-xpu-ops/issues/2113) | Update example for Distributed Data | songhappy | luoyu-intel | P2 | UT issue with few failures | module: distributed |  | distributed | ut |
| [2098](https://github.com/intel/torch-xpu-ops/issues/2098) | Upstream XPU functions in yaml | guangyey | EikanWang | P2 | UT issue with few failures | enhancement |  | aten_ops | ut |
| [2086](https://github.com/intel/torch-xpu-ops/issues/2086) | nd_item::barrier has been deprecate | dvrogozh | EikanWang | P2 | UT issue with few failures | enhancement |  | aten_ops | ut |
| [2063](https://github.com/intel/torch-xpu-ops/issues/2063) | Avoid using out-of-date term | CuiYifeng | EikanWang | P2 | UT issue with few failures | enhancement |  | aten_ops | ut |
| [2058](https://github.com/intel/torch-xpu-ops/issues/2058) | [release/2.9] llama_v2_7b_16h amp i | jianyizh | mengfei25 | P0 | Build crash - critical blocking issue | performance, regression, dependency component: community |  | aten_ops | e2e |
| [2055](https://github.com/intel/torch-xpu-ops/issues/2055) | New huggingface LLM models issues | jianyizh, mengfei25 | mengfei25 | P0 | Impacts real model/application | E2E, hw: PVC |  | aten_ops | e2e |
| [2004](https://github.com/intel/torch-xpu-ops/issues/2004) | [distributed][shared_tensor] test\d | libohao1201 | libohao1201 | P2 | UT issue with few failures | bug, module: distributed |  | distributed | ut |
| [1996](https://github.com/intel/torch-xpu-ops/issues/1996) | [TorchAO]  Memory Efficient Optimiz | arlesniak | liangan1 | P2 | UT issue with few failures | module: ao |  | AO | ut |
| [1969](https://github.com/intel/torch-xpu-ops/issues/1969) | torch._dynamo.exc.InternalTorchDyna | guangyey | shangerxin | P2 | UT issue with few failures | module: ut |  | aten_ops | ut |
| [1936](https://github.com/intel/torch-xpu-ops/issues/1936) | implement torch.linalg.cholesky xpu | mwiktor-intel | jiqing-feng | P2 | UT issue with few failures | module: op impl, bug_fix_stage5 |  | aten_ops | ut |
| [1902](https://github.com/intel/torch-xpu-ops/issues/1902) | implement torch.linalg.pinv xpu bac | mwiktor-intel | yao-matrix | P2 | UT issue with few failures | module: op impl, bug_fix_stage5 |  | aten_ops | ut |
| [1901](https://github.com/intel/torch-xpu-ops/issues/1901) | implement torch.linalg.svd xpu back | CuiYifeng | yao-matrix | P2 | UT issue with few failures | module: op impl |  | aten_ops | ut |
| [1900](https://github.com/intel/torch-xpu-ops/issues/1900) | implement torch.linalg.qr xpu backe | pbielak | yao-matrix | P2 | UT issue with few failures | module: op impl, bug_fix_stage3 |  | aten_ops | ut |
| [1894](https://github.com/intel/torch-xpu-ops/issues/1894) | [Linux][PT2E] performance test got  | jenniew | kaileiyx | P1 | E2E benchmark accuracy/functionality issue | module: quant |  | low_precision | e2e |
| [1856](https://github.com/intel/torch-xpu-ops/issues/1856) | channel last aten::hardswish_ will  | chunhuanMeng | jianyizh | P2 | UT issue with few failures | performance, hw: BMG |  | aten_ops | ut |
| [1784](https://github.com/intel/torch-xpu-ops/issues/1784) | [Performance] Torch XPU Profiler is | jfedorov | liangan1 | P2 | UT issue with few failures | module: profiler |  | profiling | ut |
| [1764](https://github.com/intel/torch-xpu-ops/issues/1764) | New kernels for concat | yucai-intel | jianyizh | P2 | UT issue with few failures | performance, kernel_optimization, hw: BMG, module: op impl, benchmark |  | aten_ops | ut |
| [1745](https://github.com/intel/torch-xpu-ops/issues/1745) | torch.xpu.empty_cache() introduces  | guangyey | songhappy | P2 | UT issue with few failures | module: core |  | aten_ops | ut |
| [1729](https://github.com/intel/torch-xpu-ops/issues/1729) | Validation Check List | chuanqi129 | EikanWang | P2 | UT issue with few failures | module: infra |  | aten_ops | ut |
| [1645](https://github.com/intel/torch-xpu-ops/issues/1645) | [For Comparison] Save reference com | mengfei25 | mengfei25 | P2 | UT issue with few failures | module: infra |  | inductor | ut |
| [1594](https://github.com/intel/torch-xpu-ops/issues/1594) | Keep track on the building warning | CuiYifeng, chunhuanMeng | toyxu | P0 | Build crash - critical blocking issue | module: build |  | aten_ops | ut |
| [1587](https://github.com/intel/torch-xpu-ops/issues/1587) | Keep track on the latest CUDA op im | CuiYifeng, yucai-intel | toyxu | P2 | UT issue with few failures | kernel_optimization |  | aten_ops | ut |
| [1574](https://github.com/intel/torch-xpu-ops/issues/1574) | The operator 'aten::_grouped_mm' is | Stonepia | githubsgi | P2 | UT issue with few failures | module: ao |  | AO | ut |
| [1571](https://github.com/intel/torch-xpu-ops/issues/1571) | [distributed] ValueError: Cannot us | zhangxiaoli73 | daisyden | P2 | UT issue with few failures | module: distributed |  | distributed | ut |
| [1159](https://github.com/intel/torch-xpu-ops/issues/1159) | [LNL Windows][Test by CD Nightly Wh | Stonepia | libohao1201 | P0 | Impacts real model/application | E2E, client, module: dependency bug, dependency: third_party packages |  | aten_ops | e2e |
| [492](https://github.com/intel/torch-xpu-ops/issues/492) | Timm_efficientdet NotImplementedErr | weishi-deng | mengfei25 | P0 | Impacts real model/application | E2E, Accuracy, module: torchbench, dtype: amp_bf16, dtype: amp_fp16, training, dtype: float16, dtype: float32, dtype: bfloat16, triaged |  | aten_ops | e2e |
| [489](https://github.com/intel/torch-xpu-ops/issues/489) | Moco NotImplementedError: xpu not s | weishi-deng | mengfei25 | P2 | UT issue with few failures | E2E, Accuracy, module: torchbench, dtype: amp_bf16, dtype: amp_fp16, training, dtype: float16, dtype: float32, dtype: bfloat16 |  | aten_ops | e2e |
| [208](https://github.com/intel/torch-xpu-ops/issues/208) | Abstract utility functions used in  | CuiYifeng | fengyuan14 | P2 | UT issue with few failures | enhancement, module: op impl, long term |  | aten_ops | ut |
| [146](https://github.com/intel/torch-xpu-ops/issues/146) | Evaluate register spill in SYCL ker | CuiYifeng, jianyizh, mengfei25 | fengyuan14 | P2 | UT issue with few failures | enhancement |  | aten_ops | ut |

---

## 5. Recent Issues (Last 10 Days)

Issues created in the last 10 days (as of 2026-04-07).

| ID | Title | Status | Owner | Priority | Reason | Labels | Module | Test Module |
|---|-------|--------|-------|---------|--------|--------|--------|-------------|
| [3259](https://github.com/intel/torch-xpu-ops/issues/3259) | New failed test cases 2026-04-02 | open | SlawomirLaba | P2 | UT issue with few failures | skipped | aten_ops | ut |
| [3258](https://github.com/intel/torch-xpu-ops/issues/3258) | Error in op: torch.ops.aten._scaled_dot_produ | open | None | P2 | UT issue with few failures |  | aten_ops | ut |
| [3257](https://github.com/intel/torch-xpu-ops/issues/3257) | [Linux][E2E][Regression] Huggingface test mod | open | None | P0 | Impacts real model/application |  | aten_ops | e2e |
| [3255](https://github.com/intel/torch-xpu-ops/issues/3255) | [Linux][PT2E][Regression] Some performance te | open | None | P0 | Regression - passed before but failed now |  | aten_ops | e2e |
| [3247](https://github.com/intel/torch-xpu-ops/issues/3247) | NotImplementedError: "dot_xpu_mkl" not implem | open | Silv3S | P2 | UT issue with few failures | ut_upstream | aten_ops | ut |
| [3246](https://github.com/intel/torch-xpu-ops/issues/3246) | AssertionError: Booleans mismatch: True is no | open | BartoszKokoszko | P2 | UT issue with few failures | skipped | aten_ops | ut |
| [3243](https://github.com/intel/torch-xpu-ops/issues/3243) | AssertionError: False is not true | open | pponikox | P2 | UT issue with few failures | module: ut, skipped | aten_ops | ut |
| [3242](https://github.com/intel/torch-xpu-ops/issues/3242) | AssertionError: Torch not compiled with CUDA  | open | None | P2 | UT issue with few failures | module: ut, skipped | aten_ops | ut |
| [3238](https://github.com/intel/torch-xpu-ops/issues/3238) | The supported dtypes of _refs.stft is not ali | open | CuiYifeng | P2 | UT issue with few failures | ut_upstream | aten_ops | ut |
| [3236](https://github.com/intel/torch-xpu-ops/issues/3236) | AssertionError: 'def [28 chars]n unbind = tor | open | jmamzax | P2 | UT issue with few failures | bug_fix_stage5 | aten_ops | ut |
| [3233](https://github.com/intel/torch-xpu-ops/issues/3233) | [distributed] RuntimeError: No backend for th | open | None | P2 | UT issue with few failures | bug, module: distributed | distributed | ut |
| [3232](https://github.com/intel/torch-xpu-ops/issues/3232) | [distributed][tensor] AssertionError: Asserti | open | None | P2 | UT issue with few failures | bug, module: distributed | distributed | ut |
| [3231](https://github.com/intel/torch-xpu-ops/issues/3231) | Dynamo failed to run FX node with fake tensor | open | None | P2 | UT issue with few failures | module: ut, skipped | aten_ops | ut |
| [3229](https://github.com/intel/torch-xpu-ops/issues/3229) | RuntimeError: No viable backend for scaled_do | open | None | P2 | UT issue with few failures | skipped, ut_upstream | aten_ops | ut |
| [3227](https://github.com/intel/torch-xpu-ops/issues/3227) | torch xpu event has ~0.1ms latency, which is  | open | guangyey | P2 | UT issue with few failures |  | aten_ops | ut |
| [3224](https://github.com/intel/torch-xpu-ops/issues/3224) | [Win][Build] Building SYCL (Device) object to | open | chunhuanMeng | P0 | Build crash - critical blocking issue |  | aten_ops | build |
| [3216](https://github.com/intel/torch-xpu-ops/issues/3216) | [OPs] Some ops of XPU have non-determinism an | open | CuiYifeng | P2 | UT issue with few failures |  | aten_ops | ut |
| [3209](https://github.com/intel/torch-xpu-ops/issues/3209) | [Win][Build] There is Cyclic dependencies err | open | Copilot | P0 | Build crash - critical blocking issue |  | aten_ops | build |
| [3206](https://github.com/intel/torch-xpu-ops/issues/3206) | [Bug Skip]: [new failures]RuntimeError: Expec | open | tszulist-hbn | P2 | UT issue with few failures | skipped | aten_ops | ut |
