# XPU Ops Bug Scrub Report

- **Repository**: `intel/torch-xpu-ops`
- **Generated**: 2026-04-21 (cutoff for Section 7: 2026-04-14)
- **Total issues in workbook**: 375
- **Classified (non-empty `action_Type`)**: 337
- **Empty `action_TBD` (no verdict)**: 38

## 1. Summary

This report groups the 375 tracked torch-xpu-ops issues into action buckets derived from the `action_Type` classification column of the triage workbook. Each issue appears in at most one Action-Required or QA section, chosen by its highest-priority category. Cross-cutting slices (duplicated issues, external dependency blockers, newly filed issues) are listed separately for visibility.

**Headline counts (primary category):**

| Bucket | Categories | Issues |
|---|---|---:|
| Developer action required | NEED_ACTION, NEEDS_OWNER, TRACK_PR, IMPLEMENT, RETRIAGE_PRS, ROOT_CAUSE | 223 |
| QA action required | CLOSE, VERIFY_AND_CLOSE, AWAIT_REPLY, SKIP, MONITOR, NOT_TARGET_CLOSE, CHECK_CASES | 103 |
| Duplicated | — | 13 |
| External dependency (non-upstream-pytorch, non-SYCL-kernel) | — | 128 |
| Filed within last 7 days | — | 27 |

<a id="sec-2"></a>
## 2. Index

- [3. Action required (Developer)](#sec-3)
  - [3.0 UNCLASSIFIED](#sec-3-0-unclassified)
  - [3.1 NEED_ACTION](#sec-3-1-need-action)
  - [3.2 NEEDS_OWNER](#sec-3-2-needs-owner)
  - [3.3 TRACK_PR](#sec-3-3-track-pr)
  - [3.4 IMPLEMENT](#sec-3-4-implement)
  - [3.5 RETRIAGE_PRS](#sec-3-5-retriage-prs)
  - [3.6 ROOT_CAUSE](#sec-3-6-root-cause)
- [4. QA](#sec-4)
  - [4.1 CLOSE](#sec-4-1-close)
  - [4.2 VERIFY_AND_CLOSE](#sec-4-2-verify-and-close)
  - [4.3 AWAIT_REPLY](#sec-4-3-await-reply)
  - [4.4 SKIP](#sec-4-4-skip)
  - [4.5 MONITOR](#sec-4-5-monitor)
  - [4.6 NOT_TARGET_CLOSE](#sec-4-6-not-target-close)
  - [4.7 CHECK_CASES](#sec-4-7-check-cases)
- [5. Duplicated issues](#sec-5)
- [6. Dependency (external blockers)](#sec-6)
- [7. New submitted issues (<7 days)](#sec-7)
- [8. Statistics](#sec-8)

<a id="sec-3"></a>
## 3. Action required (Developer)

Issues in this section require developer work before they can progress. Each subsection is split by `Category` (existing taxonomy column); rows inside each category table are sorted by `Priority` (P0 → P3).

<a id="sec-3-0-unclassified"></a>
### 3.0 UNCLASSIFIED  ·  38 issues

**UNCLASSIFIED — Phase 4b produced no verdict; needs manual triage**

| Issue | Priority | Title | Owner | Fix Approach | Reporter | Labels |
|---|---|---|---|---|---|---|
| [#2207](https://github.com/intel/torch-xpu-ops/issues/2207) | P1 | Enable FP8/MXFP8 Ops with requests and CUDA align… | Stonepia, CuiYifeng, LuF… | Continue landing the referenced PRs (#2145 arithmetic, #2152 compare/cat/where, #2154 compare, #2190 flip/index, #2258 copy) and add AT_DIS… | CuiYifeng | dtype: float8 |
| [#1587](https://github.com/intel/torch-xpu-ops/issues/1587) | P2 | Keep track on the latest CUDA op impl | CuiYifeng, yucai-intel | For each unchecked CUDA PR, replicate the algorithmic change in the corresponding torch-xpu-ops kernel (Loops.h, Copy.cpp, CatKernel.cpp, e… | toyxu | kernel_optimization |
| [#2128](https://github.com/intel/torch-xpu-ops/issues/2128) | P2 | [2.9][BMG-Windows][Torchbench] speeach_transforer… |  | Reduce to a minimal AOTI/Inductor repro by capturing the offending partitioned subgraph via TORCH_COMPILE_DEBUG=1 and TORCHINDUCTOR_CACHE_D… | libohao1201 | os: Windows |
| [#2199](https://github.com/intel/torch-xpu-ops/issues/2199) | P2 | Fix reduction and norm register spill |  | (1) Reduce vt0 to 2 for wide types (long/int64) and composite types (pair<index,value> in ArgMin/ArgMax) where spill is highest; (2) split … | jianyizh | enhancement |
| [#2217](https://github.com/intel/torch-xpu-ops/issues/2217) | P2 | AO Performance issue track | Stonepia | Drive the listed oneDNN tickets to closure: request oneDNN team to improve BF16 matmul perf on BMG to match FP16/PTL, and merge the GEMM re… | liangan1 | module: ao |
| [#2327](https://github.com/intel/torch-xpu-ops/issues/2327) | P2 | [TorchAO] benchmark enabling on XPU | LifengWang, xiaowangintel | Port the pytorch/ao/benchmarks driver scripts to accept `device=xpu`, add an XPU device path in each benchmark (e.g. quant/autoquant/gemm m… | liangan1 |  |
| [#2340](https://github.com/intel/torch-xpu-ops/issues/2340) | P2 | [distributed][_tools] AssertionError: Roofline es… | githubsgi | Close as duplicate of #2163 and track fix there. The real fix is to replace the hard torch.cuda.is_available() gate in torch/distributed/_t… | zxd1997066 | bug, duplicate, module: distributed |
| [#2390](https://github.com/intel/torch-xpu-ops/issues/2390) | P2 | SDPA in pytorch use different backend compared wi… | LuFinch | Track oneDNN MFDNN-14834; once v3.11 lands with fused training SDPA, wire it into the XPU SDPA dispatch (torch-xpu-ops `src/ATen/native/xpu… | jiqing-feng |  |
| [#2400](https://github.com/intel/torch-xpu-ops/issues/2400) | P2 | [ut_upstream] tf32_on_and_off() need xpu support | chunhuanMeng | Add an XPU counterpart in torch/testing/_internal/common_cuda.py (or a new common_xpu.py) that, when device==xpu, toggles `torch.backends.m… | daisyden |  |
| [#2404](https://github.com/intel/torch-xpu-ops/issues/2404) | P2 | [distributed][checkpoint] AssertionError: Boolean… |  | Generalize the stager to be device-agnostic: replace the torch.cuda.is_available() guard in _state_dict_stager.py:36 with `torch.accelerato… | zxd1997066 | bug |
| [#2463](https://github.com/intel/torch-xpu-ops/issues/2463) | P2 | [Bug Skip]: OSError: SYCL runtime is not dected. | xuhancn | In torch/utils/cpp_extension.py: extend _find_sycl_home() to also detect the pip-installed intel-sycl-rt layout (site-packages headers/libs… | RUIJIEZHONG66166 | skipped_windows |
| [#2465](https://github.com/intel/torch-xpu-ops/issues/2465) | P2 | [windows] ut hang | tadkrawiec, mganczarenko | First narrow the hang: instrument the runners to print the current test name before each call (PYTEST_CURRENT_TEST / printing in run_test_w… | bjarzemb | os: Windows |
| [#2482](https://github.com/intel/torch-xpu-ops/issues/2482) | P2 | test_dtypes issue introduced by pytorch test samp… | daisyden | Align the XPU OpInfo dtype overrides for nn.functional.conv_transpose{1,2,3}d: update the xpu dtype override list in third_party/torch-xpu-… | daisyden | skipped |
| [#2554](https://github.com/intel/torch-xpu-ops/issues/2554) | P2 | [upstream_ut] AssertionError: AssertionError not … | daisyden | No change needed in torch-xpu-ops; wait for intel-xpu-backend-for-triton fix (#5654) to land and bump the pinned Triton commit in PyTorch (… | daisyden | module: inductor, skipped |
| [#2562](https://github.com/intel/torch-xpu-ops/issues/2562) | P2 | Warning as Error | chunhuanMeng | In third_party/torch-xpu-ops/cmake/BuildFlags.cmake add `-Werror` (or at minimum `-Werror=return-type -Werror=unused-result -Werror=reorder… | EikanWang |  |
| [#2597](https://github.com/intel/torch-xpu-ops/issues/2597) | P2 | [TorchAO][BMG] INT4 GPTQ shows worse performance … | xiaowangintel | 1) Compare the lowered FX graph for RTN vs GPTQ quantized models to identify why GPTQ emits the extra `24x1x128` batched matmuls (likely a … | LifengWang | module: ao |
| [#2598](https://github.com/intel/torch-xpu-ops/issues/2598) | P2 | [TorchAO][BMG]The first token latency of Qwen2.5-… | Stonepia | Profile first-token with `SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=2` vs default and with onednn-verbose=1 to confirm primitive cache … | LifengWang | module: ao |
| [#2611](https://github.com/intel/torch-xpu-ops/issues/2611) | P2 | [upstream_ut] AssertionError: Tensor-likes are no… | daisyden | Close as duplicate of #2613 (or consolidate skip list into a single issue). Apply the same fix: update argmax/argmin reduce combine ops to … | daisyden | dependency component: driver, module: i… |
| [#2613](https://github.com/intel/torch-xpu-ops/issues/2613) | P2 | [upstream_ut] AssertionError: Tensor-likes are no… | daisyden | Fix the argmax/argmin reduction combine functor in ReduceArgMaxKernel.cpp / ReduceArgMinKernel.cpp to: (1) on equal values pick the pair wi… | daisyden | dependency component: driver, module: i… |
| [#2654](https://github.com/intel/torch-xpu-ops/issues/2654) | P2 | [BMG][OOB] t5 inference performance drop 2 | RUIJIEZHONG66166 | Tune the XPU branch of persistent_reduction heuristics in torch/_inductor/runtime/triton_heuristics.py so that for reduction kernels where … | jianyizh | E2E, dtype: float16, triaged, performan… |
| [#2655](https://github.com/intel/torch-xpu-ops/issues/2655) | P2 | [BMG][OOB] hf_Reformer performance drop | jianyizh | Track the IGC fix (IGC-14276) and bump the minimum IGC / compute-runtime requirement in torch-xpu-ops CI once resolved. In the meantime, ad… | jianyizh | E2E, dtype: float16, triaged, performan… |
| [#2656](https://github.com/intel/torch-xpu-ops/issues/2656) | P2 | [release/2.10] models got fail_accuracy on BMG WS… |  | Audit BatchNormKernels.cpp and GroupNormKernels.cpp to ensure: (1) welford `mean`/`m2` and `running_mean`/`running_var` accumulators use `a… | libohao1201 | os: Windows |
| [#2660](https://github.com/intel/torch-xpu-ops/issues/2660) | P2 | [release/2.10][Windows][BMG] New failed test cases | pfierek, tadkrawiec, ery… | (a) Fix the runner: install MSVC and expose cl.exe so Triton/inductor compile works, or skip those tests on Windows. (b) File a Conv2d expa… | mengfei25 | os: Windows, hw: BMG, module: ut |
| [#2689](https://github.com/intel/torch-xpu-ops/issues/2689) | P2 | [LNL][Windows] AssertionError: 'Assertion `cur_ta… | draghan, tadkrawiec | Short term: add a Windows/LNL skip for test_cross_entropy_loss_2d_out_of_bounds_class_index in torch-xpu-ops skip list (e.g. skip_list_win_… | kaileiyx | os: Windows, module: ut |
| [#2715](https://github.com/intel/torch-xpu-ops/issues/2715) | P2 | [upstream_ut] torch._dynamo.exc.Unsupported: Atte… | CuiYifeng | In torch/_dynamo/trace_rules.py, remove torch.xpu from MOD_SKIPLIST (or add torch.xpu.device to the allowed-callable list analogous to torc… | daisyden | skipped, ut_upstream |
| [#2795](https://github.com/intel/torch-xpu-ops/issues/2795) | P2 | Histc raises error with integer input when determ… | CuiYifeng | Mirror the CUDA gating in _histc_xpu: only call alertNotDeterministic when isFloatingType(self.scalar_type()) (or when floating-point weigh… | YangKai0616 |  |
| [#2800](https://github.com/intel/torch-xpu-ops/issues/2800) | P2 | AttributeError: 'torch._C._XpuDeviceProperties' o… | guangyey | Short-term: add a test-side guard that only queries '.major' on CUDA (e.g. skip/branch when device.type=='xpu' in test_scaled_matmul_cuda.p… | daisyden | dependency component: oneAPI, module: i… |
| [#2801](https://github.com/intel/torch-xpu-ops/issues/2801) | P2 | to_dense() for Sparse CSR backend cannot broadcas… | jenniew | In sparse_compressed_to_dense (TensorConversions.cpp:699-803), detect when indices batch dims are missing relative to tensor batch dims and… | jenniew |  |
| [#1594](https://github.com/intel/torch-xpu-ops/issues/1594) | P3 | Keep track on the building warning | CuiYifeng, chunhuanMeng | Continue to close remaining unchecked warnings as they appear in the nightly build log; migrate deprecated host-allocator APIs to at::getHo… | toyxu | module: build |
| [#2063](https://github.com/intel/torch-xpu-ops/issues/2063) | P3 | Avoid using out-of-date term | CuiYifeng | Enumerate terms (Tile, SIMD, possibly EU, subslice) and mechanically rename to SYCL-conformant equivalents (work-group / sub-group / comput… | EikanWang | enhancement |
| [#208](https://github.com/intel/torch-xpu-ops/issues/208) | P3 | Abstract utility functions used in ATen operator … | CuiYifeng | Audit torch-xpu-ops/src/ATen/native/xpu for duplicated utilities (e.g., sum_to, permute_dims helpers, inferExpandGeometry variants), file u… | fengyuan14 | enhancement, module: op impl, long term |
| [#2140](https://github.com/intel/torch-xpu-ops/issues/2140) | P3 | Consider how to avoid copy in FFT kernels | CuiYifeng | Refactor _fft_{c2c,c2r,r2c}_mkl and their callee _exec_fft / _mkl_dft to accept an optional output Tensor& and, when the caller's 'out' is … | CuiYifeng | enhancement |
| [#2196](https://github.com/intel/torch-xpu-ops/issues/2196) | P3 | Fix DistributionElementwiseKernelFunctor register… |  | Minor tuning only: either (a) drop the unroll factor from 2 to 1 for Normal (and other 2-output distributions) to halve per-thread state, o… | jianyizh | enhancement |
| [#2766](https://github.com/intel/torch-xpu-ops/issues/2766) | P3 | MaxPool2d - investigate memory layout performance | BBBela | Keep the layout-aware dispatch introduced by PR #2763 as the default (no implicit Contiguous->ChannelsLast conversion). Optionally add a si… | pbielak |  |
| [#3150](https://github.com/intel/torch-xpu-ops/issues/3150) | P3 | [Task] Align XPU kernel's implementation to stock… | guangyey | Port-forward each listed kernel (LayerNorm first, then GroupNorm, etc.) by diffing the XPU .cpp against the current aten/src/ATen/native/cu… | guangyey | enhancement |
| [#3189](https://github.com/intel/torch-xpu-ops/issues/3189) | P3 | Task Tracker | guangyey | Keep open as tracker; once each listed PyTorch PR merges, land the corresponding XPU-side mirror in torch-xpu-ops (e.g. XPUCachingAllocator… | guangyey |  |
| [#3266](https://github.com/intel/torch-xpu-ops/issues/3266) | P3 | [RFC] Migrate XPU kernel math functions from std:… |  | Execute the 34-item task list from the RFC as individual PRs. Start with infrastructure: (0) update XPUMathCompat.h with sycl:: equivalents… | jianyizh |  |
| [#3358](https://github.com/intel/torch-xpu-ops/issues/3358) | P3 | [v.2.12.0] Release Tracker |  | No engineering fix required; keep open through release Phase 1 (until 2026-04-27) and Phase 2 RC validation, appending cherry-pick links as… | chuanqi129 |  |


<a id="sec-3-1-need-action"></a>
### 3.1 NEED_ACTION  ·  63 issues

**NEED_ACTION — no PR and no decision; owner must start investigation**

<a id="sec-3-1-1-distributed"></a>
#### 3.1.1 Distributed  ·  7 issues

| Issue | Priority | Title | Owner | Fix Approach | Reporter | Labels |
|---|---|---|---|---|---|---|
| [#2737](https://github.com/intel/torch-xpu-ops/issues/2737) | P1 | [distributed] AttributeError: module 'torch._C' h… |  | Make the comm bindings device-agnostic: move _gather/_scatter/_broadcast(_coalesced)/_gather_out definitions from torch/csrc/cuda/python_co… | zxd1997066 | bug, module: distributed |
| [#3082](https://github.com/intel/torch-xpu-ops/issues/3082) | P1 | multithread support in distributed |  | Add 'xpu' to the devices tuple at multi_threaded_pg.py:548 (register_backend('threaded', _create_threaded_pg, devices=['cpu','cuda','xpu'])… | daisyden | module: distributed, module: ut |
| [#3233](https://github.com/intel/torch-xpu-ops/issues/3233) | P1 | [distributed] RuntimeError: No backend for the pa… | songhappy | Implement comm-splitting in ProcessGroupXCCL: add a splitGroup()/merged config override, expose supports_splitting=true, and forward comm_s… | zxd1997066 | bug, module: distributed |
| [#3270](https://github.com/intel/torch-xpu-ops/issues/3270) | P1 | [distributed][tensor] RuntimeError: Invalid scali… | syedshahbaaz | Extend scaled_mm_single_dim_strategy in torch/distributed/tensor/_ops/_matrix_ops.py to handle 2-D blockwise scales (BlockWise1x128 / Block… | zxd1997066 | bug, module: distributed |
| [#3100](https://github.com/intel/torch-xpu-ops/issues/3100) | P2 | [distributed] /handler/dump_nccl_trace_pickle and… | songhappy | (1) In FlightRecorderXCCL.cpp register a 'dump_nccl_trace_pickle' (or an alias) handler via ::c10d::control_plane::registerHandler, paralle… | zxd1997066 | module: distributed |
| [#3101](https://github.com/intel/torch-xpu-ops/issues/3101) | P2 | [distributed] 'torch._C._distributed_c10d.Process… | jenniew | Add a .def('_set_default_timeout', &::c10d::ProcessGroupXCCL::setTimeout, py::arg('timeout'), py::call_guard<py::gil_scoped_release>()) bin… | zxd1997066 | module: distributed |
| [#3139](https://github.com/intel/torch-xpu-ops/issues/3139) | P2 | [distributed][_composable] AssertionError: Expect… | Kanya-Mo | Submit an upstream PyTorch patch to replace the hard-coded device-0 in test_replicate_training.py (and the equivalent line in test_fully_sh… | zxd1997066 | bug, module: distributed |


<a id="sec-3-1-2-flash-attention"></a>
#### 3.1.2 Flash Attention  ·  6 issues

| Issue | Priority | Title | Owner | Fix Approach | Reporter | Labels |
|---|---|---|---|---|---|---|
| [#2015](https://github.com/intel/torch-xpu-ops/issues/2015) | P1 | inf is returned by nn.TransformerEncoderLayer | yucai-intel | Rebase onto PR #2336 + pytorch/pytorch#168234 and re-run pytest -v test/test_nn.py::TestNNDeviceTypeXPU -k test_transformerencoderlayer. If… | daisyden | skipped |
| [#3140](https://github.com/intel/torch-xpu-ops/issues/3140) | P2 | [upstream_ut] RuntimeError: FlashAttentionForward… | LuFinch | Implement dropout in the sycltla FlashAttention kernels: add a Philox4x32 RNG state (seed/offset plumbed through FLASH_FWD_params and FLASH… | daisyden | module: ut, skipped, ut_upstream |
| [#3142](https://github.com/intel/torch-xpu-ops/issues/3142) | P2 | [upstream_ut] RuntimeError: The sycl_ext_oneapi_w… | LuFinch | Pure oneAPI/driver-side fix: wait for oneAPI 2026.0 which lifts the graph restriction on work_group_scratch_memory (per LuFinch/daisyden). … | daisyden | dependency component: oneAPI, module: u… |
| [#3195](https://github.com/intel/torch-xpu-ops/issues/3195) | P2 | test_sdpa_unbacked_no_dde_xpu crashed |  | Reproduce under CI with `pytest -p no:xdist` to confirm xdist worker is the failure surface; if so, mark the test with @pytest.mark.forked … | daisyden | skipped, random |
| [#3326](https://github.com/intel/torch-xpu-ops/issues/3326) | P2 | Sporadic test_mem_eff_attention_large_seq_len_uni… |  | Keep the test in the skip list and tag for driver investigation: collect ZE_DEBUG and dmesg output from a failing CI run, capture the seq_l… | Silv3S | skipped, random |
| [#3356](https://github.com/intel/torch-xpu-ops/issues/3356) | P2 | [upstream_ut] dynamo/test_activation_checkpointin… |  | Land XPU device-specific tolerance override in the upstream PR pytorch/pytorch#169241 (raise atol/rtol for `xpu` like CUDA does for autocas… | shangerxin | skipped, ut_upstream |


<a id="sec-3-1-3-inductor"></a>
#### 3.1.3 Inductor  ·  23 issues

| Issue | Priority | Title | Owner | Fix Approach | Reporter | Labels |
|---|---|---|---|---|---|---|
| [#3094](https://github.com/intel/torch-xpu-ops/issues/3094) | P1 | XPUGraph tree support |  | Implement XPUGraph tree support: (1) expose a torch.xpu.graph/XPUGraph capture API backed by SYCL command-graph (or Level Zero command list… | daisyden | module: inductor, ut_upstream |
| [#3342](https://github.com/intel/torch-xpu-ops/issues/3342) | P1 | c-shim implementation is missing for aten.unsquee… | CuiYifeng | Fix in upstream PyTorch inductor: ensure aten.unsqueeze/expand/split_with_sizes are lowered as views in torch/_inductor/lowering.py for the… | kaixuanliu |  |
| [#3386](https://github.com/intel/torch-xpu-ops/issues/3386) | P1 | [Bug Skip] XPU Dynamo ocloc/IGC compilation failu… |  | Align intel-igc-cm/intel-ocloc/compute-runtime package versions in the CI image (install matching libigc-core/libigc2 and verify LD_LIBRARY… | daisyden | module: inductor, module: ut, skipped |
| [#2888](https://github.com/intel/torch-xpu-ops/issues/2888) | P2 | torch._inductor.exc.InductorError: AssertionError… | Stonepia | Test is explicitly named 'bad_cast' and expects an exception on eager vs compile path; align the XPU test expectation. Options: (1) skip te… | daisyden | module: inductor, ut_upstream |
| [#2891](https://github.com/intel/torch-xpu-ops/issues/2891) | P2 | RuntimeError: Expected to find "(262144, 0, 512, … | chunhuanMeng | Either (a) skip/xfail this test for XPU in third_party/torch-xpu-ops/test/inductor skip lists because it is a CUDA codegen-text assertion, … | daisyden | module: inductor, ut_upstream |
| [#2908](https://github.com/intel/torch-xpu-ops/issues/2908) | P2 | [release/2.11] Model fail_accuracy for 5 testcases | xuhancn | Close the 3 driver-fixed models once driver >=8531 is the minimum for 2.11 release notes. For pit_b_224: bisect torch-xpu-ops main between … | bjarzemb | E2E |
| [#2922](https://github.com/intel/torch-xpu-ops/issues/2922) | P2 | [release/2.11] UT inductor AssertionError: pass_f… | tadkrawiec | Fix upstream in torch/_inductor/test_operators or test/inductor/test_compile_subprocess.py (and the helper in torch/_inductor/compile_worke… | bjarzemb | os: Windows |
| [#2924](https://github.com/intel/torch-xpu-ops/issues/2924) | P2 | [release/2.11] xcit_large_24_p8_224 amp_bf16 trai… | jianyizh, mengfei25 | Align the eager and Inductor math paths: (1) in third_party/torch-xpu-ops/src/comm/XPUMathCompat.h replace sycl::rsqrt with 1.f/sycl::sqrt … | mengfei25 | Accuracy, dependency component: Triton |
| [#2928](https://github.com/intel/torch-xpu-ops/issues/2928) | P2 | [release/2.11] pyhpc_turbulent_kinetic_energy fp3… | jianyizh | Short term: in torch/_inductor/codegen/triton.py (or the xpu override), force libdevice.sqrt (or an fp32-safe fallback) for the XPU backend… | mengfei25 | dependency component: Triton |
| [#2938](https://github.com/intel/torch-xpu-ops/issues/2938) | P2 | [release/2.11] basic_gnn_gin and basic_gnn_sage i… | jianyizh | This is an upstream Inductor issue; track via pytorch#177117. Workaround in torch-xpu-ops CI: skip/baseline these two models until upstream… | mengfei25 | performance, dependency component: comm… |
| [#2952](https://github.com/intel/torch-xpu-ops/issues/2952) | P2 | [release/2.11][wsl] timm_models_accuracy_training… | weishi-deng | No torch-xpu-ops kernel change required. Either (a) relax the accuracy tolerance for this bf16 training model on BMG, or (b) work with the … | bjarzemb | Accuracy, hw: BMG |
| [#2960](https://github.com/intel/torch-xpu-ops/issues/2960) | P2 | [release/2.11] timm_models_xcit_large_24_p8_224_f… | pfierek, tadkrawiec | Confirm if raising `cosine` tolerance threshold to match CUDA expectations for xcit (common practice for fp16 training accuracy in benchmar… | shangerxin | os: Windows |
| [#2997](https://github.com/intel/torch-xpu-ops/issues/2997) | P2 | AssertionError of test_linear_and_cel_max_autotune | etaf | Track upstream fix pytorch/pytorch#180330 (already identified by @etaf) and land it in the PT 2.12 cherry-pick queue; per assignee, 2.12 re… | daisyden | module: inductor, ut_upstream |
| [#3006](https://github.com/intel/torch-xpu-ops/issues/3006) | P2 | AssertionError: '.to(tl.float16)' unexpectedly fo… | CuiYifeng | In the Triton reduction codegen for argmax/argmin (triton.py around line 4469 `final_argreduce` and the block-ptr store generation), ensure… | daisyden | module: inductor, ut_upstream |
| [#3080](https://github.com/intel/torch-xpu-ops/issues/3080) | P2 | cudagraph tests blocked by feature gap |  | This is a tracker/feature-gap issue. Extend Inductor+Dynamo cudagraph trees to dispatch through a generic device-graph abstraction (CUDAGra… | daisyden | module: ut |
| [#3096](https://github.com/intel/torch-xpu-ops/issues/3096) | P2 | VISIBLE_DEVICE support |  | In torch/_inductor/autotune_process.py extend TuningProcessPool.initialize / _subproc_env to set ZE_AFFINITY_MASK (or ONEAPI_DEVICE_SELECTO… | daisyden | module: ut |
| [#3148](https://github.com/intel/torch-xpu-ops/issues/3148) | P2 | [Triton] Huggingface openai/whisper-tiny got fail… | mengfei25 | Primary: wait for the Triton fix in intel/intel-xpu-backend-for-triton#6489 and bump the pinned triton-xpu commit. Short-term mitigation (p… | mengfei25 | Accuracy, hw: BMG, hw: PVC, dependency … |
| [#3187](https://github.com/intel/torch-xpu-ops/issues/3187) | P2 | PyTorch XPU gpu_cpp_wrapper fails with InductorEr… | CuiYifeng | In the XPU cpp_wrapper / AOTI codegen (torch/_inductor/codegen/cpp_wrapper_gpu.py and XPU-specific wrapper/AOTI shims) ensure that when a d… | liangan1 | ut_upstream |
| [#3361](https://github.com/intel/torch-xpu-ops/issues/3361) | P2 | [upstream_ut] test/dynamo/test_higher_order_ops.p… | kdrozd-dev | Wait for upstream PR pytorch/pytorch#174370 to land (generalizes RngStateHelper to dispatch by device, supporting XPU), then rebase and rem… | shangerxin |  |
| [#3388](https://github.com/intel/torch-xpu-ops/issues/3388) | P2 | [Bug Skip] XPU Dynamo Graph Lowering - stream_ind… |  | In Dynamo's stream variable handling, ensure XPU streams are registered via _get_stream_by_index and their integer index is captured in the… | daisyden | module: ut, skipped, module: dynamo |
| [#3389](https://github.com/intel/torch-xpu-ops/issues/3389) | P2 | [Bug Skip] XPU record_stream Tests Fail with CPU … |  | Update test_streams_xpu tests to explicitly create inputs on the XPU device (device='xpu', requires_grad=True) so record_stream routes to t… | daisyden | module: ut, skipped, module: dynamo |
| [#3393](https://github.com/intel/torch-xpu-ops/issues/3393) | P2 | [upstream_ut] test/dynamo/test_activation_checkpo… |  | Generalize the test's backend-selection branch to handle XPU (use torch.xpu.get_device_properties and the XPU-appropriate sdpa op, or gate … | shangerxin | skipped, ut_upstream |
| [#3395](https://github.com/intel/torch-xpu-ops/issues/3395) | P2 | [upstream_ut] test/dynamo/test_ctx_manager.py::Ct… |  | Wait for / cherry-pick upstream PyTorch PR #174370 which teaches Dynamo to trace torch.accelerator.device_index (e.g. by removing the accel… | shangerxin |  |


<a id="sec-3-1-4-sparse"></a>
#### 3.1.4 Sparse  ·  2 issues

| Issue | Priority | Title | Owner | Fix Approach | Reporter | Labels |
|---|---|---|---|---|---|---|
| [#3175](https://github.com/intel/torch-xpu-ops/issues/3175) | P2 | [Bug Skip]: ValueError: sampled_addmm(): all inpu… | jkosnox | In torch/sparse/_triton_ops.py:34 change the check to `t.device.type in ("cuda", "xpu")` (same pattern already applied for other _triton_op… | CuiYifeng | skipped |
| [#3177](https://github.com/intel/torch-xpu-ops/issues/3177) | P2 | Accuracy gap of BF16/FP16 test_block_addmm | jenniew | Land torch-xpu-ops PR #3273: replace result_dense.add_(input_dense, beta) at SparseCsrTensorMath.cpp:88 with explicit `result_dense.add_(in… | CuiYifeng | skipped |


<a id="sec-3-1-5-torch-operations"></a>
#### 3.1.5 Torch Operations  ·  15 issues

| Issue | Priority | Title | Owner | Fix Approach | Reporter | Labels |
|---|---|---|---|---|---|---|
| [#1574](https://github.com/intel/torch-xpu-ops/issues/1574) | P1 | The operator 'aten::_grouped_mm' is not currently… | Stonepia, LuFinch | Add XPU kernel for _grouped_mm under src/ATen/native/xpu/ that dispatches to oneDNN matmul primitive with group descriptor (requires oneDNN… | githubsgi | module: ao |
| [#2239](https://github.com/intel/torch-xpu-ops/issues/2239) | P1 | Exception: could not create a primitive descripto… | wpietka | In Deconv.cpp add pre-dispatch parameter validation mirroring oneDNN's requirements; when output_padding or dilation combinations are unsup… | zxd1997066 | skipped, bug_fix_stage5 |
| [#2479](https://github.com/intel/torch-xpu-ops/issues/2479) | P1 | [Bug] torch.rand output different result on bmg a… | Stonepia, CuiYifeng | Make the Philox counter->element mapping independent of device topology: fix the effective grid (e.g. compute num_groups from numel with a … | zufangzhu |  |
| [#2845](https://github.com/intel/torch-xpu-ops/issues/2845) | P1 | [Bug Skip]:[UT] [Windows] failed cases 2026-2-4 |  | Consolidate with #2852; they share root cause. Add the listed cases to test/xpu/skip_list_win.py temporarily and track fix under a single o… | kaileiyx | skipped_windows |
| [#2852](https://github.com/intel/torch-xpu-ops/issues/2852) | P1 | [Bug Skip]: New UT failures in 0206 nightly on Wi… |  | Bisect the torch-xpu-ops range (SpectralOps.cpp changes) and the PyTorch range (test_ops extended harness) to locate the breaking commit; c… | chuanqi129 | skipped_windows |
| [#3084](https://github.com/intel/torch-xpu-ops/issues/3084) | P1 | torch.library.register_autocast does not support … | CuiYifeng | Extend register_autocast to accept 'xpu' (and ideally any device type registered via torch._register_device_module): add self._autocast_xpu… | daisyden | module: ut |
| [#2840](https://github.com/intel/torch-xpu-ops/issues/2840) | P2 | Accuracy issue with 64 bit indexing depthwise_conv | SlawomirLaba, Silv3S | File / follow the oneDNN ticket for depthwise convolution accuracy regression under int64 offsets; in the interim, keep the skip in test/xp… | kdrozd-dev | dependency component: oneDNN, skipped, … |
| [#2862](https://github.com/intel/torch-xpu-ops/issues/2862) | P2 | accuracy issue with test_float8_scale_fast_accum_… | tszulist-hbn | Bump bundled oneDNN in cmake/External/oneDNN.cmake to v3.10.2 (hash f1d47193..) which contains PR #4923. Then re-enable the test in test/xp… | daisyden |  |
| [#2869](https://github.com/intel/torch-xpu-ops/issues/2869) | P2 | [Bug Skip]: New UT failure in 0209 nightly window… |  | Split this umbrella issue into per-module children: (1) Conv fp64/complex128 Windows: investigate oneDNN engine creation for f64/c64 conv o… | RUIJIEZHONG66166 | skipped_windows |
| [#2953](https://github.com/intel/torch-xpu-ops/issues/2953) | P2 | [release/2.11][wsl] huggingface TrOCRForCausalLM … |  | In the XPU `full`/`fill_` scalar dispatch, convert the Scalar directly to the target dtype (bfloat16/half) instead of first narrowing to fl… | bjarzemb | os: Windows |
| [#2965](https://github.com/intel/torch-xpu-ops/issues/2965) | P2 | [Bug Skip]: Random failures 2026WW10 |  | Split remaining cases out of this umbrella (conv3d/max_pool already moved to #3103 and #2676). For conv_transpose3d jvpvjp: either raise to… | CuiYifeng | hw: PVC, skipped, random |
| [#3136](https://github.com/intel/torch-xpu-ops/issues/3136) | P2 | [upstream_ut] AssertionError: False is not true i… | LuFinch | For test_disable_fastpath_xpu, land/forward-port pytorch/pytorch#179701 and re-enable. For the two fused_sdp_priority_order cases, add them… | daisyden | module: ut, skipped, ut_upstream |
| [#3137](https://github.com/intel/torch-xpu-ops/issues/3137) | P2 | [upstream_ut] RuntimeError: expected scalar type … | LuFinch | Merge/forward-port pytorch/pytorch#179701 so `_transformer_encoder_layer_fwd` takes the slow path (or casts parameters) under xpu autocast,… | daisyden | module: ut, skipped, ut_upstream, random |
| [#3180](https://github.com/intel/torch-xpu-ops/issues/3180) | P2 | [E2E] Timm/Torchbench models got "eager_two_runs_… | pbielak | Bisect by minifying one model (e.g. coat_lite_mini) to a single op that differs between two eager runs on ARC-Windows, then check whether t… | libohao1201 | Accuracy, os: Windows |
| [#3259](https://github.com/intel/torch-xpu-ops/issues/3259) | P2 | New failed test cases 2026-04-02 | SlawomirLaba | Add the two hipdnn tests to skip_list_common.py under test_convolution_xpu (they are ROCm-only; @onlyROCM or equivalent guard is missing up… | Silv3S | skipped |


<a id="sec-3-1-6-torch-runtime"></a>
#### 3.1.6 Torch Runtime  ·  8 issues

| Issue | Priority | Title | Owner | Fix Approach | Reporter | Labels |
|---|---|---|---|---|---|---|
| [#2467](https://github.com/intel/torch-xpu-ops/issues/2467) | P1 | Host may stuck when submit too many kernels when … | jianyizh | Primary fix is on the Level-Zero driver (GSD-12059) to enlarge/grow the event pool or recycle completed events. In the torch-xpu-ops/PyTorc… | jianyizh | dependency component: driver |
| [#3227](https://github.com/intel/torch-xpu-ops/issues/3227) | P1 | torch xpu event has ~0.1ms latency, which is too … | guangyey | Replace submit_profiling_tag with a lighter mechanism: either use an in-order queue's built-in timestamp (ze_event_pool with TIMESTAMP flag… | jianyizh |  |
| [#3350](https://github.com/intel/torch-xpu-ops/issues/3350) | P1 | [profiler] [XPU][Windows] torch.profiler fails to… | aostrowski-hbn | Reproduce with PTI debug logging enabled, file a bug against intel/pti-gpu (or upgrade to a newer intel-pti release that supports Windows L… | ZhaoqiongZ | module: profiler |
| [#2858](https://github.com/intel/torch-xpu-ops/issues/2858) | P2 | [Bug Skip]: test_xpu new failures |  | Validate on newer Intel GPU Windows driver; if the runtime-side bug persists, file driver ticket. In the meantime keep skip entry in test/x… | RUIJIEZHONG66166 | os: Windows, skipped_windows |
| [#3000](https://github.com/intel/torch-xpu-ops/issues/3000) | P2 | [Bug Skip]: RuntimeError: _share_fd_: only availa… | gplutop7 | Short-term: extend skip_list_common.py with 'test_dataloader_xpu.py': ('test_nested_tensor_multiprocessing_context_forkserver_xpu', 'test_n… | zxd1997066 | module: ut, skipped |
| [#3048](https://github.com/intel/torch-xpu-ops/issues/3048) | P2 | Profiler result is not correct on B70 | aostrowski-hbn | Blocked on the Intel GPU driver fix tracked as PTI-384. Once the driver delivers corrected kernel timestamps, re-validate the trace with th… | jianyizh | module: profiler |
| [#3243](https://github.com/intel/torch-xpu-ops/issues/3243) | P2 | AssertionError: False is not true | pponikox | Add an XPU equivalent to _sleep_if_cuda (e.g. a torch.xpu._sleep helper that busy-loops on device) and call it in the side-stream branch, o… | zxd1997066 | module: ut, skipped |
| [#3314](https://github.com/intel/torch-xpu-ops/issues/3314) | P2 | Test_xpu.py: Fatal Python error: Aborted on windo… |  | Wrap MemPool::~MemPool body (releasePool + emptyCache) in a try/catch that logs and swallows c10::Error/sycl::exception so destruction neve… | RUIJIEZHONG66166 | os: Windows |


<a id="sec-3-1-7-torchao"></a>
#### 3.1.7 TorchAO  ·  2 issues

| Issue | Priority | Title | Owner | Fix Approach | Reporter | Labels |
|---|---|---|---|---|---|---|
| [#3368](https://github.com/intel/torch-xpu-ops/issues/3368) | P1 | [TorchAO][BMG] DeepSeek-R1-Distill-Llama-8B RTN I… | Stonepia | Profile the two runs (links 99 vs 100) to pinpoint whether the regression comes from extra guard evaluation, additional recompiles/graph br… | LifengWang |  |
| [#3088](https://github.com/intel/torch-xpu-ops/issues/3088) | P2 | [TorchAO][BMG] INT4 RTN Flex-attention got 5% per… | hoshibara | Two-pronged: (1) file/track triton XPU backend register-spill regression (already filed as intel-xpu-backend-for-triton#6625) so codegen re… | LifengWang | dependency component: Triton, module: ao |


<a id="sec-3-2-needs-owner"></a>
### 3.2 NEEDS_OWNER  ·  35 issues

**NEEDS_OWNER — awaiting triage-lead to assign an owner**

<a id="sec-3-2-1-distributed"></a>
#### 3.2.1 Distributed  ·  8 issues

| Issue | Priority | Title | Owner | Fix Approach | Reporter | Labels |
|---|---|---|---|---|---|---|
| [#1548](https://github.com/intel/torch-xpu-ops/issues/1548) | P2 | [distributed] AssertionError: 'fused_all_gather_m… | Chao1Han | 1) Blocked on XPU SymmetricMemory enablement in torch-xpu-ops (Level Zero IPC or SYCL symm API). 2) Once available, register fused_all_gath… | PenghuiCheng | module: distributed, dependency compone… |
| [#1549](https://github.com/intel/torch-xpu-ops/issues/1549) | P2 | [distributed] AssertionError: 'fused_all_gather_s… | Chao1Han | 1) Depends on #1551: enable XPU SymmetricMemory via SYCL 2026 APIs, then register fused_all_gather_scaled_matmul for XPU and ensure the Ind… | PenghuiCheng | module: distributed, dependency compone… |
| [#1551](https://github.com/intel/torch-xpu-ops/issues/1551) | P2 | [distributed] NotImplementedError: The operator '… | Chao1Han | 1) Wait for SYCL symmetric memory support in oneAPI 2026.0 and implement an XPU SymmetricMemoryAllocator + register fused_scaled_matmul_red… | PenghuiCheng | module: distributed, dependency compone… |
| [#1555](https://github.com/intel/torch-xpu-ops/issues/1555) | P2 | [distributed] RuntimeError: aten.add.Tensor: got … | chuanqi129 | 1) Integrate the oneDNN fused attention kernel for XPU so SDPA does not fall back to the MATH decomposition under DTensor. 2) Until then, e… | PenghuiCheng | module: distributed, dependency compone… |
| [#1556](https://github.com/intel/torch-xpu-ops/issues/1556) | P2 | [distributed] NotImplementedError: Operator aten.… | pkourdis | 1) Land the oneDNN-based fused SDPA kernel for XPU so the generic overrideable op is backed by a real forward+backward implementation. 2) I… | PenghuiCheng | module: distributed, dependency compone… |
| [#1727](https://github.com/intel/torch-xpu-ops/issues/1727) | P2 | [distributed] AttributeError: module 'torch.xpu' … | guangyey | Once oneAPI 2026 is adopted, add a `_sleep(cycles)` binding in `torch/csrc/xpu/Module.cpp` backed by a SYCL kernel that busy-waits the requ… | PenghuiCheng | module: distributed, dependency compone… |
| [#2163](https://github.com/intel/torch-xpu-ops/issues/2163) | P2 | 3 distributed UT cases need to be supported by - … | githubsgi | Land the four enumerated upstream enablement changes: add an XPU entry in torch/_inductor/analysis/device_info.py, replace torch.cuda.* wit… | libohao1201 | module: distributed |
| [#2165](https://github.com/intel/torch-xpu-ops/issues/2165) | P2 | [distributed] test_device_mesh.py::TestDeviceMesh… | jemitche1 | Implement ProcessGroupXCCL::splitGroup() mirroring ProcessGroupNCCL's splitGroup (and also implement mergeRemoteGroup if needed) so it hono… | zxd1997066 | bug, module: distributed |


<a id="sec-3-2-2-flash-attention"></a>
#### 3.2.2 Flash Attention  ·  1 issues

| Issue | Priority | Title | Owner | Fix Approach | Reporter | Labels |
|---|---|---|---|---|---|---|
| [#2270](https://github.com/intel/torch-xpu-ops/issues/2270) | P2 | Backend Compatibility Error in test/xpu/test_deco… | LuFinch | Either (a) skip _flash_attention_forward in the decomp cross-ref OpInfo list for XPU (mirroring CUDA's skip in common_methods_invocations, … | libohao1201 | module: ut, skipped |


<a id="sec-3-2-3-inductor"></a>
#### 3.2.3 Inductor  ·  3 issues

| Issue | Priority | Title | Owner | Fix Approach | Reporter | Labels |
|---|---|---|---|---|---|---|
| [#1505](https://github.com/intel/torch-xpu-ops/issues/1505) | P2 | [ARC-WSL-Ubuntu24.04] 15 Timm models got fail_acc… |  | Reduce repro to a minimal failing op per model (e.g., conv backward, embedding backward for Albert token_type_embeddings), compare WSL vs L… | libohao1201 | bug, E2E, client, os: Windows, module: … |
| [#2329](https://github.com/intel/torch-xpu-ops/issues/2329) | P2 | [upstream_ut] feature missing: get_device_tflops … | etaf | Add XPU support in torch/_inductor/utils.py get_device_tflops() and get_dram_gbps(): detect device.type=='xpu' and compute peak TFLOPS from… | daisyden | duplicate, dependency component: Triton… |
| [#2331](https://github.com/intel/torch-xpu-ops/issues/2331) | P2 | [upstream_ut] AssertionError: Scalars are not equ… | hoshibara | After pytorch/pytorch#172314 lands, update torch/_inductor/choices.py::_get_exceeding_shared_memory_checker to be device-agnostic: use torc… | daisyden | dependency component: oneAPI, module: i… |


<a id="sec-3-2-4-others"></a>
#### 3.2.4 Others  ·  4 issues

| Issue | Priority | Title | Owner | Fix Approach | Reporter | Labels |
|---|---|---|---|---|---|---|
| [#1649](https://github.com/intel/torch-xpu-ops/issues/1649) | P2 | [cpp extension] Provide a clear error message whe… | dvrogozh | In torch/utils/cpp_extension.py SyclExtension helpers, record the oneAPI/ICX version used at wheel build time (e.g. in torch/version.py or … | ZhaoqiongZ | dependency component: oneAPI, module: b… |
| [#1762](https://github.com/intel/torch-xpu-ops/issues/1762) | P2 | Add an ocloc AOT target compilation test in cmake | chunhuanMeng | Add a CMake `try_compile`/`execute_process` step in the torch-xpu-ops top-level CMake that runs `ocloc compile -device <arch>` with a minim… | jingxu10 | module: build |
| [#2024](https://github.com/intel/torch-xpu-ops/issues/2024) | P2 | AssertionError: Torch not compiled with CUDA enab… | daisyden | Either (a) add these test IDs to test/xpu/skip_list_common.py with a clear TODO so CI stays green, or (b) preferred: send an upstream PR re… | mengfei25 | module: ut, skipped |
| [#2098](https://github.com/intel/torch-xpu-ops/issues/2098) | P2 | Upstream XPU functions in yaml | guangyey | Upstream XPU dispatch entries into aten/src/ATen/native/native_functions.yaml (add 'XPU: <kernel>' to each op currently registered out-of-t… | EikanWang | enhancement |


<a id="sec-3-2-5-sparse"></a>
#### 3.2.5 Sparse  ·  3 issues

| Issue | Priority | Title | Owner | Fix Approach | Reporter | Labels |
|---|---|---|---|---|---|---|
| [#2214](https://github.com/intel/torch-xpu-ops/issues/2214) | P2 | test/test_sparse.py::TestSparseAnyXPU::test_gradc… | jenniew | Extend the expected-error regex in test/test_sparse.py (around line 5297) to also accept the XPU-specific message (e.g., add 'empty_sparse_… | wincent8 | skipped, ut_upstream |
| [#2246](https://github.com/intel/torch-xpu-ops/issues/2246) | P2 | torch/sparse/_triton_ops*.py need to be ported to… |  | Port the two modules to be device-agnostic: replace torch.cuda.get_device_name() with a helper that picks the active accelerator (torch.acc… | wincent8 | skipped |
| [#2235](https://github.com/intel/torch-xpu-ops/issues/2235) | P3 | test/test_sparse_csr.py::TestSparseCompressedTrit… |  | Resolve together with #2246: make get_meta() query the current accelerator and register XPU tuning results in _operation_device_version_dat… | wincent8 | skipped |


<a id="sec-3-2-6-torch-operations"></a>
#### 3.2.6 Torch Operations  ·  8 issues

| Issue | Priority | Title | Owner | Fix Approach | Reporter | Labels |
|---|---|---|---|---|---|---|
| [#2912](https://github.com/intel/torch-xpu-ops/issues/2912) | P1 | [release/2.11] UT extended 220 new failures | unassigned | This needs decomposition, not a single fix: (1) parse the attached changed_tests.log and cluster the 220 failures by op/error signature; (2… | bjarzemb | os: Windows, hw: BMG |
| [#1856](https://github.com/intel/torch-xpu-ops/issues/1856) | P2 | channel last aten::hardswish_ will call extra copy | chunhuanMeng | Teach the hardswish (and other Activation*Kernels.cpp files using gpu_kernel) to detect NHWC/channels-last input and dispatch on a flattene… | jianyizh | performance, hw: BMG |
| [#1901](https://github.com/intel/torch-xpu-ops/issues/1901) | P2 | implement torch.linalg.svd xpu backend | CuiYifeng | Option A: implement linalg_svd XPU kernel via oneMKL gesvd/gesvdj and gate it behind a size/shape heuristic to avoid regressions vs CPU pat… | yao-matrix | module: op impl |
| [#1951](https://github.com/intel/torch-xpu-ops/issues/1951) | P2 | Functionality issues in TestCommon.test_out. | AKloniecki | Split per-op: (a) In BatchNormKernels.cpp:~585 call at::native::resize_output on out_invstd/out_mean before asserting contiguity, or reshap… | daisyden | module: ut, skipped, ut_upstream |
| [#2255](https://github.com/intel/torch-xpu-ops/issues/2255) | P2 | [upstream_ut] RuntimeError: Long is not supported… | daisyden | Either (a) land upstream PyTorch PR #169353 (referenced in the issue) which aligns OpInfo dtypes to exclude int64 for XPU matmul/conv, or (… | daisyden | skipped, ut_upstream |
| [#2257](https://github.com/intel/torch-xpu-ops/issues/2257) | P2 | Accuracy failures in test/xpu/test_unary_ufuncs_x… | pbielak | Port CUDA's explicit complex formulas (from aten/src/ATen/native/cuda/UnaryComplexKernels.cu and c10/util/complex_math.h) into the XPU sycl… | zxd1997066 | skipped, bug_fix_stage4 |
| [#2295](https://github.com/intel/torch-xpu-ops/issues/2295) | P2 | [upstream_ut][xpu][test]nn/test_embedding.py::Tes… | yucai-intel | Bump the PyTorch CI xpu docker image to use the same IGC/oneAPI version as nightly wheel builds (track via .ci/docker/common/install_xpu.sh… | wincent8 | module: inductor, skipped, ut_upstream |
| [#2301](https://github.com/intel/torch-xpu-ops/issues/2301) | P2 | [upstream_ut] dtypes not align with OpInfo | daisyden | Land the dtype alignment already prototyped in pytorch/pytorch PR #161246 (commit 7f545509) which extends OpInfo.dtypesIfXPU for einsum, in… | daisyden | skipped, ut_upstream |


<a id="sec-3-2-7-torch-runtime"></a>
#### 3.2.7 Torch Runtime  ·  2 issues

| Issue | Priority | Title | Owner | Fix Approach | Reporter | Labels |
|---|---|---|---|---|---|---|
| [#1970](https://github.com/intel/torch-xpu-ops/issues/1970) | P2 | torch._dynamo.exc.BackendCompilerFailed: backend=… |  | Generalize CUDARngStateHelper to a device-agnostic helper that dispatches on the current fake/real device: replace the hardcoded torch.cuda… | shangerxin | module: ut |
| [#2089](https://github.com/intel/torch-xpu-ops/issues/2089) | P2 | need an implementation that won't initialize gpu … | guangyey | Implement a fork-/init-safe device probe using level-zero sysman or XPU-SMI that reports device count without creating a SYCL context, mirr… | faaany | dependency component: driver |


<a id="sec-3-2-8-torchao"></a>
#### 3.2.8 TorchAO  ·  6 issues

| Issue | Priority | Title | Owner | Fix Approach | Reporter | Labels |
|---|---|---|---|---|---|---|
| [#1912](https://github.com/intel/torch-xpu-ops/issues/1912) | P2 | Implement the torch.ops.aten._weight_int4pack_mm … | liangan1 | Track oneDNN 3.11 release for float zero-point support in int4 matmul. Once available, extend the xpu int4 op (src/ATen/native/xpu/Int4Pack… | yuanwu2017 | dependency component: oneDNN |
| [#2201](https://github.com/intel/torch-xpu-ops/issues/2201) | P2 | [TorchAO][BMG] When using paged attention backend… | Stonepia | Two-part fix: (1) In the client run_generation.py (gpu-models) ensure a transformers PagedCache/DynamicCache is created and passed when --a… | MingxuZh | module: ao |
| [#2323](https://github.com/intel/torch-xpu-ops/issues/2323) | P2 | [TorchAO] MOE training enabling on XPU | karol-brejna-i | Drive the oneDNN scaled_group_gemm primitive to completion, then register `_scaled_grouped_mm` / `_scaled_grouped_mm_v2` XPU dispatches in … | liangan1 | dependency component: oneDNN, module: ao |
| [#2324](https://github.com/intel/torch-xpu-ops/issues/2324) | P2 | [TorchAO] FP8 conv support | Stonepia | Add an FP8 convolution op in torch-xpu-ops (e.g. `src/ATen/native/xpu/ScaledConv.cpp`) backed by oneDNN convolution with per-tensor/per-cha… | liangan1 | module: ao |
| [#2325](https://github.com/intel/torch-xpu-ops/issues/2325) | P2 | [TorchAO] Float8 training support on XPU | arlesniak | Track and land `_scaled_mm_xpu` via oneDNN FP8 matmul (rowwise + tensorwise scaling), enable Float8 training UTs with the emulate path firs… | liangan1 | module: ao |
| [#2326](https://github.com/intel/torch-xpu-ops/issues/2326) | P2 | [TorchAO] MX training native PyTorch on XPU | karol-brejna-i | Land/track XPU `_scaled_mm` (PR #165978) via oneDNN matmul with scale support, enable the MX training UTs in `torchao/prototype/mx_formats`… | liangan1 | module: ao |


<a id="sec-3-3-track-pr"></a>
### 3.3 TRACK_PR  ·  76 issues

**TRACK_PR — identified PR is open; wait for / push to merge**

<a id="sec-3-3-1-distributed"></a>
#### 3.3.1 Distributed  ·  1 issues

| Issue | Priority | Title | Owner | Fix Approach | Reporter | Labels |
|---|---|---|---|---|---|---|
| [#489](https://github.com/intel/torch-xpu-ops/issues/489) | P3 | Moco NotImplementedError: xpu not supported | weishi-deng | Land pytorch/benchmark#2616 (device-agnostic MoCo init using XCCL when xpu is selected), ensure ProcessGroupXCCL supports the collectives M… | mengfei25 | E2E, Accuracy, module: torchbench, dtyp… |


<a id="sec-3-3-2-flash-attention"></a>
#### 3.3.2 Flash Attention  ·  4 issues

| Issue | Priority | Title | Owner | Fix Approach | Reporter | Labels |
|---|---|---|---|---|---|---|
| [#2442](https://github.com/intel/torch-xpu-ops/issues/2442) | P1 | [Bug Skip]: NotImplementedError: Could not run 'a… | LuFinch | Implement a native XPU _flash_attention_forward SYCL kernel (e.g. under src/ATen/native/xpu/FlashAttention.cpp) and register it via native_… | CuiYifeng | skipped |
| [#2853](https://github.com/intel/torch-xpu-ops/issues/2853) | P1 | [upstream_ut] torch.ops.aten._flash_attention_for… | LuFinch | Add an XPU registration for aten::_flash_attention_forward in yaml/native/native_functions.yaml (XPU dispatch) and implement the wrapper in… | BBBela | skipped |
| [#3093](https://github.com/intel/torch-xpu-ops/issues/3093) | P1 | XPU does not support NestedTensor for SDPA operat… | tszulist-hbn | Add an XPU branch in torch/nested/_internal/sdpa.py::_select_sdp_backend that either (a) routes to a new jagged-NT SDPA overload implemente… | daisyden | module: ut, skipped |
| [#2698](https://github.com/intel/torch-xpu-ops/issues/2698) | P2 | Title: [upstream_ut] RuntimeError: FlashAttention… | chunhuanMeng, LuFinch | (1) In _scaled_dot_product_flash_attention_xpu dispatcher, check headdim eligibility before calling run_mha_fwd and fall back to math/effic… | daisyden | module: inductor, skipped, ut_upstream |


<a id="sec-3-3-3-inductor"></a>
#### 3.3.3 Inductor  ·  12 issues

| Issue | Priority | Title | Owner | Fix Approach | Reporter | Labels |
|---|---|---|---|---|---|---|
| [#3191](https://github.com/intel/torch-xpu-ops/issues/3191) | P1 | torch._inductor.exc.InductorError: AssertionError… | EikanWang, Copilot | In torch/_inductor/lowering.py, pass override_decomp=True when registering the fallback for aten.index_add (or remove the conditional decom… | mengfei25 | E2E, hw: PVC |
| [#3308](https://github.com/intel/torch-xpu-ops/issues/3308) | P1 | [Bug Skip]: [Regression] test_ctx_manager_xpu.py … | PatrykWilczewski | Re-sync test/xpu/dynamo/test_ctx_manager_xpu.py from upstream test/dynamo/test_ctx_manager.py (replace the base class with torch._dynamo.te… | kaileiyx | skipped |
| [#1877](https://github.com/intel/torch-xpu-ops/issues/1877) | P2 | Torchbench model squeezenet1_1 and functorch_dp_c… | DamJanusz | Bisect functorch_dp_cifar10 bf16 training failure to isolate the differing kernel (suspected BatchNorm backward) by comparing eager vs indu… | libohao1201 | Accuracy, hw: BMG, hw: PVC, bug_fix_sta… |
| [#2532](https://github.com/intel/torch-xpu-ops/issues/2532) | P2 | Title: [upstream_ut] AssertionError: wrong number… | yucai-intel | Align the XPU _convert_weight_to_int4pack op with the CUDA contract: (1) accept kByte weight and produce the same 4-D packed layout (N//8, … | daisyden | skipped, port_from_skiplist |
| [#2609](https://github.com/intel/torch-xpu-ops/issues/2609) | P2 | [upstream_ut] torch._inductor.exc.InductorError: … | daisyden | In pytorch upstream torch/_inductor/codegen (see cpp_wrapper_gpu.py / aoti shim generation for custom ops) and torch/csrc/inductor/aoti_tor… | daisyden | module: inductor, skipped, ut_upstream |
| [#2714](https://github.com/intel/torch-xpu-ops/issues/2714) | P2 | [upstream_ut] AssertionError: Object comparison f… | Silv3S | Update torch/_dynamo/variables/ctx_manager.py AutocastModeVariable to respect the 'xpu' device_type argument (mirror the 'cuda' branch to s… | daisyden | skipped, ut_upstream |
| [#2935](https://github.com/intel/torch-xpu-ops/issues/2935) | P2 | [release/2.11][inductor] huggingface amp_fp16 and… | jianyizh | Revert, guard, or re-tune the change in pytorch/pytorch@bc4d0bf3 for XPU. Work with the Inductor/Triton-XPU maintainers to (a) identify whi… | agnottaski | performance |
| [#2939](https://github.com/intel/torch-xpu-ops/issues/2939) | P2 | [release/2.11] gmlp_s16_224 inference amp perform… | jianyizh | In the Inductor XPU heuristics (torch/_inductor/runtime/triton_heuristics.py and hints.py), add an XPU-specific override that keeps DEFAULT… | mengfei25 | performance |
| [#3286](https://github.com/intel/torch-xpu-ops/issues/3286) | P2 | New failing test case after enabling tests from t… | BBBela | Land pytorch/pytorch#179905 (BBBela) which adds XPUDeviceVariable in torch/_dynamo/variables/ctx_manager.py mirroring CUDADeviceVariable, e… | BBBela | module: ut, skipped |
| [#3290](https://github.com/intel/torch-xpu-ops/issues/3290) | P2 | huggingface amp_bf16 inference accuracy openai/wh… | weishi-deng | Land pytorch/pytorch#180309 (already authored by weishi-deng) which extends Dynamo save_global_state to record XPU autocast state; per chua… | mengfei25 | E2E, Accuracy, hw: BMG, hw: PVC |
| [#3331](https://github.com/intel/torch-xpu-ops/issues/3331) | P2 | [ai_generated] torch.compile with slice_scatter p… | Copilot | Reproduce with TORCH_LOGS=output_code,inductor and capture the generated triton kernel for the backward; diff against the CUDA-generated ke… | laifenxiawucha | ai_generated |
| [#1963](https://github.com/intel/torch-xpu-ops/issues/1963) | P3 | [upstream_ut] MetadataMismatchError in TestFakeTe… | pbielak | Confirm PR pytorch/pytorch#178277 has merged and adds fake_autocast_device_skips["xpu"] = {"linalg.pinv", "pinverse"} in test/test_ops.py; … | daisyden | module: ut, ut_upstream |


<a id="sec-3-3-4-others"></a>
#### 3.3.4 Others  ·  5 issues

| Issue | Priority | Title | Owner | Fix Approach | Reporter | Labels |
|---|---|---|---|---|---|---|
| [#2662](https://github.com/intel/torch-xpu-ops/issues/2662) | P2 | [release/2.10][Windows][BMG] New failed test case… | tadkrawiec, kdrozd-dev | Split into sub-tasks: (a) CI side — install MSVC Build Tools and set `CC=cl`/add cl.exe to PATH for Windows XPU runners so inductor's C-shi… | mengfei25 | os: Windows, hw: BMG, module: ut |
| [#3209](https://github.com/intel/torch-xpu-ops/issues/3209) | P2 | [Win][Build] There is Cyclic dependencies error w… | Copilot | In setup_common_libraries(), drop or weaken the PUBLIC link from torch_xpu_ops to torch_xpu (e.g., replace with target_include_directories … | NeoZhangJianyu | module: build |
| [#2914](https://github.com/intel/torch-xpu-ops/issues/2914) | P3 | Test case test/test_autograd.py::TestAutogradMult… | kdrozd-dev | In tools/autograd/derivatives.yaml, register _test_autograd_multiple_dispatch_view_copy explicitly per dispatch key (AutogradCPU: grad.resh… | shangerxin |  |
| [#3362](https://github.com/intel/torch-xpu-ops/issues/3362) | P3 | test_nn_xpu.py::TestNN::test_cudnn_weight_format … | jmamzax | Either drop test_cudnn_weight_format from test_nn_xpu.py (it is intrinsically a cuDNN test), or rewrite it to use `device=device_type` and … | jmamzax | bug_fix_stage5 |
| [#492](https://github.com/intel/torch-xpu-ops/issues/492) | P3 | Timm_efficientdet NotImplementedError: The origin… | weishi-deng | Drive pytorch/benchmark PR #2374 to landing (generalize device string, replace `.cuda()` with `.to(device)`) and upstream similar changes t… | mengfei25 | E2E, Accuracy, module: torchbench, dtyp… |


<a id="sec-3-3-5-sparse"></a>
#### 3.3.5 Sparse  ·  5 issues

| Issue | Priority | Title | Owner | Fix Approach | Reporter | Labels |
|---|---|---|---|---|---|---|
| [#2283](https://github.com/intel/torch-xpu-ops/issues/2283) | P2 | [upstream_ut] sparse._sampled_addmm is not suppor… | jenniew | Implement sparse_sampled_addmm for SparseCsrXPU: add a sampled_addmm_out_sparse_csr_xpu kernel in src/ATen/native/sparse/xpu/sycl/SparseCsr… | daisyden | skipped, ut_upstream |
| [#3081](https://github.com/intel/torch-xpu-ops/issues/3081) | P2 | Sparse CSR gemm-like ops have not been supported … | tszulist-hbn | Split the fix: (a) extend SparseCsrTensorMathKernels.cpp to instantiate complex64/complex128 add kernels (AT_DISPATCH_ALL_TYPES_AND_COMPLEX… | daisyden | module: ut |
| [#3169](https://github.com/intel/torch-xpu-ops/issues/3169) | P2 | NotImplementedError: Could not run 'aten::hspmm' … | jkosnox | Port hspmm_sparse_cuda / hspmm_out_sparse_cuda from SparseCUDATensorMath.cu into torch-xpu-ops/src/ATen/native/sparse/xpu/ using SYCL kerne… | CuiYifeng | skipped, ut_upstream |
| [#2229](https://github.com/intel/torch-xpu-ops/issues/2229) | P3 | test/test_sparse_csr.py::TestSparseCompressedCPU:… | jenniew | Update the XPU/sparse test expectations in torch-xpu-ops skip/override list to match the actual emitted message (regex 'device of .* must m… | wincent8 | skipped |
| [#2921](https://github.com/intel/torch-xpu-ops/issues/2921) | P3 | [abs][complex64] - new failing test cases caused … | AKloniecki | Land pytorch/pytorch#177632 (already opened by AKloniecki) which adds the same XFAIL decorators for XPU that CPU/CUDA have. Alternatively, … | BBBela | skipped |


<a id="sec-3-3-6-torch-operations"></a>
#### 3.3.6 Torch Operations  ·  42 issues

| Issue | Priority | Title | Owner | Fix Approach | Reporter | Labels |
|---|---|---|---|---|---|---|
| [#3284](https://github.com/intel/torch-xpu-ops/issues/3284) | P1 | Optimize torch.nn.functional.one_hot | Silv3S | Land pytorch/pytorch#179831 (Silv3S) which adds at::kXPU to the device-type guards on lines 51-52 and 59-60 of Onehot.cpp so XPU follows th… | xinyu-intel | performance |
| [#1900](https://github.com/intel/torch-xpu-ops/issues/1900) | P2 | implement torch.linalg.qr xpu backend | pbielak | Unblock PR #2399 by: (1) tracking the oneMKL geqrf/orgqr perf request to completion, (2) landing the kernel even if perf is suboptimal with… | yao-matrix | module: op impl, bug_fix_stage3 |
| [#1902](https://github.com/intel/torch-xpu-ops/issues/1902) | P2 | implement torch.linalg.pinv xpu backend | mwiktor-intel | Land the pending implementation PR that registers linalg_pinv (and its _out variant) via oneMKL-backed SVD + matmul composition in third_pa… | yao-matrix | module: op impl, bug_fix_stage5 |
| [#1936](https://github.com/intel/torch-xpu-ops/issues/1936) | P2 | implement torch.linalg.cholesky xpu backend | mwiktor-intel | Implement at::linalg_cholesky_ex (the structured op that cholesky is lowered to) for XPU by wrapping oneMKL LAPACK potrf (via sycl/onemkl::… | jiqing-feng | module: op impl, bug_fix_stage5 |
| [#1973](https://github.com/intel/torch-xpu-ops/issues/1973) | P2 | AssertionError: Scalars or Tensor-likes are not e… | gplutop7 | Primary fix lives in oneDNN (MFDNN-14761): enable FP64 or chunked FP32 accumulation for the GPU matmul/conv primitives used by addmv and de… | mengfei25 | hw: PVC, module: ut, skipped, bug_fix_s… |
| [#2182](https://github.com/intel/torch-xpu-ops/issues/2182) | P2 | test_transform_bias_rescale_qkv_nested_xpu_float3… | SlawomirLaba, PawelSwide… | Use the correct max-seq-len index: replace `NestedTensor_get_max_size(...)[0]` with `NestedTensor_get_max_size(...)[1]` (or call native::Ne… | wincent8 | Accuracy, module: ut, skipped |
| [#2358](https://github.com/intel/torch-xpu-ops/issues/2358) | P2 | test/test_view_ops.py::TestOldViewOpsXPU::test_ra… | Silv3S | In third_party/torch-xpu-ops/yaml/native/native_functions.yaml, change the dispatch of _empty_affine_quantized (and _empty_per_channel_affi… | wincent8 | Ready for merge, ut_upstream, bug_fix_s… |
| [#2359](https://github.com/intel/torch-xpu-ops/issues/2359) | P2 | [upstream_ut] GradcheckError: Backward is not ree… | BBBela | Two options: (1) Add/update OpInfo entry for index_reduce in torch/testing/_internal/opinfo/definitions/_masked.py (or common_methods_invoc… | daisyden | skipped, ut_upstream |
| [#2412](https://github.com/intel/torch-xpu-ops/issues/2412) | P2 | Some NestedTensor missing XPU support | yucai-intel | For each listed site, generalize the CUDA device check to include XPU and dispatch per-device. Add XPU kernels under third_party/torch-xpu-… | daisyden | module: ut |
| [#2425](https://github.com/intel/torch-xpu-ops/issues/2425) | P2 | [upstream_ut] RuntimeError: Expected both self an… | BBBela | In aten/src/ATen/native/nested/NestedTensorBinaryOps.cpp:104 extend the guard to also accept XPU (e.g. `(self.is_cuda()\|\|self.is_xpu()) &… | daisyden | skipped, bug_fix_stage4 |
| [#2434](https://github.com/intel/torch-xpu-ops/issues/2434) | P2 | [Bug Skip]: New failures 2025-11-28 | AKloniecki | Reproduce the hypothesis failure locally with the seed printed by the test runner, compare XPU fake-quantize kernel output against the CPU … | mengfei25 | module: ut, skipped, bug_fix_stage4 |
| [#2615](https://github.com/intel/torch-xpu-ops/issues/2615) | P2 | [Bug Skip]: New failures RuntimeError: Unsupporte… | CuiYifeng | Extend the Half/ComplexHalf promotion wrapper in mkl/SpectralOps.cpp to cover all public fft entry points (_fft_c2c, _fft_r2c, _fft_c2r), i… | kaileiyx | module: ut, skipped |
| [#2722](https://github.com/intel/torch-xpu-ops/issues/2722) | P2 | [Bug Skip]: NotImplementedError: Could not run 'a… | Silv3S | Blocked on pytorch/pytorch#173923 (enable QuantizedXPU in torchgen). Once merged: (1) add QuantizedXPU dispatch entry for flip in aten/src/… | CuiYifeng | skipped, bug_fix_stage5 |
| [#2759](https://github.com/intel/torch-xpu-ops/issues/2759) | P2 | [Bug Skip]: New failed cases 2026-1-22 | AKloniecki | Two-step. (1) Immediate: gate the test so it does not run on XPU by replacing `@onlyCUDA` enforcement in the XPU test generator (third_part… | kaileiyx | skipped |
| [#2767](https://github.com/intel/torch-xpu-ops/issues/2767) | P2 | [UT] test_control_flow_xpu.py got AssertionError | PatrykWilczewski | Resync third_party/torch-xpu-ops/test/xpu/functorch/test_control_flow_xpu.py with the latest upstream pytorch/test/functorch/test_control_f… | libohao1201 | module: ut, skipped, bug_fix_stage5 |
| [#2815](https://github.com/intel/torch-xpu-ops/issues/2815) | P2 | RuntimeError: output with shape [2] doesn't match… | PawelSwider2000 | In addmm_complex_fallback (Blas.cpp:126-208), detect the addmv case (2-D mat2 collapsed from 1-D vec) or unify shapes prior to combining: s… | Silv3S | skipped, bug_fix_stage5 |
| [#3103](https://github.com/intel/torch-xpu-ops/issues/3103) | P2 | Tensor-likes are not equal for functorch and back… | BBBela | First confirm numerics by running conv3d backward on XPU vs CPU directly and measuring max diff; if within 1e-4-1e-3 relative, raise tolera… | BBBela | module: ut, skipped, random |
| [#3178](https://github.com/intel/torch-xpu-ops/issues/3178) | P2 | New failed test cases 2026-03-25 | pponikox | Land the fix already described by @pponikox: in test/xpu/functorch/test_vmap_xpu.py adjust expected backend/exception for EFFICIENT_ATTENTI… | BBBela | module: ut, skipped |
| [#3184](https://github.com/intel/torch-xpu-ops/issues/3184) | P2 | New failing UTs: test_cross_entropy_loss_2d_out_o… | wpietka | Update the XPU copy of the test in third_party/torch-xpu-ops/test/xpu/test_nn_xpu.py to also accept the XPU device-side assertion pattern (… | BBBela | module: ut, skipped |
| [#3194](https://github.com/intel/torch-xpu-ops/issues/3194) | P2 | Incorrect strides in TestCommonXPU,test_out_addmv… | AKloniecki | Trace the structured addmv.out impl on XPU: confirm whether meta::addmv's set_output_strided propagates through to_plain_output_strides in … | AKloniecki | skipped |
| [#3267](https://github.com/intel/torch-xpu-ops/issues/3267) | P2 | New failed test cases 2026-04-06 | PatrykWilczewski | For the fused Adam tests: either honor @onlyCUDA in the XPU test wrapper (filter out mixed-precision tests in test_optim_xpu.py) or, prefer… | zxd1997066 | module: ut, skipped |
| [#3349](https://github.com/intel/torch-xpu-ops/issues/3349) | P2 | [ai_generated] torch.native_batch_norm in eval mo… | Stonepia, chuanqi129, Co… | In BatchNormKernels.cpp:4157-4161 replace the resize/copy/calc_invstd logic with at::native::resize_output(save_mean, {0}) and resize_outpu… | laifenxiawucha |  |
| [#3396](https://github.com/intel/torch-xpu-ops/issues/3396) | P2 | ubind_copy cases failed due to upstream PR | Silv3S | Remove (or convert to skip when still broken) the xfail('unbind_copy') entries in test_ops_xpu.py and test_vmap_xpu.py, mirroring upstream … | daisyden | module: ut, skipped |
| [#1893](https://github.com/intel/torch-xpu-ops/issues/1893) | P3 | [upstream_ut] oneDNN accuracy issues in test_ops_… | chunhuanMeng | Bump tolerance for the remaining mv/addmv fp32 ops in XPU OpInfo (torch-xpu-ops test/xpu/xpu_test_utils.py or a toleranceOverride for mv/ad… | daisyden | skipped, ut_upstream |
| [#2132](https://github.com/intel/torch-xpu-ops/issues/2132) | P3 | [2.9][BMG-Windows][Torch-xpu-ops UT] 1 case faile… | pbielak | Add TestPoolingNNDeviceTypeXPU.test_pooling_large_xpu to the torch-xpu-ops skip list (test/xpu/skip_list_common.py or the pooling-specific … | libohao1201 | module: ut |
| [#2248](https://github.com/intel/torch-xpu-ops/issues/2248) | P3 | [upstream_ut] test_cow failures | gplutop7 | Two-part fix: (1) For ops where materialization is fundamentally required by oneDNN/oneMKL layout conversion (conv*, GEMM-family, cholesky*… | daisyden | skipped, bug_fix_stage3, ut_upstream |
| [#2250](https://github.com/intel/torch-xpu-ops/issues/2250) | P3 | Found mismatch when comparing the output of aten.… | jkosnox | Instrument the failing test and locate where bilinear backward inserts a non-view op on XPU (likely a dtype cast or contiguous call in _tri… | daisyden | skipped, bug_fix_stage3 |
| [#2287](https://github.com/intel/torch-xpu-ops/issues/2287) | P3 | [upstream_ut] test_python_ref issues | yucai-intel | Land upstream fix pytorch/pytorch PR #169565 (replace the bare aten.copy.default call in _refs.linspace around torch/_refs/__init__.py:5409… | daisyden | module: ut, ut_upstream |
| [#2446](https://github.com/intel/torch-xpu-ops/issues/2446) | P3 | [Bug Skip]: AssertionError: "Simulate error" does… | BBBela | Upstream PyTorch fix already proposed in pytorch/pytorch#179704 (making the test self-contained so prior reentrant errors do not leak). Pic… | kaileiyx | skipped, random |
| [#2530](https://github.com/intel/torch-xpu-ops/issues/2530) | P3 | Title: [upstream_ut] AssertionError: RuntimeError… | PatrykWilczewski | For test__int_mm_errors: either mark test as @onlyCUDA upstream (since XPU/oneDNN has no size-multiple-of-8/>16 requirement) or add matchin… | daisyden | skipped, bug_fix_stage5, port_from_skip… |
| [#2537](https://github.com/intel/torch-xpu-ops/issues/2537) | P3 | Title: [upstream_ut] Failed: Unexpected success | PatrykWilczewski | For each listed test, remove the obsolete XPU expected_failure entry from the xpu skip/xfail table (e.g. third_party/torch-xpu-ops/test/xpu… | daisyden | skipped, port_from_skiplist |
| [#2640](https://github.com/intel/torch-xpu-ops/issues/2640) | P3 | random issue test_vjpvjp_index_reduce_prod_xpu_fl… | wpietka | Either (a) relax tolerance / add @toleranceOverride for index_reduce prod in test/xpu/xpu_test_utils.py (or skiplist) as a short-term fix, … | daisyden | skipped, random |
| [#2779](https://github.com/intel/torch-xpu-ops/issues/2779) | P3 | Accuracy failures in logspace op | PawelSwider2000 | Change the step_type used for integer dispatch in logspace_kernel from float to double (or long double): in RangeFactoriesKernel.cpp line 2… | PawelSwider2000 | module: ut, skipped, bug_fix_stage5 |
| [#2817](https://github.com/intel/torch-xpu-ops/issues/2817) | P3 | Expected error message is different than actual | kdrozd-dev | Update test/test_linalg.py around line 607 so that only 'cuda' devices expect the CUDA-specific message, and XPU (which falls back to CPU-s… | Silv3S | skipped, bug_fix_stage5 |
| [#2837](https://github.com/intel/torch-xpu-ops/issues/2837) | P3 | Accuracy issue for Muon optimizer | DamJanusz | Loosen tolerances for this test on XPU via the toleranceOverride / tf32_on_and_off mechanism in test/test_optim.py (or add an xpu-specific … | kdrozd-dev | skipped, bug_fix_stage5 |
| [#3013](https://github.com/intel/torch-xpu-ops/issues/3013) | P3 | [upstream_ut] RuntimeError: Kernel is incompatibl… |  | Either (a) keep the index math in double but guard the kernel with `if (!syclDeviceHasFP64()) TORCH_CHECK(false, 'pdist needs fp64')` / pro… | Silv3S | bug_fix_stage5 |
| [#3014](https://github.com/intel/torch-xpu-ops/issues/3014) | P3 | [upstream_ut] AssertionError: False is not true |  | Switch pdist forward accumulation to `accscalar_t` (double) like CUDA's reference does, or widen tolerances via `assertTrue(torch.allclose(… | Silv3S | bug_fix_stage5 |
| [#3030](https://github.com/intel/torch-xpu-ops/issues/3030) | P3 | [Bug Skip] test/test_modules.py::TestModuleXPU::t… | gplutop7 | Short-term: skip/xfail this dtype entry in third_party/torch-xpu-ops/test/xpu/skip_list_common.py (or modules-specific skip list) with a to… | shangerxin | skipped |
| [#3033](https://github.com/intel/torch-xpu-ops/issues/3033) | P3 | [Bug Skip]: Softmax tolerance | chunhuanMeng | Short-term: the case is already listed in the XPU skip_list_common for extended op tests (PR merged on 2026-03-11); the 'random' label only… | chunhuanMeng | skipped, random |
| [#3296](https://github.com/intel/torch-xpu-ops/issues/3296) | P3 | accuracy gap of stft in float16 | EikanWang, Copilot | Either widen the stft fp16 tolerance for XPU in OpInfo (atol~5e-4, rtol~5e-3) similar to existing CUDA overrides, or skip stft fp16 in test… | daisyden | module: ut, ut_upstream |
| [#3329](https://github.com/intel/torch-xpu-ops/issues/3329) | P3 | [ai_generated] torch.cumprod on long float32 inpu… | Stonepia, Copilot | Switch the cumprod scan to use accscalar_t (double for fp32, float for half/bf16) for the internal accumulator, similar to CUDA's cumprod. … | laifenxiawucha | ai_generated |
| [#3330](https://github.com/intel/torch-xpu-ops/issues/3330) | P3 | [ai_generated] torch.std on large float32 input r… | Stonepia, Copilot | Promote the Welford accumulator type to double (or use Kahan/compensated accumulation) for fp32 inputs in ReduceMomentKernels.cpp, mirrorin… | laifenxiawucha | ai_generated |


<a id="sec-3-3-7-torch-runtime"></a>
#### 3.3.7 Torch Runtime  ·  7 issues

| Issue | Priority | Title | Owner | Fix Approach | Reporter | Labels |
|---|---|---|---|---|---|---|
| [#2349](https://github.com/intel/torch-xpu-ops/issues/2349) | P1 | [oneAPI][backward compatibility] libur_loader.so.… | riverliuintel | In the XPU wheel build/packaging (pytorch setup.py and torch-xpu-ops CMake install rules for libsycl/libur), convert RPATH to RUNPATH only … | dvrogozh |  |
| [#2263](https://github.com/intel/torch-xpu-ops/issues/2263) | P2 | [xpu][bug] XPU Trace event ends too late! | PawelSwider2000 | Follow PR pytorch/pytorch#172219: in XpuptiActivityProfilerSession::stop() (kineto plugin) clamp/receive the captureWindowEndTime_ value pa… | chuanqi129 | dependency component: community, module… |
| [#2712](https://github.com/intel/torch-xpu-ops/issues/2712) | P2 | [upstream_ut] RuntimeError: Cannot swap t2 becaus… | tszulist-hbn | In torch/_subclasses/fake_tensor.py / torch/_subclasses/meta_utils.py, extend the existing CUDA branch that disables weakref tracking durin… | daisyden | skipped, ut_upstream |
| [#2879](https://github.com/intel/torch-xpu-ops/issues/2879) | P2 | RuntimeError: _share_fd_: only available on CPU | DamJanusz | Fix in pytorch/torch/storage.py:395 and the parallel branch at line 409 in _new_shared: add 'xpu' to the device-type allow-list so that sha… | Silv3S | skipped, bug_fix_stage5 |
| [#3074](https://github.com/intel/torch-xpu-ops/issues/3074) | P2 | [Bug Skip] test_dlpack_exchange_api expect curren… | AKloniecki | Add a USE_XPU branch in CurrentWorkStream (torch/csrc/Module.cpp:786) that returns at::xpu::getCurrentXPUStream(device_id).queue() (the syc… | shangerxin | skipped |
| [#3077](https://github.com/intel/torch-xpu-ops/issues/3077) | P2 | [Bug Skip] test_dlpack.py::TestTorchDlPackXPU::te… | AKloniecki | Track upstream fix in pytorch/pytorch#173760 which teaches dlDeviceToTorchDevice to fall back to the device_id carried by the DLDevice when… | shangerxin | ut_upstream |
| [#2261](https://github.com/intel/torch-xpu-ops/issues/2261) | P3 | [xpu][profiler] Run with fork process has extra w… | moksiuc | No code change needed in torch-xpu-ops. Require PTI >= 0.15 (Deep Learning Essentials 2025.3.1) in the CI image and merge pytorch/pytorch#1… | chuanqi129 | dependency component: oneAPI, module: p… |


<a id="sec-3-4-implement"></a>
### 3.4 IMPLEMENT  ·  27 issues

**IMPLEMENT — new code / new PR must be written**

<a id="sec-3-4-1-distributed"></a>
#### 3.4.1 Distributed  ·  3 issues

| Issue | Priority | Title | Owner | Fix Approach | Reporter | Labels |
|---|---|---|---|---|---|---|
| [#2659](https://github.com/intel/torch-xpu-ops/issues/2659) | P1 | [distributed] test_dist2.py RuntimeError: Backend… | Chao1Han | In third_party/torch-xpu-ops/src/xccl/ProcessGroupXCCL.hpp, add an `override` right next to `getOptions()` mirroring ProcessGroupNCCL.hpp:7… | zxd1997066 | module: distributed |
| [#2738](https://github.com/intel/torch-xpu-ops/issues/2738) | P1 | [distributed] test_c10d_spawn_nccl.py ValueError:… | jenniew | Reproduce with latest wheel (artifact 24487284020). Inspect third_party/torch-xpu-ops' ProcessGroupXCCL.cpp _reduce_scatter_base size check… | zxd1997066 | bug, module: distributed |
| [#2004](https://github.com/intel/torch-xpu-ops/issues/2004) | P3 | [distributed][shared_tensor] test\distributed\_sh… | libohao1201 | Add these 12 sharded_tensor test cases to the XPU skip/exclude list (e.g., third_party/torch-xpu-ops skip_list_common.py or the distributed… | libohao1201 | bug, module: distributed |


<a id="sec-3-4-2-flash-attention"></a>
#### 3.4.2 Flash Attention  ·  1 issues

| Issue | Priority | Title | Owner | Fix Approach | Reporter | Labels |
|---|---|---|---|---|---|---|
| [#2693](https://github.com/intel/torch-xpu-ops/issues/2693) | P3 | Title: [upstream_ut] AssertionError: Scalars are … | hoshibara | Skip the test on XPU: either add an @skipIf(TEST_XPU, ...) decorator in test_cuda_repro.py (upstream) or add the case to the torch-xpu-ops … | daisyden | module: inductor, skipped, ut_upstream |


<a id="sec-3-4-3-inductor"></a>
#### 3.4.3 Inductor  ·  3 issues

| Issue | Priority | Title | Owner | Fix Approach | Reporter | Labels |
|---|---|---|---|---|---|---|
| [#2605](https://github.com/intel/torch-xpu-ops/issues/2605) | P2 | [int4][inductor] Add freezing pattern for fusing … |  | Mirror PR #170341 for XPU: register an analogous freezing pattern in torch/_inductor/fx_passes (e.g. a new xpu variant alongside the cuda o… | liangan1 |  |
| [#3095](https://github.com/intel/torch-xpu-ops/issues/3095) | P3 | cutlass support blocks some unit test cases | Triage | Skip these CUDA-cutlass-only tests on XPU via @skipIfXPU (or gate with HAS_CUTLASS) in test/inductor/test_cudacodecache.py. Longer term, if… | daisyden | module: inductor, ut_upstream |
| [#3334](https://github.com/intel/torch-xpu-ops/issues/3334) | P3 | [upstream_ut] test_repros.py ReproTests.test_part… | Triage | Wait for / track resolution of pytorch/pytorch#174370 and re-bisect against XPU once it lands; in the meantime keep test in the upstream sk… | shangerxin | skipped, ut_upstream |


<a id="sec-3-4-4-others"></a>
#### 3.4.4 Others  ·  5 issues

| Issue | Priority | Title | Owner | Fix Approach | Reporter | Labels |
|---|---|---|---|---|---|---|
| [#2086](https://github.com/intel/torch-xpu-ops/issues/2086) | P3 | nd_item::barrier has been deprecated | dvrogozh | Do a codebase sweep (grep for 'item.barrier(' / 'nd_item<...>::barrier' in src/**/*.h and *.cpp) and replace each occurrence with sycl::gro… | EikanWang | enhancement |
| [#3024](https://github.com/intel/torch-xpu-ops/issues/3024) | P3 | Enable clang-tidy checks | Silv3S | Work the checklist in the issue one check at a time: (1) run lintrunner init && lintrunner --take CLANGTIDY -a over src/ to surface violati… | Silv3S | enhancement |
| [#3196](https://github.com/intel/torch-xpu-ops/issues/3196) | P3 | vitals is not supported, the cases should be disa… | libohao1201 | Add the three TestBasicVitalSigns cases to the dynamic_skip list (or annotate with skipIfXPU) in test/xpu/skip_list or test_torch_xpu.py, t… | daisyden | skipped |
| [#3300](https://github.com/intel/torch-xpu-ops/issues/3300) | P3 | [CI] When creating PR, several pull workflows are… | mengfei25 | Refine the pull-workflow concurrency group to include the label set (or debounce via on: pull_request types: [opened, labeled, synchronize]… | BBBela |  |
| [#3345](https://github.com/intel/torch-xpu-ops/issues/3345) | P3 | Setup Self-hosted runner for Copilot auto debuggi… | chuanqi129 | Add .github/workflows/copilot-setup-steps.yml as drafted in the issue body, targeting the existing self-hosted runner label `pvc_rolling`, … | Stonepia |  |


<a id="sec-3-4-5-torch-operations"></a>
#### 3.4.5 Torch Operations  ·  7 issues

| Issue | Priority | Title | Owner | Fix Approach | Reporter | Labels |
|---|---|---|---|---|---|---|
| [#2560](https://github.com/intel/torch-xpu-ops/issues/2560) | P1 | [UT] "RuntimeError: iter.device(arg).is_xpu()" in… | CuiYifeng | Mirror CUDA's gpu_kernel support for CPU scalars: before calling gpu_kernel in addcmul_kernel, detect iter.is_cpu_scalar(3) (and/or 2), ext… | libohao1201 | bug |
| [#2244](https://github.com/intel/torch-xpu-ops/issues/2244) | P2 | test/test_sparse_csr.py::TestSparseCSRXPU::test_b… | jenniew | Add the missing BSR-result branch in addmm_out_sparse_csr: compute the dense product via addmm_calculation, convert back with result = resu… | wincent8 | module: ut, skipped |
| [#2245](https://github.com/intel/torch-xpu-ops/issues/2245) | P3 | oneDNN matmul received incorrect shape in test/te… | CuiYifeng | Invoke the standard CSR validation (at::_validate_sparse_csr_tensor_args or the same crow_indices/col_indices checks used on CUDA) at the t… | wincent8 | module: ut, skipped |
| [#3041](https://github.com/intel/torch-xpu-ops/issues/3041) | P3 | AssertionError: Expected len(flat_diff_results) >… | Silv3S, BartoszKokoszko | In torch/testing/_internal/common_methods_invocations.py, extend the existing CUDA DecorateInfo on OpInfo('normal', variant=tensor_second) … | daisyden | ut_upstream |
| [#3089](https://github.com/intel/torch-xpu-ops/issues/3089) | P3 | AssertionError: Torch not compiled with CUDA enab… | jmamzax | Add test_max_pool2d_cudnn to the torch-xpu-ops skip list (test/xpu/skip_list_common.py or the quantization-specific skip file) with reason … | jmamzax | bug_fix_stage5 |
| [#3121](https://github.com/intel/torch-xpu-ops/issues/3121) | P3 | [Bug Skip]: CUDA specific UT test_fft_half_and_ch… | Triage | Make the error regex device-aware in test_spectral_ops.py test_fft_half_and_chalf_not_power_of_two_error (e.g. 'cuFFT\|MKL\|powers? of two'… | CuiYifeng | skipped |
| [#3390](https://github.com/intel/torch-xpu-ops/issues/3390) | P3 | Clarification requested on mixed non-atomic load … | Triage | Replace the initial `*address_as_ui` read with `target.load()` (using the same sycl::atomic_ref with relaxed order) so all accesses go thro… | tonghaining |  |


<a id="sec-3-4-6-torch-runtime"></a>
#### 3.4.6 Torch Runtime  ·  4 issues

| Issue | Priority | Title | Owner | Fix Approach | Reporter | Labels |
|---|---|---|---|---|---|---|
| [#1171](https://github.com/intel/torch-xpu-ops/issues/1171) | P2 | LNL Windows got unexpected error message | xuhancn, chunhuanMeng | Confirm pytorch#167960 is included in torch-xpu-ops test expectations (loosen error_msg match on Windows) and close the issue; continue to … | daisyden | client, os: Windows, hw : LNL, hw: BMG,… |
| [#1986](https://github.com/intel/torch-xpu-ops/issues/1986) | P3 | torch.xpu._sleep is missing, | guangyey | Implement a _sleep SYCL kernel in third_party/torch-xpu-ops (analogue of aten/src/ATen/cuda/Sleep.cu), expose it as torch._C._xpu_sleep via… | githubsgi | dependency component: oneAPI |
| [#2680](https://github.com/intel/torch-xpu-ops/issues/2680) | P3 | XPU Autocast does not support fp32 dtypes | CuiYifeng | Add an XPU branch parallel to the CUDA one in torch/amp/autocast_mode.py around lines 283-297 (or extend device_supported_dtypes for 'xpu' … | kaixuanliu |  |
| [#3086](https://github.com/intel/torch-xpu-ops/issues/3086) | P3 | nvml support blocks some test cases | Triage | These tests are CUDA-specific and should be excluded on XPU (add to the skip list in third_party/torch-xpu-ops/test/xpu/skip_list_common.py… | daisyden | module: ut |


<a id="sec-3-4-7-torchao"></a>
#### 3.4.7 TorchAO  ·  4 issues

| Issue | Priority | Title | Owner | Fix Approach | Reporter | Labels |
|---|---|---|---|---|---|---|
| [#3060](https://github.com/intel/torch-xpu-ops/issues/3060) | P1 | Implement torch._scaled_grouped_mm for xpu backend | Stonepia, liangan1 | Implement _scaled_grouped_mm for XPU. Add src/ATen/native/xpu/ScaledGroupedMM.cpp with a registration for the op, and a SYCL kernel src/ATe… | kgajdamo | module: quant |
| [#2707](https://github.com/intel/torch-xpu-ops/issues/2707) | P3 | [TorchAO][BMG] INT4 GPTQ failed due to TorchAO AP… | xiaowangintel | Short-term: disable the INT4 GPTQ workload in the XPU weekly CI (TorchAO_LLM_XPU_Weekly) until torchao upstream stabilizes GPTQ, per xiaowa… | LifengWang | module: ao |
| [#2948](https://github.com/intel/torch-xpu-ops/issues/2948) | P3 | [AO] Benchmark enabling on XPU | Triage | Port the TorchAO benchmark driver to accept `xpu` as a device (parameterize device string, replace torch.cuda.* with torch.accelerator or t… | liangan1 | module: ao, bug_fix_stage6 |
| [#3032](https://github.com/intel/torch-xpu-ops/issues/3032) | P3 | [TorchAO][UT] failures in test/prototype/safetens… | Stonepia | Land the proposed skip PR pytorch/ao#4049 that parametrizes/skips the unsupported configs when CUDA (or required SM/library) is unavailable… | zxd1997066 | module: ao |


<a id="sec-3-5-retriage-prs"></a>
### 3.5 RETRIAGE_PRS  ·  12 issues

**RETRIAGE_PRS — prior PRs dead or cross-refs unverified; re-evaluate path**

<a id="sec-3-5-1-inductor"></a>
#### 3.5.1 Inductor  ·  3 issues

| Issue | Priority | Title | Owner | Fix Approach | Reporter | Labels |
|---|---|---|---|---|---|---|
| [#2694](https://github.com/intel/torch-xpu-ops/issues/2694) | P2 | Title: [upstream_ut] AssertionError: Tensor-likes… | daisyden | Land upstream PR pytorch/pytorch#171773 which aligns the XPU cpp_wrapper randint path with the eager path (seed/offset + argument marshalli… | daisyden | module: inductor, skipped, ut_upstream |
| [#2806](https://github.com/intel/torch-xpu-ops/issues/2806) | P2 | CompiledAOTI need XPU support | daisyden | Ensure torch._inductor.output_code.CompiledAOTI.__post_init__ handles `device_type.startswith('xpu')` by instantiating AOTIModelContainerRu… | daisyden | module: inductor, ut_upstream |
| [#2958](https://github.com/intel/torch-xpu-ops/issues/2958) | P3 | AssertionError of test_dtensor_basic_compile | daisyden | Verify on latest pytorch main that the test passes (etaf report), then remove the skip entry from third_party/torch-xpu-ops/test/xpu/skip_l… | daisyden | module: inductor, ut_upstream |


<a id="sec-3-5-2-sparse"></a>
#### 3.5.2 Sparse  ·  2 issues

| Issue | Priority | Title | Owner | Fix Approach | Reporter | Labels |
|---|---|---|---|---|---|---|
| [#2663](https://github.com/intel/torch-xpu-ops/issues/2663) | P3 | test_sparse_semi_structured.py gaps | Triage | Short-term: explicitly skip or xfail the whole test file for XPU to make the gap visible. Long-term: either (a) register a CPU-fallback / o… | wincent8 | module: ut, ut_upstream |
| [#3176](https://github.com/intel/torch-xpu-ops/issues/3176) | P3 | [Bug Skip]: ValueError: _scaled_dot_product_atten… |  | No code change needed — issue already resolved by the upstream change broadening check_device to accept xpu. Remove the skip entries from t… | CuiYifeng | skipped |


<a id="sec-3-5-3-torch-operations"></a>
#### 3.5.3 Torch Operations  ·  7 issues

| Issue | Priority | Title | Owner | Fix Approach | Reporter | Labels |
|---|---|---|---|---|---|---|
| [#2439](https://github.com/intel/torch-xpu-ops/issues/2439) | P2 | [oneDNN] TestDecompXPU.test_quick_addmv_xpu_float… | libohao1201 | Adopt the upstream tolerance fix proposed for addmv decomp cross-reference in pytorch/pytorch#174590 (add addmv to the per-op tolerance ove… | mengfei25 | dependency component: oneDNN, module: ut |
| [#2783](https://github.com/intel/torch-xpu-ops/issues/2783) | P2 | [Bug Skip]: Key "xpu" is missing from dict "drive… | daisyden | Upstream PR to test/test_linalg.py: add an 'xpu': (None,) entry to the drivers dict (and similarly in any other svd/lstsq helper that keys … | CuiYifeng | module: ut, skipped, dependency compone… |
| [#2618](https://github.com/intel/torch-xpu-ops/issues/2618) | P3 | [Bug Skip]: [regression] AssertionError: Scalars … | jmamzax | Since upstream pytorch is unlikely to bump numpy on python 3.10 soon, apply the workaround in torch-xpu-ops CI workflow: detect via `python… | kaileiyx | skipped, bug_fix_stage5 |
| [#2798](https://github.com/intel/torch-xpu-ops/issues/2798) | P3 | Test case test/test_dlpack.py::TestTorchDlPackCPU… | Triage | Either (a) extend the DLPack from_dlpack import logic so that when a target device argument is supplied and the capsule is on CPU, the tens… | shangerxin | ut_upstream |
| [#2993](https://github.com/intel/torch-xpu-ops/issues/2993) | P3 | [Bug Skip]: Unexpected success of test_cpu_gpu_pa… | gplutop7 | Grep skip_list_common.py / skip_list_*.py and xpu_test_utils.py _xpu_skip / xfail lists for 'test_cpu_gpu_parity_nn_ConvTranspose3d_xpu_com… | CuiYifeng | skipped |
| [#3011](https://github.com/intel/torch-xpu-ops/issues/3011) | P3 | [upstream_ut] torch.OutOfMemoryError: XPU out of … |  | Add an XPU-specific memory gate: replace the CUDA decorator with `@largeTensorTest('5GB', device='xpu')` (or dual-gate), which reads `torch… | Silv3S | bug_fix_stage5 |
| [#3025](https://github.com/intel/torch-xpu-ops/issues/3025) | P3 | New failing test in Nightly Wheel test_decomp_xpu… | Triage | Regenerate the expected list with EXPECTTEST_ACCEPT=1 python test/xpu/test_decomp_xpu.py HasDecompTest.test_has_decomposition (and test_ate… | BBBela | skipped, random |


<a id="sec-3-6-root-cause"></a>
### 3.6 ROOT_CAUSE  ·  10 issues

**ROOT_CAUSE — owner actively debugging this specific failure**

<a id="sec-3-6-1-distributed"></a>
#### 3.6.1 Distributed  ·  1 issues

| Issue | Priority | Title | Owner | Fix Approach | Reporter | Labels |
|---|---|---|---|---|---|---|
| [#3010](https://github.com/intel/torch-xpu-ops/issues/3010) | P1 | [distributed][tensor] test_random_ops.py torch._d… | jenniew | Avoid eagerly reading device RNG state inside traced regions: either (a) move `_compute_rng_offsets` / `_get_device_state` out of the trace… | zxd1997066 | bug, module: distributed |


<a id="sec-3-6-2-inductor"></a>
#### 3.6.2 Inductor  ·  2 issues

| Issue | Priority | Title | Owner | Fix Approach | Reporter | Labels |
|---|---|---|---|---|---|---|
| [#2619](https://github.com/intel/torch-xpu-ops/issues/2619) | P1 | [release/2.10] Some models inductor performance d… | mengfei25 | Cherry-pick pytorch PR #169257 (bf16 atomic_add fallback) and the reduction-heuristics fix onto release/2.10; for the cait_m36_384 fusion r… | mengfei25 | E2E, performance, regression |
| [#3058](https://github.com/intel/torch-xpu-ops/issues/3058) | P1 | [E2E] hf_GPT2_large amp_fp16/amp_bf16 training go… | weishi-deng | In torch/_inductor/fx_passes/post_grad.py:1570, make the keep_addmm_fused_for_half_dtypes guard device-aware (skip it for XPU) or gate it o… | kaileiyx | E2E, hw: PVC |


<a id="sec-3-6-3-sparse"></a>
#### 3.6.3 Sparse  ·  1 issues

| Issue | Priority | Title | Owner | Fix Approach | Reporter | Labels |
|---|---|---|---|---|---|---|
| [#3165](https://github.com/intel/torch-xpu-ops/issues/3165) | P1 | test_sparse_csr_xpu.py::TestSparseCompressedTrito… | jafraustro | File/track a pytorch-triton-xpu issue with the failing kernel reproducer and, once fixed upstream, bump the triton pin. In the meantime, ga… | CuiYifeng | skipped, ut_upstream |


<a id="sec-3-6-4-torch-operations"></a>
#### 3.6.4 Torch Operations  ·  4 issues

| Issue | Priority | Title | Owner | Fix Approach | Reporter | Labels |
|---|---|---|---|---|---|---|
| [#3216](https://github.com/intel/torch-xpu-ops/issues/3216) | P1 | [OPs] Some ops of XPU have non-determinism and ar… | CuiYifeng | Wire torch.use_deterministic_algorithms through XPU matmul/conv: pin oneDNN primitive attributes (dnnl::fpmath_mode, scratchpad mode, deter… | YangKai0616 |  |
| [#2436](https://github.com/intel/torch-xpu-ops/issues/2436) | P3 | [upstream_ut] AttributeError: 'NoneType' object h… | daisyden | Wait for the upstream fix in pytorch/pytorch#97395 (guard None .grad in _expanded_weights clone path). In the meantime redesign the XPU tes… | daisyden | skipped, dependency component: communit… |
| [#2531](https://github.com/intel/torch-xpu-ops/issues/2531) | P3 | [upstream_ut] AssertionError: Torch not compiled … | daisyden | Triage per test: (a) genuinely CUDA-specific (cufft_plan_cache, ctc_loss_cudnn_*, numeric_check_leak_tunableop_rocm, gemm_bias_offline_tuna… | daisyden | skipped, port_from_skiplist |
| [#2533](https://github.com/intel/torch-xpu-ops/issues/2533) | P3 | Title: [upstream_ut] AttributeError: 'TestQuantiz… | astachowiczhabana | Skip test_qsoftmax_qnnpack_xpu in third_party/torch-xpu-ops/test/xpu/skip_list_common.py (quantization section) with reason 'qnnpack is CPU… | daisyden | skipped, port_from_skiplist |


<a id="sec-3-6-5-torch-runtime"></a>
#### 3.6.5 Torch Runtime  ·  2 issues

| Issue | Priority | Title | Owner | Fix Approach | Reporter | Labels |
|---|---|---|---|---|---|---|
| [#2392](https://github.com/intel/torch-xpu-ops/issues/2392) | P3 | [Bug Skip]: torch.OutOfMemoryError: XPU out of me… | xuhancn | Guard the test with a minimum-memory check (e.g., require device total_memory >= 24 GB) or scale application_memory down when the allowed b… | RUIJIEZHONG66166 | skipped_windows |
| [#2513](https://github.com/intel/torch-xpu-ops/issues/2513) | P3 | [upstream_ut] RuntimeError: _share_fd_: only avai… | gplutop7 | Either (a) teach share_memory_ to move storage to CPU transparently for non-CPU devices (matches CUDA behaviour, which also errors), or mor… | libohao1201 | skipped |



<a id="sec-4"></a>
## 4. QA

Issues in this section are ready for QA action (close, verify, reply, etc.). Rows sorted by `Priority` (P0 → P3).

<a id="sec-4-1-close"></a>
### 4.1 CLOSE  ·  18 issues

**CLOSE — terminal close (CI passing, duplicate, or confirmed gap acceptable)**

| Issue | Priority | Title | Owner | Fix Approach | Reporter | Labels |
|---|---|---|---|---|---|---|
| [#2496](https://github.com/intel/torch-xpu-ops/issues/2496) | P1 | [upstream_ut] Segmentation fault when running tes… | astachowiczhabana | Reproduce locally with pytest --forked and compute-runtime debug (NEO_ENABLE_DEVICE_ENQUEUE / ZE_DEBUG) to obtain a kernel-level fault addr… | libohao1201 | skipped |
| [#2907](https://github.com/intel/torch-xpu-ops/issues/2907) | P1 | [release/2.11] Models performance regression for … | xuhancn | Bisect torch and torch-xpu-ops commits between release/2.10 and release/2.11 on Windows using the DistilBertForQuestionAnswering fp16 case … | bjarzemb | os: Windows |
| [#1624](https://github.com/intel/torch-xpu-ops/issues/1624) | P2 | [DONT CLOSE] Known UT Issue list | RUIJIEZHONG66166 | Keep as tracking issue ([DONT CLOSE]); periodically reconcile the list with the individual child issues and with test/xpu/skip_list_dist.py… | RUIJIEZHONG66166 | module: distributed, skipped, module: i… |
| [#2022](https://github.com/intel/torch-xpu-ops/issues/2022) | P2 | [Windows] [CI] [UT] AssertionError: Tensor-likes … | RUIJIEZHONG66166 | Reproduce on a Windows DGM box with XPU_VERBOSE; compare per-workgroup partial sums vs CPU. Enforce higher-precision accumulator (float for… | RUIJIEZHONG66166 | os: Windows, module: ut, skipped_windows |
| [#2541](https://github.com/intel/torch-xpu-ops/issues/2541) | P2 | Title: [upstream_ut] RuntimeError: could not cons… | yucai-intel | Land pytorch/pytorch#176875 (adjusts the stride handling before constructing the dnnl::memory::desc — e.g., falls back to a contiguous desc… | daisyden | skipped, port_from_skiplist |
| [#2720](https://github.com/intel/torch-xpu-ops/issues/2720) | P2 | [upstream_ut] RuntimeError: false INTERNAL ASSERT… | CuiYifeng | Align third_party/torch-xpu-ops/src/ATen/native/xpu/sycl/BinaryMiscOpsKernels.cpp:158 ldexp_kernel with the CUDA implementation (aten/src/A… | wincent8 | skipped |
| [#2811](https://github.com/intel/torch-xpu-ops/issues/2811) | P2 | [Bug Skip]: [Regression] failed cases 2026-2-2 | jmamzax | For the remaining failing test, add a CompositeImplicitAutograd / decomposition for _unsafe_masked_index_put_accumulate on XPU or extend th… | kaileiyx | skipped, bug_fix_stage5 |
| [#3158](https://github.com/intel/torch-xpu-ops/issues/3158) | P2 | AttributeError: module 'triton.compiler' has no a… | tadkrawiec | Replace 'triton.compiler.OutOfResources' with the compat import at third_party/torch-xpu-ops/test/xpu/test_sparse_csr_xpu.py:5274, e.g. 'fr… | kdrozd-dev |  |
| [#3160](https://github.com/intel/torch-xpu-ops/issues/3160) | P2 | compiler not found (Windows) | tadkrawiec | Update Windows CI provisioning (runner.yml / setup scripts under .github/ci_commit_pins or torch-xpu-ops CI config) to expose MSVC cl.exe o… | kdrozd-dev | os: Windows |
| [#3174](https://github.com/intel/torch-xpu-ops/issues/3174) | P2 | [Bug Skip]: Accuracy failure of test_Conv2d_group… | pbielak | Reproduce with the failing shapes and capture oneDNN verbose; raise a oneDNN ticket with the grouped-conv primitive descriptor. Short-term:… | CuiYifeng | dependency component: oneDNN, module: u… |
| [#3346](https://github.com/intel/torch-xpu-ops/issues/3346) | P2 | [PVC] Accuracy issue in Conv2d_naive_groups for f… | Silv3S | Keep the test skipped on PVC fp16 and file an upstream oneDNN ticket with the failing config (groups, shapes, dtype, PVC arch) referencing … | Silv3S | Accuracy, dtype: float16, hw: PVC, depe… |
| [#1818](https://github.com/intel/torch-xpu-ops/issues/1818) | P3 | [BMG-Windows][PT2.8]Torch-xpu-ops UT got accuracy… | kdrozd-dev | Merge #3072 to resolve the complex128 prod cases. For GroupNorm, either (a) raise to fp64 accumulators in the welford two-pass reduction in… | libohao1201 | os: Windows, hw: BMG, bug_fix_stage5 |
| [#2230](https://github.com/intel/torch-xpu-ops/issues/2230) | P3 | test_sparse_csr.py::TestSparseCompressedTritonKer… | tszulist-hbn | Either (a) port the sampled_addmm Triton kernel to support XPU multi-device dispatch similar to the CUDA path, or (b) add XPU skips/xfails … | wincent8 | skipped |
| [#2251](https://github.com/intel/torch-xpu-ops/issues/2251) | P3 | [upstream_ut] test_fake_autocase got Exception: D… | astachowiczhabana | Add linalg_pinv and pinverse (and their _out/_ex variants) to the XPU fp32 autocast promote list in third_party/torch-xpu-ops/src/ATen/auto… | daisyden | duplicate, module: ut, skipped, ut_upst… |
| [#2518](https://github.com/intel/torch-xpu-ops/issues/2518) | P3 | [upstream_ut] TypeError: Creating a Tensor subcla… | astachowiczhabana | Inspect torch/csrc/autograd/python_variable.cpp:THPVariable_as_subclass (and _make_subclass) to ensure XPU tensors go through the same subc… | libohao1201 | skipped |
| [#3161](https://github.com/intel/torch-xpu-ops/issues/3161) | P3 | Exception: Tensor-likes are not close! - test_vjp… | tadkrawiec | Add an XPU-specific tolerance override (toleranceOverride/Windows skip) for linalg_tensorsolve float32 VJP in third_party/torch-xpu-ops/tes… | kdrozd-dev | os: Windows |
| [#3280](https://github.com/intel/torch-xpu-ops/issues/3280) | P3 | [Bug Skip]: New UT failure in 0406 nightly window… | RUIJIEZHONG66166 | Add the five addcmul int cases to test/xpu/extended/skip_list_win.py to unblock Windows CI, then file the failure against Windows compute-r… | RUIJIEZHONG66166 | skipped_windows |
| [#3410](https://github.com/intel/torch-xpu-ops/issues/3410) | P3 | [Bug Skip]: test_non_contiguous_tensors_nn_Conv3d… | chunhuanMeng | Add a tolerance override entry for 'nn.Conv3d' -> ('TestModule'/'TestModuleXPU', 'test_non_contiguous_tensors') in torch-xpu-ops/test/xpu/x… | chunhuanMeng | skipped |


<a id="sec-4-2-verify-and-close"></a>
### 4.2 VERIFY_AND_CLOSE  ·  29 issues

**VERIFY_AND_CLOSE — fix merged; validate then close**

| Issue | Priority | Title | Owner | Fix Approach | Reporter | Labels |
|---|---|---|---|---|---|---|
| [#2219](https://github.com/intel/torch-xpu-ops/issues/2219) | P1 | float8_e4m3fn precision overflow | CuiYifeng, yucai-intel | Extend the CastScalarFunc specialization in CopyKernel.cpp to cover all source dtypes (float, BFloat16) for Float8_e4m3fn/e5m2 and perform … | jiqing-feng |  |
| [#2981](https://github.com/intel/torch-xpu-ops/issues/2981) | P1 | [release/2.11] T5 models performance dropped ~20% | jianyizh, weishi-deng | Cherry-pick upstream fix pytorch/pytorch#169257 into release/2.11 to restore pattern matcher ordering so view-removal runs after the fused … | mengfei25 | performance, os: Ubuntu, hw: BMG, depen… |
| [#2984](https://github.com/intel/torch-xpu-ops/issues/2984) | P1 | [release/2.11] sebotnet33ts_256 fp32 training got… | mengfei25 | Revisit torch-xpu-ops PR #2462: gate the switch from sycl::native::exp to std::exp behind a numerics flag, or restore native_exp for the si… | mengfei25 | os: Ubuntu, hw: BMG |
| [#3021](https://github.com/intel/torch-xpu-ops/issues/3021) | P1 | [distributed] all_to_all_single Compatibility Iss… | zhangxiaoli73 | (1) In alltoall_base (ProcessGroupXCCL.cpp:1977), when both inputSplitSizes and outputSplitSizes are empty, call a plain onecclAllToAll/one… | xiangyuT | module: distributed |
| [#3022](https://github.com/intel/torch-xpu-ops/issues/3022) | P1 | [distributed] batch_isend_irecv Compatibility Iss… | zhangxiaoli73 | Two parts: (1) audit ProcessGroupXCCL::send/recv (src/xccl/ProcessGroupXCCL.cpp:981 and :1025) and pointToPoint so that when coalescing_sta… | xiangyuT | module: distributed |
| [#3143](https://github.com/intel/torch-xpu-ops/issues/3143) | P1 | NotImplementedError: The operator 'aten::_scaled_… | daisyden | Already addressed by intel/torch-xpu-ops PR #3317 which adds the XPU registration for aten::_scaled_dot_product_efficient_attention_backwar… | daisyden | module: ut, skipped, ut_upstream |
| [#3258](https://github.com/intel/torch-xpu-ops/issues/3258) | P1 | huggingface accuracy inference Error in op: torch… | bjarzemb | No action needed - already fixed upstream by pytorch/pytorch#178986 / #179239 which aligned the meta kernel of _scaled_dot_product_fused_at… | bjarzemb |  |
| [#2220](https://github.com/intel/torch-xpu-ops/issues/2220) | P2 | test/test_sparse_csr.py::TestSparseCompressedTrit… |  | Dependency bug: bump/pin intel-graphics-compiler or upgrade triton-xpu to a build whose SPIR-V consumer accepts SPV_INTEL_subgroup_matrix_m… | wincent8 | duplicate, module: dependency bug, depe… |
| [#2701](https://github.com/intel/torch-xpu-ops/issues/2701) | P2 | [distributed] Barrier Timeout Error with test_dis… | syedshahbaaz | Add monitoredBarrier in ProcessGroupXCCL that performs point-to-point ack rounds (using existing ccl::send/recv at xccl.cpp:210/238) with p… | madhumitha0102 | bug, module: distributed |
| [#2702](https://github.com/intel/torch-xpu-ops/issues/2702) | P2 | [distributed] RuntimeError: Work ran time out aft… | syedshahbaaz | Implement ProcessGroupXCCL::monitoredBarrier (mirroring ProcessGroupNCCL::monitoredBarrier: per-rank send/recv based handshake with explici… | madhumitha0102 | bug, module: distributed |
| [#2919](https://github.com/intel/torch-xpu-ops/issues/2919) | P2 | [XPU][upstream_ut][COW] Fix materialization in re… | gplutop7 | For each remaining op family, audit the XPU kernel path and replace mutable .data_ptr()/.mutable_data_ptr() access on input tensors with co… | gplutop7 | bug_fix_stage3, ut_upstream |
| [#2950](https://github.com/intel/torch-xpu-ops/issues/2950) | P2 | SYCL compilation flag -fsycl-id-queries-fit-in-in… | BBBela | Two-part fix: (1) in TriangularOpsKernels.cpp make the index computation overflow-safe by casting `item.get_global_linear_id()` (or `get_gl… | BBBela |  |
| [#2966](https://github.com/intel/torch-xpu-ops/issues/2966) | P2 | [Bug Skip]: [Regression]2026-3-2 ut failures | jmamzax | CI infra: fix dut7901 so all 8 Max-1100 GPUs enumerate reliably (driver/udev) and keep the corrected --tx popen ZE_AFFINITY_MASK=0..7 confi… | kaileiyx | skipped, bug_fix_stage5, random |
| [#3141](https://github.com/intel/torch-xpu-ops/issues/3141) | P2 | [upstream_ut] RuntimeError: FlashAttentionForward… | LuFinch | Two-layer fix: (1) at the dispatcher in aten/native/transformers/xpu choose the math / mem-efficient SDPA backend when headdim is outside t… | daisyden | module: ut, skipped, ut_upstream |
| [#3163](https://github.com/intel/torch-xpu-ops/issues/3163) | P2 | [Bug Skip]: Object comparison failed: torch.int64… | chunhuanMeng | In add_out_sparse_compressed_xpu, after constructing out_dense.to_sparse_csr(), cast crow_indices and col_indices to the index dtype of the… | CuiYifeng | skipped, ut_upstream |
| [#3167](https://github.com/intel/torch-xpu-ops/issues/3167) | P2 | NotImplementedError: Could not run 'aten::triangu… | tszulist-hbn | Implement triangular_solve_out_sparse_csr_xpu in third_party/torch-xpu-ops/src/ATen/native/sparse/xpu/ backed by oneMKL sparse trsv/trsm (t… | CuiYifeng | skipped, ut_upstream |
| [#1645](https://github.com/intel/torch-xpu-ops/issues/1645) | P3 | [For Comparison] Save reference comparison run id | mengfei25 | No code change. The CICD owner (mengfei25) keeps the run-id list current for downstream comparison workflows; close when BMG entries are fu… | mengfei25 | module: infra |
| [#1689](https://github.com/intel/torch-xpu-ops/issues/1689) | P3 | [For op Perf Comparison] Save reference compariso… | RUIJIEZHONG66166 | No code action required. Owners should periodically update the reference run id in the issue body as perf baselines are refreshed and close… | RUIJIEZHONG66166 | module: infra |
| [#1778](https://github.com/intel/torch-xpu-ops/issues/1778) | P3 | [Infra] Show known issues for accuracy test | mengfei25 | Keep as meta/tracking. Wire the listed (suite, dtype, mode, model) tuples into the E2E accuracy skip/known-issues table so the dashboard ma… | mengfei25 | E2E, Accuracy, skipped, module: infra |
| [#1969](https://github.com/intel/torch-xpu-ops/issues/1969) | P3 | torch._dynamo.exc.InternalTorchDynamoError: TypeE… | guangyey | Verify on latest main that CtxManagerTests.test_gpu_event_across_graph_break either passes or remains appropriately skipped for xpu, remove… | shangerxin | module: ut |
| [#2169](https://github.com/intel/torch-xpu-ops/issues/2169) | P3 | Frame size comparison failed in test_size_compari… | guangyey | Re-run test_size_comparison_no_recompile on current main with a fresh XPU wheel to confirm the fix, then remove the TEST_XPU skip decorator… | daisyden | skipped |
| [#2215](https://github.com/intel/torch-xpu-ops/issues/2215) | P3 | Find use case example for torch-xpu-ops.lib in sy… | dvrogozh | Prototype a sycl cpp extension that calls an op implemented only in torch-xpu-ops (e.g., a kernel registered via TORCH_LIBRARY_IMPL under x… | dvrogozh |  |
| [#2253](https://github.com/intel/torch-xpu-ops/issues/2253) | P3 | the supported dtypes are not align with cuda | daisyden | Update dtypesIfXPU in upstream opinfo definitions for each failing op to match the dtype set actually supported by the XPU/oneDNN backend (… | daisyden | duplicate, skipped, ut_upstream |
| [#2444](https://github.com/intel/torch-xpu-ops/issues/2444) | P3 | [upstream_ut] RuntimeError: UR backend failed. UR… | Silv3S | Use the XPU-ported copy in intel/torch-xpu-ops PR #3340 which runs the large-tensor check on XPU with proper largeTensorTest guard, and dro… | wincent8 | skipped |
| [#2686](https://github.com/intel/torch-xpu-ops/issues/2686) | P3 | [distributed] Accuracy issues with test_distribut… | frost-intel | Mirror upstream behavior by disabling test_DistributedDataParallel on XPU/XCCL in torch-xpu-ops distributed skip list (test/xpu/distributed… | madhumitha0102 | bug, module: distributed |
| [#2729](https://github.com/intel/torch-xpu-ops/issues/2729) | P3 | [Bug Skip]: Random failures 2026WW03 | Silv3S, BartoszKokoszko | Split into per-op bug tickets for the top offenders (grid_sampler_2d decomp and conv/conv_transpose3d autograd have recurring skip historie… | CuiYifeng | skipped, bug_fix_stage5, random |
| [#2918](https://github.com/intel/torch-xpu-ops/issues/2918) | P3 | [XPU][upstream_ut][COW] Skip non-supported ops (j… | gplutop7 | Add the six OpInfo names to the XPU skip list in torch/testing/_internal/common_methods_invocations.py (via the XPU allow_list/skip_list me… | gplutop7 | skipped, bug_fix_stage3, ut_upstream |
| [#3131](https://github.com/intel/torch-xpu-ops/issues/3131) | P3 | [upstream_ut] NotImplementedError: The operator '… | daisyden | Already fixed via intel/torch-xpu-ops#3367 which added proper bias-grad validation (raising the expected RuntimeError) in the XPU mem-effic… | daisyden | module: ut, skipped, ut_upstream |
| [#3170](https://github.com/intel/torch-xpu-ops/issues/3170) | P3 | Unskip test_bmm_windows_error_xpu_float64 | libohao1201, jafraustro | In third_party/torch-xpu-ops/test/xpu/test_sparse_xpu.py:1970 remove the `@unittest.skipIf(TEST_XPU, ...)` decorator (leaving the pre-exist… | CuiYifeng | skipped, ut_upstream |


<a id="sec-4-3-await-reply"></a>
### 4.3 AWAIT_REPLY  ·  10 issues

**AWAIT_REPLY — open questions in thread; owner must respond**

| Issue | Priority | Title | Owner | Fix Approach | Reporter | Labels |
|---|---|---|---|---|---|---|
| [#1324](https://github.com/intel/torch-xpu-ops/issues/1324) | P1 | [Win] UR Error when OOM and break the tensor cont… | Stonepia | Wait for the compute-runtime / UR fix tied to GSD-11670 and oneAPI 2026.1 (per chuanqi129 2026-04-01). In the meantime, document the limita… | Stonepia | client, os: Windows, module: dependency… |
| [#1547](https://github.com/intel/torch-xpu-ops/issues/1547) | P1 | [distributed] NotImplementedError: The operator '… | Chao1Han | Implement an XPU SymmetricMemory backend (either ProcessGroupXCCL-backed or IPC/L0-based) and register the _async/fused symm_mem ops (fused… | PenghuiCheng | module: distributed, dependency compone… |
| [#1571](https://github.com/intel/torch-xpu-ops/issues/1571) | P1 | [distributed] ValueError: Cannot use ReduceOp.PRE… | zhangxiaoli73 | Once oneCCL exposes PREMUL_SUM, map it in torch/csrc/distributed/c10d/ProcessGroupXCCL (getXcclReduceOp) and remove the ValueError guard; r… | daisyden | module: distributed |
| [#1678](https://github.com/intel/torch-xpu-ops/issues/1678) | P1 | missing op support for `model.share_memory()` | kdrozd-dev | Short-term: add 'xpu' to the non-POSIX-shared-memory device list in torch/storage.py:393 so share_memory_() no longer errors; raise a clear… | jafraustro | bug_fix_stage3 |
| [#1749](https://github.com/intel/torch-xpu-ops/issues/1749) | P1 | transformers UT failure in XPU because SDPA check… | LuFinch | Blocked on oneDNN 3.12 SDPA training support. Once oneDNN is upgraded, remove/relax `check_no_grad` in `Attention.cpp:44` and register the … | sywangyi |  |
| [#1784](https://github.com/intel/torch-xpu-ops/issues/1784) | P1 | [Performance] Torch XPU Profiler is not reliable | jfedorov, aostrowski-hbn | Keep the issue open narrowed to the host-overhead item: collaborate with the PTI-SDK / SYCL runtime team already engaged (per jfedorov 2026… | liangan1 | module: profiler |
| [#1894](https://github.com/intel/torch-xpu-ops/issues/1894) | P1 | [Linux][PT2E] performance test got failed, int8 A… | jenniew | For demucs: extend oneDNN-backed qconv XPU lowering to accept conv1d + unary post-ops (relu/hardtanh/etc.), mirroring the conv2d fusion tab… | kaileiyx | module: quant |
| [#2200](https://github.com/intel/torch-xpu-ops/issues/2200) | P1 | support flash attention op on XPU device | TaoLv \| jianyizh \| dai… | Implement a flash-attention forward/backward via oneDNN SDPA primitive (micro-kernel SDPA available in oneDNN 3.5+) and register it as _fla… | Zjq9409 | dependency component: oneDNN |
| [#2232](https://github.com/intel/torch-xpu-ops/issues/2232) | P1 | sdpa backward kernel is required to reduce memory… | LuFinch | Add _scaled_dot_product_efficient_attention_backward XPU implementation (or unify with flash_attention_backward_sycltla) and register it in… | xin3he |  |
| [#3232](https://github.com/intel/torch-xpu-ops/issues/3232) | P2 | [distributed][tensor] AssertionError: AssertionEr… | madhumitha0102 | Skip the efficient-attention branch of test_attention_shard_without_cp on XPU (the path already agreed in the issue comments) by gating on … | zxd1997066 | bug, module: distributed |


<a id="sec-4-4-skip"></a>
### 4.4 SKIP  ·  9 issues

**SKIP — labeled not-target/wontfix at intake**

| Issue | Priority | Title | Owner | Fix Approach | Reporter | Labels |
|---|---|---|---|---|---|---|
| [#2536](https://github.com/intel/torch-xpu-ops/issues/2536) | P2 | Title: [upstream_ut] AttributeError: module 'torc… | daisyden | Split into three subgroups: (1) DataParallel/_scatter/_gather cases - permanently skip via skip_list_common.py with WONT_FIX note per maint… | daisyden | skipped, port_from_skiplist, not_target |
| [#3126](https://github.com/intel/torch-xpu-ops/issues/3126) | P2 | [upstream_ut] Two NestedTensor issue with flash a… | daisyden | Two-part fix: (a) In Attention.cpp's can_use_flash_attention / can_use_overrideable_attention debug-print logic, add NestedTensor-specific … | daisyden | module: ut, skipped, ut_upstream, not_t… |
| [#2164](https://github.com/intel/torch-xpu-ops/issues/2164) | P3 | skip test_no_cuda_monkeypatch as it is cuda speci… | daisyden | Remove or @unittest.skip the ported test_no_cuda_monkeypatch in third_party/torch-xpu-ops/test/xpu/test_torch_xpu.py (label it cuda-specifi… | daisyden | wontfix, skipped |
| [#2309](https://github.com/intel/torch-xpu-ops/issues/2309) | P3 | unsupported ops with PYTORCH_ENABLE_XPU_FALLBACK … | daisyden | Mirror CUDA skip decorators: add @skipXPUIf / expected-failure entries in third_party/torch-xpu-ops/test/xpu/test_linalg_xpu.py for test__d… | daisyden | wontfix, module: op impl, skipped |
| [#2999](https://github.com/intel/torch-xpu-ops/issues/2999) | P3 | KeyError: 'eager_numerics.use_pytorch_libdevice' | daisyden | Confirmed not_target label: the assignee has decided to permanently skip these tests on XPU. Add the four test_bitwise_adam*_capturable_for… | daisyden | module: inductor, ut_upstream, not_targ… |
| [#3127](https://github.com/intel/torch-xpu-ops/issues/3127) | P3 | [upstream_ut] AssertionError: AssertionError not … | daisyden | Mark wontfix (already labeled). Add a static skip for test_math_backend_high_precision_xpu in torch-xpu-ops skip_list_common.py (Silv3S req… | daisyden | wontfix, module: ut, skipped, ut_upstre… |
| [#3128](https://github.com/intel/torch-xpu-ops/issues/3128) | P3 | [upstream_ut] AssertionError: RuntimeError not ra… | daisyden | Skip the test on XPU (dtype-rejection expectation is CUDA-specific). Add test_invalid_fused_inputs_invalid_dtype_kernel1_xpu to the XPU ski… | daisyden | module: ut, skipped, ut_upstream, not_t… |
| [#3129](https://github.com/intel/torch-xpu-ops/issues/3129) | P3 | [upstream_ut] AssertionError: UserWarning not tri… | daisyden | Per maintainer guidance: set PLATFORM_SUPPORTS_CUDNN_ATTENTION=False for XPU in test_transformers_xpu.py so the three cuDNN-kernel tests ar… | daisyden | module: ut, skipped, ut_upstream, port_… |
| [#3133](https://github.com/intel/torch-xpu-ops/issues/3133) | P3 | [upstream_ut] RuntimeError: scaled_dot_product_at… | daisyden | Not a target. Add all listed nested-tensor SDPA cases to the XPU skip list (test/xpu/skip_list_common.py or test_transformers_xpu skip map)… | daisyden | module: ut, skipped, ut_upstream, not_t… |


<a id="sec-4-5-monitor"></a>
### 4.5 MONITOR  ·  4 issues

**MONITOR — long-running tracker / maintenance / scoping**

| Issue | Priority | Title | Owner | Fix Approach | Reporter | Labels |
|---|---|---|---|---|---|---|
| [#146](https://github.com/intel/torch-xpu-ops/issues/146) | P3 | Evaluate register spill in SYCL kernel | CuiYifeng, jianyizh, men… | Enumerate all spilling kernels (rebuild with `-Xs '-doubleGRF'` or inspect `compiled SIMD32 allocated ... spilled` warnings), then per-kern… | fengyuan14 | enhancement |
| [#1722](https://github.com/intel/torch-xpu-ops/issues/1722) | P3 | Ask an API to query GPU type(iGPU/dGPU). | guangyey | Check if sycl::ext::intel::info::device::device_id / architecture is sufficient; extend c10/xpu/XPUDeviceProp.h to include an is_integrated… | xuhancn | dependency component: oneAPI |
| [#1729](https://github.com/intel/torch-xpu-ops/issues/1729) | P3 | Validation Check List | chuanqi129 | Add the two checklist items as dedicated jobs in the XPU release validation workflow: (1) a no-JIT AOT smoke test that sets `SYCL_CACHE_PER… | EikanWang | module: infra |
| [#2127](https://github.com/intel/torch-xpu-ops/issues/2127) | P3 | Path Coverage enhancement | CuiYifeng | Integrate branch/path coverage instrumentation into the xpu-ops CI: enable --coverage (gcov) on the C++ build and pytest-cov --branch on th… | CuiYifeng | enhancement |


<a id="sec-4-6-not-target-close"></a>
### 4.6 NOT_TARGET_CLOSE  ·  4 issues

**NOT_TARGET_CLOSE — authoritative not-target decision (full or partial)**

| Issue | Priority | Title | Owner | Fix Approach | Reporter | Labels |
|---|---|---|---|---|---|---|
| [#2802](https://github.com/intel/torch-xpu-ops/issues/2802) | P1 | Three aten._scaled_dot_product_flash_attention is… | daisyden | (1) Relax the attn_mask+is_causal check for efficient_attention on XPU (Attention.cpp) or fix the test to not pass both. (2) Align fake/met… | daisyden | module: inductor, ut_upstream |
| [#2471](https://github.com/intel/torch-xpu-ops/issues/2471) | P2 | test_cuda.py gaps | guangyey | Per @guangyey's triage in the issue, wire up the existing accelerator equivalents (torch._C._accelerator_setAllocatorSettings, torch.xpu.me… | daisyden |  |
| [#3132](https://github.com/intel/torch-xpu-ops/issues/3132) | P2 | [upstream_ut] transfomers test reports RuntimeErr… | LuFinch | For the singleton-stride case, land/forward-port pytorch/pytorch#179800 which flips `ignore_singleton_dim` to true in SDPUtils.cpp:77-78 (m… | daisyden | module: ut, skipped, ut_upstream |
| [#2697](https://github.com/intel/torch-xpu-ops/issues/2697) | P3 | Title: [upstream_ut] RuntimeError: Expected to fi… | chunhuanMeng | Short term: skip this test for XPU (add @requires_cuda or xfail_if_xpu) since the codegen signature it checks is CUDA-EFFICIENT_ATTENTION-s… | daisyden | module: inductor, skipped, ut_upstream |


<a id="sec-4-7-check-cases"></a>
### 4.7 CHECK_CASES  ·  24 issues

**CHECK_CASES — XPU test case missing in repo; QA must verify case existence before action**

| Issue | Priority | Title | Owner | Fix Approach | Reporter | Labels |
|---|---|---|---|---|---|---|
| [#2186](https://github.com/intel/torch-xpu-ops/issues/2186) | P1 | AssertionError: Mul tiheadAttention does not supp… | daisyden | Extend _check_arg_device in torch/nn/modules/activation.py to accept 'xpu' (e.g. `x.device.type in {'cpu', 'cuda', 'xpu', torch.utils.backe… | daisyden | dependency component: oneDNN |
| [#2285](https://github.com/intel/torch-xpu-ops/issues/2285) | P1 | Support efficient attention | chunhuanMeng | Implement _efficient_attention_forward/backward for XPU via a new SYCL kernel (e.g. third_party/torch-xpu-ops/src/ATen/native/xpu/sycl/Atte… | daisyden | skipped |
| [#2572](https://github.com/intel/torch-xpu-ops/issues/2572) | P1 | [TorchAO][UT] test/dtypes/test_affine_quantized.p… | xiaowangintel | Align XPU int4pack packing and slice/copy semantics with CUDA: (1) in _convert_weight_to_int4pack_xpu ensure the packed layout is identical… | zxd1997066 | module: ao |
| [#2578](https://github.com/intel/torch-xpu-ops/issues/2578) | P1 | [TorchAO][UT] test/quantization/test_quant_api.py… | Stonepia | Either (a) implement TensorCoreTiledLayout support in _convert_weight_to_int4pack_xpu (WeightInt4Pack.cpp:17) by honoring `innerKTiles` and… | zxd1997066 | module: ao, ut_upstream |
| [#3306](https://github.com/intel/torch-xpu-ops/issues/3306) | P1 | [distributed] no attribute '_reset_fr_recording_x… | frost-intel | Land xpu-ops PR #3332 plus daisyden/pytorch#34: add a _reset_fr_recording_xccl pybind in torch/csrc/distributed/c10d/init.cpp that calls Fl… | madhumitha0102 | module: distributed |
| [#3376](https://github.com/intel/torch-xpu-ops/issues/3376) | P1 | [distributed] AttributeErrors/Feature gaps in new… | frost-intel | In ProcessGroupXCCL (torch-xpu-ops distributed backend) add the missing backend methods (_is_initialized, comm_split_count, _verify_work_ti… | madhumitha0102 | module: distributed |
| [#3377](https://github.com/intel/torch-xpu-ops/issues/3377) | P1 | [distributed] AssertionError: DistBackendError no… | frost-intel | Implement ProcessGroupXCCL::abort (see #3378 fix) so post-abort collectives raise DistBackendError, and add FlightRecorder state updates in… | madhumitha0102 | module: distributed |
| [#3378](https://github.com/intel/torch-xpu-ops/issues/3378) | P1 | [distributed] hang in test_c10d_xccl.py::ProcessG… | frost-intel | Implement ProcessGroupXCCL::abort() (and WorkXCCL::abort) that calls ccl::comm::abort / destroys the cached XCCL communicators, cancels pen… | madhumitha0102 | module: distributed |
| [#2376](https://github.com/intel/torch-xpu-ops/issues/2376) | P2 | [Bug Skip]: NotImplementedError: "logaddexp_xpu" … | daisyden, CuiYifeng | Remove the XPU-specific skips/WAs added in pytorch/pytorch#171238 (test/test_binary_ufuncs.py, test_decomp.py, test_meta.py OpInfo entries)… | mengfei25 | module: ut, skipped |
| [#2491](https://github.com/intel/torch-xpu-ops/issues/2491) | P2 | [upstream_ut] AssertionError: False is not true | PatrykWilczewski | Split into sub-fixes: (1) land pytorch/pytorch#176278 and drop storage_all_devices skip; (2) extend the Python warning-context handler (tor… | libohao1201 | skipped, bug_fix_stage5 |
| [#2630](https://github.com/intel/torch-xpu-ops/issues/2630) | P2 | Title: [upstream_ut] AssertionError: Scalars are … | jmamzax | In BinaryRemainderKernel.cpp, update FmodIntegralFunctor and RemainderIntegralFunctor to match the CUDA reference: when `b == 0` return the… | daisyden | skipped, bug_fix_stage5, port_from_skip… |
| [#2816](https://github.com/intel/torch-xpu-ops/issues/2816) | P2 | torch.logcumsumexp incorrectly returns NaNs for c… | Silv3S | Align with the CUDA implementation: in LogcumsumexpKernel.cpp ensure the initial value for complex dtypes uses opmath complex `-inf + 0j` c… | Silv3S | Ready for merge, skipped, bug_fix_stage5 |
| [#2968](https://github.com/intel/torch-xpu-ops/issues/2968) | P2 | [distributed] timeout issue in test/distributed/t… | frost-intel | Short term: skip these tests on XCCL (daisyden/pytorch#34 already does this) since the underlying blocking-wait/error-handling feature is n… | zxd1997066 | bug, module: distributed |
| [#2969](https://github.com/intel/torch-xpu-ops/issues/2969) | P2 | [distributed] AssertionError: Scalars are not equ… | frost-intel | Apply the fixes already in flight: torch-xpu-ops PR #3332 (align XCCL flight-recorder entry emission with NCCL for uneven allgather) plus d… | zxd1997066 | bug, module: distributed |
| [#2972](https://github.com/intel/torch-xpu-ops/issues/2972) | P2 | [distributed] AssertionError: ValueError not rais… | newtdms | In ProcessGroupXCCL.cpp alltoall_base (lines 1983-1984), drop the p2p=true argument so the calls become checkSingleTensor(outputTensor) / c… | zxd1997066 | bug, module: distributed |
| [#3166](https://github.com/intel/torch-xpu-ops/issues/3166) | P2 | test_consistency_SparseCSR failures | jafraustro | In torch-xpu-ops/src/ATen/native/sparse/xpu/SparseCsrTensorMath.cpp add CSC handling for mul (convert via transpose to CSR, call mul, conve… | CuiYifeng | skipped, ut_upstream |
| [#3379](https://github.com/intel/torch-xpu-ops/issues/3379) | P2 | [distributed] accuracy error in test_c10d_xccl.py | frost-intel | Extend _get_process_group_uid in torch/distributed/distributed_c10d.py to also try pg._get_backend(torch.device('xpu')) and return backend.… | madhumitha0102 | module: distributed |
| [#2508](https://github.com/intel/torch-xpu-ops/issues/2508) | P3 | TypedStorage / TypedTensors deprecation | Silv3S | Wontfix: mark all affected cases as expected failures/skips in the XPU skip list rather than via dynamic label skipping. Tracked in torch-x… | libohao1201 | wontfix, skipped |
| [#2510](https://github.com/intel/torch-xpu-ops/issues/2510) | P3 | [upstream_ut] RuntimeError: Expected output.numel… | SlawomirLaba, PawelSwide… | Either (a) mirror CUDA by registering a largeTensorTest / memory-guard skip for this slow case in the XPU skip list, or (b) add a 64-bit-in… | libohao1201 | skipped |
| [#2512](https://github.com/intel/torch-xpu-ops/issues/2512) | P3 | [upstream_ut] RuntimeError: _histc_xpu does not h… | chunhuanMeng | Edit third_party/torch-xpu-ops/src/ATen/native/xpu/SummaryOps.cpp:_histc_xpu to call `globalContext().alertNotDeterministic("_histc_xpu wit… | libohao1201 | skipped |
| [#2529](https://github.com/intel/torch-xpu-ops/issues/2529) | P3 | [upstream_ut] AssertionError: False is not true | Silv3S, BartoszKokoszko | Skip the three remaining cases via test/xpu/skip_list_common.py (or instantiate_parametrized_tests decorator) as CUDA/cuDNN specific: test_… | daisyden | skipped, port_from_skiplist |
| [#2580](https://github.com/intel/torch-xpu-ops/issues/2580) | P3 | [TorchAO][UT] test/test_low_bit_optim.py Assertio… | arlesniak | Add a small repro that computes the empirical mean of `torch.randint(0, 2**16, (32, 100_000), device='xpu')` low bits on PVC vs BMG/CPU to … | zxd1997066 | module: ao |
| [#3305](https://github.com/intel/torch-xpu-ops/issues/3305) | P3 | [distributed] shrink operation support in test/di… | frost-intel | Won't-fix at backend level; permanently skip the ten test_shrink_group_* cases in the XCCL test list (daisyden/pytorch#34) and document the… | madhumitha0102 | module: distributed |
| [#3365](https://github.com/intel/torch-xpu-ops/issues/3365) | P3 | [Bug Skip]: new found bugs in 2024/04/17 | Silv3S | Hoist the numel>0 TORCH_CHECK in foreach_tensor_max_xpu above the can_use_fast_route branch so it runs for both fast and slow paths (matchi… | LuFinch | skipped |


<a id="sec-5"></a>
## 5. Duplicated issues

Rows where `duplicated_issue` is set or `action_TBD` contains "duplicate of".  —  13 issues.

| Issue | Priority | Title | Owner | Fix Approach | Reporter | Labels |
|---|---|---|---|---|---|---|
| [#2285](https://github.com/intel/torch-xpu-ops/issues/2285) | P1 | Support efficient attention | chunhuanMeng | Implement _efficient_attention_forward/backward for XPU via a new SYCL kernel (e.g. third_party/torch-xpu-ops/src/ATen/native/xpu/sycl/Atte… | daisyden | skipped |
| [#2853](https://github.com/intel/torch-xpu-ops/issues/2853) | P1 | [upstream_ut] torch.ops.aten._flash_attention_for… | LuFinch | Add an XPU registration for aten::_flash_attention_forward in yaml/native/native_functions.yaml (XPU dispatch) and implement the wrapper in… | BBBela | skipped |
| [#2024](https://github.com/intel/torch-xpu-ops/issues/2024) | P2 | AssertionError: Torch not compiled with CUDA enab… | daisyden | Either (a) add these test IDs to test/xpu/skip_list_common.py with a clear TODO so CI stays green, or (b) preferred: send an upstream PR re… | mengfei25 | module: ut, skipped |
| [#2255](https://github.com/intel/torch-xpu-ops/issues/2255) | P2 | [upstream_ut] RuntimeError: Long is not supported… | daisyden | Either (a) land upstream PyTorch PR #169353 (referenced in the issue) which aligns OpInfo dtypes to exclude int64 for XPU matmul/conv, or (… | daisyden | skipped, ut_upstream |
| [#2301](https://github.com/intel/torch-xpu-ops/issues/2301) | P2 | [upstream_ut] dtypes not align with OpInfo | daisyden | Land the dtype alignment already prototyped in pytorch/pytorch PR #161246 (commit 7f545509) which extends OpInfo.dtypesIfXPU for einsum, in… | daisyden | skipped, ut_upstream |
| [#2536](https://github.com/intel/torch-xpu-ops/issues/2536) | P2 | Title: [upstream_ut] AttributeError: module 'torc… | daisyden | Split into three subgroups: (1) DataParallel/_scatter/_gather cases - permanently skip via skip_list_common.py with WONT_FIX note per maint… | daisyden | skipped, port_from_skiplist, not_target |
| [#2715](https://github.com/intel/torch-xpu-ops/issues/2715) | P2 | [upstream_ut] torch._dynamo.exc.Unsupported: Atte… | CuiYifeng | In torch/_dynamo/trace_rules.py, remove torch.xpu from MOD_SKIPLIST (or add torch.xpu.device to the allowed-callable list analogous to torc… | daisyden | skipped, ut_upstream |
| [#3286](https://github.com/intel/torch-xpu-ops/issues/3286) | P2 | New failing test case after enabling tests from t… | BBBela | Land pytorch/pytorch#179905 (BBBela) which adds XPUDeviceVariable in torch/_dynamo/variables/ctx_manager.py mirroring CUDADeviceVariable, e… | BBBela | module: ut, skipped |
| [#2230](https://github.com/intel/torch-xpu-ops/issues/2230) | P3 | test_sparse_csr.py::TestSparseCompressedTritonKer… | tszulist-hbn | Either (a) port the sampled_addmm Triton kernel to support XPU multi-device dispatch similar to the CUDA path, or (b) add XPU skips/xfails … | wincent8 | skipped |
| [#2251](https://github.com/intel/torch-xpu-ops/issues/2251) | P3 | [upstream_ut] test_fake_autocase got Exception: D… | astachowiczhabana | Add linalg_pinv and pinverse (and their _out/_ex variants) to the XPU fp32 autocast promote list in third_party/torch-xpu-ops/src/ATen/auto… | daisyden | duplicate, module: ut, skipped, ut_upst… |
| [#2253](https://github.com/intel/torch-xpu-ops/issues/2253) | P3 | the supported dtypes are not align with cuda | daisyden | Update dtypesIfXPU in upstream opinfo definitions for each failing op to match the dtype set actually supported by the XPU/oneDNN backend (… | daisyden | duplicate, skipped, ut_upstream |
| [#2444](https://github.com/intel/torch-xpu-ops/issues/2444) | P3 | [upstream_ut] RuntimeError: UR backend failed. UR… | Silv3S | Use the XPU-ported copy in intel/torch-xpu-ops PR #3340 which runs the large-tensor check on XPU with proper largeTensorTest guard, and dro… | wincent8 | skipped |
| [#2508](https://github.com/intel/torch-xpu-ops/issues/2508) | P3 | TypedStorage / TypedTensors deprecation | Silv3S | Wontfix: mark all affected cases as expected failures/skips in the XPU skip list rather than via dynamic label skipping. Tracked in torch-x… | libohao1201 | wontfix, skipped |


<a id="sec-6"></a>
## 6. Dependency (external blockers)

Issues with a non-blank `Dependency` value, excluding `upstream-pytorch`, `CPU fallback`, and `SYCL kernel:*` (in-repo kernel pointers). Terminal-QA rows (CLOSE / VERIFY_AND_CLOSE / SKIP / NOT_TARGET_CLOSE) are also excluded.  —  128 issues.

| Issue | Dependency | Priority | Title | Owner | Fix Approach | Reporter | Labels |
|---|---|---|---|---|---|---|---|
| [#1324](https://github.com/intel/torch-xpu-ops/issues/1324) | driver | P1 | [Win] UR Error when OOM and break the tensor cont… | Stonepia | Wait for the compute-runtime / UR fix tied to GSD-11670 and oneAPI 2026.1 (per chuanqi129 2026-04-01). In the meantime, document the limita… | Stonepia | client, os: Windows, module: dependency… |
| [#1678](https://github.com/intel/torch-xpu-ops/issues/1678) | driver | P1 | missing op support for `model.share_memory()` | kdrozd-dev | Short-term: add 'xpu' to the non-POSIX-shared-memory device list in torch/storage.py:393 so share_memory_() no longer errors; raise a clear… | jafraustro | bug_fix_stage3 |
| [#1784](https://github.com/intel/torch-xpu-ops/issues/1784) | driver | P1 | [Performance] Torch XPU Profiler is not reliable | jfedorov, aostrowski-hbn | Keep the issue open narrowed to the host-overhead item: collaborate with the PTI-SDK / SYCL runtime team already engaged (per jfedorov 2026… | liangan1 | module: profiler |
| [#2467](https://github.com/intel/torch-xpu-ops/issues/2467) | driver | P1 | Host may stuck when submit too many kernels when … | jianyizh | Primary fix is on the Level-Zero driver (GSD-12059) to enlarge/grow the event pool or recycle completed events. In the torch-xpu-ops/PyTorc… | jianyizh | dependency component: driver |
| [#2912](https://github.com/intel/torch-xpu-ops/issues/2912) | driver | P1 | [release/2.11] UT extended 220 new failures | unassigned | This needs decomposition, not a single fix: (1) parse the attached changed_tests.log and cluster the 220 failures by op/error signature; (2… | bjarzemb | os: Windows, hw: BMG |
| [#2979](https://github.com/intel/torch-xpu-ops/issues/2979) | driver | P1 | eca_halonext26ts got RuntimeError: ZE_RESULT_ERRO… | mengfei25 | File a bug against the Intel compute-runtime/IGC driver team with the failing kernel's SPIR-V / AOT reproducer extracted from /tmp/torchind… | mengfei25 | hw: BMG, dependency component: driver |
| [#3094](https://github.com/intel/torch-xpu-ops/issues/3094) | driver | P1 | XPUGraph tree support |  | Implement XPUGraph tree support: (1) expose a torch.xpu.graph/XPUGraph capture API backed by SYCL command-graph (or Level Zero command list… | daisyden | module: inductor, ut_upstream |
| [#3227](https://github.com/intel/torch-xpu-ops/issues/3227) | driver | P1 | torch xpu event has ~0.1ms latency, which is too … | guangyey | Replace submit_profiling_tag with a lighter mechanism: either use an in-order queue's built-in timestamp (ze_event_pool with TIMESTAMP flag… | jianyizh |  |
| [#3350](https://github.com/intel/torch-xpu-ops/issues/3350) | driver | P1 | [profiler] [XPU][Windows] torch.profiler fails to… | aostrowski-hbn | Reproduce with PTI debug logging enabled, file a bug against intel/pti-gpu (or upgrade to a newer intel-pti release that supports Windows L… | ZhaoqiongZ | module: profiler |
| [#2349](https://github.com/intel/torch-xpu-ops/issues/2349) | oneAPI | P1 | [oneAPI][backward compatibility] libur_loader.so.… | riverliuintel | In the XPU wheel build/packaging (pytorch setup.py and torch-xpu-ops CMake install rules for libsycl/libur), convert RPATH to RUNPATH only … | dvrogozh |  |
| [#1574](https://github.com/intel/torch-xpu-ops/issues/1574) | oneDNN | P1 | The operator 'aten::_grouped_mm' is not currently… | Stonepia, LuFinch | Add XPU kernel for _grouped_mm under src/ATen/native/xpu/ that dispatches to oneDNN matmul primitive with group descriptor (requires oneDNN… | githubsgi | module: ao |
| [#1749](https://github.com/intel/torch-xpu-ops/issues/1749) | oneDNN | P1 | transformers UT failure in XPU because SDPA check… | LuFinch | Blocked on oneDNN 3.12 SDPA training support. Once oneDNN is upgraded, remove/relax `check_no_grad` in `Attention.cpp:44` and register the … | sywangyi |  |
| [#1894](https://github.com/intel/torch-xpu-ops/issues/1894) | oneDNN | P1 | [Linux][PT2E] performance test got failed, int8 A… | jenniew | For demucs: extend oneDNN-backed qconv XPU lowering to accept conv1d + unary post-ops (relu/hardtanh/etc.), mirroring the conv2d fusion tab… | kaileiyx | module: quant |
| [#2200](https://github.com/intel/torch-xpu-ops/issues/2200) | oneDNN | P1 | support flash attention op on XPU device | TaoLv \| jianyizh \| dai… | Implement a flash-attention forward/backward via oneDNN SDPA primitive (micro-kernel SDPA available in oneDNN 3.5+) and register it as _fla… | Zjq9409 | dependency component: oneDNN |
| [#2207](https://github.com/intel/torch-xpu-ops/issues/2207) | oneDNN | P1 | Enable FP8/MXFP8 Ops with requests and CUDA align… | Stonepia, CuiYifeng, LuF… | Continue landing the referenced PRs (#2145 arithmetic, #2152 compare/cat/where, #2154 compare, #2190 flip/index, #2258 copy) and add AT_DIS… | CuiYifeng | dtype: float8 |
| [#2239](https://github.com/intel/torch-xpu-ops/issues/2239) | oneDNN | P1 | Exception: could not create a primitive descripto… | wpietka | In Deconv.cpp add pre-dispatch parameter validation mirroring oneDNN's requirements; when output_padding or dilation combinations are unsup… | zxd1997066 | skipped, bug_fix_stage5 |
| [#2769](https://github.com/intel/torch-xpu-ops/issues/2769) | oneDNN | P1 | [oneDNN] New failed test cases with 3.11 compared… | mengfei25 | Block the oneDNN 3.11 uplift until MFDNN-14584 is resolved; as mengfei25 requested, retest with oneDNN 3.11.1 to confirm whether the primit… | mengfei25 | hw: PVC, dependency component: oneDNN, … |
| [#2823](https://github.com/intel/torch-xpu-ops/issues/2823) | oneDNN | P1 | [TorchAO][BMG] Llama-3.2-1B-Instruct Dynamic INT8… | xiaowangintel, lchen2331 | Wait for oneDNN fix of MFDNN-14745 (avoid redundant H2D sync in reference SDPA path). In the meantime, in the XPU SDPA wrapper (third_party… | LifengWang | dependency component: oneDNN, module: ao |
| [#3076](https://github.com/intel/torch-xpu-ops/issues/3076) | oneDNN | P1 | [TorchAO][BMG] Llama-3.2-1B-Instruct Dynamic INT8… | LifengWang -> oneDNN tea… | Primary fix is in oneDNN (MFDNN-14792) — restore or improve the int8 matmul kernel selection for the decode shape. Once oneDNN has a fix, b… | LifengWang | dependency component: oneDNN, module: ao |
| [#3216](https://github.com/intel/torch-xpu-ops/issues/3216) | oneDNN | P1 | [OPs] Some ops of XPU have non-determinism and ar… | CuiYifeng | Wire torch.use_deterministic_algorithms through XPU matmul/conv: pin oneDNN primitive attributes (dnnl::fpmath_mode, scratchpad mode, deter… | YangKai0616 |  |
| [#2845](https://github.com/intel/torch-xpu-ops/issues/2845) | oneMKL | P1 | [Bug Skip]:[UT] [Windows] failed cases 2026-2-4 |  | Consolidate with #2852; they share root cause. Add the listed cases to test/xpu/skip_list_win.py temporarily and track fix under a single o… | kaileiyx | skipped_windows |
| [#2852](https://github.com/intel/torch-xpu-ops/issues/2852) | oneMKL | P1 | [Bug Skip]: New UT failures in 0206 nightly on Wi… |  | Bisect the torch-xpu-ops range (SpectralOps.cpp changes) and the PyTorch range (test_ops extended harness) to locate the breaking commit; c… | chuanqi129 | skipped_windows |
| [#2619](https://github.com/intel/torch-xpu-ops/issues/2619) | triton | P1 | [release/2.10] Some models inductor performance d… | mengfei25 | Cherry-pick pytorch PR #169257 (bf16 atomic_add fallback) and the reduction-heuristics fix onto release/2.10; for the cait_m36_384 fusion r… | mengfei25 | E2E, performance, regression |
| [#3058](https://github.com/intel/torch-xpu-ops/issues/3058) | triton | P1 | [E2E] hf_GPT2_large amp_fp16/amp_bf16 training go… | weishi-deng | In torch/_inductor/fx_passes/post_grad.py:1570, make the keep_addmm_fused_for_half_dtypes guard device-aware (skip it for XPU) or gate it o… | kaileiyx | E2E, hw: PVC |
| [#3151](https://github.com/intel/torch-xpu-ops/issues/3151) | triton | P1 | [Triton] Timm_models rexnet_100 / fbnetv3_b / seb… | Triton team (intel-xpu-b… | Bisect triton-xpu between 64bb0de3 and 21033c4e to identify the offending commit and file a triton-xpu upstream fix. Short term: pin triton… | kaileiyx | Accuracy, hw: BMG, dependency component… |
| [#3165](https://github.com/intel/torch-xpu-ops/issues/3165) | triton | P1 | test_sparse_csr_xpu.py::TestSparseCompressedTrito… | jafraustro | File/track a pytorch-triton-xpu issue with the failing kernel reproducer and, once fixed upstream, bump the triton pin. In the meantime, ga… | CuiYifeng | skipped, ut_upstream |
| [#1547](https://github.com/intel/torch-xpu-ops/issues/1547) | xccl | P1 | [distributed] NotImplementedError: The operator '… | Chao1Han | Implement an XPU SymmetricMemory backend (either ProcessGroupXCCL-backed or IPC/L0-based) and register the _async/fused symm_mem ops (fused… | PenghuiCheng | module: distributed, dependency compone… |
| [#1571](https://github.com/intel/torch-xpu-ops/issues/1571) | xccl | P1 | [distributed] ValueError: Cannot use ReduceOp.PRE… | zhangxiaoli73 | Once oneCCL exposes PREMUL_SUM, map it in torch/csrc/distributed/c10d/ProcessGroupXCCL (getXcclReduceOp) and remove the ValueError guard; r… | daisyden | module: distributed |
| [#2659](https://github.com/intel/torch-xpu-ops/issues/2659) | xccl | P1 | [distributed] test_dist2.py RuntimeError: Backend… | Chao1Han | In third_party/torch-xpu-ops/src/xccl/ProcessGroupXCCL.hpp, add an `override` right next to `getOptions()` mirroring ProcessGroupNCCL.hpp:7… | zxd1997066 | module: distributed |
| [#2700](https://github.com/intel/torch-xpu-ops/issues/2700) | xccl | P1 | [distributed] Hang issues with test_distributed_s… | syedshahbaaz | Implement groupStart/groupEnd (startCoalescing/endCoalescing) in ProcessGroupXCCL and override batchIsendIrecv to emit the whole list withi… | madhumitha0102 | bug, module: distributed |
| [#2738](https://github.com/intel/torch-xpu-ops/issues/2738) | xccl | P1 | [distributed] test_c10d_spawn_nccl.py ValueError:… | jenniew | Reproduce with latest wheel (artifact 24487284020). Inspect third_party/torch-xpu-ops' ProcessGroupXCCL.cpp _reduce_scatter_base size check… | zxd1997066 | bug, module: distributed |
| [#3082](https://github.com/intel/torch-xpu-ops/issues/3082) | xccl | P1 | multithread support in distributed |  | Add 'xpu' to the devices tuple at multi_threaded_pg.py:548 (register_backend('threaded', _create_threaded_pg, devices=['cpu','cuda','xpu'])… | daisyden | module: distributed, module: ut |
| [#3233](https://github.com/intel/torch-xpu-ops/issues/3233) | xccl | P1 | [distributed] RuntimeError: No backend for the pa… | songhappy | Implement comm-splitting in ProcessGroupXCCL: add a splitGroup()/merged config override, expose supports_splitting=true, and forward comm_s… | zxd1997066 | bug, module: distributed |
| [#3306](https://github.com/intel/torch-xpu-ops/issues/3306) | xccl | P1 | [distributed] no attribute '_reset_fr_recording_x… | frost-intel | Land xpu-ops PR #3332 plus daisyden/pytorch#34: add a _reset_fr_recording_xccl pybind in torch/csrc/distributed/c10d/init.cpp that calls Fl… | madhumitha0102 | module: distributed |
| [#1171](https://github.com/intel/torch-xpu-ops/issues/1171) | driver | P2 | LNL Windows got unexpected error message | xuhancn, chunhuanMeng | Confirm pytorch#167960 is included in torch-xpu-ops test expectations (loosen error_msg match on Windows) and close the issue; continue to … | daisyden | client, os: Windows, hw : LNL, hw: BMG,… |
| [#1505](https://github.com/intel/torch-xpu-ops/issues/1505) | driver | P2 | [ARC-WSL-Ubuntu24.04] 15 Timm models got fail_acc… |  | Reduce repro to a minimal failing op per model (e.g., conv backward, embedding backward for Albert token_type_embeddings), compare WSL vs L… | libohao1201 | bug, E2E, client, os: Windows, module: … |
| [#1548](https://github.com/intel/torch-xpu-ops/issues/1548) | driver | P2 | [distributed] AssertionError: 'fused_all_gather_m… | Chao1Han | 1) Blocked on XPU SymmetricMemory enablement in torch-xpu-ops (Level Zero IPC or SYCL symm API). 2) Once available, register fused_all_gath… | PenghuiCheng | module: distributed, dependency compone… |
| [#1549](https://github.com/intel/torch-xpu-ops/issues/1549) | driver | P2 | [distributed] AssertionError: 'fused_all_gather_s… | Chao1Han | 1) Depends on #1551: enable XPU SymmetricMemory via SYCL 2026 APIs, then register fused_all_gather_scaled_matmul for XPU and ensure the Ind… | PenghuiCheng | module: distributed, dependency compone… |
| [#1551](https://github.com/intel/torch-xpu-ops/issues/1551) | driver | P2 | [distributed] NotImplementedError: The operator '… | Chao1Han | 1) Wait for SYCL symmetric memory support in oneAPI 2026.0 and implement an XPU SymmetricMemoryAllocator + register fused_scaled_matmul_red… | PenghuiCheng | module: distributed, dependency compone… |
| [#1649](https://github.com/intel/torch-xpu-ops/issues/1649) | driver | P2 | [cpp extension] Provide a clear error message whe… | dvrogozh | In torch/utils/cpp_extension.py SyclExtension helpers, record the oneAPI/ICX version used at wheel build time (e.g. in torch/version.py or … | ZhaoqiongZ | dependency component: oneAPI, module: b… |
| [#1727](https://github.com/intel/torch-xpu-ops/issues/1727) | driver | P2 | [distributed] AttributeError: module 'torch.xpu' … | guangyey | Once oneAPI 2026 is adopted, add a `_sleep(cycles)` binding in `torch/csrc/xpu/Module.cpp` backed by a SYCL kernel that busy-waits the requ… | PenghuiCheng | module: distributed, dependency compone… |
| [#1762](https://github.com/intel/torch-xpu-ops/issues/1762) | driver | P2 | Add an ocloc AOT target compilation test in cmake | chunhuanMeng | Add a CMake `try_compile`/`execute_process` step in the torch-xpu-ops top-level CMake that runs `ocloc compile -device <arch>` with a minim… | jingxu10 | module: build |
| [#2089](https://github.com/intel/torch-xpu-ops/issues/2089) | driver | P2 | need an implementation that won't initialize gpu … | guangyey | Implement a fork-/init-safe device probe using level-zero sysman or XPU-SMI that reports device count without creating a SYCL context, mirr… | faaany | dependency component: driver |
| [#2295](https://github.com/intel/torch-xpu-ops/issues/2295) | driver | P2 | [upstream_ut][xpu][test]nn/test_embedding.py::Tes… | yucai-intel | Bump the PyTorch CI xpu docker image to use the same IGC/oneAPI version as nightly wheel builds (track via .ci/docker/common/install_xpu.sh… | wincent8 | module: inductor, skipped, ut_upstream |
| [#2465](https://github.com/intel/torch-xpu-ops/issues/2465) | driver | P2 | [windows] ut hang | tadkrawiec, mganczarenko | First narrow the hang: instrument the runners to print the current test name before each call (PYTEST_CURRENT_TEST / printing in run_test_w… | bjarzemb | os: Windows |
| [#2689](https://github.com/intel/torch-xpu-ops/issues/2689) | driver | P2 | [LNL][Windows] AssertionError: 'Assertion `cur_ta… | draghan, tadkrawiec | Short term: add a Windows/LNL skip for test_cross_entropy_loss_2d_out_of_bounds_class_index in torch-xpu-ops skip list (e.g. skip_list_win_… | kaileiyx | os: Windows, module: ut |
| [#2800](https://github.com/intel/torch-xpu-ops/issues/2800) | driver | P2 | AttributeError: 'torch._C._XpuDeviceProperties' o… | guangyey | Short-term: add a test-side guard that only queries '.major' on CUDA (e.g. skip/branch when device.type=='xpu' in test_scaled_matmul_cuda.p… | daisyden | dependency component: oneAPI, module: i… |
| [#2858](https://github.com/intel/torch-xpu-ops/issues/2858) | driver | P2 | [Bug Skip]: test_xpu new failures |  | Validate on newer Intel GPU Windows driver; if the runtime-side bug persists, file driver ticket. In the meantime keep skip entry in test/x… | RUIJIEZHONG66166 | os: Windows, skipped_windows |
| [#2908](https://github.com/intel/torch-xpu-ops/issues/2908) | driver | P2 | [release/2.11] Model fail_accuracy for 5 testcases | xuhancn | Close the 3 driver-fixed models once driver >=8531 is the minimum for 2.11 release notes. For pit_b_224: bisect torch-xpu-ops main between … | bjarzemb | E2E |
| [#2924](https://github.com/intel/torch-xpu-ops/issues/2924) | driver | P2 | [release/2.11] xcit_large_24_p8_224 amp_bf16 trai… | jianyizh, mengfei25 | Align the eager and Inductor math paths: (1) in third_party/torch-xpu-ops/src/comm/XPUMathCompat.h replace sycl::rsqrt with 1.f/sycl::sqrt … | mengfei25 | Accuracy, dependency component: Triton |
| [#2928](https://github.com/intel/torch-xpu-ops/issues/2928) | driver | P2 | [release/2.11] pyhpc_turbulent_kinetic_energy fp3… | jianyizh | Short term: in torch/_inductor/codegen/triton.py (or the xpu override), force libdevice.sqrt (or an fp32-safe fallback) for the XPU backend… | mengfei25 | dependency component: Triton |
| [#3048](https://github.com/intel/torch-xpu-ops/issues/3048) | driver | P2 | Profiler result is not correct on B70 | aostrowski-hbn | Blocked on the Intel GPU driver fix tracked as PTI-384. Once the driver delivers corrected kernel timestamps, re-validate the trace with th… | jianyizh | module: profiler |
| [#3142](https://github.com/intel/torch-xpu-ops/issues/3142) | driver | P2 | [upstream_ut] RuntimeError: The sycl_ext_oneapi_w… | LuFinch | Pure oneAPI/driver-side fix: wait for oneAPI 2026.0 which lifts the graph restriction on work_group_scratch_memory (per LuFinch/daisyden). … | daisyden | dependency component: oneAPI, module: u… |
| [#3180](https://github.com/intel/torch-xpu-ops/issues/3180) | driver | P2 | [E2E] Timm/Torchbench models got "eager_two_runs_… | pbielak | Bisect by minifying one model (e.g. coat_lite_mini) to a single op that differs between two eager runs on ARC-Windows, then check whether t… | libohao1201 | Accuracy, os: Windows |
| [#3314](https://github.com/intel/torch-xpu-ops/issues/3314) | driver | P2 | Test_xpu.py: Fatal Python error: Aborted on windo… |  | Wrap MemPool::~MemPool body (releasePool + emptyCache) in a try/catch that logs and swallows c10::Error/sycl::exception so destruction neve… | RUIJIEZHONG66166 | os: Windows |
| [#3326](https://github.com/intel/torch-xpu-ops/issues/3326) | driver | P2 | Sporadic test_mem_eff_attention_large_seq_len_uni… |  | Keep the test in the skip list and tag for driver investigation: collect ZE_DEBUG and dmesg output from a failing CI run, capture the seq_l… | Silv3S | skipped, random |
| [#1555](https://github.com/intel/torch-xpu-ops/issues/1555) | oneDNN | P2 | [distributed] RuntimeError: aten.add.Tensor: got … | chuanqi129 | 1) Integrate the oneDNN fused attention kernel for XPU so SDPA does not fall back to the MATH decomposition under DTensor. 2) Until then, e… | PenghuiCheng | module: distributed, dependency compone… |
| [#1556](https://github.com/intel/torch-xpu-ops/issues/1556) | oneDNN | P2 | [distributed] NotImplementedError: Operator aten.… | pkourdis | 1) Land the oneDNN-based fused SDPA kernel for XPU so the generic overrideable op is backed by a real forward+backward implementation. 2) I… | PenghuiCheng | module: distributed, dependency compone… |
| [#1912](https://github.com/intel/torch-xpu-ops/issues/1912) | oneDNN | P2 | Implement the torch.ops.aten._weight_int4pack_mm … | liangan1 | Track oneDNN 3.11 release for float zero-point support in int4 matmul. Once available, extend the xpu int4 op (src/ATen/native/xpu/Int4Pack… | yuanwu2017 | dependency component: oneDNN |
| [#1973](https://github.com/intel/torch-xpu-ops/issues/1973) | oneDNN | P2 | AssertionError: Scalars or Tensor-likes are not e… | gplutop7 | Primary fix lives in oneDNN (MFDNN-14761): enable FP64 or chunked FP32 accumulation for the GPU matmul/conv primitives used by addmv and de… | mengfei25 | hw: PVC, module: ut, skipped, bug_fix_s… |
| [#2217](https://github.com/intel/torch-xpu-ops/issues/2217) | oneDNN | P2 | AO Performance issue track | Stonepia | Drive the listed oneDNN tickets to closure: request oneDNN team to improve BF16 matmul perf on BMG to match FP16/PTL, and merge the GEMM re… | liangan1 | module: ao |
| [#2255](https://github.com/intel/torch-xpu-ops/issues/2255) | oneDNN | P2 | [upstream_ut] RuntimeError: Long is not supported… | daisyden | Either (a) land upstream PyTorch PR #169353 (referenced in the issue) which aligns OpInfo dtypes to exclude int64 for XPU matmul/conv, or (… | daisyden | skipped, ut_upstream |
| [#2323](https://github.com/intel/torch-xpu-ops/issues/2323) | oneDNN | P2 | [TorchAO] MOE training enabling on XPU | karol-brejna-i | Drive the oneDNN scaled_group_gemm primitive to completion, then register `_scaled_grouped_mm` / `_scaled_grouped_mm_v2` XPU dispatches in … | liangan1 | dependency component: oneDNN, module: ao |
| [#2324](https://github.com/intel/torch-xpu-ops/issues/2324) | oneDNN | P2 | [TorchAO] FP8 conv support | Stonepia | Add an FP8 convolution op in torch-xpu-ops (e.g. `src/ATen/native/xpu/ScaledConv.cpp`) backed by oneDNN convolution with per-tensor/per-cha… | liangan1 | module: ao |
| [#2325](https://github.com/intel/torch-xpu-ops/issues/2325) | oneDNN | P2 | [TorchAO] Float8 training support on XPU | arlesniak | Track and land `_scaled_mm_xpu` via oneDNN FP8 matmul (rowwise + tensorwise scaling), enable Float8 training UTs with the emulate path firs… | liangan1 | module: ao |
| [#2326](https://github.com/intel/torch-xpu-ops/issues/2326) | oneDNN | P2 | [TorchAO] MX training native PyTorch on XPU | karol-brejna-i | Land/track XPU `_scaled_mm` (PR #165978) via oneDNN matmul with scale support, enable the MX training UTs in `torchao/prototype/mx_formats`… | liangan1 | module: ao |
| [#2390](https://github.com/intel/torch-xpu-ops/issues/2390) | oneDNN | P2 | SDPA in pytorch use different backend compared wi… | LuFinch | Track oneDNN MFDNN-14834; once v3.11 lands with fused training SDPA, wire it into the XPU SDPA dispatch (torch-xpu-ops `src/ATen/native/xpu… | jiqing-feng |  |
| [#2439](https://github.com/intel/torch-xpu-ops/issues/2439) | oneDNN | P2 | [oneDNN] TestDecompXPU.test_quick_addmv_xpu_float… | libohao1201 | Adopt the upstream tolerance fix proposed for addmv decomp cross-reference in pytorch/pytorch#174590 (add addmv to the per-op tolerance ove… | mengfei25 | dependency component: oneDNN, module: ut |
| [#2482](https://github.com/intel/torch-xpu-ops/issues/2482) | oneDNN | P2 | test_dtypes issue introduced by pytorch test samp… | daisyden | Align the XPU OpInfo dtype overrides for nn.functional.conv_transpose{1,2,3}d: update the xpu dtype override list in third_party/torch-xpu-… | daisyden | skipped |
| [#2597](https://github.com/intel/torch-xpu-ops/issues/2597) | oneDNN | P2 | [TorchAO][BMG] INT4 GPTQ shows worse performance … | xiaowangintel | 1) Compare the lowered FX graph for RTN vs GPTQ quantized models to identify why GPTQ emits the extra `24x1x128` batched matmuls (likely a … | LifengWang | module: ao |
| [#2598](https://github.com/intel/torch-xpu-ops/issues/2598) | oneDNN | P2 | [TorchAO][BMG]The first token latency of Qwen2.5-… | Stonepia | Profile first-token with `SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=2` vs default and with onednn-verbose=1 to confirm primitive cache … | LifengWang | module: ao |
| [#2840](https://github.com/intel/torch-xpu-ops/issues/2840) | oneDNN | P2 | Accuracy issue with 64 bit indexing depthwise_conv | SlawomirLaba, Silv3S | File / follow the oneDNN ticket for depthwise convolution accuracy regression under int64 offsets; in the interim, keep the skip in test/xp… | kdrozd-dev | dependency component: oneDNN, skipped, … |
| [#2862](https://github.com/intel/torch-xpu-ops/issues/2862) | oneDNN | P2 | accuracy issue with test_float8_scale_fast_accum_… | tszulist-hbn | Bump bundled oneDNN in cmake/External/oneDNN.cmake to v3.10.2 (hash f1d47193..) which contains PR #4923. Then re-enable the test in test/xp… | daisyden |  |
| [#2869](https://github.com/intel/torch-xpu-ops/issues/2869) | oneDNN | P2 | [Bug Skip]: New UT failure in 0209 nightly window… |  | Split this umbrella issue into per-module children: (1) Conv fp64/complex128 Windows: investigate oneDNN engine creation for f64/c64 conv o… | RUIJIEZHONG66166 | skipped_windows |
| [#2965](https://github.com/intel/torch-xpu-ops/issues/2965) | oneDNN | P2 | [Bug Skip]: Random failures 2026WW10 |  | Split remaining cases out of this umbrella (conv3d/max_pool already moved to #3103 and #2676). For conv_transpose3d jvpvjp: either raise to… | CuiYifeng | hw: PVC, skipped, random |
| [#3103](https://github.com/intel/torch-xpu-ops/issues/3103) | oneDNN | P2 | Tensor-likes are not equal for functorch and back… | BBBela | First confirm numerics by running conv3d backward on XPU vs CPU directly and measuring max diff; if within 1e-4-1e-3 relative, raise tolera… | BBBela | module: ut, skipped, random |
| [#3136](https://github.com/intel/torch-xpu-ops/issues/3136) | oneDNN | P2 | [upstream_ut] AssertionError: False is not true i… | LuFinch | For test_disable_fastpath_xpu, land/forward-port pytorch/pytorch#179701 and re-enable. For the two fused_sdp_priority_order cases, add them… | daisyden | module: ut, skipped, ut_upstream |
| [#3137](https://github.com/intel/torch-xpu-ops/issues/3137) | oneDNN | P2 | [upstream_ut] RuntimeError: expected scalar type … | LuFinch | Merge/forward-port pytorch/pytorch#179701 so `_transformer_encoder_layer_fwd` takes the slow path (or casts parameters) under xpu autocast,… | daisyden | module: ut, skipped, ut_upstream, random |
| [#1900](https://github.com/intel/torch-xpu-ops/issues/1900) | oneMKL | P2 | implement torch.linalg.qr xpu backend | pbielak | Unblock PR #2399 by: (1) tracking the oneMKL geqrf/orgqr perf request to completion, (2) landing the kernel even if perf is suboptimal with… | yao-matrix | module: op impl, bug_fix_stage3 |
| [#1901](https://github.com/intel/torch-xpu-ops/issues/1901) | oneMKL | P2 | implement torch.linalg.svd xpu backend | CuiYifeng | Option A: implement linalg_svd XPU kernel via oneMKL gesvd/gesvdj and gate it behind a size/shape heuristic to avoid regressions vs CPU pat… | yao-matrix | module: op impl |
| [#1902](https://github.com/intel/torch-xpu-ops/issues/1902) | oneMKL | P2 | implement torch.linalg.pinv xpu backend | mwiktor-intel | Land the pending implementation PR that registers linalg_pinv (and its _out variant) via oneMKL-backed SVD + matmul composition in third_pa… | yao-matrix | module: op impl, bug_fix_stage5 |
| [#1936](https://github.com/intel/torch-xpu-ops/issues/1936) | oneMKL | P2 | implement torch.linalg.cholesky xpu backend | mwiktor-intel | Implement at::linalg_cholesky_ex (the structured op that cholesky is lowered to) for XPU by wrapping oneMKL LAPACK potrf (via sycl/onemkl::… | jiqing-feng | module: op impl, bug_fix_stage5 |
| [#2615](https://github.com/intel/torch-xpu-ops/issues/2615) | oneMKL | P2 | [Bug Skip]: New failures RuntimeError: Unsupporte… | CuiYifeng | Extend the Half/ComplexHalf promotion wrapper in mkl/SpectralOps.cpp to cover all public fft entry points (_fft_c2c, _fft_r2c, _fft_c2r), i… | kaileiyx | module: ut, skipped |
| [#1877](https://github.com/intel/torch-xpu-ops/issues/1877) | triton | P2 | Torchbench model squeezenet1_1 and functorch_dp_c… | DamJanusz | Bisect functorch_dp_cifar10 bf16 training failure to isolate the differing kernel (suspected BatchNorm backward) by comparing eager vs indu… | libohao1201 | Accuracy, hw: BMG, hw: PVC, bug_fix_sta… |
| [#2128](https://github.com/intel/torch-xpu-ops/issues/2128) | triton | P2 | [2.9][BMG-Windows][Torchbench] speeach_transforer… |  | Reduce to a minimal AOTI/Inductor repro by capturing the offending partitioned subgraph via TORCH_COMPILE_DEBUG=1 and TORCHINDUCTOR_CACHE_D… | libohao1201 | os: Windows |
| [#2163](https://github.com/intel/torch-xpu-ops/issues/2163) | triton | P2 | 3 distributed UT cases need to be supported by - … | githubsgi | Land the four enumerated upstream enablement changes: add an XPU entry in torch/_inductor/analysis/device_info.py, replace torch.cuda.* wit… | libohao1201 | module: distributed |
| [#2246](https://github.com/intel/torch-xpu-ops/issues/2246) | triton | P2 | torch/sparse/_triton_ops*.py need to be ported to… |  | Port the two modules to be device-agnostic: replace torch.cuda.get_device_name() with a helper that picks the active accelerator (torch.acc… | wincent8 | skipped |
| [#2329](https://github.com/intel/torch-xpu-ops/issues/2329) | triton | P2 | [upstream_ut] feature missing: get_device_tflops … | etaf | Add XPU support in torch/_inductor/utils.py get_device_tflops() and get_dram_gbps(): detect device.type=='xpu' and compute peak TFLOPS from… | daisyden | duplicate, dependency component: Triton… |
| [#2532](https://github.com/intel/torch-xpu-ops/issues/2532) | triton | P2 | Title: [upstream_ut] AssertionError: wrong number… | yucai-intel | Align the XPU _convert_weight_to_int4pack op with the CUDA contract: (1) accept kByte weight and produce the same 4-D packed layout (N//8, … | daisyden | skipped, port_from_skiplist |
| [#2554](https://github.com/intel/torch-xpu-ops/issues/2554) | triton | P2 | [upstream_ut] AssertionError: AssertionError not … | daisyden | No change needed in torch-xpu-ops; wait for intel-xpu-backend-for-triton fix (#5654) to land and bump the pinned Triton commit in PyTorch (… | daisyden | module: inductor, skipped |
| [#2654](https://github.com/intel/torch-xpu-ops/issues/2654) | triton | P2 | [BMG][OOB] t5 inference performance drop 2 | RUIJIEZHONG66166 | Tune the XPU branch of persistent_reduction heuristics in torch/_inductor/runtime/triton_heuristics.py so that for reduction kernels where … | jianyizh | E2E, dtype: float16, triaged, performan… |
| [#2655](https://github.com/intel/torch-xpu-ops/issues/2655) | triton | P2 | [BMG][OOB] hf_Reformer performance drop | jianyizh | Track the IGC fix (IGC-14276) and bump the minimum IGC / compute-runtime requirement in torch-xpu-ops CI once resolved. In the meantime, ad… | jianyizh | E2E, dtype: float16, triaged, performan… |
| [#2660](https://github.com/intel/torch-xpu-ops/issues/2660) | triton | P2 | [release/2.10][Windows][BMG] New failed test cases | pfierek, tadkrawiec, ery… | (a) Fix the runner: install MSVC and expose cl.exe so Triton/inductor compile works, or skip those tests on Windows. (b) File a Conv2d expa… | mengfei25 | os: Windows, hw: BMG, module: ut |
| [#2662](https://github.com/intel/torch-xpu-ops/issues/2662) | triton | P2 | [release/2.10][Windows][BMG] New failed test case… | tadkrawiec, kdrozd-dev | Split into sub-tasks: (a) CI side — install MSVC Build Tools and set `CC=cl`/add cl.exe to PATH for Windows XPU runners so inductor's C-shi… | mengfei25 | os: Windows, hw: BMG, module: ut |
| [#2888](https://github.com/intel/torch-xpu-ops/issues/2888) | triton | P2 | torch._inductor.exc.InductorError: AssertionError… | Stonepia | Test is explicitly named 'bad_cast' and expects an exception on eager vs compile path; align the XPU test expectation. Options: (1) skip te… | daisyden | module: inductor, ut_upstream |
| [#2935](https://github.com/intel/torch-xpu-ops/issues/2935) | triton | P2 | [release/2.11][inductor] huggingface amp_fp16 and… | jianyizh | Revert, guard, or re-tune the change in pytorch/pytorch@bc4d0bf3 for XPU. Work with the Inductor/Triton-XPU maintainers to (a) identify whi… | agnottaski | performance |
| [#2938](https://github.com/intel/torch-xpu-ops/issues/2938) | triton | P2 | [release/2.11] basic_gnn_gin and basic_gnn_sage i… | jianyizh | This is an upstream Inductor issue; track via pytorch#177117. Workaround in torch-xpu-ops CI: skip/baseline these two models until upstream… | mengfei25 | performance, dependency component: comm… |
| [#2939](https://github.com/intel/torch-xpu-ops/issues/2939) | triton | P2 | [release/2.11] gmlp_s16_224 inference amp perform… | jianyizh | In the Inductor XPU heuristics (torch/_inductor/runtime/triton_heuristics.py and hints.py), add an XPU-specific override that keeps DEFAULT… | mengfei25 | performance |
| [#2952](https://github.com/intel/torch-xpu-ops/issues/2952) | triton | P2 | [release/2.11][wsl] timm_models_accuracy_training… | weishi-deng | No torch-xpu-ops kernel change required. Either (a) relax the accuracy tolerance for this bf16 training model on BMG, or (b) work with the … | bjarzemb | Accuracy, hw: BMG |
| [#2960](https://github.com/intel/torch-xpu-ops/issues/2960) | triton | P2 | [release/2.11] timm_models_xcit_large_24_p8_224_f… | pfierek, tadkrawiec | Confirm if raising `cosine` tolerance threshold to match CUDA expectations for xcit (common practice for fp16 training accuracy in benchmar… | shangerxin | os: Windows |
| [#2997](https://github.com/intel/torch-xpu-ops/issues/2997) | triton | P2 | AssertionError of test_linear_and_cel_max_autotune | etaf | Track upstream fix pytorch/pytorch#180330 (already identified by @etaf) and land it in the PT 2.12 cherry-pick queue; per assignee, 2.12 re… | daisyden | module: inductor, ut_upstream |
| [#3006](https://github.com/intel/torch-xpu-ops/issues/3006) | triton | P2 | AssertionError: '.to(tl.float16)' unexpectedly fo… | CuiYifeng | In the Triton reduction codegen for argmax/argmin (triton.py around line 4469 `final_argreduce` and the block-ptr store generation), ensure… | daisyden | module: inductor, ut_upstream |
| [#3081](https://github.com/intel/torch-xpu-ops/issues/3081) | triton | P2 | Sparse CSR gemm-like ops have not been supported … | tszulist-hbn | Split the fix: (a) extend SparseCsrTensorMathKernels.cpp to instantiate complex64/complex128 add kernels (AT_DISPATCH_ALL_TYPES_AND_COMPLEX… | daisyden | module: ut |
| [#3088](https://github.com/intel/torch-xpu-ops/issues/3088) | triton | P2 | [TorchAO][BMG] INT4 RTN Flex-attention got 5% per… | hoshibara | Two-pronged: (1) file/track triton XPU backend register-spill regression (already filed as intel-xpu-backend-for-triton#6625) so codegen re… | LifengWang | dependency component: Triton, module: ao |
| [#3148](https://github.com/intel/torch-xpu-ops/issues/3148) | triton | P2 | [Triton] Huggingface openai/whisper-tiny got fail… | mengfei25 | Primary: wait for the Triton fix in intel/intel-xpu-backend-for-triton#6489 and bump the pinned triton-xpu commit. Short-term mitigation (p… | mengfei25 | Accuracy, hw: BMG, hw: PVC, dependency … |
| [#3175](https://github.com/intel/torch-xpu-ops/issues/3175) | triton | P2 | [Bug Skip]: ValueError: sampled_addmm(): all inpu… | jkosnox | In torch/sparse/_triton_ops.py:34 change the check to `t.device.type in ("cuda", "xpu")` (same pattern already applied for other _triton_op… | CuiYifeng | skipped |
| [#3331](https://github.com/intel/torch-xpu-ops/issues/3331) | triton | P2 | [ai_generated] torch.compile with slice_scatter p… | Copilot | Reproduce with TORCH_LOGS=output_code,inductor and capture the generated triton kernel for the backward; diff against the CUDA-generated ke… | laifenxiawucha | ai_generated |
| [#2165](https://github.com/intel/torch-xpu-ops/issues/2165) | xccl | P2 | [distributed] test_device_mesh.py::TestDeviceMesh… | jemitche1 | Implement ProcessGroupXCCL::splitGroup() mirroring ProcessGroupNCCL's splitGroup (and also implement mergeRemoteGroup if needed) so it hono… | zxd1997066 | bug, module: distributed |
| [#2968](https://github.com/intel/torch-xpu-ops/issues/2968) | xccl | P2 | [distributed] timeout issue in test/distributed/t… | frost-intel | Short term: skip these tests on XCCL (daisyden/pytorch#34 already does this) since the underlying blocking-wait/error-handling feature is n… | zxd1997066 | bug, module: distributed |
| [#2969](https://github.com/intel/torch-xpu-ops/issues/2969) | xccl | P2 | [distributed] AssertionError: Scalars are not equ… | frost-intel | Apply the fixes already in flight: torch-xpu-ops PR #3332 (align XCCL flight-recorder entry emission with NCCL for uneven allgather) plus d… | zxd1997066 | bug, module: distributed |
| [#2972](https://github.com/intel/torch-xpu-ops/issues/2972) | xccl | P2 | [distributed] AssertionError: ValueError not rais… | newtdms | In ProcessGroupXCCL.cpp alltoall_base (lines 1983-1984), drop the p2p=true argument so the calls become checkSingleTensor(outputTensor) / c… | zxd1997066 | bug, module: distributed |
| [#3100](https://github.com/intel/torch-xpu-ops/issues/3100) | xccl | P2 | [distributed] /handler/dump_nccl_trace_pickle and… | songhappy | (1) In FlightRecorderXCCL.cpp register a 'dump_nccl_trace_pickle' (or an alias) handler via ::c10d::control_plane::registerHandler, paralle… | zxd1997066 | module: distributed |
| [#3101](https://github.com/intel/torch-xpu-ops/issues/3101) | xccl | P2 | [distributed] 'torch._C._distributed_c10d.Process… | jenniew | Add a .def('_set_default_timeout', &::c10d::ProcessGroupXCCL::setTimeout, py::arg('timeout'), py::call_guard<py::gil_scoped_release>()) bin… | zxd1997066 | module: distributed |
| [#1059](https://github.com/intel/torch-xpu-ops/issues/1059) | driver | P3 | SYCL RT: Using recommended shortcut API for kerne… | CuiYifeng, jianyizh | Wait for oneAPI DLE 26.0; once available, replace the custom query in src/comm/DeviceProperties.h with `sycl::ext::oneapi::experimental::in… | fengyuan14 | dependency component: oneAPI |
| [#1722](https://github.com/intel/torch-xpu-ops/issues/1722) | driver | P3 | Ask an API to query GPU type(iGPU/dGPU). | guangyey | Check if sycl::ext::intel::info::device::device_id / architecture is sufficient; extend c10/xpu/XPUDeviceProp.h to include an is_integrated… | xuhancn | dependency component: oneAPI |
| [#3086](https://github.com/intel/torch-xpu-ops/issues/3086) | driver | P3 | nvml support blocks some test cases | Triage | These tests are CUDA-specific and should be excluded on XPU (add to the skip list in third_party/torch-xpu-ops/test/xpu/skip_list_common.py… | daisyden | module: ut |
| [#2261](https://github.com/intel/torch-xpu-ops/issues/2261) | oneAPI | P3 | [xpu][profiler] Run with fork process has extra w… | moksiuc | No code change needed in torch-xpu-ops. Require PTI >= 0.15 (Deep Learning Essentials 2025.3.1) in the CI image and merge pytorch/pytorch#1… | chuanqi129 | dependency component: oneAPI, module: p… |
| [#1893](https://github.com/intel/torch-xpu-ops/issues/1893) | oneDNN | P3 | [upstream_ut] oneDNN accuracy issues in test_ops_… | chunhuanMeng | Bump tolerance for the remaining mv/addmv fp32 ops in XPU OpInfo (torch-xpu-ops test/xpu/xpu_test_utils.py or a toleranceOverride for mv/ad… | daisyden | skipped, ut_upstream |
| [#2245](https://github.com/intel/torch-xpu-ops/issues/2245) | oneDNN | P3 | oneDNN matmul received incorrect shape in test/te… | CuiYifeng | Invoke the standard CSR validation (at::_validate_sparse_csr_tensor_args or the same crow_indices/col_indices checks used on CUDA) at the t… | wincent8 | module: ut, skipped |
| [#2248](https://github.com/intel/torch-xpu-ops/issues/2248) | oneDNN | P3 | [upstream_ut] test_cow failures | gplutop7 | Two-part fix: (1) For ops where materialization is fundamentally required by oneDNN/oneMKL layout conversion (conv*, GEMM-family, cholesky*… | daisyden | skipped, bug_fix_stage3, ut_upstream |
| [#2140](https://github.com/intel/torch-xpu-ops/issues/2140) | oneMKL | P3 | Consider how to avoid copy in FFT kernels | CuiYifeng | Refactor _fft_{c2c,c2r,r2c}_mkl and their callee _exec_fft / _mkl_dft to accept an optional output Tensor& and, when the caller's 'out' is … | CuiYifeng | enhancement |
| [#3121](https://github.com/intel/torch-xpu-ops/issues/3121) | oneMKL | P3 | [Bug Skip]: CUDA specific UT test_fft_half_and_ch… | Triage | Make the error regex device-aware in test_spectral_ops.py test_fft_half_and_chalf_not_power_of_two_error (e.g. 'cuFFT\|MKL\|powers? of two'… | CuiYifeng | skipped |
| [#3296](https://github.com/intel/torch-xpu-ops/issues/3296) | oneMKL | P3 | accuracy gap of stft in float16 | EikanWang, Copilot | Either widen the stft fp16 tolerance for XPU in OpInfo (atol~5e-4, rtol~5e-3) similar to existing CUDA overrides, or skip stft fp16 in test… | daisyden | module: ut, ut_upstream |
| [#2235](https://github.com/intel/torch-xpu-ops/issues/2235) | triton | P3 | test/test_sparse_csr.py::TestSparseCompressedTrit… |  | Resolve together with #2246: make get_meta() query the current accelerator and register XPU tuning results in _operation_device_version_dat… | wincent8 | skipped |
| [#3095](https://github.com/intel/torch-xpu-ops/issues/3095) | triton | P3 | cutlass support blocks some unit test cases | Triage | Skip these CUDA-cutlass-only tests on XPU via @skipIfXPU (or gate with HAS_CUTLASS) in test/inductor/test_cudacodecache.py. Longer term, if… | daisyden | module: inductor, ut_upstream |
| [#3176](https://github.com/intel/torch-xpu-ops/issues/3176) | triton | P3 | [Bug Skip]: ValueError: _scaled_dot_product_atten… |  | No code change needed — issue already resolved by the upstream change broadening check_device to accept xpu. Remove the skip entries from t… | CuiYifeng | skipped |
| [#3305](https://github.com/intel/torch-xpu-ops/issues/3305) | xccl | P3 | [distributed] shrink operation support in test/di… | frost-intel | Won't-fix at backend level; permanently skip the ten test_shrink_group_* cases in the XCCL test list (daisyden/pytorch#34) and document the… | madhumitha0102 | module: distributed |
| [#489](https://github.com/intel/torch-xpu-ops/issues/489) | xccl | P3 | Moco NotImplementedError: xpu not supported | weishi-deng | Land pytorch/benchmark#2616 (device-agnostic MoCo init using XCCL when xpu is selected), ensure ProcessGroupXCCL supports the collectives M… | mengfei25 | E2E, Accuracy, module: torchbench, dtyp… |


<a id="sec-7"></a>
## 7. New submitted issues (<7 days)

Issues created on or after 2026-04-14, excluding terminal-QA rows.  —  27 issues.

| Issue | Created | Priority | Title | Owner | Fix Approach | Reporter | Labels |
|---|---|---|---|---|---|---|---|
| [#3394](https://github.com/intel/torch-xpu-ops/issues/3394) | 2026-04-20 | P0 | crash still occur in sdpa | LuFinch \| oneDNN team (… | Add a sequence-length/workspace-size guard in the XPU SDPA dispatch (sdp_utils / can_use_* checks) to fall back to the math backend when se… | sywangyi |  |
| [#3386](https://github.com/intel/torch-xpu-ops/issues/3386) | 2026-04-19 | P1 | [Bug Skip] XPU Dynamo ocloc/IGC compilation failu… |  | Align intel-igc-cm/intel-ocloc/compute-runtime package versions in the CI image (install matching libigc-core/libigc2 and verify LD_LIBRARY… | daisyden | module: inductor, module: ut, skipped |
| [#3378](https://github.com/intel/torch-xpu-ops/issues/3378) | 2026-04-17 | P1 | [distributed] hang in test_c10d_xccl.py::ProcessG… | frost-intel | Implement ProcessGroupXCCL::abort() (and WorkXCCL::abort) that calls ccl::comm::abort / destroys the cached XCCL communicators, cancels pen… | madhumitha0102 | module: distributed |
| [#3377](https://github.com/intel/torch-xpu-ops/issues/3377) | 2026-04-17 | P1 | [distributed] AssertionError: DistBackendError no… | frost-intel | Implement ProcessGroupXCCL::abort (see #3378 fix) so post-abort collectives raise DistBackendError, and add FlightRecorder state updates in… | madhumitha0102 | module: distributed |
| [#3376](https://github.com/intel/torch-xpu-ops/issues/3376) | 2026-04-17 | P1 | [distributed] AttributeErrors/Feature gaps in new… | frost-intel | In ProcessGroupXCCL (torch-xpu-ops distributed backend) add the missing backend methods (_is_initialized, comm_split_count, _verify_work_ti… | madhumitha0102 | module: distributed |
| [#3368](https://github.com/intel/torch-xpu-ops/issues/3368) | 2026-04-17 | P1 | [TorchAO][BMG] DeepSeek-R1-Distill-Llama-8B RTN I… | Stonepia | Profile the two runs (links 99 vs 100) to pinpoint whether the regression comes from extra guard evaluation, additional recompiles/graph br… | LifengWang |  |
| [#3350](https://github.com/intel/torch-xpu-ops/issues/3350) | 2026-04-16 | P1 | [profiler] [XPU][Windows] torch.profiler fails to… | aostrowski-hbn | Reproduce with PTI debug logging enabled, file a bug against intel/pti-gpu (or upgrade to a newer intel-pti release that supports Windows L… | ZhaoqiongZ | module: profiler |
| [#3342](https://github.com/intel/torch-xpu-ops/issues/3342) | 2026-04-15 | P1 | c-shim implementation is missing for aten.unsquee… | CuiYifeng | Fix in upstream PyTorch inductor: ensure aten.unsqueeze/expand/split_with_sizes are lowered as views in torch/_inductor/lowering.py for the… | kaixuanliu |  |
| [#3396](https://github.com/intel/torch-xpu-ops/issues/3396) | 2026-04-20 | P2 | ubind_copy cases failed due to upstream PR | Silv3S | Remove (or convert to skip when still broken) the xfail('unbind_copy') entries in test_ops_xpu.py and test_vmap_xpu.py, mirroring upstream … | daisyden | module: ut, skipped |
| [#3395](https://github.com/intel/torch-xpu-ops/issues/3395) | 2026-04-20 | P2 | [upstream_ut] test/dynamo/test_ctx_manager.py::Ct… |  | Wait for / cherry-pick upstream PyTorch PR #174370 which teaches Dynamo to trace torch.accelerator.device_index (e.g. by removing the accel… | shangerxin |  |
| [#3393](https://github.com/intel/torch-xpu-ops/issues/3393) | 2026-04-20 | P2 | [upstream_ut] test/dynamo/test_activation_checkpo… |  | Generalize the test's backend-selection branch to handle XPU (use torch.xpu.get_device_properties and the XPU-appropriate sdpa op, or gate … | shangerxin | skipped, ut_upstream |
| [#3389](https://github.com/intel/torch-xpu-ops/issues/3389) | 2026-04-19 | P2 | [Bug Skip] XPU record_stream Tests Fail with CPU … |  | Update test_streams_xpu tests to explicitly create inputs on the XPU device (device='xpu', requires_grad=True) so record_stream routes to t… | daisyden | module: ut, skipped, module: dynamo |
| [#3388](https://github.com/intel/torch-xpu-ops/issues/3388) | 2026-04-19 | P2 | [Bug Skip] XPU Dynamo Graph Lowering - stream_ind… |  | In Dynamo's stream variable handling, ensure XPU streams are registered via _get_stream_by_index and their integer index is captured in the… | daisyden | module: ut, skipped, module: dynamo |
| [#3379](https://github.com/intel/torch-xpu-ops/issues/3379) | 2026-04-17 | P2 | [distributed] accuracy error in test_c10d_xccl.py | frost-intel | Extend _get_process_group_uid in torch/distributed/distributed_c10d.py to also try pg._get_backend(torch.device('xpu')) and return backend.… | madhumitha0102 | module: distributed |
| [#3361](https://github.com/intel/torch-xpu-ops/issues/3361) | 2026-04-16 | P2 | [upstream_ut] test/dynamo/test_higher_order_ops.p… | kdrozd-dev | Wait for upstream PR pytorch/pytorch#174370 to land (generalizes RngStateHelper to dispatch by device, supporting XPU), then rebase and rem… | shangerxin |  |
| [#3356](https://github.com/intel/torch-xpu-ops/issues/3356) | 2026-04-16 | P2 | [upstream_ut] dynamo/test_activation_checkpointin… |  | Land XPU device-specific tolerance override in the upstream PR pytorch/pytorch#169241 (raise atol/rtol for `xpu` like CUDA does for autocas… | shangerxin | skipped, ut_upstream |
| [#3349](https://github.com/intel/torch-xpu-ops/issues/3349) | 2026-04-16 | P2 | [ai_generated] torch.native_batch_norm in eval mo… | Stonepia, chuanqi129, Co… | In BatchNormKernels.cpp:4157-4161 replace the resize/copy/calc_invstd logic with at::native::resize_output(save_mean, {0}) and resize_outpu… | laifenxiawucha |  |
| [#3331](https://github.com/intel/torch-xpu-ops/issues/3331) | 2026-04-14 | P2 | [ai_generated] torch.compile with slice_scatter p… | Copilot | Reproduce with TORCH_LOGS=output_code,inductor and capture the generated triton kernel for the backward; diff against the CUDA-generated ke… | laifenxiawucha | ai_generated |
| [#3326](https://github.com/intel/torch-xpu-ops/issues/3326) | 2026-04-14 | P2 | Sporadic test_mem_eff_attention_large_seq_len_uni… |  | Keep the test in the skip list and tag for driver investigation: collect ZE_DEBUG and dmesg output from a failing CI run, capture the seq_l… | Silv3S | skipped, random |
| [#3390](https://github.com/intel/torch-xpu-ops/issues/3390) | 2026-04-19 | P3 | Clarification requested on mixed non-atomic load … | Triage | Replace the initial `*address_as_ui` read with `target.load()` (using the same sycl::atomic_ref with relaxed order) so all accesses go thro… | tonghaining |  |
| [#3365](https://github.com/intel/torch-xpu-ops/issues/3365) | 2026-04-17 | P3 | [Bug Skip]: new found bugs in 2024/04/17 | Silv3S | Hoist the numel>0 TORCH_CHECK in foreach_tensor_max_xpu above the can_use_fast_route branch so it runs for both fast and slow paths (matchi… | LuFinch | skipped |
| [#3362](https://github.com/intel/torch-xpu-ops/issues/3362) | 2026-04-16 | P3 | test_nn_xpu.py::TestNN::test_cudnn_weight_format … | jmamzax | Either drop test_cudnn_weight_format from test_nn_xpu.py (it is intrinsically a cuDNN test), or rewrite it to use `device=device_type` and … | jmamzax | bug_fix_stage5 |
| [#3358](https://github.com/intel/torch-xpu-ops/issues/3358) | 2026-04-16 | P3 | [v.2.12.0] Release Tracker |  | No engineering fix required; keep open through release Phase 1 (until 2026-04-27) and Phase 2 RC validation, appending cherry-pick links as… | chuanqi129 |  |
| [#3345](https://github.com/intel/torch-xpu-ops/issues/3345) | 2026-04-15 | P3 | Setup Self-hosted runner for Copilot auto debuggi… | chuanqi129 | Add .github/workflows/copilot-setup-steps.yml as drafted in the issue body, targeting the existing self-hosted runner label `pvc_rolling`, … | Stonepia |  |
| [#3334](https://github.com/intel/torch-xpu-ops/issues/3334) | 2026-04-15 | P3 | [upstream_ut] test_repros.py ReproTests.test_part… | Triage | Wait for / track resolution of pytorch/pytorch#174370 and re-bisect against XPU once it lands; in the meantime keep test in the upstream sk… | shangerxin | skipped, ut_upstream |
| [#3330](https://github.com/intel/torch-xpu-ops/issues/3330) | 2026-04-14 | P3 | [ai_generated] torch.std on large float32 input r… | Stonepia, Copilot | Promote the Welford accumulator type to double (or use Kahan/compensated accumulation) for fp32 inputs in ReduceMomentKernels.cpp, mirrorin… | laifenxiawucha | ai_generated |
| [#3329](https://github.com/intel/torch-xpu-ops/issues/3329) | 2026-04-14 | P3 | [ai_generated] torch.cumprod on long float32 inpu… | Stonepia, Copilot | Switch the cumprod scan to use accscalar_t (double for fp32, float for half/bf16) for the internal accumulator, similar to CUDA's cumprod. … | laifenxiawucha | ai_generated |


<a id="sec-8"></a>
## 8. Statistics

- Total rows: **375**
- Classified (non-empty `action_Type`): **337**
- Empty `action_TBD` (no verdict yet): **38**
- Issues flagged for test-case existence check (`CHECK_CASES`): **24**

### 8.1 Primary action_Type distribution (exclusive — one bucket per issue)

| Category | Issues |
|---|---:|
| NEED_ACTION | 63 |
| NEEDS_OWNER | 35 |
| TRACK_PR | 76 |
| IMPLEMENT | 27 |
| RETRIAGE_PRS | 12 |
| ROOT_CAUSE | 10 |
| CLOSE | 23 |
| VERIFY_AND_CLOSE | 29 |
| AWAIT_REPLY | 10 |
| SKIP | 9 |
| MONITOR | 4 |
| NOT_TARGET_CLOSE | 4 |
| CHECK_CASES | 24 |
| WAIT_EXTERNAL | 9 |
| FILE_ISSUE | 2 |
| CHECK_CASES | 24 |

### 8.2 action_Type distribution (multi-label — each category counted once per issue)

| Category | Issues |
|---|---:|
| CLOSE | 23 |
| NOT_TARGET_CLOSE | 4 |
| VERIFY_AND_CLOSE | 30 |
| TRACK_PR | 77 |
| IMPLEMENT | 27 |
| RETRIAGE_PRS | 18 |
| WAIT_EXTERNAL | 12 |
| ROOT_CAUSE | 10 |
| FILE_ISSUE | 2 |
| MONITOR | 4 |
| NEEDS_OWNER | 51 |
| NEED_ACTION | 64 |
| AWAIT_REPLY | 10 |
| CHECK_CASES | 24 |
| SKIP | 10 |

### 8.3 Priority distribution

| Priority | Issues |
|---|---:|
| P0 | 1 |
| P1 | 69 |
| P2 | 188 |
| P3 | 117 |

### 8.4 Status distribution

| Status | Issues |
|---|---:|
| open | 375 |

### 8.5 Category column distribution (top 20)

| Category | Issues |
|---|---:|
| Torch Operations | 123 |
| Inductor | 61 |
| Distributed | 41 |
| Others | 40 |
| Torch Runtime | 35 |
| Flash Attention | 29 |
| TorchAO | 24 |
| Sparse | 22 |

### 8.6 CHECK_CASES issue IDs

24 issues flagged for `check_case_avaliablity` (missing XPU test case in repo):

> #2186, #2285, #2376, #2491, #2508, #2510, #2512, #2529, #2572, #2578, #2580, #2630, #2816, #2968, #2969, #2972, #3166, #3305, #3306, #3365, #3376, #3377, #3378, #3379
