# XPU Ops Bug Scrub Report — UT scope

- **Repository**: `intel/torch-xpu-ops`
- **Generated**: 2026-04-21 (cutoff for Section 7: 2026-04-14)
- **Total issues in workbook**: 53
- **Classified (non-empty `action_Type`)**: 48
- **Empty `action_TBD` (no verdict)**: 5

## 1. Summary

This report groups the 53 tracked torch-xpu-ops issues into action buckets derived from the `action_Type` classification column of the triage workbook. Each issue appears in at most one Action-Required or QA section, chosen by its highest-priority category. Cross-cutting slices (duplicated issues, external dependency blockers, newly filed issues) are listed separately for visibility.

**Headline counts (primary category):**

| Bucket | Categories | Issues |
|---|---|---:|
| Developer action required | NEED PR, TRACK PR, NEEDS_OWNER | 32 |
| QA action required | CLOSE or SKIP, AWAIT_REPLY, MONITOR, CHECK_CASES | 16 |
| Duplicated | — | 5 |
| External dependency (non-upstream-pytorch, non-SYCL-kernel) | — | 13 |
| Filed within last 7 days | — | 0 |

<a id="sec-2"></a>
## 2. Index

- [3. Action required (Developer)](#sec-3)
  - [3.0 UNCLASSIFIED](#sec-3-0-unclassified)
  - [3.1 NEED PR](#sec-3-1-need-pr)
  - [3.2 TRACK PR](#sec-3-2-track-pr)
  - [3.3 NEEDS_OWNER](#sec-3-3-needs-owner)
- [4. QA](#sec-4)
  - [4.1 CLOSE or SKIP](#sec-4-1-close-or-skip)
  - [4.2 AWAIT_REPLY](#sec-4-2-await-reply)
  - [4.3 MONITOR](#sec-4-3-monitor)
  - [4.4 CHECK_CASES](#sec-4-4-check-cases)
- [5. Duplicated issues](#sec-5)
- [6. Dependency (external blockers)](#sec-6)
- [7. New submitted issues (<7 days)](#sec-7)
- [8. Statistics](#sec-8)

<a id="sec-3"></a>
## 3. Action required (Developer)

_[↑ Back to Index](#sec-2)_

Issues in this section require developer work before they can progress. Each subsection is split by `Category` (existing taxonomy column); rows inside each category table are sorted by `Priority` (P0 → P3).

<a id="sec-3-0-unclassified"></a>
### 3.0 UNCLASSIFIED  ·  5 issues

_[↑ Back to Index](#sec-2)_

**UNCLASSIFIED — Phase 4b produced no verdict; needs manual triage**

| Issue | Title | Owner | action_TBD | Fix Approach | Priority | action_reason | Reporter | Labels |
|---|---|---|---|---|---|---|---|---|
| [#2554](https://github.com/intel/torch-xpu-ops/issues/2554) | [upstream_ut] AssertionError: AssertionError not raised | daisyden |  | • No change needed in torch-xpu-ops<br>• wait for intel-xpu-backend-for-triton fix (#5654) to land and bump the pinned<br>&nbsp;&nbsp;Triton commit in PyTorch (.ci/docker/ci_commit_pins/xpu-triton.txt).<br>• In the meantime, keep these three tests in the XPU skip list, and add a<br>&nbsp;&nbsp;verification step in the UT pipeline to remove the skip once the Triton fix is<br>&nbsp;&nbsp;in.<br>• Reproducer requires pytorch PR #170056 as noted in the issue. | P2 |  | daisyden | module: inductor, skipped |
| [#2611](https://github.com/intel/torch-xpu-ops/issues/2611) | [upstream_ut] AssertionError: Tensor-likes are not equal! in test_compile_subprocess | daisyden |  | • Close as duplicate of #2613 (or consolidate skip list into a single issue).<br>• Apply the same fix: update argmax/argmin reduce combine ops to prefer lower<br>&nbsp;&nbsp;index on ties and propagate NaN per CUDA semantics.<br>• Once fixed, unskip all four tests and remove this issue's entry from the skip<br>&nbsp;&nbsp;list. | P2 |  | daisyden | dependency component: driver, module: i… |
| [#2613](https://github.com/intel/torch-xpu-ops/issues/2613) | [upstream_ut] AssertionError: Tensor-likes are not equal! in test_compile_subprocess.py | daisyden |  | • Fix the argmax/argmin reduction combine functor in `ReduceArgMaxKernel.cpp` /<br>&nbsp;&nbsp;`ReduceArgMinKernel.cpp` to: (1) on equal values pick the pair with the<br>&nbsp;&nbsp;smaller index (matching `at::native::argmax_out` CUDA semantics)<br>• (2) propagate NaN so that any NaN operand yields NaN ordering consistent with<br>&nbsp;&nbsp;eager (comparing with `a != a` style checks).<br>• Mirror the CUDA `ArgMaxOps`/`ArgMinOps` template used in<br>&nbsp;&nbsp;aten/src/ATen/native/cuda/ReduceOps.cpp.<br>• Add XPU-specific unit tests with duplicate values and NaN inputs to lock in<br>&nbsp;&nbsp;behavior.<br>• The driver-dependency label is misleading: this is a kernel bug, not a driver<br>&nbsp;&nbsp;bug. | P2 |  | daisyden | dependency component: driver, module: i… |
| [#2715](https://github.com/intel/torch-xpu-ops/issues/2715) | [upstream_ut] torch._dynamo.exc.Unsupported: Attempted to inline function marked as skipped | CuiYifeng |  | • In `torch/_dynamo/trace_rules.py`, remove torch.xpu from MOD_SKIPLIST (or add<br>&nbsp;&nbsp;torch.xpu.device to the allowed-callable list analogous to torch.cuda.device)<br>&nbsp;&nbsp;so dynamo can inline device context-manager __init__ for XPU, matching CUDA<br>&nbsp;&nbsp;behavior.<br>• Alternatively add a polyfill/TorchInGraphFunctionVariable mapping for<br>&nbsp;&nbsp;torch.xpu.device.<br>• Track via duplicate #3286. | P2 |  | daisyden | skipped, ut_upstream |
| [#2800](https://github.com/intel/torch-xpu-ops/issues/2800) | AttributeError: 'torch._C._XpuDeviceProperties' object has no attribute 'major' | guangyey |  | • Short-term: add a test-side guard that only queries `.major` on CUDA (e.g.<br>&nbsp;&nbsp;skip/branch when device.type==`xpu` in<br>&nbsp;&nbsp;`test_scaled_matmul_cuda.py`::test_scaled_mm_vs_emulated) and/or expose a<br>&nbsp;&nbsp;dummy `major`/`minor` property_readonly on _XpuDeviceProperties returning -1<br>&nbsp;&nbsp;so that inequality checks still skip the Blackwell-only path.<br>• Long-term: once DPC++ 2026.2 lands CMPLRLLVM-72166, map the new arch-version<br>&nbsp;&nbsp;API to .major/.minor in `torch/csrc/xpu/Module.cpp`<br>&nbsp;&nbsp;registerXpuDeviceProperties and update python-side `torch/xpu/__init__.py` to<br>&nbsp;&nbsp;surface them. | P2 |  | daisyden | dependency component: oneAPI, module: i… |


<a id="sec-3-1-need-pr"></a>
### 3.1 NEED PR  ·  20 issues

**NEED PR — a PR must be produced or continued (no PR yet, or owner actively debugging root cause, or new code needed)**

<a id="sec-3-1-1-flash-attention"></a>
#### 3.1.1 Flash Attention  ·  3 issues

_[↑ Back to Index](#sec-2)_

| Issue | Title | Owner | action_TBD | Fix Approach | Priority | action_reason | Reporter | Labels |
|---|---|---|---|---|---|---|---|---|
| [#2270](https://github.com/intel/torch-xpu-ops/issues/2270) | Backend Compatibility Error in test/xpu/test_decomp.py | LuFinch | Assignee @LuFinch to investigate | • Either (a) skip _flash_attention_forward in the decomp cross-ref OpInfo list<br>&nbsp;&nbsp;for XPU (mirroring CUDA's skip in common_methods_invocations, adding the op to<br>&nbsp;&nbsp;xfail_if_not_implemented / skipped_ops in `test/xpu/test_decomp_new.py`) so<br>&nbsp;&nbsp;CrossRefFakeMode does not redispatch to CPU, or (b) land PR #2341 which adds<br>&nbsp;&nbsp;the XPU flash-attention forward decomposition path so the cross-ref works<br>&nbsp;&nbsp;without hitting CPU.<br>• Short term: extend the skip_list in test/xpu/ and align with issue #2442's<br>&nbsp;&nbsp;resolution. | P2 | Issue already assigned to @LuFinch; owner to lead root-cause. | libohao1201 | module: ut, skipped |
| [#3140](https://github.com/intel/torch-xpu-ops/issues/3140) | [upstream_ut] RuntimeError: FlashAttentionForwardXPU does not only support dropout > 0.0 yet | LuFinch | Issue needs owner investigation; only weak/closed cross-ref candidates found | • Implement dropout in the sycltla FlashAttention kernels: add a Philox4x32 RNG<br>&nbsp;&nbsp;state (seed/offset plumbed through FLASH_FWD_params and FLASH_BWD_params),<br>&nbsp;&nbsp;apply the mask/scale after the online softmax in `mha_fwd.cpp` and invert it<br>&nbsp;&nbsp;in `mha_bwd.cpp`, then remove the two TORCH_CHECKs.<br>• Alternative short-term mitigation: at the ATen dispatch layer route dropout>0<br>&nbsp;&nbsp;cases to the mem-efficient/math SDPA backend (similar to the headdim fallback<br>&nbsp;&nbsp;in #3141) so user models do not crash. | P2 | Timeline cross-refs all closed unmerged with weak verdict; no actionable PR | daisyden | module: ut, skipped, ut_upstream |
| [#3142](https://github.com/intel/torch-xpu-ops/issues/3142) | [upstream_ut] RuntimeError: The sycl_ext_oneapi_work_group_scratch_memory feature is not yet available for use with SYCL Graph extension. | LuFinch | Issue needs owner investigation; only weak/closed cross-ref candidates found | • Pure oneAPI/driver-side fix: wait for oneAPI 2026.0 which lifts the graph<br>&nbsp;&nbsp;restriction on work_group_scratch_memory (per LuFinch/daisyden).<br>• Meanwhile, keep these cudagraph SDPA cases skipped in<br>&nbsp;&nbsp;`test/xpu/skip_list_common.py` with a reference to CMPLRLLVM-72057, and<br>&nbsp;&nbsp;optionally teach the torch.cuda.graph replacement for XPU to fall back to the<br>&nbsp;&nbsp;math SDPA backend when the compiled stream has an active SYCL-Graph capture so<br>&nbsp;&nbsp;that user code does not crash. | P2 | Timeline cross-refs all closed unmerged with weak verdict; no actionable PR | daisyden | dependency component: oneAPI, module: u… |


<a id="sec-3-1-2-inductor"></a>
#### 3.1.2 Inductor  ·  6 issues

_[↑ Back to Index](#sec-2)_

| Issue | Title | Owner | action_TBD | Fix Approach | Priority | action_reason | Reporter | Labels |
|---|---|---|---|---|---|---|---|---|
| [#2329](https://github.com/intel/torch-xpu-ops/issues/2329) | [upstream_ut] feature missing: get_device_tflops and get_drams_gbps | etaf | Assignee @etaf to investigate | • Add XPU support in `torch/_inductor/utils.py` get_device_tflops() and<br>&nbsp;&nbsp;get_dram_gbps(): detect device.type==`xpu` and compute peak TFLOPS from<br>&nbsp;&nbsp;torch.xpu.get_device_properties (EU count * SIMD width * frequency * ops/cycle<br>&nbsp;&nbsp;per dtype) and DRAM BW from hbm_bandwidth / memory_bus_width once exposed by<br>&nbsp;&nbsp;intel-xpu-backend-for-triton#5792 or a static PVC/BMG table as an interim.<br>• Route the flop-counter scheduler test through torch.accelerator instead of<br>&nbsp;&nbsp;torch.cuda hardcoding, then re-enable<br>&nbsp;&nbsp;test_flop_counter_op_options0_xpu_float16. | P2 | Issue already assigned to @etaf; owner to lead root-cause. | daisyden | duplicate, dependency component: Triton… |
| [#2888](https://github.com/intel/torch-xpu-ops/issues/2888) | torch._inductor.exc.InductorError: AssertionError: Conversions between float8_e5m2 and float8_e4m3fn is not supported! | Stonepia | Issue needs owner investigation; only weak/closed cross-ref candidates found | • Test is explicitly named `bad_cast` and expects an exception on eager vs<br>&nbsp;&nbsp;compile path<br>• align the XPU test expectation.<br>• Options: (1) skip test_bad_cast_xpu in the XPU Inductor skip list since<br>&nbsp;&nbsp;triton-xpu does not yet support fp8<->fp8 conversions, or (2) wrap the<br>&nbsp;&nbsp;assertion in _get_min_elements_per_thread so it raises a cleaner<br>&nbsp;&nbsp;UnsupportedOperatorError that the test already catches, or (3) lower<br>&nbsp;&nbsp;e5m2<->e4m3fn via an intermediate bf16/fp32 cast in the XPU lowering.<br>• Short-term: skip<br>• long-term: add the intermediate-cast lowering in<br>&nbsp;&nbsp;torch/_inductor/codegen/triton.py. | P2 | Timeline cross-refs all closed unmerged with weak verdict; no actionable PR | daisyden | module: inductor, ut_upstream |
| [#2891](https://github.com/intel/torch-xpu-ops/issues/2891) | RuntimeError: Expected to find "(262144, 0, 512, 1" but did not find it | chunhuanMeng | Issue needs owner investigation; only weak/closed cross-ref candidates found | • Either (a) skip/xfail this test for XPU in<br>&nbsp;&nbsp;third_party/torch-xpu-ops/test/inductor skip lists because it is a CUDA<br>&nbsp;&nbsp;codegen-text assertion, or (b) relax the FileCheck to accept the XPU stride<br>&nbsp;&nbsp;pattern `(512, 0, 1, 0)`.<br>• Preferred: add it to the upstream skip list in torch-xpu-ops with a comment<br>&nbsp;&nbsp;noting the stride layout divergence, and open a follow-up to align the XPU<br>&nbsp;&nbsp;SDPA bias expand lowering with CUDA if desired. | P2 | Timeline cross-refs all closed unmerged with weak verdict; no actionable PR | daisyden | module: inductor, ut_upstream |
| [#2997](https://github.com/intel/torch-xpu-ops/issues/2997) | AssertionError of test_linear_and_cel_max_autotune | etaf | Issue needs owner investigation; only weak/closed cross-ref candidates found | • Track upstream fix pytorch/pytorch#180330 (already identified by @etaf) and<br>&nbsp;&nbsp;land it in the PT 2.12 cherry-pick queue<br>• per assignee, 2.12 release does not need it because the failure is UT-only and<br>&nbsp;&nbsp;not observed in E2E.<br>• In the meantime, add test_linear_and_cel_max_autotune to `skip_list_common.py`<br>&nbsp;&nbsp;under `test_inplace_padding.py` (inductor UT).<br>• Verify after merging #180330 that NaN is gone on XPU by rerunning pytest -v<br>&nbsp;&nbsp;`test/inductor/test_inplace_padding.py` -k test_linear_and_cel_max_autotune. | P2 | Timeline cross-refs all closed unmerged with weak verdict; no actionable PR | daisyden | module: inductor, ut_upstream |
| [#3006](https://github.com/intel/torch-xpu-ops/issues/3006) | AssertionError: '.to(tl.float16)' unexpectedly found in '# AOT ID | CuiYifeng | Issue needs owner investigation; only weak/closed cross-ref candidates found | • In the Triton reduction codegen for argmax/argmin (`triton.py` around line<br>&nbsp;&nbsp;4469 `final_argreduce` and the block-ptr store generation), ensure the index<br>&nbsp;&nbsp;result variable's dtype is tracked as int (from `select_index_dtype()`) so the<br>&nbsp;&nbsp;dtype-propagation pass does not insert a cast to the value dtype.<br>• Concretely, when `codegen_upcast_to_fp32=False` and the reduction is<br>&nbsp;&nbsp;argmax/argmin, the post-loop index store should bypass the fp16 src-dtype<br>&nbsp;&nbsp;coercion<br>• add a guard in `TritonKernel.store`/`codegen_indirect_indexing` that skips<br>&nbsp;&nbsp;`.to(tl.float16)` when the logical dtype is integer.<br>• Validate by re-enabling the test on XPU and confirming generated code contains<br>&nbsp;&nbsp;no `.to(tl.float16)` and numerical parity vs eager argmax. | P2 | Timeline cross-refs all closed unmerged with weak verdict; no actionable PR | daisyden | module: inductor, ut_upstream |
| [#3187](https://github.com/intel/torch-xpu-ops/issues/3187) | PyTorch XPU gpu_cpp_wrapper fails with InductorError NotImplementedError | CuiYifeng | needs owner investigation | • In the XPU cpp_wrapper / AOTI codegen<br>&nbsp;&nbsp;(`torch/_inductor/codegen/cpp_wrapper_gpu.py` and XPU-specific wrapper/AOTI<br>&nbsp;&nbsp;shims) ensure that when a decomposition returns NotImplemented we fall back to<br>&nbsp;&nbsp;the aten op (aten.add for complex) via the standard fallback_kernel path used<br>&nbsp;&nbsp;by CUDA.<br>• Validate by removing the skip in pytorch/pytorch#178477 and re-running<br>&nbsp;&nbsp;gpu_cpp_wrapper complex add tests. | P2 | No comments, no linked or referenced PR found | liangan1 | ut_upstream |


<a id="sec-3-1-3-others"></a>
#### 3.1.3 Others  ·  1 issues

_[↑ Back to Index](#sec-2)_

| Issue | Title | Owner | action_TBD | Fix Approach | Priority | action_reason | Reporter | Labels |
|---|---|---|---|---|---|---|---|---|
| [#2024](https://github.com/intel/torch-xpu-ops/issues/2024) | AssertionError: Torch not compiled with CUDA enabled | daisyden | Assignee @daisyden to investigate | • Either (a) add these test IDs to `test/xpu/skip_list_common.py` with a clear<br>&nbsp;&nbsp;TODO so CI stays green, or (b) preferred: send an upstream PR replacing<br>&nbsp;&nbsp;hard-coded `cuda` with instantiate_device_type_tests / self.device_type so the<br>&nbsp;&nbsp;tests run on XPU.<br>• For test_pool3d_large_size_int64 and test_pooling_large, also check whether<br>&nbsp;&nbsp;XPU has the 49GB device memory headroom they need before enabling.<br>• Close as duplicate of #2444 after the skip list lands. | P2 | Issue already assigned to @daisyden; owner to lead root-cause. | mengfei25 | module: ut, skipped |


<a id="sec-3-1-4-sparse"></a>
#### 3.1.4 Sparse  ·  3 issues

_[↑ Back to Index](#sec-2)_

| Issue | Title | Owner | action_TBD | Fix Approach | Priority | action_reason | Reporter | Labels |
|---|---|---|---|---|---|---|---|---|
| [#3165](https://github.com/intel/torch-xpu-ops/issues/3165) | test_sparse_csr_xpu.py::TestSparseCompressedTritonKernelsXPU::test_triton_bsr_softmax meet RuntimeError: ZE_RESULT_ERROR_INVALID_KERNEL_NAME | jafraustro | assignee investigate (split-out from #2209) | • File/track a pytorch-triton-xpu issue with the failing kernel reproducer and,<br>&nbsp;&nbsp;once fixed upstream, bump the triton pin.<br>• In the meantime, gate bsr_softmax on XPU behind a device check that either (a)<br>&nbsp;&nbsp;uses the non-Triton dense-softmax fallback path similar to the CUDA fallback<br>&nbsp;&nbsp;in `_triton_ops.py`, or (b) skips the test with a clear xfail pointing to the<br>&nbsp;&nbsp;triton issue.<br>• The Triton kernel itself (_bsr_softmax_kernel) may also need to be simplified<br>&nbsp;&nbsp;(avoid unsupported tl primitives) to make it lower-able on XPU. | P1 | Skipped Triton sparse softmax case; sub-issue split from #2209; assigned to @jafraustro. | CuiYifeng | skipped, ut_upstream |
| [#2214](https://github.com/intel/torch-xpu-ops/issues/2214) | test/test_sparse.py::TestSparseAnyXPU::test_gradcheck_mm expected error message not match | jenniew | Assignee @jenniew to investigate | • Extend the expected-error regex in `test/test_sparse.py` (around line 5297) to<br>&nbsp;&nbsp;also accept the XPU-specific message (e.g., add `empty_sparse_compressed<br>&nbsp;&nbsp;expected sparse compressed.*` alternative), or add an XPU branch that calls<br>&nbsp;&nbsp;the test with the appropriate regex.<br>• Alternatively, upstream an XPU device entry in the<br>&nbsp;&nbsp;sparse_addmm_sparse_backward dispatch so the thrown error matches the CPU/CUDA<br>&nbsp;&nbsp;wording.<br>• Remove the hard-coded skip in `test/xpu/skip_list_common.py` once fixed. | P2 | Issue already assigned to @jenniew; owner to lead root-cause. | wincent8 | skipped, ut_upstream |
| [#3177](https://github.com/intel/torch-xpu-ops/issues/3177) | Accuracy gap of BF16/FP16 test_block_addmm | jenniew | needs owner investigation | • Land torch-xpu-ops PR #3273: replace result_dense.add_(input_dense, beta) at<br>&nbsp;&nbsp;`SparseCsrTensorMath.cpp`:88 with explicit `result_dense.add_(input_dense *<br>&nbsp;&nbsp;beta)` (or upcast to float32 before accumulation) to match the reference<br>&nbsp;&nbsp;semantics for fp16/bf16, then remove the skip entries from<br>&nbsp;&nbsp;test/xpu/skip_list_common.py. | P2 | No verified PR; no actionable owner requests | CuiYifeng | skipped |


<a id="sec-3-1-5-torch-operations"></a>
#### 3.1.5 Torch Operations  ·  7 issues

_[↑ Back to Index](#sec-2)_

| Issue | Title | Owner | action_TBD | Fix Approach | Priority | action_reason | Reporter | Labels |
|---|---|---|---|---|---|---|---|---|
| [#2244](https://github.com/intel/torch-xpu-ops/issues/2244) | test/test_sparse_csr.py::TestSparseCSRXPU::test_block_addmm meet RuntimeError: empty_sparse_compressed expected sparse compressed (non-block) tensor layout but got SparseBsr | jenniew | needs new PR | • Add the missing BSR-result branch in addmm_out_sparse_csr: compute the dense<br>&nbsp;&nbsp;product via addmm_calculation, convert back with result =<br>&nbsp;&nbsp;result_dense.to_sparse_bsr(block_size) (or use torch::sparse::to_sparse_bsr).<br>• Mirror the same for (CSC, CSC), (BSR, BSR) and (BSC, BSC) combinations that<br>&nbsp;&nbsp;the CUDA path supports.<br>• Add/extend GTest or bring the op back in the skip list only as an interim<br>&nbsp;&nbsp;measure.<br>• No oneDNN change required<br>• this is pure XPU dispatch plumbing. | P2 | Linked PR(s) closed without merge: intel/torch-xpu-ops#2974; fix attempt abandoned | wincent8 | module: ut, skipped |
| [#3136](https://github.com/intel/torch-xpu-ops/issues/3136) | [upstream_ut] AssertionError: False is not true in test_transformers | LuFinch | Issue needs owner investigation; only weak/closed cross-ref candidates found | • For test_disable_fastpath_xpu, land/forward-port pytorch/pytorch#179701 and<br>&nbsp;&nbsp;re-enable.<br>• For the two fused_sdp_priority_order cases, add them to the XPU skip list in<br>&nbsp;&nbsp;`test/xpu/skip_list_common.py` (cuDNN is not a supported XPU backend)<br>• optionally filter `SDPBackend.CUDNN_ATTENTION` out of the expected priority<br>&nbsp;&nbsp;list in a future upstream XPU-aware test refactor. | P2 | Timeline cross-refs all closed unmerged with weak verdict; no actionable PR | daisyden | module: ut, skipped, ut_upstream |
| [#3137](https://github.com/intel/torch-xpu-ops/issues/3137) | [upstream_ut] RuntimeError: expected scalar type Half but found Float | LuFinch | Issue needs owner investigation; only weak/closed cross-ref candidates found | • Merge/forward-port pytorch/pytorch#179701 so `_transformer_encoder_layer_fwd`<br>&nbsp;&nbsp;takes the slow path (or casts parameters) under xpu autocast, then re-enable<br>&nbsp;&nbsp;the four skipped cases.<br>• Track the accuracy delta under a separate oneDNN-SDPA-training follow-up. | P2 | Timeline cross-refs all closed unmerged with weak verdict; no actionable PR | daisyden | module: ut, skipped, ut_upstream, random |
| [#2245](https://github.com/intel/torch-xpu-ops/issues/2245) | oneDNN matmul received incorrect shape in test/test_sparse_csr.py::TestSparseCSRXPU::test_addmm_errors_xpu_float32 | CuiYifeng | Add input-tensor-expansion check on stock PyTorch side (per @CuiYifeng) | • Invoke the standard CSR validation (at::_validate_sparse_csr_tensor_args or<br>&nbsp;&nbsp;the same crow_indices/col_indices checks used on CUDA) at the top of<br>&nbsp;&nbsp;addmm_out_sparse_csr before any layout conversion, so the same error message<br>&nbsp;&nbsp;surfaces on XPU.<br>• Alternatively, update the XPU skip/xfail list to keep the test skipped until<br>&nbsp;&nbsp;the validation parity is implemented.<br>• Long term the XPU path should also fall back cleanly (or raise a clearer<br>&nbsp;&nbsp;error) instead of surfacing a raw oneDNN primitive-descriptor failure. | P3 | No fix PR; CuiYifeng identified upstream check needed. | wincent8 | module: ut, skipped |
| [#2436](https://github.com/intel/torch-xpu-ops/issues/2436) | [upstream_ut] AttributeError: 'NoneType' object has no attribute 'clone' | daisyden | Owner @daisyden to re-check case design (per @CuiYifeng request) | • Wait for the upstream fix in pytorch/pytorch#97395 (guard None .grad in<br>&nbsp;&nbsp;_expanded_weights clone path).<br>• In the meantime redesign the XPU test to explicitly reset/populate grads per<br>&nbsp;&nbsp;sub-test (zero_grad() + forward/backward before the assertion) as requested by<br>&nbsp;&nbsp;CuiYifeng, and keep the skip entry in `skip_list_common.py` labelled `random`<br>&nbsp;&nbsp;until upstream lands. | P3 | No fix PR; behavior matches CUDA per upstream pytorch/pytorch#97395 (CPU issue). | daisyden | skipped, dependency component: communit… |
| [#2531](https://github.com/intel/torch-xpu-ops/issues/2531) | [upstream_ut] AssertionError: Torch not compiled with CUDA enabled | daisyden | Owner @daisyden to triage; only github-actions auto-pass observed | • Triage per test: (a) genuinely CUDA-specific (cufft_plan_cache,<br>&nbsp;&nbsp;ctc_loss_cudnn_*, numeric_check_leak_tunableop_rocm,<br>&nbsp;&nbsp;gemm_bias_offline_tunableop, mm_submatrix_offline_tunableop) - skip in<br>&nbsp;&nbsp;`third_party/torch-xpu-ops/test/xpu/skip_list_common.py` with `CUDA/ROCm<br>&nbsp;&nbsp;only`. (b) port-able (test_sync_warning, test_layer_norm_backwards_eps,<br>&nbsp;&nbsp;test_sort_large_slice) - replace torch.cuda/CudaSyncGuard with an XPU-aware<br>&nbsp;&nbsp;guard (add XpuSyncGuard in `torch/testing/_internal/common_utils.py` or<br>&nbsp;&nbsp;parameterize CudaSyncGuard on device) and change x.to(`cuda`) to x.to(`xpu`)<br>&nbsp;&nbsp;in the XPU copy of the test. (c) quantized qrelu/qgelu - add XPU fallback<br>&nbsp;&nbsp;registration or skip until XPU quantized kernel lands.<br>• Items already passing in nightly (test_ctc_loss_cudnn_tensor_cpu_length) can<br>&nbsp;&nbsp;be removed from skiplist. | P3 | Vector 0 empty; timeline only documentation/tracker cross-refs. | daisyden | skipped, port_from_skiplist |
| [#3041](https://github.com/intel/torch-xpu-ops/issues/3041) | AssertionError: Expected len(flat_diff_results) > 0 in test_fake_crossref_backward_amp_normal_number_mean_xpu_float32 | Silv3S, BartoszKokoszko | Owner @Silv3S, BartoszKokoszko to file fix PR | • In `torch/testing/_internal/common_methods_invocations.py`, extend the<br>&nbsp;&nbsp;existing CUDA DecorateInfo on OpInfo(`normal`, variant=tensor_second) to also<br>&nbsp;&nbsp;skip on device_type=`xpu` (or drop the scalar SampleInputs and keep only the<br>&nbsp;&nbsp;tensor-valued one at line 2010).<br>• Follow-up owner is @daisyden per the latest comment. | P3 | Issue assigned to @Silv3S, BartoszKokoszko; owner to implement fix. | daisyden | ut_upstream |


<a id="sec-3-2-track-pr"></a>
### 3.2 TRACK PR  ·  12 issues

**TRACK PR — a PR is identified; track it to merge, or re-evaluate if prior PRs are dead / unverified**

<a id="sec-3-2-1-flash-attention"></a>
#### 3.2.1 Flash Attention  ·  3 issues

_[↑ Back to Index](#sec-2)_

| Issue | Title | Owner | action_TBD | Fix Approach | Priority | action_reason | Reporter | Labels |
|---|---|---|---|---|---|---|---|---|
| [#2442](https://github.com/intel/torch-xpu-ops/issues/2442) | [Bug Skip]: NotImplementedError: Could not run 'aten::_flash_attention_forward' with arguments from the 'CPU' backend | LuFinch | track PR to merge | • Implement a native XPU _flash_attention_forward SYCL kernel (e.g. under<br>&nbsp;&nbsp;`src/ATen/native/xpu/FlashAttention.cpp`) and register it via<br>&nbsp;&nbsp;`native_functions.yaml`, then remove _flash_attention_forward from<br>&nbsp;&nbsp;XPUFallback.template.<br>• As an interim, change the fallback to explicitly error or use the SDPA math<br>&nbsp;&nbsp;backend and update `skip_list_common.py` until the kernel lands. | P1 | Linked PR(s) open: [3404] | CuiYifeng | skipped |
| [#2853](https://github.com/intel/torch-xpu-ops/issues/2853) | [upstream_ut] torch.ops.aten._flash_attention_forward lack of support for XPU. | LuFinch | wait for PR #3404 review/merge | • Add an XPU registration for aten::_flash_attention_forward in<br>&nbsp;&nbsp;`yaml/native/native_functions.yaml` (XPU dispatch) and implement the wrapper<br>&nbsp;&nbsp;in `src/ATen/native/transformers/xpu/flash_attn/flash_api.cpp` that translates<br>&nbsp;&nbsp;the schema arguments (cum_seq_q/k, max_q/k, is_causal, return_debug_mask,<br>&nbsp;&nbsp;scale) to the existing flash_attention_forward_sycltla call.<br>• Also register _flash_attention_backward similarly, then remove the skip entry. | P1 | Fix PR #3404 OPEN; enables _flash_attention_forward/backward ops; explicit Fix link to #2853. | BBBela | skipped |
| [#2698](https://github.com/intel/torch-xpu-ops/issues/2698) | Title: [upstream_ut] RuntimeError: FlashAttentionForwardXPU only support headdim 64,96,128,192 | chunhuanMeng, LuFinch | Track pytorch/pytorch#180646 to merge | • (1) In _scaled_dot_product_flash_attention_xpu dispatcher, check headdim<br>&nbsp;&nbsp;eligibility before calling run_mha_fwd and fall back to math/efficient SDPA<br>&nbsp;&nbsp;(or reject with can_use_flash_attention=false in the sdp_utils heuristics)<br>&nbsp;&nbsp;when unsupported, rather than TORCH_CHECK-failing inside the kernel<br>• (2) add missing headdim specializations (e.g., 32, 72, 256) in `mha_fwd.cpp`<br>&nbsp;&nbsp;following the existing templated run_mha_fwd_specialized pattern<br>• (3) skip the affected inductor tests on XPU until (1) lands. | P2 | Verified fix PR is OPEN (VERIFIED via github_linked; PR is OPEN). | daisyden | module: inductor, skipped, ut_upstream |


<a id="sec-3-2-2-inductor"></a>
#### 3.2.2 Inductor  ·  3 issues

_[↑ Back to Index](#sec-2)_

| Issue | Title | Owner | action_TBD | Fix Approach | Priority | action_reason | Reporter | Labels |
|---|---|---|---|---|---|---|---|---|
| [#2609](https://github.com/intel/torch-xpu-ops/issues/2609) | [upstream_ut] torch._inductor.exc.InductorError: CppCompileError: C++ compile error | daisyden | Track pytorch/pytorch#171154 to merge | • In pytorch upstream torch/_inductor/codegen (see `cpp_wrapper_gpu.py` / aoti<br>&nbsp;&nbsp;shim generation for custom ops) and torch/csrc/inductor/aoti_torch/, generate<br>&nbsp;&nbsp;the `aoti_torch_xpu_fn_<op>` shim for user-registered XPU custom ops analogous<br>&nbsp;&nbsp;to the CPU path (see `torch/csrc/inductor/aoti_torch/generated/c_shim_cpu.cpp`<br>&nbsp;&nbsp;mechanism).<br>• Alternatively, codegen can fall through to `aoti_torch_cpu_fn_<op>` when the<br>&nbsp;&nbsp;custom op dispatches to CPU/XPU via the ProxyExecutor path.<br>• File a pytorch PR adding XPU custom-op shim generation<br>• in the meantime keep this test skipped.<br>• No torch-xpu-ops kernel change required. | P2 | Verified fix PR is OPEN (VERIFIED via content_match; PR is OPEN). | daisyden | module: inductor, skipped, ut_upstream |
| [#2806](https://github.com/intel/torch-xpu-ops/issues/2806) | CompiledAOTI need XPU support | daisyden | Re-evaluate: verified PR pytorch/pytorch#178385 is CLOSED unmerged | • Ensure torch._inductor.output_code.CompiledAOTI.__post_init__ handles<br>&nbsp;&nbsp;`device_type.startswith('xpu')` by instantiating AOTIModelContainerRunnerXpu<br>&nbsp;&nbsp;(present upstream at torch/csrc/inductor/aoti_runner/*).<br>• Verify the binding is compiled into libtorch_xpu (check<br>&nbsp;&nbsp;`torch/csrc/inductor/aoti_runner/model_container_runner_xpu.cpp` is built).<br>• Enable/unskip test_aot_compile_with_aoti on XPU once pytorch/pytorch#170056<br>&nbsp;&nbsp;lands. | P2 | Linked fix PR was closed without merging; need replacement fix. | daisyden | module: inductor, ut_upstream |
| [#2958](https://github.com/intel/torch-xpu-ops/issues/2958) | AssertionError of test_dtensor_basic_compile | daisyden | Triage: re-validate cross-referenced PRs (none auto-verified) | • Verify on latest pytorch main that the test passes (etaf report), then remove<br>&nbsp;&nbsp;the skip entry from `third_party/torch-xpu-ops/test/xpu/skip_list_common.py`<br>&nbsp;&nbsp;(or equivalent) and close the issue.<br>• If the assertion mismatch still reproduces, regenerate the expected-inline<br>&nbsp;&nbsp;string via `EXPECTTEST_ACCEPT=1 python<br>&nbsp;&nbsp;test/distributed/tensor/test_dtensor_compile.py<br>&nbsp;&nbsp;TestDTensorCompileWithCompiledAutograd.test_dtensor_basic_compile` and<br>&nbsp;&nbsp;upstream the updated expected string (the test is vendored upstream, so no<br>&nbsp;&nbsp;XPU-specific change is needed<br>• just track upstream PR #169867 sync). | P3 | Cross-references exist but no PR explicitly references this issue; manual review needed. | daisyden | module: inductor, ut_upstream |


<a id="sec-3-2-3-sparse"></a>
#### 3.2.3 Sparse  ·  2 issues

_[↑ Back to Index](#sec-2)_

| Issue | Title | Owner | action_TBD | Fix Approach | Priority | action_reason | Reporter | Labels |
|---|---|---|---|---|---|---|---|---|
| [#2283](https://github.com/intel/torch-xpu-ops/issues/2283) | [upstream_ut] sparse._sampled_addmm is not supported | jenniew | wait for PR | • Implement sparse_sampled_addmm for SparseCsrXPU: add a<br>&nbsp;&nbsp;sampled_addmm_out_sparse_csr_xpu kernel in<br>&nbsp;&nbsp;`src/ATen/native/sparse/xpu/sycl/SparseCsrTensorMathKernels.cpp` backed by<br>&nbsp;&nbsp;oneMKL sparse SDDMM (mkl::sparse::sddmm) or a hand-written SYCL kernel<br>&nbsp;&nbsp;iterating the CSR sparsity pattern of the output tensor, register it in<br>&nbsp;&nbsp;`SparseCsrTensorMath.cpp` via TORCH_LIBRARY_IMPL(aten, SparseCsrXPU, ...) for<br>&nbsp;&nbsp;sparse_sampled_addmm and sparse_sampled_addmm.out, and remove the<br>&nbsp;&nbsp;corresponding skips from `test/xpu/skip_list_common.py` once passing. | P2 | Verified PR(s) open: intel/torch-xpu-ops#3018 | daisyden | skipped, ut_upstream |
| [#2229](https://github.com/intel/torch-xpu-ops/issues/2229) | test/test_sparse_csr.py::TestSparseCompressedCPU::test_invalid_input meet message not match | jenniew | Move PR #3073 out of WIP and land | • Update the XPU/sparse test expectations in torch-xpu-ops skip/override list to<br>&nbsp;&nbsp;match the actual emitted message (regex `device of .* must match device of`),<br>&nbsp;&nbsp;or upstream a fix so sparse_compressed_tensor validation uses the canonical<br>&nbsp;&nbsp;`Expected all tensors to be on the same device` message from c10 before<br>&nbsp;&nbsp;emitting the component-level error.<br>• Preferred: adjust the test/error-regex in test/xpu reference to accept both<br>&nbsp;&nbsp;phrasings. | P3 | Open WIP PR by assignee jenniew with explicit Related issue: #2229; root cause identified as missing _validate_compressed_sparse_indices on… | wincent8 | skipped |


<a id="sec-3-2-4-torch-operations"></a>
#### 3.2.4 Torch Operations  ·  4 issues

_[↑ Back to Index](#sec-2)_

| Issue | Title | Owner | action_TBD | Fix Approach | Priority | action_reason | Reporter | Labels |
|---|---|---|---|---|---|---|---|---|
| [#2615](https://github.com/intel/torch-xpu-ops/issues/2615) | [Bug Skip]: New failures RuntimeError: Unsupported dtype Half / RuntimeError: Unsupported dtype torch.float16 | CuiYifeng | Track intel/torch-xpu-ops#2996 to merge \| Re-evaluate: verified PR pytorch/pytorch#171231 is CLOSE… | • Extend the Half/ComplexHalf promotion wrapper in `mkl/SpectralOps.cpp` to<br>&nbsp;&nbsp;cover all public fft entry points (_fft_c2c, _fft_r2c, _fft_c2r), including<br>&nbsp;&nbsp;the hfft2/ihfft2/ihfftn variants that currently fall through to the error<br>&nbsp;&nbsp;path.<br>• Add a common helper that casts input Half→Float / ComplexHalf→ComplexFloat<br>&nbsp;&nbsp;before calling `_exec_fft`, runs DFT, and casts the output back.<br>• Also register float16 support in op_db through an XPU override so decomp/ref<br>&nbsp;&nbsp;tests exercise the promotion path.<br>• Alternative short term: mark the remaining failing tests as skipped with<br>&nbsp;&nbsp;`@skipIfNoFP16FFT` referencing oneMKL limitation. | P2 | Verified fix PR is OPEN (VERIFIED via explicit_reference; PR is OPEN). \| Linked fix PR was closed without merging; need replacement fix. | kaileiyx | module: ut, skipped |
| [#2783](https://github.com/intel/torch-xpu-ops/issues/2783) | [Bug Skip]: Key "xpu" is missing from dict "driver" in test_svd | daisyden | Re-evaluate: verified PR pytorch/pytorch#172824 is CLOSED unmerged | • Upstream PR to `test/test_linalg.py`: add an `xpu`: (None,) entry to the<br>&nbsp;&nbsp;drivers dict (and similarly in any other svd/lstsq helper that keys on device<br>&nbsp;&nbsp;type)<br>• alternatively use drivers.get(type, (None,)).<br>• Until that lands, keep the four test_svd_xpu_* cases in<br>&nbsp;&nbsp;third_party/torch-xpu-ops/test/xpu/skip_list_common.py.<br>• Track PR #172824 to close this out. | P2 | Linked fix PR was closed without merging; need replacement fix. | CuiYifeng | module: ut, skipped, dependency compone… |
| [#1893](https://github.com/intel/torch-xpu-ops/issues/1893) | [upstream_ut] oneDNN accuracy issues in test_ops_xpu.py | chunhuanMeng | Land pytorch/pytorch#179125 (addmv stride preservation) and consider tolerance changes (per @chuanq… | • Bump tolerance for the remaining mv/addmv fp32 ops in XPU OpInfo<br>&nbsp;&nbsp;(torch-xpu-ops `test/xpu/xpu_test_utils.py` or a toleranceOverride for<br>&nbsp;&nbsp;mv/addmv on XPU) rather than modifying kernels, since the gap is within<br>&nbsp;&nbsp;expected fp32 accumulation error for oneDNN gemv.<br>• Remove now-passing cases from the skip list and add a comment linking this<br>&nbsp;&nbsp;issue. | P3 | Vector 0 link; addmv stride fix landing addresses one of the oneDNN accuracy paths. | daisyden | skipped, ut_upstream |
| [#3033](https://github.com/intel/torch-xpu-ops/issues/3033) | [Bug Skip]: Softmax tolerance | chunhuanMeng | Track PR intel/torch-xpu-ops#3036 to merge | • Short-term: the case is already listed in the XPU skip_list_common for<br>&nbsp;&nbsp;extended op tests (PR merged on 2026-03-11)<br>• the `random` label only prevents auto-close.<br>• Long-term: in torch-xpu-ops `src/ATen/native/xpu/sycl/SoftMaxKernels.cpp`,<br>&nbsp;&nbsp;align the bool-input softmax numerics with CPU (cast bool->float in fp32 and<br>&nbsp;&nbsp;match CPU's reduction order/epsilon) or relax the tolerance for bool dtype in<br>&nbsp;&nbsp;`extended/test_ops_xpu.py`, then remove the skip. | P3 | Phase 4b backfill: verified open PR from pr_analysis. | chunhuanMeng | skipped, random |


<a id="sec-3-3-needs-owner"></a>
### 3.3 NEEDS_OWNER  ·  0 issues

**NEEDS_OWNER — awaiting triage-lead to assign an owner**


<a id="sec-4"></a>
## 4. QA

_[↑ Back to Index](#sec-2)_

Issues in this section are ready for QA action (close, verify, reply, etc.). Rows sorted by `Priority` (P0 → P3).

<a id="sec-4-1-close-or-skip"></a>
### 4.1 CLOSE or SKIP  ·  12 issues

**CLOSE or SKIP — terminal QA action (close fixed, verify merged fix, skip not-target/wontfix, or label not_target and close)**

| Issue | Title | Owner | action_TBD | Fix Approach | Priority | action_reason | Reporter | Labels |
|---|---|---|---|---|---|---|---|---|
| [#2802](https://github.com/intel/torch-xpu-ops/issues/2802) | Three aten._scaled_dot_product_flash_attention issues | daisyden | label not_target (partial: 1 of 3 sub-cases) \| verify PR #3289 fix and close remaining cases | • (1) Relax the attn_mask+is_causal check for efficient_attention on XPU<br>&nbsp;&nbsp;(`Attention.cpp`) or fix the test to not pass both. (2) Align fake/meta kernel<br>&nbsp;&nbsp;for aten._scaled_dot_product_flash_attention with the XPU real implementation<br>&nbsp;&nbsp;output tuple — specifically the 5th output (cum_seq_len / philox seed layout)<br>• adjust `torch/_meta_registrations.py` or the XPU registration in<br>&nbsp;&nbsp;`third_party/torch-xpu-ops/src/ATen/native/transformers/xpu/flash_attn/flash_api.cpp`:33<br>&nbsp;&nbsp;to match the canonical schema, or ensure XPU path falls back to<br>&nbsp;&nbsp;fused_attention_overrideable for fake propagation. (3) Add XPU-specific<br>&nbsp;&nbsp;expected-inline in `test_export.py` or generalize the test to accept either<br>&nbsp;&nbsp;SDPA op variant depending on device. | P1 | 1 sub-case won't-fix; PR #3289 MERGED for fake-tensor case; 3rd case is XPU backend priority explanation. | daisyden | module: inductor, ut_upstream |
| [#3143](https://github.com/intel/torch-xpu-ops/issues/3143) | NotImplementedError: The operator 'aten::_scaled_dot_product_efficient_attention_backward' is not currently implemented for the XPU device. | daisyden | assignee verify fix and close issue | • Already addressed by intel/torch-xpu-ops PR #3317 which adds the XPU<br>&nbsp;&nbsp;registration for aten::_scaled_dot_product_efficient_attention_backward by<br>&nbsp;&nbsp;routing it to the existing mha_bwd sycltla kernel.<br>• Action: verify the PR (@daisyden is asked to re-run the listed cudagraph test<br>&nbsp;&nbsp;cases), then close the issue and unskip the tests in test/xpu/skip_list_*.py. | P1 | Fix PR #3317 MERGED 2026-04-17; daisyden to validate and close. | daisyden | module: ut, skipped, ut_upstream |
| [#2720](https://github.com/intel/torch-xpu-ops/issues/2720) | [upstream_ut] RuntimeError: false INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/native/DispatchStub.cpp":275 | CuiYifeng | Close the fixed issue | • Align<br>&nbsp;&nbsp;`third_party/torch-xpu-ops/src/ATen/native/xpu/sycl/BinaryMiscOpsKernels.cpp`:158<br>&nbsp;&nbsp;ldexp_kernel with the CUDA implementation<br>&nbsp;&nbsp;(`aten/src/ATen/native/cuda/BinaryMiscOpsKernels.cu` ldexp_kernel_cuda):<br>&nbsp;&nbsp;mirror the exact AT_DISPATCH template/dtype list it uses for the int-exponent<br>&nbsp;&nbsp;iterator, and verify REGISTER_XPU_DISPATCH(ldexp_stub, ...) in<br>&nbsp;&nbsp;`BinaryOps.cpp`:80 is linked into the XPU library (add an explicit reference<br>&nbsp;&nbsp;in `Ops.cpp` registry if stripped).<br>• Then remove the WA skip added in pytorch PR #171238. | P2 | Fixed and passed in CI | wincent8 | skipped |
| [#3132](https://github.com/intel/torch-xpu-ops/issues/3132) | [upstream_ut] transfomers test reports RuntimeError: No available kernel. Aborting execution. | LuFinch | Label issue not_target (partial); owner: @LuFinch \| Issue needs owner investigation; only weak/clo… | • For the singleton-stride case, land/forward-port pytorch/pytorch#179800 which<br>&nbsp;&nbsp;flips `ignore_singleton_dim` to true in `SDPUtils.cpp`:77-78 (mirroring CUDA<br>&nbsp;&nbsp;behavior).<br>• Skip all cuDNN-specific cases (no XPU cuDNN backend) and all nested-tensor<br>&nbsp;&nbsp;fused-kernel cases (tracked under #3133).<br>• Skip the determinism-warning case until an SDPA backend with deterministic<br>&nbsp;&nbsp;support is available<br>• record all skips in `test/xpu/skip_list_common.py` with issue links. | P2 | Partial not-target decision found in authoritative comment(s) \| Timeline cross-refs all closed unmerged with weak verdict; no actionable PR | daisyden | module: ut, skipped, ut_upstream |
| [#3141](https://github.com/intel/torch-xpu-ops/issues/3141) | [upstream_ut] RuntimeError: FlashAttentionForwardXPU only support headdim 64,96,128,192 | LuFinch | Verify fix from merged PR intel/torch-xpu-ops#3355; close issue if validated | • Two-layer fix: (1) at the dispatcher in aten/native/transformers/xpu choose<br>&nbsp;&nbsp;the math / mem-efficient SDPA backend when headdim is outside the supported<br>&nbsp;&nbsp;set instead of forcing FlashAttention (the sdp::can_use_flash_attention<br>&nbsp;&nbsp;heuristic needs an XPU-aware headdim check), as partially done in<br>&nbsp;&nbsp;pytorch/pytorch#180646<br>• (2) long-term, extend `mha_fwd.cpp` to add specialized TileShape<br>&nbsp;&nbsp;instantiations for headdim 32 (and any other required dims) so FlashAttention<br>&nbsp;&nbsp;is actually used instead of falling back. | P2 | PR intel/torch-xpu-ops#3355 merged (github_linked) | daisyden | module: ut, skipped, ut_upstream |
| [#3163](https://github.com/intel/torch-xpu-ops/issues/3163) | [Bug Skip]: Object comparison failed: torch.int64 != torch.int32 in test_sparse_add | chunhuanMeng | Verify fix from merged PR intel/torch-xpu-ops#3341; close issue if validated | • In add_out_sparse_compressed_xpu, after constructing<br>&nbsp;&nbsp;out_dense.to_sparse_csr(), cast crow_indices and col_indices to the index<br>&nbsp;&nbsp;dtype of the inputs (self.crow_indices().scalar_type()) before assigning to<br>&nbsp;&nbsp;out.<br>• Better: implement a native CSR+CSR add that preserves index dtype (mirroring<br>&nbsp;&nbsp;the CPU impl in `aten/src/ATen/native/sparse/SparseCsrTensorMath.cpp`) instead<br>&nbsp;&nbsp;of the dense round-trip, which is also inefficient.<br>• Keep the dense fallback only for CSR+strided mixed inputs. | P2 | PR intel/torch-xpu-ops#3341 merged (explicit_reference) | CuiYifeng | skipped, ut_upstream |
| [#2169](https://github.com/intel/torch-xpu-ops/issues/2169) | Frame size comparison failed in test_size_comparison_no_recompile | guangyey | Verify on latest main and close (per @guangyey already fixed) | • Re-run test_size_comparison_no_recompile on current main with a fresh XPU<br>&nbsp;&nbsp;wheel to confirm the fix, then remove the TEST_XPU skip decorator at<br>&nbsp;&nbsp;`test/test_dynamic_shapes.py`:3380-3382 and close the issue.<br>• If it still fails, bisect the dynamo ShapeEnv / GuardManager paths for a<br>&nbsp;&nbsp;device-dependent guard on sym_size that fires only on XPU. | P3 | @guangyey reports it is fixed on latest main; Vector 0 PR pytorch/pytorch#178780 covers the device fix. | daisyden | skipped |
| [#2253](https://github.com/intel/torch-xpu-ops/issues/2253) | the supported dtypes are not align with cuda | daisyden | Verify alignment landed and close (duplicate of #2289) | • Update dtypesIfXPU in upstream opinfo definitions for each failing op to match<br>&nbsp;&nbsp;the dtype set actually supported by the XPU/oneDNN backend (drop int64 for<br>&nbsp;&nbsp;GEMMs, drop complex/low-precision for conv where unsupported, set<br>&nbsp;&nbsp;histogram/histogramdd to the CPU-fallback dtype set).<br>• This is the remaining portion of PR #161246.<br>• No new kernel work required<br>• validate via `test_ops_xpu.py`::TestCommonXPU::test_dtypes_* after merge. | P3 | @daisyden noted issue is duplicate of #2289 and dtypes aligned with CUDA via pytorch/pytorch#161246 commit; nightly tests passing. | daisyden | duplicate, skipped, ut_upstream |
| [#2697](https://github.com/intel/torch-xpu-ops/issues/2697) | Title: [upstream_ut] RuntimeError: Expected to find ", 0, " but did not find it | chunhuanMeng | label not_target and close | • Short term: skip this test for XPU (add @requires_cuda or xfail_if_xpu) since<br>&nbsp;&nbsp;the codegen signature it checks is CUDA-EFFICIENT_ATTENTION-specific.<br>• Long term: add XPU support for _scaled_dot_product_efficient_attention (or<br>&nbsp;&nbsp;have XPU's SDPA lowering reuse the same padded-bias expansion pass) so that<br>&nbsp;&nbsp;the EFFICIENT_ATTENTION backend produces the expected wrapper code on XPU. | P3 | Issue assignee (MEMBER) chunhuanMeng made firm statement: 'We cannot fix this because our efficient attention falls back to math, which has… | daisyden | module: inductor, skipped, ut_upstream |
| [#2999](https://github.com/intel/torch-xpu-ops/issues/2999) | KeyError: 'eager_numerics.use_pytorch_libdevice' | daisyden | Skip issue | • Confirmed not_target label: the assignee has decided to permanently skip these<br>&nbsp;&nbsp;tests on XPU.<br>• Add the four test_bitwise_adam*_capturable_foreach_xpu tests to<br>&nbsp;&nbsp;`skip_list_common.py` (inductor section) with a comment pointing to the 1-ULP<br>&nbsp;&nbsp;pow mismatch between Intel Triton libdevice and sycl::pow.<br>• Alternatively, request an upstream option in Intel Triton that overrides the<br>&nbsp;&nbsp;bundled libdevice with a PyTorch-supplied .bc (analogous to<br>&nbsp;&nbsp;use_pytorch_libdevice on CUDA) and wire it through knobs.intel.libdevice_path<br>• without that, bitwise parity is impossible. | P3 | not target feature | daisyden | module: inductor, ut_upstream, not_targ… |
| [#3131](https://github.com/intel/torch-xpu-ops/issues/3131) | [upstream_ut] NotImplementedError: The operator 'aten::_scaled_dot_product_efficient_attention_backward' is not c | daisyden | Verify fix from merged PR intel/torch-xpu-ops#3367 and close | • Already fixed via intel/torch-xpu-ops#3367 which added proper bias-grad<br>&nbsp;&nbsp;validation (raising the expected RuntimeError) in the XPU mem-efficient<br>&nbsp;&nbsp;attention backward wrapper.<br>• Close the issue<br>• no further action needed beyond confirming CI pass. | P3 | Phase 4b backfill: verified merged PR from pr_analysis. | daisyden | module: ut, skipped, ut_upstream |
| [#3170](https://github.com/intel/torch-xpu-ops/issues/3170) | Unskip test_bmm_windows_error_xpu_float64 | libohao1201, jafraustro | Verify fix from merged PR intel/torch-xpu-ops#3225 and close | • In `third_party/torch-xpu-ops/test/xpu/test_sparse_xpu.py`:1970 remove the<br>&nbsp;&nbsp;`@unittest.skipIf(TEST_XPU, ...)` decorator (leaving the pre-existing<br>&nbsp;&nbsp;Windows/CUDA gating intact so the test is a no-op off-Windows).<br>• Verify on Windows XPU runner per daisyden's comment before closing. | P3 | Phase 4b backfill: verified merged PR from pr_analysis. | CuiYifeng | skipped, ut_upstream |


<a id="sec-4-2-await-reply"></a>
### 4.2 AWAIT_REPLY  ·  0 issues

**AWAIT_REPLY — open questions in thread; owner must respond**

_No issues._


<a id="sec-4-3-monitor"></a>
### 4.3 MONITOR  ·  0 issues

**MONITOR — long-running tracker / maintenance / scoping**

_No issues._


<a id="sec-4-4-check-cases"></a>
### 4.4 CHECK_CASES  ·  4 issues

**CHECK_CASES — XPU test case missing in repo; QA must verify case existence before action**

| Issue | Title | Owner | action_TBD | Fix Approach | Priority | action_reason | Reporter | Labels |
|---|---|---|---|---|---|---|---|---|
| [#2285](https://github.com/intel/torch-xpu-ops/issues/2285) | Support efficient attention | chunhuanMeng | check_case_avaliablity | • Implement _efficient_attention_forward/backward for XPU via a new SYCL kernel<br>&nbsp;&nbsp;(e.g.<br>&nbsp;&nbsp;`third_party/torch-xpu-ops/src/ATen/native/xpu/sycl/AttentionEfficient.cpp`)<br>&nbsp;&nbsp;and register in `native_functions.yaml/xpu_ops.yaml` dispatch keys (XPU,<br>&nbsp;&nbsp;Meta), reusing the existing Flash-Attention kernels under<br>&nbsp;&nbsp;src/ATen/native/xpu/sycl/Attention*.cpp as a starting point.<br>• For the quantized/view test cases, port CUDA test variants to XPU naming (drop<br>&nbsp;&nbsp;_cuda suffix or add XPU to DEVICES list in `test_quantized_tensor_xpu.py`) and<br>&nbsp;&nbsp;enable test_ravel_xpu/test_flatten_xpu in test_view_ops_xpu.py.<br>• Track alongside duplicate issue #2853. | P1 |  | daisyden | skipped |
| [#2578](https://github.com/intel/torch-xpu-ops/issues/2578) | [TorchAO][UT] test/quantization/test_quant_api.py AssertionError: SQNR -2.90625 is too low | Stonepia | check_case_avaliablity | • Either (a) implement TensorCoreTiledLayout support in<br>&nbsp;&nbsp;_convert_weight_to_int4pack_xpu (`WeightInt4Pack.cpp`:17) by honoring<br>&nbsp;&nbsp;`innerKTiles` and packing with the same K-tile interleave as CUDA, and<br>&nbsp;&nbsp;correspondingly update the XPU dequant/gemm kernels<br>&nbsp;&nbsp;(`sycl/Dequant_int4.cpp`:95, `sycl/LinearInt4.cpp`:220)<br>• or (b) register an XPU-specific layout in torchao (like the existing BMG/XPU<br>&nbsp;&nbsp;`Int4XPULayout`) and dispatch Int4WeightOnlyConfig to it on XPU so the packed<br>&nbsp;&nbsp;data matches what _weight_int4pack_mm_xpu expects.<br>• Add a numerics test mirroring test_workflow_e2e_numerics for xpu checking<br>&nbsp;&nbsp;SQNR>=16.5. | P1 |  | zxd1997066 | module: ao, ut_upstream |
| [#2376](https://github.com/intel/torch-xpu-ops/issues/2376) | [Bug Skip]: NotImplementedError: "logaddexp_xpu" not implemented for 'Complex' | daisyden, CuiYifeng | check_case_avaliablity | • Remove the XPU-specific skips/WAs added in pytorch/pytorch#171238<br>&nbsp;&nbsp;(`test/test_binary_ufuncs.py`, `test_decomp.py`, `test_meta.py` OpInfo<br>&nbsp;&nbsp;entries) and update torch-xpu-ops skip lists (`test/xpu/skip_list_common.py` /<br>&nbsp;&nbsp;`test_binary_ufuncs_xpu.py`) to re-enable logaddexp<br>&nbsp;&nbsp;complex64/complex128/complex32 cases.<br>• Re-run the 60+ listed op_ut cases to confirm<br>• if complex32 remains numerically unstable, keep only that dtype skipped and<br>&nbsp;&nbsp;file a narrower follow-up.<br>• Coordinate with CuiYifeng (kernel owner) to land the test-enable PR. | P2 |  | mengfei25 | module: ut, skipped |
| [#2512](https://github.com/intel/torch-xpu-ops/issues/2512) | [upstream_ut] RuntimeError: _histc_xpu does not have a deterministic implementation, but you set 'torch.use_deter | chunhuanMeng | check_case_avaliablity | • Edit `third_party/torch-xpu-ops/src/ATen/native/xpu/SummaryOps.cpp`:_histc_xpu<br>&nbsp;&nbsp;to call `globalContext().alertNotDeterministic("_histc_xpu with floating point<br>&nbsp;&nbsp;input")` only when `self.is_floating_point()`, mirroring<br>&nbsp;&nbsp;`cuda/SummaryOps.cu`:405-411.<br>• PR #3333 is already in flight implementing this<br>• merge and close the issue. | P3 |  | libohao1201 | skipped |


<a id="sec-5"></a>
## 5. Duplicated issues

_[↑ Back to Index](#sec-2)_

Rows where `duplicated_issue` is set or `action_TBD` contains "duplicate of".  —  5 issues.

| Issue | Duplicates | Title | Owner | action_TBD | Fix Approach | Priority | action_reason | Reporter | Labels |
|---|---|---|---|---|---|---|---|---|---|
| [#2285](https://github.com/intel/torch-xpu-ops/issues/2285) | [#2853](https://github.com/intel/torch-xpu-ops/issues/2853) | Support efficient attention | chunhuanMeng | check_case_avaliablity | • Implement _efficient_attention_forward/backward for XPU via a new SYCL kernel<br>&nbsp;&nbsp;(e.g.<br>&nbsp;&nbsp;`third_party/torch-xpu-ops/src/ATen/native/xpu/sycl/AttentionEfficient.cpp`)<br>&nbsp;&nbsp;and register in `native_functions.yaml/xpu_ops.yaml` dispatch keys (XPU,<br>&nbsp;&nbsp;Meta), reusing the existing Flash-Attention kernels under<br>&nbsp;&nbsp;src/ATen/native/xpu/sycl/Attention*.cpp as a starting point.<br>• For the quantized/view test cases, port CUDA test variants to XPU naming (drop<br>&nbsp;&nbsp;_cuda suffix or add XPU to DEVICES list in `test_quantized_tensor_xpu.py`) and<br>&nbsp;&nbsp;enable test_ravel_xpu/test_flatten_xpu in test_view_ops_xpu.py.<br>• Track alongside duplicate issue #2853. | P1 |  | daisyden | skipped |
| [#2853](https://github.com/intel/torch-xpu-ops/issues/2853) | [#2285](https://github.com/intel/torch-xpu-ops/issues/2285) | [upstream_ut] torch.ops.aten._flash_attention_forward lack of support for XPU. | LuFinch | wait for PR #3404 review/merge | • Add an XPU registration for aten::_flash_attention_forward in<br>&nbsp;&nbsp;`yaml/native/native_functions.yaml` (XPU dispatch) and implement the wrapper<br>&nbsp;&nbsp;in `src/ATen/native/transformers/xpu/flash_attn/flash_api.cpp` that translates<br>&nbsp;&nbsp;the schema arguments (cum_seq_q/k, max_q/k, is_causal, return_debug_mask,<br>&nbsp;&nbsp;scale) to the existing flash_attention_forward_sycltla call.<br>• Also register _flash_attention_backward similarly, then remove the skip entry. | P1 | Fix PR #3404 OPEN; enables _flash_attention_forward/backward ops; explicit Fix link to #2853. | BBBela | skipped |
| [#2024](https://github.com/intel/torch-xpu-ops/issues/2024) | [#2444](https://github.com/intel/torch-xpu-ops/issues/2444) | AssertionError: Torch not compiled with CUDA enabled | daisyden | Assignee @daisyden to investigate | • Either (a) add these test IDs to `test/xpu/skip_list_common.py` with a clear<br>&nbsp;&nbsp;TODO so CI stays green, or (b) preferred: send an upstream PR replacing<br>&nbsp;&nbsp;hard-coded `cuda` with instantiate_device_type_tests / self.device_type so the<br>&nbsp;&nbsp;tests run on XPU.<br>• For test_pool3d_large_size_int64 and test_pooling_large, also check whether<br>&nbsp;&nbsp;XPU has the 49GB device memory headroom they need before enabling.<br>• Close as duplicate of #2444 after the skip list lands. | P2 | Issue already assigned to @daisyden; owner to lead root-cause. | mengfei25 | module: ut, skipped |
| [#2715](https://github.com/intel/torch-xpu-ops/issues/2715) | [#3286](https://github.com/intel/torch-xpu-ops/issues/3286) | [upstream_ut] torch._dynamo.exc.Unsupported: Attempted to inline function marked as skipped | CuiYifeng |  | • In `torch/_dynamo/trace_rules.py`, remove torch.xpu from MOD_SKIPLIST (or add<br>&nbsp;&nbsp;torch.xpu.device to the allowed-callable list analogous to torch.cuda.device)<br>&nbsp;&nbsp;so dynamo can inline device context-manager __init__ for XPU, matching CUDA<br>&nbsp;&nbsp;behavior.<br>• Alternatively add a polyfill/TorchInGraphFunctionVariable mapping for<br>&nbsp;&nbsp;torch.xpu.device.<br>• Track via duplicate #3286. | P2 |  | daisyden | skipped, ut_upstream |
| [#2253](https://github.com/intel/torch-xpu-ops/issues/2253) | [#2289](https://github.com/intel/torch-xpu-ops/issues/2289) | the supported dtypes are not align with cuda | daisyden | Verify alignment landed and close (duplicate of #2289) | • Update dtypesIfXPU in upstream opinfo definitions for each failing op to match<br>&nbsp;&nbsp;the dtype set actually supported by the XPU/oneDNN backend (drop int64 for<br>&nbsp;&nbsp;GEMMs, drop complex/low-precision for conv where unsupported, set<br>&nbsp;&nbsp;histogram/histogramdd to the CPU-fallback dtype set).<br>• This is the remaining portion of PR #161246.<br>• No new kernel work required<br>• validate via `test_ops_xpu.py`::TestCommonXPU::test_dtypes_* after merge. | P3 | @daisyden noted issue is duplicate of #2289 and dtypes aligned with CUDA via pytorch/pytorch#161246 commit; nightly tests passing. | daisyden | duplicate, skipped, ut_upstream |


<a id="sec-6"></a>
## 6. Dependency (external blockers)

_[↑ Back to Index](#sec-2)_

Issues with a non-blank `Dependency` value, excluding `upstream-pytorch`, `CPU fallback`, and `SYCL kernel:*` (in-repo kernel pointers). Terminal-QA rows (CLOSE / VERIFY_AND_CLOSE / SKIP / NOT_TARGET_CLOSE) are also excluded.  —  13 issues.

| Issue | Dependency | Title | Owner | action_TBD | Fix Approach | Priority | action_reason | Reporter | Labels |
|---|---|---|---|---|---|---|---|---|---|
| [#3165](https://github.com/intel/torch-xpu-ops/issues/3165) | triton | test_sparse_csr_xpu.py::TestSparseCompressedTritonKernelsXPU::test_triton_bsr_softmax meet RuntimeError: ZE_RESULT_ERROR_INVALID_KERNEL_NAME | jafraustro | assignee investigate (split-out from #2209) | • File/track a pytorch-triton-xpu issue with the failing kernel reproducer and,<br>&nbsp;&nbsp;once fixed upstream, bump the triton pin.<br>• In the meantime, gate bsr_softmax on XPU behind a device check that either (a)<br>&nbsp;&nbsp;uses the non-Triton dense-softmax fallback path similar to the CUDA fallback<br>&nbsp;&nbsp;in `_triton_ops.py`, or (b) skips the test with a clear xfail pointing to the<br>&nbsp;&nbsp;triton issue.<br>• The Triton kernel itself (_bsr_softmax_kernel) may also need to be simplified<br>&nbsp;&nbsp;(avoid unsupported tl primitives) to make it lower-able on XPU. | P1 | Skipped Triton sparse softmax case; sub-issue split from #2209; assigned to @jafraustro. | CuiYifeng | skipped, ut_upstream |
| [#2800](https://github.com/intel/torch-xpu-ops/issues/2800) | driver | AttributeError: 'torch._C._XpuDeviceProperties' object has no attribute 'major' | guangyey |  | • Short-term: add a test-side guard that only queries `.major` on CUDA (e.g.<br>&nbsp;&nbsp;skip/branch when device.type==`xpu` in<br>&nbsp;&nbsp;`test_scaled_matmul_cuda.py`::test_scaled_mm_vs_emulated) and/or expose a<br>&nbsp;&nbsp;dummy `major`/`minor` property_readonly on _XpuDeviceProperties returning -1<br>&nbsp;&nbsp;so that inequality checks still skip the Blackwell-only path.<br>• Long-term: once DPC++ 2026.2 lands CMPLRLLVM-72166, map the new arch-version<br>&nbsp;&nbsp;API to .major/.minor in `torch/csrc/xpu/Module.cpp`<br>&nbsp;&nbsp;registerXpuDeviceProperties and update python-side `torch/xpu/__init__.py` to<br>&nbsp;&nbsp;surface them. | P2 |  | daisyden | dependency component: oneAPI, module: i… |
| [#3142](https://github.com/intel/torch-xpu-ops/issues/3142) | driver | [upstream_ut] RuntimeError: The sycl_ext_oneapi_work_group_scratch_memory feature is not yet available for use with SYCL Graph extension. | LuFinch | Issue needs owner investigation; only weak/closed cross-ref candidates found | • Pure oneAPI/driver-side fix: wait for oneAPI 2026.0 which lifts the graph<br>&nbsp;&nbsp;restriction on work_group_scratch_memory (per LuFinch/daisyden).<br>• Meanwhile, keep these cudagraph SDPA cases skipped in<br>&nbsp;&nbsp;`test/xpu/skip_list_common.py` with a reference to CMPLRLLVM-72057, and<br>&nbsp;&nbsp;optionally teach the torch.cuda.graph replacement for XPU to fall back to the<br>&nbsp;&nbsp;math SDPA backend when the compiled stream has an active SYCL-Graph capture so<br>&nbsp;&nbsp;that user code does not crash. | P2 | Timeline cross-refs all closed unmerged with weak verdict; no actionable PR | daisyden | dependency component: oneAPI, module: u… |
| [#3136](https://github.com/intel/torch-xpu-ops/issues/3136) | oneDNN | [upstream_ut] AssertionError: False is not true in test_transformers | LuFinch | Issue needs owner investigation; only weak/closed cross-ref candidates found | • For test_disable_fastpath_xpu, land/forward-port pytorch/pytorch#179701 and<br>&nbsp;&nbsp;re-enable.<br>• For the two fused_sdp_priority_order cases, add them to the XPU skip list in<br>&nbsp;&nbsp;`test/xpu/skip_list_common.py` (cuDNN is not a supported XPU backend)<br>• optionally filter `SDPBackend.CUDNN_ATTENTION` out of the expected priority<br>&nbsp;&nbsp;list in a future upstream XPU-aware test refactor. | P2 | Timeline cross-refs all closed unmerged with weak verdict; no actionable PR | daisyden | module: ut, skipped, ut_upstream |
| [#3137](https://github.com/intel/torch-xpu-ops/issues/3137) | oneDNN | [upstream_ut] RuntimeError: expected scalar type Half but found Float | LuFinch | Issue needs owner investigation; only weak/closed cross-ref candidates found | • Merge/forward-port pytorch/pytorch#179701 so `_transformer_encoder_layer_fwd`<br>&nbsp;&nbsp;takes the slow path (or casts parameters) under xpu autocast, then re-enable<br>&nbsp;&nbsp;the four skipped cases.<br>• Track the accuracy delta under a separate oneDNN-SDPA-training follow-up. | P2 | Timeline cross-refs all closed unmerged with weak verdict; no actionable PR | daisyden | module: ut, skipped, ut_upstream, random |
| [#2615](https://github.com/intel/torch-xpu-ops/issues/2615) | oneMKL | [Bug Skip]: New failures RuntimeError: Unsupported dtype Half / RuntimeError: Unsupported dtype torch.float16 | CuiYifeng | Track intel/torch-xpu-ops#2996 to merge \| Re-evaluate: verified PR pytorch/pytorch#171231 is CLOSE… | • Extend the Half/ComplexHalf promotion wrapper in `mkl/SpectralOps.cpp` to<br>&nbsp;&nbsp;cover all public fft entry points (_fft_c2c, _fft_r2c, _fft_c2r), including<br>&nbsp;&nbsp;the hfft2/ihfft2/ihfftn variants that currently fall through to the error<br>&nbsp;&nbsp;path.<br>• Add a common helper that casts input Half→Float / ComplexHalf→ComplexFloat<br>&nbsp;&nbsp;before calling `_exec_fft`, runs DFT, and casts the output back.<br>• Also register float16 support in op_db through an XPU override so decomp/ref<br>&nbsp;&nbsp;tests exercise the promotion path.<br>• Alternative short term: mark the remaining failing tests as skipped with<br>&nbsp;&nbsp;`@skipIfNoFP16FFT` referencing oneMKL limitation. | P2 | Verified fix PR is OPEN (VERIFIED via explicit_reference; PR is OPEN). \| Linked fix PR was closed without merging; need replacement fix. | kaileiyx | module: ut, skipped |
| [#2329](https://github.com/intel/torch-xpu-ops/issues/2329) | triton | [upstream_ut] feature missing: get_device_tflops and get_drams_gbps | etaf | Assignee @etaf to investigate | • Add XPU support in `torch/_inductor/utils.py` get_device_tflops() and<br>&nbsp;&nbsp;get_dram_gbps(): detect device.type==`xpu` and compute peak TFLOPS from<br>&nbsp;&nbsp;torch.xpu.get_device_properties (EU count * SIMD width * frequency * ops/cycle<br>&nbsp;&nbsp;per dtype) and DRAM BW from hbm_bandwidth / memory_bus_width once exposed by<br>&nbsp;&nbsp;intel-xpu-backend-for-triton#5792 or a static PVC/BMG table as an interim.<br>• Route the flop-counter scheduler test through torch.accelerator instead of<br>&nbsp;&nbsp;torch.cuda hardcoding, then re-enable<br>&nbsp;&nbsp;test_flop_counter_op_options0_xpu_float16. | P2 | Issue already assigned to @etaf; owner to lead root-cause. | daisyden | duplicate, dependency component: Triton… |
| [#2554](https://github.com/intel/torch-xpu-ops/issues/2554) | triton | [upstream_ut] AssertionError: AssertionError not raised | daisyden |  | • No change needed in torch-xpu-ops<br>• wait for intel-xpu-backend-for-triton fix (#5654) to land and bump the pinned<br>&nbsp;&nbsp;Triton commit in PyTorch (.ci/docker/ci_commit_pins/xpu-triton.txt).<br>• In the meantime, keep these three tests in the XPU skip list, and add a<br>&nbsp;&nbsp;verification step in the UT pipeline to remove the skip once the Triton fix is<br>&nbsp;&nbsp;in.<br>• Reproducer requires pytorch PR #170056 as noted in the issue. | P2 |  | daisyden | module: inductor, skipped |
| [#2888](https://github.com/intel/torch-xpu-ops/issues/2888) | triton | torch._inductor.exc.InductorError: AssertionError: Conversions between float8_e5m2 and float8_e4m3fn is not supported! | Stonepia | Issue needs owner investigation; only weak/closed cross-ref candidates found | • Test is explicitly named `bad_cast` and expects an exception on eager vs<br>&nbsp;&nbsp;compile path<br>• align the XPU test expectation.<br>• Options: (1) skip test_bad_cast_xpu in the XPU Inductor skip list since<br>&nbsp;&nbsp;triton-xpu does not yet support fp8<->fp8 conversions, or (2) wrap the<br>&nbsp;&nbsp;assertion in _get_min_elements_per_thread so it raises a cleaner<br>&nbsp;&nbsp;UnsupportedOperatorError that the test already catches, or (3) lower<br>&nbsp;&nbsp;e5m2<->e4m3fn via an intermediate bf16/fp32 cast in the XPU lowering.<br>• Short-term: skip<br>• long-term: add the intermediate-cast lowering in<br>&nbsp;&nbsp;torch/_inductor/codegen/triton.py. | P2 | Timeline cross-refs all closed unmerged with weak verdict; no actionable PR | daisyden | module: inductor, ut_upstream |
| [#2997](https://github.com/intel/torch-xpu-ops/issues/2997) | triton | AssertionError of test_linear_and_cel_max_autotune | etaf | Issue needs owner investigation; only weak/closed cross-ref candidates found | • Track upstream fix pytorch/pytorch#180330 (already identified by @etaf) and<br>&nbsp;&nbsp;land it in the PT 2.12 cherry-pick queue<br>• per assignee, 2.12 release does not need it because the failure is UT-only and<br>&nbsp;&nbsp;not observed in E2E.<br>• In the meantime, add test_linear_and_cel_max_autotune to `skip_list_common.py`<br>&nbsp;&nbsp;under `test_inplace_padding.py` (inductor UT).<br>• Verify after merging #180330 that NaN is gone on XPU by rerunning pytest -v<br>&nbsp;&nbsp;`test/inductor/test_inplace_padding.py` -k test_linear_and_cel_max_autotune. | P2 | Timeline cross-refs all closed unmerged with weak verdict; no actionable PR | daisyden | module: inductor, ut_upstream |
| [#3006](https://github.com/intel/torch-xpu-ops/issues/3006) | triton | AssertionError: '.to(tl.float16)' unexpectedly found in '# AOT ID | CuiYifeng | Issue needs owner investigation; only weak/closed cross-ref candidates found | • In the Triton reduction codegen for argmax/argmin (`triton.py` around line<br>&nbsp;&nbsp;4469 `final_argreduce` and the block-ptr store generation), ensure the index<br>&nbsp;&nbsp;result variable's dtype is tracked as int (from `select_index_dtype()`) so the<br>&nbsp;&nbsp;dtype-propagation pass does not insert a cast to the value dtype.<br>• Concretely, when `codegen_upcast_to_fp32=False` and the reduction is<br>&nbsp;&nbsp;argmax/argmin, the post-loop index store should bypass the fp16 src-dtype<br>&nbsp;&nbsp;coercion<br>• add a guard in `TritonKernel.store`/`codegen_indirect_indexing` that skips<br>&nbsp;&nbsp;`.to(tl.float16)` when the logical dtype is integer.<br>• Validate by re-enabling the test on XPU and confirming generated code contains<br>&nbsp;&nbsp;no `.to(tl.float16)` and numerical parity vs eager argmax. | P2 | Timeline cross-refs all closed unmerged with weak verdict; no actionable PR | daisyden | module: inductor, ut_upstream |
| [#1893](https://github.com/intel/torch-xpu-ops/issues/1893) | oneDNN | [upstream_ut] oneDNN accuracy issues in test_ops_xpu.py | chunhuanMeng | Land pytorch/pytorch#179125 (addmv stride preservation) and consider tolerance changes (per @chuanq… | • Bump tolerance for the remaining mv/addmv fp32 ops in XPU OpInfo<br>&nbsp;&nbsp;(torch-xpu-ops `test/xpu/xpu_test_utils.py` or a toleranceOverride for<br>&nbsp;&nbsp;mv/addmv on XPU) rather than modifying kernels, since the gap is within<br>&nbsp;&nbsp;expected fp32 accumulation error for oneDNN gemv.<br>• Remove now-passing cases from the skip list and add a comment linking this<br>&nbsp;&nbsp;issue. | P3 | Vector 0 link; addmv stride fix landing addresses one of the oneDNN accuracy paths. | daisyden | skipped, ut_upstream |
| [#2245](https://github.com/intel/torch-xpu-ops/issues/2245) | oneDNN | oneDNN matmul received incorrect shape in test/test_sparse_csr.py::TestSparseCSRXPU::test_addmm_errors_xpu_float32 | CuiYifeng | Add input-tensor-expansion check on stock PyTorch side (per @CuiYifeng) | • Invoke the standard CSR validation (at::_validate_sparse_csr_tensor_args or<br>&nbsp;&nbsp;the same crow_indices/col_indices checks used on CUDA) at the top of<br>&nbsp;&nbsp;addmm_out_sparse_csr before any layout conversion, so the same error message<br>&nbsp;&nbsp;surfaces on XPU.<br>• Alternatively, update the XPU skip/xfail list to keep the test skipped until<br>&nbsp;&nbsp;the validation parity is implemented.<br>• Long term the XPU path should also fall back cleanly (or raise a clearer<br>&nbsp;&nbsp;error) instead of surfacing a raw oneDNN primitive-descriptor failure. | P3 | No fix PR; CuiYifeng identified upstream check needed. | wincent8 | module: ut, skipped |


<a id="sec-7"></a>
## 7. New submitted issues (<7 days)

_[↑ Back to Index](#sec-2)_

Issues created on or after 2026-04-14, excluding terminal-QA rows.  —  0 issues.

| Issue | Created | Title | Owner | action_TBD | Fix Approach | Priority | action_reason | Reporter | Labels |
|---|---|---|---|---|---|---|---|---|---|


<a id="sec-8"></a>
## 8. Statistics

_[↑ Back to Index](#sec-2)_

- Total rows: **53**
- Classified (non-empty `action_Type`): **48**
- Empty `action_TBD` (no verdict yet): **5**
- Issues flagged for test-case existence check (`CHECK_CASES`): **4**

### 8.1 Primary action_Type distribution (exclusive — one bucket per issue)

_[↑ Back to Index](#sec-2)_

Merged buckets (as rendered in §3 and §4):

| Bucket | Issues |
|---|---:|
| NEED PR | 20 |
| TRACK PR | 12 |
| CLOSE or SKIP | 12 |
| CHECK_CASES | 4 |

Raw atoms (pre-merge, for reference):

| Category | Issues |
|---|---:|
| CLOSE | 1 |
| NOT_TARGET_CLOSE | 3 |
| VERIFY_AND_CLOSE | 7 |
| TRACK_PR | 9 |
| IMPLEMENT | 3 |
| RETRIAGE_PRS | 3 |
| ROOT_CAUSE | 7 |
| NEED_ACTION | 10 |
| CHECK_CASES | 4 |
| SKIP | 1 |

### 8.2 action_Type distribution (multi-label — each category counted once per issue)

_[↑ Back to Index](#sec-2)_

| Category | Issues |
|---|---:|
| CLOSE | 1 |
| NOT_TARGET_CLOSE | 3 |
| VERIFY_AND_CLOSE | 8 |
| TRACK_PR | 9 |
| IMPLEMENT | 3 |
| RETRIAGE_PRS | 4 |
| ROOT_CAUSE | 7 |
| NEED_ACTION | 11 |
| CHECK_CASES | 4 |
| SKIP | 1 |

### 8.3 Priority distribution

_[↑ Back to Index](#sec-2)_

| Priority | Issues |
|---|---:|
| P1 | 7 |
| P2 | 31 |
| P3 | 15 |

### 8.4 Status distribution

_[↑ Back to Index](#sec-2)_

| Status | Issues |
|---|---:|
| open | 53 |

### 8.5 Category column distribution (top 20)

_[↑ Back to Index](#sec-2)_

| Category | Issues |
|---|---:|
| Inductor | 16 |
| Torch Operations | 15 |
| Flash Attention | 12 |
| Sparse | 7 |
| Torch Runtime | 1 |
| TorchAO | 1 |
| Others | 1 |

### 8.6 CHECK_CASES issue IDs

_[↑ Back to Index](#sec-2)_

4 issues flagged for `check_case_avaliablity` (missing XPU test case in repo):

> #2285, #2376, #2512, #2578
