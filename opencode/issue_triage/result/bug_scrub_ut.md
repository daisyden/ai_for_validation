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
| Upstream-pytorch | — | 10 |
| CPU fallback | — | 3 |
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
  - [6.1 upstream-pytorch](#sec-6-1-upstream-pytorch)
  - [6.2 CPU fallback](#sec-6-2-cpu-fallback)
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
| [#2554](https://github.com/intel/torch-xpu-ops/issues/2554) | [upstream_ut] AssertionError: AssertionError not raised | daisyden |  | No change needed in torch-xpu-ops<br>[→ details](details/2554.md) | P2 |  | daisyden | module: inductor, skipped |
| [#2611](https://github.com/intel/torch-xpu-ops/issues/2611) | [upstream_ut] AssertionError: Tensor-likes are not equal! in test_compile_subprocess | daisyden |  | Close as duplicate of #2613 (or consolidate skip list into a single issue).<br>[→ details](details/2611.md) | P2 |  | daisyden | dependency component: driver, module: i… |
| [#2613](https://github.com/intel/torch-xpu-ops/issues/2613) | [upstream_ut] AssertionError: Tensor-likes are not equal! in test_compile_subprocess.py | daisyden |  | Fix the argmax/argmin reduction combine functor in ReduceArgMaxKernel.cpp / ReduceArgMinKernel.cpp…<br>[→ details](details/2613.md) | P2 |  | daisyden | dependency component: driver, module: i… |
| [#2715](https://github.com/intel/torch-xpu-ops/issues/2715) | [upstream_ut] torch._dynamo.exc.Unsupported: Attempted to inline function marked as skipped | CuiYifeng |  | In torch/_dynamo/trace_rules.py, remove torch.xpu from MOD_SKIPLIST (or add torch.xpu.device to the…<br>[→ details](details/2715.md) | P2 |  | daisyden | skipped, ut_upstream |
| [#2800](https://github.com/intel/torch-xpu-ops/issues/2800) | AttributeError: 'torch._C._XpuDeviceProperties' object has no attribute 'major' | guangyey |  | Short-term: add a test-side guard that only queries '.major' on CUDA (e.g. skip/branch when device.…<br>[→ details](details/2800.md) | P2 |  | daisyden | dependency component: oneAPI, module: i… |


<a id="sec-3-1-need-pr"></a>
### 3.1 NEED PR  ·  20 issues

**NEED PR — a PR must be produced or continued (no PR yet, or owner actively debugging root cause, or new code needed)**

<a id="sec-3-1-1-flash-attention"></a>
#### 3.1.1 Flash Attention  ·  3 issues

_[↑ Back to Index](#sec-2)_

| Issue | Title | Owner | action_TBD | Fix Approach | Priority | action_reason | Reporter | Labels |
|---|---|---|---|---|---|---|---|---|
| [#2270](https://github.com/intel/torch-xpu-ops/issues/2270) | Backend Compatibility Error in test/xpu/test_decomp.py | LuFinch | Assignee @LuFinch to investigate | Either (a) skip _flash_attention_forward in the decomp cross-ref OpInfo list for XPU (mirroring CUD…<br>[→ details](details/2270.md) | P2 | Issue already assigned to @LuFinch; owner to lead root-cause. | libohao1201 | module: ut, skipped |
| [#3140](https://github.com/intel/torch-xpu-ops/issues/3140) | [upstream_ut] RuntimeError: FlashAttentionForwardXPU does not only support dropout > 0.0 yet | LuFinch | Issue needs owner investigation; only weak/closed cross-ref candidates found | Implement dropout in the sycltla FlashAttention kernels: add a Philox4x32 RNG state (seed/offset pl…<br>[→ details](details/3140.md) | P2 | Timeline cross-refs all closed unmerged with weak verdict; no actionable PR | daisyden | module: ut, skipped, ut_upstream |
| [#3142](https://github.com/intel/torch-xpu-ops/issues/3142) | [upstream_ut] RuntimeError: The sycl_ext_oneapi_work_group_scratch_memory feature is not yet available for use with SYCL Graph extension. | LuFinch | Issue needs owner investigation; only weak/closed cross-ref candidates found | Pure oneAPI/driver-side fix: wait for oneAPI 2026.0 which lifts the graph restriction on work_group…<br>[→ details](details/3142.md) | P2 | Timeline cross-refs all closed unmerged with weak verdict; no actionable PR | daisyden | dependency component: oneAPI, module: u… |


<a id="sec-3-1-2-inductor"></a>
#### 3.1.2 Inductor  ·  6 issues

_[↑ Back to Index](#sec-2)_

| Issue | Title | Owner | action_TBD | Fix Approach | Priority | action_reason | Reporter | Labels |
|---|---|---|---|---|---|---|---|---|
| [#2329](https://github.com/intel/torch-xpu-ops/issues/2329) | [upstream_ut] feature missing: get_device_tflops and get_drams_gbps | etaf | Assignee @etaf to investigate | Add XPU support in torch/_inductor/utils.py get_device_tflops() and get_dram_gbps(): detect device.…<br>[→ details](details/2329.md) | P2 | Issue already assigned to @etaf; owner to lead root-cause. | daisyden | duplicate, dependency component: Triton… |
| [#2888](https://github.com/intel/torch-xpu-ops/issues/2888) | torch._inductor.exc.InductorError: AssertionError: Conversions between float8_e5m2 and float8_e4m3fn is not supported! | Stonepia | Issue needs owner investigation; only weak/closed cross-ref candidates found | Test is explicitly named 'bad_cast' and expects an exception on eager vs compile path<br>[→ details](details/2888.md) | P2 | Timeline cross-refs all closed unmerged with weak verdict; no actionable PR | daisyden | module: inductor, ut_upstream |
| [#2891](https://github.com/intel/torch-xpu-ops/issues/2891) | RuntimeError: Expected to find "(262144, 0, 512, 1" but did not find it | chunhuanMeng | Issue needs owner investigation; only weak/closed cross-ref candidates found | Either (a) skip/xfail this test for XPU in third_party/torch-xpu-ops/test/inductor skip lists becau…<br>[→ details](details/2891.md) | P2 | Timeline cross-refs all closed unmerged with weak verdict; no actionable PR | daisyden | module: inductor, ut_upstream |
| [#2997](https://github.com/intel/torch-xpu-ops/issues/2997) | AssertionError of test_linear_and_cel_max_autotune | etaf | Issue needs owner investigation; only weak/closed cross-ref candidates found | Track upstream fix pytorch/pytorch#180330 (already identified by @etaf) and land it in the PT 2.12…<br>[→ details](details/2997.md) | P2 | Timeline cross-refs all closed unmerged with weak verdict; no actionable PR | daisyden | module: inductor, ut_upstream |
| [#3006](https://github.com/intel/torch-xpu-ops/issues/3006) | AssertionError: '.to(tl.float16)' unexpectedly found in '# AOT ID | CuiYifeng | Issue needs owner investigation; only weak/closed cross-ref candidates found | In the Triton reduction codegen for argmax/argmin (triton.py around line 4469 `final_argreduce` and…<br>[→ details](details/3006.md) | P2 | Timeline cross-refs all closed unmerged with weak verdict; no actionable PR | daisyden | module: inductor, ut_upstream |
| [#3187](https://github.com/intel/torch-xpu-ops/issues/3187) | PyTorch XPU gpu_cpp_wrapper fails with InductorError NotImplementedError | CuiYifeng | needs owner investigation | In the XPU cpp_wrapper / AOTI codegen (torch/_inductor/codegen/cpp_wrapper_gpu.py and XPU-specific…<br>[→ details](details/3187.md) | P2 | No comments, no linked or referenced PR found | liangan1 | ut_upstream |


<a id="sec-3-1-3-others"></a>
#### 3.1.3 Others  ·  1 issues

_[↑ Back to Index](#sec-2)_

| Issue | Title | Owner | action_TBD | Fix Approach | Priority | action_reason | Reporter | Labels |
|---|---|---|---|---|---|---|---|---|
| [#2024](https://github.com/intel/torch-xpu-ops/issues/2024) | AssertionError: Torch not compiled with CUDA enabled | daisyden | Assignee @daisyden to investigate | Either (a) add these test IDs to test/xpu/skip_list_common.py with a clear TODO so CI stays green,…<br>[→ details](details/2024.md) | P2 | Issue already assigned to @daisyden; owner to lead root-cause. | mengfei25 | module: ut, skipped |


<a id="sec-3-1-4-sparse"></a>
#### 3.1.4 Sparse  ·  3 issues

_[↑ Back to Index](#sec-2)_

| Issue | Title | Owner | action_TBD | Fix Approach | Priority | action_reason | Reporter | Labels |
|---|---|---|---|---|---|---|---|---|
| [#3165](https://github.com/intel/torch-xpu-ops/issues/3165) | test_sparse_csr_xpu.py::TestSparseCompressedTritonKernelsXPU::test_triton_bsr_softmax meet RuntimeError: ZE_RESULT_ERROR_INVALID_KERNEL_NAME | jafraustro | assignee investigate (split-out from #2209) | File/track a pytorch-triton-xpu issue with the failing kernel reproducer and, once fixed upstream,…<br>[→ details](details/3165.md) | P1 | Skipped Triton sparse softmax case; sub-issue split from #2209; assigned to @jafraustro. | CuiYifeng | skipped, ut_upstream |
| [#2214](https://github.com/intel/torch-xpu-ops/issues/2214) | test/test_sparse.py::TestSparseAnyXPU::test_gradcheck_mm expected error message not match | jenniew | Assignee @jenniew to investigate | Extend the expected-error regex in test/test_sparse.py (around line 5297) to also accept the XPU-sp…<br>[→ details](details/2214.md) | P2 | Issue already assigned to @jenniew; owner to lead root-cause. | wincent8 | skipped, ut_upstream |
| [#3177](https://github.com/intel/torch-xpu-ops/issues/3177) | Accuracy gap of BF16/FP16 test_block_addmm | jenniew | needs owner investigation | Land torch-xpu-ops PR #3273: replace result_dense.add_(input_dense, beta) at SparseCsrTensorMath.cp…<br>[→ details](details/3177.md) | P2 | No verified PR; no actionable owner requests | CuiYifeng | skipped |


<a id="sec-3-1-5-torch-operations"></a>
#### 3.1.5 Torch Operations  ·  7 issues

_[↑ Back to Index](#sec-2)_

| Issue | Title | Owner | action_TBD | Fix Approach | Priority | action_reason | Reporter | Labels |
|---|---|---|---|---|---|---|---|---|
| [#2244](https://github.com/intel/torch-xpu-ops/issues/2244) | test/test_sparse_csr.py::TestSparseCSRXPU::test_block_addmm meet RuntimeError: empty_sparse_compressed expected sparse compressed (non-block) tensor layout but got SparseBsr | jenniew | needs new PR | Add the missing BSR-result branch in addmm_out_sparse_csr: compute the dense product via addmm_calc…<br>[→ details](details/2244.md) | P2 | Linked PR(s) closed without merge: intel/torch-xpu-ops#2974; fix attempt abandoned | wincent8 | module: ut, skipped |
| [#3136](https://github.com/intel/torch-xpu-ops/issues/3136) | [upstream_ut] AssertionError: False is not true in test_transformers | LuFinch | Issue needs owner investigation; only weak/closed cross-ref candidates found | For test_disable_fastpath_xpu, land/forward-port pytorch/pytorch#179701 and re-enable.<br>[→ details](details/3136.md) | P2 | Timeline cross-refs all closed unmerged with weak verdict; no actionable PR | daisyden | module: ut, skipped, ut_upstream |
| [#3137](https://github.com/intel/torch-xpu-ops/issues/3137) | [upstream_ut] RuntimeError: expected scalar type Half but found Float | LuFinch | Issue needs owner investigation; only weak/closed cross-ref candidates found | Merge/forward-port pytorch/pytorch#179701 so `_transformer_encoder_layer_fwd` takes the slow path (…<br>[→ details](details/3137.md) | P2 | Timeline cross-refs all closed unmerged with weak verdict; no actionable PR | daisyden | module: ut, skipped, ut_upstream, random |
| [#2245](https://github.com/intel/torch-xpu-ops/issues/2245) | oneDNN matmul received incorrect shape in test/test_sparse_csr.py::TestSparseCSRXPU::test_addmm_errors_xpu_float32 | CuiYifeng | Add input-tensor-expansion check on stock PyTorch side (per @CuiYifeng) | Invoke the standard CSR validation (at::_validate_sparse_csr_tensor_args or the same crow_indices/c…<br>[→ details](details/2245.md) | P3 | No fix PR; CuiYifeng identified upstream check needed. | wincent8 | module: ut, skipped |
| [#2436](https://github.com/intel/torch-xpu-ops/issues/2436) | [upstream_ut] AttributeError: 'NoneType' object has no attribute 'clone' | daisyden | Owner @daisyden to re-check case design (per @CuiYifeng request) | Wait for the upstream fix in pytorch/pytorch#97395 (guard None .grad in _expanded_weights clone pat…<br>[→ details](details/2436.md) | P3 | No fix PR; behavior matches CUDA per upstream pytorch/pytorch#97395 (CPU issue). | daisyden | skipped, dependency component: communit… |
| [#2531](https://github.com/intel/torch-xpu-ops/issues/2531) | [upstream_ut] AssertionError: Torch not compiled with CUDA enabled | daisyden | Owner @daisyden to triage; only github-actions auto-pass observed | Triage per test: (a) genuinely CUDA-specific (cufft_plan_cache, ctc_loss_cudnn_*, numeric_check_lea…<br>[→ details](details/2531.md) | P3 | Vector 0 empty; timeline only documentation/tracker cross-refs. | daisyden | skipped, port_from_skiplist |
| [#3041](https://github.com/intel/torch-xpu-ops/issues/3041) | AssertionError: Expected len(flat_diff_results) > 0 in test_fake_crossref_backward_amp_normal_number_mean_xpu_float32 | Silv3S, BartoszKokoszko | Owner @Silv3S, BartoszKokoszko to file fix PR | In torch/testing/_internal/common_methods_invocations.py, extend the existing CUDA DecorateInfo on…<br>[→ details](details/3041.md) | P3 | Issue assigned to @Silv3S, BartoszKokoszko; owner to implement fix. | daisyden | ut_upstream |


<a id="sec-3-2-track-pr"></a>
### 3.2 TRACK PR  ·  12 issues

**TRACK PR — a PR is identified; track it to merge, or re-evaluate if prior PRs are dead / unverified**

<a id="sec-3-2-1-flash-attention"></a>
#### 3.2.1 Flash Attention  ·  3 issues

_[↑ Back to Index](#sec-2)_

| Issue | Title | Owner | action_TBD | Fix Approach | Priority | action_reason | Reporter | Labels |
|---|---|---|---|---|---|---|---|---|
| [#2442](https://github.com/intel/torch-xpu-ops/issues/2442) | [Bug Skip]: NotImplementedError: Could not run 'aten::_flash_attention_forward' with arguments from the 'CPU' backend | LuFinch | track PR to merge | Implement a native XPU _flash_attention_forward SYCL kernel (e.g. under src/ATen/native/xpu/FlashAt…<br>[→ details](details/2442.md) | P1 | Linked PR(s) open: [3404] | CuiYifeng | skipped |
| [#2853](https://github.com/intel/torch-xpu-ops/issues/2853) | [upstream_ut] torch.ops.aten._flash_attention_forward lack of support for XPU. | LuFinch | wait for PR #3404 review/merge | Add an XPU registration for aten::_flash_attention_forward in yaml/native/native_functions.yaml (XP…<br>[→ details](details/2853.md) | P1 | Fix PR #3404 OPEN; enables _flash_attention_forward/backward ops; explicit Fix link to #2853. | BBBela | skipped |
| [#2698](https://github.com/intel/torch-xpu-ops/issues/2698) | Title: [upstream_ut] RuntimeError: FlashAttentionForwardXPU only support headdim 64,96,128,192 | chunhuanMeng, LuFinch | Track pytorch/pytorch#180646 to merge | (1) In _scaled_dot_product_flash_attention_xpu dispatcher, check headdim eligibility before calling…<br>[→ details](details/2698.md) | P2 | Verified fix PR is OPEN (VERIFIED via github_linked; PR is OPEN). | daisyden | module: inductor, skipped, ut_upstream |


<a id="sec-3-2-2-inductor"></a>
#### 3.2.2 Inductor  ·  3 issues

_[↑ Back to Index](#sec-2)_

| Issue | Title | Owner | action_TBD | Fix Approach | Priority | action_reason | Reporter | Labels |
|---|---|---|---|---|---|---|---|---|
| [#2609](https://github.com/intel/torch-xpu-ops/issues/2609) | [upstream_ut] torch._inductor.exc.InductorError: CppCompileError: C++ compile error | daisyden | Track pytorch/pytorch#171154 to merge | In pytorch upstream torch/_inductor/codegen (see cpp_wrapper_gpu.py / aoti shim generation for cust…<br>[→ details](details/2609.md) | P2 | Verified fix PR is OPEN (VERIFIED via content_match; PR is OPEN). | daisyden | module: inductor, skipped, ut_upstream |
| [#2806](https://github.com/intel/torch-xpu-ops/issues/2806) | CompiledAOTI need XPU support | daisyden | Re-evaluate: verified PR pytorch/pytorch#178385 is CLOSED unmerged | Ensure torch._inductor.output_code.CompiledAOTI.__post_init__ handles `device_type.startswith('xpu'…<br>[→ details](details/2806.md) | P2 | Linked fix PR was closed without merging; need replacement fix. | daisyden | module: inductor, ut_upstream |
| [#2958](https://github.com/intel/torch-xpu-ops/issues/2958) | AssertionError of test_dtensor_basic_compile | daisyden | Triage: re-validate cross-referenced PRs (none auto-verified) | Verify on latest pytorch main that the test passes (etaf report), then remove the skip entry from t…<br>[→ details](details/2958.md) | P3 | Cross-references exist but no PR explicitly references this issue; manual review needed. | daisyden | module: inductor, ut_upstream |


<a id="sec-3-2-3-sparse"></a>
#### 3.2.3 Sparse  ·  2 issues

_[↑ Back to Index](#sec-2)_

| Issue | Title | Owner | action_TBD | Fix Approach | Priority | action_reason | Reporter | Labels |
|---|---|---|---|---|---|---|---|---|
| [#2283](https://github.com/intel/torch-xpu-ops/issues/2283) | [upstream_ut] sparse._sampled_addmm is not supported | jenniew | wait for PR | Implement sparse_sampled_addmm for SparseCsrXPU: add a sampled_addmm_out_sparse_csr_xpu kernel in s…<br>[→ details](details/2283.md) | P2 | Verified PR(s) open: intel/torch-xpu-ops#3018 | daisyden | skipped, ut_upstream |
| [#2229](https://github.com/intel/torch-xpu-ops/issues/2229) | test/test_sparse_csr.py::TestSparseCompressedCPU::test_invalid_input meet message not match | jenniew | Move PR #3073 out of WIP and land | Update the XPU/sparse test expectations in torch-xpu-ops skip/override list to match the actual emi…<br>[→ details](details/2229.md) | P3 | Open WIP PR by assignee jenniew with explicit Related issue: #2229; root cause identified as missing _validate_compressed_sparse_indices on… | wincent8 | skipped |


<a id="sec-3-2-4-torch-operations"></a>
#### 3.2.4 Torch Operations  ·  4 issues

_[↑ Back to Index](#sec-2)_

| Issue | Title | Owner | action_TBD | Fix Approach | Priority | action_reason | Reporter | Labels |
|---|---|---|---|---|---|---|---|---|
| [#2615](https://github.com/intel/torch-xpu-ops/issues/2615) | [Bug Skip]: New failures RuntimeError: Unsupported dtype Half / RuntimeError: Unsupported dtype torch.float16 | CuiYifeng | Track intel/torch-xpu-ops#2996 to merge \| Re-evaluate: verified PR pytorch/pytorch#171231 is CLOSE… | Extend the Half/ComplexHalf promotion wrapper in mkl/SpectralOps.cpp to cover all public fft entry…<br>[→ details](details/2615.md) | P2 | Verified fix PR is OPEN (VERIFIED via explicit_reference; PR is OPEN). \| Linked fix PR was closed without merging; need replacement fix. | kaileiyx | module: ut, skipped |
| [#2783](https://github.com/intel/torch-xpu-ops/issues/2783) | [Bug Skip]: Key "xpu" is missing from dict "driver" in test_svd | daisyden | Re-evaluate: verified PR pytorch/pytorch#172824 is CLOSED unmerged | Upstream PR to test/test_linalg.py: add an 'xpu': (None,) entry to the drivers dict (and similarly…<br>[→ details](details/2783.md) | P2 | Linked fix PR was closed without merging; need replacement fix. | CuiYifeng | module: ut, skipped, dependency compone… |
| [#1893](https://github.com/intel/torch-xpu-ops/issues/1893) | [upstream_ut] oneDNN accuracy issues in test_ops_xpu.py | chunhuanMeng | Land pytorch/pytorch#179125 (addmv stride preservation) and consider tolerance changes (per @chuanq… | Bump tolerance for the remaining mv/addmv fp32 ops in XPU OpInfo (torch-xpu-ops test/xpu/xpu_test_u…<br>[→ details](details/1893.md) | P3 | Vector 0 link; addmv stride fix landing addresses one of the oneDNN accuracy paths. | daisyden | skipped, ut_upstream |
| [#3033](https://github.com/intel/torch-xpu-ops/issues/3033) | [Bug Skip]: Softmax tolerance | chunhuanMeng | Track PR intel/torch-xpu-ops#3036 to merge | Short-term: the case is already listed in the XPU skip_list_common for extended op tests (PR merged…<br>[→ details](details/3033.md) | P3 | Phase 4b backfill: verified open PR from pr_analysis. | chunhuanMeng | skipped, random |


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
| [#2802](https://github.com/intel/torch-xpu-ops/issues/2802) | Three aten._scaled_dot_product_flash_attention issues | daisyden | label not_target (partial: 1 of 3 sub-cases) \| verify PR #3289 fix and close remaining cases | (1) Relax the attn_mask+is_causal check for efficient_attention on XPU (Attention.cpp) or fix the t…<br>[→ details](details/2802.md) | P1 | 1 sub-case won't-fix; PR #3289 MERGED for fake-tensor case; 3rd case is XPU backend priority explanation. | daisyden | module: inductor, ut_upstream |
| [#3143](https://github.com/intel/torch-xpu-ops/issues/3143) | NotImplementedError: The operator 'aten::_scaled_dot_product_efficient_attention_backward' is not currently implemented for the XPU device. | daisyden | assignee verify fix and close issue | Already addressed by intel/torch-xpu-ops PR #3317 which adds the XPU registration for aten::_scaled…<br>[→ details](details/3143.md) | P1 | Fix PR #3317 MERGED 2026-04-17; daisyden to validate and close. | daisyden | module: ut, skipped, ut_upstream |
| [#2720](https://github.com/intel/torch-xpu-ops/issues/2720) | [upstream_ut] RuntimeError: false INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/native/DispatchStub.cpp":275 | CuiYifeng | Close the fixed issue | Align third_party/torch-xpu-ops/src/ATen/native/xpu/sycl/BinaryMiscOpsKernels.cpp:158 ldexp_kernel…<br>[→ details](details/2720.md) | P2 | Fixed and passed in CI | wincent8 | skipped |
| [#3132](https://github.com/intel/torch-xpu-ops/issues/3132) | [upstream_ut] transfomers test reports RuntimeError: No available kernel. Aborting execution. | LuFinch | Label issue not_target (partial); owner: @LuFinch \| Issue needs owner investigation; only weak/clo… | For the singleton-stride case, land/forward-port pytorch/pytorch#179800 which flips `ignore_singlet…<br>[→ details](details/3132.md) | P2 | Partial not-target decision found in authoritative comment(s) \| Timeline cross-refs all closed unmerged with weak verdict; no actionable PR | daisyden | module: ut, skipped, ut_upstream |
| [#3141](https://github.com/intel/torch-xpu-ops/issues/3141) | [upstream_ut] RuntimeError: FlashAttentionForwardXPU only support headdim 64,96,128,192 | LuFinch | Verify fix from merged PR intel/torch-xpu-ops#3355; close issue if validated | Two-layer fix: (1) at the dispatcher in aten/native/transformers/xpu choose the math / mem-efficien…<br>[→ details](details/3141.md) | P2 | PR intel/torch-xpu-ops#3355 merged (github_linked) | daisyden | module: ut, skipped, ut_upstream |
| [#3163](https://github.com/intel/torch-xpu-ops/issues/3163) | [Bug Skip]: Object comparison failed: torch.int64 != torch.int32 in test_sparse_add | chunhuanMeng | Verify fix from merged PR intel/torch-xpu-ops#3341; close issue if validated | In add_out_sparse_compressed_xpu, after constructing out_dense.to_sparse_csr(), cast crow_indices a…<br>[→ details](details/3163.md) | P2 | PR intel/torch-xpu-ops#3341 merged (explicit_reference) | CuiYifeng | skipped, ut_upstream |
| [#2169](https://github.com/intel/torch-xpu-ops/issues/2169) | Frame size comparison failed in test_size_comparison_no_recompile | guangyey | Verify on latest main and close (per @guangyey already fixed) | Re-run test_size_comparison_no_recompile on current main with a fresh XPU wheel to confirm the fix,…<br>[→ details](details/2169.md) | P3 | @guangyey reports it is fixed on latest main; Vector 0 PR pytorch/pytorch#178780 covers the device fix. | daisyden | skipped |
| [#2253](https://github.com/intel/torch-xpu-ops/issues/2253) | the supported dtypes are not align with cuda | daisyden | Verify alignment landed and close (duplicate of #2289) | Update dtypesIfXPU in upstream opinfo definitions for each failing op to match the dtype set actual…<br>[→ details](details/2253.md) | P3 | @daisyden noted issue is duplicate of #2289 and dtypes aligned with CUDA via pytorch/pytorch#161246 commit; nightly tests passing. | daisyden | duplicate, skipped, ut_upstream |
| [#2697](https://github.com/intel/torch-xpu-ops/issues/2697) | Title: [upstream_ut] RuntimeError: Expected to find ", 0, " but did not find it | chunhuanMeng | label not_target and close | Short term: skip this test for XPU (add @requires_cuda or xfail_if_xpu) since the codegen signature…<br>[→ details](details/2697.md) | P3 | Issue assignee (MEMBER) chunhuanMeng made firm statement: 'We cannot fix this because our efficient attention falls back to math, which has… | daisyden | module: inductor, skipped, ut_upstream |
| [#2999](https://github.com/intel/torch-xpu-ops/issues/2999) | KeyError: 'eager_numerics.use_pytorch_libdevice' | daisyden | Skip issue | Confirmed not_target label: the assignee has decided to permanently skip these tests on XPU.<br>[→ details](details/2999.md) | P3 | not target feature | daisyden | module: inductor, ut_upstream, not_targ… |
| [#3131](https://github.com/intel/torch-xpu-ops/issues/3131) | [upstream_ut] NotImplementedError: The operator 'aten::_scaled_dot_product_efficient_attention_backward' is not c | daisyden | Verify fix from merged PR intel/torch-xpu-ops#3367 and close | Already fixed via intel/torch-xpu-ops#3367 which added proper bias-grad validation (raising the exp…<br>[→ details](details/3131.md) | P3 | Phase 4b backfill: verified merged PR from pr_analysis. | daisyden | module: ut, skipped, ut_upstream |
| [#3170](https://github.com/intel/torch-xpu-ops/issues/3170) | Unskip test_bmm_windows_error_xpu_float64 | libohao1201, jafraustro | Verify fix from merged PR intel/torch-xpu-ops#3225 and close | In third_party/torch-xpu-ops/test/xpu/test_sparse_xpu.py:1970 remove the `@unittest.skipIf(TEST_XPU…<br>[→ details](details/3170.md) | P3 | Phase 4b backfill: verified merged PR from pr_analysis. | CuiYifeng | skipped, ut_upstream |


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
| [#2285](https://github.com/intel/torch-xpu-ops/issues/2285) | Support efficient attention | chunhuanMeng | check_case_avaliablity | Implement _efficient_attention_forward/backward for XPU via a new SYCL kernel (e.g. third_party/tor…<br>[→ details](details/2285.md) | P1 |  | daisyden | skipped |
| [#2578](https://github.com/intel/torch-xpu-ops/issues/2578) | [TorchAO][UT] test/quantization/test_quant_api.py AssertionError: SQNR -2.90625 is too low | Stonepia | check_case_avaliablity | Either (a) implement TensorCoreTiledLayout support in _convert_weight_to_int4pack_xpu (WeightInt4Pa…<br>[→ details](details/2578.md) | P1 |  | zxd1997066 | module: ao, ut_upstream |
| [#2376](https://github.com/intel/torch-xpu-ops/issues/2376) | [Bug Skip]: NotImplementedError: "logaddexp_xpu" not implemented for 'Complex' | daisyden, CuiYifeng | check_case_avaliablity | Remove the XPU-specific skips/WAs added in pytorch/pytorch#171238 (test/test_binary_ufuncs.py, test…<br>[→ details](details/2376.md) | P2 |  | mengfei25 | module: ut, skipped |
| [#2512](https://github.com/intel/torch-xpu-ops/issues/2512) | [upstream_ut] RuntimeError: _histc_xpu does not have a deterministic implementation, but you set 'torch.use_deter | chunhuanMeng | check_case_avaliablity | Edit third_party/torch-xpu-ops/src/ATen/native/xpu/SummaryOps.cpp:_histc_xpu to call `globalContext…<br>[→ details](details/2512.md) | P3 |  | libohao1201 | skipped |


<a id="sec-5"></a>
## 5. Duplicated issues

_[↑ Back to Index](#sec-2)_

Rows where `duplicated_issue` is set or `action_TBD` contains "duplicate of".  —  5 issues.

| Issue | Duplicates | Title | Owner | action_TBD | Fix Approach | Priority | action_reason | Reporter | Labels |
|---|---|---|---|---|---|---|---|---|---|
| [#2285](https://github.com/intel/torch-xpu-ops/issues/2285) | [#2853](https://github.com/intel/torch-xpu-ops/issues/2853) | Support efficient attention | chunhuanMeng | check_case_avaliablity | Implement _efficient_attention_forward/backward for XPU via a new SYCL kernel (e.g. third_party/tor…<br>[→ details](details/2285.md) | P1 |  | daisyden | skipped |
| [#2853](https://github.com/intel/torch-xpu-ops/issues/2853) | [#2285](https://github.com/intel/torch-xpu-ops/issues/2285) | [upstream_ut] torch.ops.aten._flash_attention_forward lack of support for XPU. | LuFinch | wait for PR #3404 review/merge | Add an XPU registration for aten::_flash_attention_forward in yaml/native/native_functions.yaml (XP…<br>[→ details](details/2853.md) | P1 | Fix PR #3404 OPEN; enables _flash_attention_forward/backward ops; explicit Fix link to #2853. | BBBela | skipped |
| [#2024](https://github.com/intel/torch-xpu-ops/issues/2024) | [#2444](https://github.com/intel/torch-xpu-ops/issues/2444) | AssertionError: Torch not compiled with CUDA enabled | daisyden | Assignee @daisyden to investigate | Either (a) add these test IDs to test/xpu/skip_list_common.py with a clear TODO so CI stays green,…<br>[→ details](details/2024.md) | P2 | Issue already assigned to @daisyden; owner to lead root-cause. | mengfei25 | module: ut, skipped |
| [#2715](https://github.com/intel/torch-xpu-ops/issues/2715) | [#3286](https://github.com/intel/torch-xpu-ops/issues/3286) | [upstream_ut] torch._dynamo.exc.Unsupported: Attempted to inline function marked as skipped | CuiYifeng |  | In torch/_dynamo/trace_rules.py, remove torch.xpu from MOD_SKIPLIST (or add torch.xpu.device to the…<br>[→ details](details/2715.md) | P2 |  | daisyden | skipped, ut_upstream |
| [#2253](https://github.com/intel/torch-xpu-ops/issues/2253) | [#2289](https://github.com/intel/torch-xpu-ops/issues/2289) | the supported dtypes are not align with cuda | daisyden | Verify alignment landed and close (duplicate of #2289) | Update dtypesIfXPU in upstream opinfo definitions for each failing op to match the dtype set actual…<br>[→ details](details/2253.md) | P3 | @daisyden noted issue is duplicate of #2289 and dtypes aligned with CUDA via pytorch/pytorch#161246 commit; nightly tests passing. | daisyden | duplicate, skipped, ut_upstream |


<a id="sec-6"></a>
## 6. Dependency (external blockers)

_[↑ Back to Index](#sec-2)_

Issues with a non-blank `Dependency` value, excluding `upstream-pytorch`, `CPU fallback`, and `SYCL kernel:*` (in-repo kernel pointers). Terminal-QA rows (CLOSE / VERIFY_AND_CLOSE / SKIP / NOT_TARGET_CLOSE) are also excluded.  —  13 issues.

| Issue | Dependency | Title | Owner | action_TBD | Fix Approach | Priority | action_reason | Reporter | Labels |
|---|---|---|---|---|---|---|---|---|---|
| [#3165](https://github.com/intel/torch-xpu-ops/issues/3165) | triton | test_sparse_csr_xpu.py::TestSparseCompressedTritonKernelsXPU::test_triton_bsr_softmax meet RuntimeError: ZE_RESULT_ERROR_INVALID_KERNEL_NAME | jafraustro | assignee investigate (split-out from #2209) | File/track a pytorch-triton-xpu issue with the failing kernel reproducer and, once fixed upstream,…<br>[→ details](details/3165.md) | P1 | Skipped Triton sparse softmax case; sub-issue split from #2209; assigned to @jafraustro. | CuiYifeng | skipped, ut_upstream |
| [#2800](https://github.com/intel/torch-xpu-ops/issues/2800) | driver | AttributeError: 'torch._C._XpuDeviceProperties' object has no attribute 'major' | guangyey |  | Short-term: add a test-side guard that only queries '.major' on CUDA (e.g. skip/branch when device.…<br>[→ details](details/2800.md) | P2 |  | daisyden | dependency component: oneAPI, module: i… |
| [#3142](https://github.com/intel/torch-xpu-ops/issues/3142) | driver | [upstream_ut] RuntimeError: The sycl_ext_oneapi_work_group_scratch_memory feature is not yet available for use with SYCL Graph extension. | LuFinch | Issue needs owner investigation; only weak/closed cross-ref candidates found | Pure oneAPI/driver-side fix: wait for oneAPI 2026.0 which lifts the graph restriction on work_group…<br>[→ details](details/3142.md) | P2 | Timeline cross-refs all closed unmerged with weak verdict; no actionable PR | daisyden | dependency component: oneAPI, module: u… |
| [#3136](https://github.com/intel/torch-xpu-ops/issues/3136) | oneDNN | [upstream_ut] AssertionError: False is not true in test_transformers | LuFinch | Issue needs owner investigation; only weak/closed cross-ref candidates found | For test_disable_fastpath_xpu, land/forward-port pytorch/pytorch#179701 and re-enable.<br>[→ details](details/3136.md) | P2 | Timeline cross-refs all closed unmerged with weak verdict; no actionable PR | daisyden | module: ut, skipped, ut_upstream |
| [#3137](https://github.com/intel/torch-xpu-ops/issues/3137) | oneDNN | [upstream_ut] RuntimeError: expected scalar type Half but found Float | LuFinch | Issue needs owner investigation; only weak/closed cross-ref candidates found | Merge/forward-port pytorch/pytorch#179701 so `_transformer_encoder_layer_fwd` takes the slow path (…<br>[→ details](details/3137.md) | P2 | Timeline cross-refs all closed unmerged with weak verdict; no actionable PR | daisyden | module: ut, skipped, ut_upstream, random |
| [#2615](https://github.com/intel/torch-xpu-ops/issues/2615) | oneMKL | [Bug Skip]: New failures RuntimeError: Unsupported dtype Half / RuntimeError: Unsupported dtype torch.float16 | CuiYifeng | Track intel/torch-xpu-ops#2996 to merge \| Re-evaluate: verified PR pytorch/pytorch#171231 is CLOSE… | Extend the Half/ComplexHalf promotion wrapper in mkl/SpectralOps.cpp to cover all public fft entry…<br>[→ details](details/2615.md) | P2 | Verified fix PR is OPEN (VERIFIED via explicit_reference; PR is OPEN). \| Linked fix PR was closed without merging; need replacement fix. | kaileiyx | module: ut, skipped |
| [#2329](https://github.com/intel/torch-xpu-ops/issues/2329) | triton | [upstream_ut] feature missing: get_device_tflops and get_drams_gbps | etaf | Assignee @etaf to investigate | Add XPU support in torch/_inductor/utils.py get_device_tflops() and get_dram_gbps(): detect device.…<br>[→ details](details/2329.md) | P2 | Issue already assigned to @etaf; owner to lead root-cause. | daisyden | duplicate, dependency component: Triton… |
| [#2554](https://github.com/intel/torch-xpu-ops/issues/2554) | triton | [upstream_ut] AssertionError: AssertionError not raised | daisyden |  | No change needed in torch-xpu-ops<br>[→ details](details/2554.md) | P2 |  | daisyden | module: inductor, skipped |
| [#2888](https://github.com/intel/torch-xpu-ops/issues/2888) | triton | torch._inductor.exc.InductorError: AssertionError: Conversions between float8_e5m2 and float8_e4m3fn is not supported! | Stonepia | Issue needs owner investigation; only weak/closed cross-ref candidates found | Test is explicitly named 'bad_cast' and expects an exception on eager vs compile path<br>[→ details](details/2888.md) | P2 | Timeline cross-refs all closed unmerged with weak verdict; no actionable PR | daisyden | module: inductor, ut_upstream |
| [#2997](https://github.com/intel/torch-xpu-ops/issues/2997) | triton | AssertionError of test_linear_and_cel_max_autotune | etaf | Issue needs owner investigation; only weak/closed cross-ref candidates found | Track upstream fix pytorch/pytorch#180330 (already identified by @etaf) and land it in the PT 2.12…<br>[→ details](details/2997.md) | P2 | Timeline cross-refs all closed unmerged with weak verdict; no actionable PR | daisyden | module: inductor, ut_upstream |
| [#3006](https://github.com/intel/torch-xpu-ops/issues/3006) | triton | AssertionError: '.to(tl.float16)' unexpectedly found in '# AOT ID | CuiYifeng | Issue needs owner investigation; only weak/closed cross-ref candidates found | In the Triton reduction codegen for argmax/argmin (triton.py around line 4469 `final_argreduce` and…<br>[→ details](details/3006.md) | P2 | Timeline cross-refs all closed unmerged with weak verdict; no actionable PR | daisyden | module: inductor, ut_upstream |
| [#1893](https://github.com/intel/torch-xpu-ops/issues/1893) | oneDNN | [upstream_ut] oneDNN accuracy issues in test_ops_xpu.py | chunhuanMeng | Land pytorch/pytorch#179125 (addmv stride preservation) and consider tolerance changes (per @chuanq… | Bump tolerance for the remaining mv/addmv fp32 ops in XPU OpInfo (torch-xpu-ops test/xpu/xpu_test_u…<br>[→ details](details/1893.md) | P3 | Vector 0 link; addmv stride fix landing addresses one of the oneDNN accuracy paths. | daisyden | skipped, ut_upstream |
| [#2245](https://github.com/intel/torch-xpu-ops/issues/2245) | oneDNN | oneDNN matmul received incorrect shape in test/test_sparse_csr.py::TestSparseCSRXPU::test_addmm_errors_xpu_float32 | CuiYifeng | Add input-tensor-expansion check on stock PyTorch side (per @CuiYifeng) | Invoke the standard CSR validation (at::_validate_sparse_csr_tensor_args or the same crow_indices/c…<br>[→ details](details/2245.md) | P3 | No fix PR; CuiYifeng identified upstream check needed. | wincent8 | module: ut, skipped |


<a id="sec-6-1-upstream-pytorch"></a>
### 6.1 upstream-pytorch

_[↑ Back to Index](#sec-2)_

Issues whose fix lives in `pytorch/pytorch` (Dynamo/Inductor, AOTAutograd, `_prims_common`, benchmark harness, test-list sync, etc.). Terminal-QA rows excluded.  —  10 issues.

| Issue | Dependency | Title | Owner | action_TBD | Fix Approach | Priority | action_reason | Reporter | Labels |
|---|---|---|---|---|---|---|---|---|---|
| [#2609](https://github.com/intel/torch-xpu-ops/issues/2609) | upstream-pytorch | [upstream_ut] torch._inductor.exc.InductorError: CppCompileError: C++ compile error | daisyden | Track pytorch/pytorch#171154 to merge | In pytorch upstream torch/_inductor/codegen (see cpp_wrapper_gpu.py / aoti shim generation for cust…<br>[→ details](details/2609.md) | P2 | Verified fix PR is OPEN (VERIFIED via content_match; PR is OPEN). | daisyden | module: inductor, skipped, ut_upstream |
| [#2715](https://github.com/intel/torch-xpu-ops/issues/2715) | upstream-pytorch | [upstream_ut] torch._dynamo.exc.Unsupported: Attempted to inline function marked as skipped | CuiYifeng |  | In torch/_dynamo/trace_rules.py, remove torch.xpu from MOD_SKIPLIST (or add torch.xpu.device to the…<br>[→ details](details/2715.md) | P2 |  | daisyden | skipped, ut_upstream |
| [#2783](https://github.com/intel/torch-xpu-ops/issues/2783) | upstream-pytorch | [Bug Skip]: Key "xpu" is missing from dict "driver" in test_svd | daisyden | Re-evaluate: verified PR pytorch/pytorch#172824 is CLOSED unmerged | Upstream PR to test/test_linalg.py: add an 'xpu': (None,) entry to the drivers dict (and similarly…<br>[→ details](details/2783.md) | P2 | Linked fix PR was closed without merging; need replacement fix. | CuiYifeng | module: ut, skipped, dependency compone… |
| [#2806](https://github.com/intel/torch-xpu-ops/issues/2806) | upstream-pytorch | CompiledAOTI need XPU support | daisyden | Re-evaluate: verified PR pytorch/pytorch#178385 is CLOSED unmerged | Ensure torch._inductor.output_code.CompiledAOTI.__post_init__ handles `device_type.startswith('xpu'…<br>[→ details](details/2806.md) | P2 | Linked fix PR was closed without merging; need replacement fix. | daisyden | module: inductor, ut_upstream |
| [#2891](https://github.com/intel/torch-xpu-ops/issues/2891) | upstream-pytorch | RuntimeError: Expected to find "(262144, 0, 512, 1" but did not find it | chunhuanMeng | Issue needs owner investigation; only weak/closed cross-ref candidates found | Either (a) skip/xfail this test for XPU in third_party/torch-xpu-ops/test/inductor skip lists becau…<br>[→ details](details/2891.md) | P2 | Timeline cross-refs all closed unmerged with weak verdict; no actionable PR | daisyden | module: inductor, ut_upstream |
| [#3187](https://github.com/intel/torch-xpu-ops/issues/3187) | upstream-pytorch | PyTorch XPU gpu_cpp_wrapper fails with InductorError NotImplementedError | CuiYifeng | needs owner investigation | In the XPU cpp_wrapper / AOTI codegen (torch/_inductor/codegen/cpp_wrapper_gpu.py and XPU-specific…<br>[→ details](details/3187.md) | P2 | No comments, no linked or referenced PR found | liangan1 | ut_upstream |
| [#2229](https://github.com/intel/torch-xpu-ops/issues/2229) | upstream-pytorch | test/test_sparse_csr.py::TestSparseCompressedCPU::test_invalid_input meet message not match | jenniew | Move PR #3073 out of WIP and land | Update the XPU/sparse test expectations in torch-xpu-ops skip/override list to match the actual emi…<br>[→ details](details/2229.md) | P3 | Open WIP PR by assignee jenniew with explicit Related issue: #2229; root cause identified as missing _validate_compressed_sparse_indices on… | wincent8 | skipped |
| [#2531](https://github.com/intel/torch-xpu-ops/issues/2531) | upstream-pytorch | [upstream_ut] AssertionError: Torch not compiled with CUDA enabled | daisyden | Owner @daisyden to triage; only github-actions auto-pass observed | Triage per test: (a) genuinely CUDA-specific (cufft_plan_cache, ctc_loss_cudnn_*, numeric_check_lea…<br>[→ details](details/2531.md) | P3 | Vector 0 empty; timeline only documentation/tracker cross-refs. | daisyden | skipped, port_from_skiplist |
| [#2958](https://github.com/intel/torch-xpu-ops/issues/2958) | upstream-pytorch | AssertionError of test_dtensor_basic_compile | daisyden | Triage: re-validate cross-referenced PRs (none auto-verified) | Verify on latest pytorch main that the test passes (etaf report), then remove the skip entry from t…<br>[→ details](details/2958.md) | P3 | Cross-references exist but no PR explicitly references this issue; manual review needed. | daisyden | module: inductor, ut_upstream |
| [#3041](https://github.com/intel/torch-xpu-ops/issues/3041) | upstream-pytorch | AssertionError: Expected len(flat_diff_results) > 0 in test_fake_crossref_backward_amp_normal_number_mean_xpu_float32 | Silv3S, BartoszKokoszko | Owner @Silv3S, BartoszKokoszko to file fix PR | In torch/testing/_internal/common_methods_invocations.py, extend the existing CUDA DecorateInfo on…<br>[→ details](details/3041.md) | P3 | Issue assigned to @Silv3S, BartoszKokoszko; owner to implement fix. | daisyden | ut_upstream |


<a id="sec-6-2-cpu-fallback"></a>
### 6.2 CPU fallback

_[↑ Back to Index](#sec-2)_

Issues where the XPU operator is missing and a CPU fallback is registered in torch-xpu-ops. Terminal-QA rows excluded.  —  3 issues.

| Issue | Dependency | Title | Owner | action_TBD | Fix Approach | Priority | action_reason | Reporter | Labels |
|---|---|---|---|---|---|---|---|---|---|
| [#2024](https://github.com/intel/torch-xpu-ops/issues/2024) | CPU fallback | AssertionError: Torch not compiled with CUDA enabled | daisyden | Assignee @daisyden to investigate | Either (a) add these test IDs to test/xpu/skip_list_common.py with a clear TODO so CI stays green,…<br>[→ details](details/2024.md) | P2 | Issue already assigned to @daisyden; owner to lead root-cause. | mengfei25 | module: ut, skipped |
| [#2214](https://github.com/intel/torch-xpu-ops/issues/2214) | CPU fallback | test/test_sparse.py::TestSparseAnyXPU::test_gradcheck_mm expected error message not match | jenniew | Assignee @jenniew to investigate | Extend the expected-error regex in test/test_sparse.py (around line 5297) to also accept the XPU-sp…<br>[→ details](details/2214.md) | P2 | Issue already assigned to @jenniew; owner to lead root-cause. | wincent8 | skipped, ut_upstream |
| [#2436](https://github.com/intel/torch-xpu-ops/issues/2436) | CPU fallback | [upstream_ut] AttributeError: 'NoneType' object has no attribute 'clone' | daisyden | Owner @daisyden to re-check case design (per @CuiYifeng request) | Wait for the upstream fix in pytorch/pytorch#97395 (guard None .grad in _expanded_weights clone pat…<br>[→ details](details/2436.md) | P3 | No fix PR; behavior matches CUDA per upstream pytorch/pytorch#97395 (CPU issue). | daisyden | skipped, dependency component: communit… |


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
