# XPU Ops Bug Scrub Report — UT scope

- **Repository**: `intel/torch-xpu-ops`
- **Generated**: 2026-04-30 (cutoff for Section 7: 2026-04-23)
- **Total issues in workbook**: 53
- **Classified (non-empty `action_Type`)**: 53
- **Empty `action_TBD` (no verdict)**: 0

## 1. Summary

This report groups the 53 tracked torch-xpu-ops issues into action buckets derived from the `action_Type` classification column of the triage workbook. Each issue appears in at most one Action-Required or QA section, chosen by its highest-priority category. Cross-cutting slices (duplicated issues, external dependency blockers, newly filed issues) are listed separately for visibility.

**Headline counts (primary category):**

| Bucket | Categories | Issues |
|---|---|---:|
| Developer action required | NEED PR, TRACK PR, NEEDS_OWNER | 40 |
| QA action required | CLOSE or SKIP, AWAIT_REPLY, MONITOR, CHECK_CASES | 12 |
| Duplicated | — | 3 |
| External dependency (non-upstream-pytorch, non-SYCL-kernel) | — | 12 |
| Upstream-pytorch | — | 19 |
| CPU fallback | — | 3 |
| Filed within last 7 days | — | 0 |
| Requests pending > 1 week | — | 15 |

<a id="sec-2"></a>
## 2. Index

- [3. Action required (Developer)](#sec-3)
  - [UNCLASSIFIED](#sec-3-0-unclassified)
  - [NEED PR](#sec-3-1-need-pr)
  - [TRACK PR](#sec-3-2-track-pr)
  - [NEEDS_OWNER](#sec-3-3-needs-owner)
- [4. QA](#sec-4)
  - [CLOSE or SKIP](#sec-4-1-close-or-skip)
  - [AWAIT_REPLY](#sec-4-2-await-reply)
  - [MONITOR](#sec-4-3-monitor)
  - [CHECK_CASES](#sec-4-4-check-cases)
- [5. Duplicated issues](#sec-5)
- [6. Dependency (external blockers)](#sec-6)
  - [Third Parties](#sec-6-1-third-parties)
  - [upstream-pytorch](#sec-6-2-upstream-pytorch)
  - [CPU fallback](#sec-6-3-cpu-fallback)
- [7. New submitted issues (<7 days)](#sec-7)
- [8. Requests pending > 1 week](#sec-8)
- [9. Statistics](#sec-9)

<a id="sec-3"></a>
## 3. Action required (Developer)

_[↑ Back to Index](#sec-2)_

Issues in this section require developer work before they can progress. Each subsection is split by `Category` (existing taxonomy column); rows inside each category table are sorted by `Priority` (P0 → P3).

Issues whose `Dependency` is a third-party blocker (`oneDNN` / `oneMKL` / `oneAPI` / `triton` / `driver` / `xccl`) are hidden here and listed only under §6 Dependency, except when their `action_Type` is `TRACK_PR` or `RETRIAGE_PRS` (a live PR to track).

<a id="sec-3-0-unclassified"></a>
- **UNCLASSIFIED**  ·  0 issues

_[↑ Back to Index](#sec-2)_

**UNCLASSIFIED — Phase 4b produced no verdict; needs manual triage**

_No issues._


<a id="sec-3-1-need-pr"></a>
- **NEED PR**  ·  11 issues

**NEED PR — a PR must be produced or continued (no PR yet, or owner actively debugging root cause, or new code needed)**

<a id="sec-3-1-1-flash-attention"></a>
#### 3.1.1 Flash Attention  ·  2 issues

_[↑ Back to Index](#sec-2)_

| Issue | Title | Owner | Owner Transferred | action_TBD | Fix Approach | Priority | action_reason | Reporter | Labels |
|---|---|---|---|---|---|---|---|---|---|
| [#2285](https://github.com/intel/torch-xpu-ops/issues/2285) | Support efficient attention | chunhuanMeng | chunhuanMeng, daisyden | <ul><li>check_case_avaliablity</li><li>No action — investigate further</li></ul> | Implement and register aten::_efficient_attention_forward / _backward for XPU in torch-xpu-ops (tra…<br>[→ details](details/2285.md) | P1 | Issue is OPEN with zero VERIFIED PR candidates: aten::_efficient_attention_forward has no XPU registration in torch-xpu-ops yet, no fix PR … | daisyden | skipped |
| [#3140](https://github.com/intel/torch-xpu-ops/issues/3140) | [upstream_ut] RuntimeError:<br>FlashAttentionForwardXPU does not only support<br>dropout > 0.0 yet | LuFinch | LuFinch | <ul><li>No action — investigate further</li></ul> | Either (a) implement dropout support in mha_fwd.cpp / mha_bwd.cpp sycltla kernels, or (b) fix the S…<br>[→ details](details/3140.md) | P1 | No fix PR exists for the dropout>0 dispatch gap in FlashAttentionForwardXPU; assignee LuFinch needs to either implement dropout in sycltla … | daisyden | module: ut, skipped, ut_upstream |


<a id="sec-3-1-2-inductor"></a>
#### 3.1.2 Inductor  ·  3 issues

_[↑ Back to Index](#sec-2)_

| Issue | Title | Owner | Owner Transferred | action_TBD | Fix Approach | Priority | action_reason | Reporter | Labels |
|---|---|---|---|---|---|---|---|---|---|
| [#1969](https://github.com/intel/torch-xpu-ops/issues/1969) | torch._dynamo.exc.InternalTorchDynamoError:<br>TypeError: cannot create weak reference to<br>'torch.Event' object | guangyey | guangyey | <ul><li>No action — investigate further</li></ul> | Upstream-pytorch fix: add weakref support to the generic torch.Event class (define __weakref__ slot…<br>[→ details](details/1969.md) | P2 | All candidate upstream PRs (pytorch/pytorch#164522, #163168, #151213) addressing torch.Event weakref support are CLOSED unmerged with no me… | shangerxin | module: ut |
| [#2715](https://github.com/intel/torch-xpu-ops/issues/2715) | [upstream_ut] torch._dynamo.exc.Unsupported:<br>Attempted to inline function marked as skipped | CuiYifeng | CuiYifeng | <ul><li>No action — investigate further</li></ul> | In upstream PyTorch torch/_dynamo/trace_rules.py, allow tracing of torch.xpu.device.__init__ (mirro…<br>[→ details](details/2715.md) | P2 | Issue is OPEN, no VERIFIED PR addresses the dynamo MOD_SKIPLIST exemption for torch.xpu.device.__init__; assignee CuiYifeng needs to invest… | daisyden | skipped, ut_upstream |
| [#2958](https://github.com/intel/torch-xpu-ops/issues/2958) | AssertionError of test_dtensor_basic_compile | daisyden | daisyden | <ul><li>Submit upstream PR from daisyden/missing_test to remove skipIfXpu on test_dtensor_basic_export and close this issue</li><li>File separate upstream issue for test-ordering / DTensorSpec flatten regression from pytorch/pytorch#178115</li></ul> | Re-run on current main to confirm<br>[→ details](details/2958.md) | P3 | Owner has already verified the fix on main and prepared a branch; only remaining step is to land the skip-removal PR. \| The ordering-depen… | daisyden | module: inductor, ut_upstream |


<a id="sec-3-1-3-others"></a>
#### 3.1.3 Others  ·  1 issues

_[↑ Back to Index](#sec-2)_

| Issue | Title | Owner | Owner Transferred | action_TBD | Fix Approach | Priority | action_reason | Reporter | Labels |
|---|---|---|---|---|---|---|---|---|---|
| [#2783](https://github.com/intel/torch-xpu-ops/issues/2783) | [Bug Skip]: Key "xpu" is missing from dict<br>"driver" in test_svd | daisyden | daisyden | <ul><li>Open upstream PR to pytorch/pytorch from branch daisyden/missing_test (adds 'xpu' entry to SVD drivers dict in test/test_linalg.py)</li><li>No action — investigate further</li></ul> | Upstream patch in pytorch/test/test_linalg.py: add an 'xpu' entry to the SVD driver dict (or extend…<br>[→ details](details/2783.md) | P3 | Assignee daisyden has a verified local fix on a personal branch but no upstream PR has been filed yet; landing that PR (or equivalent) is t… | CuiYifeng | module: ut, skipped |


<a id="sec-3-1-4-sparse"></a>
#### 3.1.4 Sparse  ·  1 issues

_[↑ Back to Index](#sec-2)_

| Issue | Title | Owner | Owner Transferred | action_TBD | Fix Approach | Priority | action_reason | Reporter | Labels |
|---|---|---|---|---|---|---|---|---|---|
| [#2214](https://github.com/intel/torch-xpu-ops/issues/2214) | test/test_sparse.py::TestSparseAnyXPU::test_gradch<br>eck_mm expected error message not match | jenniew | jenniew | <ul><li>No action — investigate further</li><li>Address comment AR from chuanqi129 (>1 week): @wincent8 remove hardcoded skips in test files (use dynamic skip-by-issue)</li><li>Address comment AR from daisyden (>1 week): confirm/dedupe with #2283 (jenniew should triage)</li></ul> | Update the assertRaisesRegex pattern in upstream test_sparse.py to also accept the 'empty_sparse_co…<br>[→ details](details/2214.md) | P3 | Issue is OPEN with zero VERIFIED fixes/supersedes PR candidates — the fix lives in upstream pytorch/pytorch test_sparse.py assertRaisesRege… | wincent8 | skipped, ut_upstream |


<a id="sec-3-1-5-torch-operations"></a>
#### 3.1.5 Torch Operations  ·  3 issues

_[↑ Back to Index](#sec-2)_

| Issue | Title | Owner | Owner Transferred | action_TBD | Fix Approach | Priority | action_reason | Reporter | Labels |
|---|---|---|---|---|---|---|---|---|---|
| [#3084](https://github.com/intel/torch-xpu-ops/issues/3084) | torch.library.register_autocast does not support<br>xpu | CuiYifeng | CuiYifeng | <ul><li>No action — investigate further</li></ul> | Upstream PR to pytorch/pytorch: in torch/library.py register_autocast, accept all autocast-capable…<br>[→ details](details/3084.md) | P2 | Issue is OPEN with no linked PRs from any of V0/VA/VB/VC/VD/VE; an upstream pytorch/pytorch fix to torch/library.py register_autocast is re… | daisyden | module: ut |
| [#2436](https://github.com/intel/torch-xpu-ops/issues/2436) | [upstream_ut] AttributeError: 'NoneType' object<br>has no attribute 'clone' | daisyden | daisyden | <ul><li>No action — investigate further</li><li>Address comment AR from daisyden (>1 week): re-check case design per CuiYifeng's request</li></ul> | Keep cases in skip list as 'random/community'<br>[→ details](details/2436.md) | P3 | Issue is OPEN with zero VERIFIED PR candidates (root cause pytorch/pytorch#97395 is an upstream issue, not a PR); needs further investigati… | daisyden | skipped, dependency component: communit… |
| [#3041](https://github.com/intel/torch-xpu-ops/issues/3041) | AssertionError: Expected len(flat_diff_results) ><br>0 in test_fake_crossref_backward_amp_normal_number<br>_mean_xpu_float32 | daisyden | daisyden | <ul><li>No action — investigate further</li></ul> | Add device_type='xpu' (or replace with allowed_dtypes/skip across non-CPU) to the existing Decorate…<br>[→ details](details/3041.md) | P3 | Issue is OPEN with no VERIFIED fix PR yet (cited PR 176690 is the trigger, not the fix, and is closed unmerged). Owner daisyden is assigned… | daisyden | ut_upstream |


<a id="sec-3-1-6-torchao"></a>
#### 3.1.6 TorchAO  ·  1 issues

_[↑ Back to Index](#sec-2)_

| Issue | Title | Owner | Owner Transferred | action_TBD | Fix Approach | Priority | action_reason | Reporter | Labels |
|---|---|---|---|---|---|---|---|---|---|
| [#2578](https://github.com/intel/torch-xpu-ops/issues/2578) | [TorchAO][UT] test/quantization/test_quant_api.py<br>AssertionError: SQNR -2.90625 is too low | Stonepia | Stonepia | <ul><li>check_case_avaliablity</li><li>No action — investigate further</li></ul> | Audit LinearInt4.cpp packing convention (inner_k_tiles=8 TensorCoreTiledLayout) vs torchao's tinyge…<br>[→ details](details/2578.md) | P1 | Issue is OPEN with no VERIFIED fixing/superseding PR; SQNR -2.9 indicates an XPU LinearInt4 packing-layout bug that the assignee Stonepia h… | zxd1997066 | module: ao, ut_upstream |


<a id="sec-3-2-track-pr"></a>
- **TRACK PR**  ·  21 issues

**TRACK PR — a PR is identified; track it to merge, or re-evaluate if prior PRs are dead / unverified**

<a id="sec-3-2-1-flash-attention"></a>
#### 3.2.1 Flash Attention  ·  3 issues

_[↑ Back to Index](#sec-2)_

| Issue | Title | Owner | Owner Transferred | action_TBD | Fix Approach | Priority | action_reason | Reporter | Labels |
|---|---|---|---|---|---|---|---|---|---|
| [#2442](https://github.com/intel/torch-xpu-ops/issues/2442) | [Bug Skip]: NotImplementedError: Could not run<br>'aten::_flash_attention_forward' with arguments<br>from the 'CPU' backend | LuFinch | LuFinch | <ul><li>Track PR intel/torch-xpu-ops#3404 to merge</li><li>Address CI failures on PR intel/torch-xpu-ops#3404</li></ul> | Add the four cases to the skip list (already labelled 'skipped')<br>[→ details](details/2442.md) | P2 | VERIFIED PR #3404 (github_linked, relationship=fixes) is OPEN and adds the XPU _flash_attention_forward/_backward registration described in… | CuiYifeng | skipped |
| [#2698](https://github.com/intel/torch-xpu-ops/issues/2698) | Title: [upstream_ut] RuntimeError:<br>FlashAttentionForwardXPU only support headdim<br>64,96,128,192 | LuFinch | LuFinch | <ul><li>Track PR pytorch/pytorch#180646 to merge</li><li>Address CI failures on PR pytorch/pytorch#180646 (>1 week)</li></ul> | In mha_fwd.cpp, instead of TORCH_CHECK(false, ...), return a status indicating unsupported head_dim…<br>[→ details](details/2698.md) | P2 | VERIFIED fixes PR is OPEN and APPROVED; needs to land. \| Latest failing required check (linux-noble-xpu-n-py3.10 / test 12/12) completed 2… | daisyden | module: inductor, skipped, ut_upstream |
| [#2853](https://github.com/intel/torch-xpu-ops/issues/2853) | [upstream_ut]<br>torch.ops.aten._flash_attention_forward lack of<br>support for XPU. | LuFinch | LuFinch | <ul><li>Track PR intel/torch-xpu-ops#3404 to merge</li></ul> | Either (a) register an XPU kernel for _flash_attention_forward in torch-xpu-ops (transformers/sycl/…<br>[→ details](details/2853.md) | P2 | PR #3404 (assignee LuFinch) is the verified fix and is currently OPEN with no review decision yet; needs to be driven through review and me… | BBBela | skipped |


<a id="sec-3-2-2-inductor"></a>
#### 3.2.2 Inductor  ·  4 issues

_[↑ Back to Index](#sec-2)_

| Issue | Title | Owner | Owner Transferred | action_TBD | Fix Approach | Priority | action_reason | Reporter | Labels |
|---|---|---|---|---|---|---|---|---|---|
| [#2554](https://github.com/intel/torch-xpu-ops/issues/2554) | [upstream_ut] AssertionError: AssertionError not<br>raised | daisyden | daisyden | <ul><li>Track PR pytorch/pytorch#181822 to merge</li></ul> | Wait for / pull in the fix from intel-xpu-backend-for-triton#5654 (TTGIR pass), then unskip the thr…<br>[→ details](details/2554.md) | P2 | PR pytorch/pytorch#181822 is the live OPEN fix re-enabling 2 of the 3 cases (the third case test_selecsls42b_misaligned_address is not list… | daisyden | module: inductor, skipped |
| [#2609](https://github.com/intel/torch-xpu-ops/issues/2609) | [upstream_ut] torch._inductor.exc.InductorError:<br>CppCompileError: C++ compile error | etaf | etaf | <ul><li>Track PR pytorch/pytorch#171154 to merge</li><li>Address CI failures on PR pytorch/pytorch#171154 (>1 week)</li><li>Address comment AR from etaf: merge the approved fix PR pytorch/pytorch#171154 to land aoti_torch_xpu_fn_<op> shim</li></ul> | Land pytorch/pytorch#171154 which generates/declares aoti_torch_xpu_fn_<op> for XPU custom ops in t…<br>[→ details](details/2609.md) | P2 | VERIFIED fixing PR pytorch/pytorch#171154 is OPEN and APPROVED but not merged. \| PR has 5 failing XPU CI checks whose latest completedAt i… | daisyden | module: inductor, skipped, ut_upstream |
| [#3187](https://github.com/intel/torch-xpu-ops/issues/3187) | PyTorch XPU gpu_cpp_wrapper fails with<br>InductorError NotImplementedError | CuiYifeng | CuiYifeng | <ul><li>PR pytorch/pytorch#178477 closed unmerged; reassess fix path</li><li>RETRIAGE_PRS</li></ul> | Investigate cpp_wrapper fallback dispatch for XPU in torch/_inductor/codegen/cpp_wrapper_gpu.py and…<br>[→ details](details/3187.md) | P2 | The only referenced PR (the upstream temporary skip workaround) was closed unmerged, and no replacement PR addresses the underlying XPU cpp… | liangan1 | ut_upstream |
| [#2891](https://github.com/intel/torch-xpu-ops/issues/2891) | RuntimeError: Expected to find "(262144, 0, 512,<br>1" but did not find it | chunhuanMeng | chunhuanMeng | <ul><li>Track PR pytorch/pytorch#180418 to merge</li></ul> | Confirm upstream pytorch PR #180418 (unskip) is landed and pulled into torch-xpu-ops test list<br>[→ details](details/2891.md) | P3 | PR pytorch/pytorch#180418 (unskip test_effn_attn_bias_padding) is OPEN, APPROVED, and CI green; only Gate 4 (merge) remains. Once merged, t… | daisyden | module: inductor, ut_upstream |


<a id="sec-3-2-3-others"></a>
#### 3.2.3 Others  ·  3 issues

_[↑ Back to Index](#sec-2)_

| Issue | Title | Owner | Owner Transferred | action_TBD | Fix Approach | Priority | action_reason | Reporter | Labels |
|---|---|---|---|---|---|---|---|---|---|
| [#2024](https://github.com/intel/torch-xpu-ops/issues/2024) | AssertionError: Torch not compiled with CUDA<br>enabled | daisyden | daisyden | <ul><li>Track PR intel/torch-xpu-ops#3510 to merge</li></ul> | Submit upstream PRs replacing hardcoded device='cuda' with the parameterized device argument (or @o…<br>[→ details](details/2024.md) | P2 | PR #3510 (OPEN, APPROVED, CI pending) explicitly enables 5 of the 7 cases listed in this issue and is one merge away from resolving the iss… | mengfei25 | module: ut, skipped |
| [#2287](https://github.com/intel/torch-xpu-ops/issues/2287) | [upstream_ut] test_python_ref issues | yucai-intel | yucai-intel | <ul><li>Track PR pytorch/pytorch#178734 to merge</li><li>Address CI failures on PR pytorch/pytorch#178734 (>1 week)</li><li>Address comment AR from tye1: pytorch/pytorch#178734 has a lint issue — yucai-intel to fix</li></ul> | Skip in test/xpu/skip_list_common.py for test_python_ref__refs_logspace_tensor_overload (and any re…<br>[→ details](details/2287.md) | P3 | PR 178734 (supersedes #169565) explicitly Fixes #2287 and is APPROVED. \| Latest failing required check (lintrunner-noclang-partial / lint … | daisyden | module: ut, ut_upstream |
| [#2531](https://github.com/intel/torch-xpu-ops/issues/2531) | [upstream_ut] AssertionError: Torch not compiled<br>with CUDA enabled | guangyey | guangyey | <ul><li>check_case_avaliablity</li><li>Track PR intel/torch-xpu-ops#3510 to merge</li><li>Address comment AR from guangyey: confirm XPU support for tunable / cufft_plan_cache / CudaSyncGuard / Miopen / quantize_per_tensor</li></ul> | Port each test in third_party/torch-xpu-ops/test/xpu/ to substitute xpu for cuda (use TEST_XPU/torc…<br>[→ details](details/2531.md) | P3 | VERIFIED partial-fix PR #3510 (relationship=fixes, OPEN, approved by chuanqi129) lands 2 of 13 cases; remaining cases need feature work tra… | daisyden | skipped, port_from_skiplist |


<a id="sec-3-2-4-sparse"></a>
#### 3.2.4 Sparse  ·  4 issues

_[↑ Back to Index](#sec-2)_

| Issue | Title | Owner | Owner Transferred | action_TBD | Fix Approach | Priority | action_reason | Reporter | Labels |
|---|---|---|---|---|---|---|---|---|---|
| [#2283](https://github.com/intel/torch-xpu-ops/issues/2283) | [upstream_ut] sparse._sampled_addmm is not<br>supported | jenniew | jenniew | <ul><li>Track PR intel/torch-xpu-ops#3018 to merge</li><li>Address CI failures on PR intel/torch-xpu-ops#3018 (>1 week)</li></ul> | Either register a CPU fallback for aten::sparse_sampled_addmm on SparseCsrXPU (analogous to other s…<br>[→ details](details/2283.md) | P1 | PR 3018 implements the missing SparseCsrXPU sampled_addmm registration; needs to land. \| linux-ut (op_ut) summary failed on 2026-03-26 (>1… | daisyden | skipped, ut_upstream |
| [#2244](https://github.com/intel/torch-xpu-ops/issues/2244) | test/test_sparse_csr.py::TestSparseCSRXPU::test_bl<br>ock_addmm meet RuntimeError:<br>empty_sparse_compressed expected sparse compressed<br>(non-block) tensor layout but got SparseBsr | jafraustro | jafraustro | <ul><li>Track PR intel/torch-xpu-ops#3476 to merge</li><li>Resolve unresolved review comments on PR intel/torch-xpu-ops#3476</li></ul> | Extend the upstream sparse compressed addmm dispatch (or torch-xpu-ops registration) so XPU SparseB…<br>[→ details](details/2244.md) | P2 | PR 3476 is the OPEN supersede of PR 2974 (closed unmerged) and addresses the residual bf16/fp16 sparse BSR addmm precision failures of #224… | wincent8 | module: ut, skipped |
| [#3177](https://github.com/intel/torch-xpu-ops/issues/3177) | Accuracy gap of BF16/FP16 test_block_addmm | jenniew | jenniew | <ul><li>Track PR intel/torch-xpu-ops#3273 to merge</li><li>Resolve unresolved review comments on PR intel/torch-xpu-ops#3273 (>1 week)</li></ul> | Promote the inner-product accumulator to float32 in the BSR addmm SYCL kernel (mirror CUDA's accumu…<br>[→ details](details/3177.md) | P2 | PR #3273 is the OPEN fixes candidate for #3177. \| Reviewer CuiYifeng asked an unanswered question on 2026-04-07 (22 days old, stale), and … | CuiYifeng | skipped |
| [#2229](https://github.com/intel/torch-xpu-ops/issues/2229) | test/test_sparse_csr.py::TestSparseCompressedCPU::<br>test_invalid_input meet message not match | jenniew | jenniew | <ul><li>Track PR intel/torch-xpu-ops#3073 to merge</li><li>Address CI failures on PR intel/torch-xpu-ops#3073 (>1 week)</li></ul> | Implement aten::_validate_compressed_sparse_indices for the XPU backend in torch-xpu-ops so the can…<br>[→ details](details/2229.md) | P3 | PR 3073 is the OPEN fix authored by assignee jenniew implementing the _validate_compressed_sparse_indices XPU kernel. \| preci-lint-check h… | wincent8 | skipped |


<a id="sec-3-2-5-torch-operations"></a>
#### 3.2.5 Torch Operations  ·  6 issues

_[↑ Back to Index](#sec-2)_

| Issue | Title | Owner | Owner Transferred | action_TBD | Fix Approach | Priority | action_reason | Reporter | Labels |
|---|---|---|---|---|---|---|---|---|---|
| [#2015](https://github.com/intel/torch-xpu-ops/issues/2015) | inf is returned by nn.TransformerEncoderLayer | yucai-intel | yucai-intel | <ul><li>check_case_avaliablity</li><li>Track PR intel/torch-xpu-ops#2336 to merge</li><li>Resolve unresolved review comments on PR intel/torch-xpu-ops#2336 (>1 week)</li><li>Address comment AR from tye1: get PR #2336 merged</li></ul> | Land/track pytorch/pytorch#168234 to add xpu device coverage in test_nn.py TransformerEncoderLayer…<br>[→ details](details/2015.md) | P1 | PR #2336 (OPEN) explicitly fixes this issue's TransformerEncoderLayer XPU test failure; needs to land. \| CuiYifeng's CHANGES_REQUESTED rev… | daisyden | skipped |
| [#2412](https://github.com/intel/torch-xpu-ops/issues/2412) | Some NestedTensor missing XPU support | yucai-intel, BBBela | BBBela | <ul><li>Track PR intel/torch-xpu-ops#2483 to merge</li><li>Address CI failures on PR intel/torch-xpu-ops#2483 (>1 week)</li><li>Resolve unresolved review comments on PR intel/torch-xpu-ops#2483 (>1 week)</li></ul> | Add XPU support to the four nested-tensor files in pytorch/pytorch upstream by relaxing `is_cuda()`…<br>[→ details](details/2412.md) | P2 | PR #2483 is the active OPEN fix; BBBela committed on 2026-04-29 to drive it to merge. \| linux-build check has been failing since 2026-03-3… | daisyden | module: ut |
| [#1893](https://github.com/intel/torch-xpu-ops/issues/1893) | [upstream_ut] oneDNN accuracy issues in<br>test_ops_xpu.py | chunhuanMeng | chunhuanMeng | <ul><li>Track PR pytorch/pytorch#179125 to merge</li><li>Address CI failures on PR pytorch/pytorch#179125 (>1 week)</li></ul> | Bump tolerance for these mv/addmv ops in test_ops xpu skip/tolerance lists (third_party/torch-xpu-o…<br>[→ details](details/1893.md) | P3 | PR pytorch/pytorch#179125 is the fixes PR addressing the addmv stride/accuracy gap and remains OPEN. \| linux-jammy-py3.10-clang18-asan tes… | daisyden | skipped, ut_upstream |
| [#2439](https://github.com/intel/torch-xpu-ops/issues/2439) | [oneDNN]<br>TestDecompXPU.test_quick_addmv_xpu_float64 got<br>fail accuracy result | libohao1201 | libohao1201 | <ul><li>PR pytorch/pytorch#174590 closed unmerged; reassess fix path</li><li>RETRIAGE_PRS</li></ul> | Pick up upstream PR pytorch/pytorch#174590 once merged (raises decomp tolerance for addmv)<br>[→ details](details/2439.md) | P3 | The only VERIFIED PR (pytorch/pytorch#174590, '[xpu] Add proper float64 handling for addmv, addmm and baddbmm.') was approved but closed un… | mengfei25 | dependency component: oneDNN, module: ut |
| [#2512](https://github.com/intel/torch-xpu-ops/issues/2512) | [upstream_ut] RuntimeError: _histc_xpu does not<br>have a deterministic implementation, but you set<br>'torch.use_deter | chunhuanMeng | chunhuanMeng | <ul><li>check_case_avaliablity</li><li>Track PR intel/torch-xpu-ops#3333 to merge</li><li>Address CI failures on PR intel/torch-xpu-ops#3333</li></ul> | Align XPU alert message with CUDA: change SummaryOps.cpp:45 to globalContext().alertNotDeterministi…<br>[→ details](details/2512.md) | P3 | VERIFIED PR #3333 (content_match, relationship=fixes) is OPEN and modifies SummaryOps.cpp + the xpu histc test as described in fix_approach… | libohao1201 | skipped |
| [#3033](https://github.com/intel/torch-xpu-ops/issues/3033) | [Bug Skip]: Softmax tolerance | chunhuanMeng | jkosnox | <ul><li>Track PR intel/torch-xpu-ops#3036 to merge</li></ul> | Investigate SoftMaxKernels.cpp accumulation/promotion path for bool inputs, align with CPU referenc…<br>[→ details](details/3033.md) | P3 | PR #3036 (Fix problems with softmax tolerance) is OPEN, not yet reviewed/approved (reviewDecision empty); track to merge so the bool softma… | chunhuanMeng | skipped, random |


<a id="sec-3-2-6-torchao"></a>
#### 3.2.6 TorchAO  ·  1 issues

_[↑ Back to Index](#sec-2)_

| Issue | Title | Owner | Owner Transferred | action_TBD | Fix Approach | Priority | action_reason | Reporter | Labels |
|---|---|---|---|---|---|---|---|---|---|
| [#2532](https://github.com/intel/torch-xpu-ops/issues/2532) | Title: [upstream_ut] AssertionError: wrong number<br>of dimensions2 for op:<br>torch.ops.aten._convert_weight_to_int4pack.defa | yucai-intel | yucai-intel | <ul><li>Track PR intel/torch-xpu-ops#3090 to merge</li><li>Resolve unresolved review comments on PR intel/torch-xpu-ops#3090 (>1 week)</li><li>Address CI failures on PR intel/torch-xpu-ops#3090 (>1 week)</li></ul> | Align the XPU implementation of _convert_weight_to_int4pack with the CUDA contract (input kByte, ma…<br>[→ details](details/2532.md) | P2 | PR #3090 is the verified fix (explicit issue link, modifies the test file for the 8 listed cases) and is OPEN with two MEMBER approvals. \|… | daisyden | skipped, port_from_skiplist |


<a id="sec-3-3-needs-owner"></a>
- **NEEDS_OWNER**  ·  0 issues

**NEEDS_OWNER — awaiting triage-lead to assign an owner**


<a id="sec-4"></a>
## 4. QA

_[↑ Back to Index](#sec-2)_

Issues in this section are ready for QA action (close, verify, reply, etc.). Rows sorted by `Priority` (P0 → P3).

<a id="sec-4-1-close-or-skip"></a>
- **CLOSE or SKIP**  ·  12 issues

**CLOSE or SKIP — terminal QA action (close fixed, verify merged fix, skip not-target/wontfix, or label not_target and close)**

| Issue | Title | Owner | Owner Transferred | action_TBD | Fix Approach | Priority | action_reason | Reporter | Labels |
|---|---|---|---|---|---|---|---|---|---|
| [#2997](https://github.com/intel/torch-xpu-ops/issues/2997) | AssertionError of test_linear_and_cel_max_autotune | daisyden | daisyden | <ul><li>Track PR pytorch/pytorch#181822 to merge</li><li>Verify fix from merged PR pytorch/pytorch#181822 and close</li></ul> | Bisect: disable inplace-padding (config.inplace_padding=False) and disable oneDNN-backed bf16 GEMM…<br>[→ details](details/2997.md) | P1 | PR 181822 (OPEN, WIP) is daisyden's verification PR removing the @skipIfXpu for #2997; once merged it confirms the community fix and closes… | daisyden | module: inductor, ut_upstream |
| [#2253](https://github.com/intel/torch-xpu-ops/issues/2253) | the supported dtypes are not align with cuda | daisyden | daisyden | <ul><li>label not_target and close</li></ul> | Either expand XPU op registrations to cover the same dtype set as CUDA (e.g. addmm/addmv/baddbmm co…<br>[→ details](details/2253.md) | P2 | Issue is labeled `duplicate` and the reporter+assignee (daisyden, MEMBER) declared it a duplicate of intel/torch-xpu-ops#2289 — consolidate… | daisyden | duplicate, skipped, ut_upstream |
| [#2270](https://github.com/intel/torch-xpu-ops/issues/2270) | Backend Compatibility Error in<br>test/xpu/test_decomp.py | LuFinch | LuFinch | <ul><li>Verify fix from merged PR intel/torch-xpu-ops#2341 and close</li><li>label not_target and close</li></ul> | Add the test to skip_list for test_decomp_xpu (decomp cross-ref doesn't make sense for backend-priv…<br>[→ details](details/2270.md) | P2 | PR #2341 (Integrate FlashAttention fwd/bwd kernels) is MERGED and provides the missing _flash_attention_forward kernel; rerun the test_deco… | libohao1201 | module: ut, skipped |
| [#2376](https://github.com/intel/torch-xpu-ops/issues/2376) | [Bug Skip]: NotImplementedError: "logaddexp_xpu"<br>not implemented for 'Complex' | daisyden, CuiYifeng | CuiYifeng, mengfei25 | <ul><li>Verify fix from merged PR intel/torch-xpu-ops#2807 and close, check_case_avaliablity</li><li>Enable complex logaddexp test cases upstream in pytorch/pytorch and unskip in torch-xpu-ops skip lists</li></ul> | Extend LogAddExpKernels.cpp dispatch to AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND* (mirroring CUDA…<br>[→ details](details/2376.md) | P2 | Highest-priority VERIFIED PR intel/torch-xpu-ops#2807 is MERGED implementing complex logaddexp kernel; per derivation rule emit VERIFY_AND_… | mengfei25 | module: ut, skipped |
| [#2720](https://github.com/intel/torch-xpu-ops/issues/2720) | [upstream_ut] RuntimeError: false INTERNAL ASSERT<br>FAILED at "/pytorch/aten/src/ATen/native/DispatchS<br>tub.cpp":275 | CuiYifeng | wincent8 \| CuiYifeng | <ul><li>Close the fixed issue</li><li>Verify fix from merged PR intel/torch-xpu-ops#3248 and close</li></ul> | Add an XPU kernel registration for ldexp_stub in torch-xpu-ops: implement at third_party/torch-xpu-…<br>[→ details](details/2720.md) | P2 | PR #3248 explicitly targeted this issue ("To solve #2720"), registered the missing ldexp_stub XPU dispatch, and merged on 2026-04-03; remai… | wincent8 | skipped |
| [#3132](https://github.com/intel/torch-xpu-ops/issues/3132) | [upstream_ut] transfomers test reports<br>RuntimeError: No available kernel. Aborting<br>execution. | daisyden | daisyden | <ul><li>label not_target and close</li><li>Track PR intel/torch-xpu-ops#3510 to merge</li></ul> | Per maintainer LuFinch: skip cuDNN cases permanently (CUDA-only backend)<br>[→ details](details/3132.md) | P2 | Skip entries for the 70 cases in this issue have been added to PR #3510 (commit a9b5b50c) per assignee daisyden's 2026-04-29 comment; PR is… | daisyden | module: ut, skipped, ut_upstream, not_t… |
| [#2245](https://github.com/intel/torch-xpu-ops/issues/2245) | oneDNN matmul received incorrect shape in test/tes<br>t_sparse_csr.py::TestSparseCSRXPU::test_addmm_erro<br>rs_xpu_float32 | CuiYifeng | CuiYifeng | <ul><li>Verify fix from merged PR intel/torch-xpu-ops#3487 and close</li></ul> | Land upstream PyTorch PR #180985 to add the expanded-input shape validation in the addmm meta/dispa…<br>[→ details](details/2245.md) | P3 | intel/torch-xpu-ops#3487 explicitly fixes #2245 by adding the missing dimension check on the XPU sparse CSR addmm path; merged 2026-04-29. | wincent8 | module: ut, skipped |
| [#2541](https://github.com/intel/torch-xpu-ops/issues/2541) | Title: [upstream_ut] RuntimeError: could not<br>construct a memory descriptor using strides | yucai-intel | daisyden \| yucai-intel | <ul><li>Close the fixed issue</li><li>Track PR pytorch/pytorch#176875 to merge</li></ul> | Land/cherry-pick upstream pytorch/pytorch#176875 which makes einsum contiguous before bmm/matmul on…<br>[→ details](details/2541.md) | P3 | Fixed and passed in CI \| Verified upstream PR fixes the einsum oneDNN memory-descriptor error and is currently OPEN awaiting review/merge. | daisyden | skipped, port_from_skiplist |
| [#2615](https://github.com/intel/torch-xpu-ops/issues/2615) | [Bug Skip]: New failures RuntimeError: Unsupported<br>dtype Half / RuntimeError: Unsupported dtype<br>torch.float16 | CuiYifeng | CuiYifeng | <ul><li>label not_target and close (for the test_comprehensive_fft_*_xpu_float16 and #2708-tracked subset)</li><li>Verify fix from merged PR intel/torch-xpu-ops#2637 and close</li><li>Track PR intel/torch-xpu-ops#2996 to merge</li><li>Resolve unresolved review comments on PR intel/torch-xpu-ops#2996</li></ul> | Land pytorch/pytorch#171231 and skip/redirect the remaining test_fft_half_and_chalf_not_power_of_tw…<br>[→ details](details/2615.md) | P3 | Most fft_*_xpu_float16 cases already pass per nightly bot once #2637 added Half/Complex32 FFT support; remaining work is verification. \| P… | kaileiyx | module: ut, skipped |
| [#3136](https://github.com/intel/torch-xpu-ops/issues/3136) | [upstream_ut] AssertionError: False is not true in<br>test_transformers | daisyden | daisyden | <ul><li>label not_target and close</li><li>Track PR intel/torch-xpu-ops#3510 to merge</li></ul> | Keep the skip entries added by PR #3510 in test/xpu/skip_list_common.py.<br>[→ details](details/3136.md) | P3 | Skip entries for the 4 cases (commit 475bdf37) have been committed to PR #3510 per assignee daisyden's 2026-04-29 comment; PR is APPROVED, … | daisyden | module: ut, skipped, ut_upstream, not_t… |
| [#3137](https://github.com/intel/torch-xpu-ops/issues/3137) | [upstream_ut] RuntimeError: expected scalar type<br>Half but found Float | daisyden | daisyden | <ul><li>label not_target and close</li></ul> | No code fix needed in torch-xpu-ops: keep the skip entries from PR #3510 in test/xpu/skip_list_comm…<br>[→ details](details/3137.md) | P3 | Authoritative owner LuFinch (MEMBER) declared the fastpath deprecated and these cases not_target; skip entries already landed in PR #3510 c… | daisyden | module: ut, skipped, ut_upstream, rando… |
| [#3170](https://github.com/intel/torch-xpu-ops/issues/3170) | Unskip test_bmm_windows_error_xpu_float64 | libohao1201, jafraustro | libohao1201 | <ul><li>Verify fix from merged PR intel/torch-xpu-ops#3225 and close</li><li>Address comment AR from libohao1201 (>1 week): verify the issue on windows</li><li>Address comment AR from chuanqi129: how to verify the Windows-only fix given CI gap</li></ul> | Remove the test_bmm_windows_error_xpu_float64 entry from the xpu skip list (test/xpu/skip_list*.py…<br>[→ details](details/3170.md) | P3 | PR #3225 explicitly references and merged the fix for #3170, modifying the cited test file. \| daisyden requested libohao1201 verify on Win… | CuiYifeng | skipped, ut_upstream |


<a id="sec-4-2-await-reply"></a>
- **AWAIT_REPLY**  ·  0 issues

**AWAIT_REPLY — open questions in thread; owner must respond**

_No issues._


<a id="sec-4-3-monitor"></a>
- **MONITOR**  ·  0 issues

**MONITOR — long-running tracker / maintenance / scoping**

_No issues._


<a id="sec-4-4-check-cases"></a>
- **CHECK_CASES**  ·  6 issues

**CHECK_CASES — XPU test case missing in repo; QA must verify case existence before action**

| Issue | Title | Owner | Owner Transferred | action_TBD | Fix Approach | Priority | action_reason | Reporter | Labels |
|---|---|---|---|---|---|---|---|---|---|
| [#2015](https://github.com/intel/torch-xpu-ops/issues/2015) | inf is returned by nn.TransformerEncoderLayer | yucai-intel | yucai-intel | <ul><li>check_case_avaliablity</li><li>Track PR intel/torch-xpu-ops#2336 to merge</li><li>Resolve unresolved review comments on PR intel/torch-xpu-ops#2336 (>1 week)</li><li>Address comment AR from tye1: get PR #2336 merged</li></ul> | Land/track pytorch/pytorch#168234 to add xpu device coverage in test_nn.py TransformerEncoderLayer…<br>[→ details](details/2015.md) | P1 | PR #2336 (OPEN) explicitly fixes this issue's TransformerEncoderLayer XPU test failure; needs to land. \| CuiYifeng's CHANGES_REQUESTED rev… | daisyden | skipped |
| [#2285](https://github.com/intel/torch-xpu-ops/issues/2285) | Support efficient attention | chunhuanMeng | chunhuanMeng, daisyden | <ul><li>check_case_avaliablity</li><li>No action — investigate further</li></ul> | Implement and register aten::_efficient_attention_forward / _backward for XPU in torch-xpu-ops (tra…<br>[→ details](details/2285.md) | P1 | Issue is OPEN with zero VERIFIED PR candidates: aten::_efficient_attention_forward has no XPU registration in torch-xpu-ops yet, no fix PR … | daisyden | skipped |
| [#2578](https://github.com/intel/torch-xpu-ops/issues/2578) | [TorchAO][UT] test/quantization/test_quant_api.py<br>AssertionError: SQNR -2.90625 is too low | Stonepia | Stonepia | <ul><li>check_case_avaliablity</li><li>No action — investigate further</li></ul> | Audit LinearInt4.cpp packing convention (inner_k_tiles=8 TensorCoreTiledLayout) vs torchao's tinyge…<br>[→ details](details/2578.md) | P1 | Issue is OPEN with no VERIFIED fixing/superseding PR; SQNR -2.9 indicates an XPU LinearInt4 packing-layout bug that the assignee Stonepia h… | zxd1997066 | module: ao, ut_upstream |
| [#2376](https://github.com/intel/torch-xpu-ops/issues/2376) | [Bug Skip]: NotImplementedError: "logaddexp_xpu"<br>not implemented for 'Complex' | daisyden, CuiYifeng | CuiYifeng, mengfei25 | <ul><li>Verify fix from merged PR intel/torch-xpu-ops#2807 and close, check_case_avaliablity</li><li>Enable complex logaddexp test cases upstream in pytorch/pytorch and unskip in torch-xpu-ops skip lists</li></ul> | Extend LogAddExpKernels.cpp dispatch to AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND* (mirroring CUDA…<br>[→ details](details/2376.md) | P2 | Highest-priority VERIFIED PR intel/torch-xpu-ops#2807 is MERGED implementing complex logaddexp kernel; per derivation rule emit VERIFY_AND_… | mengfei25 | module: ut, skipped |
| [#2512](https://github.com/intel/torch-xpu-ops/issues/2512) | [upstream_ut] RuntimeError: _histc_xpu does not<br>have a deterministic implementation, but you set<br>'torch.use_deter | chunhuanMeng | chunhuanMeng | <ul><li>check_case_avaliablity</li><li>Track PR intel/torch-xpu-ops#3333 to merge</li><li>Address CI failures on PR intel/torch-xpu-ops#3333</li></ul> | Align XPU alert message with CUDA: change SummaryOps.cpp:45 to globalContext().alertNotDeterministi…<br>[→ details](details/2512.md) | P3 | VERIFIED PR #3333 (content_match, relationship=fixes) is OPEN and modifies SummaryOps.cpp + the xpu histc test as described in fix_approach… | libohao1201 | skipped |
| [#2531](https://github.com/intel/torch-xpu-ops/issues/2531) | [upstream_ut] AssertionError: Torch not compiled<br>with CUDA enabled | guangyey | guangyey | <ul><li>check_case_avaliablity</li><li>Track PR intel/torch-xpu-ops#3510 to merge</li><li>Address comment AR from guangyey: confirm XPU support for tunable / cufft_plan_cache / CudaSyncGuard / Miopen / quantize_per_tensor</li></ul> | Port each test in third_party/torch-xpu-ops/test/xpu/ to substitute xpu for cuda (use TEST_XPU/torc…<br>[→ details](details/2531.md) | P3 | VERIFIED partial-fix PR #3510 (relationship=fixes, OPEN, approved by chuanqi129) lands 2 of 13 cases; remaining cases need feature work tra… | daisyden | skipped, port_from_skiplist |


<a id="sec-5"></a>
## 5. Duplicated issues

_[↑ Back to Index](#sec-2)_

Rows where `duplicated_issue` is set or `action_TBD` contains "duplicate of".  —  3 issues.

| Issue | Duplicates | Title | Owner | Owner Transferred | action_TBD | Fix Approach | Priority | action_reason | Reporter | Labels |
|---|---|---|---|---|---|---|---|---|---|---|
| [#2285](https://github.com/intel/torch-xpu-ops/issues/2285) | [#2853](https://github.com/intel/torch-xpu-ops/issues/2853) | Support efficient attention | chunhuanMeng | chunhuanMeng, daisyden | <ul><li>check_case_avaliablity</li><li>No action — investigate further</li></ul> | Implement and register aten::_efficient_attention_forward / _backward for XPU in torch-xpu-ops (tra…<br>[→ details](details/2285.md) | P1 | Issue is OPEN with zero VERIFIED PR candidates: aten::_efficient_attention_forward has no XPU registration in torch-xpu-ops yet, no fix PR … | daisyden | skipped |
| [#2715](https://github.com/intel/torch-xpu-ops/issues/2715) | [#3286](https://github.com/intel/torch-xpu-ops/issues/3286) | [upstream_ut] torch._dynamo.exc.Unsupported:<br>Attempted to inline function marked as skipped | CuiYifeng | CuiYifeng | <ul><li>No action — investigate further</li></ul> | In upstream PyTorch torch/_dynamo/trace_rules.py, allow tracing of torch.xpu.device.__init__ (mirro…<br>[→ details](details/2715.md) | P2 | Issue is OPEN, no VERIFIED PR addresses the dynamo MOD_SKIPLIST exemption for torch.xpu.device.__init__; assignee CuiYifeng needs to invest… | daisyden | skipped, ut_upstream |
| [#2853](https://github.com/intel/torch-xpu-ops/issues/2853) | [#2285](https://github.com/intel/torch-xpu-ops/issues/2285) | [upstream_ut]<br>torch.ops.aten._flash_attention_forward lack of<br>support for XPU. | LuFinch | LuFinch | <ul><li>Track PR intel/torch-xpu-ops#3404 to merge</li></ul> | Either (a) register an XPU kernel for _flash_attention_forward in torch-xpu-ops (transformers/sycl/…<br>[→ details](details/2853.md) | P2 | PR #3404 (assignee LuFinch) is the verified fix and is currently OPEN with no review decision yet; needs to be driven through review and me… | BBBela | skipped |


<a id="sec-6"></a>
## 6. Dependency (external blockers)

_[↑ Back to Index](#sec-2)_

Issues with a non-blank `Dependency` value, excluding `upstream-pytorch`, `CPU fallback`, and `SYCL kernel:*` (in-repo kernel pointers). Terminal-QA rows (CLOSE / VERIFY_AND_CLOSE / SKIP / NOT_TARGET_CLOSE) are also excluded.  —  12 issues.

<a id="sec-6-1-third-parties"></a>
- **Third Parties**

_[↑ Back to Index](#sec-2)_

| Issue | Dependency | Title | Owner | Owner Transferred | action_TBD | Fix Approach | Priority | action_reason | Reporter | Labels |
|---|---|---|---|---|---|---|---|---|---|---|
| [#2295](https://github.com/intel/torch-xpu-ops/issues/2295) | oneAPI | [upstream_ut][xpu][test]nn/test_embedding.py::Test<br>EmbeddingNNDeviceTypeXPU::test_embedding_bag_devic<br>e_xpu_int32_int32_float64 meet AssertionError:<br>Tensor-likes are not close! | mengfei25 | mengfei25 | <ul><li>No action — investigate further</li></ul> | Identify the IGC/oneAPI version delta between CI image and nightly build<br>[→ details](details/2295.md) | P1 | Open issue with zero VERIFIED fix PRs; cited PRs (#165886, #169168) only carry temporary WAs and are closed unmerged. Root cause is suspect… | wincent8 | module: inductor, skipped, ut_upstream |
| [#2769](https://github.com/intel/torch-xpu-ops/issues/2769) | oneDNN | [oneDNN] New failed test cases with 3.11 compared<br>with 3.10 | mengfei25 | mengfei25 | <ul><li>Address comment AR from chuanqi129 (>1 week): verify whether oneDNN 3.11.1 fixes the conv regression</li><li>No action — investigate further</li></ul> | File a oneDNN upstream issue with a minimal conv repro<br>[→ details](details/2769.md) | P1 | chuanqi129 asked mengfei25 on 2026-03-25 to verify the fix in oneDNN 3.11.1; no response in 35 days. \| No torch-xpu-ops PR exists; root ca… | mengfei25 | hw: PVC, dependency component: oneDNN, … |
| [#2613](https://github.com/intel/torch-xpu-ops/issues/2613) | driver | [upstream_ut] AssertionError: Tensor-likes are not<br>equal! in test_compile_subprocess.py | daisyden | daisyden | <ul><li>Address comment AR from daisyden (>1 week): re-verify argmax/argmin tie-break tests with the new driver per GSD-11415</li><li>No action — investigate further</li></ul> | Wait for the driver/IGC fix tracked in GSD-11415<br>[→ details](details/2613.md) | P2 | Assignee daisyden self-noted 'Need verify' 13 days ago to re-test the driver fix tracked in GSD-11415; that re-verification has not been re… | daisyden | dependency component: driver, module: i… |
| [#3142](https://github.com/intel/torch-xpu-ops/issues/3142) | oneAPI | [upstream_ut] RuntimeError: The<br>sycl_ext_oneapi_work_group_scratch_memory feature<br>is not yet available for use with SYCL Graph<br>extension. | LuFinch | LuFinch | <ul><li>No action — investigate further</li></ul> | External oneAPI compiler dependency — wait for oneAPI 2026.0 / CMPLRLLVM-72057 to lift the work_gro…<br>[→ details](details/3142.md) | P2 | Issue is blocked on external oneAPI 26.0 / CMPLRLLVM-72057; no in-tree fix PR exists. Affected cases stay skipped until the compiler depend… | daisyden | dependency component: oneAPI, module: u… |
| [#2554](https://github.com/intel/torch-xpu-ops/issues/2554) | triton | [upstream_ut] AssertionError: AssertionError not<br>raised | daisyden | daisyden | <ul><li>Track PR pytorch/pytorch#181822 to merge</li></ul> | Wait for / pull in the fix from intel-xpu-backend-for-triton#5654 (TTGIR pass), then unskip the thr…<br>[→ details](details/2554.md) | P2 | PR pytorch/pytorch#181822 is the live OPEN fix re-enabling 2 of the 3 cases (the third case test_selecsls42b_misaligned_address is not list… | daisyden | module: inductor, skipped |
| [#2888](https://github.com/intel/torch-xpu-ops/issues/2888) | triton | torch._inductor.exc.InductorError: AssertionError:<br>Conversions between float8_e5m2 and float8_e4m3fn<br>is not supported! | Stonepia | Stonepia | <ul><li>No action — investigate further</li></ul> | Either (a) upstream patch in `torch/_inductor/codegen/triton.py` to relax the guard for XPU and emi…<br>[→ details](details/2888.md) | P2 | No fix PR yet. Assignee Stonepia owns a fix targeted before June; needs upstream relaxation of the fp8_e5m2 <-> fp8_e4m3fn assert in torch/… | daisyden | module: inductor, ut_upstream |
| [#3006](https://github.com/intel/torch-xpu-ops/issues/3006) | triton | AssertionError: '.to(tl.float16)' unexpectedly<br>found in '# AOT ID | CuiYifeng | CuiYifeng | <ul><li>No action — investigate further</li></ul> | Upstream Inductor fix in torch/_inductor/codegen/triton.py (max_with_index/argmax lowering) to ensu…<br>[→ details](details/3006.md) | P2 | No VERIFIED fix PR exists. Root cause is in torch/_inductor/codegen/triton.py (max_with_index/argmax index-dtype promotion) — assignee CuiY… | daisyden | module: inductor, ut_upstream |
| [#3165](https://github.com/intel/torch-xpu-ops/issues/3165) | triton | test_sparse_csr_xpu.py::TestSparseCompressedTriton<br>KernelsXPU::test_triton_bsr_softmax meet<br>RuntimeError: ZE_RESULT_ERROR_INVALID_KERNEL_NAME | jafraustro | jafraustro | <ul><li>No action — investigate further</li></ul> | File a Triton-XPU backend bug with the failing BSR softmax kernel and the ZE_RESULT_ERROR_INVALID_K…<br>[→ details](details/3165.md) | P2 | Zero VERIFIED PR candidates with relationship fixes/supersedes; assignee jafraustro just localized the failing tiling-loop path on 2026-04-… | CuiYifeng | skipped, ut_upstream |
| [#2800](https://github.com/intel/torch-xpu-ops/issues/2800) | oneAPI | AttributeError: 'torch._C._XpuDeviceProperties'<br>object has no attribute 'major' | guangyey | guangyey | <ul><li>No action — investigate further</li></ul> | Wait for oneAPI 2026.1 (CMPLRLLVM-72166) to expose arch major/minor via SYCL device info, then add…<br>[→ details](details/2800.md) | P3 | Issue is OPEN with no fix PR; resolution is gated on oneAPI 2026.1 / CMPLRLLVM-72166 exposing arch major/minor through SYCL device info. Sh… | daisyden | dependency component: oneAPI, module: i… |
| [#1893](https://github.com/intel/torch-xpu-ops/issues/1893) | oneDNN | [upstream_ut] oneDNN accuracy issues in<br>test_ops_xpu.py | chunhuanMeng | chunhuanMeng | <ul><li>Track PR pytorch/pytorch#179125 to merge</li><li>Address CI failures on PR pytorch/pytorch#179125 (>1 week)</li></ul> | Bump tolerance for these mv/addmv ops in test_ops xpu skip/tolerance lists (third_party/torch-xpu-o…<br>[→ details](details/1893.md) | P3 | PR pytorch/pytorch#179125 is the fixes PR addressing the addmv stride/accuracy gap and remains OPEN. \| linux-jammy-py3.10-clang18-asan tes… | daisyden | skipped, ut_upstream |
| [#2439](https://github.com/intel/torch-xpu-ops/issues/2439) | oneDNN | [oneDNN]<br>TestDecompXPU.test_quick_addmv_xpu_float64 got<br>fail accuracy result | libohao1201 | libohao1201 | <ul><li>PR pytorch/pytorch#174590 closed unmerged; reassess fix path</li><li>RETRIAGE_PRS</li></ul> | Pick up upstream PR pytorch/pytorch#174590 once merged (raises decomp tolerance for addmv)<br>[→ details](details/2439.md) | P3 | The only VERIFIED PR (pytorch/pytorch#174590, '[xpu] Add proper float64 handling for addmv, addmm and baddbmm.') was approved but closed un… | mengfei25 | dependency component: oneDNN, module: ut |
| [#2329](https://github.com/intel/torch-xpu-ops/issues/2329) | triton | [upstream_ut] feature missing: get_device_tflops<br>and get_drams_gbps | etaf | etaf | <ul><li>RETRIAGE_PRS</li></ul> | Implement XPU branches in torch/_inductor/utils.py:get_device_tflops/get_dram_gbps using values fro…<br>[→ details](details/2329.md) | P3 | Highest-priority VERIFIED fixes PR (pytorch/pytorch#171291 by assignee @etaf) is CLOSED unmerged on 2026-04-15 and no replacement was found… | daisyden | duplicate, dependency component: Triton… |


<a id="sec-6-2-upstream-pytorch"></a>
- **upstream-pytorch**

_[↑ Back to Index](#sec-2)_

Issues whose fix lives in `pytorch/pytorch` (Dynamo/Inductor, AOTAutograd, `_prims_common`, benchmark harness, test-list sync, etc.). Terminal-QA rows excluded.  —  19 issues.

| Issue | Dependency | Title | Owner | Owner Transferred | action_TBD | Fix Approach | Priority | action_reason | Reporter | Labels |
|---|---|---|---|---|---|---|---|---|---|---|
| [#2015](https://github.com/intel/torch-xpu-ops/issues/2015) | upstream-pytorch | inf is returned by nn.TransformerEncoderLayer | yucai-intel | yucai-intel | <ul><li>check_case_avaliablity</li><li>Track PR intel/torch-xpu-ops#2336 to merge</li><li>Resolve unresolved review comments on PR intel/torch-xpu-ops#2336 (>1 week)</li><li>Address comment AR from tye1: get PR #2336 merged</li></ul> | Land/track pytorch/pytorch#168234 to add xpu device coverage in test_nn.py TransformerEncoderLayer…<br>[→ details](details/2015.md) | P1 | PR #2336 (OPEN) explicitly fixes this issue's TransformerEncoderLayer XPU test failure; needs to land. \| CuiYifeng's CHANGES_REQUESTED rev… | daisyden | skipped |
| [#1969](https://github.com/intel/torch-xpu-ops/issues/1969) | upstream-pytorch | torch._dynamo.exc.InternalTorchDynamoError:<br>TypeError: cannot create weak reference to<br>'torch.Event' object | guangyey | guangyey | <ul><li>No action — investigate further</li></ul> | Upstream-pytorch fix: add weakref support to the generic torch.Event class (define __weakref__ slot…<br>[→ details](details/1969.md) | P2 | All candidate upstream PRs (pytorch/pytorch#164522, #163168, #151213) addressing torch.Event weakref support are CLOSED unmerged with no me… | shangerxin | module: ut |
| [#2024](https://github.com/intel/torch-xpu-ops/issues/2024) | upstream-pytorch | AssertionError: Torch not compiled with CUDA<br>enabled | daisyden | daisyden | <ul><li>Track PR intel/torch-xpu-ops#3510 to merge</li></ul> | Submit upstream PRs replacing hardcoded device='cuda' with the parameterized device argument (or @o…<br>[→ details](details/2024.md) | P2 | PR #3510 (OPEN, APPROVED, CI pending) explicitly enables 5 of the 7 cases listed in this issue and is one merge away from resolving the iss… | mengfei25 | module: ut, skipped |
| [#2244](https://github.com/intel/torch-xpu-ops/issues/2244) | upstream-pytorch | test/test_sparse_csr.py::TestSparseCSRXPU::test_bl<br>ock_addmm meet RuntimeError:<br>empty_sparse_compressed expected sparse compressed<br>(non-block) tensor layout but got SparseBsr | jafraustro | jafraustro | <ul><li>Track PR intel/torch-xpu-ops#3476 to merge</li><li>Resolve unresolved review comments on PR intel/torch-xpu-ops#3476</li></ul> | Extend the upstream sparse compressed addmm dispatch (or torch-xpu-ops registration) so XPU SparseB…<br>[→ details](details/2244.md) | P2 | PR 3476 is the OPEN supersede of PR 2974 (closed unmerged) and addresses the residual bf16/fp16 sparse BSR addmm precision failures of #224… | wincent8 | module: ut, skipped |
| [#2412](https://github.com/intel/torch-xpu-ops/issues/2412) | upstream-pytorch | Some NestedTensor missing XPU support | yucai-intel, BBBela | BBBela | <ul><li>Track PR intel/torch-xpu-ops#2483 to merge</li><li>Address CI failures on PR intel/torch-xpu-ops#2483 (>1 week)</li><li>Resolve unresolved review comments on PR intel/torch-xpu-ops#2483 (>1 week)</li></ul> | Add XPU support to the four nested-tensor files in pytorch/pytorch upstream by relaxing `is_cuda()`…<br>[→ details](details/2412.md) | P2 | PR #2483 is the active OPEN fix; BBBela committed on 2026-04-29 to drive it to merge. \| linux-build check has been failing since 2026-03-3… | daisyden | module: ut |
| [#2609](https://github.com/intel/torch-xpu-ops/issues/2609) | upstream-pytorch | [upstream_ut] torch._inductor.exc.InductorError:<br>CppCompileError: C++ compile error | etaf | etaf | <ul><li>Track PR pytorch/pytorch#171154 to merge</li><li>Address CI failures on PR pytorch/pytorch#171154 (>1 week)</li><li>Address comment AR from etaf: merge the approved fix PR pytorch/pytorch#171154 to land aoti_torch_xpu_fn_<op> shim</li></ul> | Land pytorch/pytorch#171154 which generates/declares aoti_torch_xpu_fn_<op> for XPU custom ops in t…<br>[→ details](details/2609.md) | P2 | VERIFIED fixing PR pytorch/pytorch#171154 is OPEN and APPROVED but not merged. \| PR has 5 failing XPU CI checks whose latest completedAt i… | daisyden | module: inductor, skipped, ut_upstream |
| [#2715](https://github.com/intel/torch-xpu-ops/issues/2715) | upstream-pytorch | [upstream_ut] torch._dynamo.exc.Unsupported:<br>Attempted to inline function marked as skipped | CuiYifeng | CuiYifeng | <ul><li>No action — investigate further</li></ul> | In upstream PyTorch torch/_dynamo/trace_rules.py, allow tracing of torch.xpu.device.__init__ (mirro…<br>[→ details](details/2715.md) | P2 | Issue is OPEN, no VERIFIED PR addresses the dynamo MOD_SKIPLIST exemption for torch.xpu.device.__init__; assignee CuiYifeng needs to invest… | daisyden | skipped, ut_upstream |
| [#2853](https://github.com/intel/torch-xpu-ops/issues/2853) | upstream-pytorch | [upstream_ut]<br>torch.ops.aten._flash_attention_forward lack of<br>support for XPU. | LuFinch | LuFinch | <ul><li>Track PR intel/torch-xpu-ops#3404 to merge</li></ul> | Either (a) register an XPU kernel for _flash_attention_forward in torch-xpu-ops (transformers/sycl/…<br>[→ details](details/2853.md) | P2 | PR #3404 (assignee LuFinch) is the verified fix and is currently OPEN with no review decision yet; needs to be driven through review and me… | BBBela | skipped |
| [#3084](https://github.com/intel/torch-xpu-ops/issues/3084) | upstream-pytorch | torch.library.register_autocast does not support<br>xpu | CuiYifeng | CuiYifeng | <ul><li>No action — investigate further</li></ul> | Upstream PR to pytorch/pytorch: in torch/library.py register_autocast, accept all autocast-capable…<br>[→ details](details/3084.md) | P2 | Issue is OPEN with no linked PRs from any of V0/VA/VB/VC/VD/VE; an upstream pytorch/pytorch fix to torch/library.py register_autocast is re… | daisyden | module: ut |
| [#3187](https://github.com/intel/torch-xpu-ops/issues/3187) | upstream-pytorch | PyTorch XPU gpu_cpp_wrapper fails with<br>InductorError NotImplementedError | CuiYifeng | CuiYifeng | <ul><li>PR pytorch/pytorch#178477 closed unmerged; reassess fix path</li><li>RETRIAGE_PRS</li></ul> | Investigate cpp_wrapper fallback dispatch for XPU in torch/_inductor/codegen/cpp_wrapper_gpu.py and…<br>[→ details](details/3187.md) | P2 | The only referenced PR (the upstream temporary skip workaround) was closed unmerged, and no replacement PR addresses the underlying XPU cpp… | liangan1 | ut_upstream |
| [#2214](https://github.com/intel/torch-xpu-ops/issues/2214) | upstream-pytorch | test/test_sparse.py::TestSparseAnyXPU::test_gradch<br>eck_mm expected error message not match | jenniew | jenniew | <ul><li>No action — investigate further</li><li>Address comment AR from chuanqi129 (>1 week): @wincent8 remove hardcoded skips in test files (use dynamic skip-by-issue)</li><li>Address comment AR from daisyden (>1 week): confirm/dedupe with #2283 (jenniew should triage)</li></ul> | Update the assertRaisesRegex pattern in upstream test_sparse.py to also accept the 'empty_sparse_co…<br>[→ details](details/2214.md) | P3 | Issue is OPEN with zero VERIFIED fixes/supersedes PR candidates — the fix lives in upstream pytorch/pytorch test_sparse.py assertRaisesRege… | wincent8 | skipped, ut_upstream |
| [#2287](https://github.com/intel/torch-xpu-ops/issues/2287) | upstream-pytorch | [upstream_ut] test_python_ref issues | yucai-intel | yucai-intel | <ul><li>Track PR pytorch/pytorch#178734 to merge</li><li>Address CI failures on PR pytorch/pytorch#178734 (>1 week)</li><li>Address comment AR from tye1: pytorch/pytorch#178734 has a lint issue — yucai-intel to fix</li></ul> | Skip in test/xpu/skip_list_common.py for test_python_ref__refs_logspace_tensor_overload (and any re…<br>[→ details](details/2287.md) | P3 | PR 178734 (supersedes #169565) explicitly Fixes #2287 and is APPROVED. \| Latest failing required check (lintrunner-noclang-partial / lint … | daisyden | module: ut, ut_upstream |
| [#2436](https://github.com/intel/torch-xpu-ops/issues/2436) | upstream-pytorch | [upstream_ut] AttributeError: 'NoneType' object<br>has no attribute 'clone' | daisyden | daisyden | <ul><li>No action — investigate further</li><li>Address comment AR from daisyden (>1 week): re-check case design per CuiYifeng's request</li></ul> | Keep cases in skip list as 'random/community'<br>[→ details](details/2436.md) | P3 | Issue is OPEN with zero VERIFIED PR candidates (root cause pytorch/pytorch#97395 is an upstream issue, not a PR); needs further investigati… | daisyden | skipped, dependency component: communit… |
| [#2512](https://github.com/intel/torch-xpu-ops/issues/2512) | upstream-pytorch | [upstream_ut] RuntimeError: _histc_xpu does not<br>have a deterministic implementation, but you set<br>'torch.use_deter | chunhuanMeng | chunhuanMeng | <ul><li>check_case_avaliablity</li><li>Track PR intel/torch-xpu-ops#3333 to merge</li><li>Address CI failures on PR intel/torch-xpu-ops#3333</li></ul> | Align XPU alert message with CUDA: change SummaryOps.cpp:45 to globalContext().alertNotDeterministi…<br>[→ details](details/2512.md) | P3 | VERIFIED PR #3333 (content_match, relationship=fixes) is OPEN and modifies SummaryOps.cpp + the xpu histc test as described in fix_approach… | libohao1201 | skipped |
| [#2531](https://github.com/intel/torch-xpu-ops/issues/2531) | upstream-pytorch | [upstream_ut] AssertionError: Torch not compiled<br>with CUDA enabled | guangyey | guangyey | <ul><li>check_case_avaliablity</li><li>Track PR intel/torch-xpu-ops#3510 to merge</li><li>Address comment AR from guangyey: confirm XPU support for tunable / cufft_plan_cache / CudaSyncGuard / Miopen / quantize_per_tensor</li></ul> | Port each test in third_party/torch-xpu-ops/test/xpu/ to substitute xpu for cuda (use TEST_XPU/torc…<br>[→ details](details/2531.md) | P3 | VERIFIED partial-fix PR #3510 (relationship=fixes, OPEN, approved by chuanqi129) lands 2 of 13 cases; remaining cases need feature work tra… | daisyden | skipped, port_from_skiplist |
| [#2783](https://github.com/intel/torch-xpu-ops/issues/2783) | upstream-pytorch | [Bug Skip]: Key "xpu" is missing from dict<br>"driver" in test_svd | daisyden | daisyden | <ul><li>Open upstream PR to pytorch/pytorch from branch daisyden/missing_test (adds 'xpu' entry to SVD drivers dict in test/test_linalg.py)</li><li>No action — investigate further</li></ul> | Upstream patch in pytorch/test/test_linalg.py: add an 'xpu' entry to the SVD driver dict (or extend…<br>[→ details](details/2783.md) | P3 | Assignee daisyden has a verified local fix on a personal branch but no upstream PR has been filed yet; landing that PR (or equivalent) is t… | CuiYifeng | module: ut, skipped |
| [#2891](https://github.com/intel/torch-xpu-ops/issues/2891) | upstream-pytorch | RuntimeError: Expected to find "(262144, 0, 512,<br>1" but did not find it | chunhuanMeng | chunhuanMeng | <ul><li>Track PR pytorch/pytorch#180418 to merge</li></ul> | Confirm upstream pytorch PR #180418 (unskip) is landed and pulled into torch-xpu-ops test list<br>[→ details](details/2891.md) | P3 | PR pytorch/pytorch#180418 (unskip test_effn_attn_bias_padding) is OPEN, APPROVED, and CI green; only Gate 4 (merge) remains. Once merged, t… | daisyden | module: inductor, ut_upstream |
| [#2958](https://github.com/intel/torch-xpu-ops/issues/2958) | upstream-pytorch | AssertionError of test_dtensor_basic_compile | daisyden | daisyden | <ul><li>Submit upstream PR from daisyden/missing_test to remove skipIfXpu on test_dtensor_basic_export and close this issue</li><li>File separate upstream issue for test-ordering / DTensorSpec flatten regression from pytorch/pytorch#178115</li></ul> | Re-run on current main to confirm<br>[→ details](details/2958.md) | P3 | Owner has already verified the fix on main and prepared a branch; only remaining step is to land the skip-removal PR. \| The ordering-depen… | daisyden | module: inductor, ut_upstream |
| [#3041](https://github.com/intel/torch-xpu-ops/issues/3041) | upstream-pytorch | AssertionError: Expected len(flat_diff_results) ><br>0 in test_fake_crossref_backward_amp_normal_number<br>_mean_xpu_float32 | daisyden | daisyden | <ul><li>No action — investigate further</li></ul> | Add device_type='xpu' (or replace with allowed_dtypes/skip across non-CPU) to the existing Decorate…<br>[→ details](details/3041.md) | P3 | Issue is OPEN with no VERIFIED fix PR yet (cited PR 176690 is the trigger, not the fix, and is closed unmerged). Owner daisyden is assigned… | daisyden | ut_upstream |


<a id="sec-6-3-cpu-fallback"></a>
- **CPU fallback**

_[↑ Back to Index](#sec-2)_

Issues where the XPU operator is missing and a CPU fallback is registered in torch-xpu-ops. Terminal-QA rows excluded.  —  3 issues.

| Issue | Dependency | Title | Owner | Owner Transferred | action_TBD | Fix Approach | Priority | action_reason | Reporter | Labels |
|---|---|---|---|---|---|---|---|---|---|---|
| [#2283](https://github.com/intel/torch-xpu-ops/issues/2283) | CPU fallback | [upstream_ut] sparse._sampled_addmm is not<br>supported | jenniew | jenniew | <ul><li>Track PR intel/torch-xpu-ops#3018 to merge</li><li>Address CI failures on PR intel/torch-xpu-ops#3018 (>1 week)</li></ul> | Either register a CPU fallback for aten::sparse_sampled_addmm on SparseCsrXPU (analogous to other s…<br>[→ details](details/2283.md) | P1 | PR 3018 implements the missing SparseCsrXPU sampled_addmm registration; needs to land. \| linux-ut (op_ut) summary failed on 2026-03-26 (>1… | daisyden | skipped, ut_upstream |
| [#2442](https://github.com/intel/torch-xpu-ops/issues/2442) | CPU fallback | [Bug Skip]: NotImplementedError: Could not run<br>'aten::_flash_attention_forward' with arguments<br>from the 'CPU' backend | LuFinch | LuFinch | <ul><li>Track PR intel/torch-xpu-ops#3404 to merge</li><li>Address CI failures on PR intel/torch-xpu-ops#3404</li></ul> | Add the four cases to the skip list (already labelled 'skipped')<br>[→ details](details/2442.md) | P2 | VERIFIED PR #3404 (github_linked, relationship=fixes) is OPEN and adds the XPU _flash_attention_forward/_backward registration described in… | CuiYifeng | skipped |
| [#2229](https://github.com/intel/torch-xpu-ops/issues/2229) | CPU fallback | test/test_sparse_csr.py::TestSparseCompressedCPU::<br>test_invalid_input meet message not match | jenniew | jenniew | <ul><li>Track PR intel/torch-xpu-ops#3073 to merge</li><li>Address CI failures on PR intel/torch-xpu-ops#3073 (>1 week)</li></ul> | Implement aten::_validate_compressed_sparse_indices for the XPU backend in torch-xpu-ops so the can…<br>[→ details](details/2229.md) | P3 | PR 3073 is the OPEN fix authored by assignee jenniew implementing the _validate_compressed_sparse_indices XPU kernel. \| preci-lint-check h… | wincent8 | skipped |


<a id="sec-7"></a>
## 7. New submitted issues (<7 days)

_[↑ Back to Index](#sec-2)_

Issues created on or after 2026-04-23, excluding terminal-QA rows.  —  0 issues.

| Issue | Created | Title | Owner | Owner Transferred | action_TBD | Fix Approach | Priority | action_reason | Reporter | Labels |
|---|---|---|---|---|---|---|---|---|---|---|


<a id="sec-8"></a>
## 8. Requests pending > 1 week

_[↑ Back to Index](#sec-2)_

Issues whose `action_TBD` contains one or more verbs flagged `(>1 week)` — an unresolved comment AR, unresolved PR review comments, or unaddressed CI failures that have been sitting more than 7 days. These are the highest-priority candidates for owner follow-up.

| Issue | Title | Owner | Stale Requests | Priority | Reporter | Labels |
|---|---|---|---|---|---|---|
| [#2015](https://github.com/intel/torch-xpu-ops/issues/2015) | inf is returned by nn.TransformerEncoderLayer | yucai-intel | <ul><li>Resolve unresolved review comments on PR intel/torch-xpu-ops#2336 (>1 week)</li></ul> | P1 | daisyden | skipped |
| [#2283](https://github.com/intel/torch-xpu-ops/issues/2283) | [upstream_ut] sparse._sampled_addmm is not<br>supported | jenniew | <ul><li>Address CI failures on PR intel/torch-xpu-ops#3018 (>1 week)</li></ul> | P1 | daisyden | skipped, ut_upstream |
| [#2769](https://github.com/intel/torch-xpu-ops/issues/2769) | [oneDNN] New failed test cases with 3.11 compared<br>with 3.10 | mengfei25 | <ul><li>Address comment AR from chuanqi129 (>1 week): verify whether oneDNN 3.11.1 fixes the conv regression</li></ul> | P1 | mengfei25 | hw: PVC, dependency component: oneDNN, … |
| [#2412](https://github.com/intel/torch-xpu-ops/issues/2412) | Some NestedTensor missing XPU support | yucai-intel, BBBela | <ul><li>Address CI failures on PR intel/torch-xpu-ops#2483 (>1 week)</li><li>Resolve unresolved review comments on PR intel/torch-xpu-ops#2483 (>1 week)</li></ul> | P2 | daisyden | module: ut |
| [#2532](https://github.com/intel/torch-xpu-ops/issues/2532) | Title: [upstream_ut] AssertionError: wrong number<br>of dimensions2 for op:<br>torch.ops.aten._convert_weight_to_int4pack.defa | yucai-intel | <ul><li>Resolve unresolved review comments on PR intel/torch-xpu-ops#3090 (>1 week)</li><li>Address CI failures on PR intel/torch-xpu-ops#3090 (>1 week)</li></ul> | P2 | daisyden | skipped, port_from_skiplist |
| [#2609](https://github.com/intel/torch-xpu-ops/issues/2609) | [upstream_ut] torch._inductor.exc.InductorError:<br>CppCompileError: C++ compile error | etaf | <ul><li>Address CI failures on PR pytorch/pytorch#171154 (>1 week)</li></ul> | P2 | daisyden | module: inductor, skipped, ut_upstream |
| [#2613](https://github.com/intel/torch-xpu-ops/issues/2613) | [upstream_ut] AssertionError: Tensor-likes are not<br>equal! in test_compile_subprocess.py | daisyden | <ul><li>Address comment AR from daisyden (>1 week): re-verify argmax/argmin tie-break tests with the new driver per GSD-11415</li></ul> | P2 | daisyden | dependency component: driver, module: i… |
| [#2698](https://github.com/intel/torch-xpu-ops/issues/2698) | Title: [upstream_ut] RuntimeError:<br>FlashAttentionForwardXPU only support headdim<br>64,96,128,192 | LuFinch | <ul><li>Address CI failures on PR pytorch/pytorch#180646 (>1 week)</li></ul> | P2 | daisyden | module: inductor, skipped, ut_upstream |
| [#3177](https://github.com/intel/torch-xpu-ops/issues/3177) | Accuracy gap of BF16/FP16 test_block_addmm | jenniew | <ul><li>Resolve unresolved review comments on PR intel/torch-xpu-ops#3273 (>1 week)</li></ul> | P2 | CuiYifeng | skipped |
| [#1893](https://github.com/intel/torch-xpu-ops/issues/1893) | [upstream_ut] oneDNN accuracy issues in<br>test_ops_xpu.py | chunhuanMeng | <ul><li>Address CI failures on PR pytorch/pytorch#179125 (>1 week)</li></ul> | P3 | daisyden | skipped, ut_upstream |
| [#2214](https://github.com/intel/torch-xpu-ops/issues/2214) | test/test_sparse.py::TestSparseAnyXPU::test_gradch<br>eck_mm expected error message not match | jenniew | <ul><li>Address comment AR from chuanqi129 (>1 week): @wincent8 remove hardcoded skips in test files (use dynamic skip-by-issue)</li><li>Address comment AR from daisyden (>1 week): confirm/dedupe with #2283 (jenniew should triage)</li></ul> | P3 | wincent8 | skipped, ut_upstream |
| [#2229](https://github.com/intel/torch-xpu-ops/issues/2229) | test/test_sparse_csr.py::TestSparseCompressedCPU::<br>test_invalid_input meet message not match | jenniew | <ul><li>Address CI failures on PR intel/torch-xpu-ops#3073 (>1 week)</li></ul> | P3 | wincent8 | skipped |
| [#2287](https://github.com/intel/torch-xpu-ops/issues/2287) | [upstream_ut] test_python_ref issues | yucai-intel | <ul><li>Address CI failures on PR pytorch/pytorch#178734 (>1 week)</li></ul> | P3 | daisyden | module: ut, ut_upstream |
| [#2436](https://github.com/intel/torch-xpu-ops/issues/2436) | [upstream_ut] AttributeError: 'NoneType' object<br>has no attribute 'clone' | daisyden | <ul><li>Address comment AR from daisyden (>1 week): re-check case design per CuiYifeng's request</li></ul> | P3 | daisyden | skipped, dependency component: communit… |
| [#3170](https://github.com/intel/torch-xpu-ops/issues/3170) | Unskip test_bmm_windows_error_xpu_float64 | libohao1201, jafraustro | <ul><li>Address comment AR from libohao1201 (>1 week): verify the issue on windows</li></ul> | P3 | CuiYifeng | skipped, ut_upstream |


<a id="sec-9"></a>
## 9. Statistics

_[↑ Back to Index](#sec-2)_

- Total rows: **53**
- Classified (non-empty `action_Type`): **53**
- Empty `action_TBD` (no verdict yet): **0**
- Issues flagged for test-case existence check (`CHECK_CASES`): **6**

- **Primary action_Type distribution (exclusive — one bucket per issue)**

_[↑ Back to Index](#sec-2)_

Merged buckets (as rendered in §3 and §4):

| Bucket | Issues |
|---|---:|
| NEED PR | 19 |
| TRACK PR | 21 |
| CLOSE or SKIP | 12 |

Raw atoms (pre-merge, for reference):

| Category | Issues |
|---|---:|
| CLOSE | 2 |
| NOT_TARGET_CLOSE | 6 |
| VERIFY_AND_CLOSE | 4 |
| TRACK_PR | 19 |
| IMPLEMENT | 1 |
| RETRIAGE_PRS | 2 |
| WAIT_EXTERNAL | 1 |
| NEED_ACTION | 18 |

- **action_Type distribution (multi-label — each category counted once per issue)**

_[↑ Back to Index](#sec-2)_

| Category | Issues |
|---|---:|
| CLOSE | 2 |
| NOT_TARGET_CLOSE | 6 |
| VERIFY_AND_CLOSE | 7 |
| TRACK_PR | 24 |
| IMPLEMENT | 1 |
| RETRIAGE_PRS | 16 |
| WAIT_EXTERNAL | 1 |
| FILE_ISSUE | 1 |
| NEED_ACTION | 21 |
| AWAIT_REPLY | 9 |
| CHECK_CASES | 6 |

- **Priority distribution**

_[↑ Back to Index](#sec-2)_

| Priority | Issues |
|---|---:|
| P1 | 8 |
| P2 | 24 |
| P3 | 21 |

- **Status distribution**

_[↑ Back to Index](#sec-2)_

| Status | Issues |
|---|---:|
| open | 53 |

- **Category column distribution (top 20)**

_[↑ Back to Index](#sec-2)_

| Category | Issues |
|---|---:|
| Torch Operations | 15 |
| Inductor | 14 |
| Flash Attention | 10 |
| Sparse | 8 |
| Others | 4 |
| TorchAO | 2 |

- **CHECK_CASES issue IDs**

_[↑ Back to Index](#sec-2)_

6 issues flagged for `check_case_avaliablity` (missing XPU test case in repo):

> #2015, #2285, #2376, #2512, #2531, #2578
