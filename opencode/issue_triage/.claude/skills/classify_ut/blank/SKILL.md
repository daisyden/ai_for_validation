# classify_ut blank status_xpu cases

This skill follows agent-guidelines AND extends it with blank-XPU-status UT classification rules.
Always apply agent-guidelines rules including the mandatory post-write commit protocol.

## Purpose

Classify rows in the `Non-Inductor XPU Skip` workbook whose `Reason` is blank and whose
`status_xpu` is blank. Blank XPU status means the case has no direct XPU result in the workbook;
classification must come from deep case-existence analysis, source inspection, skip-list evidence,
and targeted local runs when needed.

## Required Inputs

- Workbook row fields: `testfile_cuda`, `classname_cuda`, `name_cuda`, `testfile_xpu`,
  `classname_xpu`, `name_xpu`, `status_xpu`, `Reason`, `DetailReason`, `Exaplaination`.
- Local PyTorch checkout: `/home/daisyden/opencode/classify/pytorch`.
- XPU test checkout: `/home/daisyden/opencode/classify/pytorch/third_party/torch-xpu-ops/test/xpu`.
- Conda environment: `pytorch_opencode_env`.
- Deep case-existence workflow:
  `/home/daisyden/opencode/ai_for_validation/opencode/issue_triage/.claude/skills/bug_scrub/analyze_ci_result/check_xpu_case_existence/SKILL.md`.

## Required Tools

- `read` - inspect base tests, XPU wrapper/direct files, distributed runner files, skip lists,
  parametrization, decorators, and helper source.
- `bash` - activate `pytorch_opencode_env`, run targeted tests or `pytest --collect-only`, inspect
  git refs, and fetch remote release files with `git show` or `gh`.
- `grep` / `rg` - locate exact class/function/decorator names after semantic analysis identifies the
  files that must be read. Do not classify from grep output alone.
- `gh` CLI - inspect GitHub files/issues when local source or `message_xpu` references them.
- Workbook tooling such as `openpyxl` may write results, but classification must come from source
  and execution evidence.

## Hard Constraints

- Do not classify by filename patterns, keyword matches, or a bulk script alone.
- Blank `status_xpu` is not proof that the case is missing. It means the row needs deep analysis.
- Always inspect the actual source that determines whether the XPU case exists, is generated, is
  skipped, or needs enablement.
- Do not use `release/2.12` for non-distributed rows.
- For distributed rows only, use release/2.12 and the remote distributed skip-list evidence described
  below.
- CUDA graph / cudagraph rows are not `Not Appliable` merely because the CUDA name contains `cuda`.
  XPU graph support exists via `_XPUGraph`, `torch.xpu.XPUGraph`, and `torch.accelerator.Graph`;
  missing/failing coverage is `To be enabled` with an XPU graph DetailReason.
- `Not Appliable` for CUDA-specific APIs must name the exact API in `DetailReason`, such as
  `CUDA-specific API: torch.cuda.jiterator` or `CUDA-specific API: cuBLAS`.
- `Not applicable / Community Changes` is only for tests removed, renamed, or no longer present in
  the source being compared. If an XPU variant exists after parametrization, do not call it community
  changes.
- Do not change `Reason TBD` after classification. Mark updated `Reason`, `DetailReason`, and
  `Exaplaination` cells blue.

## Workflow

1. Confirm the row is eligible:
   - `Reason` is blank.
   - `status_xpu` is blank.
   - CUDA metadata identifies the exact test file, class, and method.
2. Derive missing XPU metadata only as a starting point:
   - `classname_cuda` ending in `CUDA` -> `XPU`.
   - `name_cuda` ending in `_cuda` -> `_xpu`.
   - `testfile_xpu` defaults to `testfile_cuda` when blank.
   Then verify against actual XPU source; do not trust the derived names blindly.
3. Determine whether the row is distributed:
   - If `testfile_cuda` is under `test/distributed/`, follow the distributed workflow below.
   - Otherwise follow the non-distributed workflow below.
4. Write one of the canonical outcomes:
   - `Reason = To be enabled` for missing XPU registration, existing-but-unreported XPU cases,
     explicit XPU skips needing enablement, or missing XPU coverage for supported functionality.
   - `Reason = Not Appliable` for CUDA-only APIs or backend-specific features that cannot apply to
     XPU. `DetailReason` must name the exact API/feature.
   - `Reason = Not applicable`, `DetailReason = Community Changes` only when source comparison proves
     the CUDA test was removed/renamed or no longer exists.

## Distributed Blank-Status Workflow

Distributed XPU tests usually do not use `*_xpu.py` wrappers. They run upstream files through
`third_party/torch-xpu-ops/test/xpu/run_distributed.py` and distributed skip dictionaries.

1. Read `third_party/torch-xpu-ops/test/xpu/run_distributed.py` to confirm active imports.
2. For release/2.12 distributed classification, first read the remote local distributed skip list:
   `intel/torch-xpu-ops` branch `daisyden/distributed_2.12`, file
   `test/xpu/skip_list_dist_local.py`. The intended name may be described as
   `skip_list_local_dist.py`, but the verified branch filename is `skip_list_dist_local.py`.
3. Read the matching distributed skip list:
   `intel/torch-xpu-ops` branch `daisyden/distributed_2.12:test/xpu/skip_list_dist.py` when
   classifying against release/2.12, or local `third_party/torch-xpu-ops/test/xpu/skip_list_dist.py`
   for current-checkout analysis.
4. If a local override such as `skip_list_dict_local.py` exists, read it in full.
5. Normalize skip-list keys semantically:
   - `../../../../test/distributed/...` points to the upstream PyTorch distributed test.
   - `distributed/test_c10d_xccl.py` points to an XPU-native standalone distributed test.
6. Interpret entries:
   - File present with value `None` -> the whole file is enabled for XPU.
   - File present with tuple/list -> the file is enabled, but listed cases are intentionally skipped;
     all other cases run.
   - File absent -> the upstream file is not run by `run_distributed.py`; classify `To be enabled`
     with a specific DetailReason such as `Distributed file missing from remote distributed skip list`.
7. Also inspect `third_party/torch-xpu-ops/test/xpu/distributed/` for XPU-native files such as
   `test_c10d_xccl.py` and `test_c10d_ops_xccl.py`.
8. For distributed rows only, check `https://github.com/daisyden/pytorch/tree/release/2.12` or local
   `origin/release/2.12` for the upstream file and method before deciding a test was removed,
   renamed, or newly added.

## Non-Distributed Blank-Status Workflow

1. Read the local base test under `test/` and confirm the CUDA class/method still exists.
2. Enumerate all plausible XPU locations before declaring a case missing:
   - `third_party/torch-xpu-ops/test/xpu/`
   - `third_party/torch-xpu-ops/test/xpu/extended/`
   - `third_party/torch-xpu-ops/test/xpu/nn/`
   - `third_party/torch-xpu-ops/test/xpu/functorch/`
   - `third_party/torch-xpu-ops/test/xpu/quantization/`
   - other relevant subfolders discovered from imports or local naming.
3. Read wrapper/direct XPU files. Not all XPU tests use `XPUPatchForImport`; many are standalone
   copies or direct implementations.
4. If a wrapper uses `XPUPatchForImport`, understand its mode:
   - `XPUPatchForImport(False)` usually imports/instantiates upstream tests with XPU adaptations.
   - `XPUPatchForImport(True)` can disable or alter instantiation; inspect the surrounding code.
5. Inspect class definitions, imports, `instantiate_device_type_tests`,
   `instantiate_parametrized_tests`, OpInfo filters, decorators, and xfail/skip lists to determine
   whether the exact XPU case is generated.
6. Use targeted collection or local execution when source inspection is inconclusive. Run only the
   exact test or a narrow collect command in `pytorch_opencode_env`.
7. If no XPU source generates the case but the base test exists and the feature applies to XPU,
   classify `To be enabled` with a DetailReason naming the missing file/class/import/instantiation.
8. If local source proves the CUDA test no longer exists or has been renamed, classify
   `Not applicable / Community Changes` with the exact source evidence.
9. If source proves the test is CUDA-only, classify `Not Appliable` and name the exact API.

## Known Blank-Status Classifications From This Workflow

These are examples, not substitutes for analysis. Re-check source before applying.

- Remote distributed file enabled:
  - Reason: `To be enabled`.
  - DetailReason: `Distributed file enabled in remote distributed skip list: <file>`.
  - Explanation should name the remote skip-list files read and say that the file is registered for
    XPU through `run_distributed.py`.
- Remote distributed file missing:
  - Reason: `To be enabled`.
  - DetailReason: `Distributed file missing from remote distributed skip list: <file>`.
  - Explanation should name checked dictionaries, release/2.12 file presence, and enabled sibling
    files when useful.
- CUDA graph / cudagraph coverage:
  - Reason: `To be enabled`.
  - DetailReason: `XPU graph coverage missing` or a similarly specific XPU graph gap.
  - Explanation should mention that XPU graph APIs exist and identify the missing XPU test coverage.
- Jiterator blank-status rows:
  - Reason: `Not Appliable`.
  - DetailReason: `CUDA-specific API: torch.cuda.jiterator`.
  - Explanation should mention the concrete `torch.cuda.jiterator` APIs used.
- cuBLAS deterministic blank-status rows:
  - Reason: `Not Appliable`.
  - DetailReason: `CUDA-specific API: cuBLAS`.
  - Explanation should mention the cuBLAS determinism behavior and any `@onlyCUDA` evidence.
- TensorExpr CUDA fuser rows:
  - Reason: `Not Appliable`.
  - DetailReason: `CUDA-specific API: TensorExpr CUDA fuser`.
- Existing XPU wrapper/direct file with generated XPU test but no XPU workbook result:
  - Reason: `To be enabled`.
  - DetailReason: `Test exists but blank: <XPU class or file>`.
  - Explanation should name the exact XPU source file/class/function and expected XPU test name.
- Local base test missing or method removed for a non-distributed row:
  - Reason: `Not applicable`.
  - DetailReason: `Community Changes`.
  - Explanation should name the local source evidence and state that non-distributed release/2.12 was
    not used.

## Output Rules

- `Reason`: use canonical workbook labels: `To be enabled`, `Not Appliable`, or `Not applicable`.
- `DetailReason`: be specific enough to act on. Avoid generic `No XPU wrapper` and generic
  `CUDA-specific API`.
- `Exaplaination`: keep the workbook spelling. Include exact test identity, source files read,
  skip-list or wrapper evidence, and why the selected Reason follows.

## Verification

- Re-open the output workbook with `openpyxl`.
- Confirm no eligible blank `Reason` rows remain for processed blank-status rows.
- Confirm `Reason TBD` values were not flipped after classification.
- Confirm updated cells are blue.
- Spot-check at least one distributed enabled row, one distributed missing row, one CUDA-specific API
  row, one XPU-wrapper-existing row, and one community-change row when those categories are present.
