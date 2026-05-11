# classify_ut failed cases

This skill follows agent-guidelines AND extends it with failed-XPU-status UT classification rules.
Always apply agent-guidelines rules including the mandatory post-write commit protocol.

## Purpose

Classify rows in any XPU UT status workbook whose `Reason` is blank and whose
`status_xpu` is `failed`. This skill is a sub-workflow of `classify_ut`; use it after the
base workbook preparation has initialized `Reason TBD`, filled missing XPU metadata, and
preserved the original workbook.

## Required Inputs

- Workbook row fields: `testfile_cuda`, `classname_cuda`, `name_cuda`, `testfile_xpu`,
  `classname_xpu`, `name_xpu`, `status_xpu`, `message_xpu`, `Reason`, `DetailReason`,
  `Explaination`, `Reason TBD`.
- Local PyTorch checkout: `/home/daisyden/opencode/classify/pytorch`.
- XPU test checkout: `/home/daisyden/opencode/classify/pytorch/third_party/torch-xpu-ops/test/xpu`.
- Conda environment: `pytorch_opencode_env`.
- GitHub issue source of truth: `intel/torch-xpu-ops` issues first; use PyTorch issues only
  when the message or local source explicitly references a PyTorch issue.

## Required Tools

- `read` - inspect base tests, XPU test files, wrappers, skip decorators, and supporting source.
- `bash` - activate `pytorch_opencode_env`, run targeted tests, query `gh issue view`, and inspect
  git/remote files.
- `grep` / `rg` - locate exact methods, decorators, issue links, and error strings after deciding
  what must be inspected. Do not use grep hits alone as classification evidence.
- `gh` CLI - fetch issue state and body, especially for `intel/torch-xpu-ops` known issues.
- Workbook tooling such as `openpyxl` may prepare or write cells, but it must not replace
  semantic analysis.

## Hard Constraints

- Do not classify by filename, keyword, or message pattern alone.
- Do not assume every failure is an XPU bug without understanding the error and test surface.
- Always read enough code to understand what the test is validating.
- Always inspect `message_xpu`; if it is missing, empty, truncated beyond usefulness, or only says
  that a process exited, run the exact test locally to get a useful error whenever feasible.
- Local runs must use `pytorch_opencode_env`.
- Avoid importing the unbuilt source-tree `torch` accidentally. When checking the installed torch
  package directly, run from outside the PyTorch checkout, for example `/tmp/opencode`. When running
  tests, use the correct test root described below.
- If a matching known issue exists, `DetailReason` must include the issue URL.
- If no known issue exists after searching `intel/torch-xpu-ops`, `DetailReason` must start with
  `[Issue TBD]` and include the concrete error summary.
- Do not change `Reason TBD` after classification. It records whether the original `Reason` was
  blank before analysis.
- Mark updated `Reason`, `DetailReason`, and `Exaplaination` cells blue; leave unrelated cells alone.

## Local Run Rules

Run only targeted tests. Never run a whole suite.

- Inductor or Dynamo tests whose source is in `pytorch/test/` should be run from the PyTorch test
  folder:
  ```bash
  source ~/miniforge3/bin/activate pytorch_opencode_env && \
  python dynamo/test_dicts.py DictSubclassMethodsTests.test_binop_or
  ```
- XPU wrapper/direct tests should be run from `third_party/torch-xpu-ops/test/xpu`:
  ```bash
  source ~/miniforge3/bin/activate pytorch_opencode_env && \
  python test_ops_xpu.py TestCommonXPU.test_dtypes_addmm_xpu
  ```
- Distributed upstream tests generally run from `pytorch/test/` with the exact upstream file/class
  and method when feasible. If the XPU workflow requires `run_distributed.py`, first read the active
  distributed skip-list workflow from the main `classify_ut` skill.
- If the local run itself is skipped, read the decorator/helper source that caused the skip before
  deciding whether it is a feature gap, stale skip, failure, or environment limitation.

## Workflow

1. Confirm the row is eligible:
   - `Reason` is blank.
   - `status_xpu` is exactly `failed`.
   - CUDA and XPU metadata identify the exact test case.
2. Normalize `message_xpu` for readability only. Do not classify solely from text matching.
   Preserve enough of the original failure in `Exaplaination`.
3. Inspect the relevant local source:
   - Base test under `test/`.
   - XPU wrapper/direct file under `third_party/torch-xpu-ops/test/xpu/**` if present.
   - Distributed skip dictionaries and remote release branch only for distributed tests, following
     the parent `classify_ut` rules.
4. Search `intel/torch-xpu-ops` issues semantically for the concrete failing behavior. Use the
   method name, operator name, exception type, and meaningful error phrase, not just the file name.
5. If `message_xpu` is missing or too generic, run the exact test locally and use that result as
   evidence. If the local run passes, classify as `To be enabled`; if it fails, use the local error
   and continue issue search.
6. Decide the classification:
    - Known XPU implementation/runtime failure -> `Reason = Failures (xpu broken)`.
    - Known missing feature/API exposed by the failed run -> `Reason = Feature gap` only when the
      issue/source describes unsupported functionality rather than a broken implementation.
    - Local run passes and existing failure appears stale -> `Reason = Local Passed` (with evidence
      saved to local verify dir).
    - No known issue after search -> `Reason = Failures (xpu broken)` and `DetailReason` starts with
      `[Issue TBD]`.
7. Write concise evidence:
   - `DetailReason` includes issue link or `[Issue TBD]` plus the error summary.
   - `Exaplaination` names the exact test, states that `status_xpu` was `failed`, summarizes
     `message_xpu` or local-run output, and names the source/issue inspected.

## Known Failed-Case Classifications From This Workflow

These are examples, not a substitute for analysis. Re-check source and issue state before reusing.

- Jiterator failures:
  - Evidence: `message_xpu` says `Jiterator is only supported on CUDA and ROCm GPUs`.
  - Known issue: `https://github.com/intel/torch-xpu-ops/issues/2918`.
  - Reason: `Failures (xpu broken)` for failed-status rows in this workbook; the detail names that
    jiterator is not supported on XPU.
- OpInfo dtype mismatch failures:
  - Evidence: `message_xpu` says supported dtypes for an op on XPU are incorrect or dtypes worked
    but are not listed by OpInfo.
  - Known issue: `https://github.com/intel/torch-xpu-ops/issues/3574` when the issue content matches
    the op family; otherwise search more and use `[Issue TBD]` if no issue exists.
  - Reason: `Failures (xpu broken)` unless source/issue says the dtype is an intentionally missing
    feature.
- XPU compiler / `ocloc` failures:
  - Evidence: `message_xpu` contains `ocloc`, IGC initialization failure, or a Triton-to-ZEBIN
    compiler failure.
  - Known issue: `https://github.com/intel/torch-xpu-ops/issues/3386` when the failure matches.
  - Reason: `Failures (xpu broken)`.
- cuBLAS/matmul deterministic failures:
  - Evidence: test name contains `test_cublas_deterministic` and issue/source confirms matmul
    deterministic behavior is tracked for XPU.
  - Known issue: `https://github.com/intel/torch-xpu-ops/issues/2481` when the case matches.
  - Reason: `Failures (xpu broken)` for failed rows, or `Feature gap` for skipped CUDA-only rows
    handled by the skipped-case skill.

## Output Rules

- `Reason`: choose one of the workbook's canonical labels, especially `Failures (xpu broken)`,
  `Feature gap`, or `To be enabled`.
- `DetailReason`: include the issue URL when known. Otherwise start with `[Issue TBD]`.
- `Explaination`: use the workbook's requested spelling. Include exact test identity, error source,
  local-run result if used, and why the chosen Reason follows.

## Local Passed for Failed-Status Rows

When `status_xpu = failed` but the test PASSES locally in `pytorch_opencode_env`:
- Reason: `Local Passed`
- DetailReason: `Local verification passed in pytorch_opencode_env; stale failed status`
- Save evidence to `/tmp/opencode/<workbook>_local_verify/` per parent skill requirements
- This indicates the CI failure is flaky or already fixed in the current checkout

## Verification

- Re-open the output workbook with `openpyxl`.
- Confirm no eligible blank `Reason` rows remain for the processed set.
- Confirm `Reason TBD` values were not flipped after classification.
- Confirm updated cells are blue.
- Spot-check at least one known-issue row and one `[Issue TBD]` row against source/issue evidence.
