# classify_ut skipped cases

This skill follows agent-guidelines AND extends it with skipped-XPU-status UT classification rules.
Always apply agent-guidelines rules including the mandatory post-write commit protocol.

## Purpose

Classify rows in the `Non-Inductor XPU Skip` workbook whose `Reason` is blank and whose
`status_xpu` is `skipped` or `xfail`. Skipped does **not** automatically mean a test environment
limitation. Every skipped row requires semantic analysis of `message_xpu`, linked issues, local
source, and, when needed, a targeted local run.

## Required Inputs

- Workbook row fields: `testfile_cuda`, `classname_cuda`, `name_cuda`, `testfile_xpu`,
  `classname_xpu`, `name_xpu`, `status_xpu`, `message_xpu`, `Reason`, `DetailReason`,
  `Exaplaination`.
- Local PyTorch checkout: `/home/daisyden/opencode/classify/pytorch`.
- XPU test checkout: `/home/daisyden/opencode/classify/pytorch/third_party/torch-xpu-ops/test/xpu`.
- Conda environment: `pytorch_opencode_env`.
- GitHub issue sources: linked issue from `message_xpu`, `intel/torch-xpu-ops` known issues, and
  PyTorch disabled-test issues when the message links to PyTorch.

## Required Tools

- `read` - inspect test methods, decorators, skip helpers, XPU wrappers/direct tests, and skip lists.
- `bash` - activate `pytorch_opencode_env`, run exact tests, query `gh issue view`, and inspect git
  refs or remote files.
- `grep` / `rg` - find exact decorators, issue links, skip reasons, and method definitions after
  identifying the semantic target. Do not classify from grep output alone.
- `gh` CLI - fetch issue state and body for all linked issues before classification.
- Workbook tooling such as `openpyxl` may write results, but semantic classification must come from
  code/issue/local-run evidence.

## Hard Constraints

- Do not treat `status_xpu = skipped` as an environment issue by default.
- Do not classify by message pattern alone. The message points to evidence; it is not the conclusion.
- Always inspect linked issue content and state:
  - Closed PyTorch disabled-test issue -> usually `To be enabled`, because the disabling issue is no
    longer active. DetailReason must include the closed issue link and say to re-enable/verify.
  - Open PyTorch disabled-test issue -> classify by issue content as `Feature gap` or
    `Failures (XPU broken)`; do not call it an environment limitation merely because the message says
    the test is disabled on a platform.
  - Intel issue link -> classify by the issue's content, not by the link alone.
- `not-support-multithread` is a feature gap, not a test environment limitation.
- If `message_xpu` is empty, missing, truncated, or nondiagnostic (`Skipped test`, `xfail`, etc.),
  run the exact test locally when feasible and inspect the source of the skip.
- Local runs must use `pytorch_opencode_env`.
- Inductor/Dynamo tests in `pytorch/test` run from the `pytorch/test` folder; other XPU wrapper/direct
  tests run from `third_party/torch-xpu-ops/test/xpu` unless source inspection proves otherwise.
- If a skip is stale and the local run passes or the message/source says support exists, classify as
  `To be enabled`.
- Do not change `Reason TBD` after classification. Mark updated `Reason`, `DetailReason`, and
  `Exaplaination` cells blue.

## Local Run Rules

Run only targeted tests. Never run a full suite.

- Inductor/Dynamo upstream tests:
  ```bash
  source ~/miniforge3/bin/activate pytorch_opencode_env && \
  python dynamo/test_dicts.py DictSubclassMethodsTests.test_binop_or
  ```
- XPU wrapper/direct tests:
  ```bash
  source ~/miniforge3/bin/activate pytorch_opencode_env && \
  python test_ops_gradients_xpu.py TestBwdGradientsXPU.test_fn_fail_gradgrad_grid_sampler_2d_xpu_float64
  ```
- Distributed upstream tests:
  ```bash
  source ~/miniforge3/bin/activate pytorch_opencode_env && \
  PYTORCH_TEST_WITH_SLOW=1 python distributed/tensor/test_random_ops.py \
    DistTensorRandomOpTestWithLocalTensor.test_pipeline_parallel_manual_seed
  ```
- If running from inside the source tree imports an unbuilt `torch`, run package sanity checks from
  `/tmp/opencode` first. For actual tests, use the test root so test imports resolve correctly.

## Workflow

1. Confirm the row is eligible:
   - `Reason` is blank.
   - `status_xpu` is `skipped` or `xfail`.
   - CUDA and XPU metadata identify the exact test case.
2. Read and normalize `message_xpu` for readability, but do not classify from the string alone.
3. If `message_xpu` contains a URL:
   - Fetch the issue with `gh issue view`.
   - Read state, title, body, platform, sample error, and affected test path.
   - Classify by issue state and content.
4. If there is no useful URL:
   - Read the local source around the exact test method and decorators.
   - Read helper/decorator implementation if the skip reason comes from a helper.
   - Run the exact test locally if the message does not explain the skip.
5. Decide the classification:
   - Closed disabled issue or local pass -> `To be enabled`.
   - Open issue describing failure/flakiness -> `Failures (XPU broken)`.
   - Unsupported XPU functionality or missing XPU-specific coverage -> `Feature gap`.
   - True hardware/process requirement unrelated to XPU implementation, such as requiring more GPUs
     than the run provided or an explicitly slow-test gate -> `Test Enviroment limitation`.
   - Stale skip where source/message says support exists -> `To be enabled`.
6. Write evidence:
   - `DetailReason` includes the issue link when present, or a source/local-run conclusion when no
     issue is available.
   - `Exaplaination` names the exact test, summarizes `message_xpu`, issue state/content, local
     source, and local-run result if used.

## Classification Examples From This Conversation

These examples document proven decisions. Re-check issue state and source before applying elsewhere.

- `not-support-multithread`:
  - Reason: `Feature gap`.
  - DetailReason: `https://github.com/intel/torch-xpu-ops/issues/3098 - XPU distributed multithread support gap`.
  - Rationale: this is an XPU distributed multithread support gap, not an environment limitation.
- `Requires at least 2 GPUs`:
  - Reason: `Test Enviroment limitation` when the row was skipped because the run did not satisfy a
    real hardware-count requirement.
  - DetailReason: the skip reason itself.
- `test is slow; run with PYTORCH_TEST_WITH_SLOW to enable test`:
  - Reason: `Test Enviroment limitation` when the only blocker is the slow-test gate.
  - DetailReason: the skip reason itself.
- Linked Intel issue `https://github.com/intel/torch-xpu-ops/issues/1682`:
  - Reason: `Failures (XPU broken)`.
  - Rationale: issue content describes distributed pipelining accuracy failures.
- PyTorch disabled issue `https://github.com/pytorch/pytorch/issues/179687` or
  `https://github.com/pytorch/pytorch/issues/179688`:
  - Reason: `To be enabled` when the issue is closed.
  - Rationale: the disabled-test issue is no longer active; re-enable and verify on XPU.
- PyTorch disabled issue `https://github.com/pytorch/pytorch/issues/138885`:
  - Reason: `Failures (XPU broken)` while the issue is open and describes failing/flaky CI behavior.
- PyTorch XPU flaky issue `https://github.com/pytorch/pytorch/issues/110040`:
  - Reason: `To be enabled` when the issue is closed.
- `requires cuda and triton` on a cudagraph/structured-trace test:
  - Reason: `To be enabled` when local source uses CUDA/Triton-only decorators or hardcoded CUDA
    tensors but XPU graph support exists and needs XPU coverage.
- Empty `message_xpu` with `status_xpu = xfail`:
  - Run locally. In this conversation, `test/dynamo/test_dicts.py DictSubclassMethodsTests.test_binop_or`
    passed in `pytorch_opencode_env`, so Reason was `To be enabled`.
- Generic `Skipped test`:
  - Run locally and inspect source. In this conversation,
    `DistTensorRandomOpTestWithLocalTensor.test_pipeline_parallel_manual_seed` reproduced the skip;
    source documented that local tensor mode does not simulate cross-pipeline-stage seeding, so
    Reason was `Feature gap`.
- `Only runs on cuda` for `test_cublas_deterministic*`:
  - Reason: `Feature gap`.
  - DetailReason includes `https://github.com/intel/torch-xpu-ops/issues/2481` when the case matches.
- `test doesn't work on XPU backend`:
  - Reason: `Feature gap`.
  - DetailReason starts with `[Issue TBD]` unless a matching issue is found.
- `Skipped! Operation does support gradgrad`:
  - Reason: `To be enabled`.
  - Rationale: the skip is stale because the message says support exists.

## Output Rules

- `Reason`: use canonical workbook labels: `To be enabled`, `Feature gap`,
  `Failures (XPU broken)`, `Failures (xpu broken)`, or `Test Enviroment limitation` as appropriate.
- `DetailReason`: include linked issue URL and semantic conclusion. If no known issue exists and the
  row represents a failure, start with `[Issue TBD]`.
- `Exaplaination`: keep the workbook spelling and include exact test identity, message/source/issue
  evidence, local-run result if used, and the reason for the final classification.

## Verification

- Re-open the output workbook with `openpyxl`.
- Confirm no eligible blank `Reason` rows remain for the processed set.
- Confirm `Reason TBD` values were not flipped after classification.
- Confirm updated cells are blue.
- Spot-check at least:
  - one `not-support-multithread` row,
  - one closed PyTorch disabled issue row,
  - one open PyTorch disabled issue row,
  - one row verified by a local run,
  - one true environment-limitation row.
