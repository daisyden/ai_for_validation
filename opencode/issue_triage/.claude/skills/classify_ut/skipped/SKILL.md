# classify_ut skipped/xfail cases

This skill follows agent-guidelines AND extends it with skipped-XPU-status UT classification rules.

## Purpose

Classify rows whose `Reason` is blank and `status_xpu` is `skipped` or `xfail`.
Skipped does **not** automatically mean a test environment limitation. Every skipped row requires
semantic analysis of `message_xpu`, linked issues, local source, and targeted local runs.

## Required Tools

| Tool | Purpose | Example |
|------|---------|---------|
| `read` | Inspect test methods, decorators, skip helpers | Read `test_torchinductor.py` line 4110 |
| `bash` | Run tests locally, activate env | `python test_file.py -k pattern -v` |
| `grep` | Find decorators, skip entries, `TestFailure` dicts | `grep -n 'skipIfXpu' test_file.py` |
| `gh` CLI | Fetch issue state, search issues | `gh issue view 2334 --repo=intel/torch-xpu-ops --json=state,closedAt` |
| `gh search` | Find known issues by keyword | `gh search issues "test_div7 xpu" --repo=intel/torch-xpu-ops` |

## Hard Constraints

- Do not treat `status_xpu = skipped` as environment issue by default.
- Do not classify by message pattern alone. The message points to evidence; it is not the conclusion.
- Always check linked issue state (OPEN vs CLOSED) before classifying.
- If `message_xpu` is empty or nondiagnostic, run the test locally.
- Local runs must use `pytorch_opencode_env` with up-to-date nightly torch, triton-xpu, and the
  configured source checkout (`PYTORCH_SRC`, default `$HOME/upstream/pytorch`; see parent
  `classify_ut/SKILL.md` Environment Setup).
- Do not change `Reason TBD` after classification.
- Mark updated cells blue.

## Workflow Steps

### Step 1: Confirm Eligibility
- `Reason` is blank
- `status_xpu` is `skipped` or `xfail`
- CUDA/XPU metadata identifies the exact test case

### Step 1.5: Check for Community Change Regression

If `last_status_xpu = passed` (test previously passed on XPU but is now skipped):

1. This is likely a regression from an upstream commit. Check git log first:
   ```bash
   cd "$PYTORCH_SRC"
   git log --oneline -20 -- <testfile_cuda>
   ```
2. For candidate commits, inspect the diff:
   ```bash
   git show <commit_hash> -- <testfile_cuda>
   ```
3. Look for changes that would cause the skip: new skip decorator, renamed method,
   changed parametrization, restructured test class, new `TestFailure` entry, etc.
4. If a guilty commit is found:
   - Reason: `Community Change`
   - DetailReason: `Community commit <short_hash> (<author>, <date>) - <summary>.`
     Include git log + git show evidence, how the commit caused the skip
5. If no relevant commit found, continue to Step 2 (normal skip analysis).

### Step 2: Check for CPU Test (Priority Rule)

Before analyzing the skip message, check if this is a CPU test:
- Test name ends with `_cpu` or contains `_cpu_` (e.g., `test_fp8_cpu`, `test_while_loop_with_parameters_cpu`)
- Test has `cpu` as device parameter in parametrization
- Skip message says "requires GPU", "requires a GPU", "GPU_TYPE", or similar

If ANY of the above is true:
- Reason: `Not applicable`
- DetailReason: `CPU Case. CPU test (device=cpu per test name/parametrization), not relevant to XPU validation`
- Skip remaining steps.

### Step 3: Analyze `message_xpu`

Parse the skip message to identify the skip mechanism. **When `message_xpu` contains a URL,
always extract and include the full URL in `DetailReason`.**

| Message Pattern | Skip Mechanism | Next Action |
|----------------|----------------|-------------|
| `Test is disabled because an issue exists disabling it: <URL>` | PyTorch disabled-test | Extract URL -> `Community Change`, DetailReason = full URL |
| `skipIfXpu: <reason>, <issue_url>` | `@skipIfXpu` decorator | Extract URL -> Check issue state |
| `Test is disabled because an issue exists: <url>` | PyTorch disabled-test | Check issue state + run locally |
| `test is slow; run with PYTORCH_TEST_WITH_SLOW` | Slow test gate | Run with `PYTORCH_TEST_WITH_SLOW=1` |
| `Requires at least N GPUs` | Hardware requirement | `Test Enviroment limitation` |
| `not-support-multithread` | XPU feature gap | `Feature gap` + #3098 |
| `Only runs on cuda` | CUDA-only gate | Apply **CUDA-Only Judgement Rule** (parent skill): look up the API in the `Not applicable` sheet of `${ISSUE_TRIAGE_ROOT}/result/torch_xpu_ops_issues.xlsx`. Match -> `Not applicable` with Issue ID cited. No match -> `To be enabled` (often a stale gate or SM-capability check). |
| `Skipped!` (generic) | Various mechanisms | Read source to find skip dict/decorator |
| `Fails with Triton update` | Unconditional `unittest.skip` | `Test Enviroment limitation` (all backends) |
| `Fails under GCC 13` | Compiler version | `Test Enviroment limitation` |
| `sm89 errors out` / `SM90OrLater` | CUDA compute capability gate | `To be enabled` (not CUDA-specific) |
| Empty / `Skipped test` / `xfail` | Unknown | Run locally + read source |

### Step 4: Check Issue State

For ANY issue URL found in the message or source:

```bash
gh issue view <number> --repo=<org/repo> --json=state,title,closedAt
```

**Decision tree based on issue state:**

```
Issue CLOSED?
  YES -> Run test locally
    PASSES -> "To be enabled" (issue fixed, skip stale)
    FAILS  -> Search for new issue, classify as failure
  NO (OPEN) ->
    Issue describes missing feature? -> "Feature gap"
    Issue describes broken implementation? -> "Failures (xpu broken)"
    Issue describes flaky CI? -> Run locally
      PASSES -> "Local Passed"
      FAILS  -> "Failures (xpu broken)"
```

### Step 5: Handle `Skipped!` Without Clear Message

When `message_xpu` is just `Skipped!` with no explanation:

1. **Find the skip source** in the test file:
   ```bash
   grep -n 'TestFailure\|inductor_skips\|skipIfXpu\|unittest.skip' test/inductor/<file>.py
   ```

2. **Read the skip mechanism**:
   - `test_failures_*` dict with `is_skip=True` -> in-tree XPU skip
   - `inductor_skips["xpu"]` -> inductor dtype/op skip
   - `unittest.skip("reason")` -> unconditional skip (all backends)
   - Dtype restriction in test body (`raise unittest.SkipTest("Skipped!")`) -> check allowed dtypes

3. **Search for known issues**:
   ```bash
   gh search issues "<test_name> xpu" --repo=intel/torch-xpu-ops --limit=5
   gh search issues "<test_name> xpu" --repo=pytorch/pytorch --limit=5
   ```

4. **Try running without the skip**:
   - For `inductor_skips`: test the op directly via `torch.compile`
   - For `TestFailure` dicts: run the base test in the parent test file
   - For dtype restrictions: check if the dtype actually works on XPU

5. **Classify based on results**:
   - Base test passes -> `To be enabled` (skip is stale)
   - Base test fails with known issue -> `Failures (xpu broken)` + issue link in DetailReason
   - Base test fails without known issue after searching both repos -> `Failures (xpu broken)` +
     `[Issue_TBD]` prefix in DetailReason

### Step 6: Handle Slow Tests

```bash
source "${CONDA_ACTIVATE:-$HOME/miniforge3/bin/activate}" "${PYTORCH_ENV:-pytorch_opencode_env}" && \
PYTORCH_TEST_WITH_SLOW=1 python test/inductor/<file>.py -k "<test_name>" -v
```

- PASSES -> `Local Passed`, detail: `Local verification passed with PYTORCH_TEST_WITH_SLOW=1`
- FAILS -> Search known issues, classify as failure
- 0 tests collected -> `Not applicable` (test removed)

### Step 7: Handle PyTorch Disabled-Test Issues

ALL PyTorch disabled-test rows should be run locally regardless of issue state:

```bash
python test/inductor/<file>.py -k "<test_name>" -v
```

- PASSES locally -> `Local Passed` (the disabled-test mechanism is flaky CI, not a real failure)
- FAILS locally -> `Failures (xpu broken)` or `Feature gap` based on error

### Step 8: Handle `skipIfXpu` with Closed Issues

When `skipIfXpu` references a CLOSED issue:
- The issue is FIXED but the `skipIfXpu` decorator hasn't been removed yet
- Classify as `To be enabled`
- DetailReason: `<issue_url> (CLOSED <date>) - skipIfXpu decorator not yet removed; issue is fixed`

### Step 9: Handle SM89/SM90 Capability Gates

Tests skipped due to CUDA compute capability checks (e.g., `@unittest.skipIf(not SM90OrLater, ...)`):
- These test GENERAL functionality that happens to need SM90 on NVIDIA hardware
- XPU can run the same functionality without SM90
- Classify as `To be enabled`
- DetailReason: `Skipped due to SM90OrLater CUDA capability gate; XPU should support this test`

## Classification Examples (Proven Decisions)

### Closed skipIfXpu Issues

| Test | Issue | State | Classification |
|------|-------|-------|---------------|
| `test_copy_non_blocking_is_pinned_*` | #2334 | CLOSED 2026-03-25 | `To be enabled` |
| `test_div7_*` | intel-xpu-backend-for-triton#6401 | CLOSED 2026-04-22 | `To be enabled` |
| `test_comprehensive_nn_functional_linear_xpu_float16` | #2956 | CLOSED 2026-04-01 | `To be enabled` |

### Open Issues (Real Gaps)

| Test | Issue | State | Classification |
|------|-------|-------|---------------|
| `test_bad_cast` (fp8) | #2888 | OPEN | `Feature gap` |
| `gpu_cpp_wrapper` complex add tests | #3187 | OPEN | `Failures (xpu broken)` |
| `not-support-multithread` | #3098 | OPEN | `Feature gap` |

### Stale Skips Without Issues

| Test | Evidence | Classification |
|------|----------|---------------|
| `inductor_skips["xpu"]["lu"] = {f32}` | `torch.linalg.lu` passes through `torch.compile` on XPU | `To be enabled` |
| `inductor_skips["xpu"]["masked.cumprod"] = {f16}` | `torch.masked.cumprod` passes on XPU | `To be enabled` |
| `test_remove_noop_slice` compile_subprocess skip | Base test passes in `test_torchinductor.py` | `To be enabled` |

### Local Passed Examples

| Test | Evidence | Classification |
|------|----------|---------------|
| PyTorch disabled-test #176968 | `python test_compile_worker.py -k test_quiesce_repeatedly` -> OK | `Local Passed` |
| Slow test with `PYTORCH_TEST_WITH_SLOW=1` | Ran 1 test in 77.9s, OK | `Local Passed` |
| `test_low_memory_max_pool_dilation_*` | XPU variants pass (use_block_ptr_{False,True}_xpu) | `Local Passed` |

### CPU Tests (Not applicable)

| Test | Evidence | Classification |
|------|----------|---------------|
| `test_fp8_cpu` | Test name ends with `_cpu`, skip msg "requires GPU" | `Not applicable`, DetailReason=`CPU Case` |
| `test_while_loop_with_parameters_cpu` | Test name ends with `_cpu` | `Not applicable`, DetailReason=`CPU Case` |
| `test_fp8_cpu_with_stack_allocation` | Test name contains `_cpu_` | `Not applicable`, DetailReason=`CPU Case` |

### Environment Limitations

| Test | Evidence | Classification |
|------|----------|---------------|
| `Fails with Triton update` | Unconditional `unittest.skip` in source, all backends | `Test Enviroment limitation` |
| `GCC 13 vector codegen` | CPU test, compiler version dependent | `Test Enviroment limitation` |
| `Requires at least 2 GPUs` | Hardware requirement | `Test Enviroment limitation` |

## Output Rules

- `Reason`: use canonical workbook labels exactly as spelled
- `DetailReason`: include full issue/PR URL (e.g., `https://github.com/pytorch/pytorch/issues/NNNNN`),
  never bare numbers like `#NNNNN`. Extract URLs from `message_xpu` when present.
  Include test identity, tools used, evidence found, and reasoning chain.

## Verification

- Re-open output workbook with `openpyxl`
- Confirm 0 eligible blank `Reason` rows remain
- Confirm `Reason TBD` values unchanged
- Confirm updated cells are blue
- Spot-check at least one row from each classification category
