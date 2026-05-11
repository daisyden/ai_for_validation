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
- Local runs must use `pytorch_opencode_env`.
- Do not change `Reason TBD` after classification.
- Mark updated cells blue.

## Workflow Steps

### Step 1: Confirm Eligibility
- `Reason` is blank
- `status_xpu` is `skipped` or `xfail`
- CUDA/XPU metadata identifies the exact test case

### Step 2: Analyze `message_xpu`

Parse the skip message to identify the skip mechanism:

| Message Pattern | Skip Mechanism | Next Action |
|----------------|----------------|-------------|
| `skipIfXpu: <reason>, <issue_url>` | `@skipIfXpu` decorator | Check issue state |
| `Test is disabled because an issue exists: <url>` | PyTorch disabled-test | Check issue state + run locally |
| `test is slow; run with PYTORCH_TEST_WITH_SLOW` | Slow test gate | Run with `PYTORCH_TEST_WITH_SLOW=1` |
| `Requires at least N GPUs` | Hardware requirement | `Test Enviroment limitation` |
| `not-support-multithread` | XPU feature gap | `Feature gap` + #3098 |
| `Only runs on cuda` | CUDA-only gate | Analyze what API is CUDA-only |
| `Skipped!` (generic) | Various mechanisms | Read source to find skip dict/decorator |
| `Fails with Triton update` | Unconditional `unittest.skip` | `Test Enviroment limitation` (all backends) |
| `Fails under GCC 13` | Compiler version | `Test Enviroment limitation` |
| `sm89 errors out` / `SM90OrLater` | CUDA compute capability gate | `To be enabled` (not CUDA-specific) |
| Empty / `Skipped test` / `xfail` | Unknown | Run locally + read source |

### Step 3: Check Issue State

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

### Step 4: Handle `Skipped!` Without Clear Message

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
   ```

4. **Try running without the skip**:
   - For `inductor_skips`: test the op directly via `torch.compile`
   - For `TestFailure` dicts: run the base test in the parent test file
   - For dtype restrictions: check if the dtype actually works on XPU

5. **Classify based on results**:
   - Base test passes -> `To be enabled` (skip is stale)
   - Base test fails with known issue -> `Failures (xpu broken)` + issue link
   - Base test fails without known issue -> `Failures (xpu broken)` + `[Issue TBD]`

### Step 5: Handle Slow Tests

```bash
source ~/miniforge3/bin/activate pytorch_opencode_env && \
PYTORCH_TEST_WITH_SLOW=1 python test/inductor/<file>.py -k "<test_name>" -v
```

- PASSES -> `Local Passed`, detail: `Local verification passed with PYTORCH_TEST_WITH_SLOW=1`
- FAILS -> Search known issues, classify as failure
- 0 tests collected -> `Not applicable` (test removed)

### Step 6: Handle PyTorch Disabled-Test Issues

ALL PyTorch disabled-test rows should be run locally regardless of issue state:

```bash
python test/inductor/<file>.py -k "<test_name>" -v
```

- PASSES locally -> `Local Passed` (the disabled-test mechanism is flaky CI, not a real failure)
- FAILS locally -> `Failures (xpu broken)` or `Feature gap` based on error

### Step 7: Handle `skipIfXpu` with Closed Issues

When `skipIfXpu` references a CLOSED issue:
- The issue is FIXED but the `skipIfXpu` decorator hasn't been removed yet
- Classify as `To be enabled`
- DetailReason: `<issue_url> (CLOSED <date>) - skipIfXpu decorator not yet removed; issue is fixed`

### Step 8: Handle SM89/SM90 Capability Gates

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

### Environment Limitations

| Test | Evidence | Classification |
|------|----------|---------------|
| `Fails with Triton update` | Unconditional `unittest.skip` in source, all backends | `Test Enviroment limitation` |
| `GCC 13 vector codegen` | CPU test, compiler version dependent | `Test Enviroment limitation` |
| `Requires at least 2 GPUs` | Hardware requirement | `Test Enviroment limitation` |

## Output Rules

- `Reason`: use canonical workbook labels exactly as spelled
- `DetailReason`: include linked issue URL and semantic conclusion
- `Explaination`: include test identity, tools used, evidence found, reasoning chain

## Verification

- Re-open output workbook with `openpyxl`
- Confirm 0 eligible blank `Reason` rows remain
- Confirm `Reason TBD` values unchanged
- Confirm updated cells are blue
- Spot-check at least one row from each classification category
