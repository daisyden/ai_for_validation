# classify_ut

This skill follows agent-guidelines AND extends it with workbook-specific UT classification rules.
Always apply agent-guidelines rules including the mandatory post-write commit protocol.

## Purpose

Classify blank `Reason` rows in XPU UT status workbooks by performing deep source analysis,
local verification, and known-issue searches. This skill works for ANY test-case workbook
(Non-Inductor, Inductor, or other sheets) as long as the required columns are present.

## Applicable Workbooks and Sheets

This skill applies to any workbook/sheet with these columns:
- Test identification: `testfile_cuda`, `classname_cuda`, `name_cuda`
- XPU status: `status_xpu`, `message_xpu`
- Classification: `Reason`, `DetailReason`
- Tracking: `Reason TBD`

Known sheets:
- `Non-Inductor XPU Skip` in `Non_inductor_ut_status_ww*.xlsx`
- `Cuda pass xpu skip` in `Inductor_ut_status_ww*.xlsx`
- Any similarly structured sheet from weekly UT status reports

## Inputs

- Target workbook: the `.xlsx` file provided by the user
- Reference workbook (optional):
  `/home/daisyden/opencode/ai_for_validation/opencode/issue_triage/result/torch_xpu_ops_issues.xlsx`
- Deep case-existence workflow:
  `/home/daisyden/opencode/ai_for_validation/opencode/issue_triage/.claude/skills/bug_scrub/analyze_ci_result/check_xpu_case_existence/SKILL.md`
- Blank `status_xpu` workflow:
  `/home/daisyden/opencode/ai_for_validation/opencode/issue_triage/.claude/skills/classify_ut/blank/SKILL.md`
- Failed `status_xpu` workflow:
  `/home/daisyden/opencode/ai_for_validation/opencode/issue_triage/.claude/skills/classify_ut/failed/SKILL.md`
- Skipped/xfail `status_xpu` workflow:
  `/home/daisyden/opencode/ai_for_validation/opencode/issue_triage/.claude/skills/classify_ut/skipped/SKILL.md`

## Status-specific classification skills

These subskills are authoritative for case-specific `Reason`, `DetailReason`, and
`DetailReason` decisions. Always read the matching subskill before classifying rows with that
`status_xpu` value.

| `status_xpu` | Skill | Purpose |
|--------------|-------|---------|
| blank / empty | `classify_ut/blank/SKILL.md` | Deep case-existence analysis for rows with no XPU result |
| `failed` | `classify_ut/failed/SKILL.md` | Failure-message, local-run, and known-issue analysis |
| `skipped` / `xfail` | `classify_ut/skipped/SKILL.md` | Skip-message, linked-issue, source, and local-run analysis |

Do not collapse these workflows into a single pattern-matching script. Bulk scripts may prepare
workbook columns, collect candidate rows, or apply already-reviewed decisions, but the actual
classification must follow the status-specific skill.

## Column Definitions

### `Reason TBD` (Boolean)

Tracks whether the **original** Reason was blank in the SOURCE workbook (e.g.,
`Inductor_ut_status_ww18_26.xlsx`), NOT the `.agent.xlsx` output:
- Compare against the ORIGINAL workbook to determine the value
- If `Reason` is blank in the ORIGINAL workbook -> set `Reason TBD = True`
- If `Reason` is already filled in the ORIGINAL workbook -> set `Reason TBD = False`
- **NEVER change `Reason TBD` after initialization.** It is a permanent record of which rows
  required deep analysis, regardless of whether analysis is now complete.
- **NEVER set `Reason TBD = False` when filling in a Reason.** The column records the ORIGINAL
  state, not the current state. A row with `Reason TBD = True` and `Reason = "Not applicable"`
  means "this row originally had no Reason and was classified by deep analysis."

### `Reason` (String) - Canonical Labels

| Label | When to Use |
|-------|-------------|
| `To be enabled` | Test should work on XPU but skip/wrapper is stale, or the test exists but isn't in CI. Closed known issues also get this. |
| `Local Passed` | Test was run locally in `pytorch_opencode_env` and PASSED. Requires actual execution evidence saved to a local file. |
| `Feature gap` | XPU lacks a feature/API needed by the test. Known issue link required if available. |
| `Failures (xpu broken)` / `Failures (XPU broken)` | Test fails due to an XPU implementation bug. Known issue link required. |
| `Test Enviroment limitation` | True hardware/process constraint (multi-GPU, slow gate, GCC version). NOT for skips that are stale or fixable. |
| `Not Appliable` | CUDA-only API with no XPU equivalent. `DetailReason` MUST name the exact API (e.g., `CUDA-specific API: torch.cuda.jiterator`). Never use generic "CUDA-only test" or "No XPU test data" — always identify the specific API or feature that XPU does not support. |
| `Not applicable` | Test removed/renamed upstream (`DetailReason` = `Community Changes`), OR CPU-only test not relevant to XPU validation (`DetailReason` = `CPU Case`). |
| `Community Change` | Test previously passed on XPU (`last_status_xpu = passed`) but now skipped/blank due to an upstream PyTorch commit or disabled-test issue. `DetailReason` MUST include the full issue/PR URL (e.g., `https://github.com/pytorch/pytorch/issues/NNNNN`) or guilty commit hash with author and summary. |

### `DetailReason` (String)

Must be specific enough to act on. **Every `DetailReason` that references an issue or PR MUST
use a full URL** (e.g., `https://github.com/pytorch/pytorch/issues/180324`), never a bare
number like `#180324` or a truncated link. This applies to ALL Reason categories:

- `Community Change`: full issue/PR URL from `message_xpu` or guilty commit info
- `Failures (xpu broken)`: full issue URL, or `[Issue_TBD]` prefix if none found
- `Feature gap`: full issue URL when available
- `To be enabled`: full issue URL for closed issues with stale skip decorators

**Where to find issue/PR URLs:**
- `message_xpu` often contains the URL directly (e.g., PyTorch disabled-test messages say
  `Test is disabled because an issue exists disabling it: <URL>`)
- `skipIfXpu` decorators embed the URL in the skip reason
- `gh search issues` results return full URLs
- Git commit messages reference PR numbers that map to `https://github.com/pytorch/pytorch/pull/NNNNN`

**Common mistakes to avoid:**
- Writing `#181863` instead of `https://github.com/pytorch/pytorch/issues/181863`
- Omitting the URL entirely when `message_xpu` already contains it
- Writing `triton issue` without searching for the actual issue URL
- Writing a commit hash without the PR URL when one exists
- Writing generic `No XPU test data`, `CUDA-only test`, or `No XPU wrapper` without reading the
  test source and identifying the specific API or feature. Always read the test to determine whether
  it uses device-agnostic patterns (-> `To be enabled`) or CUDA-specific APIs (-> `Not Appliable`
  with exact API named)

Specific content requirements per Reason:
- Include issue links when known: `https://github.com/intel/torch-xpu-ops/issues/NNNN - description`
- Include `[Issue_TBD]` when no issue exists after searching
- Name exact APIs for `Not Appliable`: `CUDA-specific API: torch.cuda.jiterator (jiterator_binary)`
- Name exact evidence for `Local Passed`: `Local verification passed in pytorch_opencode_env; stale failed status`
- For `Community Change`: include guilty commit hash, author, date, and what the commit changed

## Deep Analysis Requirements (CRITICAL)

**DO NOT use simple pattern matching or regex scripts for classification.**

Every blank-Reason row requires:

1. **Regression check**: If `last_status_xpu` is `passed` but `status_xpu` is skipped or blank,
   this is a regression — prioritize the `Community Change` workflow below.
2. **Source inspection**: Read the test source to understand what it validates
3. **Local verification**: Run the test when status is ambiguous (failed/skipped with unclear message)
4. **Known issue search**: `gh search issues` on `intel/torch-xpu-ops` for relevant keywords
5. **Issue state check**: `gh issue view` for any referenced issue to confirm OPEN/CLOSED
6. **Evidence recording**: Save local run output to files; record in `DetailReason`

### When `last_status_xpu = passed` but `status_xpu` is skipped or blank (Community Change detection)

When a test previously passed on XPU but is now skipped or not run, an upstream PyTorch commit
likely changed the test (renamed, added skip decorator, changed parametrization, moved file, etc.).
This is a **regression from community changes**, not an XPU issue.

**Workflow:**

1. Identify the test file from `testfile_cuda` (e.g., `test/inductor/test_foo.py`).
2. Use `git log` to find recent commits that touched the test file or the specific test method:
   ```bash
   cd /home/daisyden/opencode/classify/pytorch
   git log --oneline -20 -- test/inductor/test_foo.py
   ```
3. For each candidate commit, use `git show` to inspect what changed:
   ```bash
   git show <commit_hash> -- test/inductor/test_foo.py
   ```
4. Look for changes that would cause the test to skip or not run on XPU:
   - Test method renamed or removed
   - New skip decorator added (e.g., `@skipIfXpu`, `unittest.skip`, `@requires_cuda`)
   - Parametrization changed (device list no longer includes XPU)
   - Test class restructured or moved to a different file
   - `instantiate_device_type_tests` call changed
   - New `TestFailure` or `expectedFailure` entry added
5. Once the guilty commit is identified:
   - Reason: `Community Change`
   - DetailReason: `Community commit <short_hash> (<author>, <date>) - <summary of what changed>.`
     Include full reasoning: git log output, git show diff, and how the
     commit caused the test to stop running on XPU

**Example:**
```
Reason: Community Change
DetailReason: Community commit abc1234 (John Doe, 2026-05-01) - renamed test_foo to test_bar; XPU test name no longer matches.
  last_status_xpu=passed but status_xpu=blank. git log shows commit abc1234 "Rename test_foo to test_bar".
  git show abc1234 confirms the method was renamed.
```

If git log shows NO relevant commits touching the test, fall through to the normal classification
workflow (source inspection, local run, known issue search).

### PyTorch Disabled-Test Community Changes

When `message_xpu` says `Test is disabled because an issue exists disabling it: <URL>`:
- The test was disabled by the PyTorch CI infrastructure via `.pytorch-disabled-tests.json`
- This IS a community change — the community disabled the test
- **Extract the full issue URL from `message_xpu`** and put it in `DetailReason`
- Reason: `Community Change`
- DetailReason: `<full URL>. Test previously passed (last_status_xpu=passed) but disabled by PyTorch
  disabled-test mechanism.`

**Example:**
```
message_xpu: "Test is disabled because an issue exists disabling it: https://github.com/pytorch/pytorch/issues/180324 for platform(s) linux, slow"
Reason: Community Change
DetailReason: https://github.com/pytorch/pytorch/issues/180324. Test previously passed (last_status_xpu=passed) but disabled by PyTorch disabled-test mechanism (platform: linux, slow)
```

### When to use `Local Passed`

`Local Passed` requires ALL of:
- The test was actually executed locally in `pytorch_opencode_env`
- The test PASSED (not skipped, not errored, not 0-tests-collected)
- The output is saved to a local verification file
- If 0 tests collected: the test does NOT exist -> use `Not applicable` instead

### When closed issues mean `To be enabled`

If a test is skipped by a `skipIfXpu` decorator or `inductor_skips` entry or `TestFailure` dict,
AND the linked issue is CLOSED (fixed), then:
- Reason: `To be enabled`
- DetailReason: `<issue_url> (CLOSED <date>) - <decorator/skip entry> not yet removed; issue is fixed`
- The test should be enabled by removing the stale skip

### When to run slow tests

If `message_xpu` says `test is slow; run with PYTORCH_TEST_WITH_SLOW to enable test`:
- Run locally with `PYTORCH_TEST_WITH_SLOW=1`
- If PASSES: `Local Passed` with detail `Local verification passed with PYTORCH_TEST_WITH_SLOW=1`
- If FAILS: search known issues, classify as failure
- If test doesn't exist (0 collected): `Not applicable`

### When `Skipped!` message has no clear reason

1. Search for the skip source in the test file (grep for `TestFailure`, `inductor_skips`, `skipIfXpu`, `unittest.skip`)
2. Read the skip mechanism to understand WHY it skips
3. Search `intel/torch-xpu-ops` issues for the test name or related keywords
4. If a known issue is found and is CLOSED: try running the base test without the skip
5. If the base test passes: `To be enabled`
6. If no known issue: try running the test without skip; if passes: `To be enabled`
7. If fails: `Failures (xpu broken)` with issue link or `[Issue_TBD]`

### CPU Tests (Not applicable / CPU Case)

If a test is a CPU test (determined by ANY of the following), classify as `Not applicable`:
- Test name ends with `_cpu` or contains `_cpu_` (e.g., `test_fp8_cpu`, `test_fp8_view_of_param_cpu`)
- Test has `cpu` as its device parameter in parametrization
- Skip message says "requires GPU", "requires a GPU", "GPU_TYPE", or similar GPU-requirement message
- Test class or configuration targets CPU-only execution

**Classification:**
- Reason: `Not applicable`
- DetailReason: `CPU Case. CPU test (device=cpu per test name/parametrization), not relevant to XPU validation`

This rule takes PRIORITY over other skip-message rules. Check for CPU test first before analyzing
skip messages for other classifications.

### Failures (xpu broken) Issue Link Requirement

ALL rows classified as `Failures (xpu broken)` MUST have issue tracking in `DetailReason`:
- If a known issue exists on `intel/torch-xpu-ops` or `pytorch/pytorch`: include the full issue URL
  (e.g., `https://github.com/intel/torch-xpu-ops/issues/NNNN - description`)
- If NO known issue exists after searching: prefix `DetailReason` with `[Issue_TBD]`
  (e.g., `[Issue_TBD] XPU fails with RuntimeError: unsupported dtype`)

This applies universally to ALL `Failures (xpu broken)` rows regardless of status_xpu value
(failed, skipped, or blank).

### SM89/SM90 CUDA capability gates

Tests skipped due to `SM90OrLater`, `sm89`, or similar CUDA compute capability checks:
- These are NOT `Not Appliable` (they test general functionality, not CUDA-specific APIs)
- Classify as `To be enabled` because XPU should support the underlying operation
- DetailReason: `Skipped due to <SM check> CUDA capability gate; XPU should support this test`

## Tools Required

| Tool | Purpose |
|------|---------|
| `read` | Inspect test source, wrappers, skip lists, decorators |
| `bash` | Run tests locally, activate env, directory listing |
| `grep` | Find decorators, skip entries, method definitions |
| `gh` CLI | Search issues (`gh search issues`), view issue state (`gh issue view`) |
| `openpyxl` | Read/write workbook cells |
| `pytest --collect-only` | Check if test exists without running it |
| `python <test_file> -k <pattern> -v` | Run specific tests with PyTorch test runner |

## Environment Setup (MANDATORY before any local runs)

Before running any tests, ensure the environment is up-to-date with the latest nightly builds
and source code. This prevents stale results from outdated packages or test definitions.

### Step 1: Update PyTorch source and torch-xpu-ops

```bash
# Update PyTorch source to main
cd /home/daisyden/opencode/classify/pytorch
git fetch origin main && git checkout main && git pull origin main

# Update torch-xpu-ops submodule to main
cd third_party/torch-xpu-ops
git fetch origin main && git checkout main && git pull origin main
cd ../..
```

### Step 2: Install nightly torch and triton-xpu

```bash
source ~/miniforge3/bin/activate pytorch_opencode_env

# Install latest nightly torch for XPU
pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/xpu

# Install latest nightly triton-xpu
pip download --no-deps --index-url https://download.pytorch.org/whl/nightly/xpu --pre pytorch-triton-xpu --dest /tmp/opencode/triton_whl
pip install --root-user-action=ignore /tmp/opencode/triton_whl/pytorch_triton_xpu-*.whl
```

### Step 3: Verify installation

```bash
python -c "import torch; print(f'torch={torch.__version__}, xpu_available={torch.xpu.is_available()}')"
python -c "import triton; print(f'triton={triton.__version__}')"
```

If the environment is already up-to-date from a recent run in the same session, skip this setup.

## Workflow

1. Run the environment setup steps above (update source + packages).
2. Open the target sheet in the workbook.
3. Ensure workbook column `Reason TBD` exists.
4. Initialize `Reason TBD` from the ORIGINAL workbook's `Reason` value (not the `.agent.xlsx`):
   - if `Reason` is blank in the original, set `Reason TBD = True`
   - otherwise set `Reason TBD = False`
   - Once set, NEVER modify this column again during classification or reclassification
5. If XPU test metadata is blank, derive it from CUDA metadata:
   - `classname_cuda` ending with `CUDA` -> replace with `XPU`
   - `name_cuda` ending with `_cuda` -> replace with `_xpu`
   - `testfile_xpu` defaults to `testfile_cuda` when blank
6. For each blank-Reason row, choose the status-specific skill:
   - blank `status_xpu` -> `classify_ut/blank/SKILL.md`
   - `status_xpu = failed` -> `classify_ut/failed/SKILL.md`
   - `status_xpu = skipped` or `xfail` -> `classify_ut/skipped/SKILL.md`
7. Execute the selected skill's deep analysis workflow.
8. Fill `Reason` and `DetailReason`. Mark cells blue.
9. Save local verification results to `/tmp/opencode/<workbook>_local_verify/`
10. Save output workbook as `.agent.xlsx` (do not modify original).
11. Verify: 0 blank Reason rows remaining, ZIP integrity OK, reason counts match.

## Local Verification Evidence (MANDATORY for `Local Passed`)

All `Local Passed` classifications require:
1. A JSON file saved to `/tmp/opencode/<workbook>_local_verify/` with:
   - Test file path
   - Test class and method
   - Full command used to run the test
   - PASS/FAIL/SKIP result
   - Relevant output lines
2. A summary text file with totals and any warnings

Without this evidence, `Local Passed` is NOT a valid classification.

## Notes

- Save output as `.agent.xlsx`; do not modify original workbook.
- Preserve existing `Reason` and `DetailReason` unless deep analysis justifies an update.
- Mark updated cells blue using `PatternFill(start_color='ADD8E6', end_color='ADD8E6', fill_type='solid')`.
