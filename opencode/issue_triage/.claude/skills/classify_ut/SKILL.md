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
- Source checkout for existence checks: use the user-provided path via `PYTORCH_SRC`; if none is
  provided, use `$HOME/upstream/pytorch`. Do not hard-code private checkout paths in commands or
  reusable logic.
- XPU test checkout: `$PYTORCH_SRC/third_party/torch-xpu-ops/test/xpu`, where `PYTORCH_SRC` is the
  source checkout above.
- Reference workbook (optional):
  `${ISSUE_TRIAGE_ROOT:-$HOME/opencode/ai_for_validation/opencode/issue_triage}/result/torch_xpu_ops_issues.xlsx`
- Deep case-existence workflow:
  `${ISSUE_TRIAGE_ROOT:-$HOME/opencode/ai_for_validation/opencode/issue_triage}/.claude/skills/bug_scrub/analyze_ci_result/check_xpu_case_existence/SKILL.md`
- Blank `status_xpu` workflow:
  `${CLASSIFY_UT_ROOT:-$ISSUE_TRIAGE_ROOT/.claude/skills/classify_ut}/blank/SKILL.md`
- Failed `status_xpu` workflow:
  `${CLASSIFY_UT_ROOT:-$ISSUE_TRIAGE_ROOT/.claude/skills/classify_ut}/failed/SKILL.md`
- Skipped/xfail `status_xpu` workflow:
  `${CLASSIFY_UT_ROOT:-$ISSUE_TRIAGE_ROOT/.claude/skills/classify_ut}/skipped/SKILL.md`

Recommended environment variables:
- `ISSUE_TRIAGE_ROOT=${ISSUE_TRIAGE_ROOT:-$HOME/opencode/ai_for_validation/opencode/issue_triage}`
- `CLASSIFY_UT_ROOT=${CLASSIFY_UT_ROOT:-$ISSUE_TRIAGE_ROOT/.claude/skills/classify_ut}`
- `PYTORCH_SRC=${PYTORCH_SRC:-$HOME/upstream/pytorch}`
- `PYTORCH_ENV=${PYTORCH_ENV:-pytorch_opencode_env}`
- `CONDA_ACTIVATE=${CONDA_ACTIVATE:-$HOME/miniforge3/bin/activate}`

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
| `To be enabled` | Base test exists and the functionality should apply to XPU, but the XPU case is not enabled/reported, the skip/wrapper is stale, or CI is missing coverage. Closed known issues also get this. |
| `Local Passed` | Test was run locally in `pytorch_opencode_env` and PASSED. Requires actual execution evidence saved to a local file. |
| `Feature gap` | XPU lacks a feature/API needed by the test. Known issue link required if available. |
| `Failures (xpu broken)` / `Failures (XPU broken)` | Test fails due to an XPU implementation bug. Known issue link required. |
| `Test Enviroment limitation` | True hardware/process constraint (multi-GPU, GCC version). NOT for skips that are stale or fixable. |
| `Not applicable` | Either (a) a CPU-only test not relevant to XPU validation (`DetailReason` = `CPU Case`), or (b) a CUDA-only behavior whose API/torch op is listed in the **`Not applicable` sheet** of `${ISSUE_TRIAGE_ROOT}/result/torch_xpu_ops_issues.xlsx` (column `Operation/API`). `DetailReason` MUST name the exact API/feature (e.g., `CUDA-specific API: torch.cuda.jiterator`) and cite the matching `Not applicable`-sheet row (`Issue ID`). Never use generic "CUDA-only test" or "No XPU test data" — always identify the specific API or feature that XPU does not support. See the **CUDA-Only Judgement Rule** below. |
| `Community Change` | The base function/case no longer exists in the source being compared, was renamed/refactored/moved, or the test is disabled by an upstream PyTorch community issue/commit. `DetailReason` MUST include the full issue/PR URL when available, or exact source/commit evidence. Do not require `last_status_xpu = passed`; base-function absence is enough. |
| `Need human check` | Deep analysis was performed but no category could be assigned with HIGH or MEDIUM confidence (see **Confidence Rubric & Need-Human-Check Rule** below). Only valid when `Reason TBD = True`. `DetailReason` MUST start with `[Confidence: LOW]` and enumerate which signals were checked and why each was inconclusive (no `Not applicable`-sheet match, base function ambiguous, no xpu wrapper located, no known issue in either repo, etc.). Never use this label as a shortcut to skip analysis — it is the explicit outcome of a thorough but inconclusive investigation. |

### `DetailReason` (String)

Must be specific enough to act on. **Every `DetailReason` that references an issue or PR MUST
use a full URL** (e.g., `https://github.com/pytorch/pytorch/issues/180324`), never a bare
number like `#180324` or a truncated link. This applies to ALL Reason categories:

- `Community Change`: full issue/PR URL from `message_xpu`, guilty commit info, or exact source
  evidence that the base function was removed/renamed/refactored
- `Failures (xpu broken)`: full issue URL from `intel/torch-xpu-ops` or `pytorch/pytorch`, or
  `[Issue_TBD]` prefix if none found after searching both repos
- `Feature gap`: full issue URL from `intel/torch-xpu-ops` or `pytorch/pytorch`, or
  `[Issue_TBD]` prefix if none found after searching both repos
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
  it uses device-agnostic patterns (-> `To be enabled`) or CUDA-specific APIs (-> `Not applicable`
  with exact API named)

Specific content requirements per Reason:
- Include issue links when known: `https://github.com/intel/torch-xpu-ops/issues/NNNN - description`
  or `https://github.com/pytorch/pytorch/issues/NNNN - description`
- Include `[Issue_TBD]` when no issue exists after searching both repositories
- Name exact APIs for `Not applicable`: `CUDA-specific API: torch.cuda.jiterator (jiterator_binary)`
- Name exact evidence for `Local Passed`: `Local verification passed in pytorch_opencode_env; stale failed status`
- For `Community Change`: include guilty commit hash, author, date, and what the commit changed

## CUDA-Only Judgement Rule (authoritative for `Not applicable`)

The **only** authoritative source for "CUDA-only behavior with no XPU equivalent" is the
`Not applicable` sheet of the reference workbook:

```
${ISSUE_TRIAGE_ROOT:-$HOME/opencode/ai_for_validation/opencode/issue_triage}/result/torch_xpu_ops_issues.xlsx
                                                                                              └─ sheet: "Not applicable"
```

(Historical note: this sheet was previously misspelled "Not Appliable" and has been renamed.
All `Not Appliable` logic is collapsed into `Not applicable`.)

### How to use the sheet

1. Open the workbook with `openpyxl` and load sheet `Not applicable`.
2. The sheet columns are:
   `Issue ID | Title | Operation/API | Category | Technical Details | Labels | State`.
3. To judge whether a row's failing/skipped API is CUDA-only, scan the `Operation/API` column
   for an entry that names the same torch op, ATen op, Python API, or backend feature the test
   exercises. Match by:
   - exact op name (e.g., `aten::_cudnn_rnn`, `torch._C._broadcast_coalesced`)
   - parent module/API (e.g., `torch.cuda.jiterator`, `cuBLAS`, `cuDNN`, `TensorExpr CUDA fuser`)
   - explicit test-name reference inside the cell (e.g., `test_no_cuda_monkeypatch`)
4. If a matching row exists -> the behavior IS CUDA-only -> classify `Not applicable`, and in
   `DetailReason` cite the matching `Issue ID` and the `Operation/API` value verbatim, e.g.
   `CUDA-specific API: aten::_cudnn_rnn (Not applicable sheet, Issue 2472)`.
5. **If no matching row exists, the behavior is NOT CUDA-only.** Do not classify `Not applicable`
   on the CUDA-only branch. Re-route to the correct label:
   - device-agnostic test that XPU should support -> `To be enabled`
   - XPU implementation bug -> `Failures (xpu broken)`
   - missing XPU feature -> `Feature gap`
   - base function removed/renamed/refactored upstream -> `Community Change`
   - test is CPU-only (`_cpu` suffix, `requires GPU`, etc.) -> `Not applicable` (CPU branch),
     `DetailReason = CPU Case`
6. The CPU branch of `Not applicable` (`DetailReason = CPU Case`) does NOT require a sheet match
   and is unaffected by this rule.

### Workflow snippet

```python
import openpyxl, os
xlsx = os.path.expanduser(
    os.environ.get("ISSUE_TRIAGE_ROOT", "~/opencode/ai_for_validation/opencode/issue_triage")
) + "/result/torch_xpu_ops_issues.xlsx"
wb = openpyxl.load_workbook(xlsx, read_only=True, data_only=True)
ws = wb["Not applicable"]
header = [c.value for c in next(ws.iter_rows(min_row=1, max_row=1))]
op_col = header.index("Operation/API")
id_col = header.index("Issue ID")
not_applicable_ops = [(row[id_col].value, row[op_col].value)
                      for row in ws.iter_rows(min_row=2) if row[op_col].value]
# Match candidate API against not_applicable_ops; only mark Not applicable on a hit.
```

### Anti-patterns

- Marking a row `Not applicable` because `name_cuda` contains the substring `cuda`.
- Marking a row `Not applicable` because the skip message says `Only runs on cuda` without
  confirming the underlying API is in the sheet (many such gates are SM-capability gates or
  stale skips -> `To be enabled`).
- Inventing a new CUDA-only API entry locally instead of adding it to the workbook sheet first.
  If you genuinely discover a new CUDA-only API, add a row to the `Not applicable` sheet
  (see the `create-not-applicable-sheet` skill) before classifying.

## Confidence Rubric & Need-Human-Check Rule (authoritative for `Reason TBD = True` rows)

Every row with `Reason TBD = True` MUST have its `DetailReason` prefixed with a confidence
tag reflecting the strength of the evidence behind the assigned `Reason`:

```
[Confidence: HIGH]    strong, verifiable evidence on at least one decisive axis
[Confidence: MEDIUM]  partial / indirect evidence; best-fit category but signals incomplete
[Confidence: LOW]     analysis performed but no axis yields a confident category -> Reason = "Need human check"
```

The prefix is REQUIRED for every `Reason TBD = True` row, regardless of which `Reason` is assigned.
Rows with `Reason TBD = False` keep their existing `DetailReason` untouched and do NOT need this
prefix (the original Reason was authoritative).

### Decision axes (check ALL that apply per row)

For each `Reason TBD = True` row, evaluate these axes from real sources, not pattern matching:

| Axis | HIGH signal | MEDIUM signal | LOW signal |
|------|-------------|---------------|------------|
| **CUDA-Only sheet match** | Exact `Operation/API` match in `Not applicable` sheet with cited `Issue ID` | Plausible API mentioned in the sheet but match is by family/parent, not exact | No matching entry, yet the test clearly uses a CUDA-only API in source |
| **Base function in `$PYTORCH_SRC`** | Located in `<testfile_cuda>` with file path + line range cited | Located in a refactored/renamed location; need user confirmation | Not found after thorough search of PyTorch + `third_party/torch-xpu-ops` |
| **XPU wrapper / generated case** | `_xpu`-suffixed case located in `test/xpu/**` or via `instantiate_device_type_tests(..., allow_xpu=True)` with file cited | XPU instantiation present but the specific case is not yet generated (e.g., dtype/OpInfo filter ambiguity) | No XPU wrapper / no `allow_xpu` / device list excludes XPU |
| **Known issue (both repos)** | Open/closed issue in `intel/torch-xpu-ops` or `pytorch/pytorch` with URL cited and verified via `gh issue view` | Issue mentioned in skip decorator but URL not verified | No issue found after `gh search issues` on both repos with multiple keyword variations |
| **Local verification** | Test actually executed locally with output saved to `/tmp/opencode/<workbook>_local_verify/` | Partial run / timeout / interpreted output | Not run |
| **Source-evidence for `Community Change`** | Guilty commit hash + author + date + diff that explains the regression | `git log` shows candidate commits but causal link not proven | No relevant commit found |

### Assigning a confidence level

- **HIGH** — At least ONE decisive axis is HIGH and there is no contradicting signal. Example: a
  Failure with a verified open issue URL → `Failures (xpu broken)` HIGH; or a verified
  `Not applicable`-sheet hit → `Not applicable` HIGH; or a verified guilty commit →
  `Community Change` HIGH; or a verified XPU wrapper with closed-issue skip →
  `To be enabled` HIGH.
- **MEDIUM** — Best-fit category is identifiable but the strongest signal is indirect.
  Example: base function exists, XPU wrapper *should* be generated but I cannot confirm the
  specific dtype/parameter slice → `To be enabled` MEDIUM.
- **LOW** — After running the full workflow, no axis produced a confident category. Set
  `Reason = "Need human check"`. LOW is NOT a fallback for skipping work; it is the explicit
  outcome of a complete-but-inconclusive investigation.

### When `Need human check` is and is not appropriate

USE `Need human check` (LOW) when ALL of the following hold:

1. The CUDA-Only sheet check was performed and did not produce an exact match.
2. The base function check in `$PYTORCH_SRC` (including `third_party/torch-xpu-ops/test/xpu/**`)
   was performed and the result is ambiguous or absent.
3. Known-issue search in BOTH `intel/torch-xpu-ops` and `pytorch/pytorch` was performed and
   yielded no usable evidence.
4. No other category (`To be enabled`, `Local Passed`, `Feature gap`, `Failures (xpu broken)`,
   `Test Enviroment limitation`, `Not applicable`, `Community Change`) can be assigned at HIGH
   or MEDIUM confidence.

DO NOT use `Need human check` when:

- You simply did not perform one of the checks above. Run the check first.
- The category is ambiguous between two clearly-applicable labels (e.g., `Feature gap` vs
  `To be enabled` when XPU clearly lacks the op). Pick the best-fit and mark MEDIUM.
- The test is CPU-only. Use `Not applicable` with `DetailReason = CPU Case`.

### `DetailReason` content requirements per confidence level

- **HIGH**: cite the decisive evidence directly (file path + line range, Issue URL, sheet
  `Issue ID`, commit hash). Example:
  `[Confidence: HIGH] To be enabled. Base function test_sdpa at $PYTORCH_SRC/test/test_nn.py#L1234-L1267; XPU wrapper $PYTORCH_SRC/third_party/torch-xpu-ops/test/xpu/test_nn_xpu.py uses instantiate_device_type_tests with allow_xpu=True.`

- **MEDIUM**: cite the strongest signal AND name the unresolved gap. Example:
  `[Confidence: MEDIUM] To be enabled. Base test_foo found at .../test_x.py#L42; XPU instantiation present but dtype=bfloat16 slice not confirmed in test/xpu/test_x_xpu.py.`

- **LOW** (→ `Need human check`): enumerate which axes were checked and why each was
  inconclusive. Example:
  `[Confidence: LOW] Need human check. Not applicable sheet: no match for torch.foo.bar. Base function: candidate test_foo present but signature changed (device-agnostic refactor in commit abc1234, behavior on XPU unclear). XPU wrapper: not located in test/xpu/**. Known issues: gh search on 'test_foo xpu' returned 0 results in both intel/torch-xpu-ops and pytorch/pytorch. Local run not attempted (test requires multi-GPU).`

### Workflow integration

- The status-specific subskills (`blank/`, `failed/`, `skipped/`) MUST emit a confidence level
  alongside their `Reason` decision for every `Reason TBD = True` row.
- When saving `.agent.xlsx`, blue-fill cells per the existing convention. NEVER modify
  `Reason TBD`.
- For LOW rows, `Reason` becomes `Need human check` and the original best-guess (if any) is
  preserved in `DetailReason` after the `[Confidence: LOW]` prefix, so a human reviewer can see
  what was considered.

## Deep Analysis Requirements (CRITICAL)

**DO NOT use simple pattern matching or regex scripts for classification.**

Every blank-Reason row requires:

1. **Regression check**: If `last_status_xpu` is `passed` but `status_xpu` is skipped or blank,
   this is a regression — prioritize the `Community Change` workflow below.
2. **Source inspection**: Read the test source to understand what it validates
3. **Local verification**: Run the test when status is ambiguous (failed/skipped with unclear message)
4. **Known issue search**: `gh search issues` on both `intel/torch-xpu-ops` and
   `pytorch/pytorch` for relevant keywords
5. **Issue state check**: `gh issue view` for any referenced issue to confirm OPEN/CLOSED
6. **Evidence recording**: Save local run output to files; record in `DetailReason`

### Case Existence Rule (source-of-truth order)

Use this rule before deciding that a blank-status case is `To be enabled`, `Not applicable`, or
`Community Change`:

1. Use the configured source checkout (`PYTORCH_SRC`, default `$HOME/upstream/pytorch`) as the
   source of truth. Update it first, including `$PYTORCH_SRC/third_party/torch-xpu-ops` when present.
2. Determine the **base function** in `testfile_cuda`: the function name actually defined in the
   PyTorch test source that most closely generates the workbook case. For generated/parameterized
   cases, this is the decorated function before device, dtype, OpInfo, and parameter suffixes are
   appended.
3. If the base function is not present in `$PYTORCH_SRC/<testfile_cuda>`, classify
   `Community Change`. This includes refactors where old CUDA/CPU-specific generated names were
   replaced by a different base function signature, e.g. old MinifierTests
   `test_after_dynamo_cpu_*` / `test_after_dynamo_cuda_*` cases replaced by
   `test_after_dynamo_*(self, device)`.
4. If the base function exists, check whether the XPU case exists. The expected generated XPU case
   name is normally `name_cuda` with `_cuda` replaced by `_xpu`; also account for decorators,
   `instantiate_device_type_tests(..., allow_xpu=True)`, OpInfo/device/dtype filters, and class
   instantiation.
5. For Non-Inductor rows, always check both direct PyTorch tests and
   `$PYTORCH_SRC/third_party/torch-xpu-ops/test/xpu/**` including subfolders such as `dynamo`, `nn`,
   `functorch`, `extended`, `profiler`, `quantization`, and `distributed`. Do not stop at the root
   `test/xpu` directory.
6. If the base exists and the XPU case exists or should be generated but is blank/missing from the
   workbook, classify `To be enabled` unless source/issue evidence proves an XPU implementation
   failure, feature gap, CPU-only case, or exact CUDA-only API.
7. If the base exists but no XPU case is generated, classify according to why: missing XPU
   registration/coverage for supported functionality is `To be enabled`; exact CUDA-only API is
   `Not applicable`/`Not applicable`; XPU bug is `Failures (xpu broken)`; missing feature is
   `Feature gap`.

### When `last_status_xpu = passed` but `status_xpu` is skipped or blank (Community Change detection)

When a test previously passed on XPU but is now skipped or not run, an upstream PyTorch commit
likely changed the test (renamed, added skip decorator, changed parametrization, moved file, etc.).
This is a **regression from community changes**, not an XPU issue.

**Workflow:**

1. Identify the test file from `testfile_cuda` (e.g., `test/inductor/test_foo.py`).
2. Use `git log` to find recent commits that touched the test file or the specific test method:
   ```bash
   cd "$PYTORCH_SRC"
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
- These are NOT `Not applicable` (they test general functionality, not CUDA-specific APIs)
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
# Select and update PyTorch source to main. Use the user's requested checkout if provided.
export PYTORCH_SRC="${PYTORCH_SRC:-$HOME/upstream/pytorch}"
cd "$PYTORCH_SRC"
git fetch origin main && git checkout main && git pull --rebase origin main

# Update torch-xpu-ops submodule to main
cd third_party/torch-xpu-ops
git fetch origin main && git checkout main && git pull --rebase origin main
cd ../..
```

### Step 2: Install nightly torch and triton-xpu

```bash
source "${CONDA_ACTIVATE:-$HOME/miniforge3/bin/activate}" "${PYTORCH_ENV:-pytorch_opencode_env}"

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
