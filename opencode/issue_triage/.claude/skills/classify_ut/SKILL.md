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
- Tracking: `Reason TBD`, `Explaination`

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
`Explaination` decisions. Always read the matching subskill before classifying rows with that
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

Tracks whether the **original** Reason was blank when the workbook was first processed:
- If `Reason` is blank at processing time -> set `Reason TBD = True`
- If `Reason` is already filled -> set `Reason TBD = False`
- **NEVER change `Reason TBD` after initialization.** It is a permanent record of which rows
  required deep analysis, regardless of whether analysis is now complete.

### `Explaination` (String)

Records the full reasoning chain for the classification decision. Must include:
- Exact test identity (file, class, method)
- What tools/steps were used (local run, grep, gh issue search, source read)
- What evidence was found (pass/fail output, issue state, source code content)
- Why the chosen Reason follows from the evidence

Keep the workbook spelling `Explaination` (not `Explanation`).

### `Reason` (String) - Canonical Labels

| Label | When to Use |
|-------|-------------|
| `To be enabled` | Test should work on XPU but skip/wrapper is stale, or the test exists but isn't in CI. Closed known issues also get this. |
| `Local Passed` | Test was run locally in `pytorch_opencode_env` and PASSED. Requires actual execution evidence saved to a local file. |
| `Feature gap` | XPU lacks a feature/API needed by the test. Known issue link required if available. |
| `Failures (xpu broken)` / `Failures (XPU broken)` | Test fails due to an XPU implementation bug. Known issue link required. |
| `Test Enviroment limitation` | True hardware/process constraint (multi-GPU, slow gate, GCC version). NOT for skips that are stale or fixable. |
| `Not Appliable` | CUDA-only API with no XPU equivalent. `DetailReason` MUST name the exact API. |
| `Not applicable` | Test removed/renamed upstream. `DetailReason` = `Community Changes`. |

### `DetailReason` (String)

Must be specific enough to act on:
- Include issue links when known: `https://github.com/intel/torch-xpu-ops/issues/NNNN - description`
- Include `[Issue TBD]` when no issue exists after searching
- Name exact APIs for `Not Appliable`: `CUDA-specific API: torch.cuda.jiterator (jiterator_binary)`
- Name exact evidence for `Local Passed`: `Local verification passed in pytorch_opencode_env; stale failed status`

## Deep Analysis Requirements (CRITICAL)

**DO NOT use simple pattern matching or regex scripts for classification.**

Every blank-Reason row requires:

1. **Source inspection**: Read the test source to understand what it validates
2. **Local verification**: Run the test when status is ambiguous (failed/skipped with unclear message)
3. **Known issue search**: `gh search issues` on `intel/torch-xpu-ops` for relevant keywords
4. **Issue state check**: `gh issue view` for any referenced issue to confirm OPEN/CLOSED
5. **Evidence recording**: Save local run output to files; record in `Explaination`

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
7. If fails: `Failures (xpu broken)` with issue link or `[Issue TBD]`

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

## Workflow

1. Prepare the requested `pytorch_opencode_env` environment.
2. Open the target sheet in the workbook.
3. Ensure workbook columns `Explaination` and `Reason TBD` exist.
4. Initialize `Reason TBD` from the current `Reason` value:
   - if `Reason` is blank, set `Reason TBD = True`
   - otherwise set `Reason TBD = False`
5. If XPU test metadata is blank, derive it from CUDA metadata:
   - `classname_cuda` ending with `CUDA` -> replace with `XPU`
   - `name_cuda` ending with `_cuda` -> replace with `_xpu`
   - `testfile_xpu` defaults to `testfile_cuda` when blank
6. For each blank-Reason row, choose the status-specific skill:
   - blank `status_xpu` -> `classify_ut/blank/SKILL.md`
   - `status_xpu = failed` -> `classify_ut/failed/SKILL.md`
   - `status_xpu = skipped` or `xfail` -> `classify_ut/skipped/SKILL.md`
7. Execute the selected skill's deep analysis workflow.
8. Fill `Reason`, `DetailReason`, and `Explaination`. Mark cells blue.
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

- `Explaination` keeps the requested spelling.
- Save output as `.agent.xlsx`; do not modify original workbook.
- Preserve existing `Reason`, `DetailReason`, and `Explaination` unless deep analysis justifies an update.
- Mark updated cells blue using `PatternFill(start_color='ADD8E6', end_color='ADD8E6', fill_type='solid')`.
