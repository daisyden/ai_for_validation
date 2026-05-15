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
| `Not applicable` | Either (a) a CPU-only test not relevant to XPU validation (`DetailReason` = `CPU Case`), or (b) a CUDA-only / out-of-scope behavior covered by an issue in the `Issues` sheet of `${ISSUE_TRIAGE_ROOT}/result/torch_xpu_ops_issues.xlsx` whose `Labels` column contains `not_target` OR a `wontfix` variant. `DetailReason` MUST name the exact API/feature (e.g., `CUDA-specific API: torch.cuda.jiterator`), cite the matching `Issue ID`, and quote the deciding label. Never use generic "CUDA-only test" or "No XPU test data" — always identify the specific API or feature that XPU does not support. See the **CUDA-Only Judgement Rule** below. |
| `Community Change` | The base function/case no longer exists in the source being compared, was renamed/refactored/moved, or the test is disabled by an upstream PyTorch community issue/commit. `DetailReason` MUST include the full issue/PR URL when available, or exact source/commit evidence. Do not require `last_status_xpu = passed`; base-function absence is enough. |
| `Need human check` | Deep analysis was performed but no category could be assigned with HIGH or MEDIUM confidence (see **Confidence Rubric & Need-Human-Check Rule** below). Only valid when `Reason TBD = True`. `DetailReason` MUST start with `[Confidence: LOW]` and enumerate which signals were checked and why each was inconclusive (no `not_target`-label match, base function ambiguous, no xpu wrapper located, no known issue in either repo, etc.). Never use this label as a shortcut to skip analysis — it is the explicit outcome of a thorough but inconclusive investigation. |

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

The **only** authoritative source for "CUDA-only behavior with no XPU equivalent" is an
issue in the reference workbook tagged with either of two equivalent label markers:

- `not_target` — explicitly out of XPU scope (the original CUDA-only marker)
- `wontfix` / `won't fix` / `won_t_fix` — XPU side has decided not to address the issue,
  which is treated as equivalent to "not in XPU scope" for classification purposes

```
${ISSUE_TRIAGE_ROOT:-$HOME/opencode/ai_for_validation/opencode/issue_triage}/result/torch_xpu_ops_issues.xlsx
                                                                                              └─ sheet: "Issues"
                                                                                                 └─ filter: Labels contains "not_target" OR matches /won.?t.?fix/i
```

(Historical note: the standalone `Not applicable` sheet that previously held these entries
has been consolidated into the `Issues` sheet via labels. The misspelled `Not Appliable`
sheet was first renamed to `Not applicable`, then merged. All CUDA-only judgements now flow
through label-tagged issues. Cross-reference via the `Test Cases` sheet to locate specific
test bindings.)

### How to use the labels

1. Open the workbook with `openpyxl` and load sheet `Issues`.
2. The sheet columns are:
   `Issue ID | Title | Status | Assignee | Reporter | Labels | Created Time | Updated Time | Milestone | Summary`.
3. To judge whether a row's failing/skipped API is CUDA-only, filter rows where the
   `Labels` cell contains EITHER `not_target` OR any `wontfix` variant
   (case-insensitive, allowing `wontfix`, `won't fix`, `won_t_fix`, `wont fix`).
   Then match the issue's `Title` / `Summary` against the test in question by:
   - exact op or test name in the title (e.g., `Support efficient attention`,
     `RuntimeError: expected scalar type Half but found Float`)
   - explicit test-name reference inside `Summary`
   - cross-lookup via the `Test Cases` sheet (`Test Case` column) — joining
     `Test Cases.Issue ID` -> `Issues.Issue ID` filtered by the labels above
4. If a matching `not_target`- or `wontfix`-labeled issue exists -> the behavior IS
   CUDA-only / out-of-scope -> classify `Not applicable`, and in `DetailReason` cite the
   matching `Issue ID` and the deciding label verbatim:
   `CUDA-specific behavior covered by not_target Issue #2285 ("Support efficient attention")`,
   or `Issue #NNNN labeled wontfix - XPU will not address`.
5. **If no matching `not_target`/`wontfix` issue exists, the behavior is NOT CUDA-only.**
   Do not classify `Not applicable` on the CUDA-only branch. Re-route to the correct label:
   - device-agnostic test that XPU should support -> `To be enabled`
   - XPU implementation bug -> `Failures (xpu broken)`
   - missing XPU feature -> `Feature gap`
   - base function removed/renamed/refactored upstream -> `Community Change`
   - test is CPU-only (`_cpu` suffix, `requires GPU`, etc.) -> `Not applicable` (CPU branch),
     `DetailReason = CPU Case`
6. The CPU branch of `Not applicable` (`DetailReason = CPU Case`) does NOT require a
   label match and is unaffected by this rule.

### Workflow snippet

```python
import openpyxl, os, re
xlsx = os.path.expanduser(
    os.environ.get("ISSUE_TRIAGE_ROOT", "~/opencode/ai_for_validation/opencode/issue_triage")
) + "/result/torch_xpu_ops_issues.xlsx"
wb = openpyxl.load_workbook(xlsx, read_only=True, data_only=True)
ws = wb["Issues"]
header = [c.value for c in next(ws.iter_rows(min_row=1, max_row=1))]
id_col = header.index("Issue ID")
title_col = header.index("Title")
label_col = header.index("Labels")
WONTFIX = re.compile(r"won.?t.?fix", re.I)
not_applicable_issues = [
    (row[id_col].value, row[title_col].value, row[label_col].value)
    for row in ws.iter_rows(min_row=2)
    if row[label_col].value and (
        "not_target" in str(row[label_col].value).lower()
        or WONTFIX.search(str(row[label_col].value))
    )
]
```

The matched issue must still be cross-referenced against the test by API/title; the label
makes the issue eligible to anchor a `Not applicable` verdict but does not by itself prove
the candidate test is the one the issue describes.

### Anti-patterns

- Marking a row `Not applicable` because `name_cuda` contains the substring `cuda`.
- Marking a row `Not applicable` because the skip message says `Only runs on cuda` without
  confirming the underlying behavior is covered by a `not_target`- or `wontfix`-labeled
  issue (many such gates are SM-capability gates or stale skips -> `To be enabled`).
- Inventing a CUDA-only judgement locally instead of opening / labeling an issue with
  `not_target` (or having an existing issue moved to `wontfix`) first. If you genuinely
  discover new CUDA-only behavior, open an issue in `intel/torch-xpu-ops` and apply the
  `not_target` label before classifying.
- Treating `wontfix` as a generic "won't fix this bug report" without checking that the
  issue actually scopes the API/test as out-of-scope on XPU. `wontfix` on a narrow
  reproducer that the team simply triaged away does NOT make the entire op CUDA-only.
- Citing the `Not applicable` sheet — that sheet no longer exists. Use the `not_target`
  or `wontfix` labels on the `Issues` sheet.

## Dynamic-Skip Rule (`skipped` label semantics)

The `skipped` label on an `Issues`-sheet row means the test (or test family) is currently
dynamically skipped on XPU because some issue is preventing it from running successfully —
**the test cases exist and would execute, but a skip decorator or runtime gate stops them**.
This is NOT a CUDA-only judgement; it is an XPU-side problem that needs to be reclassified
as either `Failures (xpu broken)` or `Feature gap` based on what the skip is actually
hiding.

### Decision flow

When a candidate test maps to an issue with the `skipped` label (and NO `not_target` /
`wontfix` label — those override and yield `Not applicable`):

1. **Local verify is REQUIRED.** Do not classify on issue body keywords alone. The issue
   describes WHY the case is skipped; you still need to run it with the skip lifted to
   know whether it is a broken kernel (Failures) or a missing feature (Feature gap).
2. Port / enable the case via the `port-pytorch-tests-xpu` workflow (see the
   "Local Verification via XPU Port" section below) and run it.
3. Read the actual failure mode of the executed run:
   - Crash, accuracy mismatch, hang, `RuntimeError` from an existing XPU kernel, or any
     other defect inside an XPU operator that exists -> `Failures (xpu broken)`.
     `DetailReason` must cite the `skipped`-labeled issue AND describe the observed
     failure mode from the local run.
   - `NotImplementedError`, "operator not implemented for 'XPU'", "no XPU kernel for ...",
     dispatch miss, an explicit `aten::<op>` registration gap, or a clear "this feature
     does not exist on XPU yet" signal -> `Feature gap`.
     `DetailReason` must cite the `skipped`-labeled issue AND name the missing
     operator/feature observed locally.
   - Test passes after the skip is lifted -> `To be enabled` (the skip is stale and just
     needs to be removed). `DetailReason` cites both the `skipped`-labeled issue and the
     successful local-run artifact.
4. **Same `skipped`-labeled issue → same verdict for every test in its scope.** Cluster
   the rows by their referenced issue and apply the verdict from one local run to all
   siblings, citing the same evidence.

### Anti-patterns

- Classifying a `skipped`-label row as `Not applicable`. `skipped` is an XPU-side gate,
  not a CUDA-only scope decision.
- Picking `Failures (xpu broken)` vs `Feature gap` from issue text alone when local
  verification is feasible. The two labels imply very different remediation paths
  (fix kernel vs. implement feature) and the wrong one misleads the human reviewer.
- Skipping the local run because "the issue title says feature". Issue titles drift;
  the runtime behavior is what determines the label.

## Local Verification via XPU Port (`To be enabled` and `skipped`-label cases)

When a row's verdict depends on actually running the case on XPU — specifically:

- Any `To be enabled` row where the goal is to confirm the test works once XPU is
  enabled (no tracked issue or only stale skip information).
- Any `skipped`-label row that needs the Failures-vs-Feature-gap split (see above).

…the local-run **MUST** be performed via the standard porting workflow, not via ad-hoc
device monkey-patches inside the workspace:

1. Load and follow the `port-pytorch-tests-xpu` skill at
   `${ISSUE_TRIAGE_ROOT}/.claude/skills/unittest_dev/port-pytorch-tests-xpu/SKILL.md` for
   the exact porting recipe (direct copy vs. `XPUPatchForImport` hook, hook-body parity
   rules, instantiation pattern). That skill is authoritative for HOW to port.
2. Do the porting work in a **new branch on `daisyden/pytorch`** (NOT in upstream
   `pytorch/pytorch`, NOT directly in `~/upstream/pytorch`). Branch from the same base
   commit that `~/upstream/pytorch` is synced to so the port mirrors current upstream.
3. Save the executed-test stdout/stderr (and any artifact paths) under
   `/tmp/opencode/<workbook>_local_verify/<case>.log`. This is the path that the
   Confidence Rubric recognizes as a HIGH-evidence "local verification artifact".
4. Use the run outcome to assign the Reason per the matrix below:
   - PASS -> `To be enabled` HIGH (cite log path).
   - FAIL with crash/accuracy/runtime error on an existing XPU kernel ->
     `Failures (xpu broken)` HIGH (cite log path + observed error).
   - FAIL with `NotImplementedError` / dispatch miss / "no XPU kernel" ->
     `Feature gap` HIGH (cite log path + missing op).
   - Test is CPU-only or genuinely not portable -> reclassify via the regular rules,
     do NOT use a port as the gate (porting CPU-only tests is a category error).
5. The DetailReason MUST cite the `daisyden/pytorch` branch name AND the log artifact
   path, e.g.:
   `[Confidence: HIGH] To be enabled. Ported via daisyden/pytorch branch
   xpu-port-test-foo; local run /tmp/opencode/Non_inductor_ut_status_ww18_26_v2.agent_local_verify/test_foo.log shows PASS.`

### Anti-patterns

- Running a one-off `python test_foo.py -k case` inside the workspace's PyTorch checkout
  and treating that as the local-verification artifact. Use the porting workflow so the
  run is reproducible and reviewable through a `daisyden/pytorch` branch.
- Reusing an old log from a prior run that did not actually exercise the case after
  the skip was lifted. Each local-verification claim must point at a fresh run executed
  for this classification.
- Pushing the port to `pytorch/pytorch` or to `intel/torch-xpu-ops` directly. Local
  verification ports stay on `daisyden/pytorch` until the human review approves promotion.

## Confidence Rubric & Need-Human-Check Rule (authoritative for `Reason TBD = True` rows)

Every row with `Reason TBD = True` MUST have its `DetailReason` prefixed with a confidence
tag reflecting the strength of the evidence behind the assigned `Reason`:

```
[Confidence: HIGH]    DetailReason cites STRONG EVIDENCE: a tracked issue link (intel/torch-xpu-ops or pytorch/pytorch issue URL / `#NNNN`) OR a local verification result (test actually executed, output saved under /tmp/opencode/<workbook>_local_verify/, or a cited file path + line range that directly proves the verdict).
[Confidence: MEDIUM]  Best-fit category from a deep analysis, but DetailReason does NOT cite an issue link or a local verification result (e.g., reasoning from base-function existence + XPU instantiation patterns without a tracked issue or local run).
[Confidence: LOW]     Analysis performed but no axis yields a confident category -> Reason = "Need human check". Escape hatch only; never a shortcut for skipping work.
```

**HIGH evidence is binary**: either DetailReason contains a verifiable issue reference (URL or `#NNNN` cross-checked via `gh issue view`) OR it cites a concrete local-verification artifact (executed test output OR source file path + line range that proves the claim, e.g. an in-test skip at a specific line). If DetailReason contains neither, the row is MEDIUM regardless of how confident the analyst feels.

The prefix is REQUIRED for every `Reason TBD = True` row, regardless of which `Reason` is assigned.
Rows with `Reason TBD = False` keep their existing `DetailReason` untouched and do NOT need this
prefix (the original Reason was authoritative).

### Decision axes (check ALL that apply per row)

For each `Reason TBD = True` row, evaluate these axes from real sources, not pattern matching:

| Axis | HIGH signal | MEDIUM signal | LOW signal |
|------|-------------|---------------|------------|
| **CUDA-Only label match** | Exact `Operation/API` match in an `Issues`-sheet row whose `Labels` contains `not_target` OR a `wontfix` variant, with cited `Issue ID` | Plausible API mentioned by such an issue but match is by family/parent, not exact | No matching labeled entry, yet the test clearly uses a CUDA-only API in source |
| **Base function in `$PYTORCH_SRC`** | Located in `<testfile_cuda>` with file path + line range cited | Located in a refactored/renamed location; need user confirmation | Not found after thorough search of PyTorch + `third_party/torch-xpu-ops` |
| **XPU wrapper / generated case** | `_xpu`-suffixed case located in `test/xpu/**` or via `instantiate_device_type_tests(..., allow_xpu=True)` with file cited | XPU instantiation present but the specific case is not yet generated (e.g., dtype/OpInfo filter ambiguity) | No XPU wrapper / no `allow_xpu` / device list excludes XPU |
| **Known issue (both repos)** | Open/closed issue in `intel/torch-xpu-ops` or `pytorch/pytorch` with URL cited and verified via `gh issue view`; for `skipped`-labeled issues the verdict is anchored by a local-run artifact (see below) | Issue mentioned in skip decorator but URL not verified | No issue found after `gh search issues` on both repos with multiple keyword variations |
| **Local verification** | Case ported via `port-pytorch-tests-xpu` on a `daisyden/pytorch` branch and executed, with stdout/stderr saved to `/tmp/opencode/<workbook>_local_verify/<case>.log` (cite both branch name and log path) | Partial run / timeout / interpreted output without an artifact path | Not run |
| **Source-evidence for `Community Change`** | Guilty commit hash + author + date + diff that explains the regression | `git log` shows candidate commits but causal link not proven | No relevant commit found |

### Assigning a confidence level

- **HIGH** — DetailReason cites at least ONE of:
  - A tracked issue: `https://github.com/intel/torch-xpu-ops/issues/NNNN`,
    `https://github.com/pytorch/pytorch/issues/NNNN`, or a bare `#NNNN` cross-checked
    via `gh issue view NNNN --repo <owner>/<repo>`.
  - A local verification artifact: an executed-test result (path under
    `/tmp/opencode/<workbook>_local_verify/`), OR a concrete source-code citation
    (`<file>#L<start>-L<end>`) that directly proves the verdict (e.g. an in-test
    `skipIfXpu` / `unittest.skip` at the cited line; a `not_target`-labeled issue
    matched by `Operation/API`; a guilty commit hash + diff).
- **MEDIUM** — Deep analysis assigns a best-fit category, but DetailReason does NOT
  cite an issue link or a local verification result. Typical case: base function
  exists, XPU instantiation is present in `test/xpu/**`, but the specific failure /
  skip / dtype slice is not pinned to a tracked issue or a cited source line.
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

- **HIGH**: cite the decisive evidence inline — at least one of:
  - Issue reference: full URL OR `#NNNN` (with the repo unambiguous from context).
  - Local-verification artifact: `/tmp/opencode/<workbook>_local_verify/<file>` path,
    OR a source-code citation `<file>#L<start>-L<end>` that directly proves the claim.

  Examples:
  `[Confidence: HIGH] Failures (xpu broken). PVC fp16 accuracy issue in test_Conv2d_naive_groups - https://github.com/intel/torch-xpu-ops/issues/3346.`
  `[Confidence: HIGH] To be enabled. In-test skip at $PYTORCH_SRC/test/test_transformers.py#L5165 ('_fill_mem_eff_dropout_mask too many threads') fires on both cuda and xpu; XPU wrapper at third_party/torch-xpu-ops/test/xpu/test_transformers_xpu.py uses allow_xpu=True.`

- **MEDIUM**: deep analysis assigns a best-fit category, but no issue link and no
  local-verification artifact is available. State the best-fit reasoning AND the
  evidence gap. Example:
  `[Confidence: MEDIUM] To be enabled. Base test_foo found at .../test_x.py; XPU instantiation present in test/xpu/test_x_xpu.py; no tracked issue identified and local run not attempted.`

- **LOW** (→ `Need human check`): enumerate which axes were checked and why each was
  inconclusive. Example:
  `[Confidence: LOW] Need human check. not_target label: no matching issue for torch.foo.bar in Issues sheet. Base function: candidate test_foo present but signature changed (device-agnostic refactor in commit abc1234, behavior on XPU unclear). XPU wrapper: not located in test/xpu/**. Known issues: gh search on 'test_foo xpu' returned 0 results in both intel/torch-xpu-ops and pytorch/pytorch. Local run not attempted (test requires multi-GPU).`

### Workflow integration

- The status-specific subskills (`blank/`, `failed/`, `skipped/`) MUST emit a confidence level
  alongside their `Reason` decision for every `Reason TBD = True` row.
- When saving `.agent.xlsx`, blue-fill cells per the existing convention. NEVER modify
  `Reason TBD`.
- For LOW rows, `Reason` becomes `Need human check` and the original best-guess (if any) is
  preserved in `DetailReason` after the `[Confidence: LOW]` prefix, so a human reviewer can see
  what was considered.

### `Confidence` column (workbook schema)

In addition to the `[Confidence: ...]` prefix inside `DetailReason`, every `.agent.xlsx`
workbook MUST carry a dedicated `Confidence` column immediately AFTER `Reason TBD`:

```
... | Reason | DetailReason | Reason TBD | Confidence | ...
       col22   col23          col24        col25
```

- Allowed values: `HIGH`, `MEDIUM`, `LOW` (uppercase, no other strings).
- Empty when `Reason TBD = False` (the legacy authoritative path doesn't need it).
- Required when `Reason TBD = True` — and MUST match the level in the `DetailReason` prefix.
- Blue-filled like other agent-written cells.

Purpose: enables Excel filtering / pivot-table aggregation without parsing the
`DetailReason` text. The `DetailReason` prefix remains the human-readable single source of
truth (full evidence); the column is the machine-readable index. If the two disagree, the
prefix wins and the column must be corrected.

Schema check before saving:

```python
header = [ws.cell(row=1, column=c).value for c in range(1, ws.max_column + 1)]
assert header[23] == "Reason TBD" and header[24] == "Confidence", \
    f"Confidence column missing or misplaced: got {header[23:25]}"
```

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

## Refinement Rules (Deep Analysis — NOT Pattern Matching)

This section captures **post-initial-classification refinement** rules learned from
WW18-26 review. A row may have been written with one verdict in the first pass and
require revision once deeper evidence is examined. All three rules below are
**deterministic deep analysis** — they require reading upstream source, parsing the
NA workbook, and querying issue labels. **Never substring-match `message_xpu` alone.**

Apply these passes IN ORDER, after the initial classification pass, before final save.

### Rule R1 — SDPA / Dynamic-Skip Cross-Backend Check (TEL → Community Change)

**Trigger**: row currently classified `Test Enviroment limitation` AND `message_xpu`
indicates a dynamic `skipTest(...)` from the test body (not a `@skipIf` decorator
gated on backend availability).

**Naive failure mode**: classifying as `Test Enviroment limitation` based purely on
the skip string ("Will call _fill_mem_eff_dropout_mask with too many threads!",
"Reference: ...", etc.). This is wrong when **CUDA also skips dynamically at the
same source line** — that proves the skip is a community-added guard, not an XPU
environment gap.

**Deep-analysis procedure** (deterministic, no LLM):

1. Identify the XPU wrapper that owns the row (e.g., `test_transformers_xpu.py`).
2. Extract the corresponding upstream test path (strip `_xpu` and resolve to
   `${PYTORCH_SRC}/test/<name>.py`).
3. `grep -n "skipTest" <upstream_path>` and locate every dynamic-skip line whose
   message matches the row's `message_xpu` (after stripping device tokens).
4. For each match, read ±15 lines of context and determine whether the guard fires
   for the row's parameter values (e.g., `seq_len_q > 1024`, dtype, SM capability).
   Compare against the row's `name_cuda` parameter encoding.
5. If the upstream guard fires for CUDA with the same parameters AND the message is
   identical → **the community added this skip cross-backend** → reclassify as
   **Community Change**. DetailReason MUST cite:
   - upstream file path with `:LINE` (e.g., `test/test_transformers.py:3860`)
   - introducing PR(s) found via `git log -L <line>,<line>:<file>` or
     `git blame -L <line>,<line> <file>`
   - the matching parametrize line if the skip depends on `@parametrize` values
6. If only XPU skips and CUDA runs → keep `Test Enviroment limitation` OR re-route
   to `Failures (xpu broken)` / `Feature gap` per the **Dynamic-Skip Rule**.

**Worked precedent (WW18-26)**: 576 mem_eff_attention rows reclassified TEL →
Community Change. Evidence: upstream `test/test_transformers.py:3860` and `:3978`
both contain identical `skipTest("Will call _fill_mem_eff_dropout_mask with too
many threads!")`; parametrize at `:3795` instantiates `seq_len_q=2048` for SM80+
CUDA, and all TBD rows use `seq_len_q=2048`. Lineage PRs #102038, #103704, #133049.

### Rule R2 — Strict NA Evidence Rule (NA → Human Investigation when unsupported)

**Trigger**: row currently classified `Not applicable` (any source).

**Naive failure mode**: keeping `Not applicable` because `name_cuda` contains
`cuda`, or because the test wasn't found in the XPU collection, or because skip
text says "Only runs on cuda" without a referenced issue. None of these are
sufficient evidence.

**Authoritative evidence sources** — `Not applicable` requires AT LEAST ONE:

- **E1**: `message_xpu` contains a documented skip string from a recognized NA
  family. Allowed families (must match exactly, after stripping device tokens):
  - `Only runs on cuda` / `requires CUDA` (jiterator, cuDNN-only kernels)
  - `TypedStorage is deprecated and not available on XPU`
  - `Efficient or cuDNN Attention was not built for this system`
  - `CUDA-specific API: <api>` style messages that name a CUDA-only symbol
  - Any other family pre-listed in the NA workbook (see E2)
- **E2**: An entry in `${ISSUE_TRIAGE_ROOT}/result/torch_xpu_ops_issues.xlsx`,
  sheet `Not Appliable` (sic) or `Not applicable`, whose `Test Case` /
  `Test Name` column matches the row's nodeid (full or stem match) OR whose
  `Issue ID` is referenced from the row's matched issues.
- **E3**: A matched issue (from `Issues` sheet of the same workbook, or
  `gh issue view <repo>#<id>`) whose `Labels` column contains `not_target` OR a
  `wontfix` variant (`won't fix`, `wontfix`, `wont-fix`, `wont_fix`). Both
  `intel/torch-xpu-ops` and `pytorch/pytorch` are valid repos.

**Deep-analysis procedure** (deterministic, no LLM):

1. Load NA workbook once: parse "Not Appliable" / "Not applicable" sheets to a
   map keyed by normalized nodeid → `{issue_id, justification}`. Cache as
   `na_entries.json` for reuse.
2. For each NA row:
   a. Strip device suffix (`_xpu`, `_cuda`) and parametrize tokens to a canonical
      nodeid form for matching.
   b. Check E1: scan `message_xpu` against the allowed-family list above. Record
      which family matched.
   c. Check E2: lookup canonical nodeid in NA workbook map.
   d. Check E3: for each issue in the row's `_match_iids`, fetch labels (cached
      `issues_map.json` or live `gh issue view`); record any `not_target` /
      `wontfix` hits.
3. If ANY of E1/E2/E3 produced evidence → keep `Not applicable`. DetailReason
   MUST cite the specific evidence (exact skip string OR `NA workbook row #N
   (Issue #NNNN)` OR `Issue #NNNN label='not_target'`).
4. If NONE → reclassify as **Human Investigation**. DetailReason MUST state
   *which* evidence sources were checked and that none matched, e.g.:
   `[Confidence: MEDIUM] Human Investigation. No NA evidence: message_xpu empty;
   not in NA workbook 'Not Appliable' sheet; matched issue #2618 has labels
   ['module: sdpa'] (no not_target/wontfix).`

**Worked precedent (WW18-26)**: 17 of 31 initial NA rows reclassified → Human
Investigation. They were missing from XPU collection (parametrize variants not
instantiated) rather than runtime-skipped with documented justification, so no
E1/E2/E3 evidence existed. The 14 kept-NA rows each cite a specific skip string
from the E1 allowed-family list.

**Common rejections** (do NOT use these as evidence):
- `name_cuda` substring contains `cuda` → not evidence.
- Test absent from XPU wrapper collection → that's a port gap, route to
  `To be enabled` or Human Investigation, never NA.
- Skip text "Only runs on cuda" present but no NA-workbook / `not_target`
  backing → still NA only if message exactly matches the E1 family AND the
  test exercises a clearly CUDA-only API (jiterator, cuDNN). Generic "cuda"
  mentions without a CUDA-only API → Human Investigation.

### Rule R3 — `Failures` → `Failures (xpu broken)` Relabel

**Trigger**: row classified with bare `Failures` Reason.

**Procedure**: rewrite the `Reason` cell to the canonical label
`Failures (xpu broken)`. No change to `DetailReason` or `Confidence`. This is a
pure relabel pass — apply across the whole sheet after R1 and R2 so any newly
created `Failures` verdicts also get normalized.

The canonical label is fixed by the **Column Definitions** table above; the bare
`Failures` form is reserved for legacy compatibility only and MUST NOT survive
into a final `.agent.xlsx`.

### Refinement Pass Order & Re-Verification

Run refinement in this fixed order:

1. **R1 (SDPA / dynamic-skip)** — may convert TEL → Community Change for many
   rows; do this first because it shrinks the set of rows that downstream rules
   need to inspect.
2. **R2 (strict NA evidence)** — may convert NA → Human Investigation; depends
   only on NA workbook + issue labels, independent of R1's output.
3. **R3 (Failures relabel)** — last, so it normalizes any `Failures` written by
   R1 or R2.

After each rule:

- Re-fill `Confidence` per the rubric (HIGH iff DetailReason cites
  issue # / URL / log path / `file.py:NNN`; otherwise MEDIUM).
- Blue-fill (`ADD8E6`) the changed cells. Preserve `Reason TBD` untouched.
- Read back the workbook and assert: 0 blank `Reason` rows in the
  TBD-true population, distribution matches in-memory verdict map.

Cache the per-rule diff so a reviewer can audit: which rows changed, from what,
to what, citing which evidence record.

## Notes

- Save output as `.agent.xlsx`; do not modify original workbook.
- Preserve existing `Reason` and `DetailReason` unless deep analysis justifies an update.
- Mark updated cells blue using `PatternFill(start_color='ADD8E6', end_color='ADD8E6', fill_type='solid')`.
