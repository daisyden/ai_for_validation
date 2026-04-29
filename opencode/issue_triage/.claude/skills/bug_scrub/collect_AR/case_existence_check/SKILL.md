# case_existence_check (Phase 4c)

## Overview

Phase 4c of the bug_scrub workflow. Cross-references the
`xpu_case_existence` data produced by Phase 2.4
(`analyze_ci_result/check_xpu_case_existence/`) and emits an AR entry on
the parent issue whenever at least one of its test cases could not be
found on the XPU side.

This step is mechanical — no LLM / explore-agent reasoning is involved.
All classification was already done in Phase 2.4; this skill only
aggregates per-issue and writes to the Issues sheet.

## Position in the Workflow

```
Phase 4: Collect AR
    4a close_or_skip
        ↓
    4b get_AR_from_issue (+ check_pr_status)
        ↓
    4c case_existence_check   ← THIS SKILL
```

Phase 4c runs last. It appends to `action_TBD` / `owner_transferred` on
issues that were not already closed/skipped in 4a and that carry at
least one missing case.

## Preconditions

- Phase 2.4 must have run: `Test Cases` sheet column
  `xpu_case_existence` is populated (True / False / blank) for every
  row Phase 2.4 examined.
- Issues sheet has a populated `Reporter` column (used as the owner).

## Inputs

| Source | Column(s) Used |
|---|---|
| `Test Cases` sheet | `Issue ID`, `xpu_case_existence`, `case_existence_comments` |
| `Issues` sheet | `Issue ID`, `Reporter`, existing `action_TBD`, `owner_transferred` (if already populated by 4a / 4b) |

File: `../../../../result/torch_xpu_ops_issues.xlsx` (relative from this
SKILL.md; resolved absolute path is
`opencode/issue_triage/result/torch_xpu_ops_issues.xlsx`).

## Outputs

Columns on the **Issues** sheet:

| Column | Value Written |
|---|---|
| `action_TBD` | append `"check_case_avaliablity"` |
| `owner_transferred` | set to the issue's `Reporter` |
| `action_reason` | distinct non-empty `case_existence_comments` aggregated across the issue's Test Cases rows. Single comment → plain string; multiple distinct comments → JSON array. Only written when `action_reason` is blank (does not overwrite values produced by Phase 4a / 4b). |

`action_TBD` is a delimited list of tokens (comma-separated). If the
column already has content from Phase 4a / 4b, append rather than
overwrite.

## Rule

For each issue `I` in the Issues sheet:

1. Gather `I`'s rows in the Test Cases sheet (match by `Issue ID`).
2. If **any** such row has `xpu_case_existence == False`:
   - Append `check_case_avaliablity` to `I.action_TBD`
     (de-duplicate — only add if not already present).
   - Set `I.owner_transferred` to `I.Reporter`
     (if `owner_transferred` already has a different value from 4a / 4b,
     union with the reporter so both owners are preserved).

If no test case for the issue has `xpu_case_existence == False`, do
nothing for that issue in this phase.

### Precedence: `check_case_avaliablity` overrides `No action — investigate further`

After the append step, do a final pass over every row whose
`action_TBD` contains both `check_case_avaliablity` and
`No action — investigate further`: drop the latter. Rationale: the
case-existence question must be resolved before any "investigate
upstream" verdict is meaningful — a missing test case cannot be
investigated further until its identity is verified or fixed. Phase 4b
emits `No action — investigate further` only when its 6-vector PR
search comes up empty, which is irrelevant once the test case itself is
in question.

`action_reason` is **not** modified — the Phase 4b PR-discovery
narrative is preserved so the future investigator still has that
context.

Note on spelling: the token written is literally
`check_case_avaliablity` (as specified by the workflow owner). Do not
correct the spelling when writing — downstream tooling reads this exact
string.

## Execution (Python sketch)

```python
import openpyxl
from collections import defaultdict

EXCEL = "opencode/issue_triage/result/torch_xpu_ops_issues.xlsx"
TOKEN = "check_case_avaliablity"

wb = openpyxl.load_workbook(EXCEL)
issues = wb["Issues"]
cases  = wb["Test Cases"]

# Header lookup
def col_idx(ws, name):
    return [c.value for c in ws[1]].index(name)

ci_issue_id   = col_idx(cases, "Issue ID")
ci_existence  = col_idx(cases, "xpu_case_existence")

# Aggregate: issue_id -> has at least one missing case
missing = defaultdict(bool)
for row in cases.iter_rows(min_row=2, values_only=True):
    iid = row[ci_issue_id]
    val = row[ci_existence]
    # Treat only strict False as "missing"; blanks/True are ignored.
    if val is False or (isinstance(val, str) and val.strip().lower() == "false"):
        missing[iid] = True

ii_id       = col_idx(issues, "Issue ID")
ii_reporter = col_idx(issues, "Reporter")
# action_TBD / owner_transferred are Phase 4 columns — create if absent
def ensure_col(ws, name):
    headers = [c.value for c in ws[1]]
    if name in headers:
        return headers.index(name)
    ws.cell(row=1, column=ws.max_column + 1, value=name)
    return ws.max_column - 1

ii_action = ensure_col(issues, "action_TBD")
ii_owner  = ensure_col(issues, "owner_transferred")

for row in issues.iter_rows(min_row=2):
    iid = row[ii_id].value
    if not missing.get(iid):
        continue

    # Append token to action_TBD (dedupe)
    cur = (row[ii_action].value or "").strip()
    tokens = [t.strip() for t in cur.split(",") if t.strip()]
    if TOKEN not in tokens:
        tokens.append(TOKEN)
    row[ii_action].value = ", ".join(tokens)

    # Union reporter into owner_transferred
    reporter = (row[ii_reporter].value or "").strip()
    cur_own  = (row[ii_owner].value or "").strip()
    owners = [o.strip() for o in cur_own.split(",") if o.strip()]
    if reporter and reporter not in owners:
        owners.append(reporter)
    row[ii_owner].value = ", ".join(owners)

# Back up before write (per project convention)
import shutil; shutil.copy(EXCEL, EXCEL.replace(".xlsx",
                                                "_bk_before_phase4c.xlsx"))
wb.save(EXCEL)
```

## Validation

After running, spot-check a handful of issues:

```python
# Example: confirm issues with any False xpu_case_existence now carry
# the token.
for iid in list(missing)[:10]:
    row = next(r for r in issues.iter_rows(min_row=2)
               if r[ii_id].value == iid)
    assert TOKEN in (row[ii_action].value or "")
```

Expected post-run row counts (sanity): number of Issues rows with the
token equals `len(missing)` from the aggregation step above.

## Backup Policy

Before writing, copy the Excel to
`result/torch_xpu_ops_issues_bk_before_phase4c.xlsx`. This matches the
backup convention used by Phase 3 (`_bk_before_phase3_write.xlsx`,
`_bk_before_category_normalize.xlsx`, etc.).

## Non-Goals

- Does not re-verify `xpu_case_existence`; trusts Phase 2.4's values.
- Does not produce any per-issue narrative; the token is a flag that
  downstream triage tooling expands.

## Scripts (in this folder)

| Script | Purpose |
|---|---|
| [`run_action_reason_backfill.py`](./run_action_reason_backfill.py) | For issues whose `action_TBD` contains `check_case_avaliablity` and whose `action_reason` is blank, aggregate distinct non-empty `case_existence_comments` from the Test Cases sheet and write them into `action_reason` (single → plain string, multiple → JSON array). Backs up the workbook to `_bk_before_action_reason_backfill.xlsx` before writing. Anchored via `__file__` so it runs from any CWD. |

Typical run:

```bash
python3 opencode/issue_triage/.claude/skills/bug_scrub/collect_AR/case_existence_check/run_action_reason_backfill.py
```

## Version

- v1.1.0 — 2026-04-22 — populate `action_reason` from `case_existence_comments`
  (see `run_action_reason_backfill.py`); clarified Outputs + Scripts sections.
- v1.0.0 — 2026-04-21 — initial skill.
