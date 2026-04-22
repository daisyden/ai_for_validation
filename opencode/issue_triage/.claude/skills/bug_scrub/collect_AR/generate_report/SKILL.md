# Generate Report Skill

## Overview

Final stage of the bug-scrub pipeline. After Phases 1–4 have populated the
Issues sheet of `result/torch_xpu_ops_issues.xlsx` with per-issue AR
(`action_TBD`, `action_reason`, `owner_transferred`) and triage fields
(`Category`, `Priority`, `Dependency`, `Root Cause`, `Fix Approach`), this
skill:

1. Classifies each row's free-text `action_TBD` into a structured
   `action_Type` column (leaf categories joined by `+` in priority order).
2. Renders the human-readable `result/bug_scrub.md` report — sections
   grouped by action_Type, with cross-references, duplicate detection, and
   a back-to-index anchor per section.

## When to Use

Run only after **Phase 4c** (case_existence_check) has completed and the
Issues sheet is stable. Re-running is idempotent — both scripts overwrite
their outputs cleanly.

---

## action_Type Taxonomy (17 leaf categories, priority order)

The `action_Type` column is the sort/group key for the generated report.
When a row's `action_TBD` maps to multiple leaves, they are joined with
`+` in this fixed order (so `"TRACK_PR+CHECK_CASES"` and
`"CHECK_CASES+TRACK_PR"` can never both occur).

```
CLOSE, NOT_TARGET_CLOSE, VERIFY_AND_CLOSE, TRACK_PR,
IMPLEMENT, RETRIAGE_PRS, WAIT_EXTERNAL,
ROOT_CAUSE, FILE_ISSUE, MONITOR,
NEEDS_OWNER, NEED_ACTION,
AWAIT_REPLY, CHECK_CASES, SKIP
```

Rows that don't match any leaf land in the report under **§3.0 UNCLASSIFIED**
and are the primary signal that a new leaf is needed (or upstream data
is malformed).

---

## Scripts (in this folder)

Both scripts anchor paths on the repo root via
`Path(__file__).resolve().parents[7]`, so they are safe to run from any
CWD.

| Script | Purpose |
|---|---|
| [`run_action_type.py`](./run_action_type.py) | Reads `action_TBD`, classifies into the 17-category taxonomy, writes the `action_Type` column back to the Issues sheet. Prints a Counter summary on stdout. |
| [`gen_bug_scrub_md.py`](./gen_bug_scrub_md.py) | Reads Issues + Test Cases + E2E Test Cases sheets, consumes `action_Type`, emits `result/bug_scrub.md` with section-per-category, per-issue tables, `<br>`-wrapped Fix Approach (width 80), Duplicates column, and per-section Back-to-Index anchor links. |

---

## Execution Order

```
run_action_type.py       # populate action_Type column
        ↓
gen_bug_scrub_md.py      # render bug_scrub.md
```

Typical invocation:

```bash
python3 opencode/issue_triage/.claude/skills/bug_scrub/collect_AR/generate_report/run_action_type.py
python3 opencode/issue_triage/.claude/skills/bug_scrub/collect_AR/generate_report/gen_bug_scrub_md.py
```

---

## Inputs / Outputs

| | Path (relative to repo root) |
|---|---|
| Input Excel | `opencode/issue_triage/result/torch_xpu_ops_issues.xlsx` |
| Output column | `action_Type` (Issues sheet) |
| Output report | `opencode/issue_triage/result/bug_scrub.md` |

---

## Path Reference

Both scripts locate the repo root via:

```python
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[7]
```

The skill folder is 7 directory levels under the repo root:

```
<repo>/opencode/issue_triage/.claude/skills/bug_scrub/collect_AR/generate_report/
    0       1            2       3      4       5         6              7
```

If the skill is ever moved, update `parents[N]` accordingly in both
scripts.

---

## Skill Metadata

- **Version**: 1.0.0
- **Created**: 2026-04-22
- **Consumes**: Phase 1–4 output in `result/torch_xpu_ops_issues.xlsx`
- **Produces**: `action_Type` column + `result/bug_scrub.md`
