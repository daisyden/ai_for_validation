# Generate HTML Report Skill

## Overview

Phase **5b** of the bug-scrub pipeline. A presentation alternative to
Phase 5's plain markdown — emits an interactive single-file HTML report
with per-row "Done" checkboxes (persisted in browser localStorage) and a
sticky filter bar across all sections.

This skill is purely presentational. It re-runs `gen_bug_scrub_md.py`
first (so HTML always reflects the current workbook), then converts the
markdown to HTML. It never calls `gh`, never touches the workbook, and
never rewrites verdict columns.

## When to Use

Run after Phase 4 has stabilized the Issues sheet. Re-running is
idempotent — overwrites `result/bug_scrub.html` cleanly.

The HTML report is a *triage console*. Use it when you want to:

- Walk through Action-Required items and tick them off as you complete
  them (state survives page reload, per browser).
- Slice the report by Assignee / Owner Transferred / Priority / Category
  / Dependency without re-rendering.
- Free-text search across Title and `action_TBD`.
- Hide done rows to focus on the remaining queue.
- Export the list of issue IDs you've marked done (clipboard) to share
  progress.

The markdown report (`bug_scrub.md`) remains the canonical, diffable
artifact. The HTML report is derived from it and is not committed by
default — it's regenerated on demand.

---

## Features

### Done checkbox (§3 Action required, §4 QA only)

Per the spec, each table row in §3 and §4 gets a leading "Done" checkbox
column. Checked-state is keyed by issue ID and persisted in
`localStorage` under `bug_scrub_done_<issue_id>`:

- Persistence is **per browser**. Not embedded in the HTML file. Not
  shared across machines.
- Checked rows are dimmed and struck-through.
- "Hide Done" toggle in the filter bar collapses checked rows out of
  view.
- "Export Done IDs" copies a comma-separated list of done issue IDs to
  clipboard.

§5–§8 tables do not get Done checkboxes (they aren't Action-Required).

### Filter bar (sticky, top of page, applies to all sections)

Five dropdowns + free-text search + Hide Done + Reset:

| Filter | Source |
|---|---|
| Assignee | Owner column (raw GitHub Assignee) |
| Owner Transferred | `owner_transferred` column |
| Priority | Priority column |
| Category | Category column |
| Dependency | Dependency column |
| Search (text) | Title + action_TBD substring, case-insensitive |
| Hide Done | client-side, suppresses rows in `done` state |

Dropdown options are populated dynamically from the rendered rows — they
adapt to whatever the report contains, no hard-coded lists. Filters are
**AND-combined**: a row is shown only if every active filter matches.
The stats indicator on the right shows `<visible> / <total> rows`.

### Self-contained output

CSS and JS are inlined. No CDN, no external assets. The HTML file works
offline and can be emailed / archived as-is.

---

## Scripts (in this folder)

| Script | Purpose |
|---|---|
| [`gen_bug_scrub_html.py`](./gen_bug_scrub_html.py) | Re-runs `gen_bug_scrub_md.py`, parses the resulting markdown (heading / paragraph / list / table subset), emits `result/bug_scrub.html` with per-row `data-*` attributes for filtering, Done checkboxes in §3/§4, and inlined CSS+JS. |

---

## Execution Order

This skill **wraps** Phase 5 — it re-runs Phase 5's `gen_bug_scrub_md.py`
internally before converting. You don't need to run Phase 5 separately.

```
gen_bug_scrub_html.py
    ├── (calls) gen_bug_scrub_md.py     # refresh bug_scrub.md
    └── parse markdown → render HTML    # emit bug_scrub.html
```

Typical invocation:

```bash
python3 opencode/issue_triage/.claude/skills/bug_scrub/collect_AR/generate_html_report/gen_bug_scrub_html.py
```

Open `opencode/issue_triage/result/bug_scrub.html` in a browser.

> **Note**: `run_action_type.py` is **not** auto-invoked. If `action_TBD`
> changed since the last `action_Type` classification, run Phase 5's
> `run_action_type.py` first, then this skill.

---

## Inputs / Outputs

| | Path (relative to repo root) |
|---|---|
| Input Excel (via Phase 5) | `opencode/issue_triage/result/torch_xpu_ops_issues.xlsx` |
| Intermediate (regenerated) | `opencode/issue_triage/result/bug_scrub.md` |
| Output HTML | `opencode/issue_triage/result/bug_scrub.html` |

---

## Markdown Subset Supported

We control the input shape (`gen_bug_scrub_md.py`), so the parser only
handles what that script emits:

- ATX headings (`#` … `######`) with optional preceding `<a id="…"></a>` anchor.
- Paragraphs, italic-only meta lines (`_…_`).
- Unordered lists (`- ` / `* `).
- GitHub-flavored tables with leading/trailing `|`.
- Inline: links `[text](url)`, code `` `x` ``, bold `**x**`, italic `*x*`, `<br>`.

Anything else is rendered as a plain `<p>`. Add cases as needed.

---

## Filter Column Mapping

The parser maps header-cell text to filter dimensions case-sensitively:

```python
FILTER_COLUMNS = {
    "Assignee":          "assignee",
    "Owner":             "assignee",          # report renames Assignee → Owner
    "Owner Transferred": "owner_transferred",
    "Priority":          "priority",
    "Category":          "category",
    "Dependency":        "dependency",
}
```

If a future report adds a new column the user wants to filter on, update
this dict (and `FILTER_DIMS` / `FILTER_LABELS` in the inlined JS).

---

## Done-Checkbox Section Gating

```python
DONE_CHECKBOX_PREFIXES = ("3.", "4.")
```

Done checkboxes are emitted only inside tables whose nearest preceding
heading starts with `3.` or `4.` (covers §3.0, §3.1.x, §3.2.x, §3.3.x,
§4, §4.x). To extend to additional sections, add prefixes here.

---

## Path Reference

```python
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[7]
```

The skill folder is 7 directory levels under the repo root:

```
<repo>/opencode/issue_triage/.claude/skills/bug_scrub/collect_AR/generate_html_report/
    0       1            2       3      4       5         6              7
```

If the skill is ever moved, update `parents[N]` in `gen_bug_scrub_html.py`.

---

## Skill Metadata

- **Version**: 1.0.0
- **Created**: 2026-04-27
- **Consumes**: `result/bug_scrub.md` (regenerated via Phase 5's `gen_bug_scrub_md.py`)
- **Produces**: `result/bug_scrub.html` (single self-contained file)
