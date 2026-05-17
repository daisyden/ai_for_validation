# Create Not Applicable Sheet

## Base Path Reference

Relative paths from this file location
(`bug_scrub/prepare_data/create-not-applicable-sheet/`):

```
../../../                       → issue_triage root
../../../result/                → Excel results directory
.                               → WORKDIR (this SKILL_DIR)
```

## Overview

Owns the `Not applicable` sheet of `result/torch_xpu_ops_issues.xlsx`. This
sheet collects issues that are explicitly out of scope for XPU support
(typically `wontfix`, `not_target`, deprecated upstream APIs, hardware
limitations, etc.).

The sheet is regenerated separately from the other four sheets because
`generate_excel.py` (the `issue-basic-info-extraction` skill) wipes its own
output workbook on every run. This skill restores or rebuilds the
`Not applicable` sheet without touching anything else.

## Two Modes

| Mode | When | Cost | Quality |
|---|---|---|---|
| **Carry-forward** (default in Phase 1.3) | Issues already classified in a prior backup; no new wontfix/not_target issues expected | seconds | reuses prior deep analysis |
| **Deep analysis** (initial population, or when new wontfix issues appear) | Need to classify a never-seen wontfix/not_target issue | minutes per issue (sub-agent) | high — root-cause level |

Use carry-forward whenever a known-good backup contains the rows you need.
Use deep analysis only for issues that have no prior classification.

---

## Mode A — Carry-Forward (Phase 1.3)

### When

- Phase 1 is re-running but the set of wontfix/not_target issues has not changed.
- A prior `result/torch_xpu_ops_issues_bk_*.xlsx` contains a known-good
  `Not applicable` sheet.

### Procedure

1. Identify the latest trusted backup:
   ```
   result/torch_xpu_ops_issues_bk_before_phase1_rerun_<TIMESTAMP>.xlsx
   ```
   (or any other prior file that contains the `Not applicable` sheet you want).

2. Run the carry-forward block (cell-by-cell copy preserves formatting and
   data types):

   ```python
   from openpyxl import load_workbook

   SRC = '../../../result/torch_xpu_ops_issues_bk_<TIMESTAMP>.xlsx'
   DST = '../../../result/torch_xpu_ops_issues.xlsx'

   src = load_workbook(SRC)
   dst = load_workbook(DST)

   if 'Not applicable' in dst.sheetnames:
       del dst['Not applicable']

   src_ws = src['Not applicable']
   new_ws = dst.create_sheet('Not applicable')
   for row in src_ws.iter_rows():
       for cell in row:
           new_ws.cell(row=cell.row, column=cell.column, value=cell.value)

   dst.save(DST)
   print('Carried forward rows:', new_ws.max_row - 1)
   ```

3. Verify row count matches the source. No new analysis is performed.

### Why this is the default

Deep analysis of wontfix issues changes rarely. Recomputing it on every
Phase 1 rerun wastes minutes per issue and risks regressions. Carry-forward
is deterministic and fast.

---

## Mode B — Deep Analysis

### When

- Initial population (no prior `Not applicable` sheet exists).
- New issues acquire `wontfix` or `not_target` labels since the last run.
- A reviewer asks for fresh root-cause classification.

### Why deep analysis (not regex)

The "not applicable" determination requires understanding beyond pattern
matching:

- **CUDA-specific feature** vs. **upstream-deprecated API** vs. **hardware
  limitation** all look similar in issue text but have different remediation
  paths.
- The same operator may be unavailable for different reasons in different
  contexts (e.g. missing on Windows but present on Linux).
- Technical decisions evolve as torch-xpu-ops and PyTorch evolve; only a
  reasoning agent can correlate the current codebase with the issue.

### Procedure

For each issue with `wontfix` or `not_target` label and no prior row in the
`Not applicable` sheet:

1. **Fetch full issue context**:
   ```bash
   gh issue view <NUM> --repo intel/torch-xpu-ops \
       --json title,body,labels,comments
   ```

2. **Spawn a sub-agent for deep analysis** (read-only investigation):

   ```python
   task(
       subagent_type="explore",
       run_in_background=False,
       load_skills=[],
       description=f"NA analysis #{num}",
       prompt=f"""
   INVESTIGATION: Not-applicable analysis for intel/torch-xpu-ops issue #{num}.

   CONTEXT:
     Title:  {title}
     Labels: {labels}
     Body excerpt: {body[:2000]}

   GOAL:
     Determine the root-cause category for why this issue is wontfix/not_target,
     and identify the specific torch operation, API, kernel, or feature involved.

   INVESTIGATION SCOPE (do all that apply):
     1. Identify the torch operator / API / kernel referenced in the issue.
     2. Search torch-xpu-ops source for an existing or stub implementation:
          third_party/torch-xpu-ops/src/ATen/native/
          third_party/torch-xpu-ops/src/comm/
     3. Search PyTorch core for the upstream definition:
          torch/, aten/src/ATen/native/, aten/src/ATen/native/cuda/
     4. Check whether the feature is CUDA-only, deprecated upstream, or
        gated on hardware/ISA unavailable on XPU.
     5. Note any workarounds or related operators that DO have XPU support.

   ROOT-CAUSE CATEGORIES (pick the one that fits best):
     - CUDA-specific implementation (not portable to SYCL)
     - Hardware limitation (ISA/feature unavailable on Intel GPU)
     - Deprecated/removed feature (upstream PyTorch deprecation)
     - Upstream-not-on-roadmap (PyTorch core decision)
     - License restriction
     - Third-party dependency unavailable
     - Complexity barrier (not worth implementing for current scope)

   DELIVERABLES (one block per issue):
     1. Operation/API: <name, e.g. torch.foo / aten::bar / sycl::baz>
     2. Category: <one of the categories above>
     3. Technical details: <2-4 sentences citing code paths examined>
     4. Workaround (optional): <related op that DOES work, or '' >
   """
   )
   ```

3. **Record the result** as one row in the `Not applicable` sheet:

   | Column           | Source                                       |
   |------------------|----------------------------------------------|
   | Issue ID         | GitHub issue number                          |
   | Title            | Original issue title                         |
   | Operation/API    | From sub-agent deliverable 1                 |
   | Category         | From sub-agent deliverable 2                 |
   | Technical Details| From sub-agent deliverable 3                 |
   | Workaround       | From sub-agent deliverable 4 (optional)      |
   | Labels           | Original labels (comma-joined)               |

4. **Append, don't overwrite.** Read existing rows first, append new ones at
   the bottom.

### Output

`../../../result/torch_xpu_ops_issues.xlsx` with `Not applicable` sheet
containing the columns above.

---

## Invariants

After this skill runs:

1. The `Not applicable` sheet exists in the workbook.
2. Row count ≥ 1 (header row + any data rows).
3. No issue in `Not applicable` also appears in `Test Cases`, `E2E Test Cases`,
   or `Others`. Validate with:

   ```python
   from openpyxl import load_workbook
   wb = load_workbook('../../../result/torch_xpu_ops_issues.xlsx')
   def ids(s):
       return {r[0] for r in wb[s].iter_rows(min_row=2, values_only=True)
               if r and r[0] is not None}
   na = ids('Not applicable')
   assert not (na & ids('Test Cases')),     na & ids('Test Cases')
   assert not (na & ids('E2E Test Cases')), na & ids('E2E Test Cases')
   assert not (na & ids('Others')),         na & ids('Others')
   ```

4. The Issues sheet's `Test Module` column is rewritten by
   `issue-basic-info-extraction`'s post-pass; if you want NA-routed issues
   to display `Test Module = "not_applicable"`, extend that post-pass to read
   the NA sheet IDs and set the column accordingly.

## Tools

| Tool | Purpose |
|---|---|
| `openpyxl`           | Cell-by-cell sheet copy (carry-forward) and read/write |
| `gh issue view`      | Fetch full issue text for deep analysis |
| `task(subagent_type="explore", ...)` | Spawn deep-analysis sub-agent |

## Prerequisites

- `result/torch_xpu_ops_issues.xlsx` already populated by
  `issue-basic-info-extraction` (sheets Issues / Test Cases / E2E Test Cases /
  Others).
- For carry-forward: a prior `torch_xpu_ops_issues_bk_*.xlsx` containing a
  `Not applicable` sheet.
- For deep analysis: `gh` CLI authenticated; local PyTorch checkout at
  `$PYTORCH_REPO_ROOT` for code search.

## Notes

- The previous version of this skill recommended deep analysis on every run.
  In practice this is wasteful: the wontfix set changes rarely. The current
  default (carry-forward) preserves prior deep analysis without recomputation.
- Hard-coded issue lists from any prior conversation are NOT part of the
  contract. The skill must work on whatever wontfix/not_target issues are
  open at the time it runs.
- Never edit `Not applicable` from `generate_excel.py`. That script owns
  Issues / Test Cases / E2E Test Cases / Others only.
