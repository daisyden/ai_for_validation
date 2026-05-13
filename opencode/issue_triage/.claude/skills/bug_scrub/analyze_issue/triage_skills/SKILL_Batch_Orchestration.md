# Batch Orchestration — Wave-Based Parallel Triage

## When to Use
When triaging more than ~30 issues at once (e.g., a full Phase 3 sweep over a tracking Excel). For single-issue or <10-issue runs, just invoke the explore agent directly.

## Pattern: Waves × Batches × Parallel Agents

```
Wave N
├── Batch 1 (5 issues)  ──►  explore agent #1  ──►  results_1.json
├── Batch 2 (5 issues)  ──►  explore agent #2  ──►  results_2.json
├── Batch 3 (5 issues)  ──►  explore agent #3  ──►  results_3.json
├── Batch 4 (5 issues)  ──►  explore agent #4  ──►  results_4.json
└── Batch 5 (5 issues)  ──►  explore agent #5  ──►  results_5.json
    (5 agents fired in a single tool-call message → true parallelism)
```

- **5 issues per batch**: keeps each agent's context window and runtime manageable; most issues complete in a single gh-view + 2-5 file reads.
- **5 parallel agents per wave**: empirically stable; larger fan-outs risk rate limits on `gh` and overlap in expensive file reads.
- **Waves are sequential**: you review / spot-check each wave before firing the next.
- **Final (partial) wave** can fire fewer agents if <25 issues remain.

Sizing: for N issues, use `ceil(N / 25)` waves. Example: 375 issues → 15 waves (wave 15 partial with 10 issues / 2 agents).

## Directory Layout

```
<scratch>/phase3_triage/
├── issues_all.json               # flat export of all N issues from Excel
├── wave1/
│   ├── batch_1.json .. batch_5.json     # inputs (5 issues each)
│   └── results_1.json .. results_5.json # outputs (JSON array of 5)
├── wave2/ ...
├── waveN/
└── all_triage_merged.json        # final deduped merge
```

Use `/home/daisydeng/pytorch/agent_space/phase3_triage/` as the scratch root (git-ignored).

## Step-by-step

### 1. Export issues from Excel to JSON

Read the Issues sheet, emit one object per row with the columns needed for triage (row, issue_id, title, module, test_module, existing dependency if any, existing priority if any). Write `issues_all.json`.

### 2. Pre-prepare all batches for all waves upfront

```python
# pseudo
issues = load(issues_all.json)
for w, chunk25 in enumerate(chunks(issues, 25), start=1):
    for b, chunk5 in enumerate(chunks(chunk25, 5), start=1):
        write(f"wave{w}/batch_{b}.json", chunk5)
```

Preparing batches upfront (rather than per-wave) lets you inspect / adjust before any agent runs.

### 3. Fire one wave = ONE assistant message with K parallel `task(...)` calls

Critical: all K `task()` calls must be in the **same** assistant message so the platform dispatches them concurrently. Separate messages serialize execution.

Standard per-agent prompt:

```
Triage 5 torch-xpu-ops issues with rigorous per-issue deep analysis.

BATCH:  <scratch>/wave<W>/batch_<B>.json
OUTPUT: <scratch>/wave<W>/results_<B>.json

REFERENCES:
- Skill: .../triage_skills/SKILL.md  (read "Authoritative Reference" section)
- Supporting: SKILL_Category_Analysis.md, SKILL_Priority_Analysis.md,
  SKILL_Deep_Analysis_Patterns.md, SKILL_Domain_Patterns.md
- Operator list: .../xpu_supported_operators_complete_list.md
- CI: .../ci_results/torch-xpu-ops/Inductor-XPU-UT-Data-*/op_ut/*.xml
- Source: /home/daisydeng/pytorch/third_party/torch-xpu-ops/  and  /home/daisydeng/pytorch/

WORKFLOW PER ISSUE:
1. gh issue view <id> --repo intel/torch-xpu-ops --json title,body,labels,comments,state
2. Locate test/code/error — cite file:line evidence
3. root_cause: 2-4 sentences with file:line
4. fix_approach: actionable next steps
5. Classify last using the authoritative Category Taxonomy,
   Dependency Taxonomy, and Priority rubric from SKILL.md. Do not assign
   dependency/category/priority before root_cause and fix_approach are drafted.
   Dependency must use the confirmed failing component, root_cause, and
   fix_approach as evidence. If the batch input row has a non-blank priority,
   preserve that value because it came from GitHub Projects `PyTorchXPU Priority`;
   only compute priority when the input priority is blank. New-case Excel triage
   fields may be blank; use issue/log/source evidence instead.

OUTPUT: JSON array of 5 entries only — no wrapping, no prose, no markdown fences.
Schema per entry: {row, issue_id, category, priority, dependency, root_cause, fix_approach}
```

### 4. Validate each wave before firing the next

- Confirm all `results_<B>.json` files exist and parse as 5-entry JSON arrays.
- Spot-check 1-2 entries per batch for taxonomy conformance (especially Category and Dependency).
- If any agent returns wrapped shape (`{"results": [...]}`) or wrong taxonomy, re-run just that batch.

### 5. Merge — single final step

After all waves done:

```python
all_entries = []
all_entries.extend(load("pilot_refined.json"))     # if there was a pilot
for w in range(1, N_WAVES+1):
    for f in sorted(glob(f"wave{w}/results_*.json")):
        all_entries.extend(load(f))

# Dedup by row (last-writer-wins)
merged = {e["row"]: e for e in all_entries}
assert set(merged) == set(range(2, MAX_ROW+1)), "missing/extra rows"
write("all_triage_merged.json", sorted(merged.values(), key=lambda x: x["row"]))
```

Sanity checks: row-count equality, no gaps, no extras.

### 6. Single final write to Excel

- **Always back up first**: `cp tracker.xlsx tracker_bk_before_phase3_write.xlsx`.
- Write only the triage columns (Dependency, Category, Priority, Root Cause, Fix Approach). Preserve all other columns. For `Priority`, preserve any existing non-blank value from Phase 1's `PyTorchXPU Priority` import; only write the computed Phase 3 priority into blank cells.
- Verify post-write: reload Excel, print Counter of Category / Priority / Dependency to catch any taxonomy drift.
- If drift found, fix with a targeted remap script (don't re-triage).

## Pilot First (strongly recommended)

Before running the full sweep, run a **pilot wave of 15 issues** (3 batches × 5) using the same prompt template. Use the pilot to:
- Verify the agents respect the taxonomies and output schema
- Refine the prompt if outputs deviate
- Calibrate your spot-check expectations

Persist the pilot output as `pilot_refined.json` and include it in the final merge.

## Common Pitfalls (observed)

| Pitfall | Fix |
|---|---|
| Agents invent new Category values (e.g., "Build/Compilation", "Feature Gap") | Paste the 8-bucket enum inline in the prompt AND reference `SKILL_Category_Analysis.md`. Do post-merge normalization pass if drift still occurs. |
| Wrapping JSON in `{"results": [...]}` | Say explicitly "JSON array only, no wrapping, no markdown fences". |
| `dependency` filled with free-form text | Enforce enum. Use `upstream-pytorch` explicitly — don't let it hide in blank. |
| Mixing canonical-9 categories (Test Infrastructure, Numerical Accuracy, ...) with domain-8 | Use only the domain-8 in Excel. The canonical-9 labels are OK in `root_cause` prose. |
| Running waves serially by accident | All K `task()` calls must be in ONE assistant message. |
| Over-writing Excel mid-run | Do Excel write only once, at the very end, after merge + sanity checks + backup. |

## Scaling Notes

- For very large sweeps (>500 issues), consider 7-10 parallel agents per wave with 5 issues each, but watch for `gh` rate limits.
- Keep `gh auth status` verified before firing; agents crash silently if gh is unauthenticated.
- If ~10% of batches need re-runs due to schema violations, prompt is too loose — tighten it, don't just re-fire.

## Skill Metadata

- **Version**: 1.0.0
- **Created**: 2026-04-21
- **Related**: SKILL.md (Authoritative Reference section), SKILL_Category_Analysis.md, SKILL_Priority_Analysis.md
