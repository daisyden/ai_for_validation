# Bug Scrub Workflow Diagram

Visual reference for the 5-phase torch-xpu-ops bug-scrub pipeline, showing how
each skill consumes and produces data in the shared Excel workbook
(`result/torch_xpu_ops_issues.xlsx`) and supporting artifact folders.

Source of truth for phase semantics: [`SKILL.md`](./SKILL.md) (v3.3).

---

## 1. End-to-End Pipeline

```mermaid
flowchart TD
    %% ========== EXTERNAL INPUTS ==========
    GH[(GitHub API<br/>intel/torch-xpu-ops)]:::ext
    CI[(CI artifacts<br/>torch-xpu-ops + stock pytorch)]:::ext
    PT[(pytorch/pytorch repo)]:::ext

    %% ========== PHASE 1: PREPARE DATA ==========
    subgraph P1["Phase 1 — Prepare Data"]
        direction TB
        S11["1.1 issue-basic-info-extraction<br/><i>fetch + parse issues</i>"]:::skill
        S12["1.2 download_ci_result<br/><i>download CI artifacts</i>"]:::skill
        S13["1.3 create-not-applicable-sheet<br/><i>wontfix / not_target filter</i>"]:::skill
        S14["1.4 pytorch_xpu_backend_analysis<br/><i>operator impl deep-dive</i>"]:::skill
    end

    %% ========== PHASE 2: ANALYZE CI RESULT ==========
    subgraph P2["Phase 2 — Analyze CI Result"]
        direction TB
        S21["2.1 match-ut-ci-matching"]:::skill
        S22["2.2 match-e2e-ci-matching"]:::skill
        S23["2.3 case-duplication-detection"]:::skill
        S24["2.4 check_xpu_case_existence<br/><i>first blank case per issue</i>"]:::skill
    end

    %% ========== PHASE 3: ANALYZE ISSUE ==========
    subgraph P3["Phase 3 — Analyze Issue"]
        direction TB
        S31["3.1 duplicated-issue-detection"]:::skill
        S33["3.3 triage_skills<br/><i>one-by-one deep triage<br/>(see §5 sub-workflow)</i>"]:::skill
    end

    %% ========== PHASE 4: COLLECT AR ==========
    subgraph P4["Phase 4 — Collect AR"]
        direction TB
        S4a["4a close_or_skip<br/><i>RULE 1: Fixed → Close<br/>RULE 2: not_target/wontfix → Skip</i>"]:::skill
        S4b["4b get_AR_from_issue<br/>(+ check_pr_status)"]:::skill
        S4c["4c case_existence_check"]:::skill
    end

    %% ========== PHASE 5: GENERATE REPORT ==========
    subgraph P5["Phase 5 — Generate Report"]
        direction TB
        S51["run_action_type.py<br/><i>classify action_TBD → action_Type</i>"]:::script
        S52["gen_bug_scrub_md.py<br/><i>render markdown</i>"]:::script
    end

    %% ========== ARTIFACTS ==========
    XLSX[(result/torch_xpu_ops_issues.xlsx<br/><b>Issues</b> · Test Cases · E2E · Not Applicable)]:::art
    CIART[(ci_results/)]:::art
    BACK[(pytorch_xpu_backend_analysis.md)]:::art
    REPORT[(result/bug_scrub.md<br/>result/bug_scrub_ut.md<br/>result/details/{id}.md × N)]:::out

    %% ========== FLOWS ==========
    GH --> S11
    CI --> S12
    PT --> S14

    S11 -->|"Issues · Test Cases · E2E sheets"| XLSX
    S12 --> CIART
    S13 -->|"+ Not Applicable sheet"| XLSX
    S14 --> BACK

    XLSX --> S21
    CIART --> S21
    S21 -->|"+ XPU Status · Stock Status"| XLSX

    XLSX --> S22
    CIART --> S22
    S22 -->|"+ E2E statuses"| XLSX

    XLSX --> S23
    S23 -->|"+ duplicate_group_id"| XLSX

    XLSX --> S24
    S24 -->|"+ xpu_case_existence<br/>+ case_existence_comments"| XLSX

    XLSX --> S31
    S31 -->|"+ duplicated_issue"| XLSX

    XLSX --> S33
    BACK -.reference.-> S33
    S33 -->|"+ Category · Priority<br/>+ Dependency<br/>+ Root Cause · Fix Approach"| XLSX

    XLSX --> S4a
    S4a -->|"+ action_TBD (close/skip)<br/>+ action_reason<br/>+ owner_transferred"| XLSX

    XLSX --> S4b
    GH -.PR status.-> S4b
    S4b -->|"append action_TBD<br/>append action_reason<br/>append owner_transferred"| XLSX

    XLSX --> S4c
    S4c -->|"append 'check_case_avaliablity'<br/>append case_existence_comments"| XLSX

    XLSX --> S51
    S51 -->|"+ action_Type (17-leaf taxonomy)"| XLSX

    XLSX --> S52
    S52 --> REPORT

    %% ========== STYLES ==========
    classDef ext fill:#f4e8d8,stroke:#8b6f47,stroke-width:2px,color:#000
    classDef skill fill:#d8e8f4,stroke:#2c5f8a,stroke-width:1px,color:#000
    classDef script fill:#e8d8f4,stroke:#5a2c8a,stroke-width:1px,color:#000
    classDef art fill:#fff4d8,stroke:#8a7c2c,stroke-width:1px,color:#000
    classDef out fill:#d8f4d8,stroke:#2c8a2c,stroke-width:2px,color:#000
```

**Legend**

| Shape / Color | Meaning |
|---|---|
| 🟤 Cylinder (tan) | External data source (GitHub API, CI system, pytorch repo) |
| 🟦 Rectangle (blue) | Skill (LLM-driven, `SKILL.md`-governed) |
| 🟪 Rectangle (purple) | Deterministic Python script |
| 🟨 Cylinder (yellow) | Intermediate artifact (Excel, CI dumps, analysis doc) |
| 🟩 Cylinder (green) | Final deliverable (markdown report) |
| Solid arrow | Read / write |
| Dashed arrow (`-.label.->`) | Referenced only (no mutation) |
| Edge label | Column(s) added or data passed |

---

## 2. Skill → Column Matrix

Each skill's contract, in the order the columns appear in the Excel:

| Phase | Skill | Reads | Writes (Issues sheet) | Writes (Test Cases sheet) |
|---|---|---|---|---|
| 1.1 | issue-basic-info-extraction | GitHub API | Issue ID, Title, Status, Assignee, Reporter, Labels, Created Time, Body | Test Case, Test File, Error Message, Traceback |
| 1.2 | download_ci_result | CI artifacts URL | — (produces `ci_results/`) | — |
| 1.3 | create-not-applicable-sheet | Issue labels | *(writes "Not Applicable" sheet)* | — |
| 1.4 | pytorch_xpu_backend_analysis | pytorch repo | — (produces standalone .md) | — |
| 2.1 | match-ut-ci-matching | Test Cases, CI artifacts | — | XPU Status, Stock Status |
| 2.2 | match-e2e-ci-matching | E2E Test Cases, CI artifacts | — | *(E2E sheet)* XPU Status, Stock Status |
| 2.3 | case-duplication-detection | Test Cases | — | duplicate_group_id |
| 2.4 | check_xpu_case_existence | Test Cases (first blank row per issue) | — | xpu_case_existence, case_existence_comments |
| 3.1 | duplicated-issue-detection | Issues, Test Cases | duplicated_issue | — |
| 3.3 | triage_skills | Issues body, Test Cases, pytorch_xpu_backend_analysis | Category, Priority, Dependency, Root Cause, Fix Approach | — |
| 4a | close_or_skip | Labels, Test Cases statuses | action_TBD, action_reason, owner_transferred | — |
| 4b | get_AR_from_issue | Issues body, GitHub PRs (gh api) | action_TBD *(append)*, action_reason *(append)*, owner_transferred *(append)* | — |
| 4c | case_existence_check | xpu_case_existence, case_existence_comments | action_TBD *(append `check_case_avaliablity`)*, action_reason *(append)* | — |
| 5 (script) | `run_action_type.py` | action_TBD | action_Type *(17-leaf taxonomy, `+`-joined)* | — |
| 5 (script) | `gen_bug_scrub_md.py` | Issues sheet | — (produces `bug_scrub.md`, `bug_scrub_ut.md`, `details/*.md`) | — |

---

## 3. Execution Order & Dependencies

```mermaid
flowchart LR
    subgraph Prep["Phase 1 — Prepare Data"]
        A1["1.1 issue-basic-info-extraction"] --> A2["1.2 download_ci_result"]
        A1 --> A3["1.3 create-not-applicable-sheet"]
        A1 --> A4["1.4 pytorch_xpu_backend_analysis"]
    end
    subgraph CI["Phase 2 — Analyze CI Result"]
        B1["2.1 match-ut-ci-matching"] --> B2["2.2 match-e2e-ci-matching"] --> B3["2.3 case-duplication-detection"] --> B4["2.4 check_xpu_case_existence"]
    end
    subgraph Issue["Phase 3 — Analyze Issue"]
        C1["3.1 duplicated-issue-detection"] --> C3["3.3 triage_skills"]
    end
    subgraph AR["Phase 4 — Collect AR"]
        D1["4a close_or_skip"] --> D2["4b get_AR_from_issue<br/>(+ check_pr_status)"] --> D3["4c case_existence_check"]
    end
    subgraph Report["Phase 5 — Generate Report"]
        E1["run_action_type.py"] --> E2["gen_bug_scrub_md.py"]
    end
    Prep --> CI --> Issue --> AR --> Report

    classDef ph fill:#f0f0f0,stroke:#666
    class Prep,CI,Issue,AR,Report ph
```

**Invariants**

- Phases are strictly sequential; later phases append columns to the shared Excel.
- Within a phase, sub-steps labeled N.1 → N.2 → N.3 → N.4 are also strictly sequential.
- Phase 4 sub-steps 4a → 4b → 4c are sequential because each may **append** to `action_TBD` / `action_reason`.
- Phase 5's two scripts are sequential: `action_Type` must exist before rendering.

---

## 4. Output Artifacts

```
result/
├── torch_xpu_ops_issues.xlsx          ← single source of truth, grown phase-by-phase
├── torch_xpu_ops_issues_bk_*.xlsx     ← step-by-step backups (convention)
├── pytorch_xpu_backend_analysis.md    ← from 1.4
├── bug_scrub.md                       ← from 5, full scope (all issues)
├── bug_scrub_ut.md                    ← from 5, UT-scoped subset
└── details/
    └── {issue_id}.md × N              ← from 5, one per issue
ci_results/                            ← from 1.2, per-run artifacts
```

---

## 5. Triage Skills Sub-Workflow (Phase 3.3 expansion)

`triage_skills` runs once per issue (no batch script — strictly one-by-one
unless using the wave-based parallel pattern in
[`SKILL_Batch_Orchestration.md`](./analyze_issue/triage_skills/SKILL_Batch_Orchestration.md)).
Each invocation walks 6 deterministic steps and emits one JSON object per
issue conforming to the canonical schema in
[`triage_skills/SKILL.md`](./analyze_issue/triage_skills/SKILL.md):

```mermaid
flowchart TD
    %% ========== INPUTS ==========
    IN_ISSUE[(Issue row<br/>title · body · labels · comments)]:::inp
    IN_OP[(xpu_supported_operators_complete_list.md<br/>operator → dependency)]:::inp
    IN_SRC[(~/pytorch + third_party/torch-xpu-ops<br/>source + tests)]:::inp
    IN_CI[(ci_results/.../op_ut/*.xml<br/>UT logs)]:::inp
    IN_BACK[(pytorch_xpu_backend_analysis.md)]:::inp

    %% ========== STEPS ==========
    T1["STEP 1<br/>Issue Acquisition + Version Detection<br/><sub>gh issue view · webfetch fallback<br/>extract PyTorch / IGC / Triton / oneAPI versions</sub>"]:::step
    T2["STEP 2<br/>Reproduce Command Extraction<br/><sub>identify failing test from issue body<br/>resolve test path under ~/pytorch/test/<br/>or torch-xpu-ops/test/xpu/</sub>"]:::step
    T3["STEP 3<br/>Code Exploration + Test Analysis<br/><sub>explore agent (medium depth)<br/>locate impl + test files<br/>read assertions / kernel launch</sub>"]:::step
    T4["STEP 4<br/>Runtime Verification<br/><sub>conda env: pytorch_opencode_env<br/>execute reproduce if version-compatible</sub>"]:::step
    T5["STEP 5<br/>Deep Root Cause Analysis<br/><sub>XPU vs CPU fallback diff<br/>kernel-code investigation<br/>error-pattern → cause mapping</sub>"]:::step
    T6["STEP 6<br/>Dependency Analysis + Classification<br/><sub>apply 4 taxonomies:<br/>Category · Priority · Dependency · Root Cause</sub>"]:::step

    %% ========== HELPER SKILLS ==========
    H_CAT["SKILL_Category_Analysis.md<br/><sub>8-bucket rubric</sub>"]:::help
    H_PRI["SKILL_Priority_Analysis.md<br/><sub>P0–P3 weighted scoring</sub>"]:::help
    H_DEP["SKILL_Domain_Patterns.md<br/><sub>quick-reference patterns</sub>"]:::help
    H_DEEP["SKILL_Deep_Analysis_Patterns.md<br/><sub>error-type → investigation</sub>"]:::help
    H_E2E["SKILL_E2E_Benchmark.md<br/><sub>benchmark-suite triage</sub>"]:::help
    H_TLOG["SKILL_Triage_Logic.md<br/><sub>orchestration rules</sub>"]:::help

    %% ========== OUTPUT ==========
    OUT["JSON entry per issue<br/><b>{ row, issue_id, category, priority,<br/>dependency, root_cause, fix_approach }</b>"]:::out

    POST["run_needs_owner_fix.py<br/><sub>repair: NEEDS_OWNER + has Assignee → ROOT_CAUSE</sub>"]:::script
    XLSX[(result/torch_xpu_ops_issues.xlsx<br/>Issues sheet)]:::art

    %% ========== FLOW ==========
    IN_ISSUE --> T1
    T1 --> T2
    T2 --> T3
    IN_SRC --> T3
    T3 --> T4
    T4 --> T5
    IN_CI -.evidence.-> T5
    IN_BACK -.reference.-> T5
    T5 --> T6
    IN_OP -.lookup.-> T6

    H_CAT -.guides.-> T6
    H_PRI -.guides.-> T6
    H_DEP -.guides.-> T3
    H_DEEP -.guides.-> T5
    H_E2E -.guides.-> T2
    H_TLOG -.guides.-> T1

    T6 --> OUT
    OUT --> XLSX
    XLSX --> POST
    POST --> XLSX

    %% ========== STYLES ==========
    classDef inp fill:#f4e8d8,stroke:#8b6f47,stroke-width:2px,color:#000
    classDef step fill:#d8e8f4,stroke:#2c5f8a,stroke-width:2px,color:#000
    classDef help fill:#e8e8f4,stroke:#5a5a8a,stroke-width:1px,color:#000,stroke-dasharray: 3 3
    classDef out fill:#d8f4d8,stroke:#2c8a2c,stroke-width:2px,color:#000
    classDef art fill:#fff4d8,stroke:#8a7c2c,stroke-width:1px,color:#000
    classDef script fill:#e8d8f4,stroke:#5a2c8a,stroke-width:1px,color:#000
```

### 5.1 Helper-Skill Files (governance, not executed directly)

The dashed nodes above are reference documents that the LLM consults while
performing each step. They live in `analyze_issue/triage_skills/`:

| File | Role |
|---|---|
| [`SKILL.md`](./analyze_issue/triage_skills/SKILL.md) | Authoritative output schema + 4 taxonomies |
| [`SKILL_Triage_Logic.md`](./analyze_issue/triage_skills/SKILL_Triage_Logic.md) | Orchestration of the 6 steps |
| [`SKILL_Category_Analysis.md`](./analyze_issue/triage_skills/SKILL_Category_Analysis.md) | 8-category rubric (Distributed > Flash Attention > Inductor > TorchAO > Sparse > Torch Operations > Torch Runtime > Others) |
| [`SKILL_Priority_Analysis.md`](./analyze_issue/triage_skills/SKILL_Priority_Analysis.md) | P0–P3 weighted scoring |
| [`SKILL_Dependency_Analysis.md`](./analyze_issue/triage_skills/SKILL.md) | Dependency taxonomy (driver, xccl, triton, oneDNN, oneMKL, oneAPI, CPU fallback, SYCL kernel:&lt;file&gt;, upstream-pytorch, blank) |
| [`SKILL_Deep_Analysis_Patterns.md`](./analyze_issue/triage_skills/SKILL_Deep_Analysis_Patterns.md) | Error-pattern → investigation mapping |
| [`SKILL_Domain_Patterns.md`](./analyze_issue/triage_skills/SKILL_Domain_Patterns.md) | Quick-reference patterns + tools |
| [`SKILL_E2E_Benchmark.md`](./analyze_issue/triage_skills/SKILL_E2E_Benchmark.md) | E2E/benchmark-specific triage |
| [`SKILL_Batch_Orchestration.md`](./analyze_issue/triage_skills/SKILL_Batch_Orchestration.md) | Wave-based parallel pattern (5 issues × 5 explore agents × N waves) for large-scale runs |

### 5.2 Step → Column Mapping

| Step | Produces | Excel column populated |
|---|---|---|
| 1 | version table | (none — diagnostic only) |
| 2 | reproduce command | (none — used by step 4) |
| 3 | impl + test paths | feeds Root Cause text |
| 4 | runtime PASS / FAIL evidence | feeds Root Cause + Priority |
| 5 | root-cause narrative + file:line citations | **Root Cause** |
| 6 | classified JSON | **Category**, **Priority**, **Dependency**, **Fix Approach** |
| post | NEEDS_OWNER repair | **action_Type** correction |

### 5.3 Invariants

- Steps 1–6 are strictly sequential per issue (later steps depend on earlier evidence).
- Step 4 (runtime verification) is **skipped** if the issue's reported PyTorch / driver / Triton version is incompatible with the local conda env — the version table from Step 1 gates this.
- The output JSON object **must** match the schema in [`SKILL.md`](./analyze_issue/triage_skills/SKILL.md): no markdown fences, no wrapper key, one object per issue.
- `run_needs_owner_fix.py` runs **once after the full Phase-3 pass**, not per-issue.

---

## Version

v1.1 — 2026-04-22 — added §5 Triage Skills sub-workflow (6-step expansion of Phase 3.3) with helper-skill reference matrix.
v1.0 — 2026-04-22 — initial workflow diagram accompanying bug_scrub SKILL.md v3.3.
