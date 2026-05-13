# Priority Analysis Skill

## Overview
This skill provides **deep semantic priority assessment** of torch-xpu-ops issues. Rather than relying on keyword pattern matching, the agent must read the full issue context, understand the failure mode, assess impact scope, and apply the priority definitions with human-level judgment.

---

## Priority Definitions

### P0 - Critical

| Condition | How to Verify | Examples |
|-----------|---------------|----------|
| Process crash / Segfault / GPU page fault | Stack trace shows SIGSEGV, access violation (0xC0000005), `drm_neo.cpp` page fault, `Fatal Python error: Aborted` | GPU hang, kernel panic, SIGSEGV |
| Torch build failure | CI log shows compilation/linker errors preventing a build | Compilation failures, linker errors |
| Performance regression >5% | Issue body or comments cite measured % drop AND regression confirmed between releases | 15% slower on 2.12 vs 2.11 |
| Custom/production model impact | Issue explicitly mentions customer deployment, production model, internal model NOT matching known benchmark suites | Production model failures |

### P1 - High

| Condition | How to Verify | Examples |
|-----------|---------------|----------|
| UT >6 failed cases | Count distinct test case names in issue body/title; meta-tracking issues listing many cases | Large test class failures |
| Regression (was passing, now failing) | Issue cites specific version where it passed AND current version where it fails | "Passed on 2.10, fails on 2.11" |
| Hang / timeout (process alive but stuck) | Issue describes 300s timeout, infinite wait, deadlock | Distributed test hangs |
| Accuracy regression on benchmark | Benchmark accuracy was passing in prior release, now fails (but not crash) | fail_accuracy on E2E model |

### P2 - Medium

| Condition | How to Verify | Examples |
|-----------|---------------|----------|
| Benchmark performance <5% | Measured perf drop cited but ≤5% | Minor throughput decrease |
| UT 1-6 failures | Small number of related test failures | A few op tests failing |
| Functional errors (not crash) | RuntimeError, AssertionError, wrong result but process continues | Wrong output, type errors |
| Feature gap blocking tests | Tests fail because API is not implemented, but no crash | "not implemented" errors |

### P3 - Low

| Condition | How to Verify | Examples |
|-----------|---------------|----------|
| Enhancement / feature request | Title contains "implement", "enable", "support", "RFC", "consider", "investigate", or body describes desired new functionality | Feature requests |
| Validation / error message difference | XPU raises different error message than CPU/CUDA, or doesn't raise where it should, but no incorrect computation | Error message mismatch |
| Minor / cosmetic / warning | Warning mismatch, deprecated API usage, documentation gaps | Warning mismatches |
| Alignment with CUDA (no functional break) | XPU behavior differs from CUDA but isn't incorrect per se | dtype support alignment |

---

## Deep Analysis Protocol

### Step 1: Read the Full Issue

Do NOT rely on title keywords alone. Read:
- Issue title
- Issue body (full error log, reproducer, description)
- Root cause analysis (if available in our triage data)
- Labels
- Comments (for maintainer diagnosis, version info)

### Step 2: Determine Failure Mode

Ask these questions in order:

1. **Does the process terminate abnormally?** (crash, segfault, abort, access violation, GPU page fault)
   - YES → P0 candidate
   - NO → continue

2. **Does the process hang indefinitely?** (timeout, deadlock, no response)
   - YES → P1 candidate
   - NO → continue

3. **Is there a measured performance regression >5%?**
   - Read the actual numbers. "performance dropped" without numbers is NOT sufficient for P0.
   - Verify it's a regression (compare between specific releases).
   - YES with >5% confirmed → P0
   - YES with ≤5% or unquantified → P2
   - NO → continue

4. **Is it a regression?** (confirmed working in version X, broken in version Y)
   - YES + accuracy/correctness → P1
   - YES + UT failures >6 → P1
   - YES + UT failures 1-6 → P1 (regression boost from P2)
   - NO → continue

5. **How many test cases fail?**
   - >6 distinct cases → P1
   - 1-6 cases → P2
   - 0 (enhancement/feature) → P3

6. **Is it an enhancement, feature request, or alignment issue?**
   - YES → P3

### Step 3: Apply Edge Case Rules

- **ai_generated issues**: Judge by actual failure mode, not by the `ai_generated` label. If it crashes → P0. If wrong result → P2. If validation/error message difference → P3.
- **"Bug Skip" tracking issues**: Count the actual failing cases inside. Don't assume P1 just because title says "new failures".
- **Performance without regression**: If perf is slow but was ALWAYS slow (no prior version was faster), it's P2 (improvement request), not P0/P1.
- **Benchmark vs custom model**: Only grant P0 for custom/production model impact. Benchmark issues cap at P1 for accuracy regression, P0 only for >5% perf regression.
- **Distributed feature gaps**: "not implemented" / "does not support" without crash = P2, not P1.

### Step 4: Generate Priority Reason

Write a concise (5-15 word) reason explaining the priority assignment. Examples:
- "Segfault on GPU page fault during SDPA with large seq_len"
- "15% perf regression on release/2.12 vs 2.11"
- "Regression: 245 conv tests pass on oneDNN 3.10, fail on 3.11"
- "3 UT accuracy failures in index_add bf16"
- "Enhancement: implement linalg.svd XPU backend"
- "Error message differs from CPU, no functional impact"

---

## Priority Assignment Workflow

If the input row already has a non-blank `Priority`, preserve that value. Phase 1 initializes `Priority` from the GitHub Projects `PyTorchXPU Priority` field, and that labeled priority is authoritative over computed priority. Only run computed priority assignment when the input row's `Priority` is blank.

## Anti-Patterns (DO NOT)

1. **DO NOT** classify as P0 just because title contains "regression" — verify the actual impact (crash? >5% perf? >6 failures?).
2. **DO NOT** classify as P1 just because there are "failures" — count them. 1-6 = P2.
3. **DO NOT** classify performance issues as P0 without confirmed >5% measured drop AND regression between releases.
4. **DO NOT** give P1 to feature gaps / "not implemented" issues that don't crash.
5. **DO NOT** use keyword pattern matching as the primary classification method. Keywords are hints; the agent must understand the issue semantically.
6. **DO NOT** classify ai_generated validation/error-message issues as P1 just because they mention "crash" in a hypothetical sense — verify actual crash evidence in the reproduction.

---

## Skill Metadata

- **Version**: 2.0.0
- **Created**: 2026-04-20
- **Updated**: 2026-05-12
- **Requires**: Issue text, error log, root cause analysis, comments
- **Related Skills**: SKILL_Category_Analysis.md, SKILL_Triage_Logic.md
