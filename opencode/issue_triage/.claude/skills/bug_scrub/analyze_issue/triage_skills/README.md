# Triage Skills Registry

## Skill Files

| Skill | File | Description |
|-------|------|-------------|
| Triage Logic | `SKILL_Triage_Logic.md` | Core workflow, tools, constraints, version-aware triage |
| Deep Analysis | `SKILL_Deep_Analysis_Patterns.md` | Multi-dimension analysis, error patterns, implementation investigation |
| Domain Patterns | `SKILL_Domain_Patterns.md` | Quick reference, tool map, decision tree |

---

## Load Instructions

To use these skills for triage operations:

1. Read the skill files in order:
   - First: `SKILL_Domain_Patterns.md` (overview)
   - Then: `SKILL_Triage_Logic.md` (detail)
   - Finally: `SKILL_Deep_Analysis_Patterns.md` (analysis patterns)

2. Apply workflow from `SKILL_Triage_Logic.md`

3. Use patterns from `SKILL_Deep_Analysis_Patterns.md` to draft root cause and fix approach

4. After root cause and fix approach are drafted, assign Dependency from the confirmed failing component plus those conclusions, then apply `SKILL_Priority_Analysis.md` and `SKILL_Category_Analysis.md` for final priority/category assignment. New-case Excel triage fields may be blank and must not be required inputs.

5. Reference tools from `SKILL_Domain_Patterns.md`

---

## Version-Aware Triage

### Decision Rules

```python
RULES = {
    "private_branch": {
        "condition": "issue mentions 'private' | 'unreleased' | 'internal'",
        "action": "Skip execution. Analyze only from description.",
        "note": "Cannot verify without code access"
    },
    "version_mismatch": {
        "condition": "issue.pytorch_version > env.pytorch_version",
        "action": "Mark limitation. Provide theoretical analysis.",
        "note": "May have different code paths"
    },
    "compatible": {
        "condition": "issue.pytorch_version <= env.pytorch_version AND not private",
        "action": "Execute reproduce test. Run deep analysis.",
        "note": "Can verify fix"
    }
}
```

---

## Quick Start Checklist

### Before Each Triage
- [x] Load skill files
- [x] Check PyTorch version in environment
- [x] Check IGC/Driver version
- [x] Check Triton version
- [x] Verify pytorch/third_party paths accessible

### During Triage
- [x] Verify version compatibility
- [x] Check private branch status
- [x] Extract reproduce info
- [x] Execute tests (if compatible)
- [x] Perform deep analysis
- [x] Identify dependencies
- [x] Generate fix suggestions

### After Triage
- [x] Report complete with all sections
- [x] Confidence assessed
- [x] Evidence cited

---

## Skills Version

Current version: 1.0.0
Last updated: 2026-04-20
Compatibility: PyTorch 2.12+, torch-xpu-ops 2.10+
