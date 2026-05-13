# Triage Skills Map - Quick Reference

## Skill Activation Commands

### Load Skills
```
/home/daisydeng/ai_for_validation/issue_triage/skills/triage_skills/SKILL_Triage_Logic.md
/home/daisydeng/ai_for_validation/issue_triage/skills/triage_skills/SKILL_Deep_Analysis_Patterns.md
/home/daisydeng/ai_for_validation/issue_triage/skills/triage_skills/SKILL_Domain_Patterns.md
```

### Alternative: Inline Skill Loading
Write skills directly from tool documentation to current context.

---

## Preconditions Summary

### Required Access
- GitHub CLI or web access to `github.com/intel/torch-xpu-ops`
- Conda env: `pytorch_opencode_env`
- Source paths:
  - `~/pytorch` (PyTorch source)
  - `~/pytorch/third_party/torch-xpu-ops` (XPU ops)
  - `~/ai_for_validation/opencode/issue_triage/xpu_supported_operators_complete_list.md` (Operator registry)

### Version Detection
```bash
# Essential version checks
python -c "import torch; print(torch.__version__)"           # PyTorch
python -c "import torch; print(torch.xpu.get_device_properties(0).driver_version)"  # XPU Driver
python -c "import triton; print(triton.__version__)"         # Triton
conda list | grep -E "intel|dpcpp"                           # oneAPI components
```

---

## Workflow Summary

### Step 1: Issue Acquisition
```python
# Method 1: gh CLI (if authenticated)
gh issue view {issue} --json title,body,labels,state

# Method 2: Web fetch fallback
webfetch(url=f"https://github.com/intel/torch-xpu-ops/issues/{issue}", format="markdown")
```

### Step 2: Version Check & Compatibility
```python
def check_version_compatibility(issue_body: str) -> dict:
    """
    Extract and compare versions.
    
    Returns:
    {
        "pytorch_issue": "2.x.x",
        "is_private_branch": bool,
        "can_reproduce": bool,
        "version_note": str
    }
    """
```

### Step 3: Reproduce Extraction
Extract in priority order:
1. Explicit code block with reproduce command
2. Unit test case reference (format: `op_ut,module.TestClass,test_method`)
3. Pipeline/E2E test reference

### Step 4: Execution & Verification
```python
# Only if not private branch AND version compatible
source ~/miniforge3/bin/activate pytorch_opencode_env
python -c "<reproduce_command>"
```

### Step 5: Deep Analysis
Multi-dimension analysis for:
- Memory patterns (page fault, allocation)
- Kernel implementation (crash, UB)
- Precision/dtype (numerical errors)
- API compatibility (not implemented)
- Driver/hardware (device errors)

### Step 6: Dependency Check
Run this only after deep analysis has drafted root cause and fix approach.
Use the diagnosed failing component and proposed fix path as primary evidence,
then confirm operator-based dependencies with the registry.

```python
# From xpu_supported_operators_complete_list.md
grep("operator_name", path, include="*.md")  # Find operator dependencies
```

### Step 7: Fix Generation
Provide expert-level suggestions with code references.

---

## Tool Reference

| Tool | Use | Key Parameters |
|------|-----|----------------|
| `bash` | Execute commands | `command`, `timeout`, `workdir` |
| `read` | Read files | `filePath`, `offset`, `limit` |
| `write` | Write files | `content`, `filePath` |
| `edit` | Modify files | `filePath`, `oldString`, `newString` |
| `grep` | Search patterns | `pattern`, `path`, `include` |
| `glob` | Find files | `pattern`, `path` |
| `task` | Subagent | `description`, `prompt`, `subagent_type` |
| `webfetch` | Fetch URLs | `url`, `format` |
| `question` | Ask user | `questions` |

---

## Constraint Checklist

### Environment Constraints
- [x] Conda env activated
- [x] Python path correct
- [x] XPU device available

### Execution Constraints
- [x] Timeout < 300s per test
- [x] Memory within GPU limits

### Version Constraints
- [x] Private branch → analyze only, skip execution
- [x] Older PyTorch → note limitation, analyze theory

---

## Version-Aware Triage Logic

### Decision Tree

```
Is issue PyTorch version > current PyTorch version?
├── YES → Mark "Cannot verify - older version"
│        Analyze from description only
│        Provide theoretical fix
└── NO → Continue...

Is issue from private/unreleased branch?
├── YES → Mark "Private branch - code access limited"
│        Analyze description/comments only
│        Skip execution tests
└── NO → Continue...

Is version compatible?
├── YES → Execute reproduce test
│        Run deep analysis
│        Provide verified fix
└── NO → Mark limitation
         Provide theoretical fix
```

### Version Format Patterns
```python
# Development versions
"2.13.0.dev20260419+xpu"
"2.12.0a0+gitd0d73b1"

# Release versions
"2.12.0"
"2.11.0"

# Parse pattern
r"(\d+)\.(\d+)\.(\d+)(.?dev|alpha|beta|rc)(\d*)"
```

---

## Output Expectations

### Required Sections
1. Issue Summary (1 sentence)
2. Reproduce Command/Test Case
3. Version Information Table
4. Root Cause Analysis
5. Fix Suggestions
6. Test Case Recommendations
7. Priority & Assignment

### Quality Bar
- Version table complete
- Confidence assessed
- Evidence cited
- Fixes are specific (not generic)
