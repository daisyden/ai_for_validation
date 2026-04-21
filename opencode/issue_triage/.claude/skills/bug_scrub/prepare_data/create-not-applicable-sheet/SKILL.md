# Create Not Applicable Sheet (Manual Deep Analysis)

## Base Path Reference

Relative paths from this file location (`bug_scrub/prepare_data/create-not-applicable-sheet/`):
```
../../../                    → issue_triage root
../../../result/            → Excel results directory
```../                      → WORKDIR (SKILL_DIR here)
```

## Overview

This skill creates a "Not Appliable" sheet for issues labeled with `wontfix` or `not_target`. Instead of using script-based pattern matching, this skill requires **deep analysis** using the Explore Agent to understand which torch operations or APIs are not being targeted and why.

## Distinction from Other Skills

- Uses deep investigation rather than surface pattern matching
- Focuses on understanding root cause of why feature/operation is not targeted
- Extracts specific technical information about unsupported operations

## Workflow

### Step 1: Identify Wontfix/Not Target Issues

From the Issues sheet in `../../../result/torch_xpu_ops_issues.xlsx`, filter issues with labels containing:
- `wontfix`
- `not_target`

### Step 2: For Each Identified Issue, Perform Deep Analysis

Use the Explore Agent to investigate each issue:

```python
task(description="not_applicable_deep_analysis",
     prompt=f"""
INVESTIGATION: Not Applicable Issue Analysis for Issue #{{issue_number}}

CONTEXT:
Issue titled: {{issue_title}}
Issue labels: {{issue_labels}}
This issue is marked as wontfix or not_target.

INVESTIGATION SCOPE:

1. TECHNICAL SCOPE ANALYSIS
   - Identify which torch operation or API is being requested
   - Determine if this is a CUDA-specific feature
   - Check if this is a deprecated or removed feature
   - Verify if this is an upstream PyTorch limitation
   - Identify if this requires hardware/ISA not available on XPU

2. COMPATIBILITY ASSESSMENT
   - Check torch-xpu-ops source for similar implementations
   - Review PyTorch core for relevant code
   - Verify upstream availability and compatibility

3. ROOT CAUSE CLASSIFICATION
   Classify the reason for not being targeted:
   - CUDA-specific implementation (not portable)
   - Hardware limitation (ISA/features unavailable)
   - Deprecated/removed feature (PyTorch deprecation)
   - Upstream decision (not part of PyTorch roadmap)
   - License restriction
   - Complexity/nationalization barrier
   - Third-party dependency unavailable

4. TECHNICAL SPECIFICS
   Extract specific:
   - ATen operator names if applicable
   - PyTorch API function signatures
   - Configuration parameters or requirements
   - Known limitations or constraints

SOURCE CODE LOCATIONS TO CHECK:
- ~/ai_for_validation/opencode/issue_triage/.claude/skills/bug_scrub/prepare_data/pytorch_xpu_backend_analysis/SKILL.md
- ~/ai_for_validation/pytorch/third_party/torch-xpu-ops/src/ATen/native/
- ~/ai_for_validation/pytorch/torch/

EXPECTED DELIVERABLES:
For each not_target/wontfix issue, provide:
1. Operation/API name (specific torch function or aten operator)
2. Category of reason (CUDA-specific, deprecated, hardware, etc.)
3. Technical details explaining why it's not targeted
4. Related torch operations if any workaround exists
     """,
     subagent_type="explore")
```

### Step 3: Document Findings

For each issue analyzed, populate the Not Applicable sheet with columns:

| Column | Description |
|--------|-------------|
| Issue ID | GitHub issue number |
| Title | Original issue title |
| Operation/API | Specific torch operation or API identified via deep analysis |
| Category | Classification of why not targeted (CUDA-specific, deprecated, etc.) |
| Technical Details | Explanation from deep analysis |
| Labels | Original labels |

### Step 4: Create Sheet in Excel

Manually create "Not Appliable" sheet in the Excel file with the documented findings.

## Usage

### Before Starting

1. Ensure Phase 1.1 (Issue Basic Info Extraction) is complete
2. Ensure XPU Backend Analysis (1.4) skill documentation is reviewed
3. Load PyTorch source environment for deep exploration

### Execute Deep Analysis

```python
# For each issue with wontfix/not_target labels:
# 1. Fetch issue details from GitHub
gh issue view {issue_number} --repo intel/torch-xpu-ops --json title,body,labels

# 2. Use Explore Agent for deep analysis
# (See prompts above)
```

### Manual Entry to Excel

After completing deep analysis for all wontfix/not_target issues, manually add entries to "Not Appliable" sheet with:
- Accurate operation/API names from investigation
- Proper categorization based on technical findings
- Detailed technical explanations

## Note on Why Deep Analysis Required

The "not applicable" determination requires understanding beyond simple pattern matching:

1. **Complex Relationships**: Why an operation isn't targeted often involves multiple technical factors
2. **Context Dependency**: Same operation might be unavailable for different reasons in different contexts
3. **Evolution Over Time**: Technical decisions change as torch-xpu-ops and PyTorch evolve
4. **Interdependencies**: Understanding why something isn't supported often requires tracing through multiple layers

Deep analysis ensures accurate documentation of why operations are not targeted, which is crucial for:
- Future planning and prioritization
- Communication with issue reporters
- Understanding the scope of XPU limitations

## Output

Creates "Not Appliable" sheet in `../../../result/torch_xpu_ops_issues.xlsx` with columns:
- Issue ID
- Title  
- Operation/API
- Category
- Technical Details
- Labels