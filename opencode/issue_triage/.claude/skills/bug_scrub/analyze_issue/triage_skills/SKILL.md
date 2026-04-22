# PyTorch XPU-Ops Issue Triage - Complete Workflow

## Overview
This skill provides comprehensive tooling for triaging GitHub issues from intel/torch-xpu-ops repository. It integrates version-aware analysis, deep root cause investigation, explore agent usage for code exploration, and expert-level fix suggestions.

## When to Use
- Triage new and existing GitHub issues from torch-xpu-ops
- Deep root cause analysis beyond simple pattern matching
- Explore and understand codebase structure
- Version compatibility checking
- Environment verification for reproducible results

---

## Authoritative Reference (read this first)

This section is the **single source of truth** for per-issue triage output when populating the tracking Excel. All wave/batch agents must conform to these taxonomies and the JSON schema below. Anything else in this skill is guidance only.

### Canonical Output JSON Schema

Each triaged issue MUST be emitted as one object in a JSON array. Required keys — no wrapping (`{"results": [...]}` is forbidden):

```json
{
  "row":           <int>,      // Excel row (2..N). REQUIRED.
  "issue_id":      <int>,      // GitHub issue number. REQUIRED.
  "category":      "<string>", // From Category Taxonomy below. REQUIRED.
  "priority":      "P0|P1|P2|P3",
  "dependency":    "<string>", // From Dependency Taxonomy below. Use "" for blank.
  "root_cause":    "<string>", // 2-4 sentences, cite file:line.
  "fix_approach":  "<string>"  // Actionable next steps.
}
```

Output the JSON array directly. No markdown fences, no prose, no trailing commentary.

### Category Taxonomy (8 buckets — authoritative)

This is the production taxonomy used in the tracking Excel column "Category". Pick **exactly one**. See `SKILL_Category_Analysis.md` for detailed rubric + examples.

| Category | Use when |
|---|---|
| `Torch Operations` | aten/native operator issue (conv, linalg, reduce, batchnorm, indexing, pointwise, distributions, etc.) including numerical accuracy on a specific op |
| `Inductor` | torch.compile / Dynamo / AOTAutograd / Triton codegen / FakeTensor / ExportedProgram / benchmark suites running via inductor |
| `Distributed` | ProcessGroup/XCCL/DDP/FSDP/DTensor/symm_mem/collective ops; anything tagged `[distributed]` in title |
| `Flash Attention` | SDPA / `scaled_dot_product_attention` / flash/efficient attention kernels / MultiheadAttention |
| `Torch Runtime` | torch.xpu.* runtime APIs, memory management/OOM, device context, profiler, RNG, streams, IPC/share_memory |
| `TorchAO` | quantization paths: int4/int8 weight-only/dynamic, fp8, PT2E quant, torchao integration |
| `Sparse` | sparse tensors (COO/CSR/CSC/BSR), sparse ops, `sparse_csr_tensor` APIs |
| `Others` | CI/infra/tracking issues, build/doc/test-harness bugs, upstream benchmark harness gaps, release checklists, meta-tracking — the catch-all |

**Decision order** (resolve overlaps): `Distributed` > `Flash Attention` > `Inductor` > `TorchAO` > `Sparse` > `Torch Operations` > `Torch Runtime` > `Others`. Example: a `[distributed]` SDPA issue → `Distributed` (not Flash Attention).

Notes:
- "Feature Gap" is a *sub-type* surfaced in `fix_approach` text, NOT a Category value.
- "PT2E" rolls into `Inductor` (or `TorchAO` for PT2E quantization).
- "Build/Compilation", "Documentation", "CI/CD", "Test Infrastructure", "Numerical Accuracy" are descriptive sub-types; map them to the bucket above using the domain of the failing component (e.g., Numerical Accuracy on Conv3d → `Torch Operations`; CI infra → `Others`).

### Dependency Taxonomy (authoritative)

Pick **exactly one** value. Populates Excel column "Dependency".

| Value | Use when |
|---|---|
| `driver` | ocloc / IGC / libigc / intel-igc-cm / level-zero / compute-runtime / drm_neo / SYCL runtime bug / GPU segfault at driver layer |
| `xccl` | ProcessGroupXCCL / WorkXCCL / FlightRecorderXCCL / torch.xpu.xccl / oneCCL |
| `triton` | intel-xpu-backend-for-triton codegen/compile/lowering |
| `oneDNN` | oneDNN-backed op (conv*, SDPA, linear, quantized int8, _grouped_mm, etc.) |
| `oneMKL` | oneMKL-backed op (linalg.svd/qr/pinv/cholesky, BLAS paths) |
| `oneAPI` | oneAPI compiler/runtime version mismatch or compiler regression (CMPLRLLVM-*) |
| `CPU fallback` | XPU operator missing; CPU fallback registered in torch-xpu-ops |
| `SYCL kernel: <FileName.cpp>` | Bug in a specific SYCL kernel under `torch-xpu-ops/src/ATen/native/xpu/sycl/` — cite the file name |
| `upstream-pytorch` | Bug lives in pytorch/pytorch (Dynamo/Inductor logic, AOTAutograd, `_prims_common`, test-list sync, benchmark harness); fix PR goes to pytorch repo |
| `""` (blank) | Pure torch-xpu-ops internal issue (not an external dep, not upstream) — e.g., test-list maintenance, meta-tracking, doc cleanup inside torch-xpu-ops |

**Lookup for operator-based dependency**: `/home/daisydeng/ai_for_validation/opencode/issue_triage/xpu_supported_operators_complete_list.md`
- Part I (~L33): Implementation File Index by dependency (Native SYCL / oneMKL / oneDNN)
- Part II (~L136): Operator Registry
- Part IV (~L1143): CPU Fallback Operators

### Priority Taxonomy

P0 = crash/segfault/>5% perf regression/custom-model blocker · P1 = UT >20 failures or regression · P2 = E2E issues, few failures · P3 = minor/cosmetic. See `SKILL_Priority_Analysis.md`.

### Standard Investigation Pattern (per issue)

1. **Fetch**: `gh issue view <id> --repo intel/torch-xpu-ops --json title,body,labels,comments,state`
   - Fallback: `webfetch(url="https://github.com/intel/torch-xpu-ops/issues/<id>", format="markdown")`
2. **Locate** test/code/error — read relevant files, grep for the failing symbol.
3. **Cite** file:line evidence (torch-xpu-ops source and/or pytorch source).
4. **Classify** using the four taxonomies above and write the JSON entry.

### Pinned Reference Paths

| Purpose | Path |
|---|---|
| torch-xpu-ops source | `/home/daisydeng/pytorch/third_party/torch-xpu-ops/` |
| PyTorch source | `/home/daisydeng/pytorch/` |
| Operator → dependency lookup | `/home/daisydeng/ai_for_validation/opencode/issue_triage/xpu_supported_operators_complete_list.md` |
| CI op_ut XML logs | `/home/daisydeng/ai_for_validation/opencode/issue_triage/ci_results/torch-xpu-ops/Inductor-XPU-UT-Data-*/op_ut/*.xml` |
| Tracking Excel | `/home/daisydeng/ai_for_validation/opencode/issue_triage/result/torch_xpu_ops_issues.xlsx` |
| Agent workspace (scratch) | `/home/daisydeng/pytorch/agent_space/phase3_triage/` |

### For large-scale triage (many issues)

See `SKILL_Batch_Orchestration.md` for the wave-based parallel pattern (5 issues per batch × 5 parallel explore agents per wave × N waves → merge → single Excel write).

---

## Workflow Summary

```
┌─────────────────────────────────────────────────────────────────┐
│ STEP 1: Issue Acquisition & Version Detection                   │
├─────────────────────────────────────────────────────────────────┤
│ • Fetch issue (gh CLI or web fetch)                             │
│ • Extract versions: PyTorch, IGC, Triton, oneAPI               │
│ • Check private/unreleased branch status                        │
│ • Assess version compatibility                                   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 2: Reproduce Command Extraction                            │
├─────────────────────────────────────────────────────────────────────────┤
│ • Identify test case pattern from issue body                    │
│ • Enable explore agent for code exploration                      │
│ • Access PyTorch test code (~/pytorch/test/)                  │
│ • Access torch-xpu-ops test code (~/pytorch/../torch-xpu-ops/test/)  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 3: Code Exploration & Test Analysis                        │
├─────────────────────────────────────────────────────────────────┤
│ • Use explore agent to find implementation files                 │
│ • Locate test files in PyTorch (test_linalg.py, test_ops.py)    │
│ • Locate test files in torch-xpu-ops (test/xpu/)                │
│ • Analyze test expectations and assertions                      │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 4: Runtime Verification (if compatible)                    │
├─────────────────────────────────────────────────────────────────┤
│ • Execute reproduce command in conda env                        │
│ • Capture all test execution results                             │
│ • Compare with PyTorch upstream test behavior                   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 5: Deep Root Cause Analysis                                │
├─────────────────────────────────────────────────────────────────┤
│ • Multi-dimension analysis based on explore findings            │
│ • XPU implementation vs CPU fallback comparison                 │
│ • Kernel code investigation                                      │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 6: Dependency Analysis & Report                            │
└─────────────────────────────────────────────────────────────────┘
```

## Prerequisites

### Required Access & Paths
```bash
# Environment
source ~/miniforge3/bin/activate pytorch_opencode_env

# PyTorch source and test paths
~/pytorch                                    # PyTorch source root
~/pytorch/test/                              # PyTorch test directory
~/pytorch/test/test_linalg.py                # Example: linalg tests
~/pytorch/test/test_ops.py                  # Example: ops tests
~/pytorch/aten/src/ATen/native/             # ATen native implementations

# torch-xpu-ops source and test paths
~/pytorch/third_party/torch-xpu-ops         # XPU ops source root
~/pytorch/third_party/torch-xpu-ops/test/xpu/  # XPU ops test directory
~/pytorch/third_party/torch-xpu-ops/src/    # XPU ops implementation
```

### Source Code Locations Reference

| Component | Location Path | Purpose |
|-----------|---------------|---------|
| PyTorch ATen | `~/pytorch/aten/src/ATen/native/` | Native operator implementations |
| PyTorch Tests | `~/pytorch/test/test_*.py` | Upstream test cases |
| XPU Ops Native | `~/pytorch/third_party/torch-xpu-ops/src/ATen/native/` | XPU-specific kernels |
| XPU Ops Tests | `~/pytorch/third_party/torch-xpu-ops/test/xpu/test_*.py` | XPU-specific tests |
| SYCL Kernels | `~/pytorch/third_party/torch-xpu-ops/src/ATen/native/xpu/sycl/` | XPU kernel implementations |
| Operator Registry | `~/ai_for_validation/opencode/issue_triage/xpu_supported_operators_complete_list.md` | Dependency mapping |

### Version Detection Commands
```bash
# Core versions
python -c "import torch; print(torch.__version__)"
python -c "import torch; print(torch.xpu.get_device_properties(0).driver_version)"
python -c "import triton; print(triton.__version__)"

# Check all嘶
conda list | grep -E "intel|dpcpp|oneapi"
```

## Tool Reference

### 1. Explore Agent - Code Exploration
```python
# Use explore agent for comprehensive code search
task(description="explore_xpu_ops", 
     prompt="Find all implementations and tests related to OPERATOR_NAME in torch-xpu-ops\n\nRequirements:\n1. Search ~/pytorch for native implementations\n2. Search ~/pytorch/third_party/torch-xpu-ops for XPU-specific code\n3. List relevant test files in both PyTorch and torch-xpu-ops\n4. Identify implementation files, kernel files, and test files\n\nReturn:\n- List of implementation file paths\n- List of test file paths  \n- Key function signatures\n- Related kernel files",
     subagent_type="explore")

# Medium exploration level (default)
# Use "quick" for simple searches
# Use "very thorough" for comprehensive analysis
```

### 2. Issue Fetching
```python
# Method 1: GitHub CLI (requires gh auth login)
gh issue view {issue_number} --repo intel/torch-xpu-ops --json title,body,labels

# Method 2: Web fetch fallback
webfetch(url="https://github.com/intel/torch-xpu-ops/issues/{issue_number}", format="markdown")
```

### 3. Test File Access Patterns
```python
# Access PyTorch test files (upstream tests)
read(filePath="~/pytorch/test/test_linalg.py", offset=1700, limit=100)
# Structure: ~/pytorch/test/test_{module}.py

# Access torch-xpu-ops test files
read(filePath="~/pytorch/third_party/torch-xpu-ops/test/xpu/test_transformers_xpu.py", offset=1100, limit=50)
# Structure: ~/pytorch/third_party/torch-xpu-ops/test/xpu/test_{module}_xpu.py

# Search for specific test methods
grep(pattern="def test_.*xpu", path="~/pytorch/third_party/torch-xpu-ops/test/xpu", include="*.py")
```

### 4. Code Investigation
```python
# Search implementations across both repos
grep(pattern="operator_name", path="~/pytorch", include="*.cpp")
grep(pattern="operator_name", path="~/pytorch/third_party/torch-xpu-ops/src", include="*.cpp")

# Find implementation files
glob(pattern="**/aten/**/native/**/Attention*.cpp", path="~/pytorch")
glob(pattern="**/xpu/sycl/**/Attention*.cpp", path="~/pytorch/third_party/torch-xpu-ops")

# Read implementation
read(filePath="~/pytorch/third_party/torch-xpu-ops/src/ATen/native/transformers/sycl/AttentionKernels.cpp", offset=100, limit=50)
```

### 5. Runtime Execution
```python
# Execute in conda env
bash(command="source ~/miniforge3/bin/activate pytorch_opencode_env && python -c '<code>'", timeout=180000)
```

## Explore Agent Integration

### When to Use Explore Agent
```python
EXPLORE_SCENARIOS = {
    "sdpa_kernels": {
        "trigger": "scaled_dot_product_attention，听到看到这个操作",
        "depth": "medium",
        "requirements": "Explore SDPA implementations in both default and torch-xpu-ops"
    },
    "linalg_ops": {
        "trigger": "linalg.cond, linalg.svd",
        "depth": "medium", 
        "requirements": "Find linalg operations in native ATen and XPU overrides"
    },
    "test_investigation": {
        "trigger": "Need to understand test expectations",
        "depth": "quick",
        "requirements": "Locate test file and analyze test structure"
    }
}
```

### Explore Agent Usage Templates

#### Template 1: Operator Implementation Search
```python
task(description="operator_implementation_finder",
     prompt=f"""
Find COMPLETE implementation for OPERATOR: <operator_name>

Search in order:
1. ~/pytorch/aten/src/ATen/native/ - Native implementations
2. ~/pytorch/third_party/torch-xpu-ops/src/ATen/native/xpu/ - XPU-specific
3. ~/pytorch/third_party/torch-xpu-ops/src/ATen/native/transformers/sycl/ - SYCL kernels

For each file found:
- Read key sections (first 100 lines, important kernels)
- Identify kernel launch patterns
- Note any XPU fallback paths

Output:
- Implementation file paths
- Key kernel functions
- Fallback behavior if any""",
     subagent_type="explore")
```

#### Template 2: Test Case Investigation
```python
task(description="test_case_investigator",
     prompt=f"""
Investigate test case: <test_case_name> from TEST_PATH

Access test files:
- PyTorch tests: ~/pytorch/test/
- torch-xpu-ops tests: ~/pytorch/third_party/torch-xpu-ops/test/xpu/

For the test case:
1. Read the full test method
2. Identify assertions and expectations
3. Note any setup/teardown requirements
4. Check for related helper functions

Output:
- Full test code
- Test assertions
- Expected behavior
- Related fixtures""",
     subagent_type="explore")
```

#### Template 3: Root Cause Deep Dive
```python
task(description="root_cause_explorer",
     prompt=f"""
Deep investigation for issue with ERROR_PATTERN

Areas to investigate:
1. Implementation files for failing operator
2. XPU-specific kernel implementations  
3. Test expectations
4. CPU fallback paths

Key locations:
- ~/pytorch/aten/src/ATen/native/

- ~/pytorch/third_party/torch-xpu-ops/src/ATen/native/xpu/

- ~/pytorch/third_party/torch-xpu-ops/src/ATen/native/transformers/sycl/

For each investigation point:
- Read relevant code sections
- Identify potential fault locations
- Check data type handling
- Verify memory access patterns

Output:
- Potential root causes
- Code evidence
- Investigation notes""",
     subagent_type="explore")
```

## Deep Analysis Patterns

### Error Pattern -> Investigation Mapping

| Error Type | Indicators | Investigation | Common Root Causes |
|------------|-------------|----------------|-------------------|
| **SegmentationFault** | "page fault", "longjmp" | Buffer sizes, tiling | Large allocation >16K seq, uninitialized stack |
| **PrecisionError** | "nan", "inf", "accuracy" | Input dtypes, tolerance | Mixed precision, accumulator dtype mismatch |
| **APICompatibility** | "not implemented", "fallback" | XPU fallback detection | Missing XPU implementation, CPU path assumed |
| **WarningMismatch** | "Expected X got Y" | Warning generation paths | Extra internal operations, copyback issues |
| **DriverHardware** | "gpu aborted", "device error" | Driver version check | IGC bug, hardware limitation |

### Test Code Access Patterns

```python
# Pattern 1: Access PyTorch upstream test
def access_pytorch_test(test_module: str, line_offset: int, limit: int) -> str:
    """
    Access PyTorch test file.
    
    Args:
        test_module: Module name (e.g., "test_linalg.py")
        line_offset: Line to start reading from
        limit: Number of lines to read
        
    Returns:
        Test file content
    """
    test_path = f"~/pytorch/test/{test_module}"
    return read(filePath=test_path, offset=line_offset, limit=limit)

# Pattern 2: Access torch-xpu-ops test
def access_xpu_ops_test(test_module: str, line_offset: int, limit: int) -> str:
    """
    Access torch-xpu-ops test file.
    
    Args:
        test_module: Module name (e.g., "test_transformers_xpu.py")
        line_offset: Line to start reading from
        limit: Number of lines to read
        
    Returns:
        Test file content
    """
    test_path = f"~/pytorch/third_party/torch-xpu-ops/test/xpu/{test_module}"
    return read(filePath=test_path, offset=line_offset, limit=limit)

# Pattern 3: Search test methods
def search_test_methods(test_file: str, method_pattern: str):
    """
    Search for test methods in test files.
    
    Usage:
        search_test_methods("test_linalg.py", "def test_cond")
    """
    grep(pattern=method_pattern, 
         path=f"~/pytorch/test/{test_file}",
         include="*.py")
```

### Version-Aware Analysis

```python
def analyze_with_explore_and_version(issue_data: dict, env_info: dict) -> dict:
    """
    Analysis using explore agent and version context.
    
    Steps:
    1. Use explore to find relevant code
    2. Check versions for compatibility
    3. Analyze based on explore findings
    """
    
    # Step 1: Explore for relevant implementations
    explores = task(description="find_relevant_code",
                    prompt=f"""
Find implementations and tests for OPERATOR related to issue.

Check:
- ~/pytorch/aten/src/ATen/native/ for native code
- ~/pytorch/third_party/torch-xpu-ops/ for XPU code
- Both test directories

Return paths and key findings""",
                    subagent_type="explore")
    
    # Step 2: Version compatibility on analysis
    analysis = {
        "explore_findings": explores,
        "version_compatible": True,
        "confidence": "medium",
        "test_access_info": {}
    }
    
    # Step 3: Access test code
    if "test_linalg" in issue_data.get("test_file", ""):
        analysis["test_access_info"]["pytorch_test"] = read(
            filePath="~/pytorch/test/test_linalg.py",
            offset=issue_data.get("line", 1700),
            limit=100
        )
    
    return analysis
```

## Fix Suggestion Templates

### Template 1: Sequence Length Threshold
```cpp
// For memory allocation crashes
if (seq_len > MAX_SEQ_LEN_THRESHOLD) {
    TORCH_WARN("Large sequence detected, using math backend");
    return math_backend_path;
}
```

### Template 2: Dtype Promotion Fix
```cpp
// For precision issues
auto acc_dtype = at::accumulate_type<input_scalar_t, acc_scalar_t>::type;
auto query_promoted = query.to(acc_dtype);
```

### Template 3: Warning Suppression
```cpp
// For extra warning generation
if (out_tensor_provided && shapes_match(existing_out, result)) {
    // Skip unnecessary resize operation
    return existing_out;  // No copy needed
}
```

## Output Format

### Required Sections in Report

```markdown
# Triage Report - Issue #{issue}

## Issue Summary
[One sentence description]

## 1. Version Table
| Component | Issue Ver | Env Ver | Compatible |
|-----------|-----------|---------|------------|
| PyTorch | X.Y.Z | X.Y.Z | Yes/No |
| XPU Driver | X.XX | X.XX | Yes/No |
| Triton | X.X.X | X.X.X | Yes/No |

## 2. Reproduce Info
[Test case / command reference]

## 3. Code Exploration Findings
[Explore agent results for implementation and test files]

## 4. Root Cause Analysis
[Deep analysis with evidence]

## 5. Fix Suggestions
[Expert-level suggestions]

## 6. Priority & Labels
[Recommended priority and labels]
```

## Usage Example

### Complete Triage Run with Explore
```bash
# Step 1: Fetch issue
webfetch(url="https://github.com/intel/torch-xpu-ops/issues/3394", format="markdown")

# Step 2: Extract test case from issue
# Test case: test_cond_errors_and_warnings_xpu_float64

# Step 3: Explore agent for code investigation
task(description="sdpa_crash_investigation",
     prompt="Investigate sdpa crash issue:\n1. Find SDPA implementations in ~/pytorch/third_party/torch-xpu-ops/\n2. Locate related test files\n3. Identify kernel launch patterns for large sequences",
     subagent_type="explore")

# Step 4: Access test code
read(filePath="~/pytorch/test/test_linalg.py", offset=1735, limit=50)

# Step 5: Execute reproduce if compatible
bash(command="source ~/miniforge3/bin/activate pytorch_opencode_env && python -c '<reproduce>'")

# Step 6: Generate comprehensive report
```

## Category Analysis

Add SKILL_Category_Analysis.md to the triage workflow for automatic issue categorization:

```python
# Step X: Category Analysis
task(description="category_analysis",
     prompt="Analyze issue for category classification:\n\nCategories:\n1. Distributed - XCCL, NCCL, DDP, FSDP\n2. TorchAO - torchao, quantize_, int4/int8\n3. PT2E - torch.export(), Dynamo, fake_tensor\n4. Flash Attention - flash_attention, SDPA, attention\n5. Sparse - sparse tensor, BSR, CSR, COO\n6. Inductor - torch.compile(), Triton, codegen\n7. Torch Runtime - OOM, kernel launch, memory\n8. Torch Operations - aten::, native ops\n9. Others\n\nAnalyze issue text, stack trace, and code patterns\nto determine primary and secondary categories.",
     subagent_type="explore")

# Integrate with triage report
CATEGORIES = {
    "Flash Attention/Transformer": ["scaled_dot_product", "sdpa", "flash_attention"],
    "Torch Runtime": ["OOM", "page fault", "drm_neo"],
    "Torch Operations": ["linalg.cond", "aten::"],
    "Distributed": ["ProcessGroup", "NCCL", "XCCL"],
    "Sparse": ["sparse", "BSR", "CSR"],
}
```

## Priority Analysis

Add SKILL_Priority_Analysis.md for automatic priority classification:

```python
# Step Y: Priority Analysis
# Uses weighted scoring across multiple dimensions:
# - Error type (40%): Fatal/Error/Warning
# - Test failures (30%): Many/few failures
# - Regression (20%): Was passing, now failing
# - Performance (20%): >5% = P0
# - Custom impact (40%): Production model

# Priority levels:
# - P0: Critical (crash, segfault, >5% perf, custom model)
# - P1: High (UT >20 failures, regression)
# - P2: Medium (E2E issues, few failures)
# - P3: Low (minor, cosmetic)

# Example: SDPA crash would be P0
# Example: Warning mismatch would be P3
```

## Related Skills

| Skill | File | Purpose |
|-------|------|---------|
| Priority Analysis | SKILL_Priority_Analysis.md | Automatic priority classification |
| Category Analysis | SKILL_Category_Analysis.md | Automatic issue categorization |
| Deep Analysis | SKILL_Deep_Analysis_Patterns.md | Multi-dimension analysis logic |
| Domain Patterns | SKILL_Domain_Patterns.md | Quick reference & tools |
| Issue Extraction | SKILL.md (in parent) | Basic issue collection |

## Skill Metadata

- **Version**: 1.1.0
- **Created**: 2026-04-20
- **Updated**: 2026-04-20 (added explore agent integration)
- **Compatibility**: PyTorch 2.12+, torch-xpu-ops 2.10+
- **Requires**: GitHub access, Conda pytorch_opencode_env, explore agent