# Triage Bot Skill - PyTorch XPU-Ops Issue Triage

## Overview
This skill provides a structured approach to triage GitHub issues from the Intel torch-xpu-ops repository. It emphasizes deep root cause analysis over simple pattern matching, environment verification, and version-aware triage.

---

## Preconditions

### 1. Required Access & Environment
```
- Working directory: ~/ai_for_validation/issue_triage/ or configurable
- GitHub CLI (gh) authenticated OR web access for issue fetching
- Conda environment: ~/miniforge3/bin/activate pytorch_opencode_env
- Pytorch source at: ~/pytorch
- torch-xpu-ops at: ~/pytorch/third_party/torch-xpu-ops
- xpu_supported_operators_complete_list.md at: ~/ai_for_validation/opencode/issue_triage/
```

### 2. Version Detection Prerequisites
Before triage, always verify:
```bash
# Check PyTorch version
python -c "import torch; print(torch.__version__)"

# Check IGC/XPU driver version
python -c "import torch; print(torch.xpu.get_device_properties(0).driver_version)"

# Check Triton version
python -c "import triton; print(triton.__version__)"

# Check oneAPI components
conda list | grep -E "intel|dpcpp|oneapi"
```

### 3. Issue Version Detection
Extract from issue body:
- PyTorch version from Versions section
- Intel GPU driver versions
- Check if from private branch (unreleased features)
- Identify milestone/backlog context

---

## Tools

### 1. Issue Fetching Tools
```python
# Method 1: GitHub CLI (requires authentication)
gh issue view {issue_number} --repo intel/torch-xpu-ops --json title,body,labels,state,comments

# Method 2: Web Fetch (fallback)
webfetch(url="https://github.com/intel/torch-xpu-ops/issues/{issue_number}", format="markdown")
```

### 2. Explore Agent - Code Exploration
```python
# Use explore agent for comprehensive code search across PyTorch and torch-xpu-ops
task(description="explore_xpu_ops_code", 
     prompt="Explore OPERATOR implementations in both repos:\n\n1. PyTorch native: ~/pytorch/aten/src/ATen/native/\n2. XPU ops: ~/pytorch/third_party/torch-xpu-ops/src/ATen/native/\n3. SYCL kernels: ~/pytorch/third_party/torch-xpu-ops/src/ATen/native/xpu/sycl/\n4. Tests: ~/pytorch/test/ and ~/pytorch/third_party/torch-xpu-ops/test/xpu/\n\nFind implementation files, kernel files, and test files related to the operator.\nReturn paths and key function signatures.",
     subagent_type="explore")
```

### 3. PyTorch Test Code Access

**IMPORTANT**: Always access both PyTorch upstream tests AND torch-xpu-ops tests during analysis.

```python
# Access PyTorch upstream tests (~/pytorch/test/)
read(filePath="~/pytorch/test/test_linalg.py", offset=1700, limit=100)
# Structure: ~/pytorch/test/test_{module}.py

# Access torch-xpu-ops tests  
read(filePath="~/pytorch/third_party/torch-xpu-ops/test/xpu/test_transformers_xpu.py", offset=1000, limit=50)
# Structure: ~/pytorch/third_party/torch-xpu-ops/test/xpu/test_{module}_xpu.py

# Search test methods across both repos
grep(pattern="def test_.*xpu", path="~/pytorch/third_party/torch-xpu-ops/test/xpu", include="*.py")
grep(pattern="def test_cond", path="~/pytorch/test", include="*.py")
```

### 4. Code Analysis Tools
```python
# Search patterns in PyTorch
grep(pattern="<operator_name>", path="~/pytorch/aten/src/ATen/native/", include="*.cpp")

# Search patterns in torch-xpu-ops
grep(pattern="<operator_name>", path="~/pytorch/third_party/torch-xpu-ops/src", include="*.cpp")

# Search SYCL kernels specifically
grep(pattern="kernel|Kernel", path="~/pytorch/third_party/torch-xpu-ops/src/ATen/native/xpu/sycl/", include="*.cpp")

# Find implementation files
glob(pattern="**/Attention*.cpp", path="~/pytorch/third_party/torch-xpu-ops")

# Read implementation files
# For PyTorch native: ~/pytorch/aten/src/ATen/native/
# For XPU-specific: ~/pytorch/third_party/torch-xpu-ops/src/ATen/native/
read(filePath="<absolute_pytorch_path>", offset=<line>, limit=<count>)
read(filePath="<absolute_xpu_ops_path>", offset=<line>, limit=<count>)

# Bash for specific commands
bash(command="source ~/miniforge3/bin/activate pytorch_opencode_env && <cmd>", timeout=120000)
```

### Source Code Locations Reference

| Component | PyTorch Path | torch-xpu-ops Path |
|-----------|--------------|-------------------|
| Native ATen | `~/pytorch/aten/src/ATen/native/` | `~/pytorch/third_party/torch-xpu-ops/src/ATen/native/` |
| Test Files | `~/pytorch/test/test_*.py` | `~/pytorch/third_party/torch-xpu-ops/test/xpu/test_*_xpu.py` |
| SYCL Kernels | N/A | `~/pytorch/third_party/torch-xpu-ops/src/ATen/native/xpu/sycl/` |

### 3. Runtime Verification Tools
```python
# Test execution framework
bash(command="source ~/miniforge3/bin/activate pytorch_opencode_env && python -m pytest <test_path> -xvs")

# Interactive testing with result capture
bash(command="source ~/miniforge3/bin/activate pytorch_opencode_env && python -c \"<test_code>\"")
```

### 4. Version Comparison Tools
```python
# Determine version relationship
def compare_versions(issue_version: str, current_version: str) -> str:
    """
    Returns: 'older', 'same', 'newer', 'unknown'
    Version formats: X.Y.Z.suffix+commit or similar
    """
```

---

## Primary Triage Workflow

### Step 1: Issue Acquisition & Version Detection

#### 1.1 Fetch Issue Content
```python
# Acquire full issue information
issue_data = fetch_issue(github_url)

# Key fields to extract:
# - title: Issue summary
# - body: Full description with reproduce commands
# - labels: Issue categorization
# - state: Open/Closed
# - comments: Community feedback
```

#### 1.2 Extract Version Information
```python
# From Versions section in issue body
version_info = {
    "pytorch": extract_pattern(body, r"torch==([\d.]+)"),
    "intel_gpu_driver": extract_pattern(body, r"intel-opencl.*?(\d+\.\d+\.\d+)"),
    "oneapi": extract_pattern(body, r"dpcpp-cpp-rt==([\d.]+)"),
    "triton": extract_pattern(body, r"triton==([\d.]+)"),
}

# Check if private/unreleased branch
is_private_branch = (
    "private" in body.lower() or
    "unreleased" in body.lower() or
    "internal" in body.lower() or
    "staging" in body.lower()
)
```

#### 1.3 Version Compatibility Check
```python
def check_version_compatibility(issue_version: str, current_version: str) -> bool:
    """
    Determine if current environment can reproduce issue.
    
    Logic:
    1. Parse version numbers
    2. If issue_version is NEWER than current, note limitation
    3. If private branch, skip reproduction, analyze only from description
    4. Return compatibility flag
    """
    issue_ver = parse_version(issue_version)
    current_ver = parse_version(current_version)
    
    if issue_ver.major > current_ver.major:
        return False  # May not have the code path
    if is_beta_build(issue_version) and not is_beta_build(current_version):
        return False  # Different integration
    return True
```

### Step 2: Reproduce Command Extraction

#### 2.1 Priority Order
1. **Explicit reproduce command** in issue body
2. **Minimal reproducer** if provided
3. **Unit test case references**
4. **Pipeline/E2E test references**

#### 2.2 Extraction Logic
```python
def extract_reproduce_info(issue_body: str) -> dict:
    """
    Returns:
    {
        "type": "command|unittest|pipeline|e2e",
        "content": "<extracted code or test path>",
        "file": "<test file path if applicable>",
        "test_case": "<test case name if applicable>",
        "typical_cases": ["list of test cases to run"]
    }
    """
    
    # Check for code blocks (```python, ```bash)
    code_blocks = extract_markdown_code_blocks(issue_body)
    
    # Check for test case patterns
    test_patterns = [
        r"test_(\w+)_xpu_(\w+)",  # Pattern: test_xxx_xpu_yyy
        r"(\w+\.\w+\.\w+\.)Test(\w+)\.(\w+)",  # Pattern: module.TestClass.test_method
    ]
    
    if code_blocks:
        return {"type": "command", "content": code_blocks[0]}
    
    # Check for unittest references
    unittest_refs = extract_unittest_refs(issue_body)
    if unittest_refs:
        return {"type": "unittest", "typical_cases": select_typical_cases(unittest_refs)}
    
    # Check for pipeline references
    pipeline_refs = extract_pipeline_refs(issue_body)
    if pipeline_refs:
        return {"type": "pipeline", "cases": pipeline_refs}
```

### Step 3: Code Exploration (Using Explore Agent)

#### 3.0 Explore Agent Usage
**MANDATORY for thorough triage**: Use explore agent to investigate code paths.

```python
# Template: Operator Implementation Investigation
task(description="operator_impl_finder",
     prompt=f"""
INVESTIGATE OPERATOR: <operator_name>

Search locations in priority order:
1. ~/pytorch/aten/src/ATen/native/ - PyTorch native implementations
2. ~/pytorch/third_party/torch-xpu-ops/src/ATen/native/xpu/ - XPU-specific
3. ~/pytorch/third_party/torch-xpu-ops/src/ATen/native/transformers/sycl/ - SYCL kernels
4. ~/pytorch/test/ - PyTorch upstream tests
5. ~/pytorch/third_party/torch-xpu-ops/test/xpu/ - XPU ops tests

For each location:
- Read key implementation sections
- Identify kernel launch patterns
- Note XPU fallback paths if any
- Extract function signatures

Output structured findings:
- Implementation file paths
- Key kernel functions
- Test file locations
- Fallback behavior notes""",
     subagent_type="explore")
```

#### 3.1 Test Code Access - BOTH PyTorch AND torch-xpu-ops

**IMPORTANT**: Always access test files from BOTH repositories.

```python
# Access 1: PyTorch Upstream Test (for baseline/reference)
read(filePath="~/pytorch/test/test_linalg.py", 
     offset=<line_offset>, 
     limit=<lines_to_read>)

# Access 2: torch-xpu-ops Test (for XPU behavior)
read(filePath="~/pytorch/third_party/torch-xpu-ops/test/xpu/test_transformers_xpu.py",
     offset=<line_offset>,
     limit=<lines_to_read>)

# Compare test expectations between CPU/XPU
analysis += compare_cpu_xpu_test_expectations(pytorch_test, xpu_ops_test)
```

#### 3.3 Typical Case Selection Criteria
```python
def select_typical_cases(test_cases: list) -> list:
    """
    Select 2-3 representative test cases that cover:
    1. Primary failing pattern
    2. Edge cases if applicable
    3. Different dtypes/input shapes if significant
    
    Selection rules:
    - Prefer float32/float64 for numerical stability issues
    - Prefer float16/bfloat16 for precision issues
    - Include smallest working shape + failing shape
    - Limit to 3 cases max for efficiency
    """
    
    # Deduplicate by test method name
    unique_methods = set()
    typical_cases = []
    
    for case in test_cases:
        method = extract_test_method(case)
        if method not in unique_methods:
            unique_methods.add(method)
            typical_cases.append(case)
            if len(typical_cases) >= 3:
                break
    
    return typical_cases
```

#### 3.2 Test Case Format Standardization
```python
# Standard format for torch-xpu-ops tests
CASE_FORMAT = "op_ut,{module}.{TestClass},{test_method}_{device}_{dtype}"

# Examples:
"op_ut,third_party.torch-xpu-ops.test.xpu.test_ops_xpu.TestOperatorsXPU,test_vjpvmap_nn_functional_conv3d_xpu_float32"
"op_ut,third_party.torch-xpu-ops.test.xpu.test_linalg_xpu.TestLinalgXPU,test_cond_errors_and_warnings_xpu_float64"
```

### Step 4: Runtime Verification

#### 4.1 Environment Verification Checklist
```python
VERIFICATION_CHECKLIST = {
    "torch_version": "import torch; print(torch.__version__)",
    "xpu_driver": "torch.xpu.get_device_properties(0).driver_version",
    "igc_version": "Check via: ls /opt/intel/igc* or intel_gpu_top",
    "triton_version": "import triton; print(triton.__version__)",
    "oneapi_version": "conda list | grep intel",
    "memory_available": "torch.xpu.get_device_properties(0).total_memory",
}
```

#### 4.2 IGC Version Specific Checks
```python
def check_igc_version_compatibility(igc_ver: str) -> dict:
    """
    IGC (Intel Graphics Compiler) version affects:
    - Kernel compilation
    - SIMD instruction availability
    - Memory allocation patterns
    
    Known issues by IGC version:
    - IGC < 1.0.XXXX: Memory allocation bugs with large tensors
    - IGC 1.0.XXXX-1.0.XXXX: Page fault issues with XPU backend
    - IGC >= 1.0.XXXX: Better handling of nested tensors
    """
    
    # Parse and validate IGC version
    # Return compatibility notes
    return {
        "compatible": bool,
        "warnings": [list of known issues],
        "recommendations": [list of workarounds]
    }
```

#### 4.3 Runtime Execution Template
```python
def execute_reproduce_test(reproduce_command: str, timeout: int = 180000) -> dict:
    """
    Execute reproduce command in controlled environment.
    
    Returns:
    {
        "success": bool,
        "error_type": "SegmentationFault|RuntimeError|ValueError|...",
        "error_message": "truncated error message",
        "crash_location": "if available",
        "stack_trace": "last 20 lines of stderr",
        "exit_code": int
    }
    """
    
    cmd = f"""
    source ~/miniforge3/bin/activate pytorch_opencode_env && \
    conda run -n pytorch_opencode_env python -c "{reproduce_command}"
    """
    
    result = bash(command=cmd, timeout=timeout)
    
    return parse_execution_result(result)
```

### Step 5: Root Cause Deep Analysis

#### 5.1 Analysis Framework
```python
def deep_analyze_root_cause(issue_data: dict, reproduce_result: dict = None) -> dict:
    """
    Deep analysis considering multiple dimensions:
    
    1. Memory Access Patterns
       - Buffer allocation failures
       - Page fault issues
       - Memory fragmentation
    
    2. Kernel Implementation
       - SYCL kernel correctness
       - Triton kernel issues
       - oneDNN/oneMKL integration
    
    3. Type System
       - Dtype promotion/consistency
       - Precision handling
       - Overflow/underflow conditions
    
    4. Hardware Abstraction
       - XPU-specific paths
       - Driver/kernel compatibility
       - Memory alignment requirements
    
    5. API/ABI Changes
       - Upstream API modifications
       - Backend dispatch changes
       - Fallback behavior differences
    """
    
    analysis = {
        "primary_hypothesis": str,
        "evidence": [list of supporting code/evidence],
        "confidence": "high|medium|low",
        "related_files": [list of relevant source files],
        "test_approach": str
    }
    
    # Investigation logic based on error type
    error_type = reproduce_result.get("error_type", "unknown")
    
    if error_type == "SegmentationFault":
        analysis = analyze_segfault(reproduce_result)
    elif error_type == "RuntimeError":
        analysis = analyze_runtime_error(reproduce_result)
    elif error_type == "ValueError":
        analysis = analyze_value_error(reproduce_result)
    else:
        analysis = generic_analyze(reproduce_result)
    
    return analysis
```

#### 5.2 Common Error Patterns & Analysis

##### 5.2.1 Segmentation Fault Analysis
```python
def analyze_segfault(reproduce_result: dict) -> dict:
    """
    Segmentation fault patterns in XPU context:
    
    1. Page Fault from GPU (NotPresent PDE/PDP)
       - Cause: Memory not allocated/faulted in GPU VA space
       - Common with: Large tensors (>16K sequence), tiled operations
    
    2. Uninitialized Stack Frame (longjmp)
       - Cause: SYCL runtime interruption handling
       - Common with: Kernel compilation failures, aborted operations
    
    3. Invalid Memory Access
       - Cause: Pointer miscalculation in kernel
       - Common with: Nested tensor operations, non-contiguous access
    
    Analysis steps:
    1. Extract fault address from error log
    2. Check sequence length/c tensor size
    3. Examine tiling/blocking in kernel
    4. Verify memory allocation patterns
    """
    
    error_msg = reproduce_result.get("error_message", "")
    fault_addr = extract_hex_address(error_msg)
    
    # Identify pattern
    if "PDP" in error_msg or "PDE" in error_msg:
        pattern = "page_fault_gpu_va"
    elif "longjmp" in error_msg:
        pattern = "uninitialized_stack"
    else:
        pattern = "unknown_segfault"
    
    return {
        "pattern": pattern,
        "fault_address": fault_addr,
        "likely_causes": identify_causes(error_msg),
        "investigation": "Examine memory allocation for seq_len > 16384"
    }
```

##### 5.2.2 Precision/Dtype Issue Analysis
```python
def analyze_precision_issue(issue_data: dict) -> dict:
    """
    Precision issues requiring dtype comparison:
    
    1. Input dtype: Record all input tensor dtypes
    2. Intermediate dtype: Check dtype after each operation
    3. Output dtype: Verify against expected tolerance
    
    Analysis approach:
    1. List all dtypes from reproduce command
    2. Check oneDNN/MKL precision requirements
    3. Compare with upstream CUDA behavior
    """
    
    dtypes = extract_dtypes_from_command(issue_data.get("content", ""))
    
    return {
        "input_dtypes": dtypes,
        "expected_precision": "Check tolerance (atol/rtol)",
        "related_backend": "oneDNN|oneMKL|SYCL"
    }
```

##### 5.2.3 Warning Count Mismatch Analysis
```python
def analyze_warning_mismatch(issue_data: dict) -> dict:
    """
    Warning count mismatches:
    
    1. Tensor initializer warnings
       - Extra empty() calls from XPU backend paths
       - Copyback operations triggering additional warnings
    
    2. Type conversion warnings
       - Unexpected dtype promotions
    
    Analysis:
    1. Identify which operation generates extra warnings
    2. Compare XPU vs CPU warning paths
    3. Check user-provided out tensor handling
    """
    
    # Implementation-specific analysis required
    return {
        "expected_warnings": int,
        "actual_warnings": int,
        "extra_source": "To be identified via code analysis",
        "fix_approach": "Avoid redundant operations when out provided"
    }
```

### Step 6: Dependency Analysis

#### 6.1 Operator Dependency Registry
```python
# Load from xpu_supported_operators_complete_list.md
OPERATOR_DEPS = {
    "scaled_dot_product_attention": ["Triton", "oneDNN", "SYCL"],
    "linalg.cond": ["oneMKL"],
    # ... etc
}

def get_operator_dependencies(operator: str) -> list:
    """
    Look up operator in registry to get dependencies.
    """
    return OPERATOR_DEPS.get(operator, ["Unknown - check yaml files"])
```

#### 6.2 Version-Specific Dependency Issues
```python
def check_dependency_version(dependency: str, required_ver: str) -> dict:
    """
    Check if dependency version matches requirements.
    
    Common dependencies:
    - Triton: Version affects SDPA kernel selection
    - oneDNN: Version affects conv/matmul precision
    - IGC: Version affects kernel compilation
    - Driver: Version affects memory management
    """
    
    return {
        "installed_version": str,
        "required_version": str,
        "compatible": bool
    }
```

### Step 7: Fix Suggestion Generation

#### 7.1 Expert Fix Templates
```python
FIX_TEMPLATES = {
    "memory_allocation": """
    Suggested fix for memory allocation issue:
    
    1. Add boundary check before kernel launch:
    ```cpp
    if (seq_len > MAX_SEQ_LEN_THRESHOLD) {
        TORCH_WARN("Large sequence length detected, using math backend");
        return math_backend_path;
    }
    ```
    
    2. Implement chunked processing:
    ```cpp
    // Process in chunks to avoid >1GB allocation
    for (int i = 0; i < seq_len; i += CHUNK_SIZE) {
        process_chunk(params, i, min(i + CHUNK_SIZE, seq_len));
    }
    ```
    """,
    
    "precision_fix": """
    1. Check type promotion in input processing
    2. Verify accumulator dtype matches expected precision
    3. Compare tolerance values with upstream CUDA
    """,
    
    "warning_fix": """
    1. Avoid internal empty() calls when out tensor provided
    2. Check all warning generation paths
    3. Add early return when out tensor already matches requirements
    """
}
```

#### 7.2 Test Case Suggestion Format
```python
def generate_test_case_suggestion(fix_needs: dict) -> str:
    """
    Generate regression test suggestions.
    
    Format:
    ```python
    def test_<feature>_regression_<issue_id>(self):
        # Description of regression test
        # Issue: #<issue_number>
        ...
    ```
    """
    return f"""
    Suggested regression test:
    
    ```python
    # In appropriate test file
    def test_{fix_needs['feature']}(self):
        # Regression test for issue #{fix_needs['issue_id']}
        {fix_needs['test_code']}
        self.assert...</project>
    ```
    """
```

---

## Constraints

### 1. Environment Constraints
- Cannot reproduce if current PyTorch version is older than issue's PyTorch version
- Cannot reproduce if on private branch (no code access)
- Limited by available XPU hardware

### 2. Execution Constraints
- Maximum test runtime: 300 seconds per test case
- Memory limited by GPU VRAM
- No CUDA hardware available for comparison (XPU only)

### 3. Analysis Constraints
- Deep analysis requires code access to implementation files
- Root cause may be upstream PyTorch (not torch-xpu-ops)
- Some issues are driver-level, requiring vendor fix

### 4. Version Constraints
- If issue PyTorch version > current PyTorch, note as "cannot verify"
- If private branch, skip execution tests, analyze from description only
- IGC version mismatch may cause false positives/negatives

---

## Output Format

### Triage Report Template
```markdown
# Triage Report - Issue #{issue_number}

## Issue Summary
<One sentence description>

---

## 1. Reproduce Command / Test Case
<Extracted command or test case references>
Typical Cases:
- <case 1>
- <case 2>

## 2. Version Information
| Component | Issue Version | Environment Version | Compatible |
|-----------|---------------|---------------------|------------|
| PyTorch | X.Y.Z.dev+DATE | X.Y.Z.dev | Yes/No |
| IGC/Driver | X.XX.XXXXX | X.XX.XXXXX | Yes/No |
| Triton | X.X.X | X.X.X | Yes/No |

## 3. Root Cause Analysis
<Detailed analysis following framework>

### Evidence
- <code reference or error log excerpt>
- <related implementation file>

### Confidence: <high/medium/low>

## 4. Dependency Analysis
- Affected dependencies: Triton/oneDNN/IGC/SYCL
- Version-specific issues: <if applicable>

## 5. Fix Suggestions
<Expert recommendations with code snippets>

## 6. Test Case Recommendations
<Regression tests to add>

## 7. Priority & Assignment
- Priority: P1/P2/P3/P4
- Labels: <relevant labels>
- Assign: <recommended developer>
```

---

## Example: Complete Triage Run

### Issue #3394 Analysis Example

```python
# Step 1: Issue Acquisition
issue_url = "https://github.com/intel/torch-xpu-ops/issues/3394"
issue_data = webfetch(url=issue_url, format="markdown")

# Step 2: Version Detection
pytorch_issue_ver = "2.13.0.dev20260419+xpu"  # Extracted from Versions section
pytorch_current_ver = "2.13.0.dev20260419+xpu"  # Current environment

igc_issue_driver = "1.14.36300"  # From error log

# Step 3: Reproduce Command
reproduce_code = """
import torch
batch_size, seq_len = 4, 16413
query = torch.randn(batch_size, seq_len, 3, 16, device='xpu', dtype=torch.float32)
out = torch.nn.functional.scaled_dot_product_attention(query, query, query)
"""

# Step 4: Execution
result = execute_reproduce_test(reproduce_code)

# Step 5: Analysis
analysis = deep_analyze_root_cause(issue_data, result)
# Output:
# {
#     "pattern": "page_fault_gpu_va",
#     "confidence": "high",
#     "root_cause": "Memory allocation failure for seq_len > 16384",
#     "action": "Add sequence length thresholding"
# }

# Step 6: Dependency Check
operator_deps = get_operator_dependencies("scaled_dot_product_attention")
# ["Triton", "oneDNN"]

# Step 7: Fix Suggestion
fix = generate_fix_suggestion(analysis, operator_deps)
```

---

## Quality Assurance Checklist

Before completing triage, verify:
1. [ ] Issue PyTorch version compared with environment
2. [ ] IGC/driver version noted
3. [ ] Reproduce command executed (if not private branch)
4. [ ] Root cause analysis follows multi-dimension framework
5. [ ] Dependencies identified from operator registry
6. [ ] Fix suggestions are expert-level, not generic
7. [ ] Test case format standardized
8. [ ] Report follows template structure