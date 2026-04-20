# XPU Supported Operators Analysis Skill

> **Skill Name**: xpu_supported_operators  
> **Version**: 1.0  
> **Created**: April 2026  
> **Purpose**: Comprehensive analysis of PyTorch XPU backend operator dependencies, implementations, and registry structure  

---

## Skill Overview

This skill provides deep analysis capabilities for understanding PyTorch XPU backend operator implementations, dependency libraries, and registration mechanisms. Unlike simple pattern matching approaches, this skill employs systematic codebase exploration to map 749+ operators to their implementation files and dependency specifications.

### Core Capabilities

- **Complete Operator Registry Analysis**: Process `yaml/xpu_functions.yaml` to enumerate all XPU-supported operators
- **Dependency Library Mapping**: Identify which operators use SYCL, oneMKL, oneDNN, or CPU fallback paths
- **Implementation File Tracking**: Trace operators to their source implementation files
- **Registration Mechanism Understanding**: Analyze torchgen code generation and dispatcher registration
- **Fallback Pattern Detection**: Identify operators requiring CPU fallback execution

### Target Repository

```
/home/daisydeng/test_pytorch/workdir/pytorch/third_party/torch-xpu-ops
```

---

## Tool Requirements

### Primary Analysis Tools

| Tool | Purpose | Key Functions |
|------|---------|---------------|
| **read** | Read file contents | Access YAML specs, source files, templates |
| **glob** | Pattern-based file search | Locate implementation files |
| **grep** | Content-based search | Find operator definitions, dependencies |
| **bash** | Command execution | Count files, analyze directory structures |
| **task** (explore agent) | Deep codebase exploration | Comprehensive pattern identification |

### Supporting Tools

| Tool | Purpose |
|------|---------|
| **read.offset** | Navigate large files with line ranges |
| **read.limit** | Control output volume for analysis |
| **glob pattern recursion** | Find nested directory implementations |
| **bash pipeline** | Aggregate statistics and counts |

### Tool Usage Patterns

#### File Reading Strategy

```python
# For YAML operator specifications
read(
  filePath="path/to/xpu_functions.yaml",
  offset=1,
  limit=800  # Full file for complete enumeration
)

# For implementation files
read(
  filePath="path/to/Activation.cpp",
  offset=1,
  limit=100  # Incremental for large files
)

# For templates
read(
  filePath="path/to/XPUFallback.template",
  offset=1,
  limit=300  # Fixed template size
)
```

#### File Search Strategy

```python
# Find all implementation files
glob(
  pattern="src/ATen/native/xpu/*.cpp"
)

# Find SYCL kernels
glob(
  pattern="src/ATen/native/xpu/sycl/*.cpp"
)

# Find MKL implementations
glob(
  pattern="src/ATen/native/xpu/mkl/*.cpp"
)
```

#### Content Search Strategy

```python
# Find dependency declarations
grep(
  pattern="USE_ONEMKL_XPU|USE_DNNL_XPU",
  path="src/ATen/native/xpu",
  include="*.cpp"
)

# Find registration patterns
grep(
  pattern="REGISTER_XPU_DISPATCH|TORCH_IMPL_FUNC",
  path="src/ATen/native/xpu",
  include="*.cpp"
)

# Find fallback implementations
grep(
  pattern="xpu_fallback|cpu_fallback",
  path="src/ATen/native/xpu",
  include="*.cpp"
)
```

---

## Preconditions

### Environment Requirements

| Requirement | Specification | Validation |
|-------------|---------------|------------|
| **Repository Access** | torch-xpu-ops at specified path | Verify directory exists |
| **Python Runtime** | Python 3.8+ with PyTorch build tools | Check `torchgen` availability |
| **File System Access** | Read access to source directories | Validate permissions |

### Prerequisite Knowledge

| Knowledge Area | Required Understanding |
|----------------|----------------------|
| **PyTorch ATen System** | Operator definition through YAML specs |
| **torchgen Code Generation** | Transformation of YAML to C++ registration |
| **Dispatch System** | Backend selection and routing mechanism |
| **SYCL Programming Model** | Parallel kernel execution on Intel GPUs |

### Project Structure Prerequisites

```
torch-xpu-ops/
├── cmake/                      # Must exist
│   ├── Codegen.cmake         # torchgen orchestration
│   ├── ONEMKL.cmake          # oneMKL configuration
│   └── ...                   # Other configs
├── src/ATen/native/xpu/       # Main implementations
│   ├── *.cpp                 # 106+ implementation files
│   ├── mkl/                  # oneMKL wrappers
│   └── sycl/                # SYCL kernels (201 files)
├── yaml/                      # Operator specifications
│   └── xpu_functions.yaml   # 749 operator definitions
└── tools/                    # Code generation tools
```

### Codebase State Requirements

| Requirement | Description |
|-------------|-------------|
| **Build System Functional** | CMake configuration complete |
| **YAML Specifications Valid** | No syntax errors in xpu_functions.yaml |
| **Implementation Files Present** | All referenced files exist |

---

## Constraints and Limitations

### Analysis Constraints

| Constraint | Description | Mitigation |
|------------|-------------|-----------|
| **Large File Handling** | yaml/xpu_functions.yaml is 749 lines | Use offset/limit parameters |
| **Symbol Resolution** | Function pointers prevent direct call graph | Analyze registration patterns instead |
| **Dynamic Dispatch** | Runtime selection complicates static analysis | Focus on registration macros and configs |
| **Conditional Compilation** | `#ifdef` blocks create code path variations | Check multiple build configurations |

### Technical Limitations

| Limitation | Impact | Analysis Approach |
|------------|--------|-------------------|
| **Partial Build Samples** | Not all targets compile all operators | Cross-reference multiple files |
| **Code Generation Overhead** | torchgen produces artifacts | Analyze template and generated code |
| **Library Binding Complexity** | External library linkage not visible | Trace through wrapper layers |

### Logical Constraints

| Constraint | Description | Precedence |
|------------|-------------|------------|
| **Deterministic Execution** | Analysis must produce consistent results | Fixed tool sequences |
| **Reproducibility** | Same input -> same operator mapping | Document tool order |
| **Completeness** | Must cover all 749 operators | Validate against YAML count |

---

## Core Analysis Logic

### Phase 1: Enumerate Operator Registry

```python
# Step 1.1: Read complete operator list from YAML
operator_list = read(
  filePath="yaml/xpu_functions.yaml",
  offset=5,  # Skip configuration directives (lines 1-4)
  limit=745  # 749 total lines minus 4 config lines
)

# Step 1.2: Parse configuration directives
config_directives = {
  "backend": "XPU",           # line 1
  "cpp_namespace": "at",     # line 2
  "use_out_as_primary": true, # line 3
  "device_guard": true        # line 4
}

# Step 1.3: Extract operator names
# Pattern: Lines 5-749 contain "- operator.name" format
operators = parse_yaml_operators(operator_list)
# Result: List of 745 operator signatures
```

### Phase 2: Categorize Dependencies

```python
# Step 2.1: Identify oneMKL integration patterns
mkl_files = glob(pattern="src/ATen/native/xpu/mkl/*.cpp")
# Expected: [BlasImpl.cpp, SpectralOps.cpp, BatchLinearAlgebra.cpp]

# Step 2.2: Check conditional compilation blocks
mkl_patterns = grep(
  pattern=r"#if\s+defined\(USE_ONEMKL_XPU\)",
  path="src/ATen/native/xpu",
  include="*.cpp"
)
# Parses both conditional and fallback paths

# Step 2.3: Identify oneDNN integration patterns
dnn_files = glob(pattern="aten/src/ATen/native/mkldnn/xpu/*.cpp")

# Step 2.4: Categorize by implementation type
dependency_categories = {
  "SYCL_NATIVE": operators_in_sycl_directory,
  "ONEMKL_INTEGRATION": operators_with_mkl_conditional,
  "ONEDNN_INTEGRATION": operators_with_dnn_conditional,
  "CPU_FALLBACK": operators_in_fallback_template
}
```

### Phase 3: Map Implementation Files

```python
# Step 3.1: Enumerate all XPU implementation files
main_impl_files = glob(pattern="src/ATen/native/xpu/*.cpp")
# Count: 106 files for main operators

sycl_kernel_files = glob(pattern="src/ATen/native/xpu/sycl/*.cpp")
# Count: 201 SYCL kernel files

mkl_impl_files = glob(pattern="src/ATen/native/xpu/mkl/*.cpp")
# Count: 6 files (3 .cpp + 3 .h)

# Step 3.2: Extract registration patterns
# Pattern: TORCH_IMPL_FUNC, REGISTER_XPU_DISPATCH
registration Patterns = grep(
  pattern="TORCH_IMPL_FUNC\(|REGISTER_XPU_DISPATCH\(",
  path="src/ATen/native/xpu",
  include="*.cpp"
)
# Maps function names to registration declarations

# Step 3.3: Track operator-to-file associations
operator_file_mapping = correlate(
  registration_patterns,
  operator_signatures
)
```

### Phase 4: Analyze Fallback Mechanisms

```python
# Step 4.1: Read XPUFallback.template
fallback_template = read(
  filePath="src/ATen/native/xpu/XPUFallback.template",
  offset=1,
  limit=269
)

# Step 4.2: Identify fallback control flow
# Lines 183-199: Environment variable control
# Lines 206-220: Selective fallback registration
# Lines 225-257: Hardcoded fallback list

# Step 4.3: Extract hardcoded fallback operators
hardcoded_fallback = extract_fallback_list(
  fallback_template,
  line_range=[225, 262]
)
```

### Phase 5: Validate Implementation Completeness

```python
# Step 5.1: Count total implementation files
total_impl_files = count_files(
  patterns=[
    "src/ATen/native/xpu/*.cpp",
    "src/ATen/native/xpu/sycl/*.cpp",
    "src/ATen/native/xpu/mkl/*.cpp"
  ]
)
# Expected: ~313 total files (excluding headers)

# Step 5.2: Cross-reference with YAML counts
yaml_operator_count = 749
registered_operator_count = count_registrations()
# Expected: Match or discrepancy tolerance < 5

# Step 5.3: Verify dependency distribution
expected_distribution = {
  "SYCL_NATIVE": 412,   # ~55%
  "ONEMKL": 34,         # ~5%
  "ONEDNN": 40,         # ~5%
  "CPU_FALLBACK": 263   # ~35%
}
```

---

## Implementation Details Analysis

### YAML Specification Analysis

#### Configuration Directive Interpretation

```yaml
# yaml/xpu_functions.yaml structure
backend: XPU                    # Backend identifier for dispatcher
cpp_namespace: at              # C++ namespace qualification
use_out_as_primary: true      # Output tensor optimization
device_guard: true             # Device context management
```

**Interpretation**:
- `backend: XPU` establishes dispatch routing to XPU device
- `cpp_namespace: at` ensures proper symbol qualification
- `use_out_as_primary` configures preference for in-place semantics
- `device_guard` enables automatic device context verification

#### Operator Entry Format

```yaml
# Standard operator entry
- operator.name.variant  # Single dispatch target

# Examples:
- add.Tensor            # Basic addition
- add.out               # Out-variant
- add_.Tensor           # In-place variant
- gelu_backward.grad_input  # Named output variant
```

### Dependency Library Analysis

#### SYCL Runtime Integration

```cpp
// SYCL kernel submission pattern
TORCH_IMPL_FUNC(relu_out_xpu)(
    const Tensor& input,
    const Tensor& output
) {
    c10::DeviceGuard guard(input.device());  // Ensure device context
    auto& queue = at::xpu::getCurrentSYCLQueue();  // Get SYCL queue
    
    // TensorIterator configuration
    auto iter = TensorIteratorConfig()
                    .add_output(output)
                    .add_const_input(input)
                    .build();
                    
    // Kernel submission
    sycl_kernel_submit(*iter.numel(), wg_size, queue, kernel, iter);
}
```

**Key Characteristics**:
- Immediate SYCL queue retrieval
- TensorIterator-based iteration
- DeviceGuard enforcement

#### oneMKL Integration Pattern

```cpp
// Conditional compilation for oneMKL
#if defined(USE_ONEMKL_XPU)
    // Optimized path: Direct MKL delegation
    return native::xpu::gemm_xpu_mkl(self, mat2, result);
#else
    // Fallback path: Real GEMM decomposition
    TORCH_WARN_ONCE("oneMKL not available, using fallback");
    return mm_complex_fallback(self, mat2, result);
#endif
```

**Build Configuration**:
- `USE_ONEMKL_XPU` flag controls optimization paths
- Conditional compilation enables/disables MKL integration
- Fallback maintains functional correctness without oneMKL

#### oneDNN Integration Pattern

```cpp
// oneDNN convolution primitive setup
struct ConvParams {
    std::vector<int64_t> stride;
    std::vector<int64_t> padding;
    std::vector<int64_t> dilation;
    int64_t groups{};
    // ...
};

// Primitive-based execution
dnnl::convolution_forward Primitive execution;
```

**Hardware Optimization**:
- Winograd transforms for 3x3 kernels
- Depthwise acceleration
- Quantization support (int4/uint4)

### Registration Mechanism Deep Analysis

#### torchgen Code Generation Pipeline

```cmake
# cmake/Codegen.cmake configuration
set(XPU_CODEGEN_COMMAND
  "${Python_EXECUTABLE}" -m torchgen.gen
  --source-path ${CODEGEN_XPU_YAML_DIR}
  --install-dir ${BUILD_TORCH_XPU_ATEN_GENERATED}
  --per-operator-headers
  --backend-whitelist XPU SparseXPU SparseCsrXPU NestedTensorXPU
)
```

**Generated Artifacts**:
| Artifact | Purpose |
|----------|---------|
| `RegisterXPU_0.cpp` | Standard tensor operation registration |
| `XPUFunctions.h` | Public header declarations |
| `XPUFunctions_inl.h` | Inline implementations |
| `c_shim_xpu.*` | AOTI integration interfaces |

#### Dispatch Registration Patterns

```cpp
// Pattern 1: Dispatch Stub Registration
REGISTER_XPU_DISPATCH(threshold_stub, &xpu::threshold_kernel);
REGISTER_XPU_DISPATCH(elu_stub, &xpu::elu_kernel);

// Pattern 2: Direct Implementation
TORCH_IMPL_FUNC(gelu_out_xpu)(...) {
    xpu::gelu_kernel(*this, approximate);
}

// Pattern 3: Template-based Registration
// Generated by torchgen for each operator
```

### Fallback System Architecture

#### CPU Fallback Control Flow

```
Operator Invocation
        │
        ▼
┌───────────────────────┐
│ Check Environment    │
│ PYTORCH_ENABLE_XPU_   │
│ FALLBACK              │
└───────────┬───────────┘
            │
      ┌─────┴─────┐
      ▼           ▼
   Enabled     Disabled
      │           │
      ▼           ▼
┌───────────────┐
│ xpu_fallback │  ──▶ Lazy Registration Check ──▶ xpu_fallback_impl
└───────────────┘            │
                              ▼
                      ┌───────────────┐
                      │ Device Check  │
                      │ Consistency   │
                      └───────────────┘
                              │
                              ▼
                      ┌───────────────┐
                      │ cpu_fallback  │  ──▶ CPU Execution
                      │ (native)      │
                      └───────────────┘
```

#### Fallback List Management

```cpp
// XPUFallback.template lines 225-257
// Hardcoded fallback for unsupported operations
std::vector<std::string> fallback_list = {
  "cholesky",
  "linalg_eig",
  "_efficient_attention_forward",
  // ...
};
```

### Implementation File Associations

#### Direct File Mapping

| Operator Category | Primary File | Secondary Files |
|------------------|---------------|-----------------|
| **Activation Functions** | Activation.cpp (133 lines) | sycl/Activation*.cpp (14 files) |
| **Binary Operations** | BinaryOps.cpp (184 lines) | Direct TensorIterator |
| **Linear Algebra** | Blas.cpp (489 lines) | mkl/BlasImpl.cpp, mkl/BatchLinearAlgebra.cpp |
| **CNN Operations** | oneDNN path | Conv.cpp, Linear.cpp, BatchNorm.cpp |

#### Indirect File Dependencies

| Implementation | Dependencies |
|----------------|-------------|
| **Activation.cpp** | sycl/ActivationGeluKernel.*, ... |
| **Blas.cpp** | mkl/BlasImpl.cpp (+ conditional) |
| **BatchNorm.cpp** | oneDNN primitives |

---

## Workflow Examples

### Example 1: Single Operator Dependency Lookup

```python
# GOAL: Find dependency for "mm.out" operator

# Step 1: Search registration patterns
mm_registration = grep(
  pattern=r"mm.*out.*xpu",
  path="src/ATen/native/xpu",
  include="*.cpp"
)
# Found: Blas.cpp contains "mm_out_xpu" implementation

# Step 2: Read implementation file section
blas_cm = read(
  filePath="src/ATen/native/xpu/Blas.cpp",
  offset=1,
  limit=50  # See conditional compilation at lines 27+
)

# Step 3: Check for MKL integration
mkl_check = grep(
  pattern="#if defined|USE_ONEMKL_XPU",
  path="src/ATen/native/xpu/Blas.cpp",
  include="*.cpp"
)

# Step 4: Determine dependency
result = {
  "operator": "mm.out",
  "implementation_file": "Blas.cpp",
  "dependencies": ["SYCL", "oneMKL"],
  "fallback": "real GEMM decomposition"
}
```

### Example 2: Category-Wise Operator Enumeration

```python
# GOAL: Enumerate all activation functions

# Step 1: Read YAML entries
yaml_entries = read(
  filePath="yaml/xpu_functions.yaml",
  offset=121,  # First activation entry
  limit=70     # Activation range
)

# Step 2: Filter activation operators
activation_patterns = [
  "relu", "gelu", "silu", "mish", "softplus",
  "hardtanh", "hardsigmoid", "hardswish", "leaky_relu"
]

activation_operators = filter_operators(
  yaml_entries,
  activation_patterns
)
# Result: 62 activation operators

# Step 3: Read implementation file
activation_ops = read(
  filePath="src/ATen/native/xpu/Activation.cpp",
  offset=1,
  limit=10  # Contains registration entries
)

# Step 4: Verify kernel count
activation_kernels = glob(
  pattern="src/ATen/native/xpu/sycl/Activation*.cpp"
)
# Result: 14 activation kernel files
```

### Example 3: Fallback Operator Analysis

```python
# GOAL: Identify all operators requiring CPU fallback

# Step 1: Read fallback template hardcoded list
fallback_list = read(
  filePath="src/ATen/native/xpu/XPUFallback.template",
  offset=225,
  limit=40  # Contains fallback_list definition
)

# Step 2: Extract hardcoded fallbacks
hardcoded_fallbacks = parse_fallback_list(fallback_list)
# Result: 32 operators explicitly fallback

# Step 3: Identify dynamic fallbacks
dynamic_check = grep(
  pattern="PYTORCH_ENABLE_XPU_FALLBACK|PYTORCH_XPU_FALLBACK_OP",
  path="src/ATen/native/xpu",
  include="*.cpp"
)

# Step 4: Categorize fallback types
fallback_categories = {
  "hardcoded": hardcoded_fallbacks,
  "environment_configured": dynamic_fallbacks,
  "catchall": implicit_fallbacks
}
```

### Example 4: Dependency Distribution Analysis

```python
# GOAL: Calculate operator distribution across dependencies

# Step 1: Count SYCL implementations
sycl_operators = count_sycl_registrations()  # Via grep + analysis
# Formula: Registration pattern frequency

# Step 2: Count MKL integrations
mkl_operators = count_mkl_conditionals()
# Method: Path examination through #ifdef

# Step 3: Count DNN integrations
dnn_operators = count_dnn_conditionals()

# Step 4: Calculate fallback ratio
total_operators = 749
native_xpu = total_operators - fallback_operators
# Apply distribution formula

distribution = {
  "SYCL_NATIVE": native_xpu,
  "MKL_ACCELERATED": mkl_count,
  "DNN_ACCELERATED": dnn_count,
  "CPU_FALLBACK": fallback_count
}
```

---

## Output Specifications

### Required Output Fields

| Field | Type | Description |
|-------|------|-------------|
| **operator_name** | string | Full operator signature |
| **implementation_file** | path | Source file location |
| **dependency_libraries** | list | Required libraries |
| **implementation_pattern** | enum | SYCL/MKL/DNN/Fallback |
| **kernel_files** | list | SYCL kernel file locations |
| **registration_macro** | string | Used registration macro |

### Output Format

```markdown
## Operator Entry Format

### operator.category.variant

| Field | Value |
|-------|-------|
| **Implementation File** | `path/to/file.cpp` |
| **Dependencies** | [SYCL, oneMKL, ...] |
| **Pattern** | TORCH_IMPL_FUNC / REGISTER_XPU_DISPATCH |
| **Kernel** | `object/kernel_function` |
```

### Validation Criteria

| Criterion | Threshold | Measurement |
|-----------|-----------|-------------|
| **Coverage** | 100% | All 749 operators mapped |
| **Accuracy** | 95% | Dependency assignments correct |
| **Completeness** | 100% | Required fields populated |
| **Consistency** | 98% | Cross-reference validation |

---

## Key File Reference Table

### Core Configuration Files

| File Path | Purpose | Critical Lines |
|-----------|---------|----------------|
| `cmake/Codegen.cmake` | torchgen orchestration | 39-46 (command), 107-150 (build) |
| `cmake/ONEMKL.cmake` | oneMKL configuration | 9-19 (flags) |
| `yaml/xpu_functions.yaml` | Operator registry | 1-749 (full) |

### Operator Implementation Files

| Category | Primary File | Kernel Directory |
|----------|--------------|------------------|
| Activation | `Activation.cpp` | `sycl/Activation*.cpp` |
| Binary Ops | `BinaryOps.cpp` | Direct SYCL |
| Linear Algebra | `Blas.cpp` | `mkl/BlasImpl.cpp` |
| Convolution | `Conv.cpp` | `mkldnn/xpu/` |
| Reduction | `ReduceOps.cpp` | `sycl/Reduce*.cpp` |

### Infrastructure Files

| File | Purpose | Key Content |
|------|---------|-------------|
| `XPUFallback.template` | CPU fallback handling | 225-257 (fallback list) |
| `DispatchStub.h` | Registration infrastructure | 1-120 (API) |
| `SYCLContext.h` | Runtime context | 1-19 (abstractions) |

---

## Error Handling and Diagnostics

### Common Analysis Errors

| Error | Detection | Resolution |
|-------|-----------|-----------|
| **Incomplete YAML Read** | Line count mismatch | Increase limit parameter |
| **Missing Registration** | Operator absent from results | Check alternative files |
| **False Dependency** | Conditional misinterpreted | Verify #ifdef scope |
| **Pattern Miss** | Empty grep results | Expand search patterns |

### Diagnostic Queries

```
# Verify YAML completeness
wc -l yaml/xpu_functions.yaml  # Expect: 749 lines

# Count implementation files
ls src/ATen/native/xpu/*.cpp | wc -l  # Expect: 106 files
ls src/ATen/native/xpu/sycl/*.cpp | wc -l  # Expect: ~201 files

# Validate registration patterns
grep -r "REGISTER_XPU_DISPATCH" src/ATen/native/xpu | wc -l
```

---

## Maintenance and Updates

### Registry Synchronization

When PyTorch updates operator registry:

1. **Re-read yaml/xpu_functions.yaml** to capture additions/removals
2. **Cross-reference implementation files** for new patterns
3. **Update dependency mappings** for changed implementations
4. **Verify fallback lists** for modified behavior

### Dependency Library Updates

When Intel libraries update:

1. **Check cmake/ONEMKL.cmake** for configuration changes
2. **Examine conditional compilation paths** for new integrations
3. **Update fallback documentation** for removed operations
4. **Validate new operator support** against test suites

---

## Usage Instructions

### Basic Skill Invocation

```bash
# Environment setup
cd /home/daisydeng/test_pytorch/workdir/pytorch/third_party/torch-xpu-ops

# Execute analysis following workflow phases sequentially
```

### Advanced Analysis

For specialized dependency queries:

1. **Start with Phase 1** to establish complete operator list
2. **Progress to Phase 2** for dependency categorization
3. **Apply Phase 3-4** only for operators requiring deep analysis
4. **Use Phase 5** for validation after changes

### Integration with Other Skills

This skill provides foundational operator registry understanding. Combine with:

- **operator_debugging**: For issue-specific operator analysis
- **performance_profiling**: For kernel execution analysis
- **build_configuration**: For dependency resolution

---

## Document Revision History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | April 2026 | Initial comprehensive skill document |

---

*Skill documentation for PyTorch XPU backend operator analysis. For questions regarding implementation details, refer to Intel oneAPI documentation or PyTorch developer guides.*