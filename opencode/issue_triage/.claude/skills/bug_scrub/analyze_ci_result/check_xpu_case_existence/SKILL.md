# Check XPU Test Case Existence Skill

## Purpose
Deep analysis of PyTorch XPU test case existence by tracing through code execution, comparing pytorch/test vs torch-xpu-ops/test/xpu, and examining parametrization infrastructure. Uses exploration and deep analysis based on code and execution results, not simple pattern matching.

## Preconditions

### Environment
1. conda environment `pytorch_opencode_env` must be available:
   ```bash
   source ~/miniforge3/bin/activate pytorch_opencode_env && conda activate pytorch_opencode_env
   ```

2. Required directory structure:
   - Main pytorch repo: `/home/daisydeng/pytorch`
   - XPU tests location: `/home/daisydeng/pytorch/third_party/torch-xpu-ops/test/xpu/`
   - Base tests location: `/home/daisydeng/pytorch/test/`

3. torch-xpu-ops is stored as third_party submodule at `third_party/torch-xpu-ops`

### Required Tools
1. `read` - Read files and directories with line numbers
2. `bash` - Execute shell commands for:
   - directory listing (`ls -la`)
   - pytest collection (`python3 -m pytest --collect-only`)
   - conda activation
   - find operations
3. `grep` - Search file contents with patterns
4. `glob` - File pattern matching

## Workflow

### Step 1: Locate Test Files

#### 1.1 Check torch-xpu-ops location
```bash
ls -la /home/daisydeng/pytorch/third_party/torch-xpu-ops/test/xpu/
```
Look for the xpu test file matching the base test name pattern `<base_test>_xpu.py`

#### 1.2 Check base test in pytorch/test
```bash
ls -la /home/daisydeng/pytorch/test/<base_test>.py
```
Verify base test class exists in pytorch/test

#### 1.3 Read XPU wrapper file structure
File: `third_party/torch-xpu-ops/test/xpu/<test>_xpu.py`

Typical structure:
```python
with XPUPatchForImport(False):
    from <base_module> import <BaseTestClass>

instantiate_device_type_tests(
    <BaseTestClass>, globals(), only_for="xpu", allow_xpu=True
)
```

### Step 2: Understand XPUPatchForImport Behavior

#### 2.1 Read xpu_test_utils.py XPUPatchForImport class
File: `/home/daisydeng/pytorch/third_party/torch-xpu-ops/test/xpu/xpu_test_utils.py` starting at line 972

Key behaviors:
- `XPUPatchForImport(patch_test_case=True/False)` constructor parameter controls instantiation
- `XPUPatchForImport(False)`: Does NOT disable test case instantiation, test cases WILL be generated
- `XPUPatchForImport(True)`: Disables instantiation - `instantiate_device_type_tests = DO_NOTHING`

#### 2.2 Critical line analysis (line 1143-1144)
```python
if self.patch_test_case:
    common_device_type.instantiate_device_type_tests = DO_NOTHING
    common_utils.instantiate_parametrized_tests = DO_NOTHING
```

Determines if parameterized tests will be generated or suppressed

### Step 3: Trace Base Test Parametrization

#### 3.1 Find the decorated test function
```grep
pattern: <test_function_name_without_prefix>
path: /home/daisydeng/pytorch/test
```

Extract the test function and its decorator:
```python
@ops(<op_db_name>)
def <test_function_name>(self, device, dtype, op):
    # function body
```

#### 3.2 Analyze decorator and parameters
Key elements to identify:
- `@ops(<op_db>)` - op_db parametrization from common_methods_invocations
- `@dtypes(...)` - direct dtype parametrization
- `@dtypesIfXPU(...)` - XPU-specific dtype filters
- Function signature parameters: `(self, device, dtype, op)` for @ops

#### 3.3 Find instantiate_device_type_tests call
```grep
pattern: instantiate_device_type_tests.*<TestClassName>
path: /home/daisydeng/pytorch/test/<base_test>.py
```

Key parameter: `allow_xpu=True` must be present for XPU test generation

### Step 4: Locate OpInfo Definition

#### 4.1 OpInfo Database Locations

OpInfo definitions are spread across multiple locations:

| Location | File | Description |
|----------|------|-------------|
| Main op_db | `torch/testing/_internal/common_methods_invocations.py` | Primary OpInfo definitions including `BinaryUfuncInfo`, `UnaryUfuncInfo` |
| OpInfo core | `torch/testing/_internal/opinfo/core.py` | `OpInfo` class definition |
| OpInfo definitions | `torch/testing/_internal/opinfo/definitions/` | Modularized OpInfo definitions |
| FFT ops | `torch/testing/_internal/opinfo/definitions/fft.py` | FFT-related operators |
| Linalg ops | `torch/testing/_internal/opinfo/definitions/linalg.py` | Linear algebra operators |
| Signal ops | `torch/testing/_internal/opinfo/definitions/signal.py` | Signal processing operators |
| Special ops | `torch/testing/_internal/opinfo/definitions/special.py` | Special mathematical functions |
| Masked ops | `torch/testing/_internal/opinfo/definitions/_masked.py` | Masked operations |
| Nested ops | `torch/testing/_internal/opinfo/definitions/nested.py` | Nested tensor operations |
| Sparse ops | `torch/testing/_internal/opinfo/definitions/sparse.py` | Sparse tensor operators |
| Custom ops | `torch/testing/_internal/custom_op_db.py` | Custom operator definitions |
| Hop ops | `torch/testing/_internal/hop_db.py` | Higher order operator patterns |

#### 4.2 OpInfo Groupings

Common op_db groupings defined at end of `common_methods_invocations.py`:
```python
ops_and_refs = op_db + python_ref_db
unary_ufuncs = [op for op in ops_and_refs if isinstance(op, UnaryUfuncInfo)]
binary_ufuncs = [op for op in ops_and_refs if isinstance(op, BinaryUfuncInfo)]
spectral_funcs = [op for op in ops_and_refs if isinstance(op, SpectralFuncInfo)]
sparse_unary_ufuncs = [op for op in op_db if isinstance(op, UnaryUfuncInfo) and op.supports_sparse]
reduction_ops = [op for op in ops_and_refs if isinstance(op, ReductionOpInfo)]
shape_funcs = [op for op in ops_and_refs if isinstance(op, ShapeFuncInfo)]
```

#### 4.3 Find OpInfo for the operator
```grep
pattern: BinaryUfuncInfo\('<op_name>'\)|OpInfo\('<op_name>'\)|UnaryUfuncInfo\('<op_name>'\)
path: /home/daisydeng/pytorch/torch/testing/_internal/
```

Note: The op_name extraction depends on context:
- Parameterized tests with `@ops(op_db)`: extracted from test function name
- Direct dtype tests: inferred from surrounding test context

#### 4.4 Extract dtype restrictions
From OpInfo definition, identify:
```python
dtypes=...                    # Base dtypes supported (applied if dtypesIfXPU not set)
dtypesIfXPU=...               # XPU-specific dtypes override
dtypesIfCUDA=...             # CUDA-specific for comparison
skips=...                     # Skip decorators with device_type='xpu'
```

#### 4.5 Understand dtype function expansion
File: `/home/daisydeng/pytorch/torch/testing/_internal/common_dtype.py`

Key dtype helpers:
```python
def floating_types_and(*dtypes):
    # Returns: {float32, float64} + provided dtypes
    return _floating_types + _validate_dtypes(*dtypes)

def floating_and_complex_types_and(*dtypes):
    # Returns: {float32, float64, cfloat, cdouble} + provided dtypes
    return _floating_and_complex_types + _validate_dtypes(*dtypes)

def all_types():
    # Returns: _floating_types + _integral_types
```

Used to expand macros like `floating_types_and(torch.bfloat16, torch.float16)`

### Step 5: Parse Test Name Pattern

#### 5.1 Understand generated test naming
Full test name format:
```
<test_function_name>_<op_name>_xpu_<dtype>
```

Examples:
- `test_contig_size1_large_dim_logaddexp_xpu_complex128`
- `test_contig_size1_large_dim_div_xpu_float32`

#### 5.2 Break down components
- `<test_function_name>`: The actual test method name in base test class
- `<op_name>`: Extracted from `@ops(op_db)` decorator where `op` parameter
- `_xpu_`: Device suffix indicating XPU
- `<dtype>`: The specific dtype being tested

### Step 6: Verify Test Generation with pytest

#### 6.1 Collect tests to verify parametrization
```bash
source ~/miniforge3/bin/activate pytorch_opencode_env && conda activate pytorch_opencode_env && \
cd /home/daisydeng/pytorch/third_party/torch-xpu-ops/test/xpu && \
python3 -m pytest <test_file>_xpu.py --collect-only 2>/dev/null | \
grep "<test_function_name>_<op_name>"
```

#### 6.2 Compare generated vs expected
List all collected variants to verify which dtypes are available

### Step 7: Determine Absence Reason

#### 7.1 Classify absence reason

**Reason 1: Test XPU wrapper NOT defined**
- torch-xpu-ops does not have the corresponding `_xpu.py` test file
- Action: This is a porting gap requiring new XPU test file creation

**Reason 2: XPU dtypesIfXPU restriction (MOST COMMON)**
- OpInfo has `dtypesIfXPU` that excludes specific dtypes
- If `complex128` not in `dtypesIfXPU`, no complex variant generated
- Solution: Update OpInfo definition to add missing dtype support

**Reason 3: Skip decorators applied**
- OpInfo has skip decorators for XPU: `DecorateInfo(skip, ..., device_type='xpu')`
- Check both `skips` in OpInfo and `wrappers` in decorators

**Reason 4: XPUPatchForImport(True) disables instantiation**
- Check if test file uses `XPUPatchForImport(True)`
- Tests get imported but NOT instantiated when patching is True

**Reason 5: Test renamed or moved in base pytorch/test**
- Test no longer exists in expected location
- Search for renaming in git history or similar test names

#### 7.2 Cross-reference with CUDA support
Compare `dtypesIfCUDA` vs `dtypesIfXPU` to understand:
- If XPU lags behind CUDA in dtype support
- Identifies specific dtype gaps between platforms

## Constraints

### Directory Structure Constraint
torch-xpu-ops MUST be located at `third_party/torch-xpu-ops` - NOT at workspace root as a separate repo

### Conda Environment Constraint
All Python analysis MUST be performed within `pytorch_opencode_env` conda environment to access correct pytorch modules

### pytest Collection Constraint
- pytest `--collect-only` MUST be run from `third_party/torch-xpu-ops/test/xpu/` directory
- Output filtering requires matching correct test name patterns

### OpInfo dtype Constraint
The generated test name dtype is determined by `dtypesIfXPU` (if set) NOT the base `dtypes` field

### XPUPatchForImport Behavior Constraint
- Parameter `False`: Allows instantiation, tests WILL be generated
- Parameter `True`: Disables instantiation, tests WILL NOT be generated

## Key Files Reference

| File Path | Purpose |
|-----------|---------|
| `third_party/torch-xpu-ops/test/xpu/<test>_xpu.py` | XPU test wrapper |
| `third_party/torch-xpu-ops/test/xpu/xpu_test_utils.py` | XPUPatchForImport class and patching logic |
| `test/<base_test>.py` | Base test with @ops or @dtypes decorators |
| `torch/testing/_internal/common_methods_invocations.py` | Primary OpInfo definitions with dtypes |
| `torch/testing/_internal/opinfo/core.py` | OpInfo class definition |
| `torch/testing/_internal/opinfo/definitions/__init__.py` | Aggregated op_db from submodule definitions |
| `torch/testing/_internal/opinfo/definitions/fft.py` | FFT operator OpInfos |
| `torch/testing/_internal/opinfo/definitions/linalg.py` | Linear algebra operator OpInfos |
| `torch/testing/_internal/opinfo/definitions/_masked.py` | Masked operation OpInfos |
| `torch/testing/_internal/opinfo/definitions/signal.py` | Signal processing OpInfos |
| `torch/testing/_internal/opinfo/definitions/special.py` | Special function OpInfos |
| `torch/testing/_internal/custom_op_db.py` | Custom operator OpInfos |
| `torch/testing/_internal/hop_db.py` | Higher order operator patterns |
| `torch/testing/_internal/common_device_type.py` | instantiate_device_type_tests implementation |
| `torch/testing/_internal/common_dtype.py` | dtype helper functions and expansion |

## Output Format

When analyzing a specific test case, provide two output options:

### Option A: Complete Analysis Report
```
## Analysis Summary: <test_case_name>

### 1. Test File Locations
- Base test: <path>:<line>
- XPU wrapper: <path>
- XPUPatchForImport usage: <True/False>

### 2. Test Parametrization
- Decorator: <@ops/@dtypes/other>
- Parameters: <list parameters>
- instantiate_device_type_tests: <found at line X, allow_xpu=True/False>

### 3. OpInfo Analysis
<OpInfo name>: found at <path>:<line>
- dtypes: <base dtype list>
- dtypesIfXPU: <xpu dtype list - THIS DETERMINES GENERATION>
- skips: <any xpu skip decorators>

### 4. Root Cause
<details reason classification>

### 5. Existing XPU Tests (from pytest --collect-only)
<all collected variants>

### 6. Expected But Missing Variants
<list dtypes that should be generated but aren't>

### 7. Conclusion
<summary with solution path>
```

### Option B: Minimal Presence Check
```
Test: <test_name>
Base: <path>:<line>
XPU File: <path> (exists/missing)
Generated: <yes/no>
Reason: <absence classification>
```

## Critical Decision Tree

```
START: Check if XPU wrapper file exists
├─ NO → Reason 1: XPU wrapper not ported to torch-xpu-ops
└─ YES → Continue

CHECK: XPUPatchForImport parameter
├─ True → Reason 4: Instantiation disabled by patching
└─ False → Continue (tests SHOULD be generated)

CHECK: Base test function exists
├─ NO → Reason 5: Test removed/renamed in pytorch/test
└─ YES → Continue

CHECK: OpInfo dtypesIfXPU excludes given dtype
├─ YES → Reason 2: XPU dtype restriction
└─ NO → Continue

CHECK: Skip decorators for XPU
├─ YES → Reason 3: Skip decorator applied
└─ NO → Test SHOULD exist, investigate pytest collection
```