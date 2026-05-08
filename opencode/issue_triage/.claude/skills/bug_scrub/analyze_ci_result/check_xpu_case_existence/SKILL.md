# Check XPU Test Case Existence Skill

## Purpose
Deep analysis of PyTorch XPU test case existence by tracing through code execution, comparing pytorch/test vs torch-xpu-ops/test/xpu, checking the daisyden/pytorch release/2.12 branch for distributed tests only, and examining parametrization infrastructure. Uses exploration and deep analysis based on code and execution results, not scripts, filename patterns, or regex-only matching.

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
    - Distributed release-branch reference: https://github.com/daisyden/pytorch/tree/release/2.12

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

### Step 1: Locate Test Files By Deep Inspection

**Important**: XPU tests can live in MULTIPLE locations. Check ALL before declaring a test missing:

| Location | Purpose | Example |
|----------|---------|---------|
| `pytorch/test/` | Upstream pytorch tests (CUDA/CPU) | `test/test_ops.py` |
| `pytorch/third_party/torch-xpu-ops/test/xpu/` | XPU test wrappers | `test_ops_xpu.py` |
| `pytorch/third_party/torch-xpu-ops/test/xpu/distributed/` | **XPU distributed-specific tests** | `test_c10d_xccl.py` |
| `pytorch/third_party/torch-xpu-ops/test/xpu/extended/` | Extended XPU tests | `test_ops_xpu.py` |
| `pytorch/third_party/torch-xpu-ops/test/xpu/nn/` | NN-specific XPU tests | `test_convolution_xpu.py` |
| `pytorch/third_party/torch-xpu-ops/test/xpu/functorch/` | functorch XPU tests | |
| `pytorch/third_party/torch-xpu-ops/test/xpu/quantization/` | Quantization XPU tests | |

For every case, inspect the local sources before final classification:

1. Local upstream base test in `pytorch/test/`.
2. Local XPU test files under `third_party/torch-xpu-ops/test/xpu/**`.
3. For distributed tests only, active distributed skip dictionaries, the remote release/2.12
   local distributed skip list from `intel/torch-xpu-ops` branch `daisyden/distributed_2.12`,
   standalone distributed XPU files, and the daisyden release branch
   `https://github.com/daisyden/pytorch/tree/release/2.12`.

Do not use `release/2.12` for non-distributed test classification. Non-distributed cases should
be classified from the local `pytorch/test/` and `third_party/torch-xpu-ops/test/xpu/**` sources.

Do not stop after a filename or grep hit. Read the class/function body, decorators,
parametrization, wrapper/direct XPU file, and skip-list entry that determines whether the exact
XPU case is generated, skipped, or absent.

#### 1.1 Check torch-xpu-ops main location
```bash
ls -la /home/daisydeng/pytorch/third_party/torch-xpu-ops/test/xpu/
```
Look for the xpu test file matching the base test name pattern `<base_test>_xpu.py`

#### 1.2 Check base test in pytorch/test
```bash
ls -la /home/daisydeng/pytorch/test/<base_test>.py
```
Verify base test class exists in pytorch/test

#### 1.3 **Check torch-xpu-ops distributed subfolder and run_distributed.py (CRITICAL)**

XPU distributed tests run via active skip dictionaries and standalone XPU-native files. You MUST
check all mechanisms that exist in the checkout:

---

**Mechanism A — `run_distributed.py` + distributed skip dictionaries (upstream files run directly)**

`third_party/torch-xpu-ops/test/xpu/run_distributed.py` imports the active distributed skip dict(s)
and calls `launch_test` on each path directly. No `*_xpu.py` wrapper file is created. Read
`run_distributed.py` first; do not assume which dictionary is active.

Known dictionary files to check:
- `intel/torch-xpu-ops` branch `daisyden/distributed_2.12`, file
  `test/xpu/skip_list_dist_local.py` for release/2.12 distributed classification. The file may
  be described as `skip_list_local_dist.py`, but the verified branch filename is
  `skip_list_dist_local.py`.
- `third_party/torch-xpu-ops/test/xpu/skip_list_dist.py`
- `third_party/torch-xpu-ops/test/xpu/skip_list_dict_local.py` if present in the checkout

```bash
# Step 1: Which dictionary does run_distributed.py import?
grep -n "skip_list_dist\|skip_list_dict_local\|skip_dict" \
  /home/daisyden/opencode/classify/pytorch/third_party/torch-xpu-ops/test/xpu/run_distributed.py

# Step 2: Is the upstream test file registered in every active distributed dictionary?
grep -n "<test_filename>" \
  /home/daisyden/opencode/classify/pytorch/third_party/torch-xpu-ops/test/xpu/skip_list_dist.py

# Step 3: If present, inspect local dictionary overrides too
test -f /home/daisyden/opencode/classify/pytorch/third_party/torch-xpu-ops/test/xpu/skip_list_dict_local.py && \
  grep -n "<test_filename>" \
    /home/daisyden/opencode/classify/pytorch/third_party/torch-xpu-ops/test/xpu/skip_list_dict_local.py
```

`skip_dict` / local skip-dict structure:
```python
skip_dict = {
    "../../../../test/distributed/<path>/test_foo.py": None,         # all tests run
    "../../../../test/distributed/<path>/test_bar.py": (             # specific tests skipped
        "test_skipped_method_xpu",
    ),
}
```

- Key present, value `None` → all tests in the file run on XPU
- Key present, value is a tuple/list → tests in the tuple/list are skipped; all others run
- Key absent from all active dictionaries → the file is NOT run on XPU through `run_distributed.py`

```bash
# Step 4: Read active dictionaries in full to see registered files and skipped cases
cat /home/daisyden/opencode/classify/pytorch/third_party/torch-xpu-ops/test/xpu/skip_list_dist.py
test -f /home/daisyden/opencode/classify/pytorch/third_party/torch-xpu-ops/test/xpu/skip_list_dict_local.py && \
  cat /home/daisyden/opencode/classify/pytorch/third_party/torch-xpu-ops/test/xpu/skip_list_dict_local.py
```

---

**Mechanism B — standalone XPU-native files in `distributed/` subfolder**

Some tests exist ONLY in torch-xpu-ops and have NO pytorch/test equivalent.

```bash
ls -la /home/daisyden/opencode/classify/pytorch/third_party/torch-xpu-ops/test/xpu/distributed/
```

**Known XPU-only distributed tests (no pytorch/test equivalent):**
- `test_c10d_xccl.py` - XCCL collective tests (XPU-specific, mirrors `test_c10d_nccl.py`)
- `test_c10d_ops_xccl.py` - XCCL ops tests (XPU-specific, mirrors `test_c10d_ops_nccl.py`)

These files also appear as entries in `skip_list_dist.py` under the `"distributed/test_c10d_xccl.py"` key (relative path, not `../../../../test/...`).

These files do NOT use `XPUPatchForImport` — they are standalone XPU-native tests.

---

**Decision for distributed tests:**

1. Read `run_distributed.py` to identify active distributed dictionary imports.
2. Check the remote release/2.12 local distributed skip list first, then `skip_list_dist.py` and
   `skip_list_dict_local.py` if present for the upstream file path.
3. If present with `None`, the file runs; then verify the exact test exists in the local source
   or in daisyden/pytorch release/2.12.
4. If present with tuple/list, compare the exact generated XPU test name against skipped cases.
5. Check `distributed/` subfolder for XPU-native standalone files.
6. If absent from all active dictionaries and no standalone XPU-native file exists:
   `xpu_case_existence = False`, DetailReason should be specific, e.g.
   `Distributed file missing from skip_list_dist.py` or
   `Distributed file missing from skip_list_dict_local.py`. Do not use generic `No XPU wrapper`.
   Explanation should state which dictionary files were read and what sibling files are enabled.

#### 1.4 Check other torch-xpu-ops subfolders

For tests that might be in extended/nn/functorch/quantization subfolders:
```bash
find /home/daisydeng/pytorch/third_party/torch-xpu-ops/test/xpu/ -name "<test_pattern>*.py"
```

#### 1.5 **Check daisyden release branch for distributed tests only (CRITICAL)**

For distributed tests only, local pytorch checkout may be on a different branch than what the issue
was reported against. Test cases could be **renamed, removed, or relocated** between pytorch versions due to:
- Test refactoring (e.g., class merging, method renaming)
- Parametrization changes (e.g., changing `@parametrize` args)
- Test being moved to a new file
- Test being deleted as obsolete

**Reference release branches:**
- Release 2.12: https://github.com/daisyden/pytorch/tree/release/2.12
- Release 2.11: https://github.com/daisyden/pytorch/tree/release/2.11
- Main: https://github.com/daisyden/pytorch/tree/main

**Check against pytorch release branch via GitHub:**
```bash
# Option A: Use gh CLI to check file contents at specific branch
gh api repos/daisyden/pytorch/contents/test/<base_test>.py?ref=release/2.12 \
  --jq '.content' | base64 -d | grep -n "<test_class_or_method>"

# Option B: Use raw GitHub URL via curl
curl -s https://raw.githubusercontent.com/daisyden/pytorch/release/2.12/test/<base_test>.py | \
  grep -n "<test_class_or_method>"

# Option C: Use git to check a specific ref if submodule/remote is configured
cd ~/pytorch && git show release/2.12:test/<base_test>.py 2>/dev/null | grep -n "<pattern>"
```

**Decision logic with release branch check (distributed tests only):**
```
1. Test NOT in local pytorch/test:
   → Check release/2.12 branch on GitHub
   → If found in release/2.12 → Test was REMOVED/renamed in current main
     Reason: "Test removed after release/2.12"
   → If also not in release/2.12 → Test never existed here
     Reason: "Test not in pytorch release/2.12 or main"

2. Test IS in local pytorch/test:
   → Also verify it's in release/2.12 (for issue consistency)
   → If missing in release/2.12 → Test is newer than release
     Reason: "Test added after release/2.12"
```

**When to invoke release branch check:**
- Distributed test class/method "removed/renamed" reason
- Distributed issue reported on older pytorch version but test now missing locally
- Distributed discrepancy between local source and release branch

Do not invoke release/2.12 checks for non-distributed tests.

#### 1.6 Read XPU wrapper/direct file structure
File: `third_party/torch-xpu-ops/test/xpu/<test>_xpu.py`

Typical structure:
```python
with XPUPatchForImport(False):
    from <base_module> import <BaseTestClass>

instantiate_device_type_tests(
    <BaseTestClass>, globals(), only_for="xpu", allow_xpu=True
)
```

This is only one pattern. Many XPU files are direct/standalone copies or hand-written XPU tests
that do not use `XPUPatchForImport`. For those files, inspect the XPU file itself:
- Does it define the target class/function directly?
- Does it assign/replace methods on an imported class before instantiation?
- Does it call `instantiate_device_type_tests`, `instantiate_parametrized_tests`, or both?
- Does the generated XPU test name exist after parametrization?
- Does the file use `only_for="xpu"`, `allow_xpu=True`, or a broader device list including XPU?

### Step 2: Understand XPUPatchForImport Behavior

#### 2.1 Read xpu_test_utils.py XPUPatchForImport class
File: `/home/daisydeng/pytorch/third_party/torch-xpu-ops/test/xpu/xpu_test_utils.py` starting at line 972

Critical finding: `XPUPatchForImport(False)` OVERRIDES decorators during import!

Key transformations in `__enter__` (lines 1131-1168):
```python
# CRITICAL: Decorator overrides during import!
common_device_type.onlyCUDA = common_device_type.onlyXPU  # @onlyCUDA -> @onlyXPU
common_device_type.skipXPU = _skipXPU  # @skipXPU -> NOP (_skipXPU returns obj)
common_device_type.onlyNativeDeviceTypes = common_device_type.onlyXPU
```

#### 2.2 XPUPatchForImport Override Effects
| Original Decorator | After PATCH | Impact on XPU |
|-------------------|-------------|---------------|
| `@skipXPU` | `_skipXPU` (NOP) | NOT skipped on XPU! |
| `@onlyCUDA` | `@onlyXPU` | CAN RUN on XPU! |
| `@onlyNativeDeviceTypes` | `@onlyXPU` | XPU is allowed |
| `instantiate_parametrized_tests` | `DO_NOTHING` if `patch_test_case=True` | Parametrized tests disabled |

#### 2.3 Verify PATCH Usage
Check if XPU wrapper uses PATCH:
```bash
grep -l "XPUPatchForImport" third_party/torch-xpu-ops/test/xpu/*.py
```

Typical XPU test wrapper structure:
```python
with XPUPatchForImport(False):  # False = DON'T disable instantiation
    from test_ops import TestCommon

instantiate_device_type_tests(
    TestCommon, globals(), only_for="xpu", allow_xpu=True
)
```

#### 2.4 Decision Matrix
```
IF XPU file uses XPUPatchForImport(False):
    THEN @skipXPU/@onlyCUDA are OVERRIDDEN during import → imported tests may run on XPU
ELSE IF XPU file uses XPUPatchForImport(True):
    THEN device/parametrized instantiation from the imported module may be disabled
ELSE:
    The XPU file is direct/standalone. Read its definitions and decorators literally;
    @onlyCUDA is not patched unless the file itself changes it.
```

#### 2.5 Critical Line Analysis (line 1143-1144)
```python
if self.patch_test_case:
    common_device_type.instantiate_device_type_tests = DO_NOTHING
    common_utils.instantiate_parametrized_tests = DO_NOTHING
```

Only if `patch_test_case=True` instantiation is disabled.

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

#### 7.1 Classify absence reason with specific DetailReason

**Reason 1: Specific XPU test enablement gap**
- Do not write only `No XPU wrapper` as DetailReason.
- State the concrete gap, such as:
  - `XPU test file missing in torch-xpu-ops/test/xpu/<subdir>`
  - `Class not imported by existing XPU test file`
  - `Method not defined in standalone XPU test file`
  - `Distributed file missing from skip_list_dist.py`
  - `Distributed file missing from skip_list_dict_local.py`
  - `XPUPatchForImport(True) disables instantiation`
  - `OpInfo dtypesIfXPU excludes <dtype>`
  - `Explicit XPU skip decorator`
  - `XPU graph coverage missing`

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
- Only classify `Not applicable / Community Changes` when the CUDA test itself no longer exists
  or was renamed/removed in the PyTorch source being compared. If the CUDA case and XPU case both
  exist after parametrization (for example `test_Conv1d_pad2_cuda` and `test_Conv1d_pad2_xpu`),
  it is not a community-change case.

**Reason 6: CUDA-specific API**
- If classifying `Not Appliable`, the DetailReason must name the exact API/feature, e.g.
  `CUDA-specific API: torch.cuda.jiterator`, `CUDA-specific API: cuBLAS`, or
  `CUDA-specific API: CUDA-only JIT fuser`.
- Do not classify CUDA graph/cudagraph rows as `Not Appliable` solely due to CUDA naming;
  XPU graph support exists via `_XPUGraph` and `torch.accelerator.Graph`, so missing/failing
  coverage should generally be `To be enabled / XPU graph coverage missing` after code inspection.

#### 7.2 Cross-reference with CUDA support
Compare `dtypesIfCUDA` vs `dtypesIfXPU` to understand:
- If XPU lags behind CUDA in dtype support
- Identifies specific dtype gaps between platforms

## Constraints

### Directory Structure Constraint
torch-xpu-ops MUST be located at `third_party/torch-xpu-ops` - NOT at workspace root as a separate repo

### Distributed Test Location Constraint
Tests with `origin_file` path like `test/distributed/test_c10d_xccl.py` may be XPU-native and only exist at:
`third_party/torch-xpu-ops/test/xpu/distributed/test_c10d_xccl.py` — NOT in `pytorch/test/distributed/`.
But many distributed tests run upstream `pytorch/test/distributed/**` files through
`run_distributed.py` and active skip dictionaries. Always inspect the current mechanism.

### Release Branch Verification Constraint
For distributed tests only, when tests are not found locally, MUST verify against the daisyden
release branch before declaring removed:
- Release 2.12 is the reference: https://github.com/daisyden/pytorch/tree/release/2.12
- Use `gh api` or raw.githubusercontent.com to check file contents at specific ref
- Distinguish between "removed after release/2.12" vs "never existed" reasons

For non-distributed tests, do not check release/2.12; classify from local `pytorch/test/` and
local `third_party/torch-xpu-ops/test/xpu/**` source inspection.

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
| `third_party/torch-xpu-ops/test/xpu/<test>_xpu.py` | XPU test wrapper (uses XPUPatchForImport) |
| `third_party/torch-xpu-ops/test/xpu/distributed/test_c10d_xccl.py` | **XPU-native XCCL distributed tests (no pytorch/test equivalent)** |
| `third_party/torch-xpu-ops/test/xpu/distributed/test_c10d_ops_xccl.py` | **XPU-native XCCL ops tests** |
| `third_party/torch-xpu-ops/test/xpu/extended/<test>_xpu.py` | Extended XPU tests (uses XPUPatchForImport(True)) |
| `third_party/torch-xpu-ops/test/xpu/nn/` | NN-specific XPU tests |
| `third_party/torch-xpu-ops/test/xpu/functorch/` | functorch-specific XPU tests |
| `third_party/torch-xpu-ops/test/xpu/quantization/` | Quantization-specific XPU tests |
| `third_party/torch-xpu-ops/test/xpu/xpu_test_utils.py` | XPUPatchForImport class and patching logic |
| `test/<base_test>.py` | Base test with @ops or @dtypes decorators (local main) |
| **https://github.com/daisyden/pytorch/tree/release/2.12/test/** | **Release branch - use for distributed verification only** |
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
START: Identify origin_file type
├─ origin_file is under test/distributed/ (any distributed test)
│  └─ STEP A: Read run_distributed.py to identify active dictionary imports
│     └─ STEP B: Read skip_list_dist.py and skip_list_dict_local.py if present
│        ├─ File path found in an active dict, value None
│        │  └─ Check local pytorch/test and daisyden release/2.12 for exact test method
│        │     ├─ Found → xpu_case_existence=True
│        │     └─ Not found → classify by removed/renamed evidence, not by skip-list alone
│        ├─ File path found in active dict, value tuple/list
│        │  ├─ test_method NOT in skipped list → xpu_case_existence=True
│        │  └─ test_method IN skipped list → xpu_case_existence=True but intentionally skipped
│        │     DetailReason: "Skipped in <dict file>: <specific test>"
│        └─ File path NOT in any active dict
│           └─ STEP C: Check distributed/ subfolder for XPU-native standalone file
│              ├─ Standalone XPU file exists AND test class/method found → xpu_case_existence=True
│              └─ Not found anywhere → xpu_case_existence=False
│                 DetailReason: "Distributed file missing from <dict file>"
│                 Explanation: cite all dict files read and sibling files enabled
│
└─ Standard (non-distributed) test case → Continue below

CHECK: Locate XPU test file/function under all torch-xpu-ops/test/xpu subfolders
├─ No XPU file/function/class found
│  └─ Check local pytorch/test before concluding
│     DetailReason: "XPU test file missing" or "Class not imported by XPU test file"
└─ XPU file found → Continue

CHECK: Determine XPU file mechanism
├─ Uses XPUPatchForImport(False) → imported decorators may be patched; inspect imports and instantiation
├─ Uses XPUPatchForImport(True) → imported parametrized/device tests may not be generated
└─ Does NOT use XPUPatchForImport → direct/standalone file; read definitions literally

CHECK: Base test function/class exists in local pytorch/test?
├─ NO → xpu_case_existence=False
│  Reason: "Test class removed/renamed in local pytorch/test"
└─ YES → Continue

CHECK: If both CUDA and XPU variants exist after parametrization
├─ YES → NOT Community Changes; classify by XPU status/skip/failure/enablement gap
└─ NO → For non-distributed tests, classify from local source only; release/2.12 is distributed-only

CHECK: CUDA graph / cudagraph case
├─ YES → XPU graph support exists; classify missing/failing coverage as To be enabled
│        DetailReason: "XPU graph coverage missing" or exact XPU graph failure
└─ NO → Continue

CHECK: @skipIfXPU decorator on class/method (BLOCKS at runtime, NOT overridden by PATCH)
├─ YES → xpu_case_existence=True
│  Note: @skipIfXpu is a RUNTIME skip, not a parametrization failure.
│  The test variant IS still generated by parametrization (exists as a test case),
│  it just gets skipped when executed on XPU.
│  Reason format: "Variant generated; @skipIfXpu at <file:line> skips at runtime"
└─ NO → Continue

CHECK: OpInfo dtypesIfXPU excludes given dtype
├─ YES → xpu_case_existence=False
│  DetailReason: "OpInfo dtypesIfXPU excludes <dtype>"
└─ NO → Continue

CHECK: Skip decorators for XPU in OpInfo skips list
├─ YES → xpu_case_existence=False
│  DetailReason: "OpInfo skip decorator applied: <specific decorator>"
└─ NO → xpu_case_existence=True
   Reason: "Test exists and runs on XPU"
```
