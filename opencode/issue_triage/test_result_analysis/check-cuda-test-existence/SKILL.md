# Check CUDA Test Existence in Original PyTorch Test File

## Overview
This skill provides a workflow to check whether a CUDA test case exists in the original PyTorch test file, given an XPU test name pattern.

## Input Parameters
- **Origin Test File**: Original PyTorch test file (e.g., `test/nn/test_convolution.py`)
- **Test Case**: XPU test case name (e.g., `test_conv2d_hipdnn_backend_selection_xpu`)

## Mapping Rule
Replace suffixes to find CUDA test:
- `_xpu` → `_cuda`
- `_XPU` → `_CUDA`

Example: `test_conv2d_hipdnn_backend_selection_xpu` → `test_conv2d_hipdnn_backend_selection`

## Workflow Steps

### Step 1: Map XPU test name to CUDA test name
Apply the replacement rule:
- Remove `_xpu` or `_XPU` suffix from test case name

### Step 2: Search in Origin Test File
```bash
grep -n "<cuda_test_name>" {pytorch_root}/<origin_test_file>
```

### Step 3: Read the test case and decorators
Find the test method definition and identify all decorators.

### Step 4: Analyze decorators
Common decorators that define test limitations:
- `@onlyCUDA` - Only runs on CUDA (not CPU/XPU)
- `@onlyXPU` - Only runs on XPU
- `@onlyCPU` - Only runs on CPU
- `@unittest.skip` - Skips the test unconditionally
- `@unittest.skipIf(condition, reason)` - Skips if condition is true
- `@skipCUDAIfNoCudnn` - Skips if cuDNN not available
- `@skipCUDAIfNoHipdnn` - Skips if HIPdnn not available (ROCm)
- `@skipIfXpu` - Skips on XPU
- `@requires_xccl` - Requires XCCl (Intel MPI)
- `@dtypes(...)` - Parameterizes dtypes
- `@parametrize_test(...)` - Parameterizes test arguments

### Step 5: Determine existence and applicability
- If test exists and has `@onlyCUDA` or ROCm-specific decorators (hipdnn), it's CUDA/ROCm specific
- If test has `@onlyXPU`, it's XPU-specific
- If test has no restrictive device decorators, it may run on multiple backends

## Example
Checking `test_conv2d_hipdnn_backend_selection_xpu`:

1. **Map test name**: Remove `_xpu` → `test_conv2d_hipdnn_backend_selection`
2. **Search in origin**: `grep -n "test_conv2d_hipdnn_backend_selection" test/nn/test_convolution.py`
3. **Found at line 4541**
4. **Decorators**:
   - `@onlyCUDA` - CUDA only
   - `@skipCUDAIfNoHipdnn` - ROCm specific (hipdnn)
5. **Result**: Test EXISTS but is ROCm/hipdnn specific, not applicable to XPU

## Key Points
- The CUDA test name is derived by removing `_xpu` suffix from XPU test name
- Check decorators to understand device applicability
- ROCm-specific tests (hipdnn) are not applicable to Intel XPU
- CUDA-specific tests (`@onlyCUDA`) may or may not be applicable to XPU depending on the feature