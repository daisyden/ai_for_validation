# Check XPU Test Existence in torch-xpu-ops

## Overview
This skill provides a workflow to check whether a specific test case exists in the torch-xpu-ops repo's test/xpu folder.

## Input Parameters
- **Test File**: The XPU test file path in torch-xpu-ops (e.g., `torch-xpu-ops/test/xpu/nn/test_convolution_xpu.py`)
- **Origin Test File**: The original PyTorch test file (e.g., `test/nn/test_convolution.py`)
- **Test Class**: The test class name (e.g., `TestConvolutionNNDeviceTypeXPU`)
- **Test Case**: The test method name (e.g., `test_conv2d_hipdnn_backend_selection_xpu`)

## Workflow Steps

### Step 1: Locate torch-xpu-ops repo
The torch-xpu-ops repo is located at: `{pytorch_root}/third_party/torch-xpu-ops`

### Step 2: Check if XPU test file exists
```bash
ls {pytorch_root}/third_party/torch-xpu-ops/test/xpu/<path>/<test_file>.py
```

### Step 3: Check if XPUPatchForImport is used
Search for `XPUPatchForImport` in the XPU test file:
```bash
grep -n "XPUPatchForImport" {pytorch_root}/third_party/torch-xpu-ops/test/xpu/<path>/<test_file>.py
```

- If **XPUPatchForImport is used on a class**: The test is imported from original file. Check the decorators in the **Origin Test File**.
- If **XPUPatchForImport is NOT used**: The test is directly implemented in the XPU test file. Check the decorators in the **Test File**.

### Step 4: Find the original test case
If using XPUPatchForImport, search in the Origin Test File:
```bash
grep -n "<test_case>" {pytorch_root}/<origin_test_file>
```
Example: `grep -n "test_conv2d_hipdnn_backend_selection" test/nn/test_convolution.py`

### Step 5: Analyze decorators
Common decorators that affect test validity:
- `@onlyCUDA` - Only runs on CUDA, not XPU
- `@onlyXPU` - Only runs on XPU
- `@skipCUDAIfNoHipdnn` - ROCm-specific, not applicable to XPU
- `@skipIfXpu` - Skips on XPU
- `@requires_xccl` - Requires XCCl (Intel MPI)

XPUPatchForImport may redefine some skip decorators for XPU compatibility.

### Step 6: Determine existence
- If decorators like `@onlyCUDA` or `@skipCUDAIfNoHipdnn` are present, the test is typically NOT ported to XPU (CUDA/ROCm specific)
- If `@onlyXPU` is present, the test is XPU-specific
- Search for the test name directly in XPU test file if not using XPUPatchForImport

## Example
Checking `test_conv2d_hipdnn_backend_selection_xpu`:

1. XPU test file: `third_party/torch-xpu-ops/test/xpu/nn/test_convolution_xpu.py`
2. XPUPatchForImport used at line 79 with `False`
3. Test imported from original: `test/nn/test_convolution.py`
4. Original test has decorators: `@onlyCUDA`, `@skipCUDAIfNoHipdnn`
5. **Result**: Test does NOT exist in XPU (ROCm/hipdnn specific, not applicable to XPU)

## Key Points
- hipdnn backend is ROCm-specific, not applicable to Intel XPU
- CUDA-specific tests with `@onlyCUDA` typically not ported to XPU
- Always check both the XPU test file and original test file for complete picture