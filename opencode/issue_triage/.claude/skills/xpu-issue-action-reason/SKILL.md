# XPU Issue Action Reason Workflow

## Description
Analyzes torch-xpu-ops issues where 'Action TBD' is 'Need Investigation' and provides specific fix suggestions for the Action Reason column.

## When to use
When asked to analyze XPU issues, fill Action Reason, or triage torch-xpu-ops issues.

## Workflow

### Step 1: Load and Filter Issues
```python
import pandas as pd
df = pd.read_excel('torch_xpu_ops_issues.xlsx', sheet_name='Issues')
need_inv = df[df['Action TBD'] == 'Need Investigation']
```

### Step 2: Analyze Each Issue
For each issue, examine:
- **Title**: Key operation names (e.g., stft, attention, conv, distributed)
- **Root Cause**: Category of issue (Distributed/Gloo, Dtype/Precision, Backend/Device, Inductor/Compilation, Flash Attention, Timeout/Performance, Memory, etc.)
- **Test Cases sheet**: Test file path and test case name
- **E2E Test Cases sheet**: Model name, error message, traceback

### Step 3: Search Source Code
Search PyTorch source for specific issues:
```bash
# Find test files
glob path: ~/pytorch/test **/test_*.py
glob path: ~/pytorch/third_party/torch-xpu-ops/test/xpu **/test_*.py

# Search for specific code
grep path: ~/pytorch pattern: <specific_function_or_api>
```

### Step 4: Generate Specific Fix Suggestions
Based on Root Cause and keywords:

| Root Cause | Fix Template |
|------------|--------------|
| Distributed/Gloo Issue | Fix distributed backend for XPU: Update process group initialization for XPU in torch/distributed/distributed_c10d.py or implement XPU-compatible collective operations. |
| Dtype/Precision Issue | Fix dtype/precision issue: Adjust tolerance in test file or implement proper dtype handling in the operation kernel for XPU. |
| Backend/Device Issue | Implement XPU kernel or backend support: Add proper XPU dispatch in native_functions.yaml or implement the missing kernel in the appropriate aten module. |
| Inductor/Compilation Issue | Fix Inductor XPU compilation: Add proper lowering in torch/_inductor/lowering.py or fix decomposition path for the specific operator. |
| Flash Attention/Specific Ops Issue | Fix Flash Attention on XPU: Enable or implement FlashAttentionForwardXPU with proper head_dim and dropout support. |
| Timeout/Performance Issue | Fix performance issue: Implement optimized XPU kernel or enable existing optimization path for the operation. |
| Memory/Shared Memory Issue | Fix memory management on XPU: Check memory allocation/deallocation in the operation implementation. |
| Skip/No Test Exists | Add to skiplist: Mark test as xfail or add skip decorator in the test file for XPU if not applicable. |
| Failure | Investigate and fix test failure: Analyze traceback and implement proper fix in the operation implementation or test. |
| Error | Fix error handling: Implement proper error handling or add XPU-specific error path in the operation. |
| Others | This is not a PyTorch code issue - may require CI or documentation changes. |

### Step 5: Title-based Refinements
For specific keywords in title, use more targeted fixes:
- **attention/sdpa**: Fix attention/SDPA operation on XPU: Implement proper scaled_dot_product_attention dispatch or add XPU fallback.
- **stft/fft**: Fix STFT/FFT precision on XPU: Use float32 intermediate or adjust test tolerance.
- **distributed**: Fix distributed operation on XPU: Update process group initialization or implement XPU-compatible collectives.
- **inductor/compile**: Fix Inductor compilation for XPU: Add proper lowering or decomposition.
- **sparse**: Implement sparse operation for XPU: Add missing sparse kernel implementation.
- **conv**: Fix convolution operation on XPU: Add proper backend selection or implement missing kernel.

### Step 6: For Key Issues - Deep Dive
For critical issues, do specific source code analysis:
- Check `torch/csrc/distributed/c10d/init.cpp` for distributed APIs
- Check `torch/_decomp/decompositions.py` for operator decompositions
- Check `torch/_inductor/lowering.py` for Inductor lowerings
- Check `torch/nn/functional.py` for functional operations
- Check test files in `test/` and `third_party/torch-xpu-ops/test/xpu/`

### Step 7: Update Excel
```python
# Apply fixes
for issue_id, reason in fixes.items():
    mask = df['Issue ID'] == issue_id
    df.loc[mask, 'Action Reason'] = reason

# Save
df.to_excel('torch_xpu_ops_issues.xlsx', sheet_name='Issues', index=False)
```

## Key File Locations
- Test code: `~/pytorch/test/` or `~/pytorch/third_party/torch-xpu-ops/test/xpu/`
- Operator decompositions: `torch/_decomp/decompositions.py`
- Inductor lowering: `torch/_inductor/lowering.py`
- Distributed backend: `torch/distributed/distributed_c10d.py`
- C++ distributed init: `torch/csrc/distributed/c10d/init.cpp`

## Output Format
2-sentence fix suggestion per issue, referencing specific file paths and functions when possible.