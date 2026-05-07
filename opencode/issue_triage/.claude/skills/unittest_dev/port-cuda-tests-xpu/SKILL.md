---
name: port-cuda-tests-xpu
description: Port PyTorch CUDA test files to support Intel XPU backend testing. Use when generalizing CUDA-specific tests to run on XPU, enabling GPU tests for both CUDA and XPU backends, or when user mentions porting CUDA tests to XPU or enabling XPU test coverage.
---

# Port CUDA Tests to XPU Backend

This Skill provides a workflow for generalizing PyTorch CUDA-specific test files to support Intel XPU backend testing.

## Skill Integration

**This skill follows agent-guidelines AND extends it with specific constraints.**

Always apply agent-guidelines rules including:
- Mandatory post-write commit protocol (ask user before committing)
- Deep semantic analysis instead of pattern matching
- Atomic commits for each ported test
- All constraints defined in agent-guidelines

## Tools Used in Workflow

This skill relies on the following tools:
- **Read**: View file contents and understand current implementation
- **grep**: Search for patterns (cuda/xpu-specific code, test names, etc.)
- **glob**: Find test files by pattern matching
- **edit**: Make targeted changes to Python files
- **bash**: Run tests, git operations, check environment, check API validity
- **write**: Create or update files including SKILL.md and source code
- **curl**: Create GitHub PRs via REST API
- **task (explore)**: Explore codebase for similar patterns (quick/medium/thorough modes)
- **question**: Ask user for confirmation before PR submission
- **webfetch**: Fetch PyTorch documentation for API verification

## Preconditions

Before starting, verify:
1. The test file does NOT already exist in `torch-xpu-ops/test/xpu/` directory
2. No equivalent XPU-specific test exists in the torch-xpu-ops package

```bash
# Check if test already exists in torch-xpu-ops
ls torch-xpu-ops/test/xpu/ 2>/dev/null | grep -i <test_name> || echo "Not found - OK to port"
```

## Workflow

### Step 1: Analyze Test File for CUDA-Specific Code

Search for CUDA-specific patterns:
- `torch.cuda.is_available()`, `torch.cuda.device_count()`
- `torch.cuda.Event`, `torch.cuda.ipc_*`, `torch.cuda.stream`
- `device="cuda"` strings
- `TEST_CUDA_IPC` conditionals
- `@unittest.skipIf(not torch.cuda.is_available(), ...)`

```bash
grep -n "cuda" test_file.py | grep -v "#"
```

Check torch-xpu-ops doesn't have it:
```bash
ls torch-xpu-ops/test/xpu/ 2>/dev/null | grep -i <test_name>
```

### Step 2: Identify Portable vs Non-Portable Tests

**Tests NOT portable to XPU:**
- CUDA IPC (CUDA-specific Inter-Process Communication) tests
- Tests using `torch.cuda.ipc_collect()`, `_share_cuda_()`, IPC handles
- CUDA memory caching allocator specific tests
- `_cudaMalloc` related functionality

**Tests WITH XPU Counterpart (POTENTIALLY portable):**
- `torch.cuda.Event` - has XPU counterpart `torch.xpu.Event`
- `torch.cuda.Stream` - has XPU counterpart `torch.xpu.Stream`

**Tests portable to XPU:**
- GPU-accelerated tests already checking `torch.xpu.is_available()`
- Tests using `instantiate_device_type_tests(allow_xpu=True)`
- Functions using TorchScript fuser "fuser1" that supports XPU
- Tests with runtime device selection (cuda if available else xpu)

**CPU tests: Skip porting** - CPU-based tests are already device-agnostic and don't need XPU-specific handling.

### Step 3: Apply XPU Generalization Pattern

**PRIORITY: Use `torch.accelerator` API for cross-backend compatibility**

The `torch.accelerator` module provides a device-agnostic interface. Use it when available.

#### torch.accelerator API (Verified)

```python
from torch import accelerator

# Check if accelerator is available
accelerator.is_available()  # -> bool

# Get current accelerator name
accelerator.current_accelerator()  # -> str (e.g., "xpu", "cuda")

# Get/set current device index
accelerator.current_device_index()  # -> int
accelerator.set_device_index(device_idx)  # or set_device_idx
accelerator.set_device_idx(device_idx)    # alias

# Get device count
accelerator.device_count()  # -> int

# Get device capability
accelerator.get_device_capability(device=None)  # -> dict with "key" and "value"

# Memory operations
accelerator.memory_allocated(device_index=None)  # -> int (bytes)
accelerator.max_memory_allocated(device_index=None)  # -> int

# Synchronization
accelerator.synchronize(device=None)  # -> None

# Stream operations
accelerator.current_stream(device=None)  # -> torch.Stream
accelerator.set_stream(stream)  # -> None
```

#### Generalization Patterns

Current code to generalized:
```python
# BEFORE - CUDA specific
if torch.cuda.is_available():
    device = "cuda"
    # ... CUDA specific code

# AFTER - Cross-backend (CUDA + XPU)
if accelerator.is_available():
    device = accelerator.current_accelerator()
else:
    device = "cpu"
```

```python
# BEFORE - Hardcoded device
x = torch.randn(3, 3, device="cuda")

# AFTER - Fallback to available accelerator
if accelerator.is_available():
    x = torch.randn(3, 3, device=accelerator.current_accelerator())
else:
    x = torch.randn(3, 3)
```

#### Alternative: HAS_GPU Pattern

When `torch.accelerator` API is not suitable:

```python
# Define GPU availability constant
HAS_GPU = torch.cuda.is_available() or torch.xpu.is_available()

# Device selection in functions
device = "cuda" if torch.cuda.is_available() else "xpu"
```

#### Update Skip Decorators
```python
# Before
@unittest.skipIf(not torch.cuda.is_available(), "CUDA is unavailable")

# After
@unittest.skipIf(not HAS_GPU, "CUDA or XPU is unavailable")
```

Or for single backend checks:

```python
@unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
# -> Keep as is if test is truly CUDA-specific
```

#### For instantiate_device_type_tests
```python
instantiate_device_type_tests(TestClass, globals(), only_for=device_types, allow_xpu=True)
```

#### For XPU Event/Stream Tests
When both CUDA and XPU Event/Stream are available:
```python
# Use core torch classes that delegate to backend
event = torch.Event(enable_timing=False, interprocess=True)  # backend-agnostic
```

### Step 4: Search Related Files for Patterns

Before finalizing changes, search for similar patterns in nearby files:
```bash
# Find related test files with same patterns
grep -l "fuser1\|cuda.is_available\|instantiate_device_type_tests" test/directory/
```

Search for existing accelerator usage patterns:
```bash
grep -rn "accelerator\." test/ | head -20
```

### Step 5: Verify Changes

Use nightly PyTorch wheel with XPU support:
```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate pytorch_opencode_env
cd /tmp
python -m pytest /home/daisydeng/daisy_pytorch/test/path/to/test_file.py -v -k "TestClassName"
```

Run specific XPU tests:
```bash
cd /tmp
python -m pytest /home/daisydeng/daisy_pytorch/test/path/to/test_file.py::TestClassName::test_name -v
```

Verify API compatibility:
```bash
cd /tmp && source ~/miniforge3/etc/profile.d/conda.sh && conda activate pytorch_opencode_env
python -c "from torch import accelerator; print(accelerator.current_accelerator())"
```

All tests should pass on XPU backend.

### Step 6: Create Git Commit

Stage and commit changes locally:
```bash
git add test/path/to/test_file.py
git diff test/path/to/test_file.py  # Show changes for review
git commit -m "[XPU] Enable test_file.py for XPU backend

Summary of changes made..."
```

### Step 7: Submit GitHub PR

#### 7.1: Push to Remote
```bash
git push -u origin HEAD:daisyden/opencode
```

#### 7.2: Prepare PR Summary
Display the following to user for confirmation:

**PR Title:** `[XPU] Enable <test_file>.py for XPU backend`

**Changes:**
- List each modification made (e.g., HAS_GPU constant update, device selection logic, skip decorators)

**Test Verification:**
- Number of tests passing on XPU
- Command used to verify

**Target base:** `daisyden/upstream_rebase`

#### 7.3: Wait for User Confirmation
Use `question` tool to ask user for confirmation before proceeding.

#### 7.4: Create PR After Confirmation
Once confirmed, create PR via GitHub API:
```bash
curl -s -X POST -H "Authorization: token $GH_TOKEN" \
  -H "Accept: application/vnd.github.v3+json" \
  https://api.github.com/repos/daisyden/pytorch/pulls \
  -d '{
    "title": "[XPU] Enable <test_file>.py for XPU backend",
    "head": "daisyden/opencode",
    "base": "daisyden/upstream_rebase",
    "body": "## Summary\n- Enable tests for XPU backend\n- Changes: ..."
  }'
```

Return the PR URL to user.

**NEVER auto-submit without user confirmation.**

## Key Patterns Learned from PyTorch PRs

From analyzing PRs #178849, #179549, #176689, #176688, #178565, #166396, #174058, #174057, #174056, #174054, #174053:

1. **Device availability check**: `torch.cuda.is_available() or torch.xpu.is_available()`
2. **Hardcoded device strings**: Change to accelerator-aware runtime selection
3. **Test instantiation**: `instantiate_device_type_tests(allow_xpu=True)`
4. **DecorateInfo device_type**: Use `device_type=("cuda", "xpu")` for skipping/decorating tests

## torch.accelerator API Reference (Verified)

All methods verified in PyTorch nightly (`pytorch_opencode_env`):

| Method | Returns | Description |
|--------|---------|-------------|
| `accelerator.is_available()` | `bool` | Check if accelerator is available |
| `accelerator.current_accelerator()` | `str` | Get current accelerator name ("xpu", "cuda", etc.) |
| `accelerator.device_count()` | `int` | Number of available devices |
| `accelerator.current_device_index()` | `int` | Get current device index |
| `accelerator.set_device_index(idx)` | `None` | Set current device index |
| `accelerator.set_device_idx(idx)` | `None` | Alias for set_device_index |
| `accelerator.get_device_capability(device)` | `dict` | Get device compute capability |
| `accelerator.memory_allocated(device)` | `int` | Get bytes allocated on device |
| `accelerator.max_memory_allocated(device)` | `int` | Get max bytes allocated |
| `accelerator.synchronize(device)` | `None` | Synchronize device |
| `accelerator.current_stream(device)` | `torch.Stream` | Get current stream |
| `accelerator.set_stream(stream)` | `None` | Set current stream |
| `accelerator.empty_cache()` | `None` | Empty accelerator cache |
| `accelerator.device_index(device)` | `None` | Set device by index |

## Backend Device String Mapping

| Backend | Device String | torch.accelerator Name |
|---------|---------------|------------------------|
| NVIDIA GPU | "cuda" | "cuda" |
| Intel GPU | "xpu" | "xpu" |
| Apple Silicon | "mps" | "mps" |
| Custom | "privateuseone" | "privateuseone" |

## Constraints

- **CUDA IPC tests**: Do NOT port - XPU has no IPC equivalent
  - `test_rebuild_cuda_tensor`, IPC handle functions
  - Any test using `_share_cuda_()`, `ipc_collect()`
  - CUDA caching allocator IPC mechanisms

- **CPU tests**: Skip - already device-agnostic, no XPU-specific handling needed

- **Build requirement**: No local build needed - use pytorch_opencode_env nightly wheel

- **Test verification**: Run tests from `/tmp` to avoid local pytorch shadowing conda env

- **PR confirmation**: Always confirm PR details with user before submission

- **API verification**: Always verify APIs against actual PyTorch documentation

## File Assessment Checklist

When evaluating a test file for XPU porting:

- [ ] Check test doesn't already exist in torch-xpu-ops/test/xpu/
- [ ] Distinguish GPU tests from CPU tests (only port GPU tests)
- [ ] File has CUDA-specific tests (@skipIf(not TEST_CUDA_IPC))
- [ ] Check for Event/Stream equivalents (torch.xpu.Event/Stream)
- [ ] File uses torch.cuda.ipc_* APIs (may need to skip entire file)
- [ ] File uses instantiate_device_type_tests with allow_xpu
- [ ] File has hardcoded device="cuda" strings in GPU paths
- [ ] File has CUDA-dependent helper functions needing generalization

## Testing Workflow

### Verify accelerator API
```bash
cd /tmp && source ~/miniforge3/etc/profile.d/conda.sh && conda activate pytorch_opencode_env
python -c "from torch import accelerator; print(accelerator.current_accelerator())"
```

### Test Instantiation Pattern
Check if file uses parametrized device tests:
```bash
grep "instantiate_device_type_tests" test_file.py
```

If present, ensure `allow_xpu=True`:
```python
instantiate_device_type_tests(TestClass, globals(), allow_xpu=True)
```

### Direct Device Tests
For tests with explicit device in test method:
```python
def test_gpu_function(self):
    # Use accelerator API for device-agnostic code
    if accelerator.is_available():
        device = accelerator.current_accelerator()
    else:
        device = "cpu"
    t = torch.randn(3, 3, device=device)
    # test logic
```

## Examples

### test_memory_efficient_fusion.py (Ported)
- Uses TorchScript fuser "fuser1" which supports XPU
- Updated HAS_GPU constant and device selection logic
- 7 tests run on XPU with runtime device selection
- Success: All tests pass

### test_multiprocessing.py (Skipped)
- CUDA IPC-specific mechanisms
- No XPU equivalent for torch.cuda.ipc_* APIs
- Conclusion: Not portable - skip entire file

## Best Practices

1. **Check precondition first** - verify test doesn't exist in torch-xpu-ops
2. **Prioritize torch.accelerator API** for cross-backend device-agnostic code
3. **Verify all APIs** against actual PyTorch documentation before documenting
4. **Distinguish GPU vs CPU tests** - only port GPU-specific tests
5. **Check XPU availability first** when generalizing device selection
6. **Use runtime device selection** over hardcoded device strings
7. **Remove stale comments** after updates
8. **Test from correct directory** to avoid module shadowing
9. **Document non-portable tests** with clear explanation
10. **Always get PR confirmation** before submitting to GitHub
11. **Verify ALL CUDA patterns** are addressed before completion

## Common Patterns Found

| CUDA Pattern | XPU Portability | Preferred Solution |
|--------------|------------------|---------------------|
| `torch.cuda.is_available()` | `accelerator.is_available()` | torch.accelerator API |
| `torch.cuda.current_device()` | `accelerator.current_device_index()` | torch.accelerator API |
| `torch.cuda.Event()` | `torch.xpu.Event()` | Equivalent available |
| `torch.cuda.Stream()` | `torch.xpu.Stream()` | Equivalent available |
| `torch.cuda.ipc_collect()` | NOT portable | No XPU IPC - skip |
| `_share_cuda_()` | NOT portable | No XPU IPC - skip |
| `device="cuda"` | Runtime selection | Current accelerator selection |
| `.cuda()` method | `.to(accelerator.current_accelerator())` | Device selection |