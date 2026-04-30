# SKILL — Mini Reproducer (Phase 3 / STEP 3.5)

This sub-skill defines how a triage agent writes and verifies a minimal Python
reproducer for a torch-xpu-ops issue, between STEP 3 (Code Exploration) and
STEP 4 (Runtime Verification) of the parent triage workflow (`SKILL.md`).

## Purpose

Convert the issue's prose error report into an executable Python script that:

1. Triggers the **same error** (exception type + message substring) on the
   local conda env `pytorch_opencode_env`, and
2. Is small enough that a kernel/op author can debug it without setting up
   the full test harness.

Reproducer artifacts feed both the triage JSON (`mini_reproducer` field) and
the eventual fix PR (the author can paste the script into their bug template).

## When to write a reproducer — and when to skip

| Issue Category / Symptom | Write reproducer? |
|---|---|
| `Torch Operations` numerical / shape / dtype bug | **Yes** |
| `Inductor` / `torch.compile` codegen crash | **Yes** |
| `Distributed` collective op crash on a single rank | **Yes** (single-process where possible) |
| `Flash Attention` / SDPA crash or numerical | **Yes** |
| `TorchAO` quantization error | **Yes** |
| `Sparse` op error | **Yes** |
| `Torch Runtime` OOM / device crash with minimal reproduction | **Yes** |
| `Distributed` multi-rank deadlock | **Best-effort** (document if cannot reproduce single-process) |
| `Others` — CI infra / docs / test harness / meta-tracking | **No** — skip and omit `mini_reproducer` from JSON |
| Issue requires private branch / unreleased build | **No** — set `reproduced: false` with `notes` explaining the version gap |

## Output artifacts

For issue `<id>`, write exactly two files under
`/home/daisydeng/pytorch/agent_space/phase3_triage/`:

| Path | Content |
|---|---|
| `repro_<id>.py` | The reproducer script (UTF-8, ≤30 LOC ideally, ≤80 LOC hard limit) |
| `repro_<id>.log` | Combined stdout+stderr from the verification run |

Both are referenced by the JSON entry:

```json
"mini_reproducer": {
  "path":       "agent_space/phase3_triage/repro_3530.py",
  "reproduced": true,
  "notes":      "Reproduces with bf16 input + fp32 alpha; matches IndexError in original report."
}
```

## Reproducer template

Copy this skeleton, then specialize. Keep it self-contained — no helper modules.

```python
"""Mini reproducer for issue #<id>: <one-line title>.

Source: https://github.com/intel/torch-xpu-ops/issues/<id>
Expected error: <ExceptionClass>: <substring of original message>
"""
import torch

torch.manual_seed(0)

device = "xpu"  # or "cuda"/"cpu" if comparison needed
dtype  = torch.bfloat16   # match issue

# --- minimal inputs (hard-coded shapes / values) -------------------------
x = torch.randn(4, 8, device=device, dtype=dtype)
idx = torch.tensor([0, 1, 2, 3], device=device)

# --- failing call --------------------------------------------------------
out = torch.index_add(x, 0, idx, x, alpha=1.0)

print("OK", out.dtype, out.shape)
```

### Skeleton conventions

- **Imports**: only `torch` (and `torch.distributed` if absolutely required).
  No `unittest`, no `pytest`, no internal `_internal.common_utils`.
- **Determinism**: always `torch.manual_seed(0)`.
- **Device**: prefer `xpu`; fall back to `cuda` only if the issue is
  reproducible on `cuda` and absent on `xpu`.
- **Dtype**: copy the failing dtype from the issue. If the bug is
  dtype-specific (most accuracy bugs are), make the dtype explicit.
- **Inputs**: hard-coded shapes; use `torch.randn` / `torch.arange` /
  `torch.zeros` — not `torch.testing.make_tensor`.
- **Failing call**: invoke the exact API from the issue. No try/except —
  let the exception propagate so it shows up in the log.
- **Trailing print**: a final `print("OK", ...)` lets you tell "no error
  raised" from "error raised" in the log.

## Verification loop

```bash
source ~/miniforge3/bin/activate pytorch_opencode_env
cd /home/daisydeng/pytorch
python agent_space/phase3_triage/repro_<id>.py \
   > agent_space/phase3_triage/repro_<id>.log 2>&1
echo "exit=$?"
```

### Acceptance criteria

`reproduced = true` requires **both**:

1. **Exception type match** — the Python exception class in the log is the
   same as the one the issue reports (`RuntimeError`, `IndexError`,
   `AssertionError`, `torch._dynamo.exc.BackendCompilerFailed`, etc.).
2. **Message substring match** — at least one short, distinctive phrase
   from the issue's error message appears verbatim in the log
   (e.g. `"index out of range"`, `"Scalars or Tensor-likes are not equal"`,
   `"page fault was detected"`).

If only one matches, set `reproduced = false` and explain in `notes`.

### Iteration budget

Maximum **3 iterations** before giving up:

| Iteration | Try varying |
|---|---|
| 1 | dtypes (bf16 ↔ fp16 ↔ fp32) |
| 2 | shapes (smaller / non-power-of-2 / batched) |
| 3 | flags (`torch.use_deterministic_algorithms`, `TORCH_USE_XCCL=1`, env vars from issue) |

If still not reproduced, set `reproduced: false` and document what was tried
in `notes`. **Do not** keep iterating beyond 3 — record the negative result and
move on. Triage is not blocked on a missing reproducer.

## Failure modes — how to record

| Situation | `reproduced` | `notes` example |
|---|---|---|
| Same exception, same message | `true` | "Reproduces verbatim with bf16+alpha=1.0." |
| Same exception, different message | `false` | "Got `RuntimeError: <other>` — issue's stack trace mentions a different code path; suspect intervening fix in HEAD." |
| Different exception | `false` | "Got `AssertionError` instead of reported `IndexError`; likely env-specific." |
| Silent pass | `false` | "No error raised; cannot reproduce on local HEAD `<sha>` with bf16/fp16/fp32." |
| Requires multi-rank / specific HW | `false` | "Reproducer requires 2× XPU; single-rank version exits cleanly. Documented for fix author." |
| Skipped by category | omit field | (don't emit `mini_reproducer` at all) |

## Worked example (#3530 `index_add_` bf16)

```python
"""Mini reproducer for issue #3530: index_add_ bf16/fp32-alpha accuracy.

Source: https://github.com/intel/torch-xpu-ops/issues/3530
Expected error: AssertionError: Tensor-likes are not close!
"""
import torch
torch.manual_seed(0)

device = "xpu"
src   = torch.randn(8, 4, device=device, dtype=torch.bfloat16)
idx   = torch.tensor([0, 2, 4, 6], device=device)
add_v = torch.randn(4, 4, device=device, dtype=torch.bfloat16)

out_xpu = src.clone().index_add_(0, idx, add_v, alpha=2.5)
out_cpu = src.cpu().index_add_(0, idx.cpu(), add_v.cpu(), alpha=2.5)

torch.testing.assert_close(out_xpu.cpu(), out_cpu, rtol=1e-2, atol=1e-2)
print("OK")
```

Verification:

```bash
$ python agent_space/phase3_triage/repro_3530.py
Traceback (most recent call last):
  ...
AssertionError: Tensor-likes are not close!
```

JSON emission:

```json
"mini_reproducer": {
  "path":       "agent_space/phase3_triage/repro_3530.py",
  "reproduced": true,
  "notes":      "bf16 + fp32 alpha accumulation diverges from CPU; matches issue."
}
```

## Anti-patterns

- ❌ Reproducer pulls in `torch.testing._internal.common_utils.TestCase` —
  defeats the "self-contained" goal.
- ❌ Reproducer uses `pytest.raises` / `assertRaises` and prints "OK" — the log
  must show the actual traceback for the verification grep.
- ❌ Reproducer tries 5+ shape combinations in a loop — pick one. If you
  truly need a sweep, that belongs in the fix PR's regression test, not here.
- ❌ Reproducer wraps the failing call in `try/except` — let it crash.
- ❌ Reproducer downloads a model / dataset — must run offline.
- ❌ Storing reproducers anywhere other than `agent_space/phase3_triage/` —
  that directory is gitignored and the canonical scratch location.
