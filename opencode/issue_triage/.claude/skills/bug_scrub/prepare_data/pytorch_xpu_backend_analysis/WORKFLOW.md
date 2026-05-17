# PyTorch XPU Backend Analysis — Workflow

## Position in Bug Scrub Pipeline

This skill is **Phase 1.4** of the [bug scrub pipeline](../../WORKFLOW.md). It runs in parallel with Phase 1.2 (download CI results) and Phase 1.3 (create not-applicable sheet), after Phase 1.1 (issue basic info extraction) has completed.

```
Phase 1 — Prepare Data
├── 1.1 issue-basic-info-extraction   ← must complete first
├── 1.2 download_ci_result            ← parallel
├── 1.3 create-not-applicable-sheet   ← parallel
└── 1.4 pytorch_xpu_backend_analysis  ← THIS SKILL (parallel)
```

---

## Purpose

Produce a comprehensive technical reference document (`SKILL.md`) that catalogs:

1. **Operator registration** — how the 749 operators in `xpu_functions.yaml` are declared, code-generated, and dispatched at runtime
2. **Dependency library architecture** — which libraries (SYCL, oneMKL, oneDNN, XCCL, SYCLTLA) back which operators, their build flags, and conditional compilation patterns
3. **Implementation patterns** — dispatch stub, `TORCH_IMPL_FUNC`, oneDNN primitive, SYCL kernel functor, and CPU fallback patterns

This document is consumed downstream by **Phase 3.3 (triage_skills)** as a reference for root cause analysis and dependency classification.

---

## Input

| Source | Description |
|--------|-------------|
| `pytorch/third_party/torch-xpu-ops` | Target repository (source code, YAML specs, CMake configs) |
| PyTorch mainline | ATen operator infrastructure, dispatcher, torchgen pipeline |

---

## Workflow Steps

### Step 1 — Codebase Structure Survey

Scan the torch-xpu-ops repository layout:

- `cmake/` — build configuration files (`Codegen.cmake`, `ONEMKL.cmake`, `SYCL.cmake`, etc.)
- `src/ATen/native/xpu/` — operator implementation files (`.cpp`)
- `src/ATen/native/xpu/sycl/` — SYCL kernel implementations (100+ files)
- `src/ATen/native/xpu/mkl/` — oneMKL wrapper implementations
- `yaml/xpu_functions.yaml` — operator registry (749 entries)

### Step 2 — Dependency Library Analysis

For each dependency library, document:

| Library | Build Flag | Mandatory? | Operator Families |
|---------|-----------|------------|-------------------|
| SYCL | Always enabled | Yes | All XPU operations |
| oneMKL | `USE_ONEMKL_XPU` | No (default ON) | BLAS, LAPACK, FFT |
| oneDNN | `USE_DNNL_XPU` | No | Convolution, normalization |
| XCCL | `USE_XCCL` | No | Collective communications |
| SYCLTLA | `USE_SYCLTLA` | No | Tensor linear algebra extensions |

Capture the conditional compilation pattern (`#if defined(USE_ONEMKL_XPU)` → optimized path, `#else` → CPU fallback).

### Step 3 — Operator Registration Mechanism Analysis

Document the four-layer architecture:

1. **Specification** — YAML declarations in `xpu_functions.yaml`
2. **Code generation** — `torchgen` pipeline orchestrated by `cmake/Codegen.cmake`
3. **Dispatch** — runtime routing via `TORCH_LIBRARY_IMPL` macros and fallback logic (`XPUFallback.template`)
4. **Implementation** — native SYCL kernels, oneMKL/oneDNN integration, or CPU fallback

### Step 4 — Implementation Pattern Cataloging

Identify and document each pattern with code examples:

- **Dispatch stub** (`REGISTER_XPU_DISPATCH`) — element-wise/reduction ops via `TensorIterator`
- **TORCH_IMPL_FUNC** — standalone XPU implementations (e.g., `mm_out_xpu`)
- **oneDNN primitive** — convolution/linear via cached primitives
- **SYCL kernel functor** — type-dispatched `AT_DISPATCH_*` + `gpu_kernel`
- **CPU fallback** — tensor transfer via `xpu_fallback_impl` with device consistency checks

### Step 5 — Operator Classification & Statistics

Produce the coverage matrix:

| Implementation Path | Operator Count |
|--------------------|---------------|
| Native SYCL kernels | 412 |
| oneMKL integration | 65 |
| oneDNN integration | 59 |
| CPU fallback | 213 |
| **Total** | **749** |

Break down by functional domain (arithmetic, activations, reductions, linear algebra, convolution, normalization, indexing, etc.).

### Step 6 — Anti-Patterns & Best Practices

Document common pitfalls:
- Missing `DeviceGuard`
- Incomplete type dispatch (`AT_DISPATCH_FLOATING_TYPES` vs `AT_DISPATCH_ALL_TYPES_AND_COMPLEX`)
- Naive host allocation instead of pinned memory

---

## Output

| Artifact | Path | Size | Description |
|----------|------|------|-------------|
| `SKILL.md` | `prepare_data/pytorch_xpu_backend_analysis/SKILL.md` | ~51KB | Comprehensive analysis document (12 parts + appendices) |

---

## Downstream Consumers

| Phase | Skill | How It Uses This Document |
|-------|-------|--------------------------|
| 3.3 | `triage_skills` | Reference for dependency classification (operator → library mapping) and root cause analysis (implementation path identification) |

The connection is shown in the [bug scrub workflow diagram](../../WORKFLOW.md) as a dashed "reference" arrow from `pytorch_xpu_backend_analysis.md` to the triage step.

---

## Version

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | May 9, 2026 | Initial workflow description |
