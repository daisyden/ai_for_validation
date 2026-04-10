# Torch XPU Ops Issue Report

**Repository:** [intel/torch-xpu-ops](https://github.com/intel/torch-xpu-ops)

**Issues Extracted:** April 5, 2026 (from GitHub issue data: `torch_xpu_ops_issues.json`)

**CI Data Sources:**
- Torch-XPU-OPS Nightly CI: XML files from `Inductor-XPU-UT-Data-*` + `Inductor-wheel-nightly-LTS2-XPU-E2E-Data-*` (commit: `a2d516a58c64f18b76880f3a77efbc02885d65af`)
- Stock PyTorch XPU CI: XML files from `test-default-*-linux.idc.xpu_*.zip` (run IDs: 69741866812, 69741866834, 69741866862, 69741866911, etc.)

Generated: 2026-04-10 01:53:17

---

## Index

1. [Summary](#user-content-1-summary)
2. [Statistics](#user-content-2-statistics)
   - [By Test Module](#user-content-by-test-module)
   - [By Module](#user-content-by-module)
   - [By Dependency](#user-content-by-dependency)
   - [By Action TBD](#user-content-by-action-tbd)
   - [By Category](#user-content-by-category-stats)
   - [By Priority](#user-content-by-priority)
3. [New Submitted Issues (Past Week)](#user-content-3-new-submitted-issues-past-week)
4. [Action Required](#user-content-4-action-required)
   - [Reporter Actions](#user-content-reporter-actions)
     - [Information Required](#user-content-information-required)
     - [Close Fixed Issue](#user-content-close-fixed-issue)
     - [Enable Test](#user-content-enable-test)
     - [Add to Skiplist](#user-content-add-to-skiplist)
     - [Verify the Issue](#user-content-verify-the-issue)
     - [Need Reproduce Steps](#user-content-need-reproduce-steps)
   - [Engineer Actions](#user-content-engineer-actions)
     - [Needs PyTorch Repo Changes (upstream)](#user-content-needs-pytorch-repo-changes-upstream)
     - [Revisit the PR as Case Failed](#user-content-revisit-the-pr-as-case-failed)
5. [By Category](#user-content-5-by-category)
6. [Duplicated Issues](#user-content-6-duplicated-issues)
7. [Issues with Dependency](#user-content-7-issues-with-dependency)

---

## 1. Summary

| Category | Count |
|----------|-------|
| Action Required | 1 |
| No Assignee | 2 |
| Duplicated Issues | 0 |
| With Dependency | 0 |
| Others | 2 |
| **Total** | 5 |

---

## 2. Statistics

### By Test Module

| Test Module | Count |
|-------------|-------|
| ut | 3 |
| e2e | 2 |

### By Module

| Module | Count |
|--------|-------|
| aten_ops | 5 |

### By Dependency

| Dependency | Count |
|------------|-------|

### By Action TBD

| Action TBD | Count |
|------------|-------|
| Awaiting response from reporter | 1 |

### By Category (Statistics)

| Category | Count |
|----------|-------|
| Skip/No Test Exists | 1 |
| Flash Attention/Transformer | 1 |
| Others | 1 |
| PT2E | 1 |
| Torch Operations | 1 |

### By Priority

| Priority | Count |
|----------|-------|
| P1 | 2 |
| P2 | 3 |

---

## 3. New Submitted Issues (Past Week)

Issues created in the past 7 days (as of 2026-04-10).

| ID | Title | Status | Owner | Priority | Reason | Category | Root Cause | Dependency | PR | PR Owner | PR Status | Labels | Module | Test Module |
|---|-------|--------|-------|---------|--------|----------|-----------|-----------|-------|----------|----------|--------|--------|-------------|

---

## 4. Action Required

### Reporter Actions

#### Information Required

| ID | Title | Owner | Owner Transferred | Priority | Category | Category Reason | Root Cause | Dependency | PR | PR Owner | PR Status | Test Module |
|---|-------|-------|-------------------|---------|----------|----------------|-----------|-----------|-------|----------|----------|-------------|
| [3258](https://github.com/intel/torch-xpu-ops/issues/3258) | Error in op: torch.ops.aten._scaled |  | bjarzemb | P2 | Flash Attention/Transformer | The error involves the operator `torch.ops.aten._scaled_dot_product_fused_attention_overrideable.default`, which is directly related to the scaled dot product attention mechanism used in transformer models. This categorization aligns with the Flash Attention/Transformer category due to the specific attention operation and its implementation context. | error involves scaled_dot_product_attention operator |  | https://github.com/pytorch/pytorch/pull/178986;https://github.com/pytorch/pytorch/pull/179239 | guangyey;etaf | merged;merged | ut |

#### Close Fixed Issue

| ID | Title | Owner | Owner Transferred | Priority | Category | Category Reason | Root Cause | Dependency | PR | PR Owner | PR Status | Test Module |
|---|-------|-------|-------------------|---------|----------|----------------|-----------|-----------|-------|----------|----------|-------------|

#### Enable Test

| ID | Title | Owner | Owner Transferred | Priority | Category | Category Reason | Root Cause | Dependency | PR | PR Owner | PR Status | Test Module |
|---|-------|-------|-------------------|---------|----------|----------------|-----------|-----------|-------|----------|----------|-------------|

#### Add to Skiplist

| ID | Title | Owner | Owner Transferred | Priority | Category | Category Reason | Root Cause | Dependency | PR | PR Owner | PR Status | Test Module |
|---|-------|-------|-------------------|---------|----------|----------------|-----------|-----------|-------|----------|----------|-------------|

#### Verify the Issue

| ID | Title | Owner | Owner Transferred | Priority | Category | Category Reason | Root Cause | Dependency | PR | PR Owner | PR Status | Test Module |
|---|-------|-------|-------------------|---------|----------|----------------|-----------|-----------|-------|----------|----------|-------------|

#### Need Reproduce Steps

| ID | Title | Owner | Owner Transferred | Priority | Category | Category Reason | Root Cause | Dependency | PR | PR Owner | PR Status | Test Module |
|---|-------|-------|-------------------|---------|----------|----------------|-----------|-----------|-------|----------|----------|-------------|

### Engineer Actions

#### Needs PyTorch Repo Changes (upstream)

| ID | Title | Owner | Owner Transferred | Priority | Category | Category Reason | Root Cause | Dependency | PR | PR Owner | PR Status | Test Module |
|---|-------|-------|-------------------|---------|----------|----------------|-----------|-----------|-------|----------|----------|-------------|

#### Revisit the PR as Case Failed

| ID | Title | Owner | Owner Transferred | Priority | Category | Category Reason | Root Cause | Dependency | PR | PR Owner | PR Status | Test Module |
|---|-------|-------|-------------------|---------|----------|----------------|-----------|-----------|-------|----------|----------|-------------|

---

## 5. By Category

#### Flash Attention/Transformer

| ID | Title | Status | Owner | Priority | Category Reason | Root Cause | Dependency | PR | PR Owner | PR Status | Labels | Module | Test Module |
|---|-------|--------|-------|---------|-----------------|-----------|-----------|-------|----------|----------|--------|--------|-------------|
| [3258](https://github.com/intel/torch-xpu-ops/issues/3258) | Error in op: torch.ops.aten._s | open |  | P2 | The error involves the operator `torch.ops.aten._scaled_dot_product_fused_attention_overrideable.default`, which is directly related to the scaled dot product attention mechanism used in transformer models. This categorization aligns with the Flash Attention/Transformer category due to the specific attention operation and its implementation context. | error involves scaled_dot_product_attention operator |  | https://github.com/pytorch/pytorch/pull/178986;https://github.com/pytorch/pytorch/pull/179239 | guangyey;etaf | merged;merged |  | aten_ops | ut |

#### Others

| ID | Title | Status | Owner | Priority | Category Reason | Root Cause | Dependency | PR | PR Owner | PR Status | Labels | Module | Test Module |
|---|-------|--------|-------|---------|-----------------|-----------|-----------|-------|----------|----------|--------|--------|-------------|
| [3257](https://github.com/intel/torch-xpu-ops/issues/3257) | [Linux][E2E][Regression] Huggi | open |  | P1 | The issue is categorized as Others because the provided information is insufficient to determine a specific category. The title and summary mention a failure in Huggingface test models on XPU but do not include specific keywords, operators, or error types that align with any of the predefined categories. | Backend/Device Issue - Huggingface test models fail_to_run on XPU due to improper device placement or backend compatibility issues during model execution. The failure occurs when model components are not correctly offloaded or executed on XPU, leading to silent failures without traceback. |  |  |  |  |  | aten_ops | e2e |

#### PT2E

| ID | Title | Status | Owner | Priority | Category Reason | Root Cause | Dependency | PR | PR Owner | PR Status | Labels | Module | Test Module |
|---|-------|--------|-------|---------|-----------------|-----------|-----------|-------|----------|----------|--------|--------|-------------|
| [3255](https://github.com/intel/torch-xpu-ops/issues/3255) | [Linux][PT2E][Regression] Some | open |  | P1 | The issue involves a regression in performance tests related to PyTorch Export (PT2E) as indicated by the label in the title and the mention of test failures in the context of different data types (fp32, int8 ASYMM, int8 SYMM). The test module is e2e, and the issue is tied to exported programs or export-related functionality, which falls under the PT2E category. | dtype-specific handling in PT2E compilation. The issue arises during inductor compilation when handling mixed precision inputs for ops like linear or conv2d, leading to incorrect kernel selection or execution on XPU. |  |  |  |  |  | aten_ops | e2e |

#### Skip/No Test Exists

| ID | Title | Status | Owner | Priority | Category Reason | Root Cause | Dependency | PR | PR Owner | PR Status | Labels | Module | Test Module |
|---|-------|--------|-------|---------|-----------------|-----------|-----------|-------|----------|----------|--------|--------|-------------|
| [3259](https://github.com/intel/torch-xpu-ops/issues/3259) | New failed test cases 2026-04- | open | SlawomirLaba | P2 | The issue title and summary indicate new failed test cases, but the provided test cases are marked as skipped (based on the "Labels: skipped" entry). Additionally, no specific error messages or runtime failures are described, suggesting the issue may stem from tests being skipped rather than failing due to an implementation problem. | Backend/Device Issue - The error occurs during XPU device initialization, as the traceback references torch.cuda.\_\_init\_\_.py, which is unrelated to XPU. The test involves creating a tensor on XPU with requires_grad=True, indicating a failure in device-specific setup or compatibility for XPU. |  |  |  |  | skipped | aten_ops | ut |

#### Torch Operations

| ID | Title | Status | Owner | Priority | Category Reason | Root Cause | Dependency | PR | PR Owner | PR Status | Labels | Module | Test Module |
|---|-------|--------|-------|---------|-----------------|-----------|-----------|-------|----------|----------|--------|--------|-------------|
| [3247](https://github.com/intel/torch-xpu-ops/issues/3247) | NotImplementedError: "dot_xpu_ | open | Silv3S | P2 | The error "dot_xpu_mkl not implemented for 'Long'" indicates a missing operator implementation for the Long dtype on XPU. This is a device-specific op dispatch issue, falling under Torch Operations as it relates to operator execution and kernel selection for a specific dtype and device. | Dtype/Precision Issue - The error "dot_xpu_mkl" not implemented for 'Long' indicates a dtype mismatch where the Long tensor is being used with an XPU-specific MKL-optimized dot product operation that does not support integer types. The operation likely expects floating-point dtypes like float32 or b |  |  |  |  | ut_upstream | aten_ops | ut |


---

## 6. Duplicated Issues

Issues that share test cases with other issues.

| ID | Title | Owner | Reporter | Duplicated With | Priority | Category Reason | Root Cause | Dependency | PR | PR Owner | PR Status | Labels | Module | Test Module |
|---|-------|-------|----------|-----------------|---------|-----------------|-----------|-----------|-------|----------|----------|--------|--------|-------------|

---

## 7. Issues with Dependency

Issues that have dependencies on other components.

| ID | Title | Owner | Priority | Category Reason | Root Cause | Dependency | Category | PR | PR Owner | PR Status | Labels | Test Module |
|---|-------|---------|---------|-----------------|-----------|------------|----------|-------|----------|----------|--------|-------------|
