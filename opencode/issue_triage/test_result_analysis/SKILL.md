# Test Result Analysis Pipeline

## Overview
This skill provides the test result analysis pipeline that populates CI results, error messages, test existence status, and E2E benchmark accuracy data into `torch_xpu_ops_issues.xlsx`.

## When to Use
- When need to add CI test results to the Excel file
- When need to check if XPU/CUDA test cases exist
- When need to populate E2E benchmark accuracy status
- When building complete issue analysis (after `issue_basic_info_extraction` and `issue_analysis/pr-extraction`)

## Pipeline Steps

### Step 1: Process Test_Cases Sheet
Populates CI results, error messages, test existence, and dependency information.

```bash
cd ~/ai_for_validation/opencode/issue_triage/test_result_analysis/Test_Cases
python3 test_cases_processor.py
```

**What it does:**
1. **Load CI XML Files**: Load test results from:
   - Torch-xpu-ops nightly CI: `/home/daisydeng/issue_traige/ci_results/torch-xpu-ops/Inductor-XPU-UT-Data-*/`
   - Stock PyTorch XPU CI: `/home/daisydeng/issue_traige/ci_results/stock/test-reports-*.zip`

2. **Process PASS 1** (CI Results):
   - Match test cases to XML files
   - Populate columns 11-18:
     - `XPU Status`: pass/fail/skip/error in nightly CI
     - `XPU Comments`: failure messages
     - `Commit` & `Run_id`: CI build identifiers
     - `Stock Status`: pass/fail in stock PyTorch

3. **Process PASS 2** (Test Existence Check):
   - For "not found" cases, run LLM analysis
   - Check if test exists in CUDA (stock) and XPU (torch-xpu-ops)
   - Populate columns 19-21:
     - `CUDA Case Exist`: yes/no/uncertain
     - `XPU Case Exist`: yes/no/uncertain
     - `Case Existence Comments`: LLM analysis reason

4. **Dependency Analysis (RAG-based)**:
    - `load_ops_dependency()`: Load ops_dependency.csv into list for RAG matching
    - `get_dependency_from_ops_rag()`: RAG-based score matching against ops_dependency.csv
    - Scoring: exact (100), aten prefix (95), containment (80), word overlap (>0.5: 70), SequenceMatcher >0.7 (50)
    - Deps matched: oneDNN, oneMKL, Triton, ROCm, cuSPARSE, torch CPU
    - Populate `dependency_lib` (column 23)

5. **Duplicate Detection**:
   - Find cross-issue duplicated test cases
   - Mark in `duplicated_issue` (column 22)

**Columns Populated (11-23):**

| Col | Header |
|-----|--------|
| 11 | XPU Status |
| 12 | XPU Comments |
| 13 | Commit |
| 14 | Run_id |
| 15 | XML |
| 16 | Stock Status |
| 17 | Stock Comments |
| 18 | CUDA Case Exist |
| 19 | XPU Case Exist |
| 20 | Case Existence Comments |
| 21 | Can Enable on XPU |
| 22 | Duplicated Issue |
| 23 | Dependency Lib |

---

### Step 2: Process E2E_Test_Cases Sheet
Populates E2E benchmark accuracy status from CI reports.

```bash
cd ~/ai_for_validation/opencode/issue_triage/test_result_analysis/E2E_Test_Cases
python3 e2e_cases_processor.py
```

**What it does:**

1. **Find E2E Reports**: Locate all `Inductor_E2E_Test_Report.xlsx` files:
   - Path: `/home/daisydeng/issue_traige/ci_results/torch-xpu-ops/*E2E*/Inductor_E2E_Test_Report.xlsx`

2. **Parse Sheet Names**: Extract metadata from sheet names:
   - Format: `{benchmark}_{dtype}_{phase}_acc`
   - Example: `huggingface_float32_inf_acc`, `timm_bfloat16_train_amp_acc`

3. **Build Status Mapping**: Map each (benchmark, dtype, amp, phase, model) to status:
   - Status values: pass, fail_to_run, fail_to_compile,准确度误差 (accuracy error), skipped, etc.

4. **Match and Write**: For each row in E2E_Test_Cases sheet:
   - Match based on benchmark, dtype, phase, amp, model
   - Populate accuracy status

**Columns Populated:**

| Col | Header |
|-----|--------|
| 13 | XPU Status (accuracy) |

**Supported Benchmarks:**
- `huggingface` - HuggingFace models (e.g., AlbertForMaskedLM, BertForQuestionAnswering)
- `timm_models` - TIMM models (e.g., resnet50, vit_base_patch16_224)
- `torchbench` - TorchBench models (e.g., BERT_pytorch, resnet18)

---

## Full Pipeline Usage

```bash
# Step 1: Run Test Cases processor
cd ~/ai_for_validation/opencode/issue_triage/test_result_analysis/Test_Cases
python3 test_cases_processor.py

# Step 2: Run E2E Test Cases processor  
cd ~/ai_for_validation/opencode/issue_triage/test_result_analysis/E2E_Test_Cases
python3 e2e_cases_processor.py
```

## Complete Workflow

```bash
# Prerequisite: Generate Excel with issues and test cases
cd ~/ai_for_validation/opencode/issue_triage/issue_analysis/issue_basic_info_extraction
python3 generate_excel.py

# Step 1: Extract PR references (optional - for issue-tracking)
cd ~/ai_for_validation/opencode/issue_triage/issue_analysis/pr-extraction
python3 pr_extraction.py $RESULT_DIR/torch_xpu_ops_issues.xlsx

# Step 2: Process Test_Cases (CI results, error messages, existence, deps)
cd ~/ai_for_validation/opencode/issue_triage/test_result_analysis/Test_Cases
python3 test_cases_processor.py

# Step 3: Process E2E_Test_Cases (E2E accuracy status)
cd ~/ai_for_validation/opencode/issue_triage/test_result_analysis/E2E_Test_Cases
python3 e2e_cases_processor.py

# Optional: Update categories with new info
cd ~/ai_for_validation/opencode/issue_triage/issue_analysis/category
python3 category_analyzer.py $RESULT_DIR/torch_xpu_ops_issues.xlsx
```

## Input/Output

**Input Excel:**
- `torch_xpu_ops_issues.xlsx` with Issues, Test_Cases, E2E_Test_Cases sheets

**Output Excel (updated):**
- Test_Cases sheet: columns 11-23 populated
- E2E_Test_Cases sheet: columns 13 populated

**Log File:**
- `$RESULT_DIR/pipeline.log`

## Prerequisites
- Python with: openpyxl, requests, xml
- CI XML files in place:
  - `/home/daisydeng/issue_traige/ci_results/torch-xpu-ops/Inductor-XPU-UT-Data-*/`
  - `/home/daisydeng/issue_traige/ci_results/stock/test-reports-*.zip`
  - `/home/daisydeng/issue_traige/ci_results/torch-xpu-ops/*E2E*/Inductor_E2E_Test_Report.xlsx`

## Related Skills
- issue_analysis/issue_basic_info_extraction: Create initial Excel with issues
- issue_analysis/pr-extraction: Add PR references from comments
- issue_analysis/category: Analyze and categorize issues using LLM