# PASS 4: Dependency RAG

## Overview
Match torch operations to their dependency libraries using Retrieval-Augmented Generation.

## Workflow
1. Load operation-to-dependency mapping from RAG database
2. For each torch op in test cases, query the database
3. Match ops to their required libraries (e.g., rocBLAS, cuDNN, MKL)
4. Populate dependency_lib column

## Usage
```bash
cd /home/daisydeng/ai_for_validation/opencode/issue_triage/test_result_analysis/Test_Cases
python run_processor_steps.py --steps 4
```

## Dependency Categories
- **BLAS/LAPACK**: MKL, OpenBLAS, rocBLAS
- **DNN**: cuDNN, ROCcL, OneDNN
- **RNG**: cuRAND, rocRAND
- **FFT**: cuFFT, rocFFT
- **SPARSE**: cuSPARSE, rocSPARSE
- **RNN**: cuDNN, MIOpen
- **RNG**: PhiloX

## Input
- Test Cases sheet with Torch Op column populated
- RAG database of op-to-dependency mappings

## Output
- Column 6 "dependency_lib" populated
- Fast operation (no LLM required)

## Related Files
- test_cases_processor.py (pass4_dependency_rag function)