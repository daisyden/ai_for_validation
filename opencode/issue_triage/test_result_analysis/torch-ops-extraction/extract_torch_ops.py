#!/usr/bin/env python3
"""
Torch-ops Extraction Script

Extracts torch ops from PyTorch XPU test case data in Excel format.
Uses pattern-based extraction with LLM fallback for unknown cases.

Usage:
    python3 extract_torch_ops.py <input_file> [output_file]
    python3 extract_torch_ops.py torch_xpu_ops_issues.xlsx
"""

import pandas as pd
import re
import sys
import os
import time
import requests


LLM_ENDPOINT = os.environ.get("LLM_ENDPOINT", "http://10.239.15.43/v1/chat/completions")
LLM_API_KEY = os.environ.get("OPENCODE_API_KEY", "sk-xxxxxxxxxx")
LLM_MODEL = "Qwen3-32B"


def clean_op(op):
    """Remove device and dtype suffixes from op name"""
    op = re.sub(r'_(xpu|cuda)_(float\d+|int\d+|complex\d+)$', '', op)
    op = re.sub(r'_(float\d+|int\d+|complex\d+)$', '', op)
    op = re.sub(r'_(xpu|cuda)$', '', op)
    return op


def extract_ops_from_test_name(test_name):
    """Extract ops from test name using OpDB patterns"""
    if pd.isna(test_name) or test_name == 'nan':
        return []
    test_name = str(test_name)
    
    match = re.search(r'torch_ops_aten__(\w+)', test_name)
    if match:
        return [f'aten._{clean_op(match.group(1))}']
    
    match = re.search(r'__refs_(\w+)', test_name)
    if match:
        return [f'aten._{clean_op(match.group(1))}']
    
    match = re.search(r'_nn_(\w+)', test_name)
    if match:
        return [f'nn.{clean_op(match.group(1))}']
    
    match = re.search(r'_refs_(\w+)', test_name)
    if match:
        return [f'_{clean_op(match.group(1))}']
    
    match = re.search(r'aten__(\w+)', test_name)
    if match:
        return [f'aten.{clean_op(match.group(1))}']
    
    if 'fused_attention' in test_name or 'fused_kernel' in test_name:
        return ['aten.fused_attention']
    if 'sdpa' in test_name.lower() or 'sdp' in test_name.lower():
        return ['aten.scaled_dot_product_attention']
    if 'cudnn_attention' in test_name:
        return ['aten.cudnn_attention']
    if 'transformerencoder' in test_name:
        return ['torch.nn.TransformerEncoder']
    if 'transformer' in test_name:
        return ['torch.nn.Transformer']
    if 'flash_attention' in test_name or 'flash_atteention' in test_name:
        return ['aten.flash_attention']
    if 'mem_eff_attention' in test_name:
        return ['aten.memory_efficient_attention']
    
    match = re.search(r'vjp_linalg_(\w+)', test_name)
    if match:
        return [f'torch.linalg.{clean_op(match.group(1))}']
    
    if 'csr_matvec' in test_name:
        return ['aten.csr_matvec']
    if 'sparse_csr' in test_name or 'SparseCSR' in test_name:
        return ['aten.sparse_csr']
    if 'to_sparse' in test_name:
        return ['aten.to_sparse']
    if 'sparse' in test_name.lower():
        return ['sparse_ops']
    
    if 'rms_norm_decomp' in test_name:
        return ['aten.rms_norm']
    if '_fft_' in test_name or 'fft_' in test_name:
        return ['torch.fft']
    
    if 'has_decomposition' in test_name:
        return ['decomp_ops']
    
    return []


def extract_from_error_or_traceback(text):
    """Extract torch ops from error message or traceback"""
    if pd.isna(text) or text == 'nan':
        return []
    text = str(text)
    found = []
    
    matches = re.findall(r'torch\.ops\.aten\.(\w+)\.default', text)
    for m in matches[:3]:
        found.append(f'torch.ops.aten.{m}.default')
    
    if not found:
        matches = re.findall(r'torch\.ops\.aten\.(\w+)', text)
        for m in matches[:3]:
            found.append(f'torch.ops.aten.{m}')
    
    if not found:
        matches = re.findall(r'aten::(\w+)', text)
        for m in matches[:3]:
            found.append(f'aten.{m}')
    
    return list(set(found))


def extract_from_test_case_name(test_case):
    """Extract torch ops from test case name mapping"""
    if pd.isna(test_case) or test_case == 'nan':
        return []
    test_case = str(test_case)
    name = test_case
    name = re.sub(r'^test_out_', '', name)
    name = re.sub(r'^test_quick_', '', name)
    name = re.sub(r'^test_comprehensive_', '', name)
    name = re.sub(r'^test_error_', '', name)
    name = re.sub(r'^test_noncontiguous_samples_', '', name)
    name = re.sub(r'^test_neg_view_', '', name)
    name = re.sub(r'_xpu.*$', '', name)
    name = re.sub(r'_cuda.*$', '', name)
    
    op_mappings = {
        'addmv': 'torch.addmv', 'addmm': 'torch.addmm', 'bmm': 'torch.bmm',
        'matmul': 'torch.matmul', 'dot': 'torch.dot', 'mm': 'torch.mm', 'mv': 'torch.mv',
        'conv2d': 'torch.nn.functional.conv2d', 'conv_transpose2d': 'torch.nn.functional.conv_transpose2d',
        'conv_transpose3d': 'torch.nn.functional.conv_transpose3d',
        'cross_entropy': 'torch.nn.functional.cross_entropy',
        'logaddexp': 'torch.logaddexp', 'histogram': 'torch.histogram',
        'linalg_tensorsolve': 'torch.linalg.tensorsolve',
        'baddbmm': 'aten.baddbmm', 'logspace': 'torch.logspace', 'linspace': 'torch.linspace',
        'arange': 'torch.arange', 'range': 'torch.range',
        'ones': 'torch.ones', 'zeros': 'torch.zeros', 'full': 'torch.full',
        'empty': 'torch.empty', 'rand': 'torch.rand', 'randn': 'torch.randn',
        'randint': 'torch.randint', 'tensor': 'torch.tensor', 'tensor_split': 'torch.tensor_split',
        'sum': 'torch.sum', 'mean': 'torch.mean', 'prod': 'torch.prod',
        'neg': 'torch.neg', 'abs': 'torch.abs',
        'exp': 'torch.exp', 'log': 'torch.log', 'sqrt': 'torch.sqrt',
        'sin': 'torch.sin', 'cos': 'torch.cos', 'tan': 'torch.tan',
        'tanh': 'torch.tanh', 'sigmoid': 'torch.sigmoid',
        'view': 'torch.view', 'reshape': 'torch.reshape', 'flatten': 'torch.flatten',
        'squeeze': 'torch.squeeze', 'unsqueeze': 'torch.unsqueeze',
        'transpose': 'torch.transpose', 'perm': 'torch.permute',
    }
    
    for key, op in op_mappings.items():
        if key in name:
            return [op]
    
    return []


def extract_torch_ops_with_llm(test_file, test_case, error_msg, traceback):
    """
    LLM-based torch ops extraction using Qwen3-32B.
    Used as fallback when pattern-based extraction returns empty.
    Returns list of torch ops and elapsed time.
    """
    context_parts = []
    if test_file:
        context_parts.append(f"Test file: {test_file}")
    if test_case:
        context_parts.append(f"Test case: {test_case}")
    if error_msg:
        error_sample = str(error_msg)[:1500] if len(str(error_msg)) > 1500 else str(error_msg)
        context_parts.append(f"Error message: {error_sample}")
    if traceback:
        traceback_sample = str(traceback)[:1000] if len(str(traceback)) > 1000 else str(traceback)
        context_parts.append(f"Traceback: {traceback_sample}")
    
    context = '\n'.join(context_parts)
    
    prompt = f"""You are a PyTorch expert. Extract the torch operations involved in this test failure.

Context:
{context}

Common torch ops to identify:
- aten ops: add, mm, bmm, matmul, conv2d, softmax, layernorm, gelu, dropout, linear, embedding, etc.
- torch ops: torch.add, torch.matmul, torch.nn.functional.*, torch.linalg.*, torch.fft.*, etc.
- aten.*_default ops: _flash_attention_forward, _scaled_mm, _convolution_forward, etc.

Return ONLY a JSON list of torch operation names, e.g.:
["aten.add", "aten.mm", "aten.conv2d"]

If no specific torch op can be identified, return empty list []."""
    
    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": "You are a PyTorch operation analysis assistant. Return ONLY valid JSON list."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0,
        "max_tokens": 1024
    }
    
    start_time = time.time()
    
    try:
        response = requests.post(
            LLM_ENDPOINT,
            headers=headers,
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        result = response.json()
        elapsed = time.time() - start_time
        
        if result.get('choices') and len(result['choices']) > 0:
            content = result['choices'][0].get('message', {}).get('content', '')
            json_match = re.search(r'\[[^\]]*\]', content, re.DOTALL)
            if json_match:
                ops = json.loads(json_match.group())
                if isinstance(ops, list):
                    return [str(op) for op in ops if op], elapsed
        
        return [], elapsed
    
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"  [LLM OPS ERROR] {e} (elapsed: {elapsed:.1f}s)")
        return [], elapsed


def extract_torch_ops(test_file, test_case, error_msg, traceback, use_llm_fallback=True):
    """Extract torch ops using all rules in priority order.
    Falls back to LLM when pattern-based extraction returns empty.
    Returns: (ops_list, llm_elapsed_time_or_None)
    """
    found_ops = []
    llm_elapsed = None
    
    if error_msg:
        extracted = extract_from_error_or_traceback(error_msg)
        if extracted and any('default' in e for e in extracted):
            return extracted, None
    
    if test_case:
        extracted = extract_ops_from_test_name(test_case)
        if extracted:
            return extracted, None
    
    if test_case:
        extracted = extract_from_test_case_name(test_case)
        if extracted:
            return extracted, None
    
    if error_msg:
        extracted = extract_from_error_or_traceback(error_msg)
        if extracted:
            return extracted, None
    
    if traceback:
        extracted = extract_from_error_or_traceback(traceback)
        if extracted:
            return extracted, None
    
    if use_llm_fallback and found_ops == []:
        found_ops, llm_elapsed = extract_torch_ops_with_llm(test_file, test_case, error_msg, traceback)
    
    return found_ops if found_ops else [], llm_elapsed


def process_excel(input_file, output_file=None):
    """Process Excel file and extract torch ops"""
    if output_file is None:
        output_file = input_file
    
    print(f"Reading {input_file}...")
    try:
        df_test = pd.read_excel(input_file, sheet_name='Test Cases')
    except Exception as e:
        print(f"Error reading Test Cases sheet: {e}")
        return
    
    print(f"Processing {len(df_test)} test cases...")
    
    llm_fallback_count = 0
    llm_total_time = 0
    results = []
    
    for i in range(len(df_test)):
        row = df_test.iloc[i]
        test_case = str(row['Test Case']) if pd.notna(row['Test Case']) else ''
        error_msg = str(row['Error Message']) if pd.notna(row['Error Message']) else ''
        traceback = str(row['Traceback']) if pd.notna(row['Traceback']) else ''
        
        ops, llm_elapsed = extract_torch_ops(None, test_case, error_msg, traceback)
        if not ops:
            ops = ['unknown']
        else:
            if llm_elapsed is not None:
                llm_fallback_count += 1
                llm_total_time += llm_elapsed
        
        results.append(', '.join(ops))
        
        if (i + 1) % 500 == 0:
            print(f"  Progress: {i + 1}/{len(df_test)} processed")
    
    df_test['torch-ops'] = results
    
    print(f"\nPattern extraction complete")
    if llm_fallback_count > 0:
        avg_time = llm_total_time / llm_fallback_count if llm_fallback_count > 0 else 0
        print(f"  LLM fallback used: {llm_fallback_count} calls, total: {llm_total_time:.1f}s, avg: {avg_time:.2f}s")
    
    unknown_count = (df_test['torch-ops'] == 'unknown').sum()
    print(f"  Extracted: {len(df_test) - unknown_count}, Unknown: {unknown_count}")
    
    try:
        df_issues = pd.read_excel(input_file, sheet_name='Issues')
        print(f"Found Issues sheet with {len(df_issues)} rows")
        if 'torch_ops' in df_issues.columns:
            df_issues = df_issues.drop(columns=['torch_ops'])
    except:
        print("No Issues sheet found")
    
    try:
        df_e2e = pd.read_excel(input_file, sheet_name='E2E Test Cases')
        print(f"Found E2E Test Cases sheet with {len(df_e2e)} rows")
    except:
        print("No E2E Test Cases sheet found")
    
    print(f"Writing to {output_file}...")
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        if df_issues is not None:
            df_issues.to_excel(writer, sheet_name='Issues', index=False)
        df_test.to_excel(writer, sheet_name='Test Cases', index=False)
        if df_e2e is not None:
            df_e2e.to_excel(writer, sheet_name='E2E Test Cases', index=False)
    
    print(f"\nTop 15 torch-ops:")
    vc = df_test['torch-ops'].value_counts()
    for op, count in vc.head(15).items():
        print(f"  {op}: {count}")
    
    print(f"\nSaved to {output_file}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python extract_torch_ops.py <input_file> [output_file]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    process_excel(input_file, output_file)
