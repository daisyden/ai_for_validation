#!/usr/bin/env python3
"""
Torch-ops Extraction Script

Extracts torch ops from PyTorch XPU test case data in Excel format.
"""

import pandas as pd
import re
import sys


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
    
    # torch_ops_aten__xxx -> aten._xxx
    match = re.search(r'torch_ops_aten__(\w+)', test_name)
    if match:
        return [f'aten._{clean_op(match.group(1))}']
    
    # __refs_xxx -> aten._xxx
    match = re.search(r'__refs_(\w+)', test_name)
    if match:
        return [f'aten._{clean_op(match.group(1))}']
    
    # _nn_xxx -> nn.xxx
    match = re.search(r'_nn_(\w+)', test_name)
    if match:
        return [f'nn.{clean_op(match.group(1))}']
    
    # _refs_xxx -> _refs_xxx
    match = re.search(r'_refs_(\w+)', test_name)
    if match:
        return [f'_{clean_op(match.group(1))}']
    
    # aten__xxx -> aten.xxx
    match = re.search(r'aten__(\w+)', test_name)
    if match:
        return [f'aten.{clean_op(match.group(1))}']
    
    # Attention patterns
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
    
    # vjp patterns
    match = re.search(r'vjp_linalg_(\w+)', test_name)
    if match:
        return [f'torch.linalg.{clean_op(match.group(1))}']
    
    # Sparse patterns
    if 'csr_matvec' in test_name:
        return ['aten.csr_matvec']
    if 'sparse_csr' in test_name or 'SparseCSR' in test_name:
        return ['aten.sparse_csr']
    if 'to_sparse' in test_name:
        return ['aten.to_sparse']
    if 'sparse' in test_name.lower():
        return ['sparse_ops']
    
    # Decomp patterns
    if 'rms_norm_decomp' in test_name:
        return ['aten.rms_norm']
    if '_fft_' in test_name or 'fft_' in test_name:
        return ['torch.fft']
    
    # has_decomposition
    if 'has_decomposition' in test_name:
        return ['decomp_ops']
    
    return []


def extract_from_error_or_traceback(text):
    """Extract torch ops from error message or traceback"""
    if pd.isna(text) or text == 'nan':
        return []
    text = str(text)
    found = []
    
    # torch.ops.aten.XXX.default
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


def extract_from_test_case(test_case):
    """Extract torch ops from test case name mapping"""
    if pd.isna(test_case) or test_case == 'nan':
        return []
    test_case = str(test_case)
    found = []
    
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
            found.append(op)
            break
    
    return found


def extract_torch_ops(row):
    """Extract torch ops using all rules in priority order"""
    test_case = str(row['Test Case']) if pd.notna(row['Test Case']) else ''
    error_msg = str(row['Error Message']) if pd.notna(row['Error Message']) else ''
    traceback = str(row['Traceback']) if pd.notna(row['Traceback']) else ''
    
    found_ops = []
    
    # Rule 1: Error message with torch.ops.aten.XXX.default pattern (HIGHEST)
    if error_msg:
        extracted = extract_from_error_or_traceback(error_msg)
        if extracted and any('default' in e for e in extracted):
            return extracted
    
    # Rule 2: Test name OpDB patterns
    if test_case:
        extracted = extract_ops_from_test_name(test_case)
        if extracted:
            return extracted
    
    # Rule 3: Test case name mapping
    if test_case:
        extracted = extract_from_test_case(test_case)
        if extracted:
            return extracted
    
    # Rule 4: Error message patterns
    if error_msg:
        extracted = extract_from_error_or_traceback(error_msg)
        if extracted:
            return extracted
    
    # Rule 5: Traceback patterns
    if traceback:
        extracted = extract_from_error_or_traceback(traceback)
        if extracted:
            return extracted
    
    return list(set(found_ops)) if found_ops else ['unknown']


def process_excel(input_file, output_file=None):
    """Process Excel file and extract torch ops"""
    if output_file is None:
        output_file = input_file
    
    # Read Test Cases sheet
    print(f"Reading {input_file}...")
    try:
        df_test = pd.read_excel(input_file, sheet_name='Test Cases')
    except Exception as e:
        print(f"Error reading Test Cases sheet: {e}")
        return
    
    print(f"Processing {len(df_test)} test cases...")
    
    # Process each row
    results = []
    for i in range(len(df_test)):
        row = df_test.iloc[i]
        extracted = extract_torch_ops(row)
        results.append(', '.join(extracted))
    
    df_test['torch-ops'] = results
    
    # Try to read Issues sheet
    df_issues = None
    try:
        df_issues = pd.read_excel(input_file, sheet_name='Issues')
        print(f"Found Issues sheet with {len(df_issues)} rows")
        # Remove torch_ops column if exists
        if 'torch_ops' in df_issues.columns:
            df_issues = df_issues.drop(columns=['torch_ops'])
    except:
        print("No Issues sheet found")
    
    # Save to Excel
    print(f"Writing to {output_file}...")
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        if df_issues is not None:
            df_issues.to_excel(writer, sheet_name='Issues', index=False)
        df_test.to_excel(writer, sheet_name='Test Cases', index=False)
    
    # Print summary
    unknown_count = (df_test['torch-ops'] == 'unknown').sum()
    print(f"\nSummary:")
    print(f"  Total: {len(df_test)}")
    print(f"  Unknown: {unknown_count}")
    print(f"  Known: {len(df_test) - unknown_count}")
    
    print("\nTop 15 torch-ops:")
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
