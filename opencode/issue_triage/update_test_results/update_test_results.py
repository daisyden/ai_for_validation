#!/usr/bin/env python3
"""
Script to update torch_xpu_ops_issues.xlsx with test case results from:
1. torch-xpu-ops nightly CI (XML files in Inductor-XPU-UT-Data-*-op_ut-*)
2. Stock PyTorch XPU CI (test-reports from pytorch xpu CI artifacts)
3. Case existence analysis (checking if tests exist in pytorch/test and torch-xpu-ops/test/xpu)
4. Duplicated issue detection (by Test Class + Test Case or similar Traceback)
5. E2E test case status from torch-xpu-ops nightly
6. Generate markdown report

Usage:
    python update_test_results.py

Input:
    - /home/daisydeng/ai_for_validation/opencode/issue_triage/result/torch_xpu_ops_issues.xlsx
    - /home/daisydeng/issue_traige/ci_results/torch-xpu-ops/
    - /home/daisydeng/issue_traige/ci_results/stock/
    - ~/pytorch/test/
    - ~/pytorch/third_party/torch-xpu-ops/test/xpu/

Output:
    - Updated /home/daisydeng/ai_for_validation/opencode/issue_triage/result/torch_xpu_ops_issues.xlsx
    - /home/daisydeng/ai_for_validation/opencode/issue_triage/result/issue_report.md
"""

import openpyxl
from openpyxl.styles import PatternFill, Font
import xml.etree.ElementTree as ET
import os
import re
import glob
import time

ROOT_DIR = "/home/daisydeng"
RESULT_DIR = "/home/daisydeng/ai_for_validation/opencode/issue_triage/result"


def get_torch_xpu_ops_xml_files():
    """Get all XML files from torch-xpu-ops nightly artifacts"""
    base_dir = '/home/daisydeng/issue_traige/ci_results/torch-xpu-ops'
    
    ut_folders = []
    for d in os.listdir(base_dir):
        if d.startswith('Inductor-XPU-UT-Data-'):
            match = re.match(r'Inductor-XPU-UT-Data-([a-f0-9]+)-.*-(\d+)-1$', d)
            if match:
                ut_folders.append((d, match.group(1), match.group(2)))
    
    xml_files = {}
    for folder_name, commit, run_id in ut_folders:
        folder_path = os.path.join(base_dir, folder_name, folder_name)
        if not os.path.exists(folder_path):
            continue
        for f in os.listdir(folder_path):
            if f.endswith('.xml') and (f.startswith('op_ut_with_all') or f.startswith('op_ut_with_skip')):
                xml_path = os.path.join(folder_path, f)
                try:
                    tree = ET.parse(xml_path)
                    root = tree.getroot()
                    count = len(root.findall('.//testcase'))
                    if count > 0:
                        prefix = f.replace('.xml', '')
                        xml_files[prefix] = (xml_path, commit, run_id, count)
                except:
                    pass
    
    return xml_files


def get_stock_xml_files():
    """Get all XML files from stock PyTorch XPU CI"""
    stock_base = '/home/daisydeng/issue_traige/ci_results/stock'
    stock_xml_files = {}
    
    for zip_file in glob.glob(f'{stock_base}/test-reports-*.zip'):
        pytest_dir = os.path.join(zip_file, 'test-reports', 'python-pytest')
        if not os.path.exists(pytest_dir):
            continue
        for root, dirs, files in os.walk(pytest_dir):
            for f in files:
                if f.endswith('.xml'):
                    xml_path = os.path.join(root, f)
                    test_module = os.path.basename(root)
                    stock_xml_files[test_module] = xml_path
    
    return stock_xml_files


def convert_test_file_to_xml_prefix(test_file):
    """Convert test file to XML prefix for torch-xpu-ops"""
    if not test_file:
        return None, 'No test file'
    
    test_file = str(test_file)
    
    if '/' in test_file:
        test_file = test_file.replace('torch-xpu-ops/test/xpu/', '')
        test_file = test_file.replace('.py', '')
        return 'op_ut_with_all.' + test_file.replace('/', '.'), None
    
    parts = test_file.split('.')
    parts_len = len(parts)
    
    if parts_len == 2:
        module = parts[0]
        if '_xpu' in module:
            return 'op_ut_with_skip.' + module, None
        return 'op_ut_with_all.' + module, None
    
    if parts_len == 3:
        module = parts[0]
        test_module = parts[1]
        if module == 'test':
            return 'op_ut_with_all.test_' + test_module + '_xpu', None
        if module == 'inductor':
            return None, 'inductor not in torch-xpu-ops'
        return 'op_ut_with_all.' + module + '.' + test_module, None
    
    if parts_len == 4:
        module = parts[0]
        subdir = parts[1]
        test_module = parts[2]
        if module == 'test':
            if subdir == 'functorch':
                return f'op_ut_with_all.{subdir}.{test_module}_xpu', None
            return None, f'{subdir} not in torch-xpu-ops'
        if module == 'inductor':
            return None, 'inductor not in torch-xpu-ops'
    
    return None, 'unknown pattern'


def convert_to_stock_prefix(test_file):
    """Convert test file to stock test module name"""
    if not test_file:
        return None
    
    test_file = str(test_file)
    
    if '/' in test_file:
        test_file = test_file.replace('torch-xpu-ops/test/xpu/', '')
        test_file = test_file.replace('.py', '')
        return test_file.replace('/', '.')
    
    parts = test_file.split('.')
    parts_len = len(parts)
    
    if parts_len == 2:
        return parts[0]
    if parts_len == 3:
        module = parts[0]
        test_module = parts[1]
        if module == 'test':
            return test_module
        return f'{module}.{test_module}'
    if parts_len == 4:
        module = parts[0]
        subdir = parts[1]
        test_module = parts[2]
        if module == 'test':
            return f'{subdir}.{test_module}'
        if module == 'inductor':
            return f'{module}.{test_module}'
    
    return None


def find_best_xml_match(xml_prefix, xml_files):
    """Find XML file matching prefix"""
    if not xml_prefix:
        return None
    if xml_prefix in xml_files:
        return xml_files[xml_prefix]
    return None


def parse_failure_content(content):
    """Parse failure message to extract error_msg and traceback until error type."""
    error_msg = ""
    traceback = ""

    if not content:
        return error_msg, traceback

    lines = content.split('\n')

    # Find error types - look for exception raising patterns
    # Pattern 1: Error type at start of line (RuntimeError: message)
    # Pattern 2: raise ErrorType("message") within traceback
    error_patterns = [
        (r'^RuntimeError', 'RuntimeError'),
        (r'^AssertionError', 'AssertionError'),
        (r'^ValueError', 'ValueError'),
        (r'^TypeError', 'TypeError'),
        (r'^IndexError', 'IndexError'),
        (r'^KeyError', 'KeyError'),
        (r'^ImportError', 'ImportError'),
        (r'^NotImplementedError', 'NotImplementedError'),
        (r'^AttributeError', 'AttributeError'),
        (r'^InductorError', 'InductorError'),
    ]

    exception_raise_patterns = [
        r'\braise\s+(RuntimeError|AssertionError|ValueError|TypeError|IndexError|KeyError|ImportError|NotImplementedError|AttributeError|InductorError)\s*[\(\'"]',
    ]

    error_line_idx = -1
    error_type = None
    last_error_msg = ""

    for idx, line in enumerate(lines):
        stripped = line.strip()
        # Pattern 1: Error type at start of line
        for pattern, etype in error_patterns:
            if re.match(pattern, stripped):
                error_line_idx = idx
                error_type = etype
                # Clean up the line - remove quotes and extra info
                clean_line = re.sub(r'^' + etype + r'[:\s]*', '', stripped)
                error_msg = clean_line[:200]
                break
        if error_line_idx >= 0:
            break
        # Pattern 2: raise ErrorType("message") - look for the last exception raise
        for ep in exception_raise_patterns:
            if re.search(ep, stripped):
                error_line_idx = idx
                # Extract the message part after the error type
                match = re.search(r'raise\s+\w+\s*[\(\'"](.+?)[\'\"]?', stripped)
                if match:
                    last_error_msg = match.group(1).strip()[:200]

    # Check if we have Traceback header
    traceback = ""
    if 'Traceback (most recent call last):' in content:
        # Build traceback: from Traceback line until error line (inclusive)
        tb_lines = []
        end_idx = error_line_idx if error_line_idx >= 0 else len(lines)
        for idx, line in enumerate(lines):
            if 'Traceback (most recent call last):' in line:
                # Include all lines from here until error line (inclusive)
                for j in range(idx, end_idx + 1):
                    tb_lines.append(lines[j])
                break

        if tb_lines:
            traceback = '\n'.join(tb_lines)
        # If no traceback but we found exception raise, extract from that point
        elif last_error_msg:
            for idx, line in enumerate(lines):
                stripped = line.strip()
                for ep in exception_raise_patterns:
                    if re.search(ep, stripped):
                        traceback = '\n'.join(lines[idx:])
                        break
                if traceback:
                    break
    else:
        # No traceback, just take the error message
        traceback = ""
        # Use content as error message
        error_msg = content[:200]

    # If we found an exception raise but no clear error line, use the extracted message
    if last_error_msg and not error_msg:
        error_msg = last_error_msg

    # Ensure traceback includes the actual error line if we have an error message
    if error_msg and traceback and error_msg not in traceback:
        traceback += f"\n{error_msg}"

    return error_msg, traceback[:3000] if traceback else traceback


def get_test_result(xml_path, test_case):
    """Get test case result from XML file. Returns status, error_msg, and traceback."""
    if not xml_path:
        return None, None, None

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        for testcase in root.findall('.//testcase'):
            if testcase.get('name') == test_case:
                failure = testcase.find('failure')
                if failure is not None:
                    msg = failure.text or failure.get('message', '')
                    error_msg, traceback = parse_failure_content(msg)
                    return 'failed', error_msg, traceback
                skipped = testcase.find('skipped')
                if skipped is not None:
                    msg = skipped.text or skipped.get('message', '')
                    return 'skipped', msg[:500] if msg else 'skipped', None
                return 'passed', '', None

        return 'not found', 'Test case not found', None
    except Exception as e:
        return 'error', str(e), None


def analyze_test_case_with_llm(test_file, test_class, test_case, origin_test_file):
    """
    Use LLM skills to check CUDA and XPU test case existence.
    Calls check-cuda-test-existence and check-xpu-test-existence skills via opencode.
    
    Note: This requires opencode to be available and pytorch skills to be loaded.
    """
    import subprocess
    import json
    
    if not test_file or not test_case:
        return {
            'cuda_exists': 'No',
            'xpu_exists': 'No',
            'explanation': 'No test file or test case',
            'cuda_decorators': [],
            'xpu_decorators': []
        }
    
    # Try to find pytorch root
    pytorch_root = os.path.expanduser('~/pytorch')
    if not os.path.exists(pytorch_root):
        pytorch_root = os.path.expanduser('~/issue_traige/pytorch')
    
    prompt = f"""You are in the pytorch directory: {pytorch_root}

Use the check-cuda-test-existence skill to check if the CUDA test exists in the original PyTorch test file.
Use the check-xpu-test-existence skill to check if the XPU test exists in torch-xpu-ops repo.

Paths:
- PyTorch test files: {pytorch_root}/test/
- torch-xpu-ops test files: {pytorch_root}/third_party/torch-xpu-ops/test/xpu/

Test File: {test_file}
Origin Test File: {origin_test_file if origin_test_file else 'Not provided'}
Test Class: {test_class}
Test Case: {test_case}

IMPORTANT: The base test name is NOT just removing '_xpu' suffix. The base test is the actual test function in the test file that can be parameterized to generate the XPU test case. For example:
- test_compare_cpu_add_float16_xpu has base test test_compare_cpu in test_ops.py
- test_conv2d_hipdnn_backend_selection_xpu has base test test_conv2d_hipdnn_backend_selection

The base test is often a function that gets parameterized with decorators like @dtypes, @parametrize_test, etc. to generate multiple test cases with different parameters (dtypes, devices, arguments).

IMPORTANT: In the explanation, you MUST explain WHY the XPU test does not exist if cuda_exists is "No" or xpu_exists is "No". The reasons can be:
1. SKIP DECORATORS: Test has decorators like @onlyCUDA, @skipCUDAIfNoHipdnn, @skipIfXpu, @requires_xccl that prevent it from running on XPU
2. PARAMETERIZATION: Test is generated from a parameterized base test (e.g., @dtypes, @parametrize_test) - explain what parameters are used
3. REMOVED/RENAMED: Test was removed or renamed in newer PyTorch versions
4. NOT APPLICABLE: Test is specific to CUDA/ROCm hardware (e.g., hipdnn backend)
5. OTHER: Other reasons

For each check, provide:
1. Whether CUDA test exists (Yes/No)
2. Whether XPU test exists (Yes/N/A)
3. Key decorators found (e.g., @onlyCUDA, @onlyXPU, @skipCUDAIfNoHipdnn, @skipIfXpu, @dtypes, @parametrize_test, @requires_xccl)
4. Base test name (the actual function in the test file that this test case derives from)
5. CUDA test file path if found
6. XPU test file path if found
7. CUDA test name found (full or base name)
8. XPU test name found (full or base name)
9. Detailed explanation of why XPU test exists or does not exist, including:
   - Which decorators affect XPU compatibility
   - What parameterization generates this test case
   - Whether test was removed/renamed
   - Whether test is CUDA/ROCm specific

Return ONLY valid JSON format (no additional text):
{{
    "cuda_exists": "Yes/No",
    "xpu_exists": "Yes/No/N/A", 
    "cuda_decorators": ["decorator1", "decorator2"],
    "xpu_decorators": ["decorator1"],
    "base_test_name": "original_test_function_name",
    "cuda_test_file": "path/to/test_file.py",
    "xpu_test_file": "path/to/test_xpu.py", 
    "cuda_test_name": "test_name_found",
    "xpu_test_name": "test_name_found",
    "explanation": "detailed explanation including why XPU test does or does not exist"
}}
"""
    
    try:
        # Use 'opencode run' with a faster free model
        result = subprocess.run(
            ['opencode', 'run', '-m', 'opencode/gpt-5-nano', prompt],
            capture_output=True,
            text=True,
            timeout=180,
            cwd=pytorch_root
        )
        
        if result.returncode == 0:
            output = result.stdout.strip()
            # Try to extract JSON from output
            try:
                # Look for JSON object in output
                import re
                json_match = re.search(r'\{[^{}]*\}', output, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group())
                    return data
                # If full output is JSON
                if output.startswith('{'):
                    data = json.loads(output)
                    return data
            except (json.JSONDecodeError, AttributeError) as e:
                pass
            # If no valid JSON found, return explanation
            return {
                'cuda_exists': 'Unknown',
                'xpu_exists': 'Unknown',
                'cuda_decorators': [],
                'xpu_decorators': [],
                'explanation': f"Output: {output[:500]}"
            }
        else:
            return {
                'cuda_exists': 'Error',
                'xpu_exists': 'Error',
                'cuda_decorators': [],
                'xpu_decorators': [],
                'explanation': f"LLM error (rc={result.returncode}): {result.stderr[:300]}"
            }
    except subprocess.TimeoutExpired:
        return {
            'cuda_exists': 'Timeout',
            'xpu_exists': 'Timeout',
            'cuda_decorators': [],
            'xpu_decorators': [],
            'explanation': 'LLM analysis timed out after 240s'
        }
    except FileNotFoundError:
        return {
            'cuda_exists': 'Error',
            'xpu_exists': 'Error',
            'cuda_decorators': [],
            'xpu_decorators': [],
            'explanation': 'opencode command not found'
        }
    except Exception as e:
        return {
            'cuda_exists': 'Error',
            'xpu_exists': 'Error',
            'cuda_decorators': [],
            'xpu_decorators': [],
            'explanation': f"Failed to run LLM: {str(e)}"
        }


def analyze_test_case(test_file, test_case):
    """
    Analyze why test case exists or not.
    Uses basic file existence check - for detailed analysis use check-cuda-test-existence 
    and check-xpu-test-existence skills via opencode.
    """
    result = {
        'cuda_file': None, 'cuda_exists': 'No',
        'xpu_file': None, 'xpu_exists': 'No',
        'explanation': '', 'expected_name': test_case
    }
    
    if not test_file:
        result['explanation'] = 'No test file'
        return result
    
    test_file = str(test_file)
    
    if '/' in test_file:
        cuda_rel = test_file.replace('torch-xpu-ops/test/xpu/', '').replace('_xpu', '')
        xpu_rel = test_file.replace('torch-xpu-ops/test/xpu/', '')
    elif '.' in test_file:
        parts = test_file.split('.')
        parts_len = len(parts)
        
        if parts_len == 4:
            cuda_rel = f"{parts[1]}/{parts[2]}.py"
            xpu_rel = f"{parts[1]}/{parts[2]}_xpu.py"
            result['expected_name'] = parts[3]
        elif parts_len == 3:
            if parts[0] == 'test':
                cuda_rel = f"{parts[1]}.py"
                xpu_rel = f"{parts[1]}_xpu.py"
            elif parts[0] == 'inductor':
                cuda_rel = f"inductor/{parts[1]}.py"
                xpu_rel = None
        elif parts_len == 2:
            cuda_rel = f"{parts[0].replace('_xpu', '')}.py"
            xpu_rel = f"{parts[0]}.py"
        else:
            return result
    else:
        return result
    
    pytorch_test_dir = os.path.expanduser('~/pytorch/test')
    torch_xpu_ops_dir = os.path.expanduser('~/pytorch/third_party/torch-xpu-ops/test/xpu')
    
    xpu_path = os.path.join(torch_xpu_ops_dir, xpu_rel) if xpu_rel else None
    xpu_content = None
    uses_xpu_patch = False
    
    if xpu_path and os.path.exists(xpu_path):
        with open(xpu_path, 'r') as f:
            xpu_content = f.read()
        uses_xpu_patch = 'XPUPatchForImport' in xpu_content
    
    cuda_path = os.path.join(pytorch_test_dir, cuda_rel) if cuda_rel else None
    cuda_content = None
    
    if cuda_path and os.path.exists(cuda_path):
        with open(cuda_path, 'r') as f:
            cuda_content = f.read()
    
    base_name = test_case.replace('_xpu', '') if test_case else None
    cuda_name = test_case.replace('_xpu', '_cuda') if test_case else None
    
    if cuda_content:
        result['cuda_file'] = cuda_rel
        result['cuda_exists'] = 'Yes'
        
        if test_case and test_case in cuda_content:
            result['explanation'] += 'CUDA: Found. '
        elif base_name and base_name in cuda_content:
            result['explanation'] += f"CUDA: Found (base: {base_name}). "
        elif cuda_name and cuda_name in cuda_content:
            result['explanation'] += f"CUDA: Found (cuda: {cuda_name}). "
        else:
            result['explanation'] += 'CUDA: Test not found in pytorch/test. '
            result['cuda_exists'] = 'No'
    else:
        result['cuda_file'] = f'Not found: {cuda_rel}'
        result['explanation'] += 'CUDA file missing. '
    
    uses_xpu_patch = xpu_content and 'XPUPatchForImport' in xpu_content
    
    if uses_xpu_patch:
        if cuda_content:
            if test_case in cuda_content:
                result['explanation'] += 'XPU: Found (imported from CUDA). '
            elif base_name and base_name in cuda_content:
                result['explanation'] += 'XPU: Found (imported from CUDA, base name). '
            elif cuda_name and cuda_name in cuda_content:
                result['explanation'] += 'XPU: Found (imported from CUDA, cuda name). '
            else:
                result['explanation'] += 'XPU: Imported but test not found in CUDA. '
        else:
            result['explanation'] += 'XPU: Uses XPUPatchForImport but CUDA file missing. '
    elif xpu_path and os.path.exists(xpu_path):
        result['xpu_file'] = xpu_rel
        result['xpu_exists'] = 'Yes'
        
        if test_case in xpu_content:
            result['explanation'] += 'XPU: Found in XPU file. '
        elif base_name and base_name in xpu_content:
            result['explanation'] += 'XPU: Found without _xpu. '
        else:
            result['explanation'] += 'XPU: Test not in file. '
    elif xpu_rel is None:
        result['xpu_file'] = 'N/A'
        result['explanation'] += 'XPU: Inductor tests use stock CI. '
    else:
        result['xpu_file'] = f'Not found: {xpu_rel}'
        result['explanation'] += 'XPU file missing. '
    
    return result


def find_duplicated_issues(ws):
    """Find duplicated issues based on Test Class + Test Case or similar Traceback"""
    from collections import defaultdict
    
    # Build index: (Test Class, Test Case) -> [(row, issue_id)]
    class_case_index = defaultdict(list)
    for row in range(2, ws.max_row + 1):
        test_class = ws.cell(row, 6).value
        test_case = ws.cell(row, 7).value
        issue_id = ws.cell(row, 1).value
        if test_class and test_case:
            key = (test_class, test_case)
            class_case_index[key].append((row, issue_id))
    
    # Build index for traceback similarity (excluding AssertionError about tensor-likes)
    traceback_index = defaultdict(list)
    for row in range(2, ws.max_row + 1):
        traceback = ws.cell(row, 9).value or ''
        issue_id = ws.cell(row, 1).value
        
        # Skip if it's about "AssertionError: Tensor-likes are not close!"
        if 'AssertionError: Tensor-likes are not close!' in traceback:
            continue
        
        # Normalize traceback for comparison
        norm = traceback.strip()
        if norm:
            traceback_index[norm].append((row, issue_id))
    
    # Find duplicates: Test Class + Test Case
    class_case_duplicates = {}
    for key, rows in class_case_index.items():
        if len(rows) > 1:
            issue_ids = [rid for _, rid in rows]
            for row, rid in rows:
                other_issues = [i for i in issue_ids if i != rid]
                if other_issues:
                    class_case_duplicates[row] = other_issues
    
    # Find duplicates: similar Traceback
    traceback_duplicates = {}
    for key, rows in traceback_index.items():
        if len(rows) > 1:
            issue_ids = [rid for _, rid in rows]
            for row, rid in rows:
                other_issues = [i for i in issue_ids if i != rid]
                if other_issues:
                    if row in traceback_duplicates:
                        traceback_duplicates[row].extend(other_issues)
                    else:
                        traceback_duplicates[row] = other_issues
    
    # Merge duplicates
    merged_duplicates = {}
    for row in range(2, ws.max_row + 1):
        dup_set = set()
        if row in class_case_duplicates:
            dup_set.update(class_case_duplicates[row])
        if row in traceback_duplicates:
            dup_set.update(traceback_duplicates[row])
        if dup_set:
            merged_duplicates[row] = sorted(list(dup_set))
    
    return merged_duplicates


def determine_category(title, summary, test_cases_str, traceback, test_module, labels):
    """
    Determine the category of an issue based on its content.
    Categories:
    1. Dtype / Precision Related
    2. Sparse Operations Related
    3. Inductor / Compilation Related
    4. Flash Attention / Transformer Related
    5. PT2E
    6. Distributed
    7. TorchAO
    8. Others
    """
    text = f"{title} {summary} {test_cases_str} {traceback}".lower()
    labels_lower = str(labels).lower() if labels else ""
    
    # 1. Distributed - check first as distributed is a clear module
    distributed_keywords = [
        'distributed', 'device_mesh', 'ProcessGroup', 'FSDP', 'DDP', 'c10d',
        'tensor parallel', 'all_reduce', 'all_gather', 'reduce_scatter',
        'comm', 'rank', 'world_size', 'process group'
    ]
    if any(k in text for k in distributed_keywords):
        return "Distributed"
    
    # 2. TorchAO (quantization, optimizer, etc.)
    torchao_keywords = [
        'torchao', 'quantization', 'quantize', 'int8', 'int4', 'fp8',
        'optimizer', 'Adam', 'SGD', 'adamw', 'qat', 'lora', 'adapter'
    ]
    if any(k in text for k in torchao_keywords):
        return "TorchAO"
    
    # 3. PT2E (torch.export, ExportedProgram, fake tensors)
    pt2e_keywords = [
        'torch.export', 'export', 'exported', 'dynamo', 'fake_tensor',
        'graph_code', 'graph_submodule', 'capture', 'aot', 'aotautograd',
        'forward_from_graph', '_export', 'exported_program'
    ]
    if any(k in text for k in pt2e_keywords):
        return "PT2E"
    
    # 4. Flash Attention / Transformer Related
    flash_attention_keywords = [
        'flash', 'flash_attention', 'flashattention', 'sdpa', 'scaled_dot_product',
        'scaled_dot_product_attention', 'mem_eff', 'memory efficient',
        'transformer', 'attention', 'qwen', 'llama', 'bert', 'gpt',
        'mha', 'mqa', 'gqa', 'rope', 'rms_norm', 'layernorm',
        'linear', 'mlp', 'feed forward', 'feedforward'
    ]
    if any(k in text for k in flash_attention_keywords):
        return "Flash Attention / Transformer Related"
    
    # 5. Sparse Operations Related
    sparse_keywords = [
        'sparse', 'csr', 'csc', 'coo', 'sampled_addmm', 'sampled_addmm',
        'spmm', 'sparse_ops', 'sparse_matmul', 'torch.sparse'
    ]
    if any(k in text for k in sparse_keywords):
        return "Sparse Operations Related"
    
    # 6. Inductor / Compilation Related
    inductor_keywords = [
        'inductor', 'compile', 'compilation', 'codegen', 'triton',
        'kernel', 'loop', 'schedule', 'fx', 'graph', 'lower',
        'tile', 'vectorize', 'scheduler', 'abs_float'
    ]
    if any(k in text for k in inductor_keywords):
        return "Inductor / Compilation Related"
    
    # 7. Dtype / Precision Related
    dtype_precision_keywords = [
        'dtype', 'precision', 'accuracy', 'type promotion', 'typepromotion',
        'bf16', 'fp16', 'float16', 'float32', 'int8', 'int4', 'amp',
        'atomic', 'nan', 'inf', 'numerical', 'round', 'ceil', 'floor',
        'small', 'close', 'assertionerror'
    ]
    if any(k in text for k in dtype_precision_keywords):
        return "Dtype / Precision Related"
    
    # Default: Others
    return "Others"


def extract_version_info(issue_content):
    """
    Extract version info from issue content (title, summary, comments).
    Returns version string or None.
    """
    if not issue_content:
        return None

    version_patterns = [
        r'2\.\d+(\.\d+)?[a-z0-9]*',
        r'v2\.\d+(\.\d+)?',
        r'git[a-f0-9]+',
        r' nightly',
        r'a\d+',
    ]

    for pattern in version_patterns:
        match = re.search(pattern, issue_content, re.IGNORECASE)
        if match:
            return match.group(0)

    return None


def check_reproduce_step(issue_content):
    """
    Check if reproduce step is available in issue content.
    Returns True if reproduce step exists, False otherwise.
    """
    if not issue_content:
        return False

    content_lower = issue_content.lower()

    strong_indicators = [
        'pip install',
        'git clone',
        'git checkout',
        'python3',
        'benchmark',
        'reproducer',
        '```',
        'bash',
        'sh ',
        './',
        'conda',
        'pip3 install',
    ]

    medium_indicators = [
        'reproduce',
        'steps to reproduce',
        'how to reproduce',
        'minimal example',
        'test case',
        'run the',
        'execute',
        'command',
        'script',
        'code snippet',
    ]

    soft_indicators = [
        'git ',
        'python',
        'run ',
    ]

    strong_count = sum(1 for kw in strong_indicators if kw in content_lower)
    medium_count = sum(1 for kw in medium_indicators if kw in content_lower)
    soft_count = sum(1 for kw in soft_indicators if kw in content_lower)

    if strong_count >= 1:
        return True
    if strong_count >= 1 and medium_count >= 1:
        return True
    if medium_count >= 2:
        return True
    if medium_count >= 1 and soft_count >= 2:
        return True
    if soft_count >= 3:
        return True

    return False


def check_info_requested_to_reporter(issue_content):
    """
    Check if maintainer has requested more information from reporter.
    Returns True if info was requested from reporter.
    """
    if not issue_content:
        return False

    request_keywords = [
        'could you please provide',
        'please provide more',
        'can you provide additional',
        'need more information',
        'needs more info',
        'please add',
        'please share',
        'need the reproduce',
        'we need',
        'please attach',
        'please run',
        'please check',
        'please verify',
    ]

    content_lower = issue_content.lower()
    return any(kw in content_lower for kw in request_keywords)


def is_public_branch(version_str):
    """
    Check if version indicates a public branch (main, release) vs private branch/PR.
    Returns True if public branch, False if private.
    """
    if not version_str:
        return False

    version_lower = version_str.lower()

    if 'pr' in version_lower and 'http' in version_lower:
        return False

    if version_lower in ['main', 'master']:
        return True

    if re.match(r'^v?2\.\d+(\.\d+)?$', version_lower):
        return True

    if re.match(r'^\d+\.\d+\.\d+a\d+\+git[0-9a-f]+$', version_lower):
        return True

    if '+git' in version_lower and not 'pr' in version_lower:
        return True

    return False


def analyze_root_cause_llm(issue_id, issue_title, issue_summary, test_file, test_class, test_case, error_msg, traceback, test_module=None):
    """
    Use internal Qwen3-32B LLM to analyze root cause of an issue.
    Returns brief but specific root cause description.
    Logs to ~/ai_for_validation/opencode/issue_triage/result/root_cause.txt
    """
    import subprocess
    import json
    import time
    import requests
    import os
    
    ROOT_CAUSE_LOG = os.path.expanduser('~/ai_for_validation/opencode/issue_triage/result/root_cause.txt')
    LLM_ENDPOINT = "http://10.239.15.43/v1/chat/completions"
    LLM_API_KEY = os.environ.get("OPENCODE_API_KEY", "sk-XZrfiPGmZaGLZFPNUpy6ww")
    LLM_MODEL = "Qwen3-32B"
    
    if not issue_title and not issue_summary and not error_msg and not traceback:
        return ""
    
    # Clear/create log file for this session
    with open(ROOT_CAUSE_LOG, 'a') as log:
        log.write(f"\n{'='*80}\n")
        log.write(f"LLM Root Cause Analysis Session\n")
        log.write(f"{'='*80}\n")
    
    def log_result(issue_id, root_cause, elapsed):
        """Log to file and print"""
        msg = f"Issue {issue_id}: {root_cause} ({elapsed:.2f}s)"
        print(f"  {msg}")
        with open(ROOT_CAUSE_LOG, 'a') as log:
            log.write(msg + "\n")
    
    prompt = f"""You are analyzing PyTorch XPU issue root cause.

Issue ID: {issue_id}
Title: {issue_title}
Summary: {issue_summary}
Test: {test_class}.{test_case}
Error: {error_msg}
Traceback: {traceback[:500] if traceback else 'N/A'}

Classify into ONE category:
1. Memory/Shared Memory Issue
2. Dtype/Precision Issue
3. Inductor/Compilation Issue
4. DNNL/OneDNN Issue
5. Flash Attention/Specific Ops Issue
6. Distributed/Gloo Issue
7. Skip/No Test Exists
8. Backend/Device Issue
9. API/Template Mismatch
10. Feature Not Supported
11. Timeout/Performance Issue
12. Runtime Error
13. Assertion Failure
14. Type/Value Error
15. Others

Answer format: "CATEGORY - brief_reason" (e.g., "Dtype/Precision Issue - bf16 precision mismatch")

YOUR ANSWER (no JSON, no thinking tags, just the answer):"""

    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": "Output ONLY 'Category - reason'. No markdown. No JSON. No thinking tags."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 100,
        "temperature": 0.0
    }
    
    start = time.time()
    try:
        response = requests.post(LLM_ENDPOINT, headers=headers, json=data, timeout=180)
        elapsed = time.time() - start
        
        if response.status_code == 200:
            resp_data = response.json()
            content = resp_data.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            # Clean thinking tags if present
            content = content.replace("<think>", "").replace("</think>", "").strip()
            
            # Extract CATEGORY - reason pattern
            import re
            match = re.search(r'(Memory|Dtype|Precision|Inductor|Compilation|DNNL|OneDNN|Flash|Attention|Distributed|Gloo|Backend|Device|API|Template|Mismatch|Feature|Not|Supported|Timeout|Performance|Runtime|Error|Assertion|Failure|Type|Value|Others)\s*[-–]\s*[^\n]+', content, re.IGNORECASE)
            
            if match:
                root_cause = match.group(0).strip()
            else:
                # Fallback: clean content and take first meaningful line
                lines = [l.strip() for l in content.split("\n") if l.strip() and len(l.strip()) > 5]
                root_cause = lines[-1][:80] if lines else content.strip()[:80]
            
            log_result(issue_id, root_cause, elapsed)
            return root_cause
        else:
            elapsed = time.time() - start
            error_msg = f"API Error: {response.status_code}"
            log_result(issue_id, error_msg, elapsed)
            return ""
            
    except Exception as e:
        elapsed = time.time() - start
        error_msg = f"Exception: {str(e)[:50]}"
        log_result(issue_id, error_msg, elapsed)
        return ""
    
    return ""
    
    pytorch_root = os.path.expanduser('~/pytorch')
    if not os.path.exists(pytorch_root):
        pytorch_root = os.path.expanduser('~/issue_traige/pytorch')
    
    prompt = f"""You are analyzing PyTorch XPU issue root cause.

Issue ID: {issue_id}
Issue Title: {issue_title}
Issue Summary: {issue_summary}
Test File: {test_file}
Test Class: {test_class}
Test Case: {test_case}
Error Message: {error_msg}
Traceback: {traceback[:1000] if traceback else 'N/A'}
Test Module: {test_module}

Analyze the root cause based on the information above. Classify into ONE of these categories:
1. Memory/Shared Memory Issue - OOM, allocation failures
2. Dtype/Precision Issue - precision mismatch, dtype conversion, numerical inaccuracies
3. Inductor/Compilation Issue - graph breaks, symbolic shape, compilation errors
4. DNNL/OneDNN Issue - DNNL backend primitive failures
5. Flash Attention/Specific Ops Issue - flash attention kernel failures
6. Distributed/Gloo Issue - distributed training, NCCL/Gloo backend
7. Skip/No Test Exists - missing tests, decorators preventing execution
8. Backend/Device Issue - XPU device initialization, placement
9. API/Template Mismatch - API signature mismatch, missing parameters
10. Feature Not Supported - unimplemented features
11. Import/Dependency Issue - module import failures, missing deps
12. XGBoost/External Dependency - XGBoost or other external lib issues
13. Timeout/Performance Issue - timeouts, slow execution
14. Runtime Error - general runtime failures
15. Assertion Failure - assert checks failing
16. Type/Value Error - type/value mismatches
17. Others - miscellaneous/unknown

Provide your answer as JSON:
{{
    "root_cause": "Category Name - Brief Explanation",
    "reasoning": "Why this classification"
}}

Only output the JSON, nothing else.
"""
    
    try:
        result = subprocess.run(
            ['python3', '-c', f'''
import json
import sys
sys.path.insert(0, "/home/daisydeng/ai_for_validation/opencode/issue_triage")
try:
    from openai import OpenAI
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {{"role": "system", "content": "You are a PyTorch XPU issue analyst. Analyze root causes and classify into predefined categories. Output JSON only."}},
            {{"role": "user", "content": {json.dumps(prompt)}}}
        ],
        max_tokens=200,
        temperature=0
    )
    print(response.choices[0].message.content)
except Exception as e:
    print(f"Error: {{e}}")
'''],
            capture_output=True,
            text=True,
            timeout=120
        )
        
        output = result.stdout.strip()
        if output.startswith('Error:'):
            return ""
        
        # Parse JSON response
        import re
        json_match = re.search(r'\{[^{}]+\}', output, re.DOTALL)
        if json_match:
            try:
                response_data = json.loads(json_match.group())
                return response_data.get('root_cause', '')
            except:
                pass
        
        return ""
        
    except Exception as e:
        print(f"  LLM call failed for issue {issue_id}: {e}")
        return ""


def analyze_root_cause(issue_title, issue_summary, test_file, test_class, test_case, error_msg, traceback, test_module=None):
    """
    Analyze root cause of an issue based on available information.
    Returns brief but specific root cause description.
    """
    if not any([error_msg, traceback, issue_summary]):
        return ""
    
    text = ' '.join([
        str(issue_title or ''),
        str(issue_summary or ''),
        str(error_msg or ''),
        str(traceback or ''),
        str(test_file or ''),
        str(test_class or ''),
        str(test_case or '')
    ]).lower()
    
    if 'permanent kill' in text or 'xgboost' in text:
        return "XGBoost/External Dependency"
    
    if 'out of memory' in text or 'cannot allocate memory' in text or 'allocation fails' in text:
        return "Memory/Shared Memory Issue"
    
    if any(k in text for k in ['requires xccl', 'no xccl', 'xccl not found']):
        return "XCCL/Dependency Issue"
    
    if any(k in text for k in ['inductor', 'compile', 'graph break', 'symbolic', 'fx']) and 'test/inductor' in str(test_file or '').lower():
        return "Inductor / Compilation Issue"
    
    if any(k in text for k in ['dynamo']) and ('test/dynamo' in str(test_file or '').lower() or 'testinductor' in str(test_file or '').lower()):
        return "Inductor / Compilation Issue"
    
    if 'dnnl' in text or 'onednn' in text or 'mkldnn' in text:
        return "DNNL/OneDNN Issue"
    
    if any(k in text for k in ['flash attention', 'flash_attention', 'flashattn']) and 'attention' in text:
        return "Flash Attention / Specific Ops Issue"
    
    if 'distributed' in text or 'gloo' in text or ' nccl' in text or 'nccl' in text:
        if 'test/distributed' in str(test_file or '').lower():
            return "Distributed / Gloo Issue"
    
    if 'dtype' in text or 'precision' in text or 'accuracy' in text or 'typepromotion' in text:
        return "Dtype / Precision Issue"
    
    if 'float' in text and ('16' in text or '32' in text or 'bf' in text):
        return "Dtype / Precision Issue"
    
    if 'skip' in text or 'decorator' in text or ('test' in text and 'not found' in text):
        return "Skip / No Test Exists"
    
    if 'device' in text or 'xpu' in text and 'init' in text:
        return "Backend / Device Issue"
    
    if 'api' in text or 'template' in text or 'signature' in text:
        return "API / Template Mismatch"
    
    if 'not implemented' in text or 'not support' in text:
        return "Feature Not Supported"
    
    if 'import error' in text or 'no module' in text:
        return "Import / Dependency Issue"
    
    if 'assertionerror' in text or 'assert' in text:
        if 'dtype' in text or 'precision' in text or 'float' in text:
            return "Dtype / Precision Issue"
        return "Assertion Failure"
    
    if 'runtimeerror' in text:
        if 'memory' in text or 'allocate' in text:
            return "Memory/Shared Memory Issue"
        if 'dtype' in text or 'cast' in text or 'convert' in text:
            return "Dtype / Precision Issue"
        if 'inductor' in text or 'compile' in text:
            return "Inductor / Compilation Issue"
        return "Runtime Error"
    
    if 'typeerror' in text or 'valueerror' in text:
        if 'dtype' in text or 'cast' in text:
            return "Dtype / Precision Issue"
        return "Type / Value Error"
    
    if 'timeout' in text:
        return "Timeout / Performance Issue"
    
    if 'test/nn' in str(test_file or '').lower():
        if 'conv' in str(test_case or '').lower() or 'linear' in str(test_case or '').lower():
            return "DNNL / Specific Ops Issue"
    
    return "Others"


def process_issues_sheet(wb):
    """Process Issues sheet to add owner_transfer, action_TBD, and duplicated_issue columns"""
    ws_issues = wb['Issues']
    ws_test = wb['Test Cases']
    
    # Add new columns to Issues sheet (columns 19, 20, 21, 24)
    ws_issues.cell(1, 19, 'owner_transfer')
    ws_issues.cell(1, 20, 'action_TBD')
    ws_issues.cell(1, 21, 'duplicated_issue')
    ws_issues.cell(1, 22, 'priority')
    ws_issues.cell(1, 23, 'priority_reason')
    ws_issues.cell(1, 24, 'Category')
    ws_issues.cell(1, 25, 'Root Cause')
    
    MAX_LLM_ROOT_CAUSE = 500
    llm_root_cause_count = 0
    root_cause_cache = {}
    
    # Build index: Issue ID -> test case results
    issue_test_results = {}
    issue_prs = {}
    issue_reporters = {}
    
    for row in range(2, ws_test.max_row + 1):
        issue_id = ws_test.cell(row, 1).value
        xpu_status = ws_test.cell(row, 11).value
        stock_status = ws_test.cell(row, 16).value
        cuda_case_exist = ws_test.cell(row, 18).value
        dup_issue = ws_test.cell(row, 21).value
        
        if issue_id not in issue_test_results:
            issue_test_results[issue_id] = {
                'xpu_statuses': set(),
                'stock_statuses': set(),
                'cuda_case_not_exist': False,
                'duplicated_issues': set()
            }
        
        if xpu_status:
            issue_test_results[issue_id]['xpu_statuses'].add(xpu_status)
        if stock_status:
            issue_test_results[issue_id]['stock_statuses'].add(stock_status)
        if cuda_case_exist == 'No':
            issue_test_results[issue_id]['cuda_case_not_exist'] = True
        if dup_issue:
            for dup_id in str(dup_issue).split(','):
                issue_test_results[issue_id]['duplicated_issues'].add(dup_id.strip())
    
    # Collect E2E status for issues
    ws_e2e = wb['E2E Test Cases']
    for row in range(2, ws_e2e.max_row + 1):
        issue_id = ws_e2e.cell(row, 1).value
        e2e_status = ws_e2e.cell(row, 13).value
        
        if issue_id not in issue_test_results:
            issue_test_results[issue_id] = {
                'xpu_statuses': set(),
                'stock_statuses': set(),
                'cuda_case_not_exist': False,
                'duplicated_issues': set(),
                'e2e_statuses': set()
            }
        
        if 'e2e_statuses' not in issue_test_results[issue_id]:
            issue_test_results[issue_id]['e2e_statuses'] = set()
        
        if e2e_status:
            issue_test_results[issue_id]['e2e_statuses'].add(e2e_status)
    
    # Get PR info from Issues sheet
    for row in range(2, ws_issues.max_row + 1):
        issue_id = ws_issues.cell(row, 1).value
        pr = ws_issues.cell(row, 15).value
        reporter = ws_issues.cell(row, 5).value
        assignee = ws_issues.cell(row, 4).value
        
        issue_prs[issue_id] = pr
        issue_reporters[issue_id] = reporter
    
    # Process each issue
    for row in range(2, ws_issues.max_row + 1):
        issue_id = ws_issues.cell(row, 1).value
        pr = ws_issues.cell(row, 15).value
        reporter = ws_issues.cell(row, 5).value
        assignee = ws_issues.cell(row, 4).value
        labels = ws_issues.cell(row, 6).value
        title = str(ws_issues.cell(row, 2).value) if ws_issues.cell(row, 2).value else ''
        module = str(ws_issues.cell(row, 12).value) if ws_issues.cell(row, 12).value else ''
        summary = str(ws_issues.cell(row, 10).value) if ws_issues.cell(row, 10).value else ''
        test_module = str(ws_issues.cell(row, 13).value) if ws_issues.cell(row, 13).value else ''
        traceback = ''
        for tr in range(2, ws_test.max_row + 1):
            if ws_test.cell(tr, 1).value == issue_id:
                traceback = str(ws_test.cell(tr, 9).value) if ws_test.cell(tr, 9).value else ''
                break
        test_cases_str = ''
        for tr in range(2, ws_test.max_row + 1):
            if ws_test.cell(tr, 1).value == issue_id:
                tc = str(ws_test.cell(tr, 7).value) if ws_test.cell(tr, 7).value else ''
                test_cases_str += ' ' + tc
        
        test_info = issue_test_results.get(issue_id, {
            'xpu_statuses': set(),
            'stock_statuses': set(),
            'cuda_case_not_exist': False,
            'duplicated_issues': set()
        })
        
        xpu_statuses = test_info['xpu_statuses']
        stock_statuses = test_info['stock_statuses']
        cuda_case_not_exist = test_info['cuda_case_not_exist']
        duplicated_issues = test_info['duplicated_issues']
        
        # Default values
        owner_transfer = ''
        action_tbd = ''
        
        # Check for Not target or wont fix in labels
        labels_str = str(labels).lower() if labels else ''
        is_not_target = ('not target' in labels_str or 'wont' in labels_str or "won't" in labels_str)
        
        # Rule 5: Labels contain Not target or wont fix
        if is_not_target:
            owner_transfer = reporter
            action_tbd = 'add to skiplist'
        
        # Rule 1: All test cases passed in torch-xpu-ops nightly or stock CI
        # Note: Must have passed (not skipped), and not for 'random' labeled issues
        labels_str = str(labels).lower() if labels else ''
        is_random = 'random' in labels_str
        
        # Check UT (unit test) passed
        ut_passed = False
        if not is_random and xpu_statuses and stock_statuses:
            xpu_all_passed = (xpu_statuses == {'passed'})
            stock_all_passed = (stock_statuses == {'passed'})
            
            if xpu_all_passed or stock_all_passed:
                owner_transfer = reporter
                action_tbd = 'Close fixed issue'
                ut_passed = True
        
        # Rule 1b: E2E test cases all passed
        e2e_statuses = test_info.get('e2e_statuses', set())
        if not ut_passed and e2e_statuses:
            # Check if all E2E statuses are pass (not fail)
            e2e_all_passed = all(s == 'pass' for s in e2e_statuses)
            if e2e_all_passed:
                owner_transfer = reporter
                action_tbd = 'Close fixed issue'
        
        # Check PR status
        pr_status = ws_issues.cell(row, 17).value
        pr_closed = pr_status in ['closed', 'merged']
        
        # If no owner_transfer yet and PRs are closed
        if not owner_transfer and pr_closed:
            # Rule 2: No failed cases - verify the issue
            has_failed = ('failed' in xpu_statuses) or ('failed' in stock_statuses)
            
            if not has_failed:
                owner_transfer = reporter
                action_tbd = 'Verify the issue'
            else:
                # Rule 3: Has failed cases - revisit PR
                owner_transfer = assignee
                action_tbd = 'Revisit the PR as case failed'
        
        # Rule 6: For issues still with no action_TBD, analyze content to determine TBD action
        if not action_tbd:
            labels_str = str(labels).lower() if labels else ''
            title = str(ws_issues.cell(row, 2).value).lower() if ws_issues.cell(row, 2).value else ''
            summary = str(ws_issues.cell(row, 10).value).lower() if ws_issues.cell(row, 10).value else ''
            test_module = str(ws_issues.cell(row, 13).value).lower() if ws_issues.cell(row, 13).value else ''
            
            # Get test case names for this issue
            test_cases_str = ''
            for tr in range(2, ws_test.max_row + 1):
                if ws_test.cell(tr, 1).value == issue_id:
                    tc = str(ws_test.cell(tr, 7).value).lower() if ws_test.cell(tr, 7).value else ''
                    test_cases_str += ' ' + tc
            
            # Check for "Torch not compiled with CUDA enabled" - needs enabling test
            if 'torch not compiled with cuda enabled' in summary or 'torch not compiled with cuda enabled' in title:
                owner_transfer = 'daisyden'
                action_tbd = 'Enable test'
            
            else:
                # Check for various categories
                is_upstream = 'ut_upstream' in labels_str or 'inductor' in labels_str
                is_wontfix = 'wont' in labels_str or 'not target' in labels_str
                is_not_target_upstream = is_not_target and is_upstream
                
                if is_not_target_upstream:
                    action_tbd = 'Needs Upstream Skip PR (not_target + ut_upstream)'
                    owner_transfer = assignee
                elif is_wontfix:
                    action_tbd = 'Needs Skip PR (wontfix / not_target)'
                    owner_transfer = assignee
                elif is_upstream:
                    action_tbd = 'Needs PyTorch Repo Changes (upstream)'
                    owner_transfer = assignee
        
        # Determine priority based on issue content and test results
        priority = 'P2'
        priority_reason = ''
        
        title = str(ws_issues.cell(row, 2).value) if ws_issues.cell(row, 2).value else ''
        summary = str(ws_issues.cell(row, 10).value) if ws_issues.cell(row, 10).value else ''
        test_module = str(ws_issues.cell(row, 13).value) if ws_issues.cell(row, 13).value else ''
        labels_str = str(labels).lower() if labels else ''
        
        # Check if E2E model/application issue (not benchmark/unittest)
        is_model_issue = 'model' in title.lower() or 'model' in summary.lower() or 'application' in title.lower() or 'application' in summary.lower() or 'huggingface' in title.lower() or 'timm' in title.lower() or 'torchbench' in title.lower()
        is_e2e = test_module == 'e2e'
        is_ut = test_module == 'ut'
        
        # Check for regression (passed before, failed now)
        is_regression = 'regression' in labels_str or 'regression' in title.lower() or 'was pass' in summary.lower() or 'previously pass' in summary.lower() or 'before' in summary.lower() and 'now' in summary.lower()
        
        # Check for build crash
        is_build_crash = 'build' in test_module.lower() or 'build' in title.lower() or 'crash' in title.lower() or 'segmentation' in title.lower() or 'segfault' in title.lower() or 'signal' in summary.lower()
        
        # Count failed test cases for this issue
        failed_count = 0
        for tr in range(2, ws_test.max_row + 1):
            if ws_test.cell(tr, 1).value == issue_id:
                tc_status = ws_test.cell(tr, 11).value
                if tc_status in ['failed', 'error']:
                    failed_count += 1
        
        # Determine priority
        # P0: Build crash
        if is_build_crash:
            priority = 'P0'
            priority_reason = 'Build crash - critical blocking issue'
        # P0: Real model/application impact (not unittest/benchmark)
        elif is_model_issue and not ('test' in title.lower() and 'case' in title.lower()):
            priority = 'P0'
            priority_reason = 'Impacts real model/application'
        # P0: Regression
        elif is_regression:
            priority = 'P0'
            priority_reason = 'Regression - passed before but failed now'
        # P1: E2E accuracy or functionality issue
        elif is_e2e and ('accuracy' in title.lower() or 'accuracy' in summary.lower() or 'fail' in title.lower() or 'fail' in summary.lower()):
            priority = 'P1'
            priority_reason = 'E2E benchmark accuracy/functionality issue'
        # P2: E2E performance issue
        elif is_e2e and ('performance' in title.lower() or 'slow' in title.lower() or 'latency' in title.lower()):
            priority = 'P2'
            priority_reason = 'E2E benchmark performance issue'
        # P1: UT with more than 20 failed cases
        elif is_ut and failed_count > 20:
            priority = 'P1'
            priority_reason = f'UT with {failed_count} failed test cases'
        else:
            priority = 'P2'
            priority_reason = 'UT issue with few failures'
        
        # Note: Rule 4 (cuda_case_not_exist) is intentionally not setting owner_transfer
        # as it requires long time LLM analysis
        
        # Add duplicated issues to Issues sheet
        if duplicated_issues:
            ws_issues.cell(row, 21, ','.join(sorted(duplicated_issues)))
        
        # New Rule: Check case availability and information requirements
        # IMPORTANT: Only apply if no action_tbd has been set yet and issue is on public branch
        if not action_tbd:
            labels_str = str(labels).lower() if labels else ''
            title_raw = str(ws_issues.cell(row, 2).value) if ws_issues.cell(row, 2).value else ''
            summary_raw = str(ws_issues.cell(row, 10).value) if ws_issues.cell(row, 10).value else ''

            version_info = extract_version_info(title_raw + ' ' + summary_raw)
            is_public = is_public_branch(version_info)

            # Check test case and E2E status from both sheets
            has_xpu_status = bool(xpu_statuses)
            has_stock_status = bool(stock_statuses)
            has_e2e_status = bool(test_info.get('e2e_statuses', set()))

            all_statuses_empty = (
                not has_xpu_status and
                not has_stock_status and
                not has_e2e_status
            )

            # Rule A: On public branch with no test case availability info
            if is_public and all_statuses_empty:
                action_tbd = 'Check case availability'
                owner_transfer = reporter

            # Rule B: Check if reproduce step or other information is missing
            if not action_tbd:
                issue_content = title_raw + ' ' + summary_raw

                missing_info_type = None

                if not check_reproduce_step(issue_content):
                    missing_info_type = 'reproduce step'

                if not check_info_requested_to_reporter(issue_content):
                    if missing_info_type:
                        missing_info_type += ' and more information'
                    else:
                        missing_info_type = 'more information'

                if missing_info_type:
                    # Check if this is info that was already requested
                    has_already_requested = check_info_requested_to_reporter(issue_content)
                    if has_already_requested:
                        action_tbd = 'Awaiting response from reporter'
                    else:
                        action_tbd = f'Need {missing_info_type}'
                    owner_transfer = reporter

        # Set owner_transfer, action_TBD, priority
        if owner_transfer:
            ws_issues.cell(row, 19, owner_transfer)
        if action_tbd:
            ws_issues.cell(row, 20, action_tbd)
        ws_issues.cell(row, 22, priority)
        ws_issues.cell(row, 23, priority_reason)
        
        # Determine and set Category
        category = determine_category(title, summary, test_cases_str, traceback, test_module, labels)
        ws_issues.cell(row, 24, category)
        
        # Analyze root cause only if action_TBD is blank
        current_action_tbd = ws_issues.cell(row, 20).value
        if not current_action_tbd:
            # Get test case info for this issue
            issue_test_file = ''
            issue_test_class = ''
            issue_test_case = ''
            issue_error_msg = ''
            issue_traceback = ''
            
            for tr in range(2, ws_test.max_row + 1):
                if ws_test.cell(tr, 1).value == issue_id:
                    if not issue_test_file:
                        issue_test_file = str(ws_test.cell(tr, 4).value) if ws_test.cell(tr, 4).value else ''
                    if not issue_test_class:
                        issue_test_class = str(ws_test.cell(tr, 6).value) if ws_test.cell(tr, 6).value else ''
                    issue_test_case = str(ws_test.cell(tr, 7).value) if ws_test.cell(tr, 7).value else ''
                    if not issue_error_msg:
                        issue_error_msg = str(ws_test.cell(tr, 8).value) if ws_test.cell(tr, 8).value else ''
                    if not issue_traceback:
                        issue_traceback = str(ws_test.cell(tr, 9).value) if ws_test.cell(tr, 9).value else ''
                    if issue_error_msg and issue_traceback:
                        break
            
            issue_title_raw = str(ws_issues.cell(row, 2).value) if ws_issues.cell(row, 2).value else ''
            
            # Always get rule-based root cause first as fallback
            root_cause = analyze_root_cause(
                issue_title_raw,
                summary,
                issue_test_file,
                issue_test_class,
                issue_test_case,
                issue_error_msg,
                issue_traceback,
                test_module
            )

            # Try LLM for first MAX_LLM_ROOT_CAUSE issues (always call for better results)
            if llm_root_cause_count < MAX_LLM_ROOT_CAUSE:
                print(f"  [LLM ROOT CAUSE #{llm_root_cause_count+1}] Issue {issue_id}: Calling LLM for root cause analysis...")
                llm_start = time.time()
                llm_root_cause = analyze_root_cause_llm(
                    issue_id,
                    issue_title_raw,
                    summary,
                    issue_test_file,
                    issue_test_class,
                    issue_test_case,
                    issue_error_msg,
                    issue_traceback,
                    test_module
                )
                llm_elapsed = time.time() - llm_start
                if llm_root_cause:
                    root_cause = llm_root_cause
                    llm_root_cause_count += 1
                    print(f"  [LLM ROOT CAUSE #{llm_root_cause_count}] Issue {issue_id} -> '{llm_root_cause}' ({llm_elapsed:.2f}s)")
                else:
                    print(f"  [LLM ROOT CAUSE #{llm_root_cause_count+1}] Issue {issue_id}: LLM returned empty")
            
            if root_cause:
                ws_issues.cell(row, 25, root_cause)
    
    print(f"Processed {ws_issues.max_row - 1} issues")
    return wb


def process_e2e_cases(wb):
    """Process E2E Test Cases sheet to add accuracy status from torch-xpu-ops nightly"""
    ws_e2e = wb['E2E Test Cases']
    
    # Add new column for accuracy status
    ws_e2e.cell(1, 13, 'torch-xpu-ops nightly status - accuracy')
    
    import glob
    import os
    base_dir = '/home/daisydeng/issue_traige/ci_results/torch-xpu-ops'
    
    # Build mapping: (benchmark, dtype, amp, phase, model) -> status
    # Phase: inf (inference), tra (training)
    e2e_model_status = {}
    
    for report_path in glob.glob(f'{base_dir}/*E2E*/Inductor_E2E_Test_Report.xlsx'):
        try:
            # Extract benchmark from path
            dirname = os.path.basename(os.path.dirname(report_path))
            if 'huggingface' in dirname:
                benchmark = 'huggingface'
            elif 'timm' in dirname:
                benchmark = 'timm_models'
            elif 'torchbench' in dirname:
                benchmark = 'torchbench'
            else:
                continue
            
            report_wb = openpyxl.load_workbook(report_path)
            
            # Parse all accuracy sheets (sheets ending with _acc)
            for sheet_name in report_wb.sheetnames:
                if not sheet_name.endswith('_acc'):
                    continue
                
                # Parse sheet name: huggingface_float32_inf_acc -> benchmark=huggingface, dtype=float32, phase=inf
                # Also: huggingface_amp_bf16_inf_acc -> benchmark=huggingface, dtype=bf16, amp=True, phase=inf
                parts = sheet_name.replace(f'{benchmark}_', '').replace('_acc', '').split('_')
                
                dtype = 'float32'
                amp = False
                phase = ''
                
                if 'amp' in parts:
                    amp = True
                    idx = parts.index('amp')
                    if idx + 1 < len(parts):
                        dtype = parts[idx + 1]
                else:
                    if len(parts) >= 1:
                        dtype = parts[0]
                
                if 'inf' in parts:
                    phase = 'inf'
                elif 'tra' in parts:
                    phase = 'tra'
                
                # Parse model status
                ws = report_wb[sheet_name]
                for row in range(3, ws.max_row + 1):  # Skip header rows
                    model_name = ws.cell(row, 2).value  # Column B: name
                    status = ws.cell(row, 4).value  # Column D: target/accuracy
                    
                    if model_name and status:
                        key = (benchmark, dtype, amp, phase, model_name.lower())
                        e2e_model_status[key] = status
        except Exception as e:
            print(f"  Warning: Failed to parse {report_path}: {e}")
    
    print(f"  Found {len(e2e_model_status)} E2E model status entries")
    
    # Map E2E test case row to status
    matched = 0
    for row in range(2, ws_e2e.max_row + 1):
        benchmark = ws_e2e.cell(row, 3).value  # Benchmark column
        model = ws_e2e.cell(row, 4).value  # Model column
        dtype = ws_e2e.cell(row, 6).value  # Dtype column
        amp = ws_e2e.cell(row, 7).value  # AMP column
        phase = ws_e2e.cell(row, 5).value  # Phase column
        
        if benchmark and model and dtype and phase:
            # Normalize
            dtype_key = dtype.lower().replace(' ', '_')
            phase_key = phase.lower().replace(' ', '_').replace('inference', 'inf').replace('training', 'tra')
            model_key = model.lower()
            
            # Try exact match first
            key = (benchmark, dtype_key, bool(amp), phase_key, model_key)
            if key in e2e_model_status:
                ws_e2e.cell(row, 13, e2e_model_status[key])
                matched += 1
            else:
                # Try without amp
                key = (benchmark, dtype_key, False, phase_key, model_key)
                if key in e2e_model_status:
                    ws_e2e.cell(row, 13, e2e_model_status[key])
                    matched += 1
    
    print(f"  Matched {matched} E2E cases with status")
    return wb


def main():
    # Load workbook
    wb = openpyxl.load_workbook(os.path.join(RESULT_DIR, 'torch_xpu_ops_issues.xlsx'))
    ws = wb['Test Cases']
    
    # Add new columns
    ws.cell(1, 11, 'status in torch-xpu-ops nightly')
    ws.cell(1, 12, 'comments in torch-xpu-ops nightly')
    ws.cell(1, 13, 'Commit')
    ws.cell(1, 14, 'Run_id')
    ws.cell(1, 15, 'XML')
    ws.cell(1, 16, 'status in stock CI')
    ws.cell(1, 17, 'comments in stock CI')
    ws.cell(1, 18, 'cuda_case_exist')
    ws.cell(1, 19, 'xpu_case_exist')
    ws.cell(1, 20, 'case_existence_comments')
    ws.cell(1, 21, 'duplicated_issue')
    
    # Get XML files
    print("Loading XML files...")
    xpu_xml_files = get_torch_xpu_ops_xml_files()
    print(f"  Found {len(xpu_xml_files)} torch-xpu-ops XML files")
    
    stock_xml_files = get_stock_xml_files()
    print(f"  Found {len(stock_xml_files)} stock XML files")
    
    # Process all rows
    total = ws.max_row - 1
    MAX_LLM_CASES = 3  # Process up to 3 unique issues with LLM
    llm_processed = 0
    llm_cache = {}  # Cache LLM results by issue id
    
    for i, row in enumerate(range(2, ws.max_row + 1), 1):
        test_file = ws.cell(row, 4).value
        test_class = ws.cell(row, 6).value
        test_case = ws.cell(row, 7).value
        issue_id = ws.cell(row, 1).value
        origin_test_file = ws.cell(row, 5).value
        
        # Get torch-xpu-ops nightly result from XML
        xml_prefix, reason = convert_test_file_to_xml_prefix(test_file)
        if xml_prefix:
            matched = find_best_xml_match(xml_prefix, xpu_xml_files)
            if matched:
                xml_path, commit, run_id, _ = matched
                status, error_msg, traceback = get_test_result(xml_path, test_case)
                ws.cell(row, 11, status)
                ws.cell(row, 12, error_msg if error_msg else '')
                if traceback and not ws.cell(row, 9).value:
                    ws.cell(row, 9, traceback[:3000])
                ws.cell(row, 13, commit)
                ws.cell(row, 14, run_id)
                ws.cell(row, 15, os.path.basename(xml_path))
            else:
                ws.cell(row, 11, 'not found')
                ws.cell(row, 12, f'No XML: {xml_prefix}')
        else:
            ws.cell(row, 11, 'not found')
            ws.cell(row, 12, reason)

        # Get stock CI result from XML
        stock_prefix = convert_to_stock_prefix(test_file)
        if stock_prefix and stock_prefix in stock_xml_files:
            stock_xml = stock_xml_files[stock_prefix]
            stock_status, stock_error_msg, stock_traceback = get_test_result(stock_xml, test_case)
            ws.cell(row, 16, stock_status)
            ws.cell(row, 17, stock_error_msg if stock_error_msg else '')
            if stock_traceback and not ws.cell(row, 9).value:
                ws.cell(row, 9, stock_traceback[:3000])
        else:
            ws.cell(row, 16, 'not found')
            ws.cell(row, 17, 'Not in stock CI')
        
        # Get case existence info - use LLM only for first N issues with "not found" in CI status
        
        # Initialize variables
        cuda_exists = 'No'
        xpu_exists = 'No'
        comment = ''
        
        # Check if CI status indicates not found
        xpu_status = ws.cell(row, 11).value
        stock_status = ws.cell(row, 16).value
        
        # Check if either CI result was not found or is empty (no test results)
        ci_not_found = (xpu_status == 'not found' or not xpu_status) and (stock_status == 'not found' or not stock_status)
        
        # Only run LLM for the first "not found" case of each issue
        if ci_not_found:
            if issue_id in llm_cache:
                # Use cached result
                cached = llm_cache[issue_id]
                cuda_exists = cached.get('cuda_exists', 'Unknown')
                xpu_exists = cached.get('xpu_exists', 'Unknown')
                comment = cached.get('comment', 'Cached result')
            elif llm_processed < MAX_LLM_CASES:
                try:
                    llm_result = analyze_test_case_with_llm(test_file, test_class, test_case, origin_test_file)
                    cuda_exists = llm_result.get('cuda_exists', 'Unknown')
                    xpu_exists = llm_result.get('xpu_exists', 'Unknown')
                    
                    llm_comment = llm_result.get('explanation', '')
                    cuda_decorators = ', '.join(llm_result.get('cuda_decorators', []))
                    xpu_decorators = ', '.join(llm_result.get('xpu_decorators', []))
                    
                    # Build comprehensive comment
                    base_test = llm_result.get('base_test_name', '')
                    cuda_test_file = llm_result.get('cuda_test_file', '')
                    xpu_test_file = llm_result.get('xpu_test_file', '')
                    cuda_test_name = llm_result.get('cuda_test_name', '')
                    xpu_test_name = llm_result.get('xpu_test_name', '')
                    
                    comment_parts = []
                    if base_test:
                        comment_parts.append(f"Base test: {base_test}")
                    if cuda_test_file:
                        comment_parts.append(f"CUDA file: {cuda_test_file}")
                    if xpu_test_file:
                        comment_parts.append(f"XPU file: {xpu_test_file}")
                    if cuda_test_name:
                        comment_parts.append(f"CUDA test: {cuda_test_name}")
                    if xpu_test_name:
                        comment_parts.append(f"XPU test: {xpu_test_name}")
                    if llm_comment:
                        comment_parts.append(llm_comment)
                    if cuda_decorators:
                        comment_parts.append(f"CUDA decorators: {cuda_decorators}")
                    if xpu_decorators:
                        comment_parts.append(f"XPU decorators: {xpu_decorators}")
                    
                    comment = ' | '.join(comment_parts) if comment_parts else 'Double not found - LLM analysis'
                    
                    # Cache the result
                    llm_cache[issue_id] = {
                        'cuda_exists': cuda_exists,
                        'xpu_exists': xpu_exists,
                        'comment': comment
                    }
                    llm_processed += 1
                except Exception as e:
                    comment = f"LLM error: {str(e)[:100]}"
                    cuda_exists = 'Error'
                    xpu_exists = 'Error'
                    llm_cache[issue_id] = {
                        'cuda_exists': cuda_exists,
                        'xpu_exists': xpu_exists,
                        'comment': comment
                    }
            else:
                # Max LLM cases reached, leave blank
                cuda_exists = ''
                xpu_exists = ''
                comment = ''
        else:
            # Use basic file analysis for other cases
            result = analyze_test_case(test_file, test_case)
            cuda_exists = result['cuda_exists']
            xpu_exists = result['xpu_exists']
            comment = f"CUDA file: {result.get('cuda_file', 'N/A')}. XPU file: {result.get('xpu_file', 'N/A')}. {result['explanation']}"
        
        ws.cell(row, 18, cuda_exists)
        ws.cell(row, 19, xpu_exists)
        ws.cell(row, 20, comment[:500])
        
        # Mark LLM analyzed cases with blue background
        if 'Base test:' in comment or 'CUDA decorators:' in comment or 'XPU decorators:' in comment:
            blue_fill = PatternFill(start_color='ADD8E6', end_color='ADD8E6', fill_type='solid')
            ws.cell(row, 18).fill = blue_fill
            ws.cell(row, 19).fill = blue_fill
            ws.cell(row, 20).fill = blue_fill
        
        if i % 100 == 0:
            print(f"Processed {i}/{total} (LLM: {llm_processed}/{MAX_LLM_CASES})")
        
        if i % 200 == 0:
            print(f"Processed {i}/{total}")
    
    print(f"Processed {total}/{total}")
    
    # Find and populate duplicated issues
    print("Finding duplicated issues...")
    duplicated = find_duplicated_issues(ws)
    for row, issue_ids in duplicated.items():
        ws.cell(row, 21, ','.join(str(i) for i in issue_ids))
    print(f"  Found {len(duplicated)} rows with duplicates")
    
    # Process E2E Test Cases sheet first (so Issues sheet can use E2E status)
    process_e2e_cases(wb)
    
    # Process Issues sheet (uses E2E status)
    process_issues_sheet(wb)
    
    # Save workbook
    wb.save(os.path.join(RESULT_DIR, 'torch_xpu_ops_issues.xlsx'))
    print("Saved!")
    
    # Generate markdown report
    from generate_report import generate_report
    generate_report()


if __name__ == '__main__':
    main()