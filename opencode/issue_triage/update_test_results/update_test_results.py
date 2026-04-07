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


def get_test_result(xml_path, test_case):
    """Get test case result from XML file"""
    if not xml_path:
        return None, None
    
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        for testcase in root.findall('.//testcase'):
            if testcase.get('name') == test_case:
                failure = testcase.find('failure')
                if failure is not None:
                    msg = failure.text or failure.get('message', '')
                    return 'failed', msg[:500] if msg else 'failed'
                skipped = testcase.find('skipped')
                if skipped is not None:
                    msg = skipped.text or skipped.get('message', '')
                    return 'skipped', msg[:500] if msg else 'skipped'
                return 'passed', ''
        
        return 'not found', 'Test case not found'
    except Exception as e:
        return 'error', str(e)


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


def process_issues_sheet(wb):
    """Process Issues sheet to add owner_transfer, action_TBD, and duplicated_issue columns"""
    ws_issues = wb['Issues']
    ws_test = wb['Test Cases']
    
    # Add new columns to Issues sheet (columns 19, 20, 21)
    ws_issues.cell(1, 19, 'owner_transfer')
    ws_issues.cell(1, 20, 'action_TBD')
    ws_issues.cell(1, 21, 'duplicated_issue')
    
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
                is_flash_attention = 'flash' in title or 'flash' in summary or 'flash' in test_cases_str or 'sdpa' in title or 'sdpa' in summary or 'sdpa' in test_cases_str or 'transformer' in title or 'transformer' in summary or 'transformer' in test_cases_str or 'scaled_dot_product' in title or 'scaled_dot_product' in summary or 'scaled_dot_product' in test_cases_str or 'mem_eff' in title or 'mem_eff' in summary or 'mem_eff' in test_cases_str
                is_sparse = 'sparse' in title or 'sparse' in summary or 'sparse' in test_module or 'sparse' in test_cases_str or 'csr' in title or 'csr' in summary or 'csr' in test_cases_str or 'sampled_addmm' in title or 'sampled_addmm' in summary or 'sampled_addmm' in test_cases_str
                is_inductor = 'inductor' in title or 'inductor' in summary or 'inductor' in test_cases_str or 'compile' in title or 'compile' in summary or 'compile' in test_cases_str or 'aot' in title or 'aot' in summary or 'aot' in test_cases_str
                is_dtype = 'dtype' in title or 'dtype' in summary or 'dtype' in test_cases_str or 'precision' in title or 'precision' in summary or 'precision' in test_cases_str or 'accuracy' in title or 'accuracy' in summary or 'accuracy' in test_cases_str or 'type promotion' in title or 'type promotion' in summary
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
                elif is_flash_attention:
                    action_tbd = 'Flash Attention / Transformer Related'
                    owner_transfer = assignee
                elif is_sparse:
                    action_tbd = 'Sparse Operations Related'
                    owner_transfer = assignee
                elif is_inductor:
                    action_tbd = 'Inductor / Compilation Related'
                    owner_transfer = assignee
                elif is_dtype:
                    action_tbd = 'Dtype / Precision Related'
                    owner_transfer = assignee
        
        # Note: Rule 4 (cuda_case_not_exist) is intentionally not setting owner_transfer
        # as it requires long time LLM analysis
        
        # Add duplicated issues to Issues sheet
        if duplicated_issues:
            ws_issues.cell(row, 21, ','.join(sorted(duplicated_issues)))
        
        # Set owner_transfer and action_TBD
        if owner_transfer:
            ws_issues.cell(row, 19, owner_transfer)
        if action_tbd:
            ws_issues.cell(row, 20, action_tbd)
    
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
                status, comment_xpu = get_test_result(xml_path, test_case)
                ws.cell(row, 11, status)
                ws.cell(row, 12, comment_xpu)
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
            stock_status, stock_comment = get_test_result(stock_xml, test_case)
            ws.cell(row, 16, stock_status)
            ws.cell(row, 17, stock_comment)
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