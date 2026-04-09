#!/usr/bin/env python3
"""
Test script to analyze test cases using internal Qwen3-32B API with timing.
This is a modified version of analyze_test_case_with_llm using direct API call.
"""
import os
import sys
import json
import time
import requests
from pathlib import Path

def analyze_test_case_with_llm_qwen(test_file, test_class, test_case, origin_test_file=None):
    """
    Use Qwen3-32B via internal API to check CUDA and XPU test case existence.
    Returns: dict with test existence info and measures elapsed time.
    """
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

IMPORTANT: The base test name is NOT just removing '_xpu' suffix. The base test is the actual test function in the test file that can be parameterized to generate the XPU test case.

IMPORTANT: In the explanation, you MUST explain WHY the XPU test does not exist if cuda_exists is "No" or xpu_exists is "No". The reasons can be:
1. SKIP DECORATORS: Test has decorators like @onlyCUDA, @skipCUDAIfNoHipdnn, @skipIfXpu, @requires_xccl that prevent it from running on XPU
2. PARAMETERIZATION: Test is generated from a parameterized base test (e.g., @dtypes, @parametrize_test)
3. REMOVED/RENAMED: Test was removed or renamed in newer PyTorch versions
4. NOT APPLICABLE: Test is specific to CUDA/ROCm hardware
5. OTHER: Other reasons

For each check, provide:
1. Whether CUDA test exists (Yes/No)
2. Whether XPU test exists (Yes/No/N/A)
3. Key decorators found
4. Base test name
5. CUDA test file path if found
6. XPU test file path if found
7. CUDA test name found
8. XPU test name found
9. Detailed explanation of why XPU test exists or does not exist

Return ONLY valid JSON format (no additional text):
{{
    "explanation": "detailed explanation"
    "cuda_exists": "Yes/No",
    "xpu_exists": "Yes/No/N/A",
    "cuda_decorators": ["decorator1", "decorator2"],
    "xpu_decorators": ["decorator1"],
    "base_test_name": "original_test_function_name",
    "cuda_test_file": "path/to/test_file.py",
    "xpu_test_file": "path/to/test_xpu.py",
    "cuda_test_name": "test_name_found",
    "xpu_test_name": "test_name_found",
}}
"""
    # Internal API endpoint
    api_url = "http://10.239.15.43/v1/chat/completions"
    
    # Build messages for API call
    messages = [
        {"role": "system", "content": "You are a PyTorch test analysis assistant. Return ONLY valid JSON."},
        {"role": "user", "content": prompt}
    ]
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ.get('PRIVATE_API_KEY', '')}"
    }
    
    payload = {
        "model": "Qwen3-32B",
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 4096
    }
    
    start_time = time.time()
    
    try:
        response = requests.post(
            api_url,
            headers=headers,
            json=payload,
            timeout=180
        )
        response.raise_for_status()
        result = response.json()
        
        elapsed = time.time() - start_time
        
        # Extract content from response
        if result.get('choices') and len(result['choices']) > 0:
            content = result['choices'][0].get('message', {}).get('content', '')
            # Try to parse JSON from content
            import re
            json_match = re.search(r'\{[^{}]*\}', content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                data['elapsed_time'] = elapsed
                return data
            # Try full content as JSON
            if content.strip().startswith('{'):
                data = json.loads(content)
                data['elapsed_time'] = elapsed
                return data
        
        return {
            'cuda_exists': 'Unknown',
            'xpu_exists': 'Unknown',
            'explanation': f'API response: {str(result)[:500]}',
            'elapsed_time': elapsed
        }
        
    except requests.exceptions.Timeout:
        elapsed = time.time() - start_time
        return {
            'cuda_exists': 'Timeout',
            'xpu_exists': 'Timeout',
            'explanation': 'Request timed out after 180s',
            'elapsed_time': elapsed
        }
    except Exception as e:
        elapsed = time.time() - start_time
        return {
            'cuda_exists': 'Error',
            'xpu_exists': 'Error',
            'explanation': f'API error: {str(e)}',
            'elapsed_time': elapsed
        }


if __name__ == "__main__":
    # Test cases from issue #2694
    test_cases = [
        # Test Case 1
        {
            'test_file': 'test.inductor.test_gpu_cpp_wrapper.TestGpuWrapper',
            'test_class': 'TestGpuWrapper',
            'test_case': 'test_randint_xpu_gpu_wrapper'
        },
        # Test Case 2
        {
            'test_file': 'test.inductor.test_gpu_cpp_wrapper.DynamicShapesGpuWrapperGpuTests',
            'test_class': 'DynamicShapesGpuWrapperGpuTests',
            'test_case': 'test_randint_xpu_dynamic_shapes_gpu_wrapper'
        }
    ]
    
    print("=" * 80)
    print("Running analyze_test_case_with_llm with Qwen3-32B (internal provider)")
    print("=" * 80)
    print()
    
    for i, tc in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"Test Case {i}: {tc['test_class']}::{tc['test_case']}")
        print(f"{'='*60}")
        
        result = analyze_test_case_with_llm_qwen(
            test_file=tc['test_file'],
            test_class=tc['test_class'],
            test_case=tc['test_case'],
            origin_test_file=None
        )
        
        print(f"\nElapsed time: {result.get('elapsed_time', 'N/A'):.2f} seconds")
        print(f"\nResult:")
        print(json.dumps(result, indent=2))
        print()
