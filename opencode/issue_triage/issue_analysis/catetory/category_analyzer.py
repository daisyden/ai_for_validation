#!/usr/bin/env python3
"""
Category analysis module for PyTorch XPU issue triaging.

Contains LLM-based and rule-based category determination logic.

Usage:
    from issue_analysis.catetory.category_analyzer import (
        determine_category,
        determine_category_llm
    )

    # Rule-based category
    category = determine_category(title, summary, test_cases_str, traceback, test_module, labels)

    # LLM-based category + reason
    category, reason = determine_category_llm(
        title="Issue title",
        summary="Issue summary",
        test_cases_info=[{"test_case": "test_xxx", "error_msg": "..."}],
        test_module="ut",
        labels="bug"
    )
"""

import os
import re
import time

LLM_ENDPOINT = "http://10.239.15.43/v1/chat/completions"
LLM_API_KEY = os.environ.get("OPENCODE_API_KEY", "sk-xxxxxxxxxx")
LLM_MODEL = "Qwen3-32B"


def determine_category_llm(title, summary, test_cases_info, test_module, labels):
    """
    Use Qwen3-32B via internal API to determine the category of an issue.

    Categories: Distributed, TorchAO, PT2E, Flash Attention/Transformer, Sparse,
    Inductor/Compilation, Dtype/Precision, Backend/Device, Others

    Args:
        title: Issue title
        summary: Issue summary
        test_cases_info: List of dicts with 'test_case', 'error_msg'
        test_module: Module type ('ut', 'e2e', 'build')
        labels: Labels string

    Returns:
        tuple: (category, reason)
            - category: Category string like "3 - Inductor/Compilation"
            - reason: Detailed reason for classification
    """
    import requests
    import json

    if not title and not summary:
        return "12 - Others", ''

    tc_info_str = ""
    if test_cases_info:
        for tc in test_cases_info:
            error_val = tc.get('error_msg') or ''
            tc_info_str += f"- Test: {tc.get('test_case', '')}, Error: {str(error_val)[:100]}\n"

    test_module_str = test_module or "Unknown"
    labels_str = labels or "None"

    prompt = f"""You are analyzing PyTorch XPU issue to determine its category.

Issue Title: {title}
Issue Summary: {summary}
Test Module: {test_module_str}
Labels: {labels_str}

Test Cases Info:
{tc_info_str}

Categorize this issue into ONE of:

Categories:

    1. Distributed - Keywords: distributed, XCCL, NCCL, Gloo, ProcessGroup, DDP, FSDP, torch.distributed, collective communication, multi-node, timeout, rendezvous, init_method, reduce_scatter, all_gather

    2. TorchAO - Keywords: torchao, quantize_, int4_weight_only, int8_dynamic_activation, fp8, nf4, autoquant, quantization_config, Adam8bit, AdamW4bit, Lion8bit, PagedAdam, OptimizerWithQuantization, quantized_optimizer, ao/sparsity, apply_dynamic_quant, change_linear_weights_to_int8_packed, QuantizedLinear, from_float (AO context), packed weight, dequantization, int4, int8 (when paired with torchao)

    3. PT2E - Keywords: torch.export(), Dynamo, fake_tensor, ExportedProgram, AOT, torch._export, graph break, tracing, exported_program.run_decompositions

    4. Flash Attention/Transformer - Keywords: flash_attention, scaled_dot_product_attention, SDPA, attention mask, transformer layer, F.scaled_dot_product_attention, memory-efficient attention, FlexAttention

    5. Sparse - Keywords: sparse tensor, CSR, CSC, COO, torch.sparse, sparse matrix multiplication, sparse_mask, to_sparse(), sparse_coo_tensor

    6. Inductor/Compilation - Keywords: torch.compile(), Inductor, Triton, codegen, AOT Autograd, FX graph, torch._inductor, compilation cache, torch._dynamo

    7. Torch Runtime - Keywords: CUDA runtime, cudaMalloc, cudaMemcpy, out of memory (OOM), device kernel launch, stream synchronization, cudaStreamSynchronize, device-side assert, cudaError, illegal memory access, context management, cudaSetDevice, device initialization, cudaGetDevice, driver error, CUDA_VISIBLE_DEVICES, memory leak, allocation failure, device reset

    8. Torch Operations - Keywords: operator implementation, aten::, native::, custom op, register_operator, operator overloading, tensor operation dispatch, kernel selection, op signature mismatch, unsupported op on device, op not implemented for device, device-specific op behavior, backward pass operation, autograd op

    9. Dtype/Precision - Keywords: dtype mismatch, float16, bfloat16, float32, mixed precision, autocast, GradScaler, precision loss, NaN/inf due to dtype, to(dtype=...), torch.int8 (without torchao), legacy torch.quantization

    10. Feature Not Supported - Keywords: unimplemented operator, missing kernel, feature not available in this build, unsupported combination, "not implemented for"

    11. Skip/No Test Exists - Keywords: test skipped, @unittest.skip, missing test decorator, CI test gap, skipIfNoTorchAO

    12. Others - None of the above (only if truly uncategorizable)

Classification Rules:
    - Select exactly ONE category
    - For int4/int8/fp8 errors: TorchAO takes precedence over Dtype/Precision
    - For quantized optimizers (Adam8bit, Lion8bit): TorchAO takes precedence
    - For CUDA runtime errors (memory, sync, context): Choose Torch Runtime
    - For operator dispatch/kernel errors: Choose Torch Operations
    - For device-specific op implementation issues: Choose Torch Operations
    - Use Others only as a last resort

    Distinction between Torch Runtime and Torch Operations:
        - Torch Runtime = Errors related to device management, memory allocation, synchronization, driver/runtime API calls
        - Torch Operations = Errors related to specific operator execution, kernel selection, op dispatch, custom op registration

Return the category AND a detailed reason for your classification:
- The reason is REQUIRED, not optional
- Make it detailed and specific (full sentences, 150-300 characters)
- Include: specific ops/functions mentioned, dtypes (float16, bf16, int8, Long, etc.), arguments, or patterns that led to this categorization
- Explain clearly WHY you chose this category based on the issue details

Format: "Category Name | detailed_reason"
Example: "Dtype/Precision Issue | The aten.memory_efficient_attention kernel encounters dtype mismatch when processing fp32 input tensors with bfloat16 scaling factor on XPU device. The dot_xpu_mkl operation is not implemented for Long dtype, causing NotImplementedError."

YOUR ANSWER (must include detailed reason after the pipe symbol):"""

    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": "Output ONLY 'X - Category Name | brief_reason'. No markdown. No JSON. No thinking tags."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0,
        "max_tokens": 400
    }

    start_time = time.time()

    try:
        response = requests.post(LLM_ENDPOINT, headers=headers, json=payload, timeout=180)
        elapsed = time.time() - start_time

        if response.status_code == 200:
            result = response.json()
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")

            content = re.sub(r'\[TO\]', '', content)
            content = re.sub(r'\[/TO\]', '', content)
            content = re.sub(r'\[RESULT\]', '', content)
            content = re.sub(r'\[/RESULT\]', '', content)
            content = re.sub(r'\[/TD\]', '', content)
            content = content.replace('[', ' <').replace(']', '> ')
            content = re.sub(r'<[^>]*>', '', content)
            content = re.sub(r'has ATTR\b', '', content)
            content = re.sub(r'\s+', ' ', content).strip()

            if '|' in content:
                parts = content.split('|', 1)
                category_part = parts[0].strip()
                reason = parts[1].strip() if len(parts) > 1 else ''
                return category_part, reason
            else:
                match = re.search(r'(\d+)\s*[-–]\s*[\w\s/]+', content)
                if match:
                    category = f"{match.group(1)} - " + content[match.start():match.end()].split('-')[-1].strip()
                    return category, ''
                else:
                    return content.strip()[:50], ''

        return f"API Error: {response.status_code}", ''

    except Exception as e:
        return f"Error: {str(e)[:30]}", ''


def determine_category(title, summary, test_cases_str, traceback, test_module, labels):
    """
    Rule-based category determination based on keyword matching.

    Categories:
        1. Distributed
        2. TorchAO
        3. PT2E
        4. Flash Attention / Transformer Related
        5. Sparse Operations Related
        6. Inductor / Compilation Related
        7. Dtype / Precision Related
        8. Others

    Args:
        title: Issue title
        summary: Issue summary
        test_cases_str: Test cases as string
        traceback: Error traceback
        test_module: Module type
        labels: Labels

    Returns:
        str: Category name
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
        return "1 - Distributed"

    # 2. TorchAO (quantization, optimizer, etc.)
    torchao_keywords = [
        'torchao', 'quantization', 'quantize', 'int8', 'int4', 'fp8',
        'optimizer', 'Adam', 'SGD', 'adamw', 'qat', 'lora', 'adapter'
    ]
    if any(k in text for k in torchao_keywords):
        return "2 - TorchAO"

    # 3. PT2E (torch.export, ExportedProgram, fake tensors)
    pt2e_keywords = [
        'torch.export', 'export', 'exported', 'dynamo', 'fake_tensor',
        'graph_code', 'graph_submodule', 'capture', 'aot', 'aotautograd',
        'forward_from_graph', '_export', 'exported_program'
    ]
    if any(k in text for k in pt2e_keywords):
        return "3 - PT2E"

    # 4. Flash Attention / Transformer Related
    flash_attention_keywords = [
        'flash', 'flash_attention', 'flashattention', 'sdpa', 'scaled_dot_product',
        'scaled_dot_product_attention', 'mem_eff', 'memory efficient',
        'transformer', 'attention', 'qwen', 'llama', 'bert', 'gpt',
        'mha', 'mqa', 'gqa', 'rope', 'rms_norm', 'layernorm',
        'linear', 'mlp', 'feed forward', 'feedforward'
    ]
    if any(k in text for k in flash_attention_keywords):
        return "4 - Flash Attention / Transformer Related"

    # 5. Sparse Operations Related
    sparse_keywords = [
        'sparse', 'csr', 'csc', 'coo', 'sampled_addmm', 'sampled_addmm',
        'spmm', 'sparse_ops', 'sparse_matmul', 'torch.sparse'
    ]
    if any(k in text for k in sparse_keywords):
        return "5 - Sparse Operations Related"

    # 6. Inductor / Compilation Related
    inductor_keywords = [
        'inductor', 'compile', 'compilation', 'codegen', 'triton',
        'kernel', 'loop', 'schedule', 'fx', 'graph', 'lower',
        'tile', 'vectorize', 'scheduler', 'abs_float'
    ]
    if any(k in text for k in inductor_keywords):
        return "6 - Inductor / Compilation Related"

    # 7. Dtype / Precision Related
    dtype_precision_keywords = [
        'dtype', 'precision', 'accuracy', 'type promotion', 'typepromotion',
        'bf16', 'fp16', 'float16', 'float32', 'int8', 'int4', 'amp',
        'atomic', 'nan', 'inf', 'numerical', 'round', 'ceil', 'floor',
        'small', 'close', 'assertionerror'
    ]
    if any(k in text for k in dtype_precision_keywords):
        return "7 - Dtype / Precision Related"

    # Default: Others
    return "8 - Others"


# Category constants
CATEGORY_DISTRIBUTED = "1 - Distributed"
CATEGORY_TORCHAO = "2 - TorchAO"
CATEGORY_PT2E = "3 - PT2E"
CATEGORY_FLASH_ATTENTION = "4 - Flash Attention / Transformer Related"
CATEGORY_SPARSE = "5 - Sparse Operations Related"
CATEGORY_INDUCTOR = "6 - Inductor / Compilation Related"
CATEGORY_DTYPE_PRECISION = "7 - Dtype / Precision Related"
CATEGORY_OTHERS = "8 - Others"

CATEGORY_NAMES = {
    CATEGORY_DISTRIBUTED: "Distributed",
    CATEGORY_TORCHAO: "TorchAO",
    CATEGORY_PT2E: "PT2E",
    CATEGORY_FLASH_ATTENTION: "Flash Attention/Transformer",
    CATEGORY_SPARSE: "Sparse Operations",
    CATEGORY_INDUCTOR: "Inductor/Compilation",
    CATEGORY_DTYPE_PRECISION: "Dtype/Precision",
    CATEGORY_OTHERS: "Others"
}