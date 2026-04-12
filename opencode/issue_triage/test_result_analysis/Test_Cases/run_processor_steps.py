#!/usr/bin/env python3
"""
Run Test Cases Processor - Step by Step

Usage:
    python3 run_processor_steps.py              # Run all steps
    python3 run_processor_steps.py --step 1      # Run only step 1 (PASS 1)
    python3 run_processor_steps.py --steps 1 2  # Run steps 1 and 2
    python3 run_processor_steps.py --steps 1-3    # Run steps 1, 2, 3
    python3 run_processor_steps.py --list       # List available steps
    python3 run_processor_steps.py --help       # Show help

Steps:
    1. PASS 1: Match CI results from XML files (XPU nightly + stock)
    2. PASS 2: Torch-ops extraction (pattern + LLM fallback)
    3. PASS 3: LLM analysis for test existence (CUDA/XPU)
    4. PASS 4: Dependency RAG (match ops to deps)
    5. PASS 5: Duplicate detection (cross-issue)

Example:
    # Run steps 1 and 2 only (no LLM calls - fast)
    python3 run_processor_steps.py --steps 1 2

    # Run full processor
    python3 run_processor_steps.py

    # Run specific steps
    python3 run_processor_steps.py --steps 1 2 3 4 5
"""

import os
import sys
import time
import argparse
import openpyxl

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = '/home/daisydeng'
RESULT_DIR = os.environ.get('RESULT_DIR', '/home/daisydeng/ai_for_validation/opencode/issue_triage/result')

sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, '..'))

from test_cases_processor import (
    pass1_match_ci_results,
    pass2_extract_torch_ops,
    pass3_llm_analysis_for_test_existence,
    pass4_dependency_rag,
    pass5_duplicate_detection,
    get_torch_xpu_ops_xml_files,
    get_stock_xml_files,
    log
)

STEPS_DESC = {
    1: "PASS 1: Match CI results from XML files (XPU nightly + stock)",
    2: "PASS 2: Torch-ops extraction (pattern + LLM fallback)",
    3: "PASS 3: LLM analysis for test existence (CUDA/XPU)",
    4: "PASS 4: Dependency RAG (match ops to deps)",
    5: "PASS 5: Duplicate detection (cross-issue)"
}

STEP_FUNCS = {
    1: pass1_match_ci_results,
    2: pass2_extract_torch_ops,
    3: pass3_llm_analysis_for_test_existence,
    4: pass4_dependency_rag,
    5: pass5_duplicate_detection
}

llm_steps = {3}


def parse_steps(steps_arg):
    """Parse step arguments like '1', '1-3', '1 2 3' into sorted list."""
    steps = set()
    
    for arg in steps_arg:
        arg = arg.strip()
        if '-' in arg:
            try:
                start, end = map(int, arg.split('-'))
                steps.update(range(start, end + 1))
            except ValueError:
                pass
        elif arg.isdigit():
            steps.add(int(arg))
    
    return sorted(steps)


def print_step_info(steps):
    """Print information about steps being run."""
    print("\n" + "=" * 60)
    print("Test Cases Processor - Step by Step Runner")
    print("=" * 60)
    
    print("\nSteps to run:")
    all_llm = all(s in llm_steps for s in steps)
    has_llm = any(s in llm_steps for s in steps)
    fast_mode = not has_llm
    
    for step in sorted(steps):
        desc = STEPS_DESC.get(step, f"Unknown step {step}")
        llm_marker = " [LLM - SLOW]" if step in llm_steps else ""
        print(f"  {step}. {desc}{llm_marker}")
    
    if fast_mode:
        print("\nMode: FAST (pattern-based, no LLM calls)")
    elif all_llm:
        print("\nMode: LLM ONLY (all steps require LLM)")
    else:
        print("\nMode: MIXED (some LLM calls required)")
    
    print(f"Total steps: {len(steps)}")
    print("=" * 60 + "\n")


def run_steps(steps_to_run, input_file=None, save=True):
    """
    Run specified steps of the test cases processor.
    
    Args:
        steps_to_run: list of step numbers to execute
        input_file: optional path to input Excel file
        save: whether to save results after each step
    
    Returns:
        tuple: (workbook, issues_needing_llm, issue_duplicated_map)
    """
    start_total = time.time()
    
    excel_file = input_file or os.path.join(RESULT_DIR, 'torch_xpu_ops_issues.xlsx')
    
    if not os.path.exists(excel_file):
        print(f"ERROR: File not found: {excel_file}")
        return None, None, None
    
    print(f"Loading: {excel_file}")
    wb = openpyxl.load_workbook(excel_file)
    
    if 'Test Cases' not in wb.sheetnames:
        print("ERROR: 'Test Cases' sheet not found in workbook")
        return None, None, None
    
    ws = wb['Test Cases']
    total_rows = ws.max_row - 1
    print(f"Total test cases: {total_rows}\n")
    
    issues_needing_llm = None
    issue_duplicated_map = None
    
    if 1 in steps_to_run:
        print_step_info([1])
        xpu_xml_files = get_torch_xpu_ops_xml_files()
        stock_xml_files = get_stock_xml_files()
        issues_needing_llm = pass1_match_ci_results(ws, xpu_xml_files, stock_xml_files)
        if save:
            wb.save(excel_file)
            print(f"Saved to: {excel_file}")
    
    if 2 in steps_to_run:
        print_step_info([2])
        pass2_extract_torch_ops(ws)
        if save:
            wb.save(excel_file)
            print(f"Saved to: {excel_file}")
    
    if 3 in steps_to_run:
        print_step_info([3])
        if issues_needing_llm is None:
            print("Note: Running PASS 1 first to get issues needing LLM...")
            xpu_xml_files = get_torch_xpu_ops_xml_files()
            stock_xml_files = get_stock_xml_files()
            issues_needing_llm = pass1_match_ci_results(ws, xpu_xml_files, stock_xml_files)
        pass3_llm_analysis_for_test_existence(ws, issues_needing_llm)
        if save:
            wb.save(excel_file)
            print(f"Saved to: {excel_file}")
    
    if 4 in steps_to_run:
        print_step_info([4])
        pass4_dependency_rag(ws)
        if save:
            wb.save(excel_file)
            print(f"Saved to: {excel_file}")
    
    if 5 in steps_to_run:
        print_step_info([5])
        issue_duplicated_map = pass5_duplicate_detection(ws)
        if save:
            wb.save(excel_file)
            print(f"Saved to: {excel_file}")
    
    elapsed_total = time.time() - start_total
    print("\n" + "=" * 60)
    print(f"Processing complete in {elapsed_total:.1f}s ({elapsed_total/60:.1f} min)")
    print("=" * 60)
    
    return wb, issues_needing_llm, issue_duplicated_map


def main():
    parser = argparse.ArgumentParser(
        description='Run Test Cases Processor - Step by Step',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--steps', nargs='+', default=None,
        help='Steps to run (e.g., "1", "1-3", "1 2 3", or "all")'
    )
    parser.add_argument(
        '--list', action='store_true',
        help='List all available steps with descriptions'
    )
    parser.add_argument(
        '--input', '-i', type=str, default=None,
        help='Input Excel file path (default: auto-detect from RESULT_DIR)'
    )
    parser.add_argument(
        '--no-save', action='store_true',
        help='Do not save results after each step (for debugging)'
    )
    parser.add_argument(
        '--fast', action='store_true',
        help='Run only fast steps (1, 2, 4, 5) - skip LLM dependent steps'
    )
    
    args = parser.parse_args()
    
    if args.list:
        print("\nAvailable Steps:")
        print("-" * 60)
        for step, desc in STEPS_DESC.items():
            llm = " [LLM - SLOW]" if step in llm_steps else ""
            print(f"  {step}. {desc}{llm}")
        print("-" * 60)
        return 0
    
    if args.fast:
        steps_to_run = [1, 2, 4, 5]
        print("Fast mode: Skipping step 3 (LLM analysis)")
    elif args.steps is None or (len(args.steps) == 1 and args.steps[0].lower() in ['all', 'a']):
        steps_to_run = [1, 2, 3, 4, 5]
        print("Running all steps (full processor)")
    else:
        steps_to_run = parse_steps(args.steps)
        if not steps_to_run:
            print("ERROR: No valid steps specified")
            print("Use --steps 1, --steps 1-3, or --steps 1 2 3")
            return 1
    
    os.makedirs(RESULT_DIR, exist_ok=True)
    
    save = not args.no_save
    run_steps(steps_to_run, input_file=args.input, save=save)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())