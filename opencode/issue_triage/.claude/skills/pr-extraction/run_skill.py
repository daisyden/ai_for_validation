#!/usr/bin/env python3
"""
Wrapper script to run the PR Extraction skill.

This script invokes the pr_extraction.py module with proper path setup.

Usage:
    python3 run_skill.py [--excel EXCEL_FILE] [--issues ISSUE_IDS]

Example:
    python3 run_skill.py
    python3 run_skill.py --excel /path/to/torch_xpu_ops_issues.xlsx
    python3 run_skill.py --issues 3286,3284,3258
"""

import os
import sys
import argparse

# Skill directory
SKILL_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = '/home/daisydeng/ai_for_validation/opencode/issue_triage'
RESULT_DIR = os.environ.get('RESULT_DIR', os.path.join(ROOT_DIR, 'result'))

def main():
    parser = argparse.ArgumentParser(
        description='Run PR Extraction Skill',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--excel', '-e', type=str, default=None,
        help=f'Path to Excel file (default: {RESULT_DIR}/torch_xpu_ops_issues.xlsx)'
    )
    parser.add_argument(
        '--issues', '-i', type=str, default='',
        help='Comma-separated list of issue IDs to process (default: all)'
    )
    
    args = parser.parse_args()
    
    excel_file = args.excel or os.path.join(RESULT_DIR, 'torch_xpu_ops_issues.xlsx')
    
    if not os.path.exists(excel_file):
        print(f"ERROR: Excel file not found: {excel_file}")
        return 1
    
    # Build command
    cmd = [sys.executable, os.path.join(SKILL_DIR, '..', 'pr-extraction', 'pr_extraction.py'), excel_file]
    if args.issues:
        cmd.extend(['--issues', args.issues])
    
    # Execute
    print(f"Running PR Extraction Skill...")
    print(f"Excel: {excel_file}")
    if args.issues:
        print(f"Issues: {args.issues}")
    print("-" * 50)
    
    os.chdir(os.path.join(SKILL_DIR, '..', '..', '..', 'issue_analysis', 'pr-extraction'))
    os.execv(sys.executable, cmd)

if __name__ == '__main__':
    main()