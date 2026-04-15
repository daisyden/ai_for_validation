#!/usr/bin/env python3
"""
Copy Action Reason from source Excel to target Excel.

Only updates the 'Action Reason' column in the Issues sheet.
Does not modify any other columns or sheets.
"""
import openpyxl
import sys

def copy_action_reason(
    source_path: str = '~/torch_xpu_ops_issues_action_reason2.xlsx',
    target_path: str = '~/ai_for_validation/opencode/issue_triage/result/torch_xpu_ops_issues.xlsx',
    sheet_name: str = 'Issues',
    action_reason_col: int = 20  # Column 20 = 'Action Reason'
):
    """
    Copy Action Reason from source to target Excel file.
    
    Args:
        source_path: Source Excel file with Action Reason column
        target_path: Target Excel file to update
        sheet_name: Name of sheet to update
        action_reason_col: Column index for Action Reason (1-indexed)
    """
    # Expand paths
    source_path = os.path.expanduser(source_path)
    target_path = os.path.expanduser(target_path)
    
    # Load source workbook
    print(f"Loading source: {source_path}")
    wb_source = openpyxl.load_workbook(source_path)
    
    if sheet_name not in wb_source.sheetnames:
        print(f"Error: Sheet '{sheet_name}' not found in source")
        return
    
    ws_source = wb_source[sheet_name]
    
    # Find Action Reason column in source header
    source_action_col = None
    for i, cell in enumerate(ws_source[1], 1):
        if cell.value == 'Action Reason':
            source_action_col = i
            break
    
    if source_action_col is None:
        print("Error: 'Action Reason' column not found in source sheet header")
        return
    
    print(f"Found 'Action Reason' at column {source_action_col} in source")
    
    # Get Action Reason data by Issue ID from source
    source_data = {}
    for row in ws_source.iter_rows(min_row=2, values_only=True):
        issue_id = row[0]  # Issue ID is column 1
        action_reason = row[source_action_col - 1] if source_action_col <= len(row) else None
        if issue_id is not None:
            source_data[issue_id] = action_reason
    
    print(f"Loaded {len(source_data)} Action Reason entries from source")
    
    # Load target workbook
    print(f"\nLoading target: {target_path}")
    wb_target = openpyxl.load_workbook(target_path)
    
    if sheet_name not in wb_target.sheetnames:
        print(f"Warning: Sheet '{sheet_name}' not found in target, creating it")
        ws_target = wb_target.create_sheet(sheet_name)
    else:
        ws_target = wb_target[sheet_name]
    
    # Find Action Reason column in target header
    target_action_col = None
    target_issue_col = 1  # Issue ID is always column 1
    
    for i, cell in enumerate(ws_target[1], 1):
        if cell.value == 'Action Reason':
            target_action_col = i
            break
    
    if target_action_col is None:
        print("Error: 'Action Reason' column not found in target sheet header")
        wb_target.close()
        return
    
    print(f"Found 'Action Reason' at column {target_action_col} in target")
    
    # Update Action Reason in target for matching Issue IDs
    updated_count = 0
    skipped_count = 0
    
    for row in ws_target.iter_rows(min_row=2):
        issue_id = row[target_issue_col - 1].value  # Get Issue ID from column 1
        
        if issue_id is not None and issue_id in source_data:
            new_action = source_data[issue_id]
            
            # Only update if different or target is empty
            current_action = row[target_action_col - 1].value
            
            if current_action != new_action:
                row[target_action_col - 1].value = new_action
                updated_count += 1
            else:
                skipped_count += 1
    
    print(f"\nUpdate complete:")
    print(f"  Updated: {updated_count} Action Reason entries")
    print(f"  Skipped (same value): {skipped_count}")
    
    # Save target workbook
    wb_target.save(target_path)
    print(f"\nSaved: {target_path}")
    
    wb_target.close()
    wb_source.close()

if __name__ == '__main__':
    import os
    
    source = sys.argv[1] if len(sys.argv) > 1 else '~/torch_xpu_ops_issues_action_reason2.xlsx'
    target = sys.argv[2] if len(sys.argv) > 2 else '~/ai_for_validation/opencode/issue_triage/result/torch_xpu_ops_issues.xlsx'
    
    copy_action_reason(source, target)