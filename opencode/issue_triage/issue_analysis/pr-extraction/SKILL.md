# PR Extraction Skill

Extracts PR information from GitHub issue comments and updates an Excel file with PR links, owners, and status.

## When to use

Use when you need to:
- Extract PR references from GitHub issue comments
- Update an Excel file with PR information (PR, PR Owner, PR Status columns)
- Only collect PRs that are mentioned with fix-related keywords (fix, fixes, closed, resolved, PR #, etc.)
- Handle multiple PRs per issue (separated by `;`)

## How it works

1. **Read Issues from Excel**: Load issue IDs from the Issues sheet
2. **Fetch Comments**: Get all comments for each issue from GitHub API
3. **Extract PR References**: Search for PR URLs in comments with fix-related keywords
4. **Fetch PR Info**: Get PR details (URL, owner, state) for each unique PR
5. **Apply Different Status Rules**:
   - `pytorch/pytorch` PRs: Check "Merged" label → returns 'merged', otherwise open/closed
   - `intel/torch-xpu-ops` PRs: Check merged/merged_at field → returns 'merged', otherwise open/closed
6. **Update Excel**: Write PR, PR Owner, PR Status columns (multiple PRs separated by `;`)

## Usage

```bash
python3 pr_extraction.py <excel_file>
```

## Example

```bash
python3 pr_extraction.py /path/to/torch_xpu_ops_classified.xlsx
```

## Output

Updates the following columns in the Issues sheet:
- **PR**: PR URLs (semicolon-separated for multiple PRs)
- **PR Owner**: GitHub usernames (semicolon-separated)
- **PR Status**: open/closed/merged (semicolon-separated)