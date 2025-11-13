from .github_issue import Github_Issue 

def download_issue_content(issue):
    content = ""
    if issue.body is not None:
        content += issue.body + "\n"
    content = content.split('Versions')[0]
    comments = issue.get_comments()
    for comment in comments:
        if comment.body is not None:
            content += comment.body + "\n"
    return content

def download_all_open_issues_to_file(repo, token):
    gh = Github_Issue(repo, token)
    issues = gh.get_issues(state="open")

    for issue in issues:
        content = download_issue_content(issue)
        with open(f"xpu_issues/{issue.number}.txt", "w", encoding="utf-8") as f:
            f.write(f"Issue #{issue.number}: {issue.title}\n")
            f.write(content)
            f.write("\n" + "="*80 + "\n\n")

def get_duplicated_issues(skipped:list, error_message:str, trace:str):
    duplicated_issues = []
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    for issue_file in os.listdir(os.path.join(current_dir, "xpu_issues")):
        issue_file = os.path.join(current_dir, "xpu_issues", issue_file)
        if issue_file.endswith(".txt"):
            with open(issue_file, "r", encoding="utf-8") as f:
                content = f.read()
                for skip in skipped:
                    _test_file = '/'.join(skip.split(',')[1].split('.')[:-1])
                    test_file = _test_file + '.py'
                    test_case = skip.split(',')[2].strip()
                    if (f"{test_file}".replace('_xpu','').replace('test/', '') in content) and \
                        f"{test_case}" in content and \
                        f"{error_message}" in content:
                        duplicated_issues.append(content)

            return duplicated_issues



