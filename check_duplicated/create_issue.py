import pandas as pd
from get_duplicated_issues.get_duplicated_issues import get_duplicated_issues
from get_duplicated_issues.get_duplicated_issues_ai import get_duplicated_issues_with_rag

df = pd.read_csv("ut_failure_list.csv", delimiter='|', engine='python')
df = df.rename(columns={df.columns[1]: 'Category', df.columns[2]: 'Class', df.columns[3]: 'Testcase', df.columns[4]: 'Result', df.columns[5]: 'ErrorMessage'})
df = df[['Class', 'Testcase', 'Result', 'ErrorMessage']]
header="Cases:"
id = 0

#download_all_open_issues_to_file("intel/torch-xpu-ops", os.getenv("GITHUB_TOKEN"))

env_info = ""
import os
if os.path.exists(f'collect_env.py'):
    import subprocess
    env_info = subprocess.check_output(['python', 'collect_env.py']).decode()

for group in df.groupby(['ErrorMessage']):
    skipped = []
    commands = []
    traces = []
    duplidcated = {}
    new_duplicated = {}
    for row in group[1].itertuples(index=False):
        test_class = row.Class.strip()
        _test_file = '/'.join(test_class.split('.')[:-1])
        test_file = _test_file + '.py'
        test_case = row.Testcase.strip()
        line = f"op_ut,{test_class},{test_case}"
        pytest_command = f"PYTORCH_TEST_WITH_SLOW=1 pytest -v {test_file} -k {test_case}"
        skipped.append(line)
        commands.append(pytest_command)
        xml_file = _test_file + '.xml'
        try:
            import xml.etree.ElementTree as ET
            tree = ET.parse(xml_file)
            root = tree.getroot()
            for testcase in root.iter('testcase'):
                if testcase.get('name') == test_case:
                    error = testcase.find('error')
                    failure = testcase.find('failure')
                    if error is not None:
                        traces.append(f"\n```\nCommand: {pytest_command}\n{error.text}```")
                    elif failure is not None:
                        traces.append(f"\n```\nCommand: {pytest_command}\n{failure.text}```")
                    else:
                        traces.append('')
                    break
        except Exception:
            traces.append('')

    
    duplicated_issue = get_duplicated_issues(skipped, group[0][0], traces[-1])
    
    # if len(duplicated_issue) == 0:
    #     duplicated_issue = get_duplicated_issues_with_rag(skipped, group[0][0], traces[-1], "/home/daisyden/upstream/check_duplicated/xpu_issues")
    duplidcated[id] = duplicated_issue

    if id > 0:
        new_duplicated_issue = get_duplicated_issues_with_rag(skipped, group[0][0], traces[-1], "/home/daisyden/upstream/check_duplicated/issues")
        new_duplicated[id] = list(set([ _issue["url"] for _issue in ( new_duplicated_issue["high_confidence"] + new_duplicated_issue["medium_confidence"])]))

    with open(f'issues/issue_group{id}.txt', 'w') as f:
        f.write(f"Title: [Upstream] {group[0][0]}\n")

        cases = '\n'.join(skipped)
        f.write(f"Cases:\n{cases}\n")
        
        commands_str = '\n'.join(commands)
        f.write(f"\npytest_command:\n{commands_str}\n")
        
        f.write("\nTrace Example:\n")
        f.write(traces[-1])

        if len(duplidcated) > 0 and duplidcated[id]:
            f.write("\n\nDuplicate Issues Found with RAG:\n")
            for issue in duplidcated[id]:
                f.write(f"- {issue['title']}\n  {issue['url']}\n")
        else: 
            f.write("\nNo duplicate issues found with RAG in xpu_issues.\n")

        if len(new_duplicated) > 0 and new_duplicated[id]:
            f.write("\n\nDuplicate Issues generated:\n")
            f.write(f"{new_duplicated[id]}\n")
        else: 
            f.write("\nNo duplicate issues generated.\n")

        #f.write(f"\n\nEnvironment Information:\n{env_info}\n")

    id += 1