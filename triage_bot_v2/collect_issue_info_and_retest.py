import argparse
import json
import os
from prepare import ( 
    reproduce_issues_with_docker_commands,
    extract_torch_test_details,
    extract_reproduce_commands,
    build_pytorch_docker_enviroment,
    get_issue_body,
    group_failed_test_cases,
    extract_commit_range,
    )

issue_template = "./prompts/issue_template.j2"

parser = argparse.ArgumentParser()
parser.add_argument("--issue", type=str, help="The issue number to process", default='0', required=False)
parser.add_argument("--inputs", type=str, help="The input withg issue information", default="", required=False)
parser.add_argument("--repo", type=str, help="The github repo of the issue", default='intel/torch-xpu-ops', required=False)
parser.add_argument("--container", type=str, help="The docker container to use, if not exists the script will create one", default='guilty_commit', required=False)

parser.add_argument("--since", type=str, help="The last good commit for this issue or the date when the issue does not exists", default=None, required=False)
parser.add_argument("--until", type=str, help="The commit when the issue is detected or the date when the issue is detected", default=None, required=False)

args = parser.parse_args()

def collect_issue_info_and_retest(
        issue: str = '0',
        inputs: str = '',
        repo: str = 'intel/torch-xpu-ops',
        container: str = 'guilty_commit',
        since: str = None,
        until: str = None,
):
    ######## extract issue information ##########
    token = os.environ.get("GITHUB_TOKEN", "")
    if repo is None and token is None:
        print("Please provide repo and token for issue analysis")
        exit(1)

    try:
        from jinja2 import Environment, FileSystemLoader
        env = Environment(loader=FileSystemLoader('.'))
        # Pass json module to template
        template = env.get_template(issue_template)
        json_template = template.render(json=json)  # Pass json module
        json_template = json.loads(json_template)
    except:
        print(f"Failed to load issue template from {issue_template}")
        exit(1)

    workdir = "/workdir"
    issue_information = json_template

    if len(inputs) > 0:
        user_input = inputs
        issue = '0'
    elif issue != '0':
        # Get all the commands to reproduce the issue
        body = get_issue_body(int(issue), token, repo)
        
        # last good commit extraction TBD

        user_input = body
    else:
        print("Enter cases (press Ctrl+D on Unix/Linux/Mac or Ctrl+Z on Windows when done):")
        import sys
        user_input = sys.stdin.read()
        user_input = "Cases: \n" + user_input
        issue = '0'

    enviroments = build_pytorch_docker_enviroment(False, "nightly", container)

    commands = extract_reproduce_commands(user_input)

    import pdb
    pdb.set_trace()

    since, until = "unknown", "unknown"

    if issue is not None:
        since, until = extract_commit_range(user_input)
        print(f"Extracted commit range since: {since}, until: {until} for issue #{issue}")
        
    
    if len(commands) != 0:
        reproduce_result = reproduce_issues_with_docker_commands(commands, container)

        issue_list = []
        response = ""
        for _reproduce_result in reproduce_result:
            issue_information = json_template.copy()
            issue_information["error_message"] = _reproduce_result['error_message']
            issue_information['traceback'] = _reproduce_result['traceback']
            issue_information['test_file'] = _reproduce_result['test_file']
            issue_information['test_class'] = _reproduce_result['test_class']
            issue_information['test_case'] = _reproduce_result['test_case']
            issue_information['error_message'] = _reproduce_result['error_message']
            issue_information['traceback'] = _reproduce_result['traceback']
            issue_information['reproduce_command'] = _reproduce_result['reproduce_command']
            if "_xpu.py" in _reproduce_result['test_file']:
                issue_information['test_category'] = 'torch-xpu-ops'
                issue_information['workdir'] = '/workdir/pytorch/third_party/torch-xpu-ops/test/xpu'
            else:   
                issue_information['test_category'] = 'pytorch'
                issue_information['workdir'] = '/workdir/pytorch/test'
            
            if _reproduce_result['error_message'] != "passed":
                prompt = "extraction_torch_details.j2"

                python_object = extract_torch_test_details(issue_information, prompt)
                import pdb
                pdb.set_trace()
                issue_information["torch_op"] = python_object.get("torch_op", "unknown")
                issue_information['dependency'] = python_object.get("dependency", "unknown")
                issue_information['original_test_file'] = python_object.get("original_test_file", "unknown")
                issue_information['original_test_class'] = python_object.get("original_test_class", "unknown")
                issue_information['original_test_case'] = python_object.get("original_test_case", "unknown")
                issue_information['original_test_case_lineno'] = python_object.get("original_test_case_lineno", "unknown")
                issue_information['error_type'] = python_object.get("error_type", "unknown")
                issue_information['module'] = python_object.get("module", "unknown")
                issue_information['dtype'] = python_object.get("dtype", "unknown")
            else:
                issue_information["torch_op"] = "unknown"
                issue_information['dependency'] = "unknown"
                issue_information['original_test_file'] = "unknown"
                issue_information['original_test_class'] = "unknown"
                issue_information['original_test_case'] = "unknown"
                issue_information['original_test_case_lineno'] = "unknown"
                issue_information['error_type'] = "unknown"
                issue_information['module'] = "unknown"
                issue_information['dtype'] = "unknown"

            issue_information['json_link'] = f"results/{issue}/issue#{issue}_{_reproduce_result['test_file'].replace('/', '_')}_{_reproduce_result['test_class']}_{_reproduce_result['test_case']}.json" \
                                            if issue is not None \
                                            else f"results/issue_{_reproduce_result['test_file'].replace('/', '_')}_{_reproduce_result['test_class']}_{_reproduce_result['test_case']}.json"
            
            issue_information['since'] = since
            issue_information['until'] = until

            issue_information['link'] = f"https://github.com/{repo}/issues/{issue}" if issue is not None else "unknown"
            issue_information['container'] = container
            issue_information['repo'] = repo

            issue_list.append(issue_information)
            os.makedirs(os.path.dirname(issue_information['json_link']), exist_ok=True)
            with open(issue_information['json_link'], 'w') as f:
                json.dump(issue_information, f, indent=4)
                response += f"## {issue_information['json_link']}:\n{json.dumps(issue_information, indent=4)}\n"
        
        groups = group_failed_test_cases(issue_list)

        os.makedirs(f"results/{issue}", exist_ok=True)
        with open(f"results/{issue}/classified_result_issue#{issue}.txt", 'w') as f:
            f.write(f"{groups}\n")
            f.write(f"\n\nVerification enviroment: \n {enviroments}\n")
        
        response = f"There are {len(groups)} groups of issues:\n{groups}\n\nVerification enviroment: \n\n {enviroments}\nDetails:\n" + response
        return response
        
if __name__ == "__main__":
    collect_issue_info_and_retest(
            issue=args.issue,
            inputs=args.inputs,
            repo=args.repo,
            container=args.container,
            since=args.since,
            until=args.until,
    )       