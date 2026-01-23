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
parser.add_argument("--issue", type=str, help="The issue number to process", required=False)
parser.add_argument("--repo", type=str, help="The github repo of the issue", default='intel/torch-xpu-ops', required=False)

parser.add_argument("--token", type=str, help="The github token to the repo", required=True)
parser.add_argument("--container", type=str, help="The docker container to use, if not exists the script will create one", default='guilty_commit', required=False)

parser.add_argument("--since", type=str, help="The last good commit for this issue or the date when the issue does not exists", required=False)
parser.add_argument("--until", type=str, help="The commit when the issue is detected or the date when the issue is detected", required=False)

args = parser.parse_args()

######## extract issue information ##########
if args.repo is None and args.token is None:
    print("Please provide --repo and --token for issue analysis")
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

if args.issue is None:
    print("Enter cases (press Ctrl+D on Unix/Linux/Mac or Ctrl+Z on Windows when done):")
    import sys
    user_input = sys.stdin.read()
    user_input = "Cases: \n" + user_input
    args.issue = '0'
else:
    # Get all the commands to reproduce the issue
    body = get_issue_body(int(args.issue), args.token, args.repo)
    
    # last good commit extraction TBD

    user_input = body

enviroments = build_pytorch_docker_enviroment(False, "nightly", args.container)

commands = extract_reproduce_commands(user_input)

since, until = "unknown", "unknown"

if args.issue is not None:
    since, until = extract_commit_range(user_input)
    print(f"Extracted commit range since: {since}, until: {until} for issue #{args.issue}")
    

if len(commands) != 0:
    
    reproduce_result = reproduce_issues_with_docker_commands(commands, args.container)

    issue_list = []

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
        else:   
            issue_information['test_category'] = 'pytorch'
        
        if _reproduce_result['error_message'] != "passed":
            prompt = "extraction_torch_details.j2"
            python_object = extract_torch_test_details(issue_information, prompt)
            issue_information["torch_op"] = python_object.get("torch_op", "unknown")
            issue_information['dependency'] = python_object.get("dependency", "unknown")
            issue_information['original_test_file'] = python_object.get("original_test_file", "unknown")
            issue_information['original_test_class'] = python_object.get("original_test_class", "unknown")
            issue_information['original_test_case'] = python_object.get("original_test_case", "unknown")
            issue_information['original_test_case_lineno'] = python_object.get("original_test_case_lineno", "unknown")
            issue_information['error_type'] = python_object.get("error_type", "unknown")
            issue_information['module'] = python_object.get("module", "unknown")
            issue_information['dtype'] = python_object.get("dtype", "unknown")

        issue_information['json_link'] = f"results/{args.issue}/issue#{args.issue}_{_reproduce_result['test_file'].replace('/', '_')}_{_reproduce_result['test_class']}_{_reproduce_result['test_case']}.json" \
                                        if args.issue is not None \
                                        else f"results/issue_{_reproduce_result['test_file'].replace('/', '_')}_{_reproduce_result['test_class']}_{_reproduce_result['test_case']}.json"
        if len(since) != 0 and since != "unknown":
            issue_information['since'] = since
        if len(until) != 0 and until != "unknown":
            issue_information['until'] = until
        issue_information['link'] = f"https://github.com/{args.repo}/issues/{args.issue}" if args.issue is not None else "unknown"
        issue_information['container'] = args.container
        issue_information['repo'] = args.repo

        issue_list.append(issue_information)
        os.makedirs(os.path.dirname(issue_information['json_link']), exist_ok=True)
        with open(issue_information['json_link'], 'w') as f:
            json.dump(issue_information, f, indent=4)
    
    groups = group_failed_test_cases(issue_list)

    if args.issue is not None:
        os.makedirs(f"results/{args.issue}", exist_ok=True)
        with open(f"results/{args.issue}/classified_result_issue#{args.issue}.txt", 'w') as f:
            f.write(f"{groups}\n")
            f.write(f"\n\nVerification enviroment: \n {enviroments}\n")
    else:
        os.makedirs(f"results", exist_ok=True)
        with open(f"results/classified_result.txt", 'w') as f:
            f.write(f"{groups}\n")
            f.write(f"\n\nVerification enviroment: \n {enviroments}\n") 
    

    