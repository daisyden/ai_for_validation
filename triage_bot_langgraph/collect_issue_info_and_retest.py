import os
import argparse
import json
from prepare import reproduce_issues_with_commands, extracte_information, extract_reproduce_commands, extract_issue_information, build_pytorch_enviroment

parser = argparse.ArgumentParser()
parser.add_argument("--case", type=str, help="The test case to run, e.g., test/test_nn.py::TestNN::test_function", required=False)
parser.add_argument("--issue", type=str, help="The issue number to process", required=False)
parser.add_argument("--repo", type=str, help="The github repo of the issue", default='pytorch/pytorch', required=False)
parser.add_argument("--token", type=str, help="The github token to the repo", required=True)
parser.add_argument("--issue_json", type=str, help="Path to JSON file containing issue information (alternative to --issue/--repo/--token)", required=False)
parser.add_argument("--tmux", type=str, help="The tmux session to use, if not exists the script will create one", default='guilty_commit', required=False)
parser.add_argument("--issue_template", type=str, help="The json template file for issue information", default="./prompts/issue_template.j2", required=False)

parser.add_argument("--workdir", type=str, help="The working directory to clone pytorch repo", default="~/pytorch", required=False)
parser.add_argument("--build", type=str, help="The build method, source or nightly or existing", default="existing", required=False)
parser.add_argument("--download", action="store_true", default=False, help="If set, download pytorch source code.", required=False)
parser.add_argument("--retest", action="store_true", default=True, help="If set, do retest in tmux session.", required=False)

parser.add_argument("--since", type=str, help="The since time for git log", required=False)
parser.add_argument("--until", type=str, help="The until time for git log", required=False)

args = parser.parse_args()

import pdb
pdb.set_trace()

######## extract issue information ##########
if args.repo is None and args.token is None:
    print("Please provide --repo and --token for issue analysis")
    exit(1)

if args.issue_json is None and args.case is None and args.issue is None:
    print("Please provide --issue_json or --case or --issue")
    exit(1)

if args.build not in ["source", "nightly"]:
    print("Please provided is not source or nightly, will use existing build.")
    args.build = "existing"
    
try:
    from jinja2 import Environment, FileSystemLoader
    env = Environment(loader=FileSystemLoader('.'))
    # Pass json module to template
    template = env.get_template(args.issue_template)
    json_template = template.render(json=json)  # Pass json module
    json_template = json.loads(json_template)
except:
    print(f"Failed to load issue template from {args.issue_template}")
    exit(1)

args.workdir = os.path.expanduser(args.workdir)
issue_information = json_template

if args.case is not None:
    print("Extracting issue information from the provided test case, will do a retest to collect more information.")
    test_file = args.case.split("::")[0]
    test_class = args.case.split("::")[1] if len(args.case.split("::")) > 2 else ""
    test_case = args.case.split("::")[2] if len(args.case.split("::")) > 2 else args.case.split("::")[1]
    issue_information['test_file'] = test_file
    issue_information['test_class'] = test_class
    issue_information['test_case'] = test_case
 
    commands = [[issue_information['test_file'], 
                issue_information['test_class'],
                issue_information['test_case'],
                f"pytest --junit-xml={issue_information['test_case']}.xml -v {issue_information['test_file']}::{issue_information['test_class'] if issue_information['test_class'] != '' else ''} -k {issue_information['test_case']}"],]
    args.retest = True
else:
    if args.issue_json is not None:
        issue_information = json.loads(open(args.issue_json, 'r').read())
        commands = [[{issue_information['test_file']}, 
                    {issue_information['test_class']},
                    {issue_information['test_case']},
                    f"pytest -v {issue_information['test_file']}::{issue_information['test_class'] if issue_information['test_class'] != '' else ''} -k {issue_information['test_case']}"],]
        if issue_information.get("original_test_file", None) is None or issue_information.get("traceback", None) is None:
            print("Will do retest to collect more information.")
            args.retest = True
    else:
        if args.issue is not None:
            # Get all the commands to reproduce the issue
            commands = extract_reproduce_commands(int(args.issue), args.token, args.repo)
            if args.retest is False:
                # No retest, only extract issue information from the issue
                issue_information = extract_issue_information(int(args.issue), args.token, args.repo)

                if issue_information.get("test_file", None) is None or issue_information.get("test_case", None) is None:
                    print("Failed to extract basic test case information from the issue.")
                    exit(1)

                if issue_information.get("original_test_file", None) is None:
                    issue_information["original_test_file"] = issue_information["test_file"].replace("_xpu.py", ".py")
                    issue_information["original_test_file"] = issue_information["original_test_file"].replace("/xpu/", "/")

enviroments = build_pytorch_enviroment(args.download, args.build, args.tmux, args.workdir)

with open(f"results/enviroments.log", 'w') as f:
    for env in enviroments:
        f.write(f"{env}\n")

if len(enviroments) == 0:
    print("Failed to build pytorch enviroment.")
    exit(1)

if args.retest == True:
    folder = args.workdir if args.repo == "pytorch/pytorch" else f"{args.workdir}/third_party/torch-xpu-ops/test/xpu" 
    reproduce_result = reproduce_issues_with_commands(folder, commands, args.tmux)

    if reproduce_result["status"] == "passed":
        print(f"All the cases passed, the issue is not reproduced under local enviroments.\n")
        exit(0)
    else:
        prompt = "extraction_issue_from_log.j2"
        if folder != args.workdir:
            issue_information['test_category'] = 'torch-xpu-ops'
        else:   
            issue_information['test_category'] = 'pytorch'
        python_object = extracte_information(issue_information['test_category'], reproduce_result, prompt)
        issue_information = python_object

if args.since is not None:
    issue_information["since"] = args.since
if args.until is not None:
    issue_information["until"] = args.until

issue_information.update([("tmux", args.tmux), ("repo", args.repo), ("workdir", args.workdir)])
print(f"The issue is reproduced with issue information collected {issue_information}.")

if args.issue is not None:
    json.dump(issue_information, open(f"json/issue_{args.issue}_information.json", 'w'))
    print(f"Issue information saved to json/issue_{args.issue}_information.json")
else:
    json.dump(issue_information, open(f"json/issue_{test_case}_information.json", 'w'))
    print(f"Issue information saved to json/issue_{test_case}_information.json")