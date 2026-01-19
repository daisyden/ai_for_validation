import argparse
import json
from prepare import ( 
    reproduce_issues_with_docker_commands,
    extracte_information_from_log,
    extract_reproduce_commands,
    extract_issue_information,
    build_pytorch_docker_enviroment 
    )

issue_template = "./prompts/issue_template.j2"

parser = argparse.ArgumentParser()
parser.add_argument("--repo", type=str, help="The github repo of the issue", default='pytorch/pytorch', required=False)
parser.add_argument("--token", type=str, help="The github token to the repo", required=True)
parser.add_argument("--container", type=str, help="The docker container to use, if not exists the script will create one", default='guilty_commit', required=False)

parser.add_argument("--issue", type=str, help="The issue number to process", required=False)
parser.add_argument("--case", type=str, help="The test case to run, e.g., " \
"op_ut,third_party.torch-xpu-ops.test.xpu.test_autograd_xpu.TestAutograd,test_checkpointing_without_reentrant_detached_tensor_use_reentrant_True, or" \
"test_autograd_xpu.py,TestAutograd,test_checkpointing_without_reentrant_detached_tensor_use_reentrant_True", required=False)

parser.add_argument("--retest", action="store_true", default=False, help="If set, do retest in docker container.", required=False)

parser.add_argument("--since", type=str, help="The last good commit for this issue or the date when the issue does not exists", required=False)
parser.add_argument("--until", type=str, help="The commit when the issue is detected or the date when the issue is detected", required=False)

args = parser.parse_args()

######## extract issue information ##########
if args.repo is None and args.token is None:
    print("Please provide --repo and --token for issue analysis")
    exit(1)

if args.build not in ["source", "nightly"]:
    print("The args.build neither source nor nightly, will use existing build.")
    args.build = "existing"
    
try:
    from jinja2 import Environment, FileSystemLoader
    import pdb
    pdb.set_trace()
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

if args.case is not None:
    print("Extracting issue information from the provided test case, will do a retest to collect more information.")
    if args.case.beginswith("op_ut"):
        test_file = args.case.replace("third_party.torch-xpu-ops.test.xpu.", "")
        test_file = "/".join(test_file.split(",")[:-1]) + ".py"
        test_class = args.case.split(",")[2].split(".")[-1]
        test_case = args.case.split(",")[3]
        issue_information["test_category"] = "torch-xpu-ops"
    else:
        test_file = args.case.split(",")[0]
        test_class = args.case.split(",")[1] if len(args.case.split(",")) > 2 else ""
        test_case = args.case.split(",")[2] if len(args.case.split(",")) > 2 else args.case.split(",")[1]
    issue_information['test_file'] = test_file
    issue_information['test_class'] = test_class
    issue_information['test_case'] = test_case
 
    commands = [[issue_information['test_file'], 
                issue_information['test_class'],
                issue_information['test_case'],
                f"pytest --junit-xml={issue_information['test_case']}.xml -v {issue_information['test_file']}::{issue_information['test_class'] if issue_information['test_class'] != '' else ''} -k {issue_information['test_case']}"],]
    args.retest = True
else:
    if args.issue is not None:
        # Get all the commands to reproduce the issue
        commands = extract_reproduce_commands(int(args.issue), args.token, args.repo)
        
        issue_information = extract_issue_information(int(args.issue), args.token, args.repo)

        if issue_information.get("test_file", None) is None or issue_information.get("test_case", None) is None:
            print("Failed to extract basic test case information from the issue.")
            exit(1)

        if issue_information.get("original_test_file", None) is None:
            issue_information["original_test_file"] = issue_information["test_file"].replace("_xpu.py", ".py")
            issue_information["original_test_file"] = issue_information["original_test_file"].replace("/xpu/", "/")

if args.retest == True:
    folder = f"{workdir}/pytorch" if args.repo == "pytorch/pytorch" else f"{workdir}/pytorch/third_party/torch-xpu-ops/test/xpu" 
    reproduce_result = reproduce_issues_with_docker_commands(folder, commands, args.container)

    if reproduce_result["status"] == "passed":
        print(f"All the cases passed, the issue is not reproduced under local enviroments.\n")
        exit(0)
    else:
        prompt = "extraction_issue_from_log.j2"
        if folder != workdir:
            issue_information['test_category'] = 'torch-xpu-ops'
        else:   
            issue_information['test_category'] = 'pytorch'
        python_object = extracte_information_from_log(issue_information['test_category'], reproduce_result, prompt)
        
        since = issue_information["since"]
        until = issue_information["until"]
        if python_object is not None:
            issue_information = python_object
            issue_information["since"] = since
            issue_information["until"] = until
        
if args.since is not None:
    issue_information["since"] = args.since
if args.until is not None:
    issue_information["until"] = args.until

issue_information.update([("container", args.container), ("repo", args.repo), ("workdir", folder)])
print(f"The issue is reproduced with issue information collected {issue_information}.")

if args.issue is not None:
    json.dump(issue_information, open(f"json/issue_{args.issue}_information.json", 'w'))
    print(f"Issue information saved to json/issue_{args.issue}_information.json")
else:
    json.dump(issue_information, open(f"json/issue_{test_case}_information.json", 'w'))
    print(f"Issue information saved to json/issue_{test_case}_information.json")