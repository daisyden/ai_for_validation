import argparse
import json
from jinja2 import Environment, FileSystemLoader
from guilty_commit import git_show_in_tmux, get_blamed_commits_case_update, get_git_log
from common_nodes import stream_graph_updates, classify_depsrag_graph, classify_graph
from prepare import reproduce_issues_with_commands, extracte_information, extract_reproduce_commands, extract_issue_information, get_tmux_sessions, build_pytorch_enviroment, reproduce_issue_in_tmux
import time


parser = argparse.ArgumentParser()
parser.add_argument("--issue_json", type=str, help="Path to JSON file containing issue information (alternative to --issue/--repo/--token)", required=False)
parser.add_argument("--match", type=str, help="The pattern to match in git log", default="test")
parser.add_argument("--since", type=str, help="The start time for git blame, e.g., '2023-10-01'", required=False)
parser.add_argument("--until", type=str, help="The end time for git blame, e.g., '2023-10-31'", required=False) 
parser.add_argument("--blame_range", type=str, help="The git blame range, e.g., 'commit1..commit2'", required=False)
parser.add_argument("--blame_callee", action="store_true", default=False, help="If set, use call tracer to trace the function calls and git blame callee functions.", required=False)

args = parser.parse_args()

start = time.time()

import pdb
pdb.set_trace()
try:
    issue_information = json.load(open(args.issue_json, 'r'))
except:
    print(f"Failed to load issue information from {args.issue_json}")
    exit(1)

tmux = issue_information.get("tmux", "guilty_commit")
workdir = issue_information.get("workdir", "./pytorch")
test_file = issue_information.get("test_file", None)
original_test_file = issue_information.get("original_test_file", None)
test_class = issue_information.get("test_class", None)
test_case = issue_information.get("test_case", None)
error_message = issue_information.get("error_message", None)
last_good_commit = issue_information.get("last_good_commit", None)
since = issue_information.get("since", None) if args.since is None else args.since
until = issue_information.get("until", None) if args.until is None else args.until
match = args.match

if args.blame_range is not None:
    blame_range = args.blame_range
else:
    import pdb
    pdb.set_trace()
    blame_range = f"--since {since}" if since is not None else ""
    blame_range += f" --until {until}" if until is not None else ""
    if since == "" and until == "":
        blame_range = "HEAD~200..HEAD"



######## Find guilty commit ###########
# Prase git log to get possible guilty commits
# Parse the git show of each candidate commit to confirm if it is the guilty commit
# Git blame each callee function to get more candidate commits if needed
#######################################
graph = classify_graph()

if tmux is not None and workdir is not None: 
    try:
        failure_info = issue_information
        print(f"Failure information: {failure_info}")
        ###############################
        # Collect realted commit:
        # 1. commit updated the test function
        # 2. commit updated the callee functions of the test function
        # 3. commit updated the op_db definition of the test ops
        # 4. commit updated the related module
        # 5. commit updated the related dependency component
        ###############################
        # 

        import pdb
        pdb.set_trace()
        git_log_output = get_git_log(issue_information, blame_range, match=match)
        def generate_prompt_git_log(git_log: str, python_object: dict) -> str:
            env = Environment(loader=FileSystemLoader('prompts'))
            template = env.get_template('guilty_commit_gitlog.j2')
            # use original test file because we want to check pytorch git log
            return template.render(git_log=git_log,
                                  test_file=python_object["original_test_file"],
                                  test_case=python_object["original_test_case"],
                                  torch_op=python_object["torch_op"],
                                  module=python_object["module"],
                                  dependency=python_object["dependency"],
                                  error_message=python_object["error_message"])
        prompt = generate_prompt_git_log(git_log=git_log_output, python_object=failure_info)
        user_input = prompt
        json_string = stream_graph_updates(user_input, graph)

        python_object = json.loads(json_string.replace("```json\n","").replace("```", ""))
        blamed_commits_git_log = []
        for key in ["test_case_update", "related_components_update", "error_message_related_update"]:
            if python_object.get(key) not in [False, "false"]:
                blamed_commits_git_log += [ v.strip() for v in python_object.get(key).split(",")]
        blamed_commits_git_log = list(set(blamed_commits_git_log))
        print(f"Potential blamed commits from git log: {blamed_commits_git_log}")

        blamed_commits = []
        called_functions = []
        if args.blame_callee == True:
            blamed_commits, called_functions = get_blamed_commits_case_update(tmux, failure_info, blame_range, match)
            called_functions = list(set([func[0] for func in called_functions]))
            print(f"Potential blamed commits from git blame and call tracer: {blamed_commits}.")

        for commit in set(list(blamed_commits) + list(blamed_commits_git_log)):
            print(f"Check blamed commit: {commit}.")

            try:
                show_results = git_show_in_tmux(commit, tmux)
                
                def generate_prompt_gitshow(commit_sha: str, git_show: str, python_object: dict, called_functions) -> str:
                    env = Environment(loader=FileSystemLoader('prompts'))
                    template = env.get_template('guilty_commit_gitshow.j2')
                    # use original test file because we want to check pytorch git log
                    return template.render(commit_sha=commit_sha, 
                                          git_show=git_show,
                                          original_test_file=python_object["original_test_file"],
                                          original_test_case=python_object["original_test_case"],
                                          original_test_class=python_object["original_test_class"],
                                          called_functions=called_functions,
                                          torch_op=python_object["torch_op"],
                                          module=python_object["module"],
                                          dependency=python_object["dependency"],
                                          error_message=python_object["error_message"])
                
                prompt = generate_prompt_gitshow(commit_sha=commit, 
                                          git_show=show_results,
                                          python_object=failure_info,
                                          called_functions=",".join(called_functions) if len(called_functions) > 0 else "")
                user_input = prompt
                
                json_string = stream_graph_updates(user_input, graph)

                index0 = json_string.find("```json\n")
                json_string = json_string[index0 + len("```json\n"):] if index0 > 0 else json_string
                index1 = json_string.rfind("}\n```")
                json_string = json_string[:index1+1] if index1 > 0 else json_string

                python_object = json.loads(json_string)    
                print(f"Potential guilty commit is found from git show of commit {commit}. {python_object}")                                               

            except json.JSONDecodeError as e:
                print(f"Failed to decode JSON from git show of commit {commit}: {e}")
            except Exception as e:
                print(f"An unexpected error occurred while processing commit {commit}: {e}")
        
        end = time.time()
        print(f"Time used: {end - start} seconds.")
        # ##################
        # # Check git log by module #
        # ##################

        # # The tmux by default is under pytorch folder!
        # git_log = get_git_log_by_module_in_tmux(blame_range, tmux, match=module_map[python_object["module"]])

        # def generate_prompt_gitlog(python_object: dict) -> str:
        #     env = Environment(loader=FileSystemLoader('prompts'))
        #     template = env.get_template('guilty_commit_gitlog.j2')
        #     # use original test file because we want to check pytorch git log
        #     return template.render(git_log=git_log,
        #                             original_test_file=python_object["original_test_file"],
        #                             original_test_case=python_object["original_test_case"],
        #                             original_test_class=python_object["original_test_class"],
        #                             test_ops=python_object["torch_op"],
        #                             module=python_object["module"],
        #                             dependency=python_object["dependency"],
        #                             error_message=python_object["error_message"])
        
        # prompt = generate_prompt_gitlog(failure_info)
        # user_input = prompt
        # json_string = stream_graph_updates(user_input, graph)
        # print(json_string)

        # python_object = json.loads(json_string.replace("```json\n","").replace("```", ""))

        # if python_object.get("test_case_update") == "false" and python_object.get("related_components_update") == "false" and python_object.get("error_message_related_update") == "false":
        #     print(f"Cannot find the guilty commit from git log.")
        # else:
        #     print(f"Potential guilty commit is found from git log. {python_object}")

    except json.JSONDecodeError as e:
        print(f"Failed to decode JSON: {e}")
