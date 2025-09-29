import argparse

import json

from langgraph.graph import StateGraph, START, END
from jinja2 import Environment, FileSystemLoader
from common import git_show_in_tmux, get_git_log_in_tmux, get_blamed_commits_case_update, get_blamed_commits_module_update
from common_nodes import State, classify, depsRAG, stream_graph_updates

parser = argparse.ArgumentParser()
parser.add_argument("--last-good-commit", type=str, help="Last known good commit (SHA or ref)")
args = parser.parse_args()
last_good_commit = args.last_good_commit


graph_builder = StateGraph(State)
graph_builder.add_node("classify", classify)
graph_builder.add_node("depsRAG", depsRAG)
graph_builder.add_edge(START, "classify")
graph_builder.add_edge("classify", "depsRAG")
graph_builder.add_edge("depsRAG", END)
graph = graph_builder.compile()

while True:
    try:
        user_input = input("User: ")

        if user_input == "1":
            test_file = "test/xpu/nn/test_pooling_xpu.py"
            original_test_file = "test/nn/test_pooling.py"
            test_class = "TestPoolingNN" 
            test_case = "test_adaptive_pooling_avg_nhwc_launch_config_backward"
            tmux = "test_nn" 
            error_message ="""
____________________________________________________________________________________ TestPoolingNN.test_adaptive_pooling_avg_nhwc_launch_config_backward ____________________________________________________________________________________
Traceback (most recent call last):
  File "/home/daisyden/upstream/test_nn_s1_ci/third_party/torch-xpu-ops/test/xpu/../../../../test/nn/test_pooling.py", line 317, in test_adaptive_pooling_avg_nhwc_launch_config_backward
    input = torch.randint(
  File "/home/daisyden/miniforge3/envs/test_nn_s1/lib/python3.10/site-packages/torch/cuda/__init__.py", line 403, in _lazy_init
    raise AssertionError("Torch not compiled with CUDA enabled")
AssertionError: Torch not compiled with CUDA enabled

To execute this test, run the following from the base repo dir:
    python ../../test/nn/test_pooling.py TestPoolingNN.test_adaptive_pooling_avg_nhwc_launch_config_backward
"""
        
        def generate_prompt(test_file: str, original_test_file: str, test_class: str, test_case: str, error_message: str, tmux: str) -> str:
            env = Environment(loader=FileSystemLoader('prompts'))
            template = env.get_template('classification_prompt.j2')
            return template.render(test_file=test_file, original_test_file=original_test_file, test_class=test_class, test_case=test_case, error_message=error_message, tmux=tmux)

        prompt = generate_prompt(test_file, original_test_file, test_class, test_case, error_message, tmux)
        user_input = prompt
        # collect the failure information
        json_string = stream_graph_updates(user_input, graph)

        if len(json_string) > 0:
            try:
                failure_info = json.loads(json_string.replace("```json\n","").replace("```", ""))

                ##################
                # Check git log  #
                ##################

                # The tmux by default is under pytorch folder!
                git_log = get_git_log_in_tmux(args.last_good_commit, tmux)

                def generate_prompt_gitlog(python_object: dict) -> str:
                    env = Environment(loader=FileSystemLoader('prompts'))
                    template = env.get_template('guilty_commit_gitlog.j2')
                    # use original test file because we want to check pytorch git log
                    return template.render(git_log=git_log,
                                            original_test_file=python_object["original_test_file"],
                                            original_test_case=python_object["original_test_case"],
                                            original_test_class=python_object["original_test_class"],
                                            test_ops=python_object["torch_op"],
                                            module=python_object["module"],
                                            dependency=python_object["dependency"],
                                            error_message=python_object["error_message"])
                
                # prompt = generate_prompt_gitlog(failure_info)
                # user_input = prompt
                # json_string = stream_graph_updates(user_input, graph)
                # print(json_string)

                # python_object = json.loads(json_string.replace("```json\n","").replace("```", ""))

                # if python_object.get("test_case_update") == "false" and python_object.get("related_components_update") == "false" and python_object.get("error_message_related_update") == "false":
                #     print(f"Cannot find the guilty commit from git log.")
                # else:
                #     print(f"Potential guilty commit is found from git log. {python_object}")
                
                ###############################
                # Collect realted commit:
                # 1. commit updated the test function
                # 2. commit updated the callee functions of the test function
                # 3. commit updated the op_db definition of the test ops
                # 4. commit updated the related module
                # 5. commit updated the related dependency component
                ###############################
             

                blamed_commits, called_functions = get_blamed_commits_case_update(tmux, failure_info, last_good_commit)

                for commit in blamed_commits:
                    print(f"Check blamed commit: {commit}.")

                    show_results = git_show_in_tmux(commit, tmux)
                    import pdb
                    pdb.set_trace()
                    
                    def generate_prompt_gitshow(commit_sha: str, git_show: str, python_object: dict, called_functions) -> str:
                        env = Environment(loader=FileSystemLoader('prompts'))
                        template = env.get_template('guilty_commit_gitshow_newcase.j2')
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
                    called_functions = [func[0] for func in called_functions]
                    prompt = generate_prompt_gitshow(commit_sha=commit, 
                                               git_show=show_results,
                                               python_object=failure_info,
                                               called_functions=called_functions)
                    user_input = prompt
                    json_string = stream_graph_updates(user_input, graph)

                    import pdb
                    pdb.set_trace()

                    index = json_string.find("```json\n")

                    python_object = json.loads(json_string[index + len("```json\n"):].replace("```", ""))

                    if python_object.get("test_case_update") == "true":
                        print(f"Guilty commit is found from git show: {commit}. {python_object}")
                        break

            except json.JSONDecodeError as e:
                print(f"Failed to decode JSON: {e}")
    except Exception as e:
        # Catch any other unexpected exceptions
        print(f"An unexpected error occurred: {e}")