import argparse
import json
from jinja2 import Environment, FileSystemLoader
from guilty_commit import git_show_in_tmux, get_blamed_commits_case_update, get_git_log
from common_nodes import stream_graph_updates, classify_depsrag_graph, classify_graph
from prepare import extract_issue_information, get_tmux_sessions, build_pytorch_enviroment, reproduce_issue_in_tmux

parser = argparse.ArgumentParser()
parser.add_argument("--issue", type=str, help="The issue number to process", required=True)
parser.add_argument("--repo", type=str, help="The github repo of the issue", default='pytorch/pytorch', required=False)
parser.add_argument("--token", type=str, help="The github token to the repo", required=True)
parser.add_argument("--tmux", type=str, help="The tmux session to use, if not exists the script will create one", required=False)
parser.add_argument("--build", type=str, help="The build method, source or nightly or existing", default="nightly", required=False)
parser.add_argument("--match", type=str, help="The pattern to match in git log", default="test")
parser.add_argument("--since", type=str, help="The start time for git blame, e.g., '2023-10-01'", required=False)
parser.add_argument("--until", type=str, help="The end time for git blame, e.g., '2023-10-31'", required=False) 
parser.add_argument("--blame_range", type=str, help="The git blame range, e.g., 'commit1..commit2'", required=False)
parser.add_argument("--retest", action="store_true", default=True, help="If set, do retest in tmux session.", required=False)
parser.add_argument("--blame_callee", action="store_true", default=False, help="If set, use call tracer to trace the function calls and git blame callee functions.", required=False)
args = parser.parse_args()

import time
start = time.time()
######## extract issue information ##########
#if args.input is None:
if args.issue is None or args.repo is None or args.token is None:
    print("Please provide --input or --issue, --repo and --token when input is not provided")
    exit(1)

json_string = extract_issue_information(args.repo, int(args.issue), args.token)
issue_information = json.loads(json_string)
issue_information["tmux"] = args.tmux
if args.since is not None:
    issue_information["since"] = args.since
if args.until is not None:
    issue_information["until"] = args.until
# else:
#     with open(args.input, 'r') as f:
#         issue_input = json.load(f)

#         test_file = issue_input["test_file"]
#         original_test_file = issue_input["original_test_file"]
#         test_class = issue_input["test_class"]
#         test_case = issue_input["test_case"]
#         error_message = issue_input["error_message"]
#         since = issue_input.get("since", None)
#         until = issue_input.get("until", None)
#         last_good_commit = issue_input.get("last_good_commit", None)
              
#         def generate_prompt(test_file: str, original_test_file: str, test_class: str, test_case: str, error_message: str, since: str, until: str, last_good_commit: str) -> str:
#             env = Environment(loader=FileSystemLoader('prompts'))
#             template = env.get_template('classification_prompt.j2')
#             return template.render(test_file=test_file, original_test_file=original_test_file, test_class=test_class, test_case=test_case, error_message=error_message, since=since, until=until, last_good_commit=last_good_commit)

#         prompt = generate_prompt(test_file, original_test_file, test_class, test_case, error_message, since, until, last_good_commit)
#         user_input = prompt
#         # collect the failure information
#         graph = classify_depsrag_graph()
#         json_string = stream_graph_updates(user_input, graph)
#         issue_information = json.loads(json_string)
#         issue_information["tmux"] = args.tmux

test_file = issue_information["test_file"]
original_test_file = issue_information["original_test_file"]
test_class = issue_information["test_class"]
test_case = issue_information["test_case"]
error_message = issue_information["error_message"]
#last_good_commit = issue_information["last_good_commit"]
since = issue_information["since"]
until = issue_information["until"]
match = args.match

# information would collect in retest
graph = classify_graph()

######## Prepare the environment ##########
tmux = issue_information["tmux"] = args.tmux
get_tmux_sessions(args.tmux)
#build_pytorch_enviroment(args.build, args.tmux)

issue_information["reproduced"], python_object = reproduce_issue_in_tmux(issue_information, match=match, blame_callee=args.blame_callee) 
if issue_information["reproduced"] == False:
    print("Cannot reproduce the issue, exit the script.")
    exit(1)
else:
    print("The issue is reproduced, start to find the guilty commit.")
    issue_information["error_message"] = python_object["error_message"]
    issue_information["module"] = python_object["module"]
    issue_information["dtype"] = python_object["dtype"]
    issue_information["torch_op"] = python_object["torch_op"]
    issue_information["error_type"] = python_object["error_type"]

######## Pepare the blame range ##########
if args.blame_range is not None:
    blame_range = args.blame_range
else:
    blame_range = f"--since {since}" if since is not None else ""
    blame_range += f" --until {until}" if until is not None else ""

# while True:
#     try:
#         user_input = input("User: ")

#         if user_input == "1":
#             test_file = "test/xpu/nn/test_pooling_xpu.py"
#             original_test_file = "test/nn/test_pooling.py"
#             test_class = "TestPoolingNN" 
#             test_case = "test_adaptive_pooling_avg_nhwc_launch_config_backward"
#             tmux = "test_nn" 
#             error_message ="""
# ____________________________________________________________________________________ TestPoolingNN.test_adaptive_pooling_avg_nhwc_launch_config_backward ____________________________________________________________________________________
# Traceback (most recent call last):
#   File "/home/daisyden/upstream/test_nn_s1_ci/third_party/torch-xpu-ops/test/xpu/../../../../test/nn/test_pooling.py", line 317, in test_adaptive_pooling_avg_nhwc_launch_config_backward
#     input = torch.randint(
#   File "/home/daisyden/miniforge3/envs/test_nn_s1/lib/python3.10/site-packages/torch/cuda/__init__.py", line 403, in _lazy_init
#     raise AssertionError("Torch not compiled with CUDA enabled")
# AssertionError: Torch not compiled with CUDA enabled

# To execute this test, run the following from the base repo dir:
#     python ../../test/nn/test_pooling.py TestPoolingNN.test_adaptive_pooling_avg_nhwc_launch_config_backward
# """
#         elif user_input == "2":
#             test_file = "test/inductor/test_compile_subprocess.py"
#             original_test_file = "test/inductor/test_compile_subprocess.py"
#             test_class = "GPUTests"
#             test_case = "test_conv1d_with_permute_xpu"
#             tmux = "guilty"
#             error_message = """
# python inductor/test_compile_subprocess.py GPUTests.test_conv1d_with_permute_xpu
# /home/daisyden/miniforge3/envs/guilty/lib/python3.10/site-packages/hypothesis/entry_points.py:23: UserWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html. The pkg_res
# ources package is slated for removal as early as 2025-11-30. Refrain from using this package or pin to Setuptools<81.
#   import pkg_resources
# W1001 13:51:20.364000 1616416 site-packages/torch/_inductor/compile_fx_ext.py:491] [0/0] Unable to pickle input graph or example inputs
# W1001 13:51:20.364000 1616416 site-packages/torch/_inductor/compile_fx_ext.py:491] [0/0] Traceback (most recent call last):
# W1001 13:51:20.364000 1616416 site-packages/torch/_inductor/compile_fx_ext.py:491] [0/0]   File "/home/daisyden/miniforge3/envs/guilty/lib/python3.10/site-packages/torch/_inductor/compile_fx_ext.py", line 484, in serialize
# _compile
# W1001 13:51:20.364000 1616416 site-packages/torch/_inductor/compile_fx_ext.py:491] [0/0]     ).serialize()
# W1001 13:51:20.364000 1616416 site-packages/torch/_inductor/compile_fx_ext.py:491] [0/0]   File "/home/daisyden/miniforge3/envs/guilty/lib/python3.10/site-packages/torch/_inductor/compile_fx_ext.py", line 210, in serialize
# W1001 13:51:20.364000 1616416 site-packages/torch/_inductor/compile_fx_ext.py:491] [0/0]     return _WireProtocolPickledInput(GraphPickler.dumps(self))
# W1001 13:51:20.364000 1616416 site-packages/torch/_inductor/compile_fx_ext.py:491] [0/0]   File "/home/daisyden/miniforge3/envs/guilty/lib/python3.10/site-packages/torch/fx/_graph_pickler.py", line 124, in dumps
# W1001 13:51:20.364000 1616416 site-packages/torch/_inductor/compile_fx_ext.py:491] [0/0]     pickler.dump(obj)
# W1001 13:51:20.364000 1616416 site-packages/torch/_inductor/compile_fx_ext.py:491] [0/0] AttributeError: Can't pickle local object 'CommonTemplate.test_conv1d_with_permute.<locals>.ConvModel'
# inline_call []
# stats [('calls_captured', 2), ('unique_graphs', 1)]
# inductor [('triton_bundler_save_kernel', 14), ('async_compile_cache_miss', 2), ('benchmarking.TritonBenchmarker.benchmark_gpu', 2), ('fxgraph_cache_miss', 1), ('extern_calls', 1), ('async_compile_cache_hit', 1)]
# aot_autograd [('total', 1), ('autograd_cache_miss', 1), ('ok', 1)]
# graph_break []
# F
# ======================================================================
# FAIL: test_conv1d_with_permute_xpu (__main__.GPUTests)
# ----------------------------------------------------------------------
# Traceback (most recent call last):
#   File "/home/daisyden/miniforge3/envs/guilty/lib/python3.10/site-packages/torch/testing/_internal/common_utils.py", line 3213, in wrapper
#     method(*args, **kwargs)
#   File "/home/daisyden/upstream/pytorch/test/inductor/test_torchinductor.py", line 14257, in new_test
#     return value(self)
#   File "/home/daisyden/upstream/pytorch/test/inductor/test_torchinductor.py", line 4654, in test_conv1d_with_permute
#     self.common(ConvModel(), (torch.randn([32, 100, 1]),), check_lowp=False)
#   File "/home/daisyden/miniforge3/envs/guilty/lib/python3.10/contextlib.py", line 79, in inner
#     return func(*args, **kwds)
#   File "/home/daisyden/upstream/pytorch/test/inductor/test_torchinductor.py", line 687, in check_model_gpu
#     check_model(
#   File "/home/daisyden/upstream/pytorch/test/inductor/test_torchinductor.py", line 561, in check_model
#     assert_equal_fn(
#   File "/home/daisyden/miniforge3/envs/guilty/lib/python3.10/site-packages/torch/_dynamo/test_case.py", line 111, in assertEqual
#     return super().assertEqual(x, y, *args, **kwargs)
#   File "/home/daisyden/miniforge3/envs/guilty/lib/python3.10/site-packages/torch/testing/_internal/common_utils.py", line 4168, in assertEqual
#     raise error_metas.pop()[0].to_error(  # type: ignore[index]
# AssertionError: Tensor-likes are not close!

# Mismatched elements: 204479 / 204800 (99.8%)
# Greatest absolute difference: 5.426783084869385 at index (8, 35, 56) (up to 1e-05 allowed)
# Greatest relative difference: 206132.828125 at index (8, 27, 85) (up to 1.3e-06 allowed)

# To execute this test, run the following from the base repo dir:
#     python test/inductor/test_compile_subprocess.py GPUTests.test_conv1d_with_permute_xpu

# This message can be suppressed by setting PYTORCH_PRINT_REPRO_ON_FAILURE=0

# ----------------------------------------------------------------------
# Ran 1 test in 16.714s

# FAILED (failures=1)


# """  


if args.retest == False or issue_information.get("reproduced", False) == True:
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

        git_log_output = get_git_log(issue_information, blame_range, match=match)
        def generate_prompt_git_log(git_log: str, python_object: dict) -> str:
            env = Environment(loader=FileSystemLoader('prompts'))
            template = env.get_template('guilty_commit_gitlog.j2')
            # use original test file because we want to check pytorch git log
            return template.render(git_log=git_log,
                                  test_file=python_object["test_file"],
                                  test_case=python_object["test_case"],
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
