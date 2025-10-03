import argparse

import json

from langgraph.graph import StateGraph, START, END
from jinja2 import Environment, FileSystemLoader
from common import git_show_in_tmux, get_git_log_by_module_in_tmux, get_blamed_commits_case_update
from common_nodes import State, classify, depsRAG, stream_graph_updates

parser = argparse.ArgumentParser()
parser.add_argument("--last-good-commit", type=str, help="Last known good commit (SHA or ref)")
parser.add_argument("--since", type=str, help="Blame since a date (e.g., '2023-01-01') or '4 days ago'")
parser.add_argument("--until", type=str, help="Blame until a date (e.g., '2023-01-01') or '4 days ago'")
parser.add_argument("--match", type=str, help="The pattern to match in git log", default="test")
args = parser.parse_args()
last_good_commit = args.last_good_commit
since = args.since
until = args.until
match = args.match

if since is not None or until is not None:
    blame_range = f"--since {since}" if since is not None else ""
    blame_range += f" --until {until}" if until is not None else ""
elif last_good_commit is not None:
    blame_range = f"{last_good_commit}..HEAD"
else:
    blame_range = ""

module_map = {
    "torch operation": "aten",
    "inductor": "torch/csrc/inductor",
    "distributed": "torch/csrc/distributed",
    "runtime": "torch/csrc/*.*",
}

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
        elif user_input == "2":
            test_file = "test/inductor/test_compile_subprocess.py"
            original_test_file = "test/inductor/test_compile_subprocess.py"
            test_class = "GPUTests"
            test_case = "test_conv1d_with_permute_xpu"
            tmux = "guilty"
            error_message = """
python inductor/test_compile_subprocess.py GPUTests.test_conv1d_with_permute_xpu
/home/daisyden/miniforge3/envs/guilty/lib/python3.10/site-packages/hypothesis/entry_points.py:23: UserWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html. The pkg_res
ources package is slated for removal as early as 2025-11-30. Refrain from using this package or pin to Setuptools<81.
  import pkg_resources
W1001 13:51:20.364000 1616416 site-packages/torch/_inductor/compile_fx_ext.py:491] [0/0] Unable to pickle input graph or example inputs
W1001 13:51:20.364000 1616416 site-packages/torch/_inductor/compile_fx_ext.py:491] [0/0] Traceback (most recent call last):
W1001 13:51:20.364000 1616416 site-packages/torch/_inductor/compile_fx_ext.py:491] [0/0]   File "/home/daisyden/miniforge3/envs/guilty/lib/python3.10/site-packages/torch/_inductor/compile_fx_ext.py", line 484, in serialize
_compile
W1001 13:51:20.364000 1616416 site-packages/torch/_inductor/compile_fx_ext.py:491] [0/0]     ).serialize()
W1001 13:51:20.364000 1616416 site-packages/torch/_inductor/compile_fx_ext.py:491] [0/0]   File "/home/daisyden/miniforge3/envs/guilty/lib/python3.10/site-packages/torch/_inductor/compile_fx_ext.py", line 210, in serialize
W1001 13:51:20.364000 1616416 site-packages/torch/_inductor/compile_fx_ext.py:491] [0/0]     return _WireProtocolPickledInput(GraphPickler.dumps(self))
W1001 13:51:20.364000 1616416 site-packages/torch/_inductor/compile_fx_ext.py:491] [0/0]   File "/home/daisyden/miniforge3/envs/guilty/lib/python3.10/site-packages/torch/fx/_graph_pickler.py", line 124, in dumps
W1001 13:51:20.364000 1616416 site-packages/torch/_inductor/compile_fx_ext.py:491] [0/0]     pickler.dump(obj)
W1001 13:51:20.364000 1616416 site-packages/torch/_inductor/compile_fx_ext.py:491] [0/0] AttributeError: Can't pickle local object 'CommonTemplate.test_conv1d_with_permute.<locals>.ConvModel'
inline_call []
stats [('calls_captured', 2), ('unique_graphs', 1)]
inductor [('triton_bundler_save_kernel', 14), ('async_compile_cache_miss', 2), ('benchmarking.TritonBenchmarker.benchmark_gpu', 2), ('fxgraph_cache_miss', 1), ('extern_calls', 1), ('async_compile_cache_hit', 1)]
aot_autograd [('total', 1), ('autograd_cache_miss', 1), ('ok', 1)]
graph_break []
F
======================================================================
FAIL: test_conv1d_with_permute_xpu (__main__.GPUTests)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/home/daisyden/miniforge3/envs/guilty/lib/python3.10/site-packages/torch/testing/_internal/common_utils.py", line 3213, in wrapper
    method(*args, **kwargs)
  File "/home/daisyden/upstream/pytorch/test/inductor/test_torchinductor.py", line 14257, in new_test
    return value(self)
  File "/home/daisyden/upstream/pytorch/test/inductor/test_torchinductor.py", line 4654, in test_conv1d_with_permute
    self.common(ConvModel(), (torch.randn([32, 100, 1]),), check_lowp=False)
  File "/home/daisyden/miniforge3/envs/guilty/lib/python3.10/contextlib.py", line 79, in inner
    return func(*args, **kwds)
  File "/home/daisyden/upstream/pytorch/test/inductor/test_torchinductor.py", line 687, in check_model_gpu
    check_model(
  File "/home/daisyden/upstream/pytorch/test/inductor/test_torchinductor.py", line 561, in check_model
    assert_equal_fn(
  File "/home/daisyden/miniforge3/envs/guilty/lib/python3.10/site-packages/torch/_dynamo/test_case.py", line 111, in assertEqual
    return super().assertEqual(x, y, *args, **kwargs)
  File "/home/daisyden/miniforge3/envs/guilty/lib/python3.10/site-packages/torch/testing/_internal/common_utils.py", line 4168, in assertEqual
    raise error_metas.pop()[0].to_error(  # type: ignore[index]
AssertionError: Tensor-likes are not close!

Mismatched elements: 204479 / 204800 (99.8%)
Greatest absolute difference: 5.426783084869385 at index (8, 35, 56) (up to 1e-05 allowed)
Greatest relative difference: 206132.828125 at index (8, 27, 85) (up to 1.3e-06 allowed)

To execute this test, run the following from the base repo dir:
    python test/inductor/test_compile_subprocess.py GPUTests.test_conv1d_with_permute_xpu

This message can be suppressed by setting PYTORCH_PRINT_REPRO_ON_FAILURE=0

----------------------------------------------------------------------
Ran 1 test in 16.714s

FAILED (failures=1)


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
                
                ###############################
                # Collect realted commit:
                # 1. commit updated the test function
                # 2. commit updated the callee functions of the test function
                # 3. commit updated the op_db definition of the test ops
                # 4. commit updated the related module
                # 5. commit updated the related dependency component
                ###############################
             

                blamed_commits, called_functions = get_blamed_commits_case_update(tmux, failure_info, blame_range, match)
                called_functions = list(set([func[0] for func in called_functions]))

                import pdb
                pdb.set_trace()

                print(f"Blamed commits: {blamed_commits}.")
                for commit in blamed_commits:
                    print(f"Check blamed commit: {commit}.")

                    try:
                        show_results = git_show_in_tmux(commit, tmux)
                        
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
                        
                        prompt = generate_prompt_gitshow(commit_sha=commit, 
                                                  git_show=show_results,
                                                  python_object=failure_info,
                                                  called_functions=",".join(called_functions))
                        user_input = prompt
                        json_string = stream_graph_updates(user_input, graph)

                        index0 = json_string.find("```json\n")
                        json_string = json_string[index0 + len("```json\n"):]
                        index1 = json_string.rfind("}\n```")

                        json_string = json_string[:index1+1]                        

                        python_object = json.loads(json_string)                                                   

                        if python_object.get("test_case_update") == True:
                            print(f"Guilty commit is found from git show: {commit}. {python_object}")
                            break
                    except json.JSONDecodeError as e:
                        print(f"Failed to decode JSON from git show of commit {commit}: {e}")
                    except Exception as e:
                        print(f"An unexpected error occurred while processing commit {commit}: {e}")
                

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
    except Exception as e:
        # Catch any other unexpected exceptions
        print(f"An unexpected error occurred: {e}")