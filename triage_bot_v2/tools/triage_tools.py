from langchain.tools import tool
import subprocess
import time
from utils import run_in_docker


################################
# Tools 
################################
@tool
def do_nothing_tool(test_file: str, test_case: str, container: str) -> str:
    """ Do nothing """
    return "do nothing tool is called"

@tool
def onednn_verbose_tool(test_file: str, test_case: str, workdir: str, container: str, env: str) -> str:
    """Executes gdb catch throw comamnd to capture the trace of a pytest failure."""
    result = run_in_docker(f"sh -c 'cd {workdir} && {env} pytest -v {test_file} -k {test_case}'", container, workdir)
    return "onednn_verbose tool is called: " + result

@tool
def gdb_catch_throw_tool(test_file: str, test_case: str, workdir: str, container: str) -> str:
    """Executes gdb catch throw comamnd to capture the trace of a pytest failure."""
    
    command = f"/bin/bash -c 'source ~/miniforge3/bin/activate pytorch_guilty_commit && \
                source /tools/env.sh && \
                echo $CONDA_DEFAULT_ENV && \
                PYTORCH_TEST_WITH_SLOW=1 gdb -batch -ex \"catch throw\" -ex \"run\" -ex \"bt\" --args python -m pytest -v {test_file} -k {test_case} 2>&1 | tee log.txt ' "
    
    print("Running command:", command)
    result = run_in_docker(command, container, workdir)
    
    
    # Parse log.txt to extract stack frames until _PyEval_EvalFrameDefault
    filtered_lines = []
    for line in result.split('\n'):
        if not line.startswith('#'):
            continue
        line = line.strip()
        if line.startswith('#') and any(c.isdigit() for c in line.split(' ')[0].split('#')[-1]):
            filtered_lines.append(line)
            if '_PyEval_EvalFrameDefault' in line:
                break        
    
    filtered_result = '\n'.join(filtered_lines)

    return "gdb_catch_throw tool is called: " + filtered_result
    
@tool
def instrument_tool(test_file: str, test_case: str, base_test_file: str, original_test_case: str, lineno: int, workdir: str, container: str) -> str:
    """Instrument print in the code and rerun the test to collect debug information."""
    command = f"sh -c '. ~/miniforge3/bin/activate pytorch_guilty_commit && python -m pytest -v {test_file} -k {test_case} --capture=no --tb=native '"
    command = f"/bin/bash -c 'source ~/miniforge3/bin/activate pytorch_guilty_commit && \
                source /tools/env.sh && \
                echo $CONDA_DEFAULT_ENV && \
                PYTORCH_TEST_WITH_SLOW=1 pytest -p call_tracer -v {test_file} -k {test_case} 2>&1 | tee call_tracer.txt ' "
    result = run_in_docker(command, container, workdir)
    
    return "instrument tool is called: " + result

@tool
def inductor_tool(test_file: str, test_case: str, op: str, workdir: str, container: str) -> str:
    """First rerun test with TORCHINDUCTOR_FALLBACK_OPS="failed_op", if passed it is an eager problem, if failed dump triton output_code."""
    try:
        result1 = ""
        # Check if the issue pass with specific op fallback
        if op != 'None':
            command = f"sh -c \". ~/miniforge3/bin/activate pytorch_guilty_commit && TORCHINDUCTOR_FALLBACK_OPS=aten::{op} pytest -vs {test_file} -k {test_case} --capture=no --tb=native \""
                            
            result1 = run_in_docker(command, container, workdir)

        command = f"sh -c \". ~/miniforge3/bin/activate pytorch_guilty_commit && TORCH_LOGS=\"+inductor,+output_code\" TORCHINDUCTOR_DUMP_CODE=readable TORCHINDUCTOR_DUMP_TRITON=1 TORCHINDUCTOR_DEBUG=1  pytest -vs {test_file} -k {test_case} --capture=no --tb=native \""
        result2 = run_in_docker(command, container, workdir)
        return f"inductor tool is called: Result with op fallback:\n{result1}\n\nResult with triton dump:\n{result2}"
    except Exception as e:
        return f"Error during inductor tool execution: {e}"

