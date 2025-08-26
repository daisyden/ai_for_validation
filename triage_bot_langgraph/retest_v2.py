from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from vllm_service import llm
from langgraph.prebuilt import ToolNode
from langchain_community.tools.shell.tool import ShellTool
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
import subprocess
from jinja2 import Environment, FileSystemLoader
from langchain_core.messages import AIMessage
import json
import time

from datetime import datetime
# Getting the current date and time
dt = datetime.now()
# getting the timestamp
ts = datetime.timestamp(dt)
id = "daisyden"

import os
os.mkdir(f"/tmp/{ts}")

################################
# State
################################
class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]
    execution_path: list[str]        

def get_history(state: State, node: str):
    executed_nodes = state.get('execution_path')
    last_node = executed_nodes[-1] if executed_nodes else "START"
    
    print(f"=== Node {node} executed after {last_node} ====\n")
    return executed_nodes, last_node 



################################
# Tools 
################################
@tool
def do_nothing_tool(test_file: str, test_case: str, tmux: str) -> str:
    """ Do nothing """
    return "do nothing tool is called"

@tool
def verbose_tool(test_file: str, test_case: str, tmux: str, env: str) -> str:
    """Executes gdb catch throw comamnd to capture the trace of a pytest failure."""

    command = f"tmux send-keys -R -t {tmux}  \"tmux clear-history; clear ; {env} pytest -v {test_file} -k {test_case} 2>&1|tee /tmp/{ts}/verbose.log ; tmux wait -S my_lock \" C-m; tmux wait my_lock; tmux capture-pane -S - -E - -p -t {tmux}"
    try:
        result = subprocess.run(
            command,
            shell=True,  # Set to True to allow shell features like piping
            check=True,  # Raise an exception for non-zero exit codes
            text=True,   # Decode stdout/stderr as text
            capture_output=True # Capture stdout and stderr
        )
        return "verbose_tool output: \n" + result.stdout.strip().split("test session starts")[-1][0:min(50000, len(result.stdout))] if result.stdout else "Command executed successfully."
    except subprocess.CalledProcessError as e:
        return f"Error executing command: {e.stderr.strip()}"
    except FileNotFoundError:
        return f"Error: Command '{command.split()[0]}' not found."

@tool
def gdb_catch_throw_tool(test_file: str, test_case: str, tmux: str) -> str:
    """Executes gdb catch throw comamnd to capture the trace of a pytest failure."""
   
    command = f"tmux send-keys -R -t {tmux}  \"tmux clear-history; clear ; PYTORCH_ENABLE_XPU_FALLBACK=1 PYTORCH_TEST_WITH_SLOW=1 gdb -batch -ex \'catch throw\' -ex \"run\" -ex \"bt\" --args python -m pytest -v {test_file} -k {test_case} 2>&1|tee /tmp/{ts}/gdb.log ; tmux wait-for -S my_lock \" C-m; tmux wait-for my_lock; tmux capture-pane -S - -E - -p -t {tmux}"
    try:
        result = subprocess.run(
            command,
            shell=True,  # Set to True to allow shell features like piping
            check=True,  # Raise an exception for non-zero exit codes
            text=True,   # Decode stdout/stderr as text
            capture_output=True # Capture stdout and stderr
        )
        return "gdb_catch_throw_tool output: \n" + result.stdout.strip().split("test session starts")[-1][0:min(50000, len(result.stdout))] if result.stdout else "Command executed successfully."
    except subprocess.CalledProcessError as e:
        return f"Error executing command: {e.stderr.strip()}"
    except FileNotFoundError:
        return f"Error: Command '{command.split()[0]}' not found."

@tool
def retest_reproduce_tool(test_file: str, test_case: str, base_test_file: str, original_test_case: str, lineno: int, tmux: str) -> str:
    """Reprudce the failure in tmux enviroment"""    
    command = f"tmux send-keys -R -t {tmux}  \"tmux clear-history; clear ; tmux wait -S my_lock \" C-m ; tmux wait my_lock ; tmux capture-pane -S - -E - -p -t {tmux}"
    print(command)
    try:
        result = subprocess.run(
            command,
            shell=True,  # Set to True to allow shell features like piping
            check=True,  # Raise an exception for non-zero exit codes
            text=True,   # Decode stdout/stderr as text
            capture_output=True # Capture stdout and stderr
        )
        return "debug_prints_tool output: \n" + result.stdout.strip().split("test session starts")[-1][0:min(50000, len(result.stdout))] if result.stdout else "Command executed successfully."
    except subprocess.CalledProcessError as e:
        return f"Error executing command: {e.stderr.strip()}"
    except FileNotFoundError:
        return f"Error: Command '{command.split()[0]}' not found."
    
@tool
def retest_debug_prints_tool(test_file: str, test_case: str, base_test_file: str, original_test_case: str, lineno: int, tmux: str) -> str:
    """Instrument print in the code and rerun the test to collect debug information."""
    
    command = f"tmux send-keys -R -t {tmux}  \"tmux clear-history; clear ; cp {base_test_file} {base_test_file}.saved ; python /tmp/visit_ast.py {base_test_file} {original_test_case} {lineno} >{base_test_file}.updated; cp {base_test_file}.updated {base_test_file} ; tmux wait -S my_lock \" C-m; tmux wait my_lock; tmux capture-pane -S - -E - -p -t {tmux}"
    print(command)
    try:
        result = subprocess.run(
            command,
            shell=True,  # Set to True to allow shell features like piping
            check=True,  # Raise an exception for non-zero exit codes
            text=True,   # Decode stdout/stderr as text
            capture_output=True # Capture stdout and stderr
        )
        time.sleep(20)
    except subprocess.CalledProcessError as e:
        return f"Error executing command: {e.stderr.strip()}"
    except FileNotFoundError:
        return f"Error: Command '{command.split()[0]}' not found."
    
    command = f"tmux send-keys -R -t {tmux}  \"tmux clear-history; clear ; PYTORCH_ENABLE_XPU_FALLBACK=1 PYTORCH_TEST_WITH_SLOW=1 pytest --show-capture=no -vs {test_file} -k {test_case} --capture=no --tb=native 2>&1|tee /tmp/{ts}/dbg.log ; mv {base_test_file}.saved {base_test_file} ; tmux wait -S my_lock \" C-m ; tmux wait my_lock ; tmux capture-pane -S - -E - -p -t {tmux}"
    print(command)
    try:
        result = subprocess.run(
            command,
            shell=True,  # Set to True to allow shell features like piping
            check=True,  # Raise an exception for non-zero exit codes
            text=True,   # Decode stdout/stderr as text
            capture_output=True # Capture stdout and stderr
        )
        return "debug_prints_tool output: \n" + result.stdout.strip().split("test session starts")[-1][0:min(50000, len(result.stdout))] if result.stdout else "Command executed successfully."
    except subprocess.CalledProcessError as e:
        return f"Error executing command: {e.stderr.strip()}"
    except FileNotFoundError:
        return f"Error: Command '{command.split()[0]}' not found."


@tool
def retest_inductor_tool(test_file: str, test_case: str, op: str, tmux: str) -> str:
    """First rerun test with TORCHINDUCTOR_FALLBACK_OPS="failed_op", if passed it is an eager problem, if failed dump triton output_code."""
    try:
        result1 = ""
        # Check if the issue pass with specific op fallback
        if op != 'None':
            command = f"tmux send-keys -R -t {tmux}  \"tmux clear-history; clear ; rm -rf /tmp/torchinductor_{id}; rm -rf ./torch_compile_debug ; TORCHINDUCTOR_FALLBACK_OPS=\"aten::{op}\" pytest -vs {test_file} -k {test_case} --capture=no --tb=native 2>&1|tee /tmp/{ts}/eager.log ; tmux wait -S my_lock \" C-m ; tmux wait my_lock ; tmux capture-pane -S - -E - -p -t {tmux}"
                            
            result = subprocess.run(
                command,
                shell=True,  # Set to True to allow shell features like piping
                check=True,  # Raise an exception for non-zero exit codes
                text=True,   # Decode stdout/stderr as text
                capture_output=True # Capture stdout and stderr
            )

            result1 = f"\n### Result of {command}: \n" + result.stdout.strip().split("test session starts")[-1][0:min(50000, len(result.stdout))]
        # "+all", TORCH_COMPILE_DEBUG=1
        command = f"tmux send-keys -R -t {tmux}  \"tmux clear-history; clear ; rm -rf /tmp/torchinductor_{id}; rm -rf ./torch_compile_debug ; TORCH_LOGS=\"+inductor,+output_code\" TORCHINDUCTOR_DUMP_CODE=readable TORCHINDUCTOR_DUMP_TRITON=1 TORCHINDUCTOR_DEBUG=1  pytest -vs {test_file} -k {test_case} --capture=no --tb=native 2>&1|tee /tmp/{ts}/inductor.log ; cp -rf ./torch_compile_debug /tmp/{ts}/. ; mv /tmp/torchinductor_{id} /tmp/{ts}/. ; echo \"find torch_compile_debug\" ; find torch_compile_debug ; tmux wait -S my_lock \" C-m ; tmux wait my_lock ; tmux capture-pane -S - -E - -p -t {tmux}"
        try:
            result = subprocess.run(
                command,
                shell=True,  # Set to True to allow shell features like piping
                check=True,  # Raise an exception for non-zero exit codes
                text=True,   # Decode stdout/stderr as text
                capture_output=True # Capture stdout and stderr
            )
            result = f"\n### Result of {command}: \n" + result.stdout.strip().split("test session starts")[-1][0:min(50000, len(result.stdout))] if result.stdout else "Command executed successfully."

            # output_code = []
            # output_code_content = ""
            # output_code_bwd_content = ""
            # with open("/tmp/{ts}/torch_compile_debug/run_*/torchinductor/*forward*/output_code.py", 'r') as f:
            #     output_code = f.readlines()
            #     output_code_content = "".join(output_code)
            # output_code_bwd = []
            # with open("/tmp/{ts}/torch_compile_debug/run_*/torchinductor/*backward*/output_code.py", 'r') as f:
            #     output_code_bwd = f.readlines()
            #     output_code_bwd_content = "".join(output_code)

            return f"Detailed inductor logs are in /tmp/{ts}/torch_compile_debug and /tmp/{ts}/torchinductor_{id}. \n {result1} \n {result} \n"
            
        except subprocess.CalledProcessError as e:
            return f"Error executing command: {e.stderr.strip()}"
        except FileNotFoundError:
            return f"Error: Command '{command.split()[0]}' not found."
    except subprocess.CalledProcessError as e:
        return f"Error executing command: {e.stderr.strip()}"
    except FileNotFoundError:
        return f"Error: Command '{command.split()[0]}' not found."


retest_errtype_tools = [gdb_catch_throw_tool, do_nothing_tool]
llm_with_errortype_tools = llm.bind_tools(retest_errtype_tools, tool_choice="auto")

retest_deps_tools = [verbose_tool, do_nothing_tool]
llm_with_dep_stools = llm.bind_tools(retest_deps_tools, tool_choice="auto")

llm_with_reprtools = llm.bind_tools([retest_reproduce_tool,])
llm_with_dbgprint_tools = llm.bind_tools([retest_debug_prints_tool,])

retest_module_tools = [retest_inductor_tool, do_nothing_tool]
llm_with_module_tools = llm.bind_tools(retest_module_tools, tool_choice="auto")
################################
# Nodes 
################################

def classify(state: State):
    executed_nodes, last_node = get_history(state, "classify")

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that classify pytest faiures."),
        ("human", "{text}")
    ])
    chain = {"text": RunnablePassthrough()} | prompt | llm
    return {"messages": chain.invoke(state["messages"]),
            "execution_path": state.get("execution_path", []) + ["classify"]
    }

def call_reproduce_tool(state: State):
    executed_nodes, last_node = get_history(state, "call_reproduce_tool")
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful debugging assistant that can reproduce pytest faiures. Please rerpoduce the failure in tmux enviroment and check whether the error message is the same. 
        If the same, return 'Reproduced' only, otherwise return 'CannotReproduce' only.   
        """),
        ("human", "{text}")
    ])

    chain = {"text": RunnablePassthrough()} | prompt | llm_with_reprtools
 
    return {"messages": chain.invoke(state["messages"][-1]),
            "execution_path": state.get("execution_path", []) + ["call_reproduce_tool"]
    }

def call_debug_prints_tool(state: State):
    executed_nodes, last_node = get_history(state, "call_debug_prints_tool")
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful debugging assistant that can instrument prints in test cases and rerun test.    
        """),
        ("human", "{text}")
    ])

    chain = {"text": RunnablePassthrough()} | prompt | llm_with_dbgprint_tools
 
    return {"messages": chain.invoke(state["messages"][-1]),
            "execution_path": state.get("execution_path", []) + ["call_debug_prints_tool"]
    }

def call_eager_comp_tool(state: State):
    executed_nodes, last_node = get_history(state, "call_eager_comp_tool")
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful debugging assistant that can run pytorch inductor UT with the failed op fallback to CPU.    
        """),
        ("human", "{text}")
    ])

    chain = {"text": RunnablePassthrough()} | prompt | llm_with_eager_comp_tools
 
    return {"messages": chain.invoke(state["messages"][-1]),
            "execution_path": state.get("execution_path", []) + ["call_eager_comp_tool"]
    }

def call_inductor_dump_tool(state: State):
    executed_nodes, last_node = get_history(state, "call_inductor_dump_tool")
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful debugging assistant that can run pytorch inductor UT with output_code.py dump.    
        """),
        ("human", "{text}")
    ])

    chain = {"text": RunnablePassthrough()} | prompt | llm_with_eager_comp_tools
 
    return {"messages": chain.invoke(state["messages"][-1]),
            "execution_path": state.get("execution_path", []) + ["llm_with_inductor_dump_tools"]
    }

################################
# RAG node
###############################
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
loader = CSVLoader("rag_doc/ops_oneDNN.csv")
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# Create vector store
vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
retriever = vectorstore.as_retriever()

# Define prompt template
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# Define the chain
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

def depsRAG(
    state: State,
) -> State:

    executed_nodes, last_node = get_history(state, "depsRAG")

    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")

    json_string = ai_message.content
    python_object = json.loads(json_string.replace("```json\n","").replace("```", ""))
    
    if python_object['torch_op'] != None:
        question = f"Please get the depedency library or tool for the torch-xpu-ops op {python_object['torch_op']} based on the knowledge of the context. Only return the final answer and without any explainations.If no dependency library is detected, return 'sycl'" 
        answer = rag_chain.invoke(question)
    else:
        answer = "sycl"

    python_object["dependency"] = answer
    json_string = json.dumps(python_object)
    return {"messages": AIMessage(content=json_string),
            "execution_path": state.get("execution_path", []) + ["depRAG"]
    }

def triage_start(
    state: State,
):
    executed_nodes, last_node = get_history(state, "triage_start")
    return {"messages": state.get("messages", []),
            "execution_path": state.get("execution_path", []) + ["triage_start"]
    }
  

def check_for_dependency(state: State):
    executed_nodes, last_node = get_history(state, "check_for_dependency")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that rerun the failed pytest test case according to torch ops, rerun with \"DNNL_VERBOSE=1\" for oneDNN dependent ops or \"MKL_VERBOSE=1\" for MKL dependent ops in tmux session, for other dependency do nothing."),
        ("human", "{text}")
    ])
    chain = {"text": RunnablePassthrough()} | prompt | llm_with_dep_stools
    return {"messages": chain.invoke(state["messages"][-1]),
            "execution_path": state.get("execution_path", []) + ["check_for_dependency"]
    }

def check_for_error_type(state: State): 
    executed_nodes, last_node = get_history(state, "check_for_error_type")
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an intelligent test debugging assistant specialized in automatically rerunning failed pytest cases based on the error type. 
         Follow these precise rules:
:
         1. For RuntimeError failures: Rerun in GDB debug mode to capture stack traces by tool calling. 
         2. For other error_types, do nothing.
         """
         ),
        ("human", "{text}")
    ])
    chain = {"text": RunnablePassthrough()} | prompt | llm_with_errortype_tools
    return {"messages": chain.invoke(state["messages"][-1]),
            "execution_path": state.get("execution_path", []) + ["check_for_error_type"]
    }

def check_debug_trace(state: State):
    executed_nodes, last_node = get_history(state, "check_debug_trace")
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
You are an advanced debugging assistant specialized in PyTorch error analysis. Please do the following and output a beautified json with keys "code_execution_trace",  "error_root_cuase", and "reproduce_script".

1. CODE TRACE ANALYSIS:
   - Extract and list all relevant code execution paths from the log and the backtrace of the original error message, please avoid to use the new backtrace. 
   - Format as: `<code snippet> [file:line_number]`
   - Include all variable assignments and function calls leading to the error

2. ERROR ROOT CAUSE:
   - Identify the exact operation causing failure
   - Determine the problematic tensor characteristics:
     * Shape
     * Data type
     * Device (CPU/XPU/CUDA)
     * Values (if special values like inf/nan are involved)
   - Explain why the operation fails with these inputs

3. ERROR REPRODUCTION SCRIPT:
   - Create a minimal Python script that reproduces the error
   - Mimic the input data shape and data distributions in the exeuction trace of the log
   - For XPU-related errors:
     * Use `torch.xpu` device when available
     * Include proper device availability checks
     * Maintain XPU-specific syntax correctness
   - If the module is inductor, write the script with inductor enabled on the related ops.
   - Include all necessary imports
   - Add clear comments explaining each step
   - Ensure the script can run without syntax errors
   - If the case failed in output comparision, also include expected vs actual output comparison in the small script and ensure the expected and actual data are on the same device

4. OUTPUT REQUIREMENTS:
   - Present information in this exact structure:
     a) Code Execution Trace
     b) Error Root Cause Analysis
     c) Reproduction Script
   - Use clear section headers with border lines
   - Maintain proper code formatting
   - Keep all technical details accurate

5. SPECIAL CONSIDERATIONS:
   - For shape-related errors: show tensor shape propagation
   - For dtype errors: highlight type mismatches
   - For device errors: compare CPU vs XPU behavior
   - For numerical errors: show value differences
   - Always verify the script runs correctly on the target device
          
         """),
        ("human", "{text}")
    ])

    
    chain = {"text": RunnablePassthrough()} | prompt | llm    
    
    if isinstance(state, list):
        ai_message = state[-3]
    elif messages := state.get("messages", []):
        ai_message = messages[-3]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")

    json_string = ai_message.content
    
    python_object = json.loads(json_string.replace("```json\n","").replace("```", ""))
    error_message =  python_object['error_message']

    result = chain.invoke(state["messages"][-1] + "\n ### original error message: \n" + error_message)

    python_object['reproduce_code'] = result.content
    json_string = json.dumps(python_object)
    print('"""\n' + json_string + '"""')
    return {"messages": AIMessage(content=json_string),
            "execution_path": state.get("execution_path", []) + ["check_debug_trace"]
    }

def check_for_module(state: State): 
    executed_nodes, last_node = get_history(state, "check_for_module")
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an intelligent test debugging assistant specialized in automatically rerunning failed pytest cases according to test module. 
         Follow these precise rules:
:
         1. For inductor failures: first rerun with failed op fallback to eager mode, if still fail dump triton output code.
         2. For other module, do nothing.
         """
         ),
        ("human", "{text}")
    ])
    chain = {"text": RunnablePassthrough()} | prompt | llm_with_module_tools
    return {"messages": chain.invoke(state["messages"][-1]),
            "execution_path": state.get("execution_path", []) + ["check_for_module"]
    }


def summary(state: State):

    executed_nodes, last_node = get_history(state, "summary")

    if isinstance(state, list):
        ai_message2 = state[-1]
    elif messages := state.get("messages", []):
        ai_message2 = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")

    try:
        _reversed_node = list(reversed(executed_nodes))
        index = _reversed_node.index("triage_start") 

        checker = ""
        if ( "check_for_dependency" in _reversed_node and _reversed_node.index("check_for_dependency") + 1 == index ):
            index = index + 2
            checker = "check_for_dependency"

        if ( "check_for_error_type" in _reversed_node and _reversed_node.index("check_for_error_type") + 1 == index ):
            index = index + 2
            checker = "check_for_error_type"
        
        if ( "check_for_module" in _reversed_node and _reversed_node.index("check_for_module") + 1 == index ):
            index = index + 2
            checker = "check_for_module"

        if isinstance(state, list):
            ai_message = state[-index]
        elif messages := state.get("messages", []):
            ai_message = messages[-index]
        else:
            raise ValueError(f"No messages found in input state to tool_edge: {state}")
    
        json_string = ai_message.content
        python_object = json.loads(json_string.replace("```json\n","").replace("```", ""))

        if "Detailed inductor logs are in" not in ai_message2.content:
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a helpful assistant that summary the bug root cause based on the input. For each failure according to the input log only pick up one work from the following tasks:
                        * If oneDNN verbose in the log and there are lines staring with 'onednn_verbose', please create a benchdnn command to reproduce the issue and summary the root cause. 
                        * If a gdb traceback in the log, please highlight the C++ component or code caused the issue. 
                        * Otherwise, just return N/A
                """),
                ("human", "{text}")
            ])
        else:
            # 2. Dynamo Issue (Graph Capture):

            # Does the error occur during the graph capture phase?

            # Look for: Logs containing torchdynamo compile, capturing graph, graph break

            # Stack trace showing torch/_dynamo/ files (convert_frame.py, symbolic_convert.py)

            # Errors about unsupported Python features, graph breaks, or tracing failures

            # Messages like WARNING: graph break or could not trace
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a helpful assistant that triage the bug root cause based on the input for pytorch inductor problems using this framework:
                                
                                    1. Eager Mode Issue (Before Compilation):
                                    Does the error occur with TORCHINDUCTOR_FALLBACK_OPS, if passed with TORCHINDUCTOR_FALLBACK_OPS the issue could be in torch-xpu-ops eager mode. 

                                    Does the error occur on the direct call to the model without any compilation logs?

                                    Look for: No mentions of dynamo, inductor, or compile in the stack trace

                                    Stack trace points to core PyTorch operations (torch/__init__.py, ATen)

                                    Errors like shape mismatches, type errors, or CUDA/XPU errors in plain PyTorch code

                                    2. Inductor Issue (Graph Lowering):

                                    Does the error occur during graph lowering (after successful capture)?

                                    Look for: Logs containing "Lowering" or "IR dump for"

                                    Stack trace showing torch/_inductor/graph.py, torch/_inductor/ir.py, or torch/_inductor/scheduler.py

                                    Assertion errors, key errors, or type errors in Inductor's logic

                                    Errors about unsupported operators or internal logic failures

                                    3. Triton Issue (Code Generation/Execution):

                                    Does the error occur during/after code generation?

                                    Look for: Logs showing "Lowering done" and "Generating Triton code"

                                    Messages about "Compiling kernel" from triton.compiler

                                    Triton compiler errors (syntax errors, assertion failures)

                                    XPU runtime errors (illegal memory access, device-side assert)
                 
                                    4. If the content of output_code.py is in the log
                                    
                                    Examine the code to check possible grammar issues, memory access issues, type mismatches, indexing errors, grid and block configurations.
                 

                                    Please output in a beutified format:

                                    Summary which component is failing (Eager/Inductor/Triton) and explain the nature of the error, the related ops and shapes

                                    The output_code is labeld with [__output_code], please list the output oude. If possible please analyze output code and the traceback, then point the lines or statements of the output code that caused the error, also list the arugments, related ops and address the possible root cause.

                                    Also provide the explainations with:
                                    
                                    Point to the specific log lines that confirm this diagnosis

                                    Point to the dump logs location on local machine for further investigations, such as files under torch_compile_debug and torchinductor_<id> folder

                                    Based on the failed components and reproducing code, suggest appropriate next steps for debugging
                 
                        * If a do nothing tool is called just return N/A."""),
                ("human", "{text}")
            ])

        chain = {"text": RunnablePassthrough()} | prompt | llm

        if checker == "check_for_dependency":
            summary_ai_message = chain.invoke(ai_message2.content)
            python_object["dependency_triaged"] = summary_ai_message.content
    
        if checker == "check_for_error_type":
            summary_ai_message = chain.invoke(ai_message2.content)
            python_object["error_type_triaged"] = summary_ai_message.content

        if checker == "check_for_module":
            summary_ai_message = chain.invoke(ai_message2.content)
            python_object["module_triaged"] = summary_ai_message.content
 
    except:
        print("Impossible, cannot find triage_start")
        raise ValueError(f"Cannot find triage_start node in state.")

    json_string = json.dumps(python_object)
    return {"messages": AIMessage(content=json_string),
            "execution_path": state.get("execution_path", []) + ["summary"]
    }


################################
# Routers 
###############################

# Define your router function for reprodcue tool
def route_on_reproduce_tool(state):
    """
    Use in the conditional_edge to route to the retest_reproduce_tool node, if no tool calling found return END
    """
    
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")

    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        if isinstance(state, list):
            ai_message = state[-2]
        elif messages := state.get("messages", []):
            ai_message = messages[-2]
        else:
            raise ValueError(f"No messages found in input state to tool_edge: {state}")
    
        json_string = ai_message.content
        python_object = json.loads(json_string.replace("```json\n","").replace("```", ""))
    
        print(f"\n### route check_for_error_type: {python_object}\n")
        return 'retest_reproduce_tool'
    else:
        return END

# Define your router function for reprodcue tool
def route_on_debug_prints_tool(state):
    """
    Use in the conditional_edge to route to the retest_reproduce_tool node, if no tool calling found return END
    """
    
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")

    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        if isinstance(state, list):
            ai_message = state[-2]
        elif messages := state.get("messages", []):
            ai_message = messages[-2]
        else:
            raise ValueError(f"No messages found in input state to tool_edge: {state}")
    
        json_string = ai_message.content
        python_object = json.loads(json_string.replace("```json\n","").replace("```", ""))
    
        print(f"\n### route check_for_error_type: {python_object}\n")
        return 'retest_debug_prints_tool'
    else:
        return END
    

def router(
    state: State
):
    """
    Use in the conditional_edge to route to the restest_with_tools node according to the failure type. 
    If no failure type is matched, route to the end.
    """
    #executed_nodes, last_node = get_history(state, "router")

    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")

    json_string = ai_message.content
    python_object = json.loads(json_string.replace("```json\n","").replace("```", ""))
    
    if python_object['error_type_triaged'] == "":
        return 'check_for_error_type'
    elif python_object['module'] != 'inductor' and python_object['dependency_triaged'] == "":
        return 'check_for_dependency'
    elif python_object['module_triaged'] == "":
        return 'check_for_module'
    else:
        return END


def route_error_type(
    state: State
):
    """
    Use in the conditional_edge to route to the restest_with_tools node according to the failure type. 
    If no failure type is matched, route to the end.
    """
    
    #executed_nodes, last_node = get_history(state, "route_error_type")

    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")

    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        if isinstance(state, list):
            ai_message = state[-2]
        elif messages := state.get("messages", []):
            ai_message = messages[-2]
        else:
            raise ValueError(f"No messages found in input state to tool_edge: {state}")
    
        json_string = ai_message.content
        python_object = json.loads(json_string.replace("```json\n","").replace("```", ""))
    
        print(f"\n### route check_for_error_type: {python_object}\n")
        return 'retest_errtype_tools'
    else:
        return END

def route_dependency(
    state: State
):
    """
    Use in the conditional_edge to route to the restest_with_tools node according to the failure type. 
    If no failure type is matched, route to the end.
    """
    #executed_nodes, last_node = get_history(state, "route_dependency")

    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")

    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        if isinstance(state, list):
            ai_message = state[-2]
        elif messages := state.get("messages", []):
            ai_message = messages[-2]
        else:
            raise ValueError(f"No messages found in input state to tool_edge: {state}")
 
        json_string = ai_message.content
        python_object = json.loads(json_string.replace("```json\n","").replace("```", ""))
        
        print(f"\n### route check_for_error_type: {python_object}\n")
        return 'retest_deps_tools'
    else:
        return END 

def route_module(
    state: State
):
    """
    Use in the conditional_edge to route to the restest_with_tools node according to the failure type. 
    If no failure type is matched, route to the end.
    """
    #executed_nodes, last_node = get_history(state, "route_dependency")

    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")

    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        if isinstance(state, list):
            ai_message = state[-2]
        elif messages := state.get("messages", []):
            ai_message = messages[-2]
        else:
            raise ValueError(f"No messages found in input state to tool_edge: {state}")
 
        json_string = ai_message.content
        python_object = json.loads(json_string.replace("```json\n","").replace("```", ""))
        
        return 'retest_module_tools'
    else:
        return END 
    
def stream_graph_updates(user_input: str, graph: StateGraph):
    messages = [HumanMessage(user_input)]

    for event in graph.stream({"messages": [{"role": "user", "content": user_input}], "execution_path": []}):
        for value in event.values():
            if isinstance(value["messages"], list):
                json_string = value["messages"][-1].content
            else:
                 json_string =value["messages"].content

            if len(json_string) > 0:
                try:
                    python_object = json.loads(json_string.replace("```json\n","").replace("```", ""))
                    json_string = json.dumps(python_object, indent=4)
                    print(f"### Assistant output json: {json_string}")
                except:
                    print(f"### Assistant output text: {json_string}")
                
 


graph_builder = StateGraph(State)
graph_builder.add_node("classify", classify)
graph_builder.add_node("depsRAG", depsRAG)
graph_builder.add_node("triage_start", triage_start)
graph_builder.add_node("check_for_error_type", check_for_error_type)
graph_builder.add_node("check_for_dependency", check_for_dependency)
graph_builder.add_node("check_for_module", check_for_module)

graph_builder.add_node("retest_errtype_tools", ToolNode(retest_errtype_tools))
graph_builder.add_node("retest_deps_tools", ToolNode(retest_deps_tools))
graph_builder.add_node("retest_debug_prints_tool", ToolNode([retest_debug_prints_tool,]))
graph_builder.add_node("retest_module_tools", ToolNode(retest_module_tools))

graph_builder.add_node("summary", summary)
graph_builder.add_node("call_debug_prints_tool", call_debug_prints_tool)
graph_builder.add_node("call_eager_comp_tool", call_eager_comp_tool)
graph_builder.add_node("call_inductor_dump_tool", call_inductor_dump_tool)

graph_builder.add_node("check_debug_trace", check_debug_trace)

graph_builder.add_edge(START, "classify")
graph_builder.add_edge(
    "classify",
    "call_debug_prints_tool",
)
graph_builder.add_conditional_edges(
    "call_debug_prints_tool",
    route_on_debug_prints_tool,
    {"retest_debug_prints_tool": "retest_debug_prints_tool", END: END},
)
graph_builder.add_edge(
    "retest_debug_prints_tool",
    "check_debug_trace"
)
graph_builder.add_edge(
    "check_debug_trace",
    "depsRAG",
)
graph_builder.add_edge(
    "depsRAG",
    "triage_start",
)
graph_builder.add_conditional_edges(
    "triage_start",
    router,
    # The following dictionary lets you tell the graph to interpret the condition's outputs as a specific node
    # It defaults to the identity function, but if you
    # want to use a node named something else apart from "tools",
    # You can update the value of the dictionary to something else
    # e.g., "tools": "my_tools"
    {"check_for_dependency": "check_for_dependency", 
     "check_for_error_type": "check_for_error_type", 
     "check_for_module": "check_for_module",
     END: END},
)
graph_builder.add_conditional_edges(
    "check_for_module",
    route_module,
    {
     "retest_module_tools": "retest_module_tools",
     END: END
     }
)
graph_builder.add_edge("retest_module_tools", "summary")
graph_builder.add_conditional_edges(
    "check_for_error_type",
    route_error_type,
    # The following dictionary lets you tell the graph to interpret the condition's outputs as a specific node
    # It defaults to the identity function, but if you
    # want to use a node named something else apart from "tools",
    # You can update the value of the dictionary to something else
    # e.g., "tools": "my_tools"
    {"retest_errtype_tools": "retest_errtype_tools", END: END},
)
graph_builder.add_conditional_edges(
    "check_for_dependency",
    route_dependency,
    # The following dictionary lets you tell the graph to interpret the condition's outputs as a specific node
    # It defaults to the identity function, but if you
    # want to use a node named something else apart from "tools",
    # You can update the value of the dictionary to something else
    # e.g., "tools": "my_tools"
    {"retest_deps_tools": "retest_deps_tools", END: END},
)
graph_builder.add_edge("retest_errtype_tools", "summary")
graph_builder.add_edge("retest_deps_tools", "summary")
graph_builder.add_edge("summary", "triage_start")

# The `tools_condition` function returns "tools" if the chatbot asks to use a tool, and "END" if
# it is fine directly responding. This conditional routing defines the main agent loop.

graph = graph_builder.compile()


from IPython.display import Image, display

try:
    display_image = Image(graph.get_graph().draw_mermaid_png())
    output_filename = 'saved_image.png'
    with open(output_filename, 'wb') as f:
            f.write(display_image.data)
except Exception:
   # This requires some extra dependencies and is optional
   pass


while True:
    try:
        user_input = input("User: ")

        if user_input == "1":
            test_file = "test/xpu/quantization/core/test_workflow_ops_xpu.py"
            base_test_file = "test/quantization/core/test_workflow_ops.py"
            test_class = "TestFusedObsFakeQuantXPU" 
            test_case = "test_fused_obs_fake_quant_moving_avg_per_channel_xpu"
            tmux = "python_test" 
            error_message ="TypeError: accept..test_fused_obs_fake_quant_moving_avg_per_channel() missing 1 required positional argument: 'use_bool'" 
        elif user_input == "2":
            test_file = "test/xpu/test_ops_xpu.py"
            base_test_file = "test/test_ops.py"
            test_class = "TestMathBitsXPU" 
            test_case = "test_conj_view_nn_functional_conv_transpose2d_xpu_complex64"
            tmux = "python_test" 
            error_message ="""
Traceback (most recent call last):
  File "/home/daisyden/miniforge3/envs/pytorch/lib/python3.10/site-packages/torch/testing/_internal/common_device_type.py", line 1135, in test_wrapper
    return test(*args, **kwargs)
  File "/home/daisyden/upstream/pytorch/third_party/torch-xpu-ops/test/xpu/../../../../test/test_ops.py", line 2155, in test_conj_view
    self._test_math_view(
  File "/home/daisyden/upstream/pytorch/third_party/torch-xpu-ops/test/xpu/../../../../test/test_ops.py", line 2087, in _test_math_view
    expected_forward = op(sample.input, *sample.args, **sample.kwargs)
  File "/home/daisyden/miniforge3/envs/pytorch/lib/python3.10/site-packages/torch/testing/_internal/opinfo/core.py", line 1188, in __call__
    return self.op(*args, **kwargs)
RuntimeError: could not create a primitive descriptor for the deconvolution forward propagation primitive. Run workload with environment variable ONEDNN_VERBOSE=all to get additional diagnostic information.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/home/daisyden/miniforge3/envs/pytorch/lib/python3.10/site-packages/torch/testing/_internal/common_utils.py", line 3223, in wrapper
    method(*args, **kwargs)
  File "/home/daisyden/miniforge3/envs/pytorch/lib/python3.10/site-packages/torch/testing/_internal/common_device_type.py", line 426, in instantiated_test
    result = test(self, **param_kwargs)
  File "/home/daisyden/miniforge3/envs/pytorch/lib/python3.10/site-packages/torch/testing/_internal/common_utils.py", line 1644, in wrapper
    fn(*args, **kwargs)
  File "/home/daisyden/miniforge3/envs/pytorch/lib/python3.10/site-packages/torch/testing/_internal/common_device_type.py", line 1147, in test_wrapper
    raise e_tracked from e
Exception: Caused by sample input at index 2: SampleInput(input=Tensor[size=(2, 2, 4, 4), device="xpu:0", dtype=torch.complex64], args=TensorList[Tensor[size=(2, 2, 4, 5), device="xpu:0", dtype=torch.complex64], Tensor[size=(4,), device="xpu:0", dtype=torch.complex64]], kwargs={'stride': '(3,2)', 'padding': '(1,2)', 'output_padding': '(2,3)', 'groups': '2', 'dilation': '(4,4)'}, broadcasts_input=False, name='')

To execute this test, run the following from the base repo dir:
    PYTORCH_OPINFO_SAMPLE_INPUT_INDEX=2 python ../../test/test_ops.py TestMathBitsXPU.test_conj_view_nn_functional_conv_transpose2d_xpu_complex64

This message can be suppressed by setting PYTORCH_PRINT_REPRO_ON_FAILURE=0
"""
        elif user_input == "3":
            test_file = "test/xpu/test_unary_ufuncs_xpu.py"
            base_test_file = "test/test_unary_ufuncs.py"
            test_class = "TestUnaryUfuncsXPU" 
            test_case = "test_reference_numerics_extremal__refs_asinh_xpu_complex64"
            tmux = "python_test" 
            error_message ="""
__________________________ TestUnaryUfuncsXPU.test_reference_numerics_extremal__refs_asinh_xpu_complex64 ___________________________
Traceback (most recent call last):
  File "/home/daisyden/upstream/pytorch/third_party/torch-xpu-ops/test/xpu/../../../../test/test_unary_ufuncs.py", line 311, in test_reference_numerics_extremal
    self._test_reference_numerics(dtype, op, tensors)
  File "/home/daisyden/upstream/pytorch/third_party/torch-xpu-ops/test/xpu/../../../../test/test_unary_ufuncs.py", line 260, in _test_reference_numerics
    _helper_reference_numerics(
  File "/home/daisyden/upstream/pytorch/third_party/torch-xpu-ops/test/xpu/../../../../test/test_unary_ufuncs.py", line 226, in _helper_reference_numerics
    self.assertEqualHelper(
  File "/home/daisyden/upstream/pytorch/third_party/torch-xpu-ops/test/xpu/../../../../test/test_unary_ufuncs.py", line 171, in assertEqualHelper
    self.assertEqual(
  File "/home/daisyden/miniforge3/envs/pytorch/lib/python3.10/site-packages/torch/testing/_internal/common_utils.py", line 4178, in assertEqual
    raise error_metas.pop()[0].to_error(  # type: ignore[index]
AssertionError: Tensor-likes are not close!

Mismatched elements: 36 / 81 (44.4%)
Greatest absolute difference: nan at index (0,) (up to 1e-05 allowed)
Greatest relative difference: nan at index (0,) (up to 1.3e-06 allowed)

To execute this test, run the following from the base repo dir:
    python ../../test/test_unary_ufuncs.py TestUnaryUfuncsXPU.test_reference_numerics_extremal__refs_asinh_xpu_complex64

This message can be suppressed by setting PYTORCH_PRINT_REPRO_ON_FAILURE=0

            """
        elif user_input == "4":
            test_file = "test/inductor/test_torchinductor_opinfo.py"
            base_test_file = "test/inductor/test_torchinductor_opinfo.py"
            test_class = "TestInductorOpInfoXPU" 
            test_case = "test_comprehensive_grid_sampler_3d_xpu_float16"
            tmux = "inductor" 
            error_message ="""
____________________________________________________________________________ TestInductorOpInfoXPU.test_comprehensive_grid_sampler_3d_xpu_float16 ____________________________________________________________________________
Traceback (most recent call last):
  File "/home/daisyden/miniforge3/envs/inductor/lib/python3.10/site-packages/torch-2.9.0a0+gitb6364b0-py3.10-linux-x86_64.egg/torch/testing/_internal/common_device_type.py", line 1135, in test_wrapper
    return test(*args, **kwargs)
  File "/home/daisyden/miniforge3/envs/inductor/lib/python3.10/site-packages/torch-2.9.0a0+gitb6364b0-py3.10-linux-x86_64.egg/torch/testing/_internal/common_device_type.py", line 1434, in only_fn
    return fn(self, *args, **kwargs)
  File "/home/daisyden/miniforge3/envs/inductor/lib/python3.10/site-packages/torch-2.9.0a0+gitb6364b0-py3.10-linux-x86_64.egg/torch/testing/_internal/common_utils.py", line 2361, in wrapper
    fn(*args, **kwargs)
  File "/home/daisyden/miniforge3/envs/inductor/lib/python3.10/site-packages/torch-2.9.0a0+gitb6364b0-py3.10-linux-x86_64.egg/torch/testing/_internal/common_device_type.py", line 1215, in dep_fn
    return fn(slf, *args, **kwargs)
  File "/home/daisyden/miniforge3/envs/inductor/lib/python3.10/site-packages/torch-2.9.0a0+gitb6364b0-py3.10-linux-x86_64.egg/torch/testing/_internal/common_device_type.py", line 1215, in dep_fn
    return fn(slf, *args, **kwargs)
  File "/home/daisyden/miniforge3/envs/inductor/lib/python3.10/site-packages/torch-2.9.0a0+gitb6364b0-py3.10-linux-x86_64.egg/torch/testing/_internal/common_device_type.py", line 1215, in dep_fn
    return fn(slf, *args, **kwargs)
  File "/home/daisyden/miniforge3/envs/inductor/lib/python3.10/site-packages/torch-2.9.0a0+gitb6364b0-py3.10-linux-x86_64.egg/torch/testing/_internal/common_utils.py", line 1645, in wrapper
    fn(*args, **kwargs)
  File "/home/daisyden/miniforge3/envs/inductor/lib/python3.10/site-packages/torch-2.9.0a0+gitb6364b0-py3.10-linux-x86_64.egg/torch/testing/_internal/common_utils.py", line 1560, in wrapper
    fn(*args, **kwargs)
  File "/home/daisyden/miniforge3/envs/inductor/lib/python3.10/unittest/mock.py", line 1370, in patched
    return func(*newargs, **newkeywargs)
  File "/home/daisyden/miniforge3/envs/inductor/lib/python3.10/contextlib.py", line 79, in inner
    return func(*args, **kwds)
  File "/home/daisyden/miniforge3/envs/inductor/lib/python3.10/contextlib.py", line 79, in inner
    return func(*args, **kwds)
  File "/home/daisyden/miniforge3/envs/inductor/lib/python3.10/contextlib.py", line 79, in inner
    return func(*args, **kwds)
  File "/home/daisyden/upstream/gh_162/test/inductor/test_torchinductor_opinfo.py", line 1119, in inner
    raise e
  File "/home/daisyden/upstream/gh_162/test/inductor/test_torchinductor_opinfo.py", line 1111, in inner
    fn(self, device, dtype, op)
  File "/home/daisyden/upstream/gh_162/test/inductor/test_torchinductor_opinfo.py", line 1376, in test_comprehensive
    raise e                                                                                                                                                                                                                     File "/home/daisyden/upstream/gh_162/test/inductor/test_torchinductor_opinfo.py", line 1351, in test_comprehensive
    self.check_model_gpu(
  File "/home/daisyden/miniforge3/envs/inductor/lib/python3.10/contextlib.py", line 79, in inner
    return func(*args, **kwds)
  File "/home/daisyden/upstream/gh_162/test/inductor/test_torchinductor.py", line 685, in check_model_gpu
    check_model(
  File "/home/daisyden/upstream/gh_162/test/inductor/test_torchinductor.py", line 507, in check_model                                                                                                                             actual = run(*example_inputs, **kwargs)                                                                                                                                                                                     File "/home/daisyden/miniforge3/envs/inductor/lib/python3.10/site-packages/torch-2.9.0a0+gitb6364b0-py3.10-linux-x86_64.egg/torch/_dynamo/eval_frame.py", line 817, in compile_wrapper
    raise e.remove_dynamo_frames() from None  # see TORCHDYNAMO_VERBOSE=1
 File "/home/daisyden/miniforge3/envs/inductor/lib/python3.10/site-packages/torch-2.9.0a0+gitb6364b0-py3.10-linux-x86_64.egg/torch/_inductor/compile_fx.py", line 987, in _compile_fx_inner
    raise InductorError(e, currentframe()).with_traceback(
  File "/home/daisyden/miniforge3/envs/inductor/lib/python3.10/site-packages/torch-2.9.0a0+gitb6364b0-py3.10-linux-x86_64.egg/torch/_inductor/compile_fx.py", line 971, in _compile_fx_inner
    mb_compiled_graph = fx_codegen_and_compile(
  File "/home/daisyden/miniforge3/envs/inductor/lib/python3.10/site-packages/torch-2.9.0a0+gitb6364b0-py3.10-linux-x86_64.egg/torch/_inductor/compile_fx.py", line 1673, in fx_codegen_and_compile
    return scheme.codegen_and_compile(gm, example_inputs, inputs_to_check, graph_kwargs)
  File "/home/daisyden/miniforge3/envs/inductor/lib/python3.10/site-packages/torch-2.9.0a0+gitb6364b0-py3.10-linux-x86_64.egg/torch/_inductor/compile_fx.py", line 1440, in codegen_and_compile
    graph.run(*example_inputs)
  File "/home/daisyden/miniforge3/envs/inductor/lib/python3.10/site-packages/torch-2.9.0a0+gitb6364b0-py3.10-linux-x86_64.egg/torch/_inductor/graph.py", line 937, in run
    return super().run(*args)
  File "/home/daisyden/miniforge3/envs/inductor/lib/python3.10/site-packages/torch-2.9.0a0+gitb6364b0-py3.10-linux-x86_64.egg/torch/fx/interpreter.py", line 174, in run
    self.env[node] = self.run_node(node)
  File "/home/daisyden/miniforge3/envs/inductor/lib/python3.10/site-packages/torch-2.9.0a0+gitb6364b0-py3.10-linux-x86_64.egg/torch/_inductor/graph.py", line 1623, in run_node
    result = super().run_node(n)
  File "/home/daisyden/miniforge3/envs/inductor/lib/python3.10/site-packages/torch-2.9.0a0+gitb6364b0-py3.10-linux-x86_64.egg/torch/fx/interpreter.py", line 256, in run_node
    return getattr(self, n.op)(n.target, args, kwargs)
  File "/home/daisyden/miniforge3/envs/inductor/lib/python3.10/site-packages/torch-2.9.0a0+gitb6364b0-py3.10-linux-x86_64.egg/torch/_inductor/graph.py", line 1240, in call_function
    raise MissingOperatorWithoutDecomp(target, args, kwargs)
torch._inductor.exc.InductorError: MissingOperatorWithoutDecomp: missing lowering
  target: aten.grid_sampler_3d.default
  args[0]: TensorBox(StorageBox(
    InputBuffer(name='primals_1', layout=FixedLayout('xpu:0', torch.float16, size=[1, 1, 2, 2, 2], stride=[8, 8, 4, 2, 1]))
  ))
  args[1]: TensorBox(StorageBox(
    InputBuffer(name='primals_2', layout=FixedLayout('xpu:0', torch.float16, size=[1, 1, 1, 1, 3], stride=[3, 3, 3, 3, 1]))
  ))
  args[2]: 0
  args[3]: 0
  args[4]: False

Set TORCHDYNAMO_VERBOSE=1 for the internal stack trace (please do this especially if you're reporting a bug to PyTorch). For even more developer context, set TORCH_LOGS="+dynamo"


The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/home/daisyden/miniforge3/envs/inductor/lib/python3.10/site-packages/torch-2.9.0a0+gitb6364b0-py3.10-linux-x86_64.egg/torch/testing/_internal/common_utils.py", line 3224, in wrapper
    method(*args, **kwargs)
  File "/home/daisyden/miniforge3/envs/inductor/lib/python3.10/site-packages/torch-2.9.0a0+gitb6364b0-py3.10-linux-x86_64.egg/torch/testing/_internal/common_device_type.py", line 426, in instantiated_test
    result = test(self, **param_kwargs)
  File "/home/daisyden/miniforge3/envs/inductor/lib/python3.10/site-packages/torch-2.9.0a0+gitb6364b0-py3.10-linux-x86_64.egg/torch/testing/_internal/common_utils.py", line 1645, in wrapper
    fn(*args, **kwargs)
  File "/home/daisyden/miniforge3/envs/inductor/lib/python3.10/site-packages/torch-2.9.0a0+gitb6364b0-py3.10-linux-x86_64.egg/torch/testing/_internal/common_device_type.py", line 1147, in test_wrapper
    raise e_tracked from e
Exception: Caused by sample input at index 0: SampleInput(input=Tensor[size=(1, 1, 2, 2, 2), device="xpu:0", dtype=torch.float16], args=(Tensor[size=(1, 1, 1, 1, 3), device="xpu:0", dtype=torch.float16],0,0,False), kwargs=
{}, broadcasts_input=False, name='')

To execute this test, run the following from the base repo dir:
    PYTORCH_OPINFO_SAMPLE_INPUT_INDEX=0 python test/inductor/test_torchinductor_opinfo.py TestInductorOpInfoXPU.test_comprehensive_grid_sampler_3d_xpu_float16

            """
        elif user_input == "5":
            test_file = "test/inductor/test_fxir_backend.py"
            base_test_file = "test/inductor/test_fxir_backend.py"
            test_class = "AOTFxirTestCase" 
            test_case = "test_aoti_fx_dynamic"
            tmux = "inductor" 
            error_message ="""
____________________________________________________________________________________________ AOTFxirTestCase.test_aoti_fx_dynamic ____________________________________________________________________________________________
Traceback (most recent call last):
  File "/home/daisyden/upstream/gh_162/test/inductor/test_fxir_backend.py", line 603, in test_aoti_fx_dynamic
    self.check(
  File "/home/daisyden/upstream/gh_162/test/inductor/test_fxir_backend.py", line 558, in check
    self.assertTrue(torch.allclose(model(*inp), gm(*inp)))
  File "/home/daisyden/miniforge3/envs/inductor/lib/python3.10/site-packages/torch-2.9.0a0+gitb6364b0-py3.10-linux-x86_64.egg/torch/fx/graph_module.py", line 837, in call_wrapped
    return self._wrapped_call(self, *args, **kwargs)
  File "/home/daisyden/miniforge3/envs/inductor/lib/python3.10/site-packages/torch-2.9.0a0+gitb6364b0-py3.10-linux-x86_64.egg/torch/fx/graph_module.py", line 413, in __call__
    raise e
  File "/home/daisyden/miniforge3/envs/inductor/lib/python3.10/site-packages/torch-2.9.0a0+gitb6364b0-py3.10-linux-x86_64.egg/torch/fx/graph_module.py", line 400, in __call__
    return super(self.cls, obj).__call__(*args, **kwargs)  # type: ignore[misc]
  File "/home/daisyden/miniforge3/envs/inductor/lib/python3.10/site-packages/torch-2.9.0a0+gitb6364b0-py3.10-linux-x86_64.egg/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/daisyden/miniforge3/envs/inductor/lib/python3.10/site-packages/torch-2.9.0a0+gitb6364b0-py3.10-linux-x86_64.egg/torch/nn/modules/module.py", line 1786, in _call_impl
    return forward_call(*args, **kwargs)
  File "<eval_with_key>.15", line 9, in forward
    triton_kernel_wrapper_mutation = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 0, constant_args_idx = 0, grid = [(floordiv, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'in_ptr0': arg0_1, 'in_ptr1': arg1_1, 'out_ptr0': buf0, 'xnumel': sym_size_int, 'XBLOCK': 4});  floordiv = arg0_1 = arg1_1 = sym_size_int = triton_kernel_wrapper_mutation = None
  File "/home/daisyden/miniforge3/envs/inductor/lib/python3.10/site-packages/torch-2.9.0a0+gitb6364b0-py3.10-linux-x86_64.egg/torch/_higher_order_ops/triton_kernel_wrap.py", line 978, in __call__
    return super().__call__(
  File "/home/daisyden/miniforge3/envs/inductor/lib/python3.10/site-packages/torch-2.9.0a0+gitb6364b0-py3.10-linux-x86_64.egg/torch/_ops.py", line 535, in __call__
    return wrapper()
  File "/home/daisyden/miniforge3/envs/inductor/lib/python3.10/site-packages/torch-2.9.0a0+gitb6364b0-py3.10-linux-x86_64.egg/torch/_ops.py", line 531, in wrapper
    return self.dispatch(
  File "/home/daisyden/miniforge3/envs/inductor/lib/python3.10/site-packages/torch-2.9.0a0+gitb6364b0-py3.10-linux-x86_64.egg/torch/_ops.py", line 519, in dispatch
    return kernel(*args, **kwargs)
  File "/home/daisyden/miniforge3/envs/inductor/lib/python3.10/site-packages/torch-2.9.0a0+gitb6364b0-py3.10-linux-x86_64.egg/torch/_higher_order_ops/triton_kernel_wrap.py", line 1092, in triton_kernel_wrapper_mutation_dense
    kernel[grid_fn](*args, **kwargs, **constant_args)
  File "/home/daisyden/miniforge3/envs/inductor/lib/python3.10/site-packages/triton/runtime/jit.py", line 374, in <lambda>
    return lambda *args, **kwargs: self.run(grid=grid, warmup=False, *args, **kwargs)
  File "/home/daisyden/miniforge3/envs/inductor/lib/python3.10/site-packages/triton/runtime/jit.py", line 578, in run
    kernel = self.compile(src, target=target, options=options.__dict__)
  File "/home/daisyden/miniforge3/envs/inductor/lib/python3.10/site-packages/triton/compiler/compiler.py", line 342, in compile
    module = src.make_ir(options, codegen_fns, module_map, context)
  File "/home/daisyden/miniforge3/envs/inductor/lib/python3.10/site-packages/triton/compiler/compiler.py", line 84, in make_ir
    return ast_to_ttir(self.fn, self, context=context, options=options, codegen_fns=codegen_fns,
  File "/home/daisyden/miniforge3/envs/inductor/lib/python3.10/site-packages/triton/runtime/jit.py", line 914, in get_jit_fn_file_line
    file_name = base_fn.fn.__code__.co_filename
AttributeError: 'NoneType' object has no attribute '__code__'

To execute this test, run the following from the base repo dir:
    python test/inductor/test_fxir_backend.py AOTFxirTestCase.test_aoti_fx_dynamic

This message can be suppressed by setting PYTORCH_PRINT_REPRO_ON_FAILURE=0. Did you mean: '__bool__'?
---------------------------------------------------------------------------------------------------- Captured stdout call ----------------------------------------------------------------------------------------------------
inductor [('async_compile_cache_miss', 2), ('async_compile_cache_hit', 1)]
graph_break []
================================================================================================== short test summary info ===================================================================================================
FAILED [15.5186s] test/inductor/test_fxir_backend.py::AOTFxirTestCase::test_aoti_fx_dynamic - AttributeError: 'NoneType' object has no attribute '__code__'

            """
        #RuntimeError: could not create a primitive descriptor for a deconvolution forward propagation primitive" 
        else:
            continue

        def generate_prompt(test_file: str, base_test_file: str, test_class: str, test_case: str, error_message: str, tmux: str) -> str:
            env = Environment(loader=FileSystemLoader('prompts'))
            template = env.get_template('classification_prompt.j2')
            return template.render(test_file=test_file, base_test_file=base_test_file, test_class=test_class, test_case=test_case, error_message=error_message, tmux=tmux)

        prompt = generate_prompt(test_file, base_test_file, test_class, test_case, error_message, tmux)
        user_input = prompt

        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        print(f"User: {user_input}\n")
        stream_graph_updates(user_input, graph)
    except Exception as e:
        # Catch any other unexpected exceptions
        print(f"An unexpected error occurred: {e}")
        import sys
        exc_type, exc_val, exc_tb = sys.exc_info()
        import traceback
        traceback.print_exception(exc_type, exc_val, exc_tb)

        ## fallback if input() is not available
        #user_input = "What do you know about LangGraph?"
        #print("User: " + user_input)
        #stream_graph_updates(user_input, graph)
        break
