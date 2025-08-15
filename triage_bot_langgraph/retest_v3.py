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

    command = f"tmux send-keys -R -t {tmux}  \"tmux clear-history; clear ; {env} pytest -v {test_file} -k {test_case} 2>&1|tee /tmp/log ; tmux wait -S my_lock \" C-m; tmux wait my_lock; tmux capture-pane -S - -E - -p -t {tmux}"
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
   
    command = f"tmux send-keys -R -t {tmux}  \"tmux clear-history; clear ; PYTORCH_ENABLE_XPU_FALLBACK=1 PYTORCH_TEST_WITH_SLOW=1 gdb -batch -ex \'catch throw\' -ex \"run\" -ex \"bt\" --args python -m pytest -v {test_file} -k {test_case} 2>&1|tee /tmp/log ; tmux wait-for -S my_lock \" C-m; tmux wait-for my_lock; tmux capture-pane -S - -E - -p -t {tmux}"
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
    
    command = f"tmux send-keys -R -t {tmux}  \"tmux clear-history; clear ; PYTORCH_ENABLE_XPU_FALLBACK=1 PYTORCH_TEST_WITH_SLOW=1 pytest -vs {test_file} -k {test_case} 2>&1|tee /tmp/log ; mv {base_test_file}.saved {base_test_file} ; tmux wait -S my_lock \" C-m ; tmux wait my_lock ; tmux capture-pane -S - -E - -p -t {tmux}"
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
    
retest_errtype_tools = [gdb_catch_throw_tool, do_nothing_tool]
llm_with_errortools = llm.bind_tools(retest_errtype_tools, tool_choice="auto")

retest_deps_tools = [verbose_tool, do_nothing_tool]
llm_with_depstools = llm.bind_tools(retest_deps_tools, tool_choice="auto")

llm_with_reprtools = llm.bind_tools([retest_reproduce_tool,])
llm_with_dbgtools = llm.bind_tools([retest_debug_prints_tool,])

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
    import pdb
    pdb.set_trace()
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

    chain = {"text": RunnablePassthrough()} | prompt | llm_with_dbgtools
 
    return {"messages": chain.invoke(state["messages"][-1]),
            "execution_path": state.get("execution_path", []) + ["call_debug_prints_tool"]
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
    chain = {"text": RunnablePassthrough()} | prompt | llm_with_depstools
    return {"messages": chain.invoke(state["messages"][-1]),
            "execution_path": state.get("execution_path", []) + ["check_for_dependency"]
    }

def check_for_error_type(state: State):
 
    executed_nodes, last_node = get_history(state, "check_for_error_type")

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an intelligent test debugging assistant specialized in automatically rerunning failed pytest cases with appropriate debugging instrumentation based on the error type. 
         Follow these precise rules:
:
         1. For RuntimeError failures: Rerun in GDB debug mode to capture stack traces by tool calling. 
         2. For other error_types, do nothing.
         """
         ),
        ("human", "{text}")
    ])
    chain = {"text": RunnablePassthrough()} | prompt | llm_with_errortools
    return {"messages": chain.invoke(state["messages"][-1]),
            "execution_path": state.get("execution_path", []) + ["check_for_error_type"]
    }

def check_reproducible(state: State):
    executed_nodes, last_node = get_history(state, "check_reproducible")

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that check pytest log to determine whether a failure defined in json is reproduced in the log, if yes return 'Reproduced' as output, otherwise return 'NotReproduce'."),
        ("human", "{text}")
    ])
    chain = {"text": RunnablePassthrough()} | prompt | llm    
    result = chain.invoke(state["messages"][-1])

    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")

    json_string = ai_message.content
    python_object = json.loads(json_string.replace("```json\n","").replace("```", ""))
    
    python_object['reproduced'] = result
    json_string = json.dumps(python_object)
    return {"messages": AIMessage(content=json_string),
            "execution_path": state.get("execution_path", []) + ["check_reproducible"]
    }


def summary(state: State):

    executed_nodes, last_node = get_history(state, "summary")

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that summary the bug root cause based on the input. If a do nothing tool is called just return N/A."),
        ("human", "{text}")
    ])
    chain = {"text": RunnablePassthrough()} | prompt | llm

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

        if isinstance(state, list):
            ai_message = state[-index]
        elif messages := state.get("messages", []):
            ai_message = messages[-index]
        else:
            raise ValueError(f"No messages found in input state to tool_edge: {state}")
    
        json_string = ai_message.content
        python_object = json.loads(json_string.replace("```json\n","").replace("```", ""))

        if checker == "check_for_dependency":
            summary_ai_message = chain.invoke(ai_message2.content)
            python_object["dependency_triaged"] = summary_ai_message.content
    
        if checker == "check_for_error_type":
            summary_ai_message = chain.invoke(ai_message2.content)
            python_object["error_type_triaged"] = summary_ai_message.content
 
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

def route_on_reproduce_result(state):
    """
    Use in the conditional_edge to route to next step or END, if the failure is reproduced do next step triage, otherwise go to END
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
    
    if python_object["reproduced"] == "true":
        return END
    
    if python_object['error_type_triaged'] == "":
        return 'check_for_error_type'
    elif python_object['dependency_triaged'] == "":
        return 'check_for_dependency'
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
graph_builder.add_node("retest_errtype_tools", ToolNode(retest_errtype_tools))
graph_builder.add_node("retest_deps_tools", ToolNode(retest_deps_tools))
graph_builder.add_node("retest_reproduce_tool", ToolNode([retest_reproduce_tool,]))
graph_builder.add_node("retest_debug_prints_tool", ToolNode([retest_debug_prints_tool,]))
graph_builder.add_node("summary", summary)
graph_builder.add_node("call_reproduce_tool", call_reproduce_tool)
graph_builder.add_node("call_debug_prints_tool", call_debug_prints_tool)
graph_builder.add_node("check_reproducible", check_reproducible)

graph_builder.add_edge(START, "classify")
graph_builder.add_edge(
    "classify",
    "call_reproduce_tool",
)
graph_builder.add_conditional_edges(
    "call_reproduce_tool",
    route_on_reproduce_tool,
    {"retest_reproduce_tool": "retest_reproduce_tool", END: END},
)
graph_builder.add_edge(
    "retest_reproduce_tool",
    "check_reproducible"
)
graph_builder.add_edge(
    "check_reproducible",
    "call_debug_prints_tool"
)
graph_builder.add_conditional_edges(
    "call_debug_prints_tool",
    route_on_debug_prints_tool,
    {"retest_debug_prints_tool": "retest_debug_prints_tool", END: END},
)
graph_builder.add_edge(
    "retest_debug_prints_tool",
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
    {"check_for_dependency": "check_for_dependency", "check_for_error_type": "check_for_error_type", END: END},
)
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
TypeError: could not create a primitive descriptor for the deconvolution forward propagation primitive. Run workload with environment variable ONEDNN_VERBOSE=all to get additional diagnostic information.

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
