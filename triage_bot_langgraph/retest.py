from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from vllm_service import llm
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools.shell.tool import ShellTool
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
import subprocess
from jinja2 import Environment, FileSystemLoader

#shell_tool = ShellTool()

#tools = [shell_tool]
#llm_with_tools = llm.bind_tools(tools)

@tool
def gdb_catch_throw_tool(test_file: str, test_case: str, tmux: str) -> str:
    """Executes gdb catch throw comamnd to capture the trace of a pytest failure."""

    #command = f"tmux send-keys -t {tmux}  \"PYTORCH_ENABLE_XPU_FALLBACK=1 PYTORCH_TEST_WITH_SLOW=1 gdb -batch -ex \"catch throw\" -ex \"run\" -ex \"bt\" --args python -m pytest -v {test_file} -k {test_case} 2>&1|tee /tmp/log ; tmux wait-for -U my_lock \" C-m; tmux wait-for -L my_lock"
    command = f"tmux send-keys -t {tmux}  \"tmux clear-history; PYTORCH_ENABLE_XPU_FALLBACK=1 PYTORCH_TEST_WITH_SLOW=1 gdb -batch -ex \'catch throw\' -ex \"run\" -ex \"bt\" --args python -m pytest -v {test_file} -k {test_case} 2>&1|tee /tmp/log ; tmux wait -S my_lock \" C-m; tmux wait my_lock; tmux capture-pane -S - -N -p -t {tmux}"
    #command = f"tmux send-keys -t {tmux}  \"python -m pytest -v {test_file} -k {test_case} 2>&1|tee log \" C-m"
    #command = f"tmux at -t {tmux}; which pytest; pwd"
    try:
        result = subprocess.run(
            command,
            shell=True,  # Set to True to allow shell features like piping
            check=True,  # Raise an exception for non-zero exit codes
            text=True,   # Decode stdout/stderr as text
            capture_output=True # Capture stdout and stderr
        )
        #return result.stdout.strip().replace("^.*\"python\" hit Catchpoint 1 (exception thrown)", "")[0:min(50000, len(result.stdout))] if result.stdout else "Command executed successfully."
        return result.stdout.strip().split("test session starts")[-1][0:min(50000, len(result.stdout))] if result.stdout else "Command executed successfully."
    except subprocess.CalledProcessError as e:
        return f"Error executing command: {e.stderr.strip()}"
    except FileNotFoundError:
        return f"Error: Command '{command.split()[0]}' not found."


tools = [gdb_catch_throw_tool]
llm_with_tools = llm.bind_tools(tools)


class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]

def classify(state: State):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that classify pytest faiures."),
        ("human", "{text}")
    ])
    chain = {"text": RunnablePassthrough()} | prompt | llm
    return {"messages": chain.invoke(state["messages"])}


def retest_with_tools(state: State):
    #return {"messages": [llm_with_tools.invoke(state["messages"])]}
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that rerun pytest with gdb catch throw in tmux session"),
        ("human", "{text}")
    ])
    chain = {"text": RunnablePassthrough()} | prompt | llm_with_tools
    return {"messages": chain.invoke(state["messages"])}


def summary(state: State):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that summary the bug root cause based on gdb traceback."),
        ("human", "{text}")
    ])
    chain = {"text": RunnablePassthrough()} | prompt | llm
    return {"messages": chain.invoke(state["messages"][-1])}


def stream_graph_updates(user_input: str, graph: StateGraph):
    messages = [HumanMessage(user_input)]

    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        for value in event.values():
            if isinstance(value["messages"], list):
                print("Assistant:", value["messages"][-1].content)
            else:
                print("Assistant:", value["messages"].content)

def route_failure_type(
    state: State
):
    """
    Use in the conditional_edge to route to the restest_with_tools node according to the failure type. 
    If no failure type is matched, route to the end.
    """

    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")

    import json
    json_string = ai_message.content
    python_object = json.loads(json_string.replace("```json\n","").replace("```", ""))
    
    if python_object['error_type'] == 'RuntimeError':
        #print(f"route retest_with_tools: {python_object}\n")
        return 'retest_with_tools'
    else:
        #print("route END retest_with_tools\n")
        return END


def route_tools(
    state: State,
):
    """
    Use in the conditional_edge to route to the ToolNode if the last message
    has tool calls. Otherwise, route to the end.
    """
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        #print(f"route tools: {ai_message}\n")
        return "tools"

    #print("route END tools\n")
    return END

graph_builder = StateGraph(State)
graph_builder.add_node("classify", classify)
graph_builder.add_node("retest_with_tools", retest_with_tools)
graph_builder.add_node("tools", ToolNode(tools))
graph_builder.add_node("summary", summary)

graph_builder.add_edge(START, "classify")
graph_builder.add_conditional_edges(
    "classify",
    route_failure_type,
    # The following dictionary lets you tell the graph to interpret the condition's outputs as a specific node
    # It defaults to the identity function, but if you
    # want to use a node named something else apart from "tools",
    # You can update the value of the dictionary to something else
    # e.g., "tools": "my_tools"
    {"retest_with_tools": "retest_with_tools", END: END},
)
graph_builder.add_conditional_edges(
    "retest_with_tools",
    route_tools,
    # The following dictionary lets you tell the graph to interpret the condition's outputs as a specific node
    # It defaults to the identity function, but if you
    # want to use a node named something else apart from "tools",
    # You can update the value of the dictionary to something else
    # e.g., "tools": "my_tools"
    {"tools": "tools", END: END},
)
graph_builder.add_edge("tools", "summary")
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
            base_test_file = "../../../test/quantization/core/test_workflow_ops.py"
            test_class = "TestFusedObsFakeQuantXPU" 
            test_case = "test_fused_obs_fake_quant_moving_avg_per_channel_xpu"
            tmux = "python_test" 
            error_message ="TypeError: accept..test_fused_obs_fake_quant_moving_avg_per_channel() missing 1 required positional argument: 'use_bool'" 
        elif user_input == "2":
            test_file = "test/xpu/test_ops_xpu.py"
            base_test_file = "../../../test/test_ops.py"
            test_class = "TestMathBitsXPU" 
            test_case = "test_conj_view_nn_functional_conv_transpose2d_xpu_complex64"

            tmux = "python_test" 
            error_message ="RuntimeError: could not create a primitive descriptor for a deconvolution forward propagation primitive" 
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

        ## fallback if input() is not available
        #user_input = "What do you know about LangGraph?"
        #print("User: " + user_input)
        #stream_graph_updates(user_input, graph)
        break
