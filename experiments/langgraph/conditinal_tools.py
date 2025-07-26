from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from vllm_service import llm
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools.shell.tool import ShellTool
from langchain_core.messages import HumanMessage


shell_tool = ShellTool()
tools = [shell_tool]
llm = llm.bind_tools(tools)

class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]

def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

def stream_graph_updates(user_input: str, graph: StateGraph, llm):
    messages = [HumanMessage(user_input)]

    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        for value in event.values():
            import pdb
            pdb.set_trace()
            print("Assistant:", value["messages"][-1].content)
        #if 'chatbot' in event.keys():
        #    messages.append(event['chatbot']['messages'][0])
        #elif 'tools' in event.keys():
        #    messages.append(event['tools']['messages'][0])

            #if len(event['tools']['messages'][0].tool_calls) > 0: 
            #    for tool_call in event['tools']['messages'][0].tool_calls:
            #        tool_result = shell_tool.invoke(tool_call)
            #        messages.append(tool_result)
    #print(llm.invoke(messages))
           
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
    import pdb
    pdb.set_trace()
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return END



## Any time a tool is called, we return to the chatbot to decide the next step
#graph_builder.add_edge("tools", "chatbot")
#graph_builder.add_edge(START, "chatbot")
#graph = graph_builder.compile()


graph_builder = StateGraph(State)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", ToolNode(tools))
# Any time a tool is called, we return to the chatbot to decide the next step
graph_builder.add_edge("tools", "chatbot")
# The `tools_condition` function returns "tools" if the chatbot asks to use a tool, and "END" if
# it is fine directly responding. This conditional routing defines the main agent loop.
graph_builder.add_conditional_edges(
    "chatbot",
    route_tools,
    # The following dictionary lets you tell the graph to interpret the condition's outputs as a specific node
    # It defaults to the identity function, but if you
    # want to use a node named something else apart from "tools",
    # You can update the value of the dictionary to something else
    # e.g., "tools": "my_tools"
    {"tools": "tools", END: END},
)
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
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        stream_graph_updates(user_input, graph, llm)
    except:
        # fallback if input() is not available
        user_input = "What do you know about LangGraph?"
        print("User: " + user_input)
        stream_graph_updates(user_input, graph, llm)
        break
