
from langgraph.graph import END
from vllm_service import llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langgraph.prebuilt import ToolNode
from langchain_core.messages import AIMessage, ToolMessage
import json
from jinja2 import Environment, FileSystemLoader
from common_nodes import State, get_history
from tools.triage_tools import (
    gdb_catch_throw_tool,
    do_nothing_tool,
    onednn_verbose_tool,
    instrument_tool,
    inductor_tool
)
import json


retest_tools = [gdb_catch_throw_tool, onednn_verbose_tool, inductor_tool, instrument_tool, do_nothing_tool]
retest_tool_node = ToolNode(retest_tools)

def get_base64_encoded_message(info: dict) -> str:
    import base64
    json_str = json.dumps(info)
    info_encoded = base64.b64encode(json_str.encode()).decode()
    return info_encoded

def get_message_from_state(state: State):
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")

    json_string = ai_message.content
    try:
        import json
        python_object = json.loads(json_string.replace("```json\n","").replace("```", ""))
    except json.JSONDecodeError:
        python_object = None
        print("Failed to decode JSON from message content.")
    return python_object, json_string

def get_issue_info_from_state(state: State):
    python_object = json.loads(state.get("issue_info", {}).replace("```json\n","").replace("```", ""))
    return python_object

def triage_plan_agent(
    state: State,
):
    executed_nodes, last_node = get_history(state, "triage_plan_agent")
    issue_info, _ = get_message_from_state(state)
                 
    prompt = ChatPromptTemplate.from_messages([
            ("system", """
             You create triage plans for failed pytest tests based on user proivided issue information.
**Inductor issue**: The issue with module inductor.
**Non-inductor issue**: The issue without module inductor.
**Depends on oneDNN**: The dependency is oneDNN
**Non-oneDNN op issue**: The issue is not depend on oneDNN
**RuntimeError**: The error type is RuntimeError
**Non-RuntimeError**: The error type is not RuntimeError

Rule to pick triage steps:
1. **Inductor issue** → Use inductor_tool & triage_inductor_issue_agent agent
2. **Depends on oneDNN ONLY** → Use verbose_tool & triage_verbose_agent agent  
3. **Non-inductor + RuntimeError** → Use gdb_catch_throw_tool tool & triage_gdb_catch_throw_agent agent
4. **Non-inductor ** → Use instrument_tool & draft_reproduce_agent agent

For example:
1. Inductor issue with op depends on oneDNN → Step 1 and Step 2
2. Inductor issue with op without oneDNN dependency → Step 1
3. Non-inductor oneDNN op issue with Asserterror → Step 2 and Step 4
4. Non-inductor non-oneDNN op issue and not Runtimeerror → Step 4
5. Non-inductor non-oneDNN op issue with Runtimeerror → Step 3 and Step 4
             
Output ONLY this JSON format, and make sure to no duplicated triage_steps:
```json
{{
    "total_steps": "number_of_steps",
    "triage_steps": [
        {{
        "retest_tool": "tool_name",
        "triage_agent": "agent_name"
        "reason": "reason_for_choosing_this_step"
        }},
    ],
    "current_step": <current_step_index, integer number starting from 0>
}}
```
Pick tools/agents based on the rules above.
             """),
            ("human", "{text}")
        ])
    chain = {"text": RunnablePassthrough()} | prompt | llm
    
    triage_plan = chain.invoke(AIMessage(content=state["messages"][-1].content))
    try:
        if isinstance(triage_plan, list):
            json_string = triage_plan[-1].content
        else:
            json_string =triage_plan.content
        if "```json" in json_string:
            index0 = json_string.find("```json\n")
            json_string = json_string[index0 + len("```json\n"):]
            index1 = json_string.rfind("}\n```")
            json_string = json_string[:index1+1]                    

        issue_info['triage_plan'] = json.loads(json_string)
    except json.JSONDecodeError:
        issue_info['triage_plan'] = None
        import pdb
        pdb.set_trace()
        exit("Failed to decode JSON from triage plan message content.")
    import pdb
    pdb.set_trace()
    return {"messages": triage_plan,
            "execution_path": state.get("execution_path", []) + ["triage_plan_agent"],
            "issue_info": json.dumps(issue_info)
    }

def retest_agent(
    state: State,
):
    executed_nodes, last_node = get_history(state, "retest_agent")

    issue_info = get_issue_info_from_state(state)
    issue_info_json = json.dumps(issue_info)
    
    
    total_step = int(issue_info['triage_plan']["total_steps"])
    current_step = int(issue_info['triage_plan']["current_step"])
    if current_step < total_step:
        next_tool = issue_info['triage_plan']["triage_steps"][current_step]
    else:
        next_tool = None

    import pdb
    pdb.set_trace()
    
    if next_tool is not None:
        next_tool_name = json.dumps(next_tool)
        prompt = ChatPromptTemplate.from_messages([
            ("system", """
             You are a helpful assistant that rerun the failed pytest test case according to the triage plan, get workdir and container in the input and call the next retest tool: 
             
             {{next_tool_name}}
             
             in the workdir in the container.
             """),
            ("human", "{issue_info_json}"
             )
        ])

        llm_with_tools = llm.bind_tools(retest_tools, tool_choice="auto")
        
        chain = {"issue_info_json": lambda _: issue_info_json} | prompt | llm_with_tools
        return {"messages": chain.invoke(AIMessage(content=json.dumps(issue_info))),
                "execution_path": state.get("execution_path", []) + ["retest_agent"],
                "issue_info": json.dumps(issue_info)
        }
    else:
        return {"messages": AIMessage(content=json.dumps(issue_info)),
                "execution_path": state.get("execution_path", []) + ["retest_agent"],
                "issue_info": json.dumps(issue_info)
        }        


# Check if agent called tools
def check_retest_tool_calls(state: State) -> str:
    messages = state["messages"]
    last_message = messages[-1]
    
    import pdb
    pdb.set_trace()
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "retest_tools"
    # elif hasattr(last_message, 'content') and "<tool_call>" in last_message.content:
    #     return "retest_tools"
    else:
        return "summary_agent"

def triage_agent(state: State):
    executed_nodes, last_node = get_history(state, "triage_agent")

    return {"messages": AIMessage(content=state.get("messages", [])[-1].content),
            "execution_path": state.get("execution_path", []) + ["triage_agent"],
            "issue_info": state.get("issue_info", {})
    }

def route_triage_agent(state: State):
    executed_nodes, last_node = get_history(state, "triage_agent")
    
    issue_info = get_issue_info_from_state(state)
    _, message = get_message_from_state(state)
    
    i = int(issue_info["triage_plan"]["current_step"])
    next_triage_step = issue_info["triage_plan"]["triage_steps"][i]["triage_agent"]
    triage_target = next_triage_step.replace('triage_', '').replace('_agent', '')
    if "instrument tool" in message:
        return "draft_reproduce_agent"
    elif triage_target in message:
        return "triage_" + triage_target + "_agent"
    else:
        return "summary_agent"

   
def triage_verbose_agent(state: State):
    executed_nodes, last_node = get_history(state, "triage_verbose_agent")
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant that triage pytorch unit test failures based on verbose information:
         1. Analysis the oneDNN verbos of \"DNNL_VERBOSE=1\" for oneDNN dependent ops, detemine the backend, ops and shape, also create a benchdnn command to reproduce the issue 
         2. Analysis the MKL verbose of \"MKL_VERBOSE=1\" for MKL dependent ops, detemine the backend, ops and shape
         """
        ),         
        ("human", "{text}")
    ])
    chain = {"text": RunnablePassthrough()} | prompt | llm
    issue_info = get_issue_info_from_state(state)
    message = chain.invoke(state["messages"][-1])
    issue_info['onednn_issue_triaged'] = message
    issue_info['triage_plan']['current_step'] += 1
    return {"messages": message,
            "execution_path": state.get("execution_path", []) + ["triage_verbose_agent"],
            "issue_info": json.dumps(issue_info)
    }

def triage_gdb_catch_throw_agent(state: State):
    executed_nodes, last_node = get_history(state, "triage_gdb_catch_throw_agent")
    
    import json
    issue_info = get_issue_info_from_state(state)
    issue_info_json = json.dumps(issue_info)
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an intelligent test debugging assistant that root cause pytest issue based on GDB catch trough result.
         1. Detemine the failed op, function and backend of the issue
         2. Root cause the issue and provide possible fix or workaround
         """
         ),
        ("human", """The collected issue information is as follows:
         {issue_info_json}
         {text}
         """)
    ])    
    
    chain = {"text": RunnablePassthrough(), "issue_info_json": lambda _: issue_info_json} | prompt | llm
    message = chain.invoke(state["messages"][-1])    
    issue_info['gdb_catch_throw_triaged'] = message.content
    issue_info['gdb_catch_throw'] = state["messages"][-1].content
    issue_info['triage_plan']['current_step'] += 1
    import pdb
    pdb.set_trace()
    return {"messages": AIMessage(content=message.content),
            "execution_path": state.get("execution_path", []) + ["triage_gdb_catch_throw_agent"],
            "issue_info": json.dumps(issue_info)
    }


def triage_inductor_issue_agent(state: State): 
    executed_nodes, last_node = get_history(state, "triage_inductor_issue_agent")
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an intelligent test debugging assistant that triage inductor related pytest failures:
         1. If the fallback test failed, it is an eager problem.
         2. if fallback test passed, analysis inductor logs to determine the root cause.
         """
         ),
        ("human", "{text}")
    ])
    chain = {"text": RunnablePassthrough()} | prompt | llm
    issue_info = get_issue_info_from_state(state)
    message = chain.invoke(state["messages"][-1])
    issue_info['inductor_issue_triaged'] = message
    return {"messages": message,
            "execution_path": state.get("execution_path", []) + ["triage_inductor_issue_agent"],
            "issue_info": json.dumps(issue_info)
    }

def triage_general_agent(state: State):
    executed_nodes, last_node = get_history(state, "triage_general_agent")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that triage pytest failures."),
        ("human", "{text}")
    ])
    chain = {"text": RunnablePassthrough()} | prompt | llm
    issue_info = get_issue_info_from_state(state)
    message = chain.invoke(state["messages"][-1])
    issue_info['general_issue_triaged'] = message.content
    return {"messages": message,
            "execution_path": state.get("execution_path", []) + ["triage_general_agent"],
            "issue_info": json.dumps(issue_info)
    }

def draft_reproduce_agent(state: State):
    executed_nodes, last_node = get_history(state, "draft_reproduce_test")
    
    def generate_prompt(issue_info: dict) -> str:
        env = Environment(loader=FileSystemLoader('prompts'))            
        template = env.get_template('draft_reproduce_test.j2')
        return template.render(test_case=issue_info.get("test_case", ""),
                               test_file=issue_info.get("test_file", ""),
                               test_class=issue_info.get("test_class", ""),
                               error_message=issue_info.get("error_message", ""),
                               traceback=issue_info.get("traceback", ""),
                               module=issue_info.get("module", ""),
                               dependency=issue_info.get("dependency", ""),
                               torch_op=issue_info.get("torch_op", ""),
                               )

    issue_info = get_issue_info_from_state(state)
    prompt_text = generate_prompt(issue_info)    
    
    print("Draft reproduce prompt:", prompt_text)
    prompt = ChatPromptTemplate.from_messages([
                                            ("system", prompt_text),
                                            ("human", "{text}")
                                            ])
    chain = {"text": RunnablePassthrough()} | prompt | llm
    
    issue_info["call_tracing"] = state["messages"][-1].content
    message = chain.invoke(state["messages"][-1])

    # Extract code from "Reproduce test script\n```<code>```" format
    reproduce_script = message.content
    if "```python" in reproduce_script:
        start_idx = reproduce_script.find("```python") + len("```python\n")
        end_idx = reproduce_script.rfind("```")
        reproduce_script = reproduce_script[start_idx:end_idx].strip()
    elif "```" in reproduce_script:
        start_idx = reproduce_script.find("```") + len("```\n")
        end_idx = reproduce_script.rfind("```")
        reproduce_script = reproduce_script[start_idx:end_idx].strip()

    issue_info["reproduce_test_script"] = reproduce_script
    issue_info["reproduce_test_details"] = message.content
    issue_info['triage_plan']['current_step'] += 1
    
    return {"messages": message,
            "execution_path": state.get("execution_path", []) + ["draft_reproduce_agent"],
            "issue_info": json.dumps(issue_info)
    }



def summary_agent(state: State):

    executed_nodes, last_node = get_history(state, "triage_general_agent")
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
         You are a helpful assistant that can create beautiful summaries for pytest failures based on the issue information provided. When generating the summary try the best to exact answer from user input.
         1. Brief root cause analysis of the issue, no need minium reproduce test script.
         2. Suggest possible fixes or workarounds.
         """),
        ("human", "issue information {issue_info}")
    ])
    issue_info = get_issue_info_from_state(state)
    chain = {"text": RunnablePassthrough(), "issue_info": lambda _: json.dumps(issue_info)} | prompt | llm
    
    message = chain.invoke(state["messages"][-1])
    issue_info['summary'] = message.content
    print("Triage summary: ", message.content)
    print("Details: ", json.dumps(issue_info, indent=4))
    issue = issue_info['link'].split('/')[-1]
    with open(f"results/summary_issue{issue}.txt", "w") as f:
        f.write("# Triage summary: \n" + message.content)   
        f.write("\n\n# Details: \n")
        f.write("## Test case: " + issue_info.get("test_case", "N/A") + "\n")
        f.write("## Test file: " + issue_info.get("test_file", "N/A") + "\n")
        f.write("## Test class: " + issue_info.get("test_class", "N/A") + "\n")
        f.write("## Error message: " + issue_info.get("error_message", "N/A") + "\n")
        f.write("## Traceback: " + f"\n```\n" + issue_info.get("traceback", "N/A") + "\n```\n")
        f.write("## Reproduce command: " + issue_info.get("reproduce_command", "N/A") + "\n")
        f.write("## onednn issue triaged: " + issue_info.get("onednn_issue_triaged", "N/A") + "\n")
        f.write("## inductor issue triaged: " + issue_info.get("inductor_issue_triaged", "N/A") + "\n")
        f.write("## gdb catch throw triaged: " + issue_info.get("gdb_catch_throw_triaged", "N/A") + "\n")
        f.write("## Call tracing: \n" + f"\n```\n" + issue_info.get("call_tracing", "N/A") + "\n```\n")
        f.write("## Reproduce test script: \n" + f"\n```\n" + issue_info.get("reproduce_test_script", "N/A") + "\n```\n")
    return {"messages": message,
            "execution_path": state.get("execution_path", []) + ["summary_agent"],
            "issue_info": json.dumps(issue_info)
    }
    