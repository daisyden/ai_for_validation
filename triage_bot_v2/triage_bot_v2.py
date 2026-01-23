
import argparse
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
import json
from common_nodes import stream_graph_updates, State

from triage_bot_nodes import (
    triage_general_agent,
    triage_gdb_catch_throw_agent,
    triage_plan_agent,
    triage_verbose_agent,
    triage_inductor_issue_agent,
    draft_reproduce_agent,
    summary_agent,
    retest_agent,
    retest_tool_node,
    check_retest_tool_calls,
    triage_agent,
    route_triage_agent,
)

def main():
    with open(args.issue_json, 'r') as f:
        issue_info = json.load(f)
        # issue_info["triage_plan"] = {}
        # issue_info["runtime_error_triaged"] = ""
        # issue_info["onednn_issue_triaged"] = ""
        # issue_info["inductor_issue_triaged"] = ""
        # issue_info["reproduce_test_drafted"] = ""

    try:
        user_input = json.dumps(issue_info)
        print("User: " + user_input)

        def build_graph():
            graph_builder = StateGraph(State)
            graph_builder.add_node("triage_plan_agent", triage_plan_agent)
            graph_builder.add_node("retest_agent", retest_agent)
            graph_builder.add_node("triage_agent", triage_agent)
            
            graph_builder.add_node("triage_gdb_catch_throw_agent", triage_gdb_catch_throw_agent)
            graph_builder.add_node("triage_verbose_agent", triage_verbose_agent)
            graph_builder.add_node("triage_inductor_issue_agent", triage_inductor_issue_agent)
            #graph_builder.add_node("triage_general_agent", triage_general_agent)
            graph_builder.add_node("draft_reproduce_agent", draft_reproduce_agent)
            graph_builder.add_node("summary_agent", summary_agent)
            
            graph_builder.add_node("retest_tools", retest_tool_node)

            graph_builder.add_edge(START, "triage_plan_agent")
            graph_builder.add_edge("triage_plan_agent", "retest_agent")
            graph_builder.add_conditional_edges(
                "retest_agent",
                check_retest_tool_calls,
                #{"retest_tools": "retest_tools", END: END}
                {"retest_tools": "retest_tools", "summary_agent": "summary_agent"}
            )
            
            graph_builder.add_edge("retest_tools", "triage_agent")
            graph_builder.add_conditional_edges(
                "triage_agent",
                route_triage_agent,
                {
                "triage_verbose_agent": "triage_verbose_agent",
                "triage_gdb_catch_throw_agent": "triage_gdb_catch_throw_agent",
                "summary_agent": "summary_agent",
                "draft_reproduce_agent": "draft_reproduce_agent",
                "triage_inductor_issue_agent": "triage_inductor_issue_agent",
                END: END
                }
            )
            graph_builder.add_edge("triage_verbose_agent", "retest_agent")
            graph_builder.add_edge("triage_gdb_catch_throw_agent", "retest_agent")
            graph_builder.add_edge("triage_inductor_issue_agent", "retest_agent")
            #graph_builder.add_edge("triage_general_agent", "retest_agent")
            graph_builder.add_edge("draft_reproduce_agent", "retest_agent")
            graph_builder.add_edge("summary_agent", END)
            


            graph = graph_builder.compile().with_config({"recursion_limit": 20})
            return graph

        graph = build_graph()

        from IPython.display import Image, display

        try:
            display_image = Image(graph.get_graph().draw_mermaid_png())
            output_filename = 'saved_image.png'
            with open(output_filename, 'wb') as f:
                    f.write(display_image.data)
        except Exception:
            # This requires some extra dependencies and is optional
            pass

        
        stream_graph_updates(user_input, graph)

    except Exception as e:
        # Catch any other unexpected exceptions
        print(f"An unexpected error occurred: {e}")
        import sys
        exc_type, exc_val, exc_tb = sys.exc_info()
        import traceback
        traceback.print_exception(exc_type, exc_val, exc_tb)
   
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Triage Bot for Pytest Failures")
    parser.add_argument("--issue_json", type=str, required=True, help="The json file containing the issue details")
    args = parser.parse_args()
    main()