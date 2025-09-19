from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from vllm_service import llm
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import AIMessage
import json

from datetime import datetime


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
    
def stream_graph_updates(user_input: str, graph: StateGraph):
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}], "execution_path": []}):
        for value in event.values():
            if isinstance(value["messages"], list):
                json_string = value["messages"][-1].content
            else:
                 json_string =value["messages"].content

            if len(json_string) > 0:
                try:
                    python_object = json.loads(json_string.replace("```json\n","").replace("```", ""))
                    print(python_object["module_triaged"] + "=========\n" + python_object["reproduce_code"] + "=========\n")
                    json_string = json.dumps(python_object, indent=4)
                    print(f"### Assistant output json: {json_string}")
                except:
                    print(f"### Assistant output text: {json_string}")
            return json_string
                
