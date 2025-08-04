from langchain_community.llms import VLLMOpenAI
#from langchain_community.chat_models import ChatVLLMOpenAI
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    openai_api_key="EMPTY",
    openai_api_base="http://skyrex.jf.intel.com:8000/v1",
    model_name="meta-llama/Llama-3.3-70B-Instruct",
    #model_kwargs={"stop": ["."]},
    max_tokens=2000,
)

