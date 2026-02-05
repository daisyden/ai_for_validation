from langchain_community.llms import VLLMOpenAI
#from langchain_community.chat_models import ChatVLLMOpenAI
from langchain_openai import ChatOpenAI

import os
proxy = os.environ.get("http_proxy", "")
if proxy != "": 
    os.environ["http_proxy"] = ""
    os.environ["https_proxy"] = ""
    os.environ["no_proxy"] = "10.112.110.111"

llm = ChatOpenAI(
    openai_api_key="EMPTY",
    #openai_api_base="http://skyrex.jf.intel.com:8000/v1",
    #model_name="meta-llama/Llama-3.3-70B-Instruct",
    openai_api_base="http://10.112.110.111/v1",
    model_name="glm-4.6-fp8",
    #model_kwargs={"stop": ["."]},
    api_key="sk-XZrfiPGmZaGLZFPNUpy6ww",
    max_tokens=2000,

)

os.environ["http_proxy"] = proxy
os.environ["https_proxy"] = proxy
os.environ["no_proxy"] = ""