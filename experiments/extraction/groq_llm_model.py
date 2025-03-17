import os
import getpass
if not os.environ.get("GROQ_API_KEY"):
    #os.environ["GROQ_API_KEY"] = getpass.getpass("Enter API key for Groq: ")
    os.environ["GROQ_API_KEY"] = ""

from langchain.chat_models import init_chat_model

llm_model = init_chat_model("llama3-8b-8192", model_provider="groq")
