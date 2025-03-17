from pydantic import BaseModel, Field
from llm_model import llm_model
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

class GetWeather(BaseModel):
    '''Get the current weather in a given location'''

    location: str = Field(..., description="The city and state, e.g. San Francisco, CA")

class GetPopulation(BaseModel):
    '''Get the current population in a given location'''

    location: str = Field(..., description="The city and state, e.g. San Francisco, CA")

chat = ChatHuggingFace(llm=llm_model, verbose=True)

#from langchain_core.utils.function_calling import convert_to_openai_tool
#GetWeather = convert_to_openai_tool(GetWeather)
#GetPopulation = convert_to_openai_tool(GetPopulation)

chat_with_tools = chat.bind_tools([GetWeather, GetPopulation], strict=True)
ai_msg = chat_with_tools.invoke("Which city is hotter today and which is bigger: LA or NY?")

print(ai_msg.tool_calls)
