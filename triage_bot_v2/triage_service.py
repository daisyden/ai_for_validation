# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


import uvicorn

from collect_issue_info_and_retest import collect_issue_info_and_retest

app = FastAPI()

# Request/Response models
class ChatRequest(BaseModel):
    message: str
    temperature: float = 0.7

class ChatResponse(BaseModel):
    response: str

@app.post("/guilty_commit", response_model=ChatResponse)
async def guilty_commit(request: ChatRequest):
    try:
        print("&&&&&&&&&&&&&&&&&&&&&&&\n")
        response = collect_issue_info_and_retest(issue=request.message)
        return ChatResponse(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)