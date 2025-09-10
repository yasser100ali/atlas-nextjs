from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Any, Dict

from .agents.chat import stream_chat_py

app = FastAPI()

class ChatRequest(BaseModel):
    messages: List[Dict[str, Any]]
    selectedChatModel: str
    requestHints: Dict[str, Any]

@app.post("/api/chat")
async def chat_endpoint(chat_request: ChatRequest):
    return StreamingResponse(
        stream_chat_py(
            chat_request.messages,
            chat_request.selectedChatModel,
            chat_request.requestHints,
        ),
        media_type="text/event-stream",
    )
