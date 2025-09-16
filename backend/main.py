from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Any, Dict
import logging
import os
from dotenv import load_dotenv

from .chat_agents.chat import stream_chat_py


# Configure simple logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

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
            chat_request.requestHints
        ),
        media_type="text/event-stream",
    )
