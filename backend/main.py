from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Any, Dict
import logging
import openai
import os
from dotenv import load_dotenv

from .agents.chat import stream_chat_py

# Load environment variables from .env file
load_dotenv()

# Configure simple logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

app = FastAPI()

client = openai.AsyncOpenAI()


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
            client,  # Pass the client instance
        ),
        media_type="text/event-stream",
    )
