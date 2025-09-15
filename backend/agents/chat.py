import json
import os
import time
import logging
from typing import Dict, List, Any
from openai import OpenAI, AsyncOpenAI
from dotenv import load_dotenv
# Set up logging
logger = logging.getLogger(__name__)

# Tool definitions for function calling
# No tools are defined for now, but the structure is here for future use.
TOOLS = []

load_dotenv()



async def stream_chat_py(messages: List[Dict[str, Any]], selected_chat_model: str, request_hints: Dict[str, Any]):
    """
    Handles chat streaming logic with tool support.
    Yields data in Server-Sent Events format.
    """
    start_time = time.time()

    client = AsyncOpenAI()

    try:
        system_prompt = """
        You are a healthcare and Data Analyst Assistant that works for Kaiser Permanente. 
        The user may ask you to research into various healthcare topics or business topics related to healthcare or Kaiser or they may have general chats which won't require you to search things up on the internet.
        They may also ask you to analyze their csv or excel files they uploaded at which point you will call 'data_analyst_agent'
        """

        def to_responses_input(msgs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            out: List[Dict[str, Any]] = []

            for m in msgs:
                role = m.get('role', 'user')
                text = m.get('content', '')

                # Map roles: system -> developer; keep user/assistant as-is.
                if role == "system":
                    role_out = "developer"
                elif role in ("user", "assistant"):
                    role_out = role
                else:
                    # Fall back to user for unknown roles
                    role_out = "user"


                part_type = "input_text" if role_out == "user" else "text"
                
                out.append({
                    "role": "developer" if role == "system" else "user",
                    "content": [{"type": part_type, "text": str(text)}],
                })

            return out 

        responses_api_input = to_responses_input(msgs=messages)

        yield f"data: {json.dumps({"type": "start-step"})}\n\n"
        yield f"data: {json.dumps({"type": "text-start"})}\n\n"

        try:
            async with client.responses.stream(
                model="gpt-5",
                instructions=system_prompt,
                input=responses_api_input,
                tools=[{"type": "web_search"}],
                tool_choice="auto"
            ) as stream:
                async for event in stream:
                    if event.type=="response.output_text.delta":
                        yield f"data: {json.dumps({"type": "text-delta", "delta": event.delta})}\n\n"
                    elif event.type == "response.error":
                        yield f"data: {json.dumps({"type": "error", "message": getattr(event, 'error', 'unknown error')})}\n\n"
                    elif event.type == "response.completed":
                        break

            yield f"data: {json.dumps({"type": "text-end"})}\n\n"
            yield f"data: {json.dumps({"type": "step-end"})}\n\n"
        
        except Exception as e:
            yield f"data: {json.dumps({"type": "error", "message": str(e)})}\n\n"

    finally:
        duration = time.time() - start_time
        logger.info(f"Chat Agent completed in {duration:.2f} seconds")
