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
        system_prompt = ""
        all_messages = [{"role": "system", "content": system_prompt}] + messages

        yield f"data: {json.dumps({'type': 'start-step'})}\n\n"
        yield f"data: {json.dumps({'type': 'text-start'})}\n\n"

        try:
            response_stream = await client.chat.completions.create(
                model='gpt-4o-mini',
                messages=all_messages,
                tools=TOOLS,
                tool_choice="auto",
                stream=True
            )

            async for chunk in response_stream:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    yield f"data: {json.dumps({'type': 'text-delta', 'delta': delta.content})}\n\n"

                

        except Exception as e:
            error_message = f"\n[chat error] {str(e)}"
            logger.error(error_message, exc_info=True)
            yield f"data: {json.dumps({'type': 'text-delta', 'delta': error_message})}\n\n"
            raise
        finally:
            yield f"data: {json.dumps({'type': 'text-end'})}\n\n"
            yield f"data: {json.dumps({'type': 'finish-step'})}\n\n"
            yield f"data: {json.dumps({'type': 'final'})}\n\n"

    finally:
        duration = time.time() - start_time
        logger.info(f"Chat Agent completed in {duration:.2f} seconds")
