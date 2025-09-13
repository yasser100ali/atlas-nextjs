import json
import os
import time
import logging
from typing import Dict, List, Any

# Set up logging
logger = logging.getLogger(__name__)

# Tool definitions for function calling
# No tools are defined for now, but the structure is here for future use.
TOOLS = []

def build_system_prompt(request_hints: Dict[str, Any]) -> str:
    """Build the system prompt with location context."""
    req = f"""About the origin of user's request:
- lat: {request_hints.get('latitude')}
- lon: {request_hints.get('longitude')}
- city: {request_hints.get('city')}
- country: {request_hints.get('country')}"""

    artifacts_prompt = """
Artifacts is a special user interface mode that helps users with writing, editing, and other content creation tasks. When artifact is open, it is on the right side of the screen, while the conversation is on the left side. When creating or updating documents, changes are reflected in real-time on the artifacts and visible to the user.
"""

    regular = 'You are a friendly assistant! Keep your responses concise and helpful.'

    return f"{regular}\n\n{req}\n\n{artifacts_prompt}"

def handle_tool_call(tool_call: Dict[str, Any], client) -> str:
    """Handle tool calls from the model."""
    function_name = tool_call.get('function', {}).get('name')
    # Since there are no tools, this will just return an unknown tool message.
    return f"Unknown tool: {function_name}"

async def stream_chat_py(messages: List[Dict[str, Any]], selected_chat_model: str, request_hints: Dict[str, Any], client):
    """
    Handles chat streaming logic with tool support.
    Yields data in Server-Sent Events format.
    """
    start_time = time.time()

    try:
        system_prompt = build_system_prompt(request_hints)
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

                if delta and delta.tool_calls:
                    # This block will likely not be hit if TOOLS is empty
                    for tool_call in delta.tool_calls:
                        tool_result = handle_tool_call(
                            {'function': {'name': tool_call.function.name, 'arguments': tool_call.function.arguments}},
                            client
                        )
                        yield f"data: {json.dumps({'type': 'text-delta', 'delta': f'[Tool call result: {tool_result}]'})}\n\n"

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
