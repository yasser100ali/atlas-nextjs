import json
import os
from typing import Dict, List, Any, Optional
from .research_agent import research_agent_stream, client

# Tool definitions for function calling
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather information for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA"
                    }
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_document",
            "description": "Create a document or artifact based on user request",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Title for the document"
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to include in the document"
                    },
                    "type": {
                        "type": "string",
                        "enum": ["text", "code", "sheet"],
                        "description": "Type of document to create"
                    }
                },
                "required": ["title", "type"]
            }
        }
    }
]

def build_system_prompt(request_hints: Dict[str, Any]) -> str:
    """Build the system prompt with location context."""
    req = f"""About the origin of user's request:
- lat: {request_hints.get('latitude')}
- lon: {request_hints.get('longitude')}
- city: {request_hints.get('city')}
- country: {request_hints.get('country')}"""

    artifacts_prompt = """
Artifacts is a special user interface mode that helps users with writing, editing, and other content creation tasks. When artifact is open, it is on the right side of the screen, while the conversation is on the left side. When creating or updating documents, changes are reflected in real-time on the artifacts and visible to the user.

When asked to write code, always use artifacts. When writing code, specify the language in the backticks, e.g. ```python```code here```. The default language is Python. Other languages are not yet supported, so let the user know if they request a different language.

DO NOT UPDATE DOCUMENTS IMMEDIATELY AFTER CREATING THEM. WAIT FOR USER FEEDBACK OR REQUEST TO UPDATE IT.

This is a guide for using artifacts tools: `createDocument` and `updateDocument`, which render content on a artifacts beside the conversation.

**When to use `createDocument`:**
- For substantial content (>10 lines) or code
- For content users will likely save/reuse (emails, code, essays, etc.)
- When explicitly requested to create a document
- For when content contains a single code snippet

**When NOT to use `createDocument`:**
- For informational/explanatory content
- For conversational responses
- When asked to keep it in chat

**Using `updateDocument`:**
- Default to full document rewrites for major changes
- Use targeted updates only for specific, isolated changes
- Follow user instructions for which parts to modify

**When NOT to use `updateDocument`:**
- Immediately after creating a document
"""

    regular = 'You are a friendly assistant! Keep your responses concise and helpful.'

    # Only include artifacts prompt for non-reasoning models
    return f"{regular}\n\n{req}\n\n{artifacts_prompt}"


def handle_tool_call(tool_call: Dict[str, Any]) -> str:
    """Handle tool calls from the model."""
    function_name = tool_call.get('function', {}).get('name')
    arguments = tool_call.get('function', {}).get('arguments', '{}')

    try:
        args = json.loads(arguments)
    except:
        args = {}

    if function_name == 'get_weather':
        location = args.get('location', 'Unknown')
        return mock_weather_tool(location)
    elif function_name == 'create_document':
        title = args.get('title', 'Untitled')
        content = args.get('content', '')
        doc_type = args.get('type', 'text')
        return mock_create_document(title, content, doc_type)

    return f"Unknown tool: {function_name}"

async def stream_chat_py(messages: List[Dict[str, Any]], selected_chat_model: str, request_hints: Dict[str, Any]):
    """
    Handles chat streaming logic with tool support.
    For 'chat-model-reasoning', it uses the research_agent.
    For other models, it uses standard chat completion with tools.
    Yields data in Server-Sent Events format.
    """
    system_prompt = build_system_prompt(request_hints)

    if selected_chat_model == 'chat-model-reasoning':
        # Use research agent for deep research
        last_user_message = next((m['content'] for m in reversed(messages) if m['role'] == 'user'), '')
        combined_input = f"{system_prompt}\n\n{last_user_message}"

        yield f"data: {json.dumps({'type': 'start-step'})}\n\n"
        yield f"data: {json.dumps({'type': 'text-start'})}\n\n"

        try:
            async for delta in research_agent_stream(combined_input):
                if delta:
                    yield f"data: {json.dumps({'type': 'text-delta', 'delta': delta})}\n\n"

        except Exception as e:
            error_message = f"\n[research error] {str(e)}"
            yield f"data: {json.dumps({'type': 'text-delta', 'delta': error_message})}\n\n"
        finally:
            yield f"data: {json.dumps({'type': 'text-end'})}\n\n"
            yield f"data: {json.dumps({'type': 'finish-step'})}\n\n"
            yield f"data: {json.dumps({'type': 'final'})}\n\n"

    else:
        # Use standard chat completion with tools for base model
        all_messages = [{"role": "system", "content": system_prompt}] + messages

        yield f"data: {json.dumps({'type': 'start-step'})}\n\n"
        yield f"data: {json.dumps({'type': 'text-start'})}\n\n"

        try:
            response_stream = await client.chat.completions.create(
                model='gpt-4o-mini',  # Changed from gpt-4.1 to gpt-4o-mini which exists
                messages=all_messages,
                tools=TOOLS,
                tool_choice="auto",
                stream=True
            )

            async for chunk in response_stream:
                if hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    yield f"data: {json.dumps({'type': 'text-delta', 'delta': content})}\n\n"
                elif hasattr(chunk.choices[0].delta, 'tool_calls') and chunk.choices[0].delta.tool_calls:
                    # Handle tool calls
                    tool_calls = chunk.choices[0].delta.tool_calls
                    for tool_call in tool_calls:
                        if hasattr(tool_call, 'function') and tool_call.function:
                            tool_result = handle_tool_call({
                                'function': {
                                    'name': tool_call.function.name,
                                    'arguments': tool_call.function.arguments
                                }
                            })
                            yield f"data: {json.dumps({'type': 'text-delta', 'delta': f'\n{tool_result}\n'})}\n\n"

        except Exception as e:
            error_message = f"\n[chat error] {str(e)}"
            yield f"data: {json.dumps({'type': 'text-delta', 'delta': error_message})}\n\n"
        finally:
            yield f"data: {json.dumps({'type': 'text-end'})}\n\n"
            yield f"data: {json.dumps({'type': 'finish-step'})}\n\n"
            yield f"data: {json.dumps({'type': 'final'})}\n\n"
