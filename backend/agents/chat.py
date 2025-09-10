import json
from .research_agent import research_agent, client

async def stream_chat_py(messages, selected_chat_model, request_hints):
    """
    Handles chat streaming logic.
    For 'chat-model-reasoning', it uses the research_agent.
    Otherwise, it uses a standard chat completion model.
    Yields data in Server-Sent Events format.
    """
    def build_system_prompt():
        req = f"""About the origin of user's request:
- lat: {request_hints.get('latitude')}
- lon: {request_hints.get('longitude')}
- city: {request_hints.get('city')}
- country: {request_hints.get('country')}"""
        regular = 'You are a friendly assistant! Keep your responses concise and helpful.'
        return f"{regular}\n\n{req}"

    if selected_chat_model == 'chat-model-reasoning':
        sys_prompt = build_system_prompt()
        last_user_message = next((m['content'] for m in reversed(messages) if m['role'] == 'user'), '')
        combined_input = f"{sys_prompt}\n\n{last_user_message}"

        yield f"data: {json.dumps({'type': 'start-step'})}\n\n"
        yield f"data: {json.dumps({'type': 'text-start'})}\n\n"

        try:
            response_stream = await research_agent(combined_input, stream=True)

            async for chunk in response_stream:
                content = chunk.choices[0].delta.content
                if content:
                    yield f"data: {json.dumps({'type': 'text-delta', 'delta': content})}\n\n"
        except Exception as e:
            error_message = f"\\n[research error] {e}"
            yield f"data: {json.dumps({'type': 'text-delta', 'delta': error_message})}\n\n"
        finally:
            yield f"data: {json.dumps({'type': 'text-end'})}\n\n"
            yield f"data: {json.dumps({'type': 'finish-step'})}\n\n"
            yield f"data: {json.dumps({'type': 'final'})}\n\n"

    else:
        # Fallback for other models.
        system_prompt = build_system_prompt()
        
        all_messages = [{"role": "system", "content": system_prompt}] + messages

        yield f"data: {json.dumps({'type': 'start-step'})}\n\n"
        yield f"data: {json.dumps({'type': 'text-start'})}\n\n"

        try:
            response_stream = await client.chat.completions.create(
                model='gpt-4.1', 
                messages=all_messages,
                stream=True
            )
            
            async for chunk in response_stream:
                content = chunk.choices[0].delta.content
                if content:
                    yield f"data: {json.dumps({'type': 'text-delta', 'delta': content})}\n\n"

        except Exception as e:
            error_message = f"\\n[chat error] {e}"
            yield f"data: {json.dumps({'type': 'text-delta', 'delta': error_message})}\n\n"
        finally:
            yield f"data: {json.dumps({'type': 'text-end'})}\n\n"
            yield f"data: {json.dumps({'type': 'finish-step'})}\n\n"
            yield f"data: {json.dumps({'type': 'final'})}\n\n"
