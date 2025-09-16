import json 
import time
import logging 
from typing import List, Any, Dict, AsyncIterator
from dotenv import load_dotenv
from agents import Agent, Runner, WebSearchTool

load_dotenv()

logger = logging.getLogger(__name__)



def to_agent_messages(history: List[Dict[str, Any]]):
    msgs = []
    for m in history:
        role = m.get("role", "user").lower()
        text = str(m.get("content", ""))

        if role == "system":
            msgs.append({"content": text, "role": "developer", "type": "message"})
        elif role == "assistant":
            msgs.append({"content": text, "role": "assistant", "type": "message"})
        else:
            msgs.append({"content": text, "role": "user", "type": "message"})

    return msgs

async def stream_chat_py(
    messages: List[Dict[str, Any]],
    selected_chat_mode: str,
    request_hints: Dict[str, Any] | None 
) -> AsyncIterator[str]:

    start_time = time.time()

    agent = Agent(
        name="agent",
        model="gpt-4.1",
        instructions="You are a healthcare and Data Analyst Assistant for Kaiser Permanente. Use web_search for current facts and cite sources. If the user uploads CSV/Excel and asks for analysis, you will call 'data_analyst_agent'. Be concise.",
        tools=[WebSearchTool()]
    )

    agent_input = to_agent_messages(messages)

    # Prologue 
    yield f"data: {json.dumps({"type": "start-step"})}\n\n"
    yield f"data: {json.dumps({"type": "text-start"})}\n\n"

    try: 
        streamed = Runner.run_streamed(agent, input=agent_input)

        async for ev in streamed.stream_events():
            et = getattr(ev, "type", "")

            if et in ("text.delta", "response.text.delta", "agent.output_text.delta"):
                chunk = getattr(ev, "delta", None) or getattr(ev, "text", "")
                if chunk: 
                    yield f"data: {json.dumps({"type": "text-delta", "delta": chunk})}\n\n"

            elif et in ("error", "agent.error", "run.error"):
                msg = str(getattr(ev, "error", "unknown_error"))
                yield f"data: {json.dumps({"type": "error", "message": msg})}\n\n"
        
        yield f"data: {json.dumps({"type": "text-end"})}\n\n"
        yield f"data: {json.dumps({"type": "end-step"})}\n\n"

    except Exception as e:
        yield f"data: {json.dumps({"type": "error", "message": str(e)})}\n\n"

    finally: 
        end_time = time.time()
        duration = end_time - start_time
        logger.info(f"Chat Agent completed in {duration:.2f} seconds.")
    