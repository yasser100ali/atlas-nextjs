import json 
import time
import logging 
import os
from typing import List, Any, Dict, AsyncIterator
from dotenv import load_dotenv
from agents import Agent, Runner, WebSearchTool
load_dotenv()

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)



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
    
    logger.info(f"Starting stream_chat_py with {len(messages)} messages")
    logger.info(f"OpenAI API key present: {'OPENAI_API_KEY' in os.environ}")

    agent = Agent(
        name="agent",
        model="gpt-4.1",  # Use a valid model
        instructions="You are a healthcare and Data Analyst Assistant for Kaiser Permanente. Use web_search for current facts and cite sources. If the user uploads CSV/Excel and asks for analysis, you will call 'data_analyst_agent'. Be concise.",
        tools=[WebSearchTool()]
    )

    agent_input = to_agent_messages(messages)
    logger.info(f"Agent input: {agent_input}")

    # Prologue 
    yield f"data: {json.dumps({"type": "start-step"})}\n\n"
    yield f"data: {json.dumps({"type": "text-start"})}\n\n"

    try: 
        logger.info("Creating Runner.run_streamed")
        streamed = Runner.run_streamed(agent, input=agent_input)
        logger.info("Starting to iterate over stream events")

        async for ev in streamed.stream_events():
            et = getattr(ev, "type", "")
            logger.debug(f"Received event type: {et}, event: {ev}")

            # Handle raw_response_event with ResponseTextDeltaEvent
            if et == "raw_response_event":
                data = getattr(ev, "data", None)
                if data and hasattr(data, '__class__') and 'ResponseTextDeltaEvent' in str(data.__class__):
                    delta = getattr(data, "delta", "")
                    if delta:
                        logger.info(f"Streaming text chunk: {delta}")
                        yield f"data: {json.dumps({"type": "text-delta", "delta": delta})}\n\n"

            elif et in ("text.delta", "response.text.delta", "agent.output_text.delta"):
                chunk = getattr(ev, "delta", None) or getattr(ev, "text", "")
                if chunk: 
                    logger.info(f"Streaming text chunk: {chunk}")
                    yield f"data: {json.dumps({"type": "text-delta", "delta": chunk})}\n\n"

            elif et in ("error", "agent.error", "run.error"):
                msg = str(getattr(ev, "error", "unknown_error"))
                logger.error(f"Agent error: {msg}")
                yield f"data: {json.dumps({"type": "error", "message": msg})}\n\n"
                
        yield f"data: {json.dumps({"type": "text-end"})}\n\n"
        yield f"data: {json.dumps({"type": "end-step"})}\n\n"

    except Exception as e:
        yield f"data: {json.dumps({"type": "error", "message": str(e)})}\n\n"

    finally: 
        end_time = time.time()
        duration = end_time - start_time
        logger.info(f"Chat Agent completed in {duration:.2f} seconds.")
    