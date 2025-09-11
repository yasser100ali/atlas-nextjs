import os
import openai
from dotenv import load_dotenv

load_dotenv()

client = openai.AsyncOpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    timeout=60.0,
)

async def research_agent_stream(input_str: str):
    async with client.responses.stream(
        model="o4-mini-deep-research",
        input=input_str,
    ) as stream:
        async for event in stream:
            # Stream text deltas
            if event.type == "response.output_text.delta":
                yield event.delta or ""
            # Optional: surface errors
            elif event.type == "response.error":
                msg = getattr(event, "error", None)
                raise Exception(getattr(msg, "message", "unknown responses error"))

async def research_agent(input_str: str, stream: bool = False):
    if stream:
        # For backward-compat callers, return the async iterator
        return research_agent_stream(input_str)
    # Non-streaming one-shot
    resp = await client.responses.create(
        model="o4-mini-deep-research",
        input=input_str,
        tools=["web_search_preview"]
    )
    # Extract aggregate text
    try:
        return "".join(
            o.content[0].text.value
            for o in (resp.output or [])
            if getattr(o, "type", None) == "output_text"
            and getattr(o, "content", None)
            and getattr(o.content[0], "type", None) == "output_text"
        )
    except Exception:
        return ""
