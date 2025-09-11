import os
import openai
from dotenv import load_dotenv

load_dotenv()

client = openai.AsyncOpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    timeout=60.0,
    default_headers={
        "OpenAI-Beta": "assistants=v2",
    },
)

async def research_agent_stream(input_str: str):
    async with client.responses.stream(
        model="o4-mini-deep-research",
        input=input_str,
        tools=[{"type": "web_search_preview"}],
    ) as stream:
        async for event in stream:
            # Stream text deltas
            if event.type == "response.output_text.delta":
                yield event.delta or ""
            # Optional: surface errors
            elif event.type == "response.error":
                msg = getattr(event, "error", None)
                raise Exception(getattr(msg, "message", "unknown responses error"))


