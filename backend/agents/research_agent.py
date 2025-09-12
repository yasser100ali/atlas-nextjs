import os
import openai
import time
import logging
from dotenv import load_dotenv

# Set up logging
logger = logging.getLogger()

load_dotenv()

client = openai.AsyncOpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    timeout=60.0,
    default_headers={
        "OpenAI-Beta": "assistants=v2",
    },
)

async def research_agent_stream(input_str: str):
    """Research agent stream with simple timing."""
    start_time = time.time()

    try:
        async with client.responses.stream(
            model="gpt-5",
            input=input_str,
            tools=[{"type": "web_search"}],
        ) as stream:
            async for event in stream:
                # Stream text deltas
                if event.type == "response.output_text.delta":
                    delta = event.delta or ""
                    yield delta
                # Optional: surface errors
                elif event.type == "response.error":
                    msg = getattr(event, "error", None)
                    raise Exception(getattr(msg, "message", "unknown responses error"))

    finally:
        duration = time.time() - start_time
        logger.info(f"Research Agent completed in {duration:.2f} seconds")


