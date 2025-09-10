import os
import openai
from dotenv import load_dotenv

load_dotenv()

client = openai.AsyncOpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    timeout=60.0,
)

async def research_agent(input_str: str, stream: bool = False):
    messages = [{"role": "user", "content": input_str}]
    response = await client.chat.completions.create(
        model="o4-mini-deep-research",
        messages=messages,
        stream=stream,
    )
    return response
