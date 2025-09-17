import json 
import time
import logging 
import os
import re
import requests
from io import BytesIO
from typing import List, Any, Dict, AsyncIterator
from dotenv import load_dotenv
from agents.agent import Agent
from agents.run import Runner
import PyPDF2
import pandas as pd
load_dotenv()

logger = logging.getLogger(__name__)

def extract_pdf_text(url: str) -> str:
    """Extract text from PDF file"""
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        pdf_file = BytesIO(response.content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
            
        return text.strip()
    except Exception as e:
        logger.error(f"Error extracting PDF text: {e}")
        return f"[Error reading PDF: {str(e)}]"

def extract_excel_data(url: str) -> str:
    """Extract data from Excel file"""
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        excel_file = BytesIO(response.content)
        
        # Try reading as Excel first
        try:
            df = pd.read_excel(excel_file)
        except:
            # If Excel fails, try CSV
            excel_file.seek(0)
            df = pd.read_csv(excel_file)
        
        # Convert to string representation
        return df.to_string(max_rows=100)  # Limit rows to avoid huge output
    except Exception as e:
        logger.error(f"Error extracting Excel/CSV data: {e}")
        return f"[Error reading Excel/CSV: {str(e)}]"

def process_file_content(content: str) -> str:
    """Process message content and extract file contents"""
    # Pattern to match file references: [File: filename (mediaType) - URL: url]
    file_pattern = r'\[File: ([^(]+) \(([^)]+)\) - URL: ([^\]]+)\]'
    
    def replace_file_ref(match):
        filename = match.group(1).strip()
        media_type = match.group(2).strip()
        url = match.group(3).strip()
        
        if media_type == 'application/pdf':
            file_content = extract_pdf_text(url)
            return f"[PDF File: {filename}]\n{file_content}\n[End of PDF]"
        elif media_type in ['text/csv', 'application/vnd.ms-excel', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet']:
            file_content = extract_excel_data(url)
            return f"[Excel/CSV File: {filename}]\n{file_content}\n[End of Excel/CSV]"
        else:
            return f"[File: {filename} ({media_type}) - Content not processed]"
    
    return re.sub(file_pattern, replace_file_ref, content)



def to_agent_messages(history: List[Dict[str, Any]]):
    msgs = []
    for m in history:
        role = m.get("role", "user").lower()
        text = str(m.get("content", ""))
        
        # Process file content if present
        processed_text = process_file_content(text)

        if role == "system":
            msgs.append({"content": processed_text, "role": "developer", "type": "message"})
        elif role == "assistant":
            msgs.append({"content": processed_text, "role": "assistant", "type": "message"})
        else:
            msgs.append({"content": processed_text, "role": "user", "type": "message"})

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
    )

    agent_input = to_agent_messages(messages)

    # Prologue 
    yield f"data: {json.dumps({"type": "start-step"})}\n\n"
    yield f"data: {json.dumps({"type": "text-start"})}\n\n"

    try: 
        streamed = Runner.run_streamed(agent, input=agent_input)

        async for ev in streamed.stream_events():
            et = getattr(ev, "type", "")

            # Handle raw_response_event with ResponseTextDeltaEvent
            if et == "raw_response_event":
                data = getattr(ev, "data", None)
                if data and hasattr(data, '__class__') and 'ResponseTextDeltaEvent' in str(data.__class__):
                    delta = getattr(data, "delta", "")
                    if delta:
                        yield f"data: {json.dumps({"type": "text-delta", "delta": delta})}\n\n"

            elif et in ("text.delta", "response.text.delta", "agent.output_text.delta"):
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
    