import os
import json
import base64
from io import BytesIO
from typing import List, Optional, Dict
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from urllib.parse import quote
from pypdf import PdfReader

from .utils.prompt import ClientMessage
from .utils.attachment import Attachment
from .agents.orchestrator import atlas_agent, create_ephemeral_session
from agents import SQLiteSession, Runner, ItemHelpers
from openai.types.responses import ResponseTextDeltaEvent

load_dotenv(".env")

app = FastAPI()
SESSIONS: Dict[str, SQLiteSession] = {}

class Request(BaseModel):
    messages: List[ClientMessage]
    data: Optional[dict] = None
    chatId: Optional[str] = "default"

@app.post("/api/chat")
async def handle_chat_data(request: Request):
    print("Received request in handle_chat_data")

    if not request.messages:
        return JSONResponse(content={"error": "No messages provided"}, status_code=400)

    user_message = request.messages[-1].content if isinstance(request.messages[-1].content, str) else "".join(
        part.text for part in request.messages[-1].content
    )

    attachments: List[Attachment] = []
    if request.data and isinstance(request.data, dict) and request.data.get("attachments"):
        attachments = [Attachment(**att) for att in request.data["attachments"]]

    pdf_texts: List[str] = []
    for att in attachments:
        if att.type == "application/pdf" and att.content:
            try:
                b64 = att.content.split(",", 1)[1] if "," in att.content else att.content
                pdf_bytes = base64.b64decode(b64)
                pdf_reader = PdfReader(BytesIO(pdf_bytes))
                text = "".join([(page.extract_text() or "") for page in pdf_reader.pages])
                if text.strip():
                    pdf_texts.append(text)
            except Exception as e:
                print(f"Failed to extract PDF text from {att.name}:", e)

    combined_text = user_message
    if pdf_texts:
        combined_text = user_message + "\n\nPDF Content:\n" + "\n\n".join(pdf_texts)

    async def ndjson_stream():
        try:
            yield json.dumps({"event": "thinking", "data": "Starting agent..."}) + "\n"

            chat_id = request.chatId or "default"
            session = SESSIONS.get(chat_id)
            if session is None:
                session = create_ephemeral_session()
                SESSIONS[chat_id] = session

            result = Runner.run_streamed(atlas_agent, input=combined_text, session=session)

            accumulated_text = ""
            last_pdf: Optional[dict] = None

            async for event in result.stream_events():
                if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                    delta = event.data.delta or ""
                    if delta:
                        accumulated_text += delta
                        yield json.dumps({"event": "final", "response": accumulated_text}) + "\n"
                    continue

                if event.type == "agent_updated_stream_event":
                    payload = {
                        "event": "thinking",
                        "data": {"type": "agent_updated", "new_agent": event.new_agent.name},
                    }
                    yield json.dumps(payload) + "\n"
                    continue

                if event.type == "run_item_stream_event":
                    if event.item.type == "tool_call_item":
                        tool_name = (
                            getattr(event.item, "tool_name", None)
                            or getattr(getattr(event.item, "tool", None), "name", None)
                            or getattr(getattr(event.item, "tool_call", None), "name", None)
                            or "unknown_tool"
                        )
                        yield json.dumps({
                            "event": "thinking",
                            "data": {"type": "tool_call", "tool": tool_name},
                        }) + "\n"
                    elif event.item.type == "tool_call_output_item":
                        tool_name = (
                            getattr(event.item, "tool_name", None)
                            or getattr(getattr(event.item, "tool", None), "name", None)
                            or getattr(getattr(event.item, "tool_call", None), "name", None)
                            or "unknown_tool"
                        )
                        output = getattr(event.item, "output", None)
                        if output is not None:
                            parsed = None
                            if isinstance(output, str):
                                try:
                                    parsed = json.loads(output)
                                except Exception:
                                    parsed = None
                            elif isinstance(output, dict):
                                parsed = output

                            if isinstance(parsed, dict):
                                pdf_path = parsed.get("pdf_path")
                                if not pdf_path:
                                    out_dir = parsed.get("output_folder")
                                    fname = parsed.get("filename")
                                    if isinstance(out_dir, str) and isinstance(fname, str):
                                        pdf_path = os.path.join(out_dir, fname)
                                filename = parsed.get("filename") or "resume.pdf"

                                file_url = None
                                if isinstance(pdf_path, str):
                                    base_dir = os.path.abspath("generated_resumes")
                                    real = os.path.abspath(pdf_path)
                                    if real.startswith(base_dir) and os.path.exists(real):
                                        file_url = f"/api/file?path={quote(real)}"
                                if not file_url:
                                    b64 = parsed.get("pdf_b64")
                                    if isinstance(b64, str) and len(b64) > 0:
                                        file_url = f"data:application/pdf;base64,{b64}"

                                if file_url:
                                    last_pdf = {"url": file_url, "name": filename, "contentType": "application/pdf"}
                                    yield json.dumps({"event": "resume_ready", "data": last_pdf}) + "\n"
                        yield json.dumps({
                            "event": "thinking",
                            "data": {"type": "tool_output", "tool": tool_name, "status": "completed"},
                        }) + "\n"
                    elif event.item.type == "message_output_item":
                        text = ItemHelpers.text_message_output(event.item) or ""
                        snippet = (text[:200] + ("..." if len(text) > 200 else "")) if text else ""
                        if snippet:
                            yield json.dumps({
                                "event": "thinking",
                                "data": {"type": "message_output", "text": snippet},
                            }) + "\n"
                    continue
            if last_pdf is not None:
                yield json.dumps({"event": "resume_ready", "data": last_pdf}) + "\n"

        except Exception as e:
            import traceback
            err = {"event": "error", "message": str(e), "trace": traceback.format_exc()[:2000]}
            yield json.dumps(err) + "\n"

    return StreamingResponse(ndjson_stream(), media_type="application/x-ndjson")


@app.get("/api/file")
async def get_file(path: str):
    base_dir = os.path.abspath("generated_resumes")
    tmp_base = "/tmp/generated_resumes"
    real = os.path.abspath(path)
    if not (real.startswith(base_dir) or real.startswith(tmp_base)):
        return JSONResponse(content={"error": "Forbidden"}, status_code=403)
    media = "application/pdf" if real.lower().endswith(".pdf") else "application/octet-stream"
    if media == "application/pdf":
        return FileResponse(
            real,
            media_type=media,
            headers={"Content-Disposition": f"inline; filename=\"{os.path.basename(real)}\""},
        )
    return FileResponse(real, media_type=media, filename=os.path.basename(real))


class ResetRequest(BaseModel):
    chatId: Optional[str] = "default"


@app.post("/api/session/reset")
async def reset_session(req: ResetRequest):
    chat_id = req.chatId or "default"
    if chat_id in SESSIONS:
        try:
            del SESSIONS[chat_id]
        except Exception:
            pass
    return JSONResponse(content={"ok": True, "chatId": chat_id})
