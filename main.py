import os
import uuid
import tempfile
from datetime import datetime, timezone
import torch
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from transformers import pipeline

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

transcript_history: list[dict] = []

device = "cuda:0" if torch.cuda.is_available() else "cpu"

asr_pipeline = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-large-v3-turbo",
    device=device,
    chunk_length_s=30,
)


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename)[-1] or ".audio"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        result = asr_pipeline(
            tmp_path,
            generate_kwargs={"task": "transcribe"},
            return_timestamps=False,
        )
        transcript = result.get("text", "").strip()
        entry = {
            "id": str(uuid.uuid4()),
            "filename": file.filename,
            "transcript": transcript,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        transcript_history.append(entry)
        return JSONResponse({"transcript": transcript, "history": transcript_history})
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.get("/history")
async def get_history():
    return JSONResponse({"history": transcript_history})


@app.get("/samples")
async def list_samples():
    audio_exts = {".mp3", ".wav", ".ogg", ".flac", ".m4a", ".webm", ".opus"}
    static_dir = "static"
    files = []
    if os.path.isdir(static_dir):
        files = [
            f for f in os.listdir(static_dir)
            if os.path.splitext(f)[1].lower() in audio_exts
        ]
    return JSONResponse({"samples": sorted(files)})
