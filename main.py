import os
import uuid
import logging
from datetime import datetime, timezone

import groq
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_api_key = os.environ.get("GROQ_API_KEY")
if not _api_key:
    raise RuntimeError("GROQ_API_KEY environment variable is not set.")

client = groq.Groq(api_key=_api_key)

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

transcript_history: list[dict] = []


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    audio_bytes = await file.read()

    # Transcribe in the source language and detect what it is
    try:
        transcript_response = client.audio.transcriptions.create(
            file=(file.filename, audio_bytes),
            model="whisper-large-v3",
            response_format="verbose_json",
        )
    except Exception as e:
        logger.error("Groq transcription failed: %s", e)
        raise HTTPException(status_code=500, detail="Transcription failed. Please try again.")

    original_text     = transcript_response.text.strip()
    detected_language = (getattr(transcript_response, "language", None) or "").lower()
    logger.info("Detected language: %s", detected_language)

    # Skip translation when audio is already in English
    if detected_language in ("english", "en"):
        english_translation = original_text
    else:
        try:
            chat_response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a professional translator. "
                            "Translate the user's transcript into natural, fluent English. "
                            "Preserve the original meaning, tone, and context exactly. "
                            "Return only the translated text with no commentary or explanation."
                        ),
                    },
                    {
                        "role": "user",
                        "content": original_text,
                    },
                ],
                temperature=0.2,
            )
            english_translation = chat_response.choices[0].message.content.strip()
        except Exception as e:
            logger.error("Groq LLM translation failed: %s", e)
            raise HTTPException(status_code=500, detail="Translation failed. Please try again.")

    entry = {
        "id": str(uuid.uuid4()),
        "filename": file.filename,
        "original_text": original_text,
        "english_translation": english_translation,
        "detected_language": detected_language,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    transcript_history.append(entry)
    return JSONResponse({
        "original_text": original_text,
        "english_translation": english_translation,
        "detected_language": detected_language,
        "history": transcript_history,
    })


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
