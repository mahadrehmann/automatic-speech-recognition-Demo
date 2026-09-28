# Automatic Speech Recognition Demo

A Dockerized web application for multilingual speech-to-text transcription, powered by OpenAI's **Whisper large-v3-turbo** model via the Hugging Face `transformers` library.

## Features

- **Multilingual** — auto-detects and transcribes English, Urdu, Bengali, Chinese, and more
- **Dynamic sample library** — automatically lists any audio files placed in the `static/` folder
- **Drag-and-drop upload** — upload your own MP3, WAV, OGG, FLAC, M4A, or WebM files
- **In-browser playback** — preview any selected audio before transcribing
- **Transcript history** — every transcription is stored in-memory for the session
- **Keyword search** — real-time filtering across filenames and transcript text with highlighted matches
- **GPU-aware** — automatically uses CUDA if available, falls back to CPU

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI + Uvicorn |
| ML Model | `openai/whisper-large-v3-turbo` via HuggingFace Transformers |
| Frontend | Vanilla HTML / CSS / JavaScript |
| Container | Docker (python:3.10-slim) |

## Quick Start

### With Docker (recommended)

```bash
# Build the image (downloads the Whisper model on first run)
docker build -t asr-demo .

# Run on CPU
docker run -p 8000:8000 asr-demo

# Run on GPU
docker run --gpus all -p 8000:8000 asr-demo
```

Then open **http://localhost:8000** in your browser.

### Without Docker

```bash
# Install system dependencies (Ubuntu/Debian)
sudo apt-get install ffmpeg

# Install Python dependencies
pip install -r requirements.txt

# Start the server
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Adding Sample Audio Files

Drop any audio file into the `static/` directory and refresh the page — it will appear automatically as a sample button. Supported formats: `.mp3`, `.wav`, `.ogg`, `.flac`, `.m4a`, `.webm`, `.opus`.

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Serves the demo UI |
| `POST` | `/transcribe` | Accepts an audio file, returns transcript + updated history |
| `GET` | `/history` | Returns the full in-session transcript history |
| `GET` | `/samples` | Returns a list of audio files available in `/static` |

### `POST /transcribe`

**Request:** `multipart/form-data` with a field named `file`.

**Response:**
```json
{
  "transcript": "The transcribed text goes here.",
  "history": [
    {
      "id": "uuid",
      "filename": "audio.mp3",
      "transcript": "...",
      "timestamp": "2026-09-28T10:00:00+00:00"
    }
  ]
}
```

## Project Structure

```
asr-demo/
├── Dockerfile
├── requirements.txt
├── main.py               # FastAPI app + Whisper pipeline
├── static/               # Audio sample files (auto-listed in UI)
└── templates/
    └── index.html        # Single-page frontend
```

## Notes

- The Whisper model (~1.6 GB) is downloaded from Hugging Face on the first startup. Subsequent starts use the local cache.
- Transcript history is stored in memory and resets when the server restarts.
- `chunk_length_s=30` is set on the pipeline so files longer than 30 seconds are processed in chunks without crashing.
