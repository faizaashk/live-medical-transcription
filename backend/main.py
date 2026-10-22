"""Live Medical Transcription - backend.

POST /api/transcribe  : one WAV speech segment  -> {"text": "..."}
POST /api/summarize   : full transcript         -> structured ClinicalSummary
GET  /                : serves the frontend (../frontend)
"""
import json
import os
import struct
from pathlib import Path

import numpy as np

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from faster_whisper import WhisperModel

from schemas import ClinicalSummary, TranscriptIn

load_dotenv()

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")

# Which LLM to use for the summary: "openai" (works for OpenAI, Groq, Gemini and
# local Ollama, since they all speak the OpenAI API format) or "anthropic".
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
LLM_BASE_URL = os.getenv("LLM_BASE_URL") or None
LLM_API_KEY = os.getenv("LLM_API_KEY", "")

app = FastAPI(title="Live Medical Transcription")

# Load once at startup (first run downloads the model weights).
asr = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")

# ---------------------------------------------------------------- ASR ------


def wav_to_float32(data: bytes) -> np.ndarray:
    """Decode a WAV file (16-bit PCM or 32-bit float) to 16 kHz mono float32.

    Done by hand so we don't depend on PyAV/ffmpeg, which caused version errors.
    """
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("Not a WAV file")
    pos, fmt, pcm = 12, None, None
    while pos + 8 <= len(data):
        cid = data[pos:pos + 4]
        size = struct.unpack("<I", data[pos + 4:pos + 8])[0]
        body = data[pos + 8:pos + 8 + size]
        if cid == b"fmt ":
            fmt = struct.unpack("<HHIIHH", body[:16])  # tag, channels, rate, _, _, bits
        elif cid == b"data":
            pcm = body
            break
        pos += 8 + size + (size & 1)
    if fmt is None or pcm is None:
        raise ValueError("Incomplete WAV file")
    tag, channels, rate, _, _, bits = fmt

    if tag == 3 and bits == 32:
        audio = np.frombuffer(pcm[: len(pcm) // 4 * 4], dtype="<f4")
    elif tag == 1 and bits == 16:
        audio = np.frombuffer(pcm[: len(pcm) // 2 * 2], dtype="<i2").astype(np.float32) / 32768.0
    else:
        raise ValueError(f"Unsupported WAV format (tag={tag}, bits={bits})")

    if channels > 1:
        audio = audio[: len(audio) // channels * channels].reshape(-1, channels).mean(axis=1)
    audio = audio.astype(np.float32)
    if rate != 16000 and len(audio):
        n = int(len(audio) * 16000 / rate)
        audio = np.interp(np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio).astype(np.float32)
    return audio


@app.post("/api/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    data = await audio.read()
    try:
        samples = wav_to_float32(data)
    except ValueError as e:
        raise HTTPException(400, f"Bad audio chunk: {e}")
    if len(samples) < 4000:  # under ~0.25 s, too short to contain real speech
        return {"text": ""}

    # VAD already ran in the browser, so we don't need Whisper's own filter.
    segments, _info = asr.transcribe(
        samples, language="en", beam_size=1, vad_filter=False,
        condition_on_previous_text=False,
    )
    text = " ".join(s.text.strip() for s in segments).strip()
    return {"text": text}


# ---------------------------------------------------------------- LLM ------

SYSTEM_PROMPT = """You are a clinical documentation assistant. You read a
transcript of a doctor-patient consultation and extract structured information
for a medical record.

Rules:
- Use ONLY information stated in the transcript. Never invent or infer
  diagnoses, doses, vitals or history.
- If something was not mentioned, use null (for single values) or [] (for lists).
- "symptoms.positive" = symptoms the patient has; "symptoms.negative" =
  symptoms the patient explicitly denied. Keep them separate.
- "chief_complaint" should use the patient's own words where possible.
- "assessment" is what the CLINICIAN stated. If no diagnosis was stated, null.
- The transcript may contain speech-recognition errors; fix obvious ones only
  when the meaning is unambiguous.
- "clinical_summary" is 2-4 sentences, written for a medical record.
- Output ONLY one valid JSON object, no markdown fences, no commentary."""

SCHEMA_EXAMPLE = {
    "patient": {"name": None, "age": None, "sex": None, "identifiers": []},
    "chief_complaint": None,
    "hpi": {"onset": None, "duration": None, "progression": None,
            "aggravating": [], "relieving": []},
    "symptoms": {"positive": [], "negative": []},
    "past_medical_history": {"conditions": [], "surgeries": [], "hospitalisations": []},
    "medications": {"current": [{"name": "", "dose": None, "adherence": None}],
                    "allergies": []},
    "observations": {"vitals": [], "exam_findings": []},
    "assessment": {"provisional": None, "differentials": []},
    "plan": {"investigations": [], "prescriptions": [], "advice": [], "follow_up": None},
    "clinical_summary": "",
}


def _user_prompt(transcript: str) -> str:
    return (
        "Return JSON matching this shape:\n"
        f"{json.dumps(SCHEMA_EXAMPLE, indent=2)}\n\n"
        f"Transcript:\n\"\"\"\n{transcript}\n\"\"\""
    )


def _call_llm(transcript: str) -> str:
    """Send the transcript to the configured LLM and return its raw text reply."""
    if LLM_PROVIDER == "anthropic":
        import anthropic

        key = LLM_API_KEY or os.getenv("ANTHROPIC_API_KEY", "")
        if not key:
            raise HTTPException(500, "Set LLM_API_KEY (or ANTHROPIC_API_KEY) in backend/.env")
        client = anthropic.Anthropic(api_key=key)
        msg = client.messages.create(
            model=LLM_MODEL, max_tokens=4000, temperature=0,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": _user_prompt(transcript)}],
        )
        return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")

    # OpenAI-compatible providers: OpenAI, Groq, Gemini, Ollama, ...
    from openai import OpenAI

    # Local servers such as Ollama need no real key, but the client wants a value.
    if not LLM_API_KEY and not (LLM_BASE_URL and "localhost" in LLM_BASE_URL):
        raise HTTPException(500, "Set LLM_API_KEY in backend/.env")
    client = OpenAI(api_key=LLM_API_KEY or "not-needed", base_url=LLM_BASE_URL)
    resp = client.chat.completions.create(
        model=LLM_MODEL, temperature=0, max_tokens=4000,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _user_prompt(transcript)},
        ],
    )
    return resp.choices[0].message.content or ""


def _parse_json(raw: str) -> dict:
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object found")
    return json.loads(raw[start:end + 1])


@app.post("/api/summarize", response_model=ClinicalSummary)
async def summarize(body: TranscriptIn):
    text = body.text.strip()
    if len(text.split()) < 5:
        raise HTTPException(400, "Transcript is too short to summarise.")

    last_err = None
    for _ in range(2):  # one retry if the model returns malformed JSON
        try:
            data = _parse_json(_call_llm(text))
            return ClinicalSummary.model_validate(data)
        except HTTPException:
            raise
        except Exception as e:  # noqa: BLE001
            last_err = e
    raise HTTPException(502, f"Could not parse the LLM response: {last_err}")


# ----------------------------------------------------------- Frontend ------

FRONTEND = Path(__file__).resolve().parent.parent / "frontend"
app.mount("/", StaticFiles(directory=FRONTEND, html=True), name="frontend")