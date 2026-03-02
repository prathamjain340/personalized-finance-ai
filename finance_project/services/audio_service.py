# finance_project/services/audio_service.py

import base64
import io
from openai import OpenAI

client = OpenAI()


def _resolve_audio_extension(mime_type: str | None = None, filename: str | None = None) -> str:
    known = {
        "audio/wav": "wav",
        "audio/x-wav": "wav",
        "audio/mpeg": "mp3",
        "audio/mp3": "mp3",
        "audio/mp4": "m4a",
        "audio/m4a": "m4a",
        "audio/webm": "webm",
        "audio/ogg": "ogg",
    }

    mime = str(mime_type or "").strip().lower()
    if mime in known:
        return known[mime]

    name = str(filename or "").strip().lower()
    if "." in name:
        ext = name.rsplit(".", 1)[-1]
        if ext in {"wav", "mp3", "m4a", "webm", "ogg"}:
            return ext

    return "m4a"


def speech_to_text(base64_audio: str, mime_type: str | None = None, filename: str | None = None) -> str:
    """
    Converts base64 encoded audio to text using OpenAI Whisper.
    Supports mp3, wav, m4a etc.
    """

    audio_bytes = base64.b64decode(base64_audio)

    # Use in-memory buffer instead of temp file
    audio_file = io.BytesIO(audio_bytes)

    # IMPORTANT: give it a filename with appropriate extension
    extension = _resolve_audio_extension(mime_type=mime_type, filename=filename)
    audio_file.name = f"input.{extension}"

    transcript = client.audio.transcriptions.create(
        # model="gpt-4o-mini-transcribe",
        model="gpt-4o-mini-transcribe",
        file=audio_file
    )
    return transcript.text


def text_to_speech(text: str) -> str:
    """
    Converts text to base64 encoded speech using OpenAI TTS.
    """
    response = client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice="sage",
        input=text
    )

    audio_bytes = response.read()
    return base64.b64encode(audio_bytes).decode("utf-8")
