from finance_project.services.audio_service import text_to_speech


DEFAULT_GREETING_NAME = "Guest"
DEFAULT_FILLER_TEXT = "Give me a moment while I work that out for you."
_GREETING_AUDIO_CACHE: dict[str, str] = {}


def _first_name(name: str | None) -> str:
    if not name:
        return DEFAULT_GREETING_NAME
    token = name.strip().split()[0]
    return token if token else DEFAULT_GREETING_NAME


def build_greeting(name: str | None) -> str:
    opening = f"Hi {_first_name(name)}"
    return f"{opening}, how can I help you today?"


def get_greeting_audio_base64(name: str | None) -> str:
    """
    Returns cached TTS for greeting.
    - Guest greeting is cached and reused.
    - Personalized greeting is generated only when a user name is present.
    """
    greeting_text = build_greeting(name)

    cached = _GREETING_AUDIO_CACHE.get(greeting_text)
    if cached:
        return cached

    try:
        encoded = text_to_speech(greeting_text)
    except Exception:
        # Fallback to guest greeting audio if personalized TTS fails.
        guest_text = build_greeting(None)
        guest_cached = _GREETING_AUDIO_CACHE.get(guest_text)
        if guest_cached:
            return guest_cached
        encoded = text_to_speech(guest_text)
        _GREETING_AUDIO_CACHE[guest_text] = encoded
        return encoded

    _GREETING_AUDIO_CACHE[greeting_text] = encoded
    return encoded


def get_filler_text() -> str:
    return DEFAULT_FILLER_TEXT


def get_filler_audio_base64(text: str | None = None) -> str:
    """
    Returns cached filler TTS clip.
    This can be played by the client while waiting for the final response.
    """
    filler_text = (text or DEFAULT_FILLER_TEXT).strip() or DEFAULT_FILLER_TEXT
    cached = _GREETING_AUDIO_CACHE.get(filler_text)
    if cached:
        return cached

    encoded = text_to_speech(filler_text)
    _GREETING_AUDIO_CACHE[filler_text] = encoded
    return encoded


# Pre-warm guest greeting audio once for faster init responses.
try:
    _GREETING_AUDIO_CACHE[build_greeting(None)] = text_to_speech(build_greeting(None))
except Exception:
    # Avoid startup failure if TTS is temporarily unavailable.
    pass

# Pre-warm filler audio once so first voice turn can use it immediately.
try:
    _GREETING_AUDIO_CACHE[DEFAULT_FILLER_TEXT] = text_to_speech(DEFAULT_FILLER_TEXT)
except Exception:
    pass
