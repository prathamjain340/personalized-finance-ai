import os
import re

from finance_project.core.llm.client import generate_response


_VOICE_RENDER_ENABLED = os.getenv("VOICE_RENDER_ENABLED", "true").lower() not in {"0", "false", "off", "no"}
_FAILURE_FALLBACK_TEXT = "I am having trouble processing this request right now."


def is_voice_render_enabled() -> bool:
    return _VOICE_RENDER_ENABLED


def detect_script_hint(text: str) -> str:
    sample = str(text or "").strip()
    if not sample:
        return "unknown"

    has_devanagari = bool(re.search(r"[\u0900-\u097F]", sample))
    has_arabic = bool(re.search(r"[\u0600-\u06FF]", sample))
    has_latin = bool(re.search(r"[A-Za-z]", sample))

    if has_arabic:
        return "arabic"
    if has_devanagari:
        return "devanagari"
    if has_latin and not has_devanagari and not has_arabic:
        return "latin"
    return "mixed"


def _language_policy(script_hint: str) -> str:
    if script_hint == "arabic":
        return (
            "Language policy: user transcript is Urdu/Arabic script. "
            "Respond in Hindi using Devanagari script and do not use Arabic script."
        )
    if script_hint == "devanagari":
        return "Language policy: respond in Hindi using Devanagari script."
    if script_hint == "latin":
        return (
            "Language policy: match Latin-script style. "
            "If user intent is Hindi in Latin script, respond in Hinglish Latin. "
            "If user intent is English, respond in English."
        )
    return "Language policy: match the user's dominant script and language style."


def _build_voice_render_prompt(
    user_transcript: str,
    primary_answer: str,
    conversation_stage: str | None,
    response_channel: str,
    script_hint: str,
) -> str:
    stage = str(conversation_stage or "unknown")
    transcript = str(user_transcript or "").strip()
    answer = str(primary_answer or "").strip()

    return (
        "You are a voice response renderer for a finance assistant.\n"
        "Rewrite the primary answer for spoken delivery without changing meaning.\n"
        "Requirements:\n"
        "- Preserve all critical facts, numbers, percentages, and rupee amounts.\n"
        "- Preserve cautionary and risk notes.\n"
        "- Keep it concise and natural for speech.\n"
        "- Ensure the response is complete and ends naturally.\n"
        "- Avoid partial numbered lists unless the list is fully complete.\n"
        f"- {_language_policy(script_hint)}\n\n"
        f"Conversation stage: {stage}\n"
        f"Response channel: {response_channel}\n\n"
        f"User transcript:\n{transcript}\n\n"
        f"Primary answer:\n{answer}\n\n"
        "Return only the rewritten final response text."
    )


def render_voice_response(
    user_transcript: str,
    primary_answer: str,
    conversation_stage: str | None = None,
    response_channel: str = "voice",
) -> str:
    base = " ".join(str(primary_answer or "").split()).strip()
    if not base:
        return base

    if not is_voice_render_enabled():
        return base

    script_hint = detect_script_hint(user_transcript)
    prompt = _build_voice_render_prompt(
        user_transcript=user_transcript,
        primary_answer=base,
        conversation_stage=conversation_stage,
        response_channel=response_channel,
        script_hint=script_hint,
    )

    try:
        rendered = " ".join(str(generate_response(prompt, operation="voice_render") or "").split()).strip()
    except Exception:
        return base

    if not rendered:
        return base

    if rendered == _FAILURE_FALLBACK_TEXT or _FAILURE_FALLBACK_TEXT.lower() in rendered.lower():
        return base

    return rendered
