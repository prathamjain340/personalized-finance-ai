# app/api/finance_routes.py
# TODO (PRODUCTION):
# - Move SESSION_STORE to Redis
# - Add session expiration
# - Add authentication middleware
# - Add rate limiting
# - Validate profile schema

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from uuid import uuid4
from time import perf_counter
import re

from finance_project.domains.finance.engine import FinanceEngine
from finance_project.domains.finance.intent import TurnControl, infer_turn_control
from finance_project.core.chat.session import ChatSession
from finance_project.core.logging.logger import log_event
from finance_project.core.postprocess.dispatcher import enqueue_profile_updates
from finance_project.core.profile.repository import get_profile, merge_profiles
from finance_project.services.profile_client import fetch_financial_profile
from finance_project.services.greeting_service import (
    build_greeting,
    get_filler_audio_base64,
    get_filler_text,
    get_greeting_audio_base64,
)
from finance_project.services.audio_service import speech_to_text, text_to_speech

router = APIRouter()

# TODO (PRODUCTION): Replace with Redis or DB-backed session store
SESSION_STORE = {}

engine = FinanceEngine()

CORE_FINANCE_FIELDS = ("monthly_income", "monthly_expenses", "monthly_savings")
PORTFOLIO_REQUIRED_FIELDS = ("monthly_income", "monthly_expenses", "monthly_savings", "existing_investments")
_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
_ARABIC_RE = re.compile(r"[\u0600-\u06FF]")


def _to_number(value) -> float | None:
    if value in (None, "", "null"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_missing_profile_field(profile: dict, field: str | None) -> bool:
    if not field:
        return False
    if field in CORE_FINANCE_FIELDS:
        value = _to_number((profile or {}).get(field))
        return value is None or value <= 0
    value = (profile or {}).get(field)
    return value in (None, "", "null")


def _sanitize_profile(profile: dict | None) -> dict:
    normalized = dict(profile or {})
    for field in CORE_FINANCE_FIELDS:
        value = _to_number(normalized.get(field))
        if value is None or value <= 0:
            normalized[field] = None
        elif value.is_integer():
            normalized[field] = int(value)
        else:
            normalized[field] = value
    return normalized


def _non_empty_fields(payload: dict | None) -> dict:
    if not isinstance(payload, dict):
        return {}

    cleaned = {}
    for field, value in payload.items():
        if value is None:
            continue
        if isinstance(value, str) and value.strip().lower() in {"", "null"}:
            continue
        cleaned[field] = value
    return cleaned


def _context_window_for_scope(scope: str | None, response_channel: str) -> int:
    if response_channel == "voice":
        return 1
    normalized = str(scope or "general").strip().lower()
    if normalized == "profile_specific":
        return 2
    if normalized == "hybrid":
        return 2
    return 1


def _has_concrete_live_slot(turn_control: TurnControl) -> bool:
    slots = turn_control.live_data_slots or {}
    return bool(slots.get("ticker") or slots.get("company") or slots.get("location") or slots.get("topic"))


def _should_rewrite_live_followup(turn_control: TurnControl) -> bool:
    if turn_control.profile_updates:
        return False
    kind = turn_control.live_data_kind
    if kind == "weather":
        return not (turn_control.live_data_slots or {}).get("location")
    if kind in {"stock", "stock_history"}:
        slots = turn_control.live_data_slots or {}
        return not (slots.get("ticker") or slots.get("company"))
    return False


def _rewrite_live_followup_query(user_message: str, turn_control: TurnControl) -> str:
    message = str(user_message or "").strip()
    if not message:
        return message

    if len(message) > 80:
        return message

    kind = turn_control.live_data_kind
    if kind == "weather":
        return f"weather in {message}"
    if kind == "stock":
        return f"stock price of {message}"
    if kind == "stock_history":
        return f"stock history of {message}"
    return message


def _is_portfolio_collection_turn(session: ChatSession, turn_control: TurnControl) -> bool:
    if str(turn_control.routing.financial_category or "").lower() == "investment_advice":
        return True
    return bool(session.profile_collection_goal == "portfolio_planning")


def _refresh_portfolio_collection_state(session: ChatSession, turn_control: TurnControl) -> None:
    should_collect = _is_portfolio_collection_turn(session, turn_control)
    if should_collect and session.profile_collection_goal is None:
        session.profile_collection_goal = "portfolio_planning"

    if session.profile_collection_goal != "portfolio_planning":
        return

    missing = [field for field in PORTFOLIO_REQUIRED_FIELDS if _is_missing_profile_field(session.profile, field)]
    session.profile_collection_queue = missing

    if missing:
        session.pending_field = missing[0]
        return

    session.profile_collection_goal = None
    session.profile_collection_queue = []
    if session.pending_field in PORTFOLIO_REQUIRED_FIELDS:
        session.pending_field = None


def _sanitize_turn_control_for_profile_flow(
    session: ChatSession,
    turn_control: TurnControl,
) -> tuple[TurnControl, bool, bool]:
    live_kind = turn_control.live_data_kind
    live_slots = dict(turn_control.live_data_slots or {})
    has_core_profile_update = any(field in (turn_control.profile_updates or {}) for field in PORTFOLIO_REQUIRED_FIELDS)
    is_portfolio_turn = _is_portfolio_collection_turn(session, turn_control)
    has_concrete_live_slot = _has_concrete_live_slot(turn_control)

    route_guard_applied = False
    enable_live_data = True

    if has_core_profile_update:
        live_kind = None
        live_slots = {}
        enable_live_data = False
        route_guard_applied = True
    elif is_portfolio_turn:
        if live_kind and has_concrete_live_slot:
            enable_live_data = True
        else:
            live_kind = None
            live_slots = {}
            enable_live_data = False
            route_guard_applied = True

    if live_kind == turn_control.live_data_kind and live_slots == (turn_control.live_data_slots or {}):
        return turn_control, route_guard_applied, enable_live_data

    guarded_turn_control = TurnControl(
        routing=turn_control.routing,
        profile_updates=turn_control.profile_updates,
        active_goal=turn_control.active_goal,
        pending_field=turn_control.pending_field,
        question_scope=turn_control.question_scope,
        self_knowledge_request=turn_control.self_knowledge_request,
        live_data_kind=live_kind,
        live_data_slots=live_slots,
    )
    return guarded_turn_control, route_guard_applied, enable_live_data


def _expected_response_language(user_text: str) -> str:
    text = str(user_text or "").strip()
    if _DEVANAGARI_RE.search(text) or _ARABIC_RE.search(text):
        return "hi"
    if not re.search(r"[A-Za-z]", text):
        return "hi"
    return "en"


def _enforce_response_script(response: str, expected_language: str) -> str:
    text = str(response or "").strip()
    if not text:
        return text

    has_non_latin_script = bool(_DEVANAGARI_RE.search(text) or _ARABIC_RE.search(text))

    if expected_language == "hi":
        if has_non_latin_script:
            return "Main Hinglish mein jawab deti hoon. Kripya apna sawaal phir se poochho."
        return text

    if expected_language == "en":
        if has_non_latin_script:
            return "Please ask again and I will respond in English."
        return text

    return text


def _apply_turn_control(session: ChatSession, turn_control: TurnControl) -> None:
    if not isinstance(session.profile, dict):
        session.profile = {}

    updates = turn_control.profile_updates or {}
    previous_goal = session.active_goal

    # Update short-term conversation state first.
    if turn_control.active_goal:
        session.active_goal = turn_control.active_goal

    for field, amount in updates.items():
        session.profile[field] = amount
        if field in CORE_FINANCE_FIELDS:
            _recompute_derived_financials(session.profile, updated_field=field)

    # Resolve pending field if answered.
    if session.pending_field and session.pending_field in updates:
        session.pending_field = None

    # Guard against stale pending hints when profile already has this value.
    if session.pending_field and not _is_missing_profile_field(session.profile, session.pending_field):
        session.pending_field = None

    # Keep latest pending field hint if controller provided one.
    if turn_control.pending_field:
        if _is_missing_profile_field(session.profile, turn_control.pending_field):
            session.pending_field = turn_control.pending_field
        elif session.pending_field == turn_control.pending_field:
            session.pending_field = None
    elif turn_control.active_goal and turn_control.active_goal != previous_goal:
        session.pending_field = None


def _recompute_derived_financials(profile: dict, updated_field: str) -> None:
    """
    Keep derived savings coherent when income/expenses are provided in chat.
    """
    if updated_field == "monthly_savings":
        return

    income = _to_number(profile.get("monthly_income"))
    expenses = _to_number(profile.get("monthly_expenses"))
    savings = _to_number(profile.get("monthly_savings"))

    if income is None or expenses is None:
        return

    if savings in (None, 0):
        derived_savings = income - expenses
        profile["monthly_savings"] = int(derived_savings) if float(derived_savings).is_integer() else derived_savings


# -----------------------------
# Request Models
# -----------------------------

class InitSessionRequest(BaseModel):
    user_id: str


class AskRequest(BaseModel):
    session_id: str
    message: str


class AskVoiceRequest(BaseModel):
    session_id: str
    audio_base64: str
    audio_mime_type: str | None = None
    audio_filename: str | None = None


# -----------------------------
# Initialize Session
# -----------------------------

@router.post("/session/init")
def init_session(request: InitSessionRequest):
    api_profile = _sanitize_profile(fetch_financial_profile(request.user_id))
    persistent_profile = _sanitize_profile(get_profile(request.user_id))
    profile = _sanitize_profile(merge_profiles(api_profile, persistent_profile))

    # if not is_valid_profile(profile):
    #     raise HTTPException(
    #         status_code=404,
    #         detail="Financial profile not found or incomplete."
    #     )

    session_id = str(uuid4())

    session = ChatSession(
        user_id=request.user_id,
        domain="finance"
    )

    name = profile.get("full_name") or profile.get("name")
    session.user_name = name if name else None
    session.greeted = False

    # Store profile inside session (cache for conversation)
    session.profile = profile

    SESSION_STORE[session_id] = session

    enqueue_profile_updates(
        user_id=request.user_id,
        updates=_non_empty_fields(api_profile),
        source="api",
        confidence=0.9,
    )

    greeting = build_greeting(session.user_name)
    greeting_audio_base64 = get_greeting_audio_base64(session.user_name)
    filler_text = get_filler_text()
    filler_audio_base64 = get_filler_audio_base64(filler_text)
    session.has_started = True
    session.greeted = True
    session.last_assistant_response = greeting

    return {
        "session_id": session_id,
        "greeting": greeting,
        "greeting_audio_base64": greeting_audio_base64,
        "filler_text": filler_text,
        "filler_audio_base64": filler_audio_base64,
        "message": "Session initialized successfully."
    }


# -----------------------------
# Ask Question (Text Only)
# -----------------------------

@router.post("/session/ask")
def ask_question(request: AskRequest):
    request_start = perf_counter()
    session = SESSION_STORE.get(request.session_id)

    if not session:
        raise HTTPException(status_code=404, detail="Invalid session")

    turn_control = infer_turn_control(
        raw_query=request.message,
        last_assistant_text=session.last_assistant_response,
        active_goal=session.active_goal,
        pending_field=session.pending_field,
    )
    turn_control, route_guard_applied, enable_live_data = _sanitize_turn_control_for_profile_flow(session, turn_control)
    effective_query = request.message
    if not route_guard_applied and turn_control.live_data_kind and _should_rewrite_live_followup(turn_control):
        effective_query = _rewrite_live_followup_query(request.message, turn_control)
    _apply_turn_control(session, turn_control)
    _refresh_portfolio_collection_state(session, turn_control)
    enqueue_profile_updates(
        user_id=session.user_id,
        updates=turn_control.profile_updates,
        source="explicit",
        confidence=0.95,
    )

    contextual_query = session.build_query(
        request.message,
        max_turns=_context_window_for_scope(turn_control.question_scope, response_channel="text"),
    )

    engine_start = perf_counter()
    expected_language = _expected_response_language(request.message)
    response, memories_used = engine.handle_request(
        user_id=session.user_id,
        raw_query=contextual_query,
        current_query=effective_query,
        last_assistant_response=session.last_assistant_response,
        routing=turn_control.routing,
        self_knowledge_request=turn_control.self_knowledge_request,
        active_goal=session.active_goal,
        pending_field=session.pending_field,
        question_scope=turn_control.question_scope,
        profile=session.profile,
        session_memory_usage=session.session_memory_usage,
        conversation_stage=session.stage,
        response_channel="text",
        live_data_kind=turn_control.live_data_kind,
        live_data_slots=turn_control.live_data_slots,
        enable_live_data=enable_live_data,
    )
    response = _enforce_response_script(response, expected_language)
    text_generation_ms = round((perf_counter() - engine_start) * 1000, 2)

    session.update(
        user_message=request.message,
        assistant_response=response,
        memories_used=memories_used
    )

    log_event(
        event="request_timing",
        metadata={
            "route": "text",
            "user_id": session.user_id,
            "session_id": request.session_id,
            "text_generation_ms": text_generation_ms,
            "route_guard_applied": route_guard_applied,
            "total_request_ms": round((perf_counter() - request_start) * 1000, 2),
        },
    )

    return {
        "response": response,
        "conversation_stage": session.stage
    }


# -----------------------------
# Ask Question (Voice)
# -----------------------------

@router.post("/session/ask-voice")
def ask_question_voice(request: AskVoiceRequest):
    request_start = perf_counter()
    session = SESSION_STORE.get(request.session_id)

    if not session:
        raise HTTPException(status_code=404, detail="Invalid session")

    # 1) STT
    stt_start = perf_counter()
    try:
        user_text = speech_to_text(
            request.audio_base64,
            mime_type=request.audio_mime_type,
            filename=request.audio_filename,
        )
    except Exception as exc:
        log_event(
            event="stt_failed",
            level="WARNING",
            metadata={
                "user_id": session.user_id,
                "session_id": request.session_id,
                "audio_mime_type": request.audio_mime_type,
                "audio_filename": request.audio_filename,
                "error": str(exc),
            },
        )
        raise HTTPException(
            status_code=400,
            detail="Unable to transcribe this audio. Please retry with clear speech or upload wav/mp3/m4a/webm.",
        )
    stt_ms = round((perf_counter() - stt_start) * 1000, 2)

    turn_control_start = perf_counter()
    turn_control = infer_turn_control(
        raw_query=user_text,
        last_assistant_text=session.last_assistant_response,
        active_goal=session.active_goal,
        pending_field=session.pending_field,
    )
    turn_control, route_guard_applied, enable_live_data = _sanitize_turn_control_for_profile_flow(session, turn_control)
    effective_query = user_text
    if not route_guard_applied and turn_control.live_data_kind and _should_rewrite_live_followup(turn_control):
        effective_query = _rewrite_live_followup_query(user_text, turn_control)
    _apply_turn_control(session, turn_control)
    _refresh_portfolio_collection_state(session, turn_control)
    enqueue_profile_updates(
        user_id=session.user_id,
        updates=turn_control.profile_updates,
        source="explicit",
        confidence=0.95,
    )
    turn_control_ms = round((perf_counter() - turn_control_start) * 1000, 2)
    expected_language = _expected_response_language(user_text)

    # 2) Build contextual query
    contextual_query = session.build_query(
        user_text,
        max_turns=_context_window_for_scope(turn_control.question_scope, response_channel="voice"),
    )

    engine_start = perf_counter()
    primary_response, memories_used = engine.handle_request(
        user_id=session.user_id,
        raw_query=contextual_query,
        current_query=effective_query,
        last_assistant_response=session.last_assistant_response,
        routing=turn_control.routing,
        self_knowledge_request=turn_control.self_knowledge_request,
        active_goal=session.active_goal,
        pending_field=session.pending_field,
        question_scope=turn_control.question_scope,
        profile=session.profile,
        session_memory_usage=session.session_memory_usage,
        conversation_stage=session.stage,
        response_channel="voice",
        apply_voice_trim=False,
        live_data_kind=turn_control.live_data_kind,
        live_data_slots=turn_control.live_data_slots,
        enable_live_data=enable_live_data,
    )
    text_generation_ms = round((perf_counter() - engine_start) * 1000, 2)

    # 3) Voice rendering pass removed to reduce latency.
    voice_render_ms = 0.0
    response = _enforce_response_script(primary_response, expected_language)
    response = " ".join(str(response or "").split()).strip()
    if not response:
        response = "Could you please repeat that?"

    # 3) Update session
    session.update(
        user_message=user_text,
        assistant_response=response,
        memories_used=memories_used
    )

    # 4) TTS
    tts_start = perf_counter()
    audio_response_base64 = text_to_speech(response)
    tts_generation_ms = round((perf_counter() - tts_start) * 1000, 2)
    total_request_ms = round((perf_counter() - request_start) * 1000, 2)
    other_overhead_ms = round(
        max(0.0, total_request_ms - stt_ms - turn_control_ms - text_generation_ms - voice_render_ms - tts_generation_ms),
        2,
    )

    log_event(
        event="request_timing",
        metadata={
            "route": "voice",
            "user_id": session.user_id,
            "session_id": request.session_id,
            "stt_ms": stt_ms,
            "turn_control_ms": turn_control_ms,
            "text_generation_ms": text_generation_ms,
            "voice_render_ms": voice_render_ms,
            "tts_generation_ms": tts_generation_ms,
            "other_overhead_ms": other_overhead_ms,
            "total_request_ms": total_request_ms,
            "route_guard_applied": route_guard_applied,
        },
    )

    return {
        "transcript": user_text,
        "response_text": response,
        "response_audio_base64": audio_response_base64,
        "conversation_stage": session.stage,
        "timing": {
            "stt_ms": stt_ms,
            "turn_control_ms": turn_control_ms,
            "text_generation_ms": text_generation_ms,
            "voice_render_ms": voice_render_ms,
            "tts_generation_ms": tts_generation_ms,
            "other_overhead_ms": other_overhead_ms,
            "total_request_ms": total_request_ms,
        },
    }
