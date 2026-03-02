import json
from typing import List, Optional

from finance_project.core.memory.types import Memory
from finance_project.domains.finance.prompt.base import (
    ADVICE_GUARDRAIL,
    ASSISTANT_NAME,
    ASSISTANT_PERSONA,
    SAFETY_BOUNDARY,
)


def _clip_text(text: str, max_chars: int) -> str:
    normalized = " ".join(str(text or "").split())
    if len(normalized) <= max_chars:
        return normalized
    return normalized[:max_chars].rstrip() + "..."


def _compact_profile(profile: dict) -> dict:
    if not isinstance(profile, dict) or not profile:
        return {}

    preferred_keys = (
        "preferred_name",
        "full_name",
        "name",
        "age",
        "city",
        "occupation",
        "monthly_income",
        "monthly_expenses",
        "monthly_savings",
        "existing_investments",
        "has_health_insurance",
        "has_life_insurance",
        "children_count",
        "parents_dependent",
        "dependents",
    )

    compact: dict = {}
    for key in preferred_keys:
        if key not in profile:
            continue
        value = profile.get(key)
        if value in (None, "", "null"):
            continue
        if isinstance(value, (list, tuple, set, dict)):
            serialized = json.dumps(value, ensure_ascii=True, default=str)
            compact[key] = _clip_text(serialized, 120)
        else:
            compact[key] = value

    if compact:
        return compact

    # Fallback: include up to 8 non-empty scalar fields.
    for key, value in profile.items():
        if value in (None, "", "null"):
            continue
        if isinstance(value, (list, tuple, set, dict)):
            continue
        compact[str(key)] = value
        if len(compact) >= 8:
            break
    return compact


def assemble_prompt(
    raw_query: str,
    intent: str,
    profile: dict,
    signals: dict,
    domain_summary: Optional[str],
    memories: List[Memory],
    response_mode: str,
    question_scope: str,
    conversation_stage,
    missing_fields: Optional[List[str]] = None,
    profile_state: Optional[str] = None,
    previous_assistant_response: Optional[str] = None,
    active_goal: Optional[str] = None,
    pending_field: Optional[str] = None,
    response_channel: str = "text",
) -> str:
    compact_query = _clip_text(raw_query, max_chars=900)
    prompt_sections = [
        _role_and_boundaries(),
        _user_context(
            profile=profile,
            signals=signals,
            domain_summary=domain_summary,
            memories=memories,
        ),
        _task_definition(raw_query=compact_query, intent=intent),
        _scope_rules(question_scope=question_scope),
        _response_rules(response_mode=response_mode, response_channel=response_channel),
        _personalization_rules(profile=profile),
        _conversation_stage_rules(stage=conversation_stage),
        _verbosity_rules(stage=conversation_stage.value),
        _data_completeness_rules(
            profile_state=profile_state,
            missing_fields=missing_fields or [],
        ),
        _goal_state_rules(
            active_goal=active_goal,
            pending_field=pending_field,
        ),
        _continuity_rules(previous_assistant_response=previous_assistant_response),
    ]
    return "\n\n".join(part for part in prompt_sections if part)


def _role_and_boundaries() -> str:
    return (
        f"{ASSISTANT_PERSONA}\n"
        f"{SAFETY_BOUNDARY}\n"
        f"{ADVICE_GUARDRAIL}\n"
        "When needed, clearly state assumptions before advice."
    )


def _user_context(
    profile: dict,
    signals: dict,
    domain_summary: Optional[str],
    memories: List[Memory],
) -> str:
    sections = ["KNOWN USER CONTEXT:"]
    compact_profile = _compact_profile(profile)
    if compact_profile:
        sections.append(f"- Profile data: {compact_profile}")
    if signals:
        sections.append(f"- Derived financial signals: {signals}")
    if domain_summary:
        sections.append(f"- Finance summary: {_clip_text(domain_summary, 280)}")
    if memories:
        sections.append("- Memory cues (internal, do not quote verbatim):")
        for memory in memories[:3]:
            sections.append(f"  - {_clip_text(memory.content, 110)}")
    return "\n".join(sections)


def _task_definition(raw_query: str, intent: str) -> str:
    return (
        f"USER MESSAGE:\n{raw_query}\n\n"
        f"INTENT: {intent}\n"
        "Answer this exact turn first, then add one natural next step only if useful."
    )


def _scope_rules(question_scope: str) -> str:
    scope = (question_scope or "general").strip().lower()
    if scope == "profile_specific":
        return (
            "QUESTION SCOPE:\n"
            "- Scope is profile_specific.\n"
            "- Lead with a direct personalized answer first.\n"
            "- Keep it concise: answer + one practical next step."
        )
    if scope == "hybrid":
        return (
            "QUESTION SCOPE:\n"
            "- Scope is hybrid.\n"
            "- Give a brief concept answer, then one line tailored to the user's profile."
        )
    return (
        "QUESTION SCOPE:\n"
        "- Scope is general.\n"
        "- Prioritize clarity and explanation over deep personalization."
    )


def _response_rules(response_mode: str, response_channel: str = "text") -> str:
    common = (
        "RESPONSE RULES:\n"
        f"- Speak in first person as {ASSISTANT_NAME} only if asked your name; otherwise stay natural\n"
        "- Sound human and confident, not scripted\n"
        "- Keep the answer specific to the latest user message\n"
        "- Avoid repetitive phrasing and repeated profile labels\n"
        "- If user asks for a simple fact, answer directly first\n"
        "- Ask at most one follow-up question, and only when it improves advice quality\n"
        "- Prefer short paragraphs; use bullets only when they add real clarity\n"
        "- Default to INR/rupees unless user specifies another currency\n"
    )

    mode_specific = {
        "direct_framework": (
            "- Give a clear recommendation path with a brief reason.\n"
            "- Keep it actionable."
        ),
        "assumptive_with_clarifier": (
            "- Give best-effort guidance with one explicit assumption.\n"
            "- End with one focused clarifying question."
        ),
        "clarifying_first": (
            "- Briefly explain why guidance is limited.\n"
            "- Ask one focused question to unblock better advice."
        ),
        "reflective_preventive": (
            "- Acknowledge concern calmly.\n"
            "- Give grounded steps that reduce risk."
        ),
        "educational_redirect": (
            "- Decline unsafe asks briefly.\n"
            "- Redirect to safe educational guidance."
        ),
    }

    channel_rules = ""
    if response_channel == "voice":
        channel_rules = (
            "VOICE RULES:\n"
            "- Keep it compact and easy to speak\n"
            "- Use complete sentences only (no sentence fragments)\n"
            "- Prefer 1 short sentence, max 2\n"
            "- Target roughly 12-22 words total\n"
            "- For Hindi/Hinglish, keep language conversational and avoid long list-style narration\n"
            "- No bullet points"
        )
    else:
        channel_rules = (
            "TEXT RULES:\n"
            "- Keep most responses to 2-5 short lines\n"
            "- Avoid long blocks unless user explicitly asks for detail"
        )

    return (
        common
        + "\n"
        + mode_specific.get(response_mode, "- Keep it concise and practical.")
        + "\n"
        + channel_rules
    )


def _personalization_rules(profile: dict) -> str:
    has_financial_profile = (
        bool(profile)
        and profile.get("monthly_income") not in (None, "")
        and profile.get("monthly_expenses") not in (None, "")
    )

    if has_financial_profile:
        return (
            "PERSONALIZATION:\n"
            "- Personalize advice using relevant profile and memory context.\n"
            "- Use personal details only when they improve this answer."
        )

    return (
        "PERSONALIZATION:\n"
        "- Give general guidance without pretending to know missing facts.\n"
        "- Mention uncertainty naturally without repeating the same disclaimer wording."
    )


def _conversation_stage_rules(stage) -> str:
    return (
        "CONVERSATION STATE:\n"
        f"- Current stage: {stage}\n"
        "- Continue existing thread unless user clearly switches topic.\n"
        "- Move the conversation forward; do not reset framing every turn."
    )


def _verbosity_rules(stage) -> str:
    limits = {
        "initial_guidance": "Keep it short and clear. One practical step is enough.",
        "explanation": "Explain simply with one short example only if needed.",
        "comparison": "Compare options directly and succinctly.",
        "decision_support": "Be concise and decision-oriented.",
        "execution_help": "Give practical next steps in minimal words.",
        "wrap_up": "Keep it very brief.",
    }
    return f"LENGTH GUIDANCE:\n- {limits.get(stage, 'Be concise and natural.')}"


def _data_completeness_rules(profile_state: Optional[str], missing_fields: List[str]) -> str:
    if not profile_state:
        return ""
    if profile_state == "partial_data" and missing_fields:
        return (
            "DATA STATE:\n"
            f"- Missing details: {', '.join(missing_fields)}\n"
            "- Give best-effort guidance, then ask one targeted follow-up."
        )
    if profile_state == "insufficient_data" and missing_fields:
        return (
            "DATA STATE:\n"
            f"- Missing details: {', '.join(missing_fields)}\n"
            "- Give a brief provisional response and ask one critical question."
        )
    return (
        "DATA STATE:\n"
        "- Use available context confidently without over-claiming certainty."
    )


def _goal_state_rules(active_goal: Optional[str], pending_field: Optional[str]) -> str:
    return (
        "GOAL STATE:\n"
        f"- Active goal: {active_goal or 'none'}\n"
        f"- Pending field: {pending_field or 'none'}\n"
        "- Prioritize the active goal unless user clearly changes topic.\n"
        "- If pending field is still needed, ask for it naturally."
    )


def _continuity_rules(previous_assistant_response: Optional[str]) -> str:
    if not previous_assistant_response:
        return "CONTINUITY:\n- No previous assistant turn; respond fresh."
    clipped = previous_assistant_response.strip()
    if len(clipped) > 320:
        clipped = clipped[:320].rstrip() + "..."
    return (
        "CONTINUITY:\n"
        "- Previous assistant response (for thread continuity and anti-repetition):\n"
        f"{clipped}\n"
        "- Do not reuse the same framing unless user asks for repetition."
    )


def assemble_clarification_prompt(
    raw_query: str,
    missing_fields: List[str],
    profile: dict,
    response_channel: str = "text",
) -> str:
    length_rule = (
        "Respond in 1 concise complete sentence, with one short question only if essential."
        if response_channel == "voice"
        else "Respond in 2-4 short lines."
    )
    return (
        f"You are {ASSISTANT_NAME}, a personal finance guide.\n"
        f"{SAFETY_BOUNDARY}\n"
        f"User message: {raw_query}\n"
        f"Missing fields: {', '.join(missing_fields) if missing_fields else 'key financial details'}\n"
        f"Known profile: {profile}\n"
        f"{length_rule}\n"
        "Give brief provisional guidance, then ask one focused follow-up question."
    )


def assemble_out_of_domain_prompt(
    raw_query: str,
    small_talk: bool,
    response_channel: str = "text",
    profile: Optional[dict] = None,
    memories: Optional[List[Memory]] = None,
) -> str:
    profile = profile or {}
    memories = memories or []
    known_name = profile.get("preferred_name") or profile.get("full_name") or profile.get("name")
    name_hint = f"Known user name: {known_name}" if known_name else "Known user name: unknown"
    memory_lines = [memory.content for memory in memories[:3]]
    memory_hint = f"Known memory cues: {memory_lines}" if memory_lines else "Known memory cues: none"

    if small_talk:
        length_rule = (
            "Reply in one short complete sentence, optionally with a soft bridge to finance."
            if response_channel == "voice"
            else "Reply briefly in 1-2 lines, friendly and human."
        )
        return (
            f"You are {ASSISTANT_NAME}, a personal finance assistant.\n"
            f"{name_hint}\n"
            f"{memory_hint}\n"
            f"User message: {raw_query}\n"
            f"{length_rule}\n"
            "Acknowledge naturally. Keep it warm.\n"
            "If user asks what you know about them, answer from known memory cues and profile only.\n"
            "Do not force a finance redirect in every reply.\n"
            "Only add a finance offer if the user asks for help or if it fits naturally."
        )

    length_rule = (
        "Reply in 1 concise complete sentence, max 2 if needed."
        if response_channel == "voice"
        else "Reply in 2-3 short lines."
    )
    return (
        f"You are {ASSISTANT_NAME}, a finance-first assistant.\n"
        f"{SAFETY_BOUNDARY}\n"
        f"{name_hint}\n"
        f"{memory_hint}\n"
        f"User message: {raw_query}\n"
        f"{length_rule}\n"
        "If the request is harmless but non-finance, answer directly and briefly.\n"
        "Do not add a finance segue by default.\n"
        "You may add one short optional finance offer only when it is contextually natural.\n"
        "If user asks a personal-memory question, answer directly from known memory cues without inventing.\n"
        "If the request is explicit/unsafe/illegal, decline briefly and redirect to safe topics."
    )


def assemble_live_data_prompt(
    raw_query: str,
    live_data: dict,
    response_channel: str = "text",
    profile: Optional[dict] = None,
) -> str:
    profile = profile or {}
    name = profile.get("preferred_name") or profile.get("full_name") or profile.get("name")
    facts = live_data.get("facts") or []
    sources = live_data.get("sources") or []
    kind = live_data.get("kind", "update")
    as_of = live_data.get("as_of", "")

    if response_channel == "voice":
        length_rule = (
            "Respond in 1 complete sentence, max 2. Keep it around 12-22 words. "
            "For Hindi/Hinglish, keep it crisp and conversational. "
            "No bullet points."
        )
    else:
        length_rule = (
            "Respond in 2-4 short lines. "
            "Use bullets only if needed for readability (max 2)."
        )

    return (
        f"You are {ASSISTANT_NAME}, a trusted finance-first assistant.\n"
        f"{SAFETY_BOUNDARY}\n"
        f"{ADVICE_GUARDRAIL}\n"
        "LIVE-UPDATE MODE: The user explicitly asked for a real-time update.\n"
        "Provide the requested live update directly, even when it is weather or general news.\n"
        "After answering, you may add one short optional bridge back to finance.\n"
        f"User name (if known): {name or 'unknown'}\n"
        f"User message: {raw_query}\n"
        f"Live data category: {kind}\n"
        f"Live data facts: {facts}\n"
        f"Sources: {sources}\n"
        f"Data timestamp (UTC): {as_of}\n"
        f"{length_rule}\n"
        "Ground the response only in provided live facts.\n"
        "If the user asks for investment action, provide caution and decision framework, not a direct buy/sell command.\n"
        "Mention source freshness briefly in natural language."
    )
