import json
import re
from typing import Any, Dict, List

from finance_project.core.llm.client import generate_response


ALLOWED_MEMORY_TYPES = {"goal", "preference", "interest", "dislike", "behavioral", "emotional"}
HIGH_SIGNAL_MEMORY_TYPES = {"goal", "preference", "interest", "dislike"}


def reflect_on_response(
    user_id: str,
    raw_query: str,
    response: str,
    intent: str,
) -> List[Dict]:
    result = reflect_on_response_with_audit(
        user_id=user_id,
        raw_query=raw_query,
        response=response,
        intent=intent,
    )
    return result["memory_candidates"]


def reflect_on_response_with_audit(
    user_id: str,
    raw_query: str,
    response: str,
    intent: str,
) -> dict[str, Any]:
    """
    Post-response reflection hook with extraction audit metadata.
    """
    _ = user_id
    _ = response
    _ = intent

    utterance = _extract_current_utterance(raw_query)
    query = " ".join((utterance or "").strip().split())
    lowered = query.lower()
    memory_candidates: list[dict] = []
    dropped_reason_counts: dict[str, int] = {}

    # --- Emotional signals (low confidence, fast decay) ---
    if any(word in lowered for word in ["scared", "afraid", "worried", "anxious", "stress", "nervous"]):
        memory_candidates.append(
            {
                "type": "emotional",
                "content": "User expressed anxiety or fear related to financial decisions.",
                "confidence": 0.35,
                "importance": 0.45,
            }
        )

    # --- Risk preference signals (medium confidence) ---
    if any(phrase in lowered for phrase in ["safe", "secure", "low risk", "stable", "conservative"]):
        memory_candidates.append(
            {
                "type": "preference",
                "content": "User shows preference for lower-risk financial options.",
                "confidence": 0.65,
                "importance": 0.65,
            }
        )

    # --- Learning style signals ---
    if any(phrase in lowered for phrase in ["explain", "understand", "step by step", "in detail"]):
        memory_candidates.append(
            {
                "type": "preference",
                "content": "User prefers clear explanations before making financial decisions.",
                "confidence": 0.6,
                "importance": 0.55,
            }
        )

    llm_extracted = _extract_user_statements_llm(query)
    extraction_source = "llm"
    if llm_extracted is None:
        llm_extracted = _extract_user_statements_regex(query)
        extraction_source = "regex_fallback"

    normalized_extracted = _normalize_candidates(llm_extracted, dropped_reason_counts)
    memory_candidates.extend(normalized_extracted)

    deduped = _dedupe(memory_candidates)
    dedupe_dropped = max(0, len(memory_candidates) - len(deduped))
    if dedupe_dropped:
        dropped_reason_counts["dedupe"] = dropped_reason_counts.get("dedupe", 0) + dedupe_dropped

    if deduped:
        print("[MEMORY CANDIDATES DETECTED]")
        for candidate in deduped:
            print(candidate)

    high_signal_count = sum(1 for item in deduped if str(item.get("type") or "").lower() in HIGH_SIGNAL_MEMORY_TYPES)
    return {
        "memory_candidates": deduped,
        "audit": {
            "source": extraction_source,
            "raw_extracted_count": len(memory_candidates),
            "deduped_count": len(deduped),
            "high_signal_count": high_signal_count,
            "dropped_reason_counts": dropped_reason_counts,
            "utterance_chars": len(query),
        },
    }


def _extract_user_statements_llm(query: str) -> list[dict] | None:
    if not query:
        return []

    prompt = (
        "You extract personal memory facts from one user utterance.\n"
        "Return STRICT JSON only with key 'candidates'.\n"
        "Output format:\n"
        "{\n"
        '  "candidates": [\n'
        '    {"type":"goal|preference|interest|dislike|behavioral|emotional","value":"...",'
        ' "confidence":0.0-1.0,"importance":0.0-1.0}\n'
        "  ]\n"
        "}\n"
        "Rules:\n"
        "- Capture only stable personal facts/preferences/goals from the current utterance.\n"
        "- Split compound lists into atomic values.\n"
        "- Keep values concise; no commentary.\n"
        "- If no memory fact is present, return {\"candidates\":[]}.\n"
        f"Utterance: {query}"
    )
    raw = generate_response(prompt, operation="turn_control")
    payload = _extract_json_payload(raw)
    if not isinstance(payload, dict):
        return None
    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        return None
    return [item for item in candidates if isinstance(item, dict)]


def _extract_user_statements_regex(query: str) -> list[dict]:
    lowered = query.lower()
    extracted: list[dict] = []

    hobby_match = re.search(r"\bmy hobbies include\s+(.+)", lowered)
    if hobby_match:
        value = _clean_phrase(hobby_match.group(1), max_len=180, max_words=60)
        if value:
            for item in _split_atomic_values(value):
                extracted.append(
                    {
                        "type": "interest",
                        "value": item,
                        "confidence": 0.76,
                        "importance": 0.58,
                    }
                )

    # Likes
    like_patterns = [
        r"\bi like\s+(.+)",
        r"\bi love\s+(.+)",
        r"\bi enjoy\s+(.+)",
    ]
    for pattern in like_patterns:
        match = re.search(pattern, lowered)
        if match:
            value = _clean_phrase(match.group(1), max_len=180, max_words=60)
            if value:
                for item in _split_atomic_values(value):
                    extracted.append(
                        {
                            "type": "interest",
                            "value": item,
                            "confidence": 0.75,
                            "importance": 0.55,
                        }
                    )
            break

    # Dislikes
    dislike_patterns = [
        r"\bi (?:do not|don't) like\s+(.+)",
        r"\bi hate\s+(.+)",
        r"\bi dislike\s+(.+)",
    ]
    for pattern in dislike_patterns:
        match = re.search(pattern, lowered)
        if match:
            value = _clean_phrase(match.group(1), max_len=180, max_words=60)
            if value:
                for item in _split_atomic_values(value):
                    extracted.append(
                        {
                            "type": "dislike",
                            "value": item,
                            "confidence": 0.75,
                            "importance": 0.55,
                        }
                    )
            break

    # Explicit preference
    prefer_match = re.search(r"\bi prefer\s+(.+)", lowered)
    if prefer_match:
        value = _clean_phrase(prefer_match.group(1))
        if value:
            extracted.append(
                {
                    "type": "preference",
                    "value": value,
                    "confidence": 0.78,
                    "importance": 0.58,
                }
            )

    # Goal intent
    goal_patterns = [
        r"\bmy goal is to\s+(.+)",
        r"\bi want to\s+(.+)",
        r"\bi am planning to\s+(.+)",
    ]
    for pattern in goal_patterns:
        match = re.search(pattern, lowered)
        if match:
            value = _clean_phrase(match.group(1))
            if value:
                extracted.append(
                    {
                        "type": "goal",
                        "value": value,
                        "confidence": 0.72,
                        "importance": 0.68,
                    }
                )
            break

    # Spending categories (useful for cards/offers/product fit)
    category_match = re.search(r"\b(?:i spend most on|i spend on)\s+(.+)", lowered)
    if category_match:
        value = _clean_phrase(category_match.group(1))
        if value:
            extracted.append(
                {
                    "type": "behavioral",
                    "value": value,
                    "confidence": 0.7,
                    "importance": 0.62,
                }
            )

    return extracted


def _clean_phrase(text: str, max_len: int = 80, max_words: int = 12) -> str | None:
    if not text:
        return None
    cleaned = re.sub(r"[\n\r\t]+", " ", text).strip(" .,!?:;")
    # Keep only the first clause to avoid noisy multi-intent captures.
    cleaned = re.split(r"\b(?:and i|but i|also i|i also|whereas)\b", cleaned, maxsplit=1, flags=re.IGNORECASE)[0]
    cleaned = cleaned.strip(" .,!?:;")
    if not cleaned:
        return None
    # Keep short factual chunks only, truncating at a word boundary.
    if len(cleaned) > max_len:
        raw = cleaned[:max_len].strip()
        last_space = raw.rfind(" ")
        cleaned = raw[:last_space].rstrip(" .,;:") if last_space > max_len // 2 else raw
    # Avoid turning long multi-clause chat into noisy memory.
    if max_words > 0 and len(cleaned.split()) > max_words:
        return None
    return cleaned


def _extract_current_utterance(raw_query: str) -> str:
    text = str(raw_query or "").strip()
    marker = "Current user follow-up:"
    if marker in text:
        _, tail = text.rsplit(marker, 1)
        followup = tail.strip()
        if followup:
            return followup
    return text


def _extract_json_payload(raw_text: str) -> dict | None:
    if not raw_text:
        return None

    text = raw_text.strip()
    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None
    try:
        payload = json.loads(match.group(0))
        return payload if isinstance(payload, dict) else None
    except json.JSONDecodeError:
        return None


def _split_atomic_values(value: str) -> list[str]:
    cleaned = _clean_phrase(value, max_len=180, max_words=60)
    if not cleaned:
        return []

    if "," not in cleaned and " and " not in cleaned.lower() and "/" not in cleaned:
        return [cleaned]

    parts = re.split(r",|/|\band\b|&", cleaned, flags=re.IGNORECASE)
    atoms: list[str] = []
    seen: set[str] = set()
    for part in parts:
        token = _clean_phrase(part, max_len=40, max_words=8)
        if not token:
            continue
        if len(token.split()) > 5:
            continue
        words = token.lower().split()
        if words:
            tail = words[-1]
            if len(tail) <= 2 and tail not in {"tv", "ai", "vr", "ux"}:
                continue
        norm = token.lower()
        if norm in seen:
            continue
        seen.add(norm)
        atoms.append(token)

    if len(atoms) >= 2:
        return atoms[:6]
    return [cleaned]


def _normalize_candidates(candidates: list[dict], dropped_reason_counts: dict[str, int]) -> list[dict]:
    normalized: list[dict] = []
    for item in candidates:
        memory_type = str(item.get("type") or "").strip().lower()
        if memory_type in {"hobby", "hobbies"}:
            memory_type = "interest"
        if memory_type == "likes":
            memory_type = "interest"
        if memory_type not in ALLOWED_MEMORY_TYPES:
            dropped_reason_counts["unsupported_type"] = dropped_reason_counts.get("unsupported_type", 0) + 1
            continue

        raw_value = str(item.get("value") or item.get("content") or "").strip()
        value = _clean_phrase(raw_value, max_len=90)
        if not value:
            dropped_reason_counts["empty_value"] = dropped_reason_counts.get("empty_value", 0) + 1
            continue

        chunks = _split_atomic_values(value) if memory_type in {"interest", "dislike"} else [value]
        for chunk in chunks:
            content = _to_memory_content(memory_type, chunk)
            if not content:
                dropped_reason_counts["empty_content"] = dropped_reason_counts.get("empty_content", 0) + 1
                continue
            normalized.append(
                {
                    "type": memory_type,
                    "content": content,
                    "confidence": _clamp(item.get("confidence"), default=0.7),
                    "importance": _clamp(item.get("importance"), default=0.6),
                }
            )
    return normalized


def _to_memory_content(memory_type: str, value: str) -> str | None:
    cleaned = _clean_phrase(value, max_len=90)
    if not cleaned:
        return None
    if memory_type == "goal":
        return f"User goal: {cleaned}."
    if memory_type == "preference":
        return f"User prefers {cleaned}."
    if memory_type == "interest":
        return f"User interest: {cleaned}."
    if memory_type == "dislike":
        return f"User dislikes {cleaned}."
    if memory_type == "behavioral":
        return f"User major spending categories: {cleaned}."
    if memory_type == "emotional":
        return f"User emotional cue: {cleaned}."
    return f"User {cleaned}."


def _clamp(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return max(0.0, min(1.0, parsed))


def _dedupe(candidates: list[dict]) -> list[dict]:
    seen = set()
    unique = []
    for candidate in candidates:
        key = (
            candidate.get("type"),
            (candidate.get("content") or "").strip().lower(),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique
