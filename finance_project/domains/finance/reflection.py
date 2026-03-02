import re
from typing import Dict, List


def reflect_on_response(
    user_id: str,
    raw_query: str,
    response: str,
    intent: str,
) -> List[Dict]:
    """
    Post-response reflection hook.

    Detects lightweight memory candidates from user text.
    Storage remains asynchronous and outside request critical path.
    """
    query = " ".join((raw_query or "").strip().split())
    lowered = query.lower()
    memory_candidates: list[dict] = []

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

    # --- Structured personal preference extraction (step 1) ---
    extracted = _extract_user_statements(query)
    memory_candidates.extend(extracted)

    if memory_candidates:
        print("[MEMORY CANDIDATES DETECTED]")
        for candidate in memory_candidates:
            print(candidate)

    return _dedupe(memory_candidates)


def _extract_user_statements(query: str) -> list[dict]:
    lowered = query.lower()
    extracted: list[dict] = []

    # Likes
    like_patterns = [
        r"\bi like\s+(.+)",
        r"\bi love\s+(.+)",
        r"\bi enjoy\s+(.+)",
    ]
    for pattern in like_patterns:
        match = re.search(pattern, lowered)
        if match:
            value = _clean_phrase(match.group(1))
            if value:
                extracted.append(
                    {
                        "type": "preference",
                        "content": f"User likes {value}.",
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
            value = _clean_phrase(match.group(1))
            if value:
                extracted.append(
                    {
                        "type": "preference",
                        "content": f"User dislikes {value}.",
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
                    "content": f"User prefers {value}.",
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
                        "content": f"User goal: {value}.",
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
                    "content": f"User major spending categories: {value}.",
                    "confidence": 0.7,
                    "importance": 0.62,
                }
            )

    return extracted


def _clean_phrase(text: str, max_len: int = 80) -> str | None:
    if not text:
        return None
    cleaned = re.sub(r"[\n\r\t]+", " ", text).strip(" .,!?:;")
    # Keep only the first clause to avoid noisy multi-intent captures.
    cleaned = re.split(r"\b(?:and i|but i|also i|whereas)\b", cleaned, maxsplit=1, flags=re.IGNORECASE)[0]
    cleaned = cleaned.strip(" .,!?:;")
    if not cleaned:
        return None
    # Keep short factual chunks only.
    cleaned = cleaned[:max_len].strip()
    # Avoid turning long multi-clause chat into noisy memory.
    if len(cleaned.split()) > 12:
        return None
    return cleaned


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
