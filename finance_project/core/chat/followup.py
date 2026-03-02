# app/core/chat/followup.py

def detect_followup_intent(query: str) -> str:
    q = query.lower().strip()

    if any(word in q for word in ["explain", "simple", "easier", "understand"]):
        return "simplification"

    if any(word in q for word in ["which is better", "compare", "difference"]):
        return "comparison"

    if q.startswith("what if") or "what if" in q:
        return "scenario"

    if any(word in q for word in ["scared", "afraid", "anxious", "fear"]):
        return "emotional_reassurance"

    if any(word in q for word in ["so", "then", "what should i do", "next"]):
        return "decision_confirmation"

    return "general_followup"