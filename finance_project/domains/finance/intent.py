# Finance turn controller
# app/domains/finance/intent.py

from dataclasses import dataclass
import json
import re
from typing import Any

from finance_project.core.llm.client import generate_response


ALLOWED_INTENTS = {"decision_support", "education", "reflection", "general"}
ALLOWED_FINANCIAL_CATEGORIES = {
    "affordability",
    "investment_advice",
    "insurance_planning",
    "debt_management",
    "income_lookup",
    "expense_lookup",
    "savings_lookup",
    "general",
}
ALLOWED_PENDING_FIELDS = {"monthly_income", "monthly_expenses", "monthly_savings", "existing_investments"}
ALLOWED_QUESTION_SCOPES = {"general", "profile_specific", "hybrid"}
ALLOWED_LIVE_DATA_KINDS = {"weather", "stock", "stock_history", "news"}

PROFILE_UPDATE_FIELD_TYPES = {
    "monthly_income": "number",
    "monthly_expenses": "number",
    "monthly_savings": "number",
    "existing_investments": "text",
    "full_name": "text",
    "preferred_name": "text",
    "occupation": "text",
    "city": "text",
    "relationship_status": "text",
    "preferred_language": "text",
}
PROFILE_FIELD_ALIASES = {
    "name": "full_name",
}


@dataclass(frozen=True)
class QueryRouting:
    finance_query: bool
    small_talk: bool
    intent: str
    financial_category: str


@dataclass(frozen=True)
class TurnControl:
    routing: QueryRouting
    profile_updates: dict[str, Any]
    active_goal: str | None
    pending_field: str | None
    question_scope: str
    self_knowledge_request: bool
    live_data_kind: str | None
    live_data_slots: dict[str, Any]


DEFAULT_ROUTING = QueryRouting(
    finance_query=True,
    small_talk=False,
    intent="general",
    financial_category="general",
)

DEFAULT_TURN_CONTROL = TurnControl(
    routing=DEFAULT_ROUTING,
    profile_updates={},
    active_goal=None,
    pending_field=None,
    question_scope="general",
    self_knowledge_request=False,
    live_data_kind=None,
    live_data_slots={},
)


def _to_bool(value, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        norm = value.strip().lower()
        if norm in {"true", "yes", "1"}:
            return True
        if norm in {"false", "no", "0"}:
            return False
    if isinstance(value, (int, float)):
        return value != 0
    return default


def _to_positive_int(value) -> int | None:
    try:
        amount = int(float(value))
    except (TypeError, ValueError):
        return None
    if amount <= 0:
        return None
    return amount


def _to_text_value(value, max_len: int = 80) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"null", "none", "n/a"}:
        return None
    text = re.sub(r"\s+", " ", text)
    if len(text) > max_len:
        text = text[:max_len].strip()
    return text or None


def _sanitize_active_goal(value) -> str | None:
    if value is None:
        return None
    goal = str(value).strip()
    if not goal or goal.lower() in {"none", "null"}:
        return None
    if len(goal) > 80:
        goal = goal[:80].strip()
    return goal


def _sanitize_pending_field(value) -> str | None:
    if value is None:
        return None
    field = str(value).strip().lower()
    if field in {"none", "null", ""}:
        return None
    if field not in ALLOWED_PENDING_FIELDS:
        return None
    return field


def _sanitize_question_scope(value) -> str:
    scope = str(value or "").strip().lower()
    if scope not in ALLOWED_QUESTION_SCOPES:
        return "general"
    return scope


def _infer_scope_heuristic(query: str) -> str:
    return "general"


def _sanitize_live_data_kind(value) -> str | None:
    kind = str(value or "").strip().lower()
    if kind in {"", "none", "null"}:
        return None
    if kind not in ALLOWED_LIVE_DATA_KINDS:
        return None
    return kind


def _to_window_days(value) -> int | None:
    if value is None or value == "":
        return None
    try:
        days = int(float(value))
    except (TypeError, ValueError):
        return None
    if days < 7 or days > 3650:
        return None
    return days


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


def _build_turn_control_prompt(
    raw_query: str,
    last_assistant_text: str | None = None,
    active_goal: str | None = None,
    pending_field: str | None = None,
) -> str:
    return (
        "You are a turn controller for a personal finance assistant.\n"
        "Classify ONLY the current user message and extract any useful profile updates.\n\n"
        f"Previous assistant message: {last_assistant_text or ''}\n"
        f"Current active goal: {active_goal or ''}\n"
        f"Current pending field: {pending_field or ''}\n"
        f"Current user message: {raw_query}\n\n"
        "Return STRICT JSON only with keys:\n"
        "- finance_query: boolean\n"
        "- small_talk: boolean\n"
        "- intent: one of [decision_support, education, reflection, general]\n"
        "- financial_category: one of [affordability, investment_advice, insurance_planning, debt_management, income_lookup, expense_lookup, savings_lookup, general]\n"
        "- question_scope: one of [general, profile_specific, hybrid]\n"
        "- self_knowledge_request: boolean\n"
        "- live_data_kind: one of [weather, stock, stock_history, news, none]\n"
        "- live_ticker: string or null\n"
        "- live_company: string or null\n"
        "- live_location: string or null\n"
        "- live_topic: string or null\n"
        "- live_window_days: integer days or null\n"
        "- profile_updates: object with zero or more keys [monthly_income, monthly_expenses, monthly_savings, existing_investments, full_name, preferred_name, occupation, city, relationship_status, preferred_language]\n"
        "  - For monthly_* fields, values must be integer INR monthly amounts\n"
        "  - If user explicitly says they currently have no investments, set existing_investments to 'none_declared'\n"
        "  - For personal fields, values should be short strings\n"
        "- active_goal: short string or null\n"
        "- pending_field: one of [monthly_income, monthly_expenses, monthly_savings, existing_investments, none]\n\n"
        "Guidance:\n"
        "- small_talk=true ONLY for pure social niceties without an informational request.\n"
        "- Keep the assistant finance-anchored but conversationally soft.\n"
        "- If user shares personal context (name, occupation, location, preferences), keep finance_query=true so context can be retained.\n"
        "- If user asks for deeply non-finance tasks (coding, medical diagnosis, legal drafting, explicit/adult content), set finance_query=false and small_talk=false.\n"
        "- Short acknowledgements (yes/yeah/ok/sure) after a finance follow-up should remain finance_query=true.\n"
        "- Short fragment replies (product names, categories, preferences) to a prior finance question should remain finance_query=true.\n"
        "- For profile lookup requests (my income/expenses/savings), use the respective lookup category.\n"
        "- question_scope=profile_specific when user asks about their own situation or affordability.\n"
        "- question_scope=general for conceptual questions not requiring user-specific data.\n"
        "- question_scope=hybrid when user asks concept + personal implication in same turn.\n"
        "- self_knowledge_request=true only when user asks what you know/remember about them (profile, likes, dislikes, preferences, memories).\n"
        "- For personal-memory lookup like 'what is my name', keep finance_query=true with intent=general.\n"
        "- Definitions inside finance (e.g., what is savings account) use financial_category=general.\n"
        "- Extract multiple profile fields if user provides them in one message.\n"
        "- If any update is unclear, leave that field out.\n"
        "- Set live_data_kind only when user explicitly asks for current or historical data updates.\n"
        "- For stock-history performance questions use stock_history and fill live_window_days when available.\n"
        "- Use live slots only when explicitly present in the user turn.\n"
        "- If the user is answering a profile question with financial amounts, live_data_kind must be none.\n"
        "- Keep active_goal stable unless user clearly switches topics.\n"
        "- If user still needs to answer a missing detail, set pending_field accordingly; otherwise pending_field=none.\n"
        "- If uncertain, choose finance_query=true, small_talk=false, intent=general, financial_category=general.\n"
        "Output JSON only. No markdown."
    )


def infer_turn_control(
    raw_query: str,
    last_assistant_text: str | None = None,
    active_goal: str | None = None,
    pending_field: str | None = None,
) -> TurnControl:
    query = (raw_query or "").strip()
    if not query:
        return TurnControl(
            routing=DEFAULT_ROUTING,
            profile_updates={},
            active_goal=active_goal,
            pending_field=pending_field,
            question_scope="general",
            self_knowledge_request=False,
            live_data_kind=None,
            live_data_slots={},
        )

    prompt = _build_turn_control_prompt(
        raw_query=query,
        last_assistant_text=last_assistant_text,
        active_goal=active_goal,
        pending_field=pending_field,
    )
    raw_response = generate_response(prompt, operation="turn_control")
    payload = _extract_json_payload(raw_response)

    if not payload:
        return TurnControl(
            routing=DEFAULT_ROUTING,
            profile_updates={},
            active_goal=active_goal,
            pending_field=pending_field,
            question_scope="general",
            self_knowledge_request=False,
            live_data_kind=None,
            live_data_slots={},
        )

    finance_query = _to_bool(payload.get("finance_query"), default=True)
    small_talk = _to_bool(payload.get("small_talk"), default=False)

    intent = str(payload.get("intent", "general")).strip().lower()
    if intent not in ALLOWED_INTENTS:
        intent = "general"

    financial_category = str(payload.get("financial_category", "general")).strip().lower()
    if financial_category not in ALLOWED_FINANCIAL_CATEGORIES:
        financial_category = "general"

    updates_raw = payload.get("profile_updates")
    profile_updates: dict[str, Any] = {}
    if isinstance(updates_raw, dict):
        for field, value in updates_raw.items():
            field_name = str(field).strip().lower()
            field_name = PROFILE_FIELD_ALIASES.get(field_name, field_name)
            expected_type = PROFILE_UPDATE_FIELD_TYPES.get(field_name)
            if not expected_type:
                continue
            if expected_type == "number":
                amount = _to_positive_int(value)
                if amount is None:
                    continue
                profile_updates[field_name] = amount
                continue
            if expected_type == "text":
                text = _to_text_value(value)
                if text is None:
                    continue
                profile_updates[field_name] = text

    next_active_goal = _sanitize_active_goal(payload.get("active_goal"))
    next_pending_field = _sanitize_pending_field(payload.get("pending_field"))
    question_scope = _sanitize_question_scope(payload.get("question_scope"))
    self_knowledge_request = _to_bool(payload.get("self_knowledge_request"), default=False)
    live_data_kind = _sanitize_live_data_kind(payload.get("live_data_kind"))
    live_data_slots = {
        "ticker": _to_text_value(payload.get("live_ticker")),
        "company": _to_text_value(payload.get("live_company")),
        "location": _to_text_value(payload.get("live_location")),
        "topic": _to_text_value(payload.get("live_topic")),
        "window_days": _to_window_days(payload.get("live_window_days")),
    }

    if any(field in profile_updates for field in ("monthly_income", "monthly_expenses", "monthly_savings", "existing_investments")):
        live_data_kind = None
        live_data_slots = {}

    if small_talk:
        finance_query = False

    return TurnControl(
        routing=QueryRouting(
            finance_query=finance_query,
            small_talk=small_talk,
            intent=intent,
            financial_category=financial_category,
        ),
        profile_updates=profile_updates,
        active_goal=next_active_goal,
        pending_field=next_pending_field,
        question_scope=question_scope,
        self_knowledge_request=self_knowledge_request,
        live_data_kind=live_data_kind,
        live_data_slots=live_data_slots,
    )


def infer_query_metadata(raw_query: str, last_assistant_text: str | None = None) -> QueryRouting:
    return infer_turn_control(
        raw_query=raw_query,
        last_assistant_text=last_assistant_text,
    ).routing


def classify_intent(raw_query: str, last_assistant_text: str | None = None) -> str:
    return infer_query_metadata(raw_query, last_assistant_text=last_assistant_text).intent


def map_intent_to_financial_category(raw_query: str, last_assistant_text: str | None = None) -> str:
    return infer_query_metadata(raw_query, last_assistant_text=last_assistant_text).financial_category


def is_small_talk(raw_query: str, last_assistant_text: str | None = None) -> bool:
    return infer_query_metadata(raw_query, last_assistant_text=last_assistant_text).small_talk


def is_finance_query(raw_query: str, last_assistant_text: str | None = None) -> bool:
    return infer_query_metadata(raw_query, last_assistant_text=last_assistant_text).finance_query
