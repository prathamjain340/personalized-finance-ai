# app/domains/finance/signals.py

from typing import Dict


def _to_float(value, default: float = 0.0) -> float:
    if value in (None, "", "null"):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_non_negative_int(value, default: int = 0) -> int:
    try:
        parsed = int(float(value))
    except (TypeError, ValueError):
        return default
    return max(0, parsed)


def _to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1"}:
            return True
        if lowered in {"false", "no", "0"}:
            return False
    if isinstance(value, (int, float)):
        return value != 0
    return False


def _normalize_investments(value) -> list[str]:
    if value in (None, "", "null"):
        return []
    if isinstance(value, str):
        return [value.strip().lower()] if value.strip() else []
    if isinstance(value, (list, tuple, set)):
        normalized = []
        for item in value:
            text = str(item).strip().lower()
            if text:
                normalized.append(text)
        return normalized
    return []


def compute_financial_signals(profile: Dict) -> Dict:
    """
    Derives financial signals from raw profile data.
    These signals are deterministic, explainable, and finance-specific.
    """

    income = _to_float(profile.get("monthly_income"), default=0.0)
    expenses = _to_float(profile.get("monthly_expenses"), default=0.0)
    savings_raw = profile.get("monthly_savings")
    if savings_raw in (None, "", "null"):
        savings = income - expenses
    else:
        savings = _to_float(savings_raw, default=income - expenses)

    signals = {}

    # --- Savings rate ---
    if income > 0:
        savings_rate = savings / income
    else:
        savings_rate = 0

    signals["savings_rate"] = round(savings_rate, 2)

    # --- Expense pressure ---
    if savings_rate < 0.1:
        signals["expense_pressure"] = "high"
    elif savings_rate < 0.25:
        signals["expense_pressure"] = "medium"
    else:
        signals["expense_pressure"] = "low"

    # --- Investment readiness ---
    if savings_rate < 0.1:
        signals["investment_readiness"] = "not_ready"
    elif savings_rate < 0.25:
        signals["investment_readiness"] = "cautious"
    else:
        signals["investment_readiness"] = "ready"

    # --- Inflation risk (simplified, conservative) ---
    existing_investments = _normalize_investments(profile.get("existing_investments"))

    conservative_baskets = {"bank", "fd", "fixed deposit", "savings account"}
    if not existing_investments or set(existing_investments).issubset(conservative_baskets):
        signals["inflation_risk"] = "high"
    else:
        signals["inflation_risk"] = "moderate"

    # --- Investment status ---
    if existing_investments:
        signals["investment_status"] = "invested"
    else:
        signals["investment_status"] = "not_invested"

    # --- Protection status ---
    has_health = _to_bool(profile.get("has_health_insurance", False))
    has_life = _to_bool(profile.get("has_life_insurance", False))
    dependents = _to_non_negative_int(profile.get("children_count", 0)) + _to_non_negative_int(
        profile.get("parents_dependent", 0)
    )

    if has_health and (has_life or dependents == 0):
        signals["protection_status"] = "adequate"
    elif has_health or has_life:
        signals["protection_status"] = "partial"
    else:
        signals["protection_status"] = "missing"

    # --- Cash flow health (human-friendly) ---
    if savings_rate < 0.1:
        signals["cash_flow_health"] = "fragile"
    elif savings_rate < 0.25:
        signals["cash_flow_health"] = "stable"
    else:
        signals["cash_flow_health"] = "strong"

    return signals
