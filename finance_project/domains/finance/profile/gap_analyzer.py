# finance_project/domains/finance/profile/gap_analyzer.py

from enum import Enum
from typing import Dict, List, Tuple


class ProfileDataState(str, Enum):
    ENOUGH_DATA = "enough_data"
    PARTIAL_DATA = "partial_data"
    INSUFFICIENT_DATA = "insufficient_data"


# Which profile fields matter for which question types
INTENT_REQUIREMENTS = {
    "affordability": ["monthly_income", "monthly_expenses", "monthly_savings"],
    "investment_advice": ["monthly_income", "monthly_savings"],
    "insurance_planning": ["monthly_income", "dependents"],
    "debt_management": ["monthly_income", "monthly_expenses"],
    "income_lookup": ["monthly_income"],
    "expense_lookup": ["monthly_expenses"],
    "savings_lookup": ["monthly_savings"],
}


def _normalize_profile(profile: Dict) -> Dict:
    normalized = dict(profile)

    # Support existing schema where dependents are split into two fields.
    if normalized.get("dependents") is None:
        children = normalized.get("children_count", 0) or 0
        parents = normalized.get("parents_dependent", 0) or 0
        try:
            normalized["dependents"] = int(children) + int(parents)
        except (TypeError, ValueError):
            normalized["dependents"] = None

    return normalized


def _is_positive_number(value) -> bool:
    try:
        return float(value) > 0
    except (TypeError, ValueError):
        return False


def _is_non_negative_number(value) -> bool:
    try:
        return float(value) >= 0
    except (TypeError, ValueError):
        return False


def _is_present(value) -> bool:
    return value not in (None, "", "null")


def analyze_profile_gaps(
    intent: str,
    profile: Dict | None,
) -> Tuple[ProfileDataState, List[str]]:
    """
    Checks whether we have enough user data
    to safely answer the financial question.
    """

    required_fields = INTENT_REQUIREMENTS.get(intent, ["monthly_income", "monthly_expenses"])

    if not profile:
        return ProfileDataState.INSUFFICIENT_DATA, required_fields.copy()

    normalized_profile = _normalize_profile(profile)

    validators = {
        "monthly_income": _is_positive_number,
        "monthly_expenses": _is_positive_number,
        "monthly_savings": _is_positive_number,
        "dependents": _is_non_negative_number,
    }

    missing = []
    present = 0

    for field in required_fields:
        value = normalized_profile.get(field)
        validator = validators.get(field, _is_present)

        if validator(value):
            present += 1
        else:
            missing.append(field)

    if present == len(required_fields):
        return ProfileDataState.ENOUGH_DATA, []

    if present == 0:
        return ProfileDataState.INSUFFICIENT_DATA, missing

    return ProfileDataState.PARTIAL_DATA, missing
