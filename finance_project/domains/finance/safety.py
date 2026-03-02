# Finance safety rules
# app/domains/finance/safety.py

from finance_project.domains.finance.prompt.modes import ResponseMode


def select_response_mode(
    intent: str,
    profile: dict,
    memories: list
) -> ResponseMode:
    """
    Selects the appropriate response mode based on
    intent, profile completeness, and memory signals.
    """

    # Unsafe or asset-picking intent
    if intent == "decision_support":
        if not profile:
            return ResponseMode.CLARIFYING_FIRST

        if memories:
            return ResponseMode.REFLECTIVE_PREVENTIVE

        return ResponseMode.ASSUMPTIVE_WITH_CLARIFIER

    if intent == "education":
        return ResponseMode.DIRECT_FRAMEWORK

    if intent == "reflection":
        return ResponseMode.REFLECTIVE_PREVENTIVE

    return ResponseMode.DIRECT_FRAMEWORK