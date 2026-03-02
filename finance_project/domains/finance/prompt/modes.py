# Response modes (1–5)
# app/domains/finance/prompt/modes.py

from enum import Enum


class ResponseMode(str, Enum):
    DIRECT_FRAMEWORK = "direct_framework"
    ASSUMPTIVE_WITH_CLARIFIER = "assumptive_with_clarifier"
    CLARIFYING_FIRST = "clarifying_first"
    REFLECTIVE_PREVENTIVE = "reflective_preventive"
    EDUCATIONAL_REDIRECT = "educational_redirect"
