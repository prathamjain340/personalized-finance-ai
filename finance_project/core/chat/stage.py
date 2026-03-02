from enum import Enum


class ConversationStage(str, Enum):
    INITIAL = "initial_guidance"
    EXPLANATION = "explanation"
    COMPARISON = "comparison"
    DECISION_SUPPORT = "decision_support"
    EXECUTION = "execution_help"
    WRAP_UP = "wrap_up"