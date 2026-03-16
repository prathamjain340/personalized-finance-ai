# app/core/chat/session.py

from typing import Optional
from finance_project.core.chat.stage import ConversationStage
from datetime import datetime


class ChatSession:
    """
    Maintains short-term conversational context for a user.
    This is NOT long-term memory.
    """

    def __init__(self, user_id: str, domain: str):
        self.user_id = user_id
        self.domain = domain
        self.session_memory_usage: dict[str, int] = {}
        self.last_user_message: Optional[str] = None
        self.last_assistant_response: Optional[str] = None
        self.recent_turns: list[tuple[str, str]] = []
        self.active_goal: Optional[str] = None
        self.pending_field: Optional[str] = None
        self.profile_collection_goal: Optional[str] = None
        self.profile_collection_queue: list[str] = []
        self.stage: ConversationStage = ConversationStage.INITIAL
        self.has_started = False

    def build_query(self, raw_query: str, max_turns: int = 3) -> str:
        """
        Builds a contextual query using the recent turns.
        Keeps context minimal to avoid prompt bloat.
        """

        if not self.recent_turns:
            return raw_query

        window = max(1, max_turns)
        turns = self.recent_turns[-window:]
        lines = [
            "Session state:",
            f"Active goal: {self.active_goal or 'none'}",
            f"Pending field: {self.pending_field or 'none'}",
            f"Profile collection goal: {self.profile_collection_goal or 'none'}",
            f"Profile collection queue: {', '.join(self.profile_collection_queue) if self.profile_collection_queue else 'none'}",
            "",
            "Recent conversation turns:",
        ]
        for idx, (user_text, assistant_text) in enumerate(turns, start=1):
            clipped_user = self._clip_turn_text(user_text, max_chars=180)
            clipped_assistant = self._clip_turn_text(assistant_text, max_chars=240)
            lines.append(f"Turn {idx} user: {clipped_user}")
            lines.append(f"Turn {idx} assistant: {clipped_assistant}")
        lines.append("")
        lines.append("Current user follow-up:")
        lines.append(raw_query)
        return "\n".join(lines)

    @staticmethod
    def _clip_turn_text(text: str, max_chars: int) -> str:
        normalized = " ".join(str(text or "").split())
        if len(normalized) <= max_chars:
            return normalized
        return normalized[:max_chars].rstrip() + "..."

    def update(
        self,
        user_message: str,
        assistant_response: str,
        memories_used=None
    ):
        # ---- always update last turn ----
        self.last_user_message = user_message
        self.last_assistant_response = assistant_response
        self.recent_turns.append((user_message, assistant_response))
        if len(self.recent_turns) > 4:
            self.recent_turns = self.recent_turns[-4:]

        # ---- stage progression (enum-aligned) ----
        if self.stage == ConversationStage.INITIAL:
            self.stage = ConversationStage.EXPLANATION

        elif self.stage == ConversationStage.EXPLANATION:
            self.stage = ConversationStage.COMPARISON

        elif self.stage == ConversationStage.COMPARISON:
            self.stage = ConversationStage.DECISION_SUPPORT

        elif self.stage == ConversationStage.DECISION_SUPPORT:
            self.stage = ConversationStage.EXECUTION

        # NOTE: do NOT auto-advance beyond EXECUTION yet

        # ---- memory exposure tracking ----
        if not memories_used:
            return

        for mem in memories_used:
            # session-level exposure
            self.session_memory_usage[mem.id] = (
                self.session_memory_usage.get(mem.id, 0) + 1
            )

            # memory-level exposure
            mem.exposure_count += 1
            mem.last_used_at = datetime.utcnow()

            # decay after repeated exposure
            if mem.exposure_count >= 2:
                mem.decay()


    def update_stage(self, followup_intent: str):
        """
        Advances conversation stage based on user behavior.
        """

        if followup_intent in ("simplification",):
            self.stage = ConversationStage.EXPLANATION

        elif followup_intent in ("comparison",):
            self.stage = ConversationStage.COMPARISON

        elif followup_intent in ("scenario",):
            self.stage = ConversationStage.DECISION_SUPPORT

        elif followup_intent in ("confirmation", "yes"):
            self.stage = ConversationStage.EXECUTION

        elif followup_intent in ("gratitude", "exit"):
            self.stage = ConversationStage.WRAP_UP

