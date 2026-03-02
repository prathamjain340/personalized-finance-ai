# Finance Engine (orchestrator)
# This file is the brainstem of your system
# app/domains/finance/engine.py

from typing import Optional
import re

from finance_project.domains.finance.intent import (
    QueryRouting,
    infer_query_metadata,
)
from finance_project.domains.finance.prompt.assembler import (
    assemble_clarification_prompt,
    assemble_out_of_domain_prompt,
    assemble_prompt,
)
from finance_project.domains.finance.reflection import reflect_on_response

from finance_project.core.summary.repository import get_domain_summary
from finance_project.core.memory.retriever import retrieve_memories
from finance_project.core.llm.client import generate_response
from finance_project.core.logging.logger import log_conversation
from finance_project.domains.finance.safety import select_response_mode
from finance_project.core.postprocess.dispatcher import enqueue_memory_candidates
from finance_project.domains.finance.signals import compute_financial_signals
from finance_project.domains.finance.profile.gap_analyzer import analyze_profile_gaps, ProfileDataState
from finance_project.core.chat.stage import ConversationStage
from finance_project.services.live_data_service import maybe_fetch_live_data


class FinanceEngine:
    """
    FinanceEngine orchestrates the full lifecycle of a finance-domain request.
    It does NOT:
    - contain business logic
    - infer memory
    - talk directly to databases
    - enforce HTTP concerns
    """

    def handle_request(
        self,
        user_id: str,
        raw_query: str,
        current_query: Optional[str] = None,
        last_assistant_response: Optional[str] = None,
        routing: Optional[QueryRouting] = None,
        self_knowledge_request: bool = False,
        active_goal: Optional[str] = None,
        pending_field: Optional[str] = None,
        question_scope: Optional[str] = None,
        profile: Optional[dict] = None,
        session_memory_usage: Optional[dict] = None,
        conversation_stage: Optional[ConversationStage] = None,
        response_channel: str = "text",
    ):
        # Backward-compatible defaults for older callers (tests/UI).
        if profile is None:
            profile = {}

        if session_memory_usage is None:
            session_memory_usage = {}

        if conversation_stage is None:
            conversation_stage = ConversationStage.INITIAL
        if not question_scope:
            question_scope = "general"

        query_for_routing = current_query or raw_query

        live_data = maybe_fetch_live_data(query=query_for_routing, profile=profile)
        if live_data:
            response = self._handle_live_data_response(
                query=query_for_routing,
                live_data=live_data,
                response_channel=response_channel,
                profile=profile,
            )
            memory_candidates = reflect_on_response(
                user_id=user_id,
                raw_query=query_for_routing,
                response=response,
                intent="live_data",
            )
            enqueue_memory_candidates(
                user_id=user_id,
                domain="finance",
                memory_candidates=memory_candidates,
            )
            log_conversation(
                user_id=user_id,
                domain="finance",
                request=raw_query,
                response=response,
                metadata={
                    "intent": "live_data",
                    "live_kind": live_data.get("kind"),
                    "live_status": live_data.get("status"),
                },
            )
            return response, []

        if routing is None:
            routing = infer_query_metadata(
                query_for_routing,
                last_assistant_text=last_assistant_response,
            )

        if self_knowledge_request:
            memories = retrieve_memories(
                user_id=user_id,
                domain="finance",
                query=query_for_routing,
                session_memory_usage=session_memory_usage,
                limit=8,
            )
            response = self._build_self_knowledge_response(
                profile=profile,
                memories=memories,
                response_channel=response_channel,
            )
            log_conversation(
                user_id=user_id,
                domain="finance",
                request=raw_query,
                response=response,
                metadata={"intent": "self_knowledge"},
            )
            return response, memories

        if not routing.finance_query:
            memories = retrieve_memories(
                user_id=user_id,
                domain="finance",
                query=query_for_routing,
                session_memory_usage=session_memory_usage,
                limit=3,
            )
            prompt = assemble_out_of_domain_prompt(
                raw_query=query_for_routing,
                small_talk=routing.small_talk,
                response_channel=response_channel,
                profile=profile,
                memories=memories,
            )
            response = generate_response(
                prompt,
                operation="out_of_domain_response",
                max_tokens_override=self._max_tokens_for_generation(
                    operation="out_of_domain_response",
                    response_channel=response_channel,
                    query=query_for_routing,
                ),
            )
            response = self._trim_voice_response(response, response_channel=response_channel)
            memory_candidates = reflect_on_response(
                user_id=user_id,
                raw_query=query_for_routing,
                response=response,
                intent="out_of_domain",
            )
            enqueue_memory_candidates(
                user_id=user_id,
                domain="finance",
                memory_candidates=memory_candidates,
            )
            log_conversation(
                user_id=user_id,
                domain="finance",
                request=raw_query,
                response=response,
                metadata={"intent": "out_of_domain", "small_talk": routing.small_talk},
            )
            return response, memories

        # 1. Classify user intent (finance-only)
        intent = routing.intent

        # Map to financial decision type
        financial_category = routing.financial_category

        # Check profile sufficiency
        profile_state, missing_fields = analyze_profile_gaps(financial_category, profile)

        signals = self._safe_compute_signals(profile)

        # 3. Fetch finance domain summary (if exists)
        domain_summary = get_domain_summary(user_id, domain="finance")

        # 4. Retrieve relevant long-term memories (read-only)
        memories = retrieve_memories(
            user_id=user_id,
            domain="finance",
            query=query_for_routing,
            session_memory_usage=session_memory_usage,
            limit=3,
        )

        response_mode = select_response_mode(
            intent=intent,
            profile=profile,
            memories=memories,
        )

        if self._needs_clarification_first(
            intent=intent,
            financial_category=financial_category,
            profile_state=profile_state,
            missing_fields=missing_fields,
        ):
            prompt = assemble_clarification_prompt(
                raw_query=query_for_routing,
                missing_fields=missing_fields,
                profile=profile,
                response_channel=response_channel,
            )
            response = generate_response(
                prompt,
                operation="clarification_response",
                max_tokens_override=self._max_tokens_for_generation(
                    operation="clarification_response",
                    response_channel=response_channel,
                    query=query_for_routing,
                ),
            )
            response = self._trim_voice_response(response, response_channel=response_channel)
            log_conversation(
                user_id=user_id,
                domain="finance",
                request=raw_query,
                response=response,
                metadata={"intent": intent, "profile_state": profile_state.value, "missing_fields": missing_fields},
            )
            return response, []

        # 5. Assemble prompt using finance prompt framework
        prompt = assemble_prompt(
            raw_query=raw_query,
            intent=intent,
            profile=profile,
            signals=signals,
            domain_summary=domain_summary,
            memories=memories,
            response_mode=response_mode.value,
            question_scope=question_scope,
            conversation_stage=conversation_stage,
            missing_fields=missing_fields,
            profile_state=profile_state.value,
            previous_assistant_response=last_assistant_response,
            active_goal=active_goal,
            pending_field=pending_field,
            response_channel=response_channel,
        )

        # 6. Call LLM (stateless reasoning)
        response = generate_response(
            prompt,
            operation="finance_response",
            max_tokens_override=self._max_tokens_for_generation(
                operation="finance_response",
                response_channel=response_channel,
                query=query_for_routing,
            ),
        )
        response = self._trim_voice_response(response, response_channel=response_channel)

        # 7. Log conversation (immutable)
        log_conversation(
            user_id=user_id,
            domain="finance",
            request=raw_query,
            response=response,
            metadata={
                "intent": intent,
                "financial_category": financial_category,
                "question_scope": question_scope,
                "profile_state": profile_state.value,
                "active_goal": active_goal,
                "pending_field": pending_field,
            },
        )

        # 8. Post-response reflection (non-blocking, probabilistic)
        memory_candidates = reflect_on_response(
            user_id=user_id,
            raw_query=query_for_routing,
            response=response,
            intent=intent,
        )

        enqueue_memory_candidates(
            user_id=user_id,
            domain="finance",
            memory_candidates=memory_candidates,
        )

        return response, memories

    @staticmethod
    def _needs_clarification_first(
        intent: str,
        financial_category: str,
        profile_state: ProfileDataState,
        missing_fields: list[str],
    ) -> bool:
        """
        Ask follow-up first only when data is too limited for decision guidance.
        """
        if profile_state != ProfileDataState.INSUFFICIENT_DATA or not missing_fields:
            return False

        if financial_category != "general":
            return True

        return intent == "decision_support"

    @staticmethod
    def _safe_compute_signals(profile: dict) -> dict:
        if not profile:
            return {}

        # Avoid overconfident derived claims when core numbers are absent.
        income = profile.get("monthly_income")
        expenses = profile.get("monthly_expenses")
        savings = profile.get("monthly_savings")
        if income in (None, "", 0) or (expenses in (None, "") and savings in (None, "")):
            return {}

        try:
            return compute_financial_signals(profile)
        except Exception:
            return {}

    @staticmethod
    def _max_tokens_for_generation(
        operation: str,
        response_channel: str,
        query: str,
    ) -> int | None:
        is_hindi_script = FinanceEngine._contains_devanagari(query)
        channel = str(response_channel or "text").strip().lower()

        if channel == "voice":
            if operation == "finance_response":
                return 220 if is_hindi_script else 170
            if operation in {"clarification_response", "out_of_domain_response"}:
                return 140 if is_hindi_script else 110
            return None

        if operation == "finance_response":
            return 300 if is_hindi_script else 240
        if operation in {"clarification_response", "out_of_domain_response"}:
            return 200 if is_hindi_script else 170
        return None

    @staticmethod
    def _trim_voice_response(
        text: str,
        response_channel: str,
        max_words: int = 28,
        max_sentences: int = 2,
    ) -> str:
        if response_channel != "voice":
            return text

        normalized = " ".join((text or "").split())
        if not normalized:
            return "Could you please repeat that?"

        # Keep complete sentences only (supports English + Hindi sentence markers).
        sentences = FinanceEngine._split_sentences(normalized)
        if not sentences:
            return FinanceEngine._ensure_terminal(normalized)

        # Always keep the first complete sentence, then optionally add one more
        # if still compact by word count.
        selected = [sentences[0]]
        total_words = len(sentences[0].split())

        for sentence in sentences[1:max_sentences]:
            sentence_words = len(sentence.split())
            if total_words + sentence_words <= max_words:
                selected.append(sentence)
                total_words += sentence_words
            else:
                break

        return FinanceEngine._ensure_terminal(" ".join(selected).strip())

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        if not text:
            return []
        parts = [p.strip() for p in re.split(r"(?<=[.!?।])\s+", text) if p.strip()]
        if parts:
            return [FinanceEngine._ensure_terminal(part) for part in parts]
        return [FinanceEngine._ensure_terminal(text.strip())]

    @staticmethod
    def _contains_devanagari(text: str) -> bool:
        return bool(re.search(r"[\u0900-\u097F]", text or ""))

    @staticmethod
    def _ensure_terminal(text: str) -> str:
        if not text:
            return text
        if text[-1] in ".!?।":
            return text
        terminal = "।" if FinanceEngine._contains_devanagari(text) else "."
        return f"{text}{terminal}"

    def _handle_live_data_response(
        self,
        query: str,
        live_data: dict,
        response_channel: str,
        profile: dict,
    ) -> str:
        status = live_data.get("status")
        if status in {"needs_input", "error", "blocked"}:
            return self._trim_voice_response(
                str(live_data.get("message") or "I couldn't fetch that update right now."),
                response_channel=response_channel,
            )

        kind = str(live_data.get("kind") or "").lower()
        facts = [str(item).strip() for item in (live_data.get("facts") or []) if str(item).strip()]
        as_of = str(live_data.get("as_of") or "")

        if kind == "weather":
            response = self._format_weather_update(facts=facts, as_of=as_of)
        elif kind == "stock":
            response = self._format_stock_update(facts=facts, as_of=as_of)
        elif kind == "news":
            response = self._format_news_update(facts=facts, as_of=as_of)
        else:
            response = self._format_generic_live_update(facts=facts, as_of=as_of)

        return self._trim_voice_response(response, response_channel=response_channel)

    @staticmethod
    def _format_weather_update(facts: list[str], as_of: str) -> str:
        temperature = FinanceEngine._first_fact_value(facts, "Temperature:")
        feels_like = FinanceEngine._first_fact_value(facts, "Feels like:")
        humidity = FinanceEngine._first_fact_value(facts, "Humidity:")
        location = FinanceEngine._first_fact_value(facts, "Location:")

        parts = []
        if location:
            parts.append(f"Weather update for {location}")
        if temperature:
            temp_line = f"temperature is {temperature}"
            if feels_like:
                temp_line += f", feels like {feels_like}"
            parts.append(temp_line)
        if humidity:
            parts.append(f"humidity is {humidity}")

        body = ". ".join(parts) if parts else "Here is the latest weather update."
        freshness = FinanceEngine._freshness_suffix(as_of)
        return f"{body}.{freshness}"

    @staticmethod
    def _format_stock_update(facts: list[str], as_of: str) -> str:
        instrument = FinanceEngine._first_fact_value(facts, "Instrument:")
        price = FinanceEngine._first_fact_value(facts, "Last price:")
        change = FinanceEngine._first_fact_value(facts, "Change:")

        parts = []
        if instrument:
            parts.append(f"{instrument}")
        if price:
            parts.append(f"last price is {price}")
        if change:
            parts.append(f"change is {change}")

        body = ", ".join(parts) if parts else "Here is the latest stock update."
        freshness = FinanceEngine._freshness_suffix(as_of)
        caution = " This is market data, not a buy or sell recommendation."
        return f"{body}.{freshness}{caution}"

    @staticmethod
    def _format_news_update(facts: list[str], as_of: str) -> str:
        headlines = [item for item in facts if item.startswith("Headline ")]
        top = headlines[:2]
        if top:
            cleaned = [re.sub(r"^Headline\s+\d+:\s*", "", item).strip() for item in top]
            body = "Top updates: " + " | ".join(cleaned)
        else:
            body = "Here are the latest updates I found."
        freshness = FinanceEngine._freshness_suffix(as_of)
        return f"{body}.{freshness}"

    @staticmethod
    def _format_generic_live_update(facts: list[str], as_of: str) -> str:
        body = " ".join(facts[:2]).strip() if facts else "Here is the latest update."
        freshness = FinanceEngine._freshness_suffix(as_of)
        return f"{body}.{freshness}"

    @staticmethod
    def _first_fact_value(facts: list[str], prefix: str) -> str | None:
        for fact in facts:
            if fact.startswith(prefix):
                return fact[len(prefix) :].strip()
        return None

    @staticmethod
    def _freshness_suffix(as_of: str) -> str:
        if not as_of:
            return ""
        return f" Data as of {as_of}"

    @staticmethod
    def _memory_text_tokens(text: str) -> set[str]:
        return {token for token in re.findall(r"[a-z0-9]+", text.lower()) if len(token) >= 3}

    @staticmethod
    def _prepare_self_knowledge_memory_lines(memories: list, limit: int = 6) -> list[str]:
        entries: list[dict] = []
        for memory in memories[:10]:
            memory_type = str(getattr(memory, "type", "") or "").strip().lower()
            content = str(getattr(memory, "content", "")).strip()
            if not content:
                continue
            if content.lower().startswith("user "):
                content = content[5:]
            content = content[:1].upper() + content[1:] if content else content
            line = content[:120].strip()
            if not line:
                continue
            normalized = re.sub(r"\s+", " ", line.lower()).strip(" .,!?:;")
            tokens = FinanceEngine._memory_text_tokens(normalized)
            entries.append(
                {
                    "type": memory_type,
                    "line": line,
                    "norm": normalized,
                    "tokens": tokens,
                }
            )

        entries.sort(key=lambda item: len(item["norm"]))
        compact: list[dict] = []

        for entry in entries:
            duplicate = False
            for kept in compact:
                if entry["type"] != kept["type"]:
                    continue

                if not entry["norm"] or entry["norm"] == kept["norm"]:
                    duplicate = True
                    break

                # If the new line only extends an existing shorter line, skip it.
                if kept["norm"] in entry["norm"]:
                    duplicate = True
                    break

                entry_tokens = entry["tokens"]
                kept_tokens = kept["tokens"]
                if entry_tokens and kept_tokens:
                    union = entry_tokens | kept_tokens
                    if union:
                        jaccard = len(entry_tokens & kept_tokens) / len(union)
                        if jaccard >= 0.85:
                            duplicate = True
                            break

            if not duplicate:
                compact.append(entry)
            if len(compact) >= limit:
                break

        return [item["line"] for item in compact[:limit]]

    @staticmethod
    def _build_self_knowledge_response(profile: dict, memories: list, response_channel: str) -> str:
        points: list[str] = []
        name = profile.get("preferred_name") or profile.get("full_name") or profile.get("name")
        city = profile.get("city")
        income = profile.get("monthly_income")
        expenses = profile.get("monthly_expenses")
        savings = profile.get("monthly_savings")

        if name:
            points.append(f"Name: {name}")
        if city:
            points.append(f"City: {city}")
        if income not in (None, ""):
            points.append(f"Monthly income: INR {income}")
        if expenses not in (None, ""):
            points.append(f"Monthly expenses: INR {expenses}")
        if savings not in (None, ""):
            points.append(f"Monthly savings: INR {savings}")

        memory_lines = FinanceEngine._prepare_self_knowledge_memory_lines(memories, limit=6)

        if response_channel == "voice":
            if not points and not memory_lines:
                return "I only know limited details so far. You can share your preferences and I'll remember them."
            summary_bits = []
            if name:
                summary_bits.append(f"your name is {name}")
            if city:
                summary_bits.append(f"you are in {city}")
            if memory_lines:
                summary_bits.append(memory_lines[0].rstrip("."))
            return "I know that " + ", and ".join(summary_bits) + "."

        lines = ["Here is what I currently know about you:"]
        for point in points:
            lines.append(f"- {point}")
        for item in memory_lines:
            lines.append(f"- {item}")

        if len(lines) == 1:
            lines.append("- I have only limited details so far.")
            lines.append("- Share your likes, goals, and preferences and I will remember them.")
        return "\n".join(lines)
