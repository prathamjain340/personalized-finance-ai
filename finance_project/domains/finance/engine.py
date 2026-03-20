# Finance Engine (orchestrator)
# This file is the brainstem of your system
# app/domains/finance/engine.py

import json
from typing import Any, Optional
import datetime
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
from finance_project.domains.finance.reflection import reflect_on_response_with_audit

from finance_project.core.summary.repository import get_domain_summary
from finance_project.core.profile.repository import get_profile, merge_profiles
from finance_project.core.memory.retriever import retrieve_memories
from finance_project.core.llm.client import generate_response
from finance_project.core.logging.logger import log_conversation, log_event
from finance_project.domains.finance.safety import select_response_mode
from finance_project.core.postprocess.dispatcher import enqueue_memory_candidates
from finance_project.core.memory.store import store_memory
from finance_project.core.storage.sqlite_db import get_connection, init_db
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
        apply_voice_trim: bool = True,
        live_data_kind: Optional[str] = None,
        live_data_slots: Optional[dict] = None,
        enable_live_data: bool = True,
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
        current_utterance = self._extract_current_utterance(query_for_routing)

        live_data = None
        if enable_live_data:
            live_data = maybe_fetch_live_data(
                query=query_for_routing,
                profile=profile,
                live_kind=live_data_kind,
                slots=live_data_slots,
            )
        if live_data:
            response = self._handle_live_data_response(
                query=query_for_routing,
                live_data=live_data,
                response_channel=response_channel,
                profile=profile,
                apply_voice_trim=apply_voice_trim,
            )
            memory_audit = self._capture_memory_candidates(
                user_id=user_id,
                raw_query=current_utterance,
                response=response,
                intent="live_data",
            )
            assistant_note_stored_count = self._store_assistant_note(
                user_id=user_id,
                response=response,
                intent="live_data",
                financial_category=str(live_data.get("kind") or "general"),
                pending_field=None,
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
                    "live_route": live_data.get("route", "structured"),
                    "live_provider": live_data.get("provider", "unknown"),
                    "ambiguity_clarification": bool(live_data.get("ambiguity_clarification")),
                    "live_failure_reason": live_data.get("failure_reason"),
                    "provider_attempts_count": len(live_data.get("provider_attempts") or []),
                    "memory_extracted_count": memory_audit.get("extracted_count", 0),
                    "memory_stored_count": memory_audit.get("stored_count", 0),
                    "assistant_note_stored_count": assistant_note_stored_count,
                },
            )
            return response, []

        if routing is None:
            routing = infer_query_metadata(
                query_for_routing,
                last_assistant_text=last_assistant_response,
            )

        if self_knowledge_request:
            self_knowledge_focus = self._infer_self_knowledge_focus(current_utterance)
            include_assistant_notes = (
                self_knowledge_focus == "assistant_recall"
                and self._should_include_assistant_notes_for_query(current_utterance)
            )
            memories = retrieve_memories(
                user_id=user_id,
                domain="finance",
                query=current_utterance,
                session_memory_usage=session_memory_usage,
                limit=28,
                allow_exposure_reuse=True,
                preferred_types=self._preferred_memory_types_for_focus(self_knowledge_focus),
                candidate_scan_limit=160,
                include_assistant_notes=include_assistant_notes,
            )
            selected_memories, selection_breakdown, goals_suppressed = self._select_self_knowledge_memories(
                utterance=current_utterance,
                memories=memories,
                focus=self_knowledge_focus,
                limit=7,
            )
            profile_snapshot = self._refresh_profile_snapshot(
                user_id=user_id,
                in_session_profile=profile,
            )
            response = self._build_self_knowledge_response(
                profile=profile_snapshot,
                memories=selected_memories,
                response_channel=response_channel,
                focus=self_knowledge_focus,
            )
            log_conversation(
                user_id=user_id,
                domain="finance",
                request=raw_query,
                response=response,
                metadata={
                    "intent": "self_knowledge",
                    "profile_fields_count": self._non_empty_profile_fields_count(profile_snapshot),
                    "memory_count": len(selected_memories),
                    "self_knowledge_focus": self_knowledge_focus,
                    "self_knowledge_selection_breakdown": selection_breakdown,
                    "self_knowledge_goal_suppressed": goals_suppressed,
                },
            )
            return response, selected_memories

        if not routing.finance_query:
            memories = retrieve_memories(
                user_id=user_id,
                domain="finance",
                query=current_utterance,
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
            prompt = self._with_language_lock(prompt, query_for_routing)
            response = generate_response(prompt, operation="out_of_domain_response")
            response = self._finalize_voice_response(
                response,
                response_channel=response_channel,
                apply_voice_trim=apply_voice_trim,
            )
            memory_audit = self._capture_memory_candidates(
                user_id=user_id,
                raw_query=current_utterance,
                response=response,
                intent="out_of_domain",
            )
            assistant_note_stored_count = self._store_assistant_note(
                user_id=user_id,
                response=response,
                intent="out_of_domain",
                financial_category=routing.financial_category,
                pending_field=pending_field,
            )
            log_conversation(
                user_id=user_id,
                domain="finance",
                request=raw_query,
                response=response,
                metadata={
                    "intent": "out_of_domain",
                    "small_talk": routing.small_talk,
                    "memory_extracted_count": memory_audit.get("extracted_count", 0),
                    "memory_stored_count": memory_audit.get("stored_count", 0),
                    "assistant_note_stored_count": assistant_note_stored_count,
                },
            )
            return response, memories

        # 1. Classify user intent (finance-only)
        intent = routing.intent

        # Map to financial decision type
        financial_category = routing.financial_category

        # Check profile sufficiency
        profile_state, missing_fields = analyze_profile_gaps(financial_category, profile)
        if pending_field and pending_field in missing_fields:
            missing_fields = [pending_field]

        signals = self._safe_compute_signals(profile)

        # 3. Fetch finance domain summary (if exists)
        domain_summary = get_domain_summary(user_id, domain="finance")

        # 4. Retrieve relevant long-term memories (read-only)
        memories = retrieve_memories(
            user_id=user_id,
            domain="finance",
            query=current_utterance,
            session_memory_usage=session_memory_usage,
            limit=3,
        )

        response_mode = select_response_mode(
            intent=intent,
            profile=profile,
            memories=memories,
        )

        force_pending_clarification = bool(pending_field and pending_field in missing_fields)
        if force_pending_clarification or self._needs_clarification_first(
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
            prompt = self._with_language_lock(prompt, query_for_routing)
            response = generate_response(prompt, operation="clarification_response")
            response = self._finalize_voice_response(
                response,
                response_channel=response_channel,
                apply_voice_trim=apply_voice_trim,
            )
            memory_audit = self._capture_memory_candidates(
                user_id=user_id,
                raw_query=current_utterance,
                response=response,
                intent=intent,
            )
            assistant_note_stored_count = self._store_assistant_note(
                user_id=user_id,
                response=response,
                intent=intent,
                financial_category=financial_category,
                pending_field=missing_fields[0] if missing_fields else pending_field,
            )
            log_conversation(
                user_id=user_id,
                domain="finance",
                request=raw_query,
                response=response,
                metadata={
                    "intent": intent,
                    "profile_state": profile_state.value,
                    "missing_fields": missing_fields,
                    "memory_extracted_count": memory_audit.get("extracted_count", 0),
                    "memory_stored_count": memory_audit.get("stored_count", 0),
                    "assistant_note_stored_count": assistant_note_stored_count,
                },
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
        prompt = self._with_language_lock(prompt, query_for_routing)

        # 6. Call LLM (stateless reasoning)
        finance_operation = "finance_response_voice" if response_channel == "voice" else "finance_response"
        response = generate_response(prompt, operation=finance_operation)
        response = self._finalize_voice_response(
            response,
            response_channel=response_channel,
            apply_voice_trim=apply_voice_trim,
        )

        # 7. Log conversation (immutable)
        memory_audit = self._capture_memory_candidates(
            user_id=user_id,
            raw_query=current_utterance,
            response=response,
            intent=intent,
        )
        assistant_note_stored_count = self._store_assistant_note(
            user_id=user_id,
            response=response,
            intent=intent,
            financial_category=financial_category,
            pending_field=pending_field,
        )

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
                "memory_extracted_count": memory_audit.get("extracted_count", 0),
                "memory_stored_count": memory_audit.get("stored_count", 0),
                "assistant_note_stored_count": assistant_note_stored_count,
            },
        )

        return response, memories

    @staticmethod
    def _extract_current_utterance(raw_query: str) -> str:
        text = str(raw_query or "").strip()
        marker = "Current user follow-up:"
        if marker in text:
            _, tail = text.rsplit(marker, 1)
            followup = tail.strip()
            if followup:
                return followup
        return text

    @staticmethod
    def _preferred_response_language(raw_query: str) -> str:
        utterance = FinanceEngine._extract_current_utterance(raw_query)
        # Non-Latin user inputs should get Hindi content in Latin script (Hinglish).
        if re.search(r"[\u0900-\u097F]", utterance or ""):
            return "hi"
        if re.search(r"[\u0600-\u06FF]", utterance or ""):
            return "hi"
        if not re.search(r"[A-Za-z]", utterance or ""):
            return "hi"
        return "en"

    @staticmethod
    def _language_lock_instruction(raw_query: str) -> str:
        language = FinanceEngine._preferred_response_language(raw_query)
        if language == "hi":
            return (
                "LANGUAGE POLICY:\n"
                "- Reply in Hindi using Latin script only (Hinglish).\n"
                "- Do not use Devanagari or Urdu/Arabic scripts.\n"
                "- Keep numbers and tickers in Latin script."
            )
        return (
            "LANGUAGE POLICY:\n"
            "- Reply only in English using Latin script.\n"
            "- Do not use Hindi/Devanagari or Urdu/Arabic scripts."
        )

    @staticmethod
    def _with_language_lock(prompt: str, raw_query: str) -> str:
        return f"{prompt}\n\n{FinanceEngine._language_lock_instruction(raw_query)}"

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
    def _trim_voice_response(
        text: str,
        response_channel: str,
        max_words: int = 32,
        max_chars: int = 190,
    ) -> str:
        if response_channel != "voice":
            return text

        normalized = " ".join((text or "").split())
        if not normalized:
            return "Could you please repeat that?"

        # Prefer one sentence; optionally include a second only if still compact.
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", normalized) if s.strip()]
        if not sentences:
            sentences = [normalized]

        selected: list[str] = []
        word_count = 0
        for sentence in sentences:
            words = sentence.split()
            if not words:
                continue

            if not selected:
                if len(words) > max_words:
                    trimmed = " ".join(words[:max_words])
                    return FinanceEngine._ensure_terminal(trimmed[:max_chars].rstrip())
                selected.append(sentence)
                word_count += len(words)
                continue

            candidate = " ".join(selected + [sentence])
            if len(candidate.split()) <= max_words and len(candidate) <= max_chars:
                selected.append(sentence)
                word_count = len(candidate.split())
            break

        compact = " ".join(selected) if selected else normalized
        compact_words = compact.split()
        if len(compact_words) > max_words:
            compact = " ".join(compact_words[:max_words])
        if len(compact) > max_chars:
            compact = compact[:max_chars].rstrip()
        return FinanceEngine._ensure_terminal(compact)

    @staticmethod
    def _ensure_terminal(text: str) -> str:
        if not text:
            return text
        return text if text[-1] in ".!?" else f"{text}."

    @staticmethod
    def _finalize_voice_response(
        text: str,
        response_channel: str,
        apply_voice_trim: bool,
    ) -> str:
        normalized = " ".join((text or "").split()).strip()
        if response_channel != "voice":
            return normalized or text
        if not normalized:
            return "Could you please repeat that?"
        if not apply_voice_trim:
            return normalized
        return FinanceEngine._trim_voice_response(normalized, response_channel=response_channel)

    def _handle_live_data_response(
        self,
        query: str,
        live_data: dict,
        response_channel: str,
        profile: dict,
        apply_voice_trim: bool = True,
    ) -> str:
        status = live_data.get("status")
        if status in {"needs_input", "error", "blocked"}:
            return self._finalize_voice_response(
                str(live_data.get("message") or "I couldn't fetch that update right now."),
                response_channel=response_channel,
                apply_voice_trim=apply_voice_trim,
            )

        kind = str(live_data.get("kind") or "").lower()
        facts = [str(item).strip() for item in (live_data.get("facts") or []) if str(item).strip()]
        as_of = str(live_data.get("as_of") or "")

        if kind == "weather":
            response = self._format_weather_update(facts=facts, as_of=as_of)
        elif kind == "stock":
            response = self._format_stock_update(facts=facts, as_of=as_of)
        elif kind == "stock_history":
            response = self._format_stock_history_update(facts=facts, as_of=as_of)
        elif kind == "crypto":
            response = self._format_crypto_update(facts=facts, as_of=as_of)
        elif kind == "crypto_history":
            response = self._format_crypto_history_update(facts=facts, as_of=as_of)
        elif kind == "gold":
            response = self._format_gold_update(facts=facts, as_of=as_of)
        elif kind == "commodity":
            response = self._format_gold_update(facts=facts, as_of=as_of)
        elif kind == "gold_history":
            response = self._format_stock_history_update(facts=facts, as_of=as_of)
        elif kind == "commodity_history":
            response = self._format_stock_history_update(facts=facts, as_of=as_of)
        elif kind == "fx":
            response = self._format_fx_update(facts=facts, as_of=as_of)
        elif kind == "news":
            response = self._format_news_update(facts=facts, as_of=as_of)
        elif kind == "mf_nav":
            response = self._format_mf_nav_update(facts=facts, as_of=as_of)
        elif kind == "mf_history":
            response = self._format_mf_history_update(facts=facts, as_of=as_of)
        else:
            response = self._format_generic_live_update(facts=facts, as_of=as_of)

        return self._finalize_voice_response(
            response,
            response_channel=response_channel,
            apply_voice_trim=apply_voice_trim,
        )

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

        label = FinanceEngine._ticker_to_label(instrument) if instrument else None
        price_clean = FinanceEngine._clean_price(price) if price else None

        if label and price_clean:
            body = f"{label} is currently at {price_clean}"
        elif label:
            body = f"Here is the latest update for {label}"
        else:
            body = "Here is the latest stock update."

        if change:
            body += f", with a change of {FinanceEngine._clean_price(change)}"

        caution = " This is market data, not a buy or sell recommendation."
        return f"{body}.{caution}"

    @staticmethod
    def _format_stock_history_update(facts: list[str], as_of: str) -> str:
        instrument = FinanceEngine._first_fact_value(facts, "Instrument:")
        period = FinanceEngine._first_fact_value(facts, "Period:")
        start_price = FinanceEngine._first_fact_value(facts, "Start price:")
        end_price = FinanceEngine._first_fact_value(facts, "End price:")
        total_return = FinanceEngine._first_fact_value(facts, "Total return:")

        label = FinanceEngine._ticker_to_label(instrument) if instrument else None
        period_clean = FinanceEngine._clean_period(period) if period else None
        start_clean = FinanceEngine._clean_price(start_price) if start_price else None
        end_clean = FinanceEngine._clean_price(end_price) if end_price else None
        return_clean = FinanceEngine._clean_return(total_return) if total_return else None

        if label and start_clean and end_clean and period_clean and return_clean:
            body = f"{label} went from {start_clean} to {end_clean} between {period_clean}, a total return of {return_clean}"
        elif label and start_clean and end_clean and return_clean:
            body = f"{label} moved from {start_clean} to {end_clean}, a total return of {return_clean}"
        elif label and return_clean:
            body = f"{label} had a total return of {return_clean}"
        else:
            body = "Here is the historical performance."

        caution = " This is market data, not a buy or sell recommendation."
        return f"{body}.{caution}"

    @staticmethod
    def _format_crypto_update(facts: list[str], as_of: str) -> str:
        instrument = FinanceEngine._first_fact_value(facts, "Instrument:")
        usd_price = FinanceEngine._first_fact_value(facts, "Last price:")
        day_change = FinanceEngine._first_fact_value(facts, "24h change:")

        inr_price = FinanceEngine._first_fact_value(facts, "Last price INR:")
        label = FinanceEngine._ticker_to_label(instrument) if instrument else None
        price_clean = FinanceEngine._clean_price(usd_price) if usd_price else None

        if label and inr_price:
            body = f"{label} is currently at {inr_price} rupees"
        elif label and price_clean:
            body = f"{label} is currently at {price_clean}"
        elif label:
            body = f"Here is the latest update for {label}"
        else:
            body = "Here is the latest crypto update."

        if day_change:
            body += f", with a 24 hour change of {FinanceEngine._clean_return(day_change)}"

        caution = " This is market data, not a buy or sell recommendation."
        return f"{body}.{caution}"

    @staticmethod
    def _format_crypto_history_update(facts: list[str], as_of: str) -> str:
        instrument = FinanceEngine._first_fact_value(facts, "Instrument:")
        period = FinanceEngine._first_fact_value(facts, "Period:")
        start_price = FinanceEngine._first_fact_value(facts, "Start price:")
        end_price = FinanceEngine._first_fact_value(facts, "End price:")
        total_return = FinanceEngine._first_fact_value(facts, "Total return:")

        label = FinanceEngine._ticker_to_label(instrument) if instrument else None
        period_clean = FinanceEngine._clean_period(period) if period else None
        start_clean = FinanceEngine._clean_price(start_price) if start_price else None
        end_clean = FinanceEngine._clean_price(end_price) if end_price else None
        return_clean = FinanceEngine._clean_return(total_return) if total_return else None

        if label and start_clean and end_clean and period_clean and return_clean:
            body = f"{label} went from {start_clean} to {end_clean} between {period_clean}, a total return of {return_clean}"
        elif label and return_clean:
            body = f"{label} had a total return of {return_clean}"
        else:
            body = "Here is the historical crypto performance."

        caution = " This is market data, not a buy or sell recommendation."
        return f"{body}.{caution}"

    @staticmethod
    def _format_gold_update(facts: list[str], as_of: str) -> str:
        instrument = FinanceEngine._first_fact_value(facts, "Instrument:")
        price = FinanceEngine._first_fact_value(facts, "Last price:")
        change = FinanceEngine._first_fact_value(facts, "Change:")

        inr_10g = FinanceEngine._first_fact_value(facts, "Last price INR per 10g:")
        inr_price = FinanceEngine._first_fact_value(facts, "Last price INR:")
        label = FinanceEngine._ticker_to_label(instrument) if instrument else None
        price_clean = FinanceEngine._clean_price(price) if price else None

        if label and inr_10g:
            body = f"{label} is currently at {inr_10g} rupees per 10 grams"
        elif label and inr_price:
            body = f"{label} is currently at {inr_price} rupees per troy ounce"
        elif label and price_clean:
            body = f"{label} is currently at {price_clean}"
        elif label:
            body = f"Here is the latest update for {label}"
        else:
            body = "Here is the latest commodity update."

        if change:
            body += f", with a change of {FinanceEngine._clean_price(change)}"

        caution = " This is market data, not a buy or sell recommendation."
        return f"{body}.{caution}"

    @staticmethod
    def _format_fx_update(facts: list[str], as_of: str) -> str:
        pair = FinanceEngine._first_fact_value(facts, "Pair:")
        rate = FinanceEngine._first_fact_value(facts, "Rate:")

        if pair and rate:
            rate_clean = FinanceEngine._clean_price(rate)
            body = f"The exchange rate for {pair} is {rate_clean}"
        elif pair:
            body = f"Here is the latest exchange rate for {pair}"
        else:
            body = "Here is the latest forex update."

        caution = " This is market data, not a buy or sell recommendation."
        return f"{body}.{caution}"

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
    def _format_mf_nav_update(facts: list[str], as_of: str) -> str:
        _ = as_of
        name = FinanceEngine._first_fact_value(facts, "Fund:")
        nav = FinanceEngine._first_fact_value(facts, "NAV:")
        nav_date = FinanceEngine._first_fact_value(facts, "NAV Date:")
        if name and nav:
            nav_clean = FinanceEngine._clean_price(nav)
            date_part = f" as of {nav_date}" if nav_date else ""
            return f"The NAV of {name} is {nav_clean} rupees{date_part}. This is market data, not a buy or sell recommendation."
        return "I couldn't find the NAV for that fund right now."

    @staticmethod
    def _format_mf_history_update(facts: list[str], as_of: str) -> str:
        _ = as_of
        name = FinanceEngine._first_fact_value(facts, "Fund:")
        period = FinanceEngine._first_fact_value(facts, "Period:")
        start_nav = FinanceEngine._first_fact_value(facts, "Start NAV:")
        end_nav = FinanceEngine._first_fact_value(facts, "End NAV:")
        total_return = FinanceEngine._first_fact_value(facts, "Total return:")
        if name and start_nav and end_nav and total_return:
            start_clean = FinanceEngine._clean_price(start_nav)
            end_clean = FinanceEngine._clean_price(end_nav)
            return_clean = FinanceEngine._clean_return(total_return)
            period_clean = FinanceEngine._clean_period(period) if period else ""
            period_part = f" between {period_clean}" if period_clean else ""
            return (
                f"{name} went from {start_clean} to {end_clean} rupees{period_part}, "
                f"a total return of {return_clean}. This is market data, not a buy or sell recommendation."
            )
        return "I couldn't fetch the historical performance for that fund right now."

    @staticmethod
    def _first_fact_value(facts: list[str], prefix: str) -> str | None:
        for fact in facts:
            if fact.startswith(prefix):
                return fact[len(prefix) :].strip()
        return None

    @staticmethod
    def _ticker_to_label(ticker: str) -> str:
        """Convert a raw ticker/symbol to a human-readable name."""
        _KNOWN = {
            "xauusd": "Gold", "xagusd": "Silver", "xptusd": "Platinum",
            "xpdusd": "Palladium", "xrhusd": "Rhodium",
        }
        t = str(ticker or "").strip()
        lower = t.lower()
        if lower in _KNOWN:
            return _KNOWN[lower]
        # Strip common exchange suffixes (.us, .in, .ns, .bo, .l, etc.)
        clean = re.sub(r"\.(us|in|ns|bo|l|pa|de|hk|ax|to)$", "", lower, flags=re.IGNORECASE)
        return clean.upper() if clean else t

    @staticmethod
    def _clean_price(value_str: str) -> str:
        """Round and format a price string: '1978.36 USD' → '1,978 USD', '80.13' → '80'."""
        text = str(value_str or "").strip()
        m = re.match(r"^([+-]?[0-9,]+(?:\.[0-9]*)?)(.*)$", text)
        if not m:
            return text
        num_part = m.group(1).replace(",", "")
        suffix = m.group(2).strip()
        try:
            val = float(num_part)
        except ValueError:
            return text
        rounded = round(val)
        formatted = f"{rounded:,}"
        return f"{formatted} {suffix}".strip() if suffix else formatted

    @staticmethod
    def _clean_return(value_str: str) -> str:
        """'115.86%' → 'about 116 percent', '-5.2%' → 'about -5 percent'."""
        text = str(value_str or "").strip()
        m = re.match(r"^([+-]?\s*[0-9]+(?:\.[0-9]*)?)%?", text)
        if not m:
            return text
        try:
            val = float(m.group(1).replace(" ", ""))
        except ValueError:
            return text
        rounded = round(val)
        return f"about {rounded} percent"

    @staticmethod
    def _clean_period(period_str: str) -> str:
        """'2023-03-20 to 2026-03-16' → 'March 2023 to March 2026'."""
        text = str(period_str or "").strip()
        m = re.match(r"^(\d{4}-\d{2}-\d{2})\s+to\s+(\d{4}-\d{2}-\d{2})$", text)
        if not m:
            return text
        _MONTHS = ["January", "February", "March", "April", "May", "June",
                   "July", "August", "September", "October", "November", "December"]
        try:
            d1 = datetime.date.fromisoformat(m.group(1))
            d2 = datetime.date.fromisoformat(m.group(2))
            return f"{_MONTHS[d1.month - 1]} {d1.year} to {_MONTHS[d2.month - 1]} {d2.year}"
        except Exception:
            return text

    @staticmethod
    def _freshness_suffix(as_of: str) -> str:
        return ""

    @staticmethod
    def _memory_text_tokens(text: str) -> set[str]:
        return {token for token in re.findall(r"[a-z0-9]+", text.lower()) if len(token) >= 3}

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        normalized = " ".join(str(text or "").split()).strip()
        if not normalized:
            return []
        parts = [segment.strip() for segment in re.split(r"(?<=[.!?])\s+", normalized) if segment.strip()]
        return parts or [normalized]

    @staticmethod
    def _extract_last_question(text: str) -> str | None:
        sentences = FinanceEngine._split_sentences(text)
        for sentence in reversed(sentences):
            if sentence.endswith("?"):
                return sentence
        return None

    @staticmethod
    def _build_assistant_note_content(
        response: str,
        intent: str,
        financial_category: str | None = None,
        pending_field: str | None = None,
    ) -> str | None:
        normalized = " ".join(str(response or "").split()).strip()
        if not normalized:
            return None

        sentences = FinanceEngine._split_sentences(normalized)
        if not sentences:
            return None

        summary_sentence = ""
        for sentence in sentences:
            if not sentence.endswith("?"):
                summary_sentence = sentence
                break
        if not summary_sentence:
            summary_sentence = sentences[0]

        summary_sentence = summary_sentence.strip()
        if len(summary_sentence) > 96:
            summary_sentence = summary_sentence[:96].rstrip(" ,;:") + "."

        pending_question = FinanceEngine._extract_last_question(normalized)
        if pending_question and len(pending_question) > 96:
            pending_question = pending_question[:96].rstrip(" ,;:") + "?"

        topic = str(financial_category or intent or "general").strip().lower()
        if not topic:
            topic = "general"

        components = [f"topic={topic}", f"summary={summary_sentence}"]
        if pending_field:
            components.append(f"pending_field={pending_field}")
        if pending_question:
            components.append(f"pending_question={pending_question}")
        return "; ".join(components)

    @staticmethod
    def _should_include_assistant_notes_for_query(utterance: str) -> bool:
        text = str(utterance or "").strip()
        if not text:
            return False

        prompt = (
            "Determine whether the user is explicitly asking to recall prior assistant suggestions/advice.\n"
            "Return STRICT JSON only with key 'include_assistant_notes' as true or false.\n"
            "Use true only when the user clearly asks what the assistant suggested/recommended earlier.\n"
            f"User message: {text}"
        )
        raw = generate_response(prompt, operation="turn_control")
        payload = FinanceEngine._extract_json_payload(raw)
        if isinstance(payload, dict):
            return bool(payload.get("include_assistant_notes"))
        return False

    @staticmethod
    def _prune_assistant_notes(user_id: str, keep: int = 20) -> int:
        keep_limit = max(5, int(keep))
        try:
            init_db()
            with get_connection() as conn:
                cursor = conn.execute(
                    """
                    DELETE FROM user_memories
                    WHERE user_id = ?
                      AND domain = 'finance'
                      AND type = 'assistant_note'
                      AND id NOT IN (
                          SELECT id
                          FROM user_memories
                          WHERE user_id = ?
                            AND domain = 'finance'
                            AND type = 'assistant_note'
                          ORDER BY updated_at DESC
                          LIMIT ?
                      )
                    """,
                    (user_id, user_id, keep_limit),
                )
                return int(cursor.rowcount or 0)
        except Exception:
            return 0

    def _store_assistant_note(
        self,
        user_id: str,
        response: str,
        intent: str,
        financial_category: str | None = None,
        pending_field: str | None = None,
    ) -> int:
        # Keep these notes focused on assistant guidance, not on social chatter or self-knowledge echoes.
        if str(intent or "").strip().lower() in {"self_knowledge", "out_of_domain"}:
            log_event(
                event="assistant_note_pipeline_audit",
                metadata={
                    "user_id": user_id,
                    "intent": intent,
                    "extracted_count": 0,
                    "stored_count": 0,
                    "dropped_reason": "intent_filtered",
                },
            )
            return 0

        content = self._build_assistant_note_content(
            response=response,
            intent=intent,
            financial_category=financial_category,
            pending_field=pending_field,
        )
        if not content:
            log_event(
                event="assistant_note_pipeline_audit",
                metadata={
                    "user_id": user_id,
                    "intent": intent,
                    "extracted_count": 0,
                    "stored_count": 0,
                    "dropped_reason": "empty_content",
                },
            )
            return 0

        stored_count = 0
        try:
            stored = store_memory(
                user_id=user_id,
                domain="finance",
                memory={
                    "type": "assistant_note",
                    "content": content,
                    "confidence": 0.72,
                    "importance": 0.6,
                },
            )
            if stored is not None:
                stored_count = 1
        except Exception:
            stored_count = 0

        pruned_count = 0
        if stored_count:
            pruned_count = self._prune_assistant_notes(user_id=user_id, keep=20)

        log_event(
            event="assistant_note_pipeline_audit",
            metadata={
                "user_id": user_id,
                "intent": intent,
                "extracted_count": 1,
                "stored_count": stored_count,
                "pruned_count": pruned_count,
            },
        )
        return stored_count

    def _capture_memory_candidates(
        self,
        user_id: str,
        raw_query: str,
        response: str,
        intent: str,
    ) -> dict[str, Any]:
        result = reflect_on_response_with_audit(
            user_id=user_id,
            raw_query=raw_query,
            response=response,
            intent=intent,
        )
        candidates = list(result.get("memory_candidates") or [])
        audit = dict(result.get("audit") or {})
        dropped = dict(audit.get("dropped_reason_counts") or {})

        sync_types = {"goal", "preference", "interest", "dislike"}
        sync_candidates: list[dict] = []
        async_candidates: list[dict] = []
        for item in candidates:
            memory_type = str(item.get("type") or "").strip().lower()
            if memory_type in sync_types:
                sync_candidates.append(item)
            else:
                async_candidates.append(item)

        stored_sync = 0
        sync_failures = 0
        for candidate in sync_candidates:
            try:
                stored = store_memory(user_id=user_id, domain="finance", memory=candidate)
                if stored is not None:
                    stored_sync += 1
                else:
                    sync_failures += 1
            except Exception:
                sync_failures += 1

        if sync_failures:
            dropped["sync_store_failed"] = dropped.get("sync_store_failed", 0) + sync_failures

        queued_async = len(async_candidates)
        if async_candidates:
            enqueue_memory_candidates(
                user_id=user_id,
                domain="finance",
                memory_candidates=async_candidates,
            )

        summary = {
            "source": audit.get("source"),
            "extracted_count": int(audit.get("deduped_count", len(candidates))),
            "stored_count": stored_sync + queued_async,
            "stored_sync_count": stored_sync,
            "queued_async_count": queued_async,
            "dropped_reason_counts": dropped,
        }
        log_event(
            event="memory_pipeline_audit",
            metadata={
                "user_id": user_id,
                "intent": intent,
                "source": summary["source"],
                "extracted_count": summary["extracted_count"],
                "stored_count": summary["stored_count"],
                "stored_sync_count": summary["stored_sync_count"],
                "queued_async_count": summary["queued_async_count"],
                "dropped_reason_counts": summary["dropped_reason_counts"],
            },
        )
        return summary

    @staticmethod
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

    @staticmethod
    def _infer_self_knowledge_focus(utterance: str) -> str:
        text = str(utterance or "").strip()
        if not text:
            return "profile"

        prompt = (
            "Classify the user's self-knowledge question focus.\n"
            "Return STRICT JSON only with key 'focus'.\n"
            "Allowed values: general, profile, goal, interest, assistant_recall.\n"
            "Use 'interest' for hobbies/likes/dislikes/preferences.\n"
            "Use 'assistant_recall' when the user asks what the assistant suggested/recommended earlier.\n"
            f"User message: {text}"
        )
        raw = generate_response(prompt, operation="turn_control")
        payload = FinanceEngine._extract_json_payload(raw)
        if isinstance(payload, dict):
            focus = str(payload.get("focus") or "").strip().lower()
            if focus in {"general", "profile", "goal", "interest", "assistant_recall"}:
                return focus

        return "profile"

    @staticmethod
    def _preferred_memory_types_for_focus(focus: str) -> list[str]:
        if focus == "assistant_recall":
            return ["assistant_note"]
        if focus == "interest":
            return ["interest", "preference", "dislike"]
        if focus == "goal":
            return ["goal", "interest", "preference"]
        if focus == "profile":
            return ["interest", "preference", "dislike", "behavioral"]
        return ["interest", "preference", "dislike", "behavioral"]

    @staticmethod
    def _select_self_knowledge_memories(
        utterance: str,
        memories: list,
        focus: str,
        limit: int = 5,
    ) -> tuple[list, dict[str, int], bool]:
        _ = utterance
        interest_types = {"interest", "preference", "dislike", "hobby"}
        goal_types = {"goal"}
        behavioral_types = {"behavioral", "emotional"}
        assistant_note_types = {"assistant_note"}

        goals = [m for m in memories if str(getattr(m, "type", "") or "").lower() in goal_types]
        interests = [m for m in memories if str(getattr(m, "type", "") or "").lower() in interest_types]
        behavioral = [m for m in memories if str(getattr(m, "type", "") or "").lower() in behavioral_types]
        assistant_notes = [m for m in memories if str(getattr(m, "type", "") or "").lower() in assistant_note_types]
        other = [
            m
            for m in memories
            if str(getattr(m, "type", "") or "").lower()
            not in (goal_types | interest_types | behavioral_types | assistant_note_types)
        ]
        goals_suppressed = focus in {"general", "profile", "interest", "assistant_recall"} and bool(goals)

        ordered: list = []
        fallback_pool: list = []
        if focus == "assistant_recall":
            ordered.extend(assistant_notes[:4])
            fallback_pool = assistant_notes + other
        elif focus == "interest":
            ordered.extend(interests[:3])
            ordered.extend(behavioral[:1])
            fallback_pool = interests + behavioral + other
        elif focus == "goal":
            ordered.extend(goals[:2])
            ordered.extend(interests[:2])
            ordered.extend(behavioral[:1])
            fallback_pool = goals + interests + behavioral + other
        elif focus == "profile":
            ordered.extend(interests[:2])
            ordered.extend(behavioral[:1])
            ordered.extend(other[:1])
            fallback_pool = interests + behavioral + other
        else:
            ordered.extend(interests[:2])
            ordered.extend(behavioral[:1])
            ordered.extend(other[:1])
            fallback_pool = interests + behavioral + other

        selected: list = []
        seen: set[str] = set()
        for memory in ordered + fallback_pool:
            content = str(getattr(memory, "content", "") or "").strip().lower()
            if not content or content in seen:
                continue
            seen.add(content)
            selected.append(memory)
            if len(selected) >= limit:
                break

        breakdown: dict[str, int] = {}
        for memory in selected:
            memory_type = str(getattr(memory, "type", "") or "").strip().lower()
            breakdown[memory_type] = breakdown.get(memory_type, 0) + 1
        return selected, breakdown, goals_suppressed

    @staticmethod
    def _prepare_self_knowledge_memory_items(memories: list, limit: int = 6) -> list[dict[str, str]]:
        items: list[dict[str, str]] = []
        seen: set[str] = set()
        for memory in memories:
            memory_type = str(getattr(memory, "type", "") or "").strip().lower()
            content = str(getattr(memory, "content", "") or "").strip()
            line = FinanceEngine._paraphrase_memory(memory_type, content)
            if not line:
                continue
            norm = re.sub(r"\s+", " ", line.lower()).strip(" .,!?:;")
            if not norm or norm in seen:
                continue
            seen.add(norm)
            items.append({"type": memory_type, "line": line.rstrip(".")})
            if len(items) >= limit:
                break
        return items

    @staticmethod
    def _paraphrase_memory(memory_type: str, content: str) -> str:
        text = re.sub(r"\s+", " ", str(content or "")).strip().strip(" .")
        if not text:
            return ""

        lowered = text.lower()
        if lowered.startswith("user "):
            text = text[5:].strip()
            lowered = text.lower()

        if memory_type == "goal":
            value = re.sub(r"^(?:goal\s*:|goal is to)\s*", "", text, flags=re.IGNORECASE).strip(" .")
            value = re.split(r"\b(?:should i|can i|what should)\b", value, maxsplit=1, flags=re.IGNORECASE)[0].strip(" .")
            if value.lower().startswith("to "):
                value = value[3:].strip()
            return f"your goal is to {value}" if value else ""

        if memory_type in {"interest", "hobby"}:
            value = re.sub(r"^(?:interest\s*:|hobby\s*:|hobbies\s*:|likes?\s*)", "", text, flags=re.IGNORECASE).strip(" .")
            if value.lower().startswith("eat "):
                value = "eating " + value[4:].strip()
            return f"you enjoy {value}" if value else ""

        if memory_type == "dislike":
            value = re.sub(r"^(?:dislikes?\s*)", "", text, flags=re.IGNORECASE).strip(" .")
            return f"you dislike {value}" if value else ""

        if memory_type == "preference":
            if re.match(r"^likes?\s+", text, flags=re.IGNORECASE):
                value = re.sub(r"^likes?\s+", "", text, flags=re.IGNORECASE).strip(" .")
                return f"you like {value}" if value else ""
            if re.match(r"^prefers?\s+", text, flags=re.IGNORECASE):
                value = re.sub(r"^prefers?\s+", "", text, flags=re.IGNORECASE).strip(" .")
                return f"you prefer {value}" if value else ""
            return f"you prefer {text.strip(' .')}"

        if memory_type == "behavioral":
            value = re.sub(r"^major spending categories\s*:\s*", "", text, flags=re.IGNORECASE).strip(" .")
            return f"your major spending categories are {value}" if value else ""

        if memory_type == "emotional":
            value = re.sub(r"^emotional cue\s*:\s*", "", text, flags=re.IGNORECASE).strip(" .")
            return value or ""

        if memory_type == "assistant_note":
            body = re.sub(r"^\s*assistant\s*note\s*:\s*", "", text, flags=re.IGNORECASE).strip(" .")
            parts = [segment.strip() for segment in body.split(";") if segment.strip()]
            summary = ""
            pending = ""
            for part in parts:
                lowered = part.lower()
                if lowered.startswith("summary="):
                    summary = part.split("=", 1)[1].strip(" .")
                elif lowered.startswith("pending_question="):
                    pending = part.split("=", 1)[1].strip()
            if summary and pending:
                return f"earlier I suggested {summary} and asked {pending}"
            if summary:
                return f"earlier I suggested {summary}"
            if pending:
                return f"earlier I asked {pending}"
            return ""

        return text

    @staticmethod
    def _refresh_profile_snapshot(user_id: str, in_session_profile: dict | None) -> dict:
        session_profile = dict(in_session_profile or {})
        try:
            persisted = get_profile(user_id)
            return merge_profiles(persisted, session_profile)
        except Exception:
            return session_profile

    @staticmethod
    def _non_empty_profile_fields_count(profile: dict | None) -> int:
        data = dict(profile or {})
        count = 0
        for value in data.values():
            if value is None:
                continue
            if isinstance(value, str) and not value.strip():
                continue
            count += 1
        return count

    @staticmethod
    def _build_self_knowledge_response(profile: dict, memories: list, response_channel: str, focus: str = "general") -> str:
        profile_bits: list[str] = []
        name = profile.get("preferred_name") or profile.get("full_name") or profile.get("name")
        city = profile.get("city")
        income = profile.get("monthly_income")
        expenses = profile.get("monthly_expenses")
        savings = profile.get("monthly_savings")

        if income not in (None, ""):
            profile_bits.append(f"monthly income is INR {income}")
        if expenses not in (None, ""):
            profile_bits.append(f"monthly expenses are INR {expenses}")
        if savings not in (None, ""):
            profile_bits.append(f"monthly savings are INR {savings}")
        if name:
            profile_bits.insert(0, f"your name is {name}")
        if city:
            profile_bits.append(f"you are in {city}")

        memory_items = FinanceEngine._prepare_self_knowledge_memory_items(memories, limit=6)
        grouped: dict[str, list[str]] = {
            "goal": [],
            "interest": [],
            "preference": [],
            "dislike": [],
            "behavioral": [],
            "assistant_note": [],
            "other": [],
        }
        for item in memory_items:
            memory_type = item["type"]
            line = item["line"]
            if memory_type in grouped:
                grouped[memory_type].append(line)
            elif memory_type in {"hobby"}:
                grouped["interest"].append(line)
            else:
                grouped["other"].append(line)
        if focus != "assistant_recall":
            grouped["assistant_note"] = []
        effective_focus = focus
        if focus == "assistant_recall" and not grouped["assistant_note"]:
            effective_focus = "profile"

        if response_channel == "voice":
            if not profile_bits and not memory_items:
                return "I only know limited details so far. You can share your preferences and I'll remember them."

            facts_used = 0
            parts: list[str] = []
            if profile_bits:
                selected_profile = profile_bits[:3]
                facts_used += len(selected_profile)
                parts.append("Profile: " + ", ".join(selected_profile))

            memory_sentences: list[str] = []
            if effective_focus == "assistant_recall":
                memory_sentences.extend(grouped["assistant_note"][:3])
            elif effective_focus == "goal":
                memory_sentences.extend(grouped["goal"][:2])
                memory_sentences.extend(grouped["interest"][:1])
            elif effective_focus == "interest":
                memory_sentences.extend(grouped["interest"][:2])
                memory_sentences.extend(grouped["preference"][:2])
            elif effective_focus == "profile":
                memory_sentences.extend(grouped["interest"][:3])
                memory_sentences.extend(grouped["preference"][:2])
                memory_sentences.extend(grouped["dislike"][:1])
                memory_sentences.extend(grouped["behavioral"][:1])
            else:
                memory_sentences.extend(grouped["interest"][:2])
                memory_sentences.extend(grouped["preference"][:1])
                memory_sentences.extend(grouped["dislike"][:1])
                memory_sentences.extend(grouped["behavioral"][:1])

            memory_sentences = [item for item in memory_sentences if item]
            non_profile_limit = max(1, 6 - facts_used)
            selected_memory = memory_sentences[:non_profile_limit]
            if selected_memory:
                facts_used += len(selected_memory)
                if effective_focus == "assistant_recall":
                    parts.append("Earlier suggestions: " + "; ".join(selected_memory))
                else:
                    parts.append("Personal details: " + "; ".join(selected_memory))

            if not parts:
                return "I only know limited details so far. You can share your preferences and I'll remember them."
            return "Here is what I know: " + ". ".join(parts) + "."

        lines = ["Here is what I currently know about you:"]
        if profile_bits:
            lines.append(f"- Profile: {', '.join(profile_bits[:4])}.")
        if effective_focus == "assistant_recall":
            if grouped["assistant_note"]:
                lines.append(f"- Earlier suggestions: {'; '.join(grouped['assistant_note'][:3])}.")
        elif effective_focus == "goal" and grouped["goal"]:
            lines.append(f"- Goals: {'; '.join(grouped['goal'][:2])}.")
        interests = grouped["interest"] + grouped["preference"] + grouped["dislike"]
        if interests:
            lines.append(f"- Interests and preferences: {'; '.join(interests[:3])}.")
        if grouped["behavioral"]:
            lines.append(f"- Behavior notes: {'; '.join(grouped['behavioral'][:2])}.")
        if grouped["other"]:
            lines.append(f"- Other notes: {'; '.join(grouped['other'][:2])}.")

        if len(lines) == 1:
            lines.append("- I have only limited details so far.")
            lines.append("- Share your goals, hobbies, and preferences and I will remember them.")
        return "\n".join(lines)
