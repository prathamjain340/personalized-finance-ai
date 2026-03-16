# Technical Decisions Log

Last updated: 2026-03-16

## 2026-03-16 - Baseline Promotion Followed By Variant Cleanup
- Promoted the validated rebuild baseline into root `finance_project/` and checkpointed on GitHub (`personalized-finance-ai`).
- After validation, removed side-by-side rollback lanes from workspace: `finance_project_rebuild/` and `finance_project_backup/`.
- Root workspace is now the single source of truth for ongoing development.

Rationale:
- Reduce workspace clutter and avoid accidental edits against outdated variants.
- Keep delivery velocity high by maintaining one canonical path.

## 2026-03-16 - Profile-First Routing Guardrails
- Added stronger guardrails in `intent.py` so core profile updates (income/expenses/savings) clear `live_data_kind`.
- Added fast path for direct profile-answer turns to avoid unnecessary turn-control LLM latency and prevent live-data drift.
- Restricted expensive profile amount fallback LLM calls to pending-field scenarios only.

Rationale:
- Logs showed user profile answers being misrouted into stock-history/live-data flows.
- Duplicate `turn_control` LLM calls were contributing to latency.

## 2026-03-16 - Live Data Augmentation Scope
- Limited web-search live-data augmentation in `engine.py` to `news` only.
- Weather/stock updates now prefer direct structured formatter path without extra web-search LLM roundtrip.

Rationale:
- Weather and price facts are already structured and fresh from dedicated providers.
- Extra web-search pass added latency without consistent value.

## 2026-03-16 - Clarification Trigger Policy
- Updated clarification gating so profile follow-up is asked not only for `insufficient_data`, but also for `partial_data` in explicit finance categories.
- Personalized decision-support turns now ask for missing profile details earlier.

Rationale:
- User requirement: profile should be collected when needed for personalization, not skipped.

## 2026-03-16 - Voice Latency Trims
- Reduced turn-control default token budget and voice-response token budget in LLM client defaults.
- Tightened voice trimming word budget in `engine.py`.

Rationale:
- Reduce end-to-end voice latency while preserving answer quality.

## 2026-03-16 - Live Data Misroute Guard (Profile Turns)
- Added API-level sanitization to force `live_data_kind=None` when core profile updates are extracted in the same turn.
- Added guard to disable live-data routing when a pending profile field exists but no concrete live-data slot is present.

Rationale:
- Prevent profile answers (income/expenses/savings) from being misrouted into stock-history/live-data flow.
- Keep the flow language-agnostic and not dependent on keyword lists.

## 2026-03-16 - GitHub-Base Rebuild Minimal Core
- Rebuilt from GitHub base commit `03abff6` in side-by-side folder `finance_project_rebuild`.
- Added deterministic portfolio collection state in session: `profile_collection_goal` + ordered `profile_collection_queue`.
- Enforced ordered pending progression for portfolio planning: income -> expenses -> savings -> existing investments.
- Added route guard metadata (`route_guard_applied`) and engine flag (`enable_live_data`) to prevent live-data preemption during guarded turns.
- Updated turn-control schema to carry model-signaled `live_data_kind`/slots and `existing_investments` extraction.
- Removed extra voice-render LLM pass in voice route to reduce latency.

Rationale:
- Fix continuity drops where profile-answer turns were incorrectly jumping to stock/live-data prompts.
- Keep guarding model/context-driven and avoid introducing keyword-only routing rules.

## 2026-03-16 - Voice Completeness Over Aggressive Trim
- Disabled voice response trimming in the voice route (`apply_voice_trim=False`) to prevent mid-thought truncation.
- Increased explicit token budgets for voice/clarification operations in LLM client defaults.
- Updated voice prompt rules to require complete spoken sentences and ban numbered/bulleted list formatting.

Rationale:
- User validated that cutoff quality is a higher pain point than incremental latency.
- Better complete responses reduce perceived errors even with modestly higher response time.
