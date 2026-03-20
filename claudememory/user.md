# User And Project Snapshot

Last updated: 2026-03-20

## Project Purpose
- Voice-first personal finance assistant API.
- Goal: give personalized finance guidance using user profile, memory, and live data when needed.

## Runtime Surface
- FastAPI app import: `finance_project.main:app`
- Start: `python -m uvicorn finance_project.main:app --host 127.0.0.1 --port 9003`
- Core routes:
  - `POST /finance/session/init`
  - `POST /finance/session/ask`
  - `POST /finance/session/ask-voice`
  - `GET /voice-lab`
- Voice Lab is used for latency and behavior testing.

## Major Components
- `finance_project/api/finance_routes.py`: request orchestration, timing, STT/TTS flow.
- `finance_project/domains/finance/intent.py`: turn control (routing, profile extraction, live-data intent).
- `finance_project/domains/finance/engine.py`: end-to-end decision flow and prompt execution.
- `finance_project/domains/finance/prompt/`: prompt assembly and policy blocks.
- `finance_project/services/live_data_service.py`: weather/stock/news/commodity fetchers with provider-chain fallback.
- `finance_project/services/live_data_search.py`: DuckDuckGo web-search fallback.
- `finance_project/services/audio_service.py`: STT + TTS.
- `finance_project/core/postprocess/dispatcher.py`: async profile/memory write pipeline.
- `finance_project/services/memory_extraction_service.py`: LLM-based durable memory extraction.

## Current Focus
- Response naturalness for voice: shorter responses, no tickers/ISO timestamps in output, INR prices.
- MF knowledge injection working via `knowledge.py` distilled block.
- Pending: NISM Series V exam content, Indian MF NAV live data.

## Live Data Kinds (working as of 2026-03-20)
- weather, stock, stock_history, crypto, crypto_history, gold, gold_history, commodity, commodity_history, fx, news
- All prices converted to INR where applicable.
- Commodity/precious metal spot prices use LLM-extracted stooq ticker (e.g. XAGUSD for silver).
- Gold history uses dedicated stooq XAUUSD provider (bypasses .us suffix normalization).
- Stock fallback: `needs_input` removed from `_PROVIDER_SUCCESS_STATUSES` so unresolved equities fall through to DuckDuckGo web fallback.

## Knowledge Injection
- `finance_project/domains/finance/prompt/knowledge.py`: distilled MF knowledge block (~2,000 tokens).
- Injected into prompts when query contains MF-related keywords (MF_KEYWORDS set).
- Source: Eazyhaina 130 Q&A guide + NISM MCQ material (raw .docx files kept at project root).

## Memory / Self-Knowledge
- Stored memories use first-person ("my hobbies") — pronouns normalized to second-person ("your hobbies") in voice responses.
- Memory content truncation bug was in `reflection.py` storage, not engine display.

## Workspace Baseline
- Canonical working copy: root project folder (`finance_project/` package within repository root).
- Side-by-side variant folders were retired after baseline validation on 2026-03-16.
- Ongoing feature work should be applied only to the root baseline and checkpointed to GitHub.
