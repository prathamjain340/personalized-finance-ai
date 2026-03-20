# User And Project Snapshot

Last updated: 2026-03-17

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
- Improve profile-first personalization (income, expenses, savings, investments).
- Prevent wrong live-data jumps when user shares profile values.
- Keep high answer quality while expanding live coverage through hybrid routing.
- Commodity prices (silver, oil, etc.) now route to `commodity` live_data_kind → web fallback for accurate live data.
- Stock/history equity resolution failures now fall through to web fallback instead of dead-ending.

## Workspace Baseline
- Canonical working copy: root project folder (`finance_project/` package within repository root).
- Side-by-side variant folders were retired after baseline validation on 2026-03-16.
- Ongoing feature work should be applied only to the root baseline and checkpointed to GitHub.
