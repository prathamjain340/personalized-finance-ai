# User And Project Snapshot

Last updated: 2026-03-16

## Project Purpose
- Voice-first personal finance assistant API.
- Goal: give personalized finance guidance using user profile, memory, and live data when needed.

## Runtime Surface
- FastAPI app import: `finance_project.main:app`
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
- `finance_project/services/live_data_service.py`: weather/stock/news fetchers.
- `finance_project/services/live_data_search.py`: optional web-search augmentation.
- `finance_project/services/audio_service.py`: STT + TTS.
- `finance_project/core/postprocess/dispatcher.py`: async profile/memory write pipeline.
- `finance_project/services/memory_extraction_service.py`: LLM-based durable memory extraction.

## Current Focus
- Improve profile-first personalization (income, expenses, savings, investments).
- Prevent wrong live-data jumps when user shares profile values.
- Reduce voice latency (turn control + response generation + TTS perceived delay).

## Workspace Baseline
- Canonical working copy: root project folder (`finance_project/` package within repository root).
- Side-by-side variant folders were retired after baseline validation on 2026-03-16.
- Ongoing feature work should be applied only to the root baseline and checkpointed to GitHub.
