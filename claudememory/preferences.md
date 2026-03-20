# User Preferences

Last updated: 2026-03-20

## Product Behavior Priorities
- Ask for missing core profile data when advice is personalized.
- Core fields to prioritize: monthly income, monthly expenses, monthly savings, existing investments.
- Avoid topic drift: profile answers must not be rerouted into unrelated live-data prompts.
- Prioritize complete, high-quality voice answers over aggressive latency trimming.
- Keep voice answers concise but fully finished (no cutoff or mid-sentence truncation).
- Avoid hardcoded keyword-only routing/matching; prefer model/context-driven and language-agnostic logic.
- Prefer hybrid live-data architecture: structured providers first, generalized web fallback for unsupported/failing live asks.
- Do not hardcode specific names, symbols, or terms in routing logic — let the model classify and web search resolve.

## Voice UX Priorities
- Responses must sound natural when read aloud — no tickers, ISO timestamps, or raw decimals.
- Prices should be in INR, rounded, comma-formatted.
- Prefer shorter responses; reduce TTS time without cutting answers mid-thought.

## Testing Workflow Preferences
- Compare multiple app variants in Voice Lab side-by-side before merging features.
- Favor practical fixes that can be validated quickly in `/voice-lab`.

## Collaboration Preferences
- Keep context persistent in `claudememory` so future sessions need less prompt history.
- Update memory files when behavior, constraints, or decisions change.
- Keep changes minimal and simple — avoid large rewrites or over-engineering.
