# User Preferences

Last updated: 2026-03-16

## Product Behavior Priorities
- Ask for missing core profile data when advice is personalized.
- Core fields to prioritize: monthly income, monthly expenses, monthly savings, existing investments.
- Avoid topic drift: profile answers must not be rerouted into unrelated live-data prompts.
- Prioritize complete, high-quality voice answers over aggressive latency trimming.
- Keep voice answers concise but fully finished (no cutoff or mid-sentence truncation).
- Avoid hardcoded keyword-only routing/matching; prefer model/context-driven and language-agnostic logic.
- Prefer hybrid live-data architecture: structured providers first, generalized web fallback for unsupported/failing live asks.

## Testing Workflow Preferences
- Compare multiple app variants in Voice Lab side-by-side before merging features.
- Favor practical fixes that can be validated quickly in `/voice-lab`.

## Collaboration Preferences
- Keep context persistent in `codexmemory` so future sessions need less prompt history.
- Update memory files when behavior, constraints, or decisions change.
