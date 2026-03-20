# Claude Session Protocol

At the start of every session, read these files before making any changes:
- `claudememory/user.md` — project snapshot, runtime, major components, current focus
- `claudememory/preferences.md` — product behavior priorities and collaboration preferences
- `claudememory/decisions.md` — technical decisions log with rationale
- `claudememory/people.md` — stakeholders and roles

## Session Workflow

### Start
1. Read all `claudememory/*.md` files.
2. Use them as the compact context baseline before scanning code.
3. Ask clarifying questions if the task is ambiguous before starting.

### During Work
- Keep changes minimal and focused — avoid over-engineering.
- Prefer model/context-driven logic over hardcoded keyword rules.
- Update memory files when behavior, architecture, or preferences change.

### End
- Update `claudememory/decisions.md` with new technical decisions and rationale (dated entry).
- Update `claudememory/preferences.md` if user priorities or collaboration style changed.
- Update `claudememory/user.md` with the current project snapshot and active focus areas.
- Update `claudememory/people.md` only when team or stakeholder information changes.

## Rules
- Keep entries concise, factual, and dated.
- Do not store secrets, API keys, or credentials in memory files.
- Do not store raw logs — store distilled insights only.
- Avoid hardcoded behavior rules and keyword-only routing in code.
- Prefer language-agnostic and model-driven decisions.
