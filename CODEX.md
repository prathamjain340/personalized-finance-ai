# CODEX Session Protocol

At the start of every session, read these files before making changes:
- `codexmemory/user.md`
- `codexmemory/preferences.md`
- `codexmemory/decisions.md`
- `codexmemory/people.md`

Session workflow:
1. Start:
   - Read all `codexmemory/*.md` files.
   - Use them as the compact context baseline before scanning code.
2. During work:
   - Keep notes short and decision-focused.
   - Prefer updating memory files when behavior, architecture, or user preferences change.
3. End:
   - Update `codexmemory/decisions.md` with new technical decisions and rationale.
   - Update `codexmemory/preferences.md` if user priorities changed.
   - Update `codexmemory/user.md` with current project snapshot and active issues.
   - Update `codexmemory/people.md` only when team/stakeholder info changes.

Rules:
- Keep entries concise, factual, and dated.
- Avoid secrets, keys, tokens, or private credentials.
- Do not store raw logs; store distilled insights only.
- Avoid hardcoded behavior rules and keyword-only routing where possible.
- Prefer model/context-driven decisions and language-agnostic logic over fixed word lists.
