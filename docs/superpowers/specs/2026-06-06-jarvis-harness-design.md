# Jarvis Harness (`~/.jarvis`) — Design

**Date:** 2026-06-06
**Status:** Approved by Lucca

## Goal

Give Jarvis a persistent home folder — memories about the user, a user TODO
list, and self-written markdown skills — inspired by Hermes/OpenClaw-style
harnesses, but deliberately sandboxed: Jarvis can only ever write inside
`~/.jarvis`. PC interaction remains exclusively through the existing tools
(Apple Shortcuts, `open`, `mdfind`, `mo`).

## Decisions made during brainstorming

| Question | Decision |
|---|---|
| What is a "skill"? | Markdown instructions only (no executable code) |
| Memory recall | Index always in system prompt + read tool for full files |
| TODO design | `TODO.md` is source of truth; optional sync to Apple Reminders via a user-created shortcut |
| Memory autonomy | Jarvis saves memories autonomously and announces it aloud (user can veto) |
| Folder location | `~/.jarvis`, overridable via `JARVIS_HOME` env var |

## Folder layout

```
~/.jarvis/
├── MEMORY.md          # index: one line per memory (auto-maintained by code)
├── SKILLS.md          # index: one line per skill (auto-maintained by code)
├── TODO.md            # user todos, markdown checkboxes (- [ ] / - [x])
├── memories/
│   └── <slug>.md      # one fact/preference per file
└── skills/
    └── <slug>.md      # markdown recipes Jarvis writes for himself
```

- Created on first launch by `init_harness()`.
- Plain markdown everywhere; the user can read/edit the folder directly.
- `JARVIS_HOME` overrides the location (tests point it at tmp dirs).

## New module: `src/jarvis/harness.py`

Single module owning everything under `~/.jarvis`.

### Sandbox (enforced in code, not prompt)

A single `_resolve(kind, name)` helper:
1. Slugifies `name` (lowercase, alphanumerics and hyphens only).
2. Resolves the candidate path and verifies it is inside `JARVIS_HOME`.
3. Rejects anything else (`../` traversal, absolute paths, empty slugs) by
   returning an error string — never raises into the pipeline.

Every read/write goes through `_resolve`. No tool accepts arbitrary paths.

### Functions

- `init_harness()` — create folder structure if missing
- `save_memory(name, content)` / `read_memory(name)`
- `save_skill(name, content)` / `read_skill(name)`
- `add_todo(item)` / `complete_todo(item)` / `remove_todo(item)` / `list_todos()`
- `build_context()` — returns the system-prompt addendum (see below)

Notes:
- `save_memory` / `save_skill` rewrite the relevant index automatically; the
  model never maintains indexes.
- Saving to an existing slug updates the file (no duplicates).
- Todo matching for complete/remove is case-insensitive substring; ambiguous
  matches return all candidates so Jarvis can ask the user.
- Writes are atomic: write to a temp file in the same directory, then rename.

### Limits

- Memory/skill content capped at 10,000 characters; longer content is
  rejected with an error asking Jarvis to summarize.
- Each index capped at 200 entries; at the cap, new saves (to new slugs) are
  rejected with a "memory full, consider cleanup" error that Jarvis speaks
  aloud. Updates to existing slugs still succeed.

## Brain integration

### Four new tools (schemas live in `harness.py`, same pattern as `hands.py`)

| Tool | Parameters | Maps to |
|---|---|---|
| `save_memory` | `name`, `content` | `harness.save_memory` |
| `save_skill` | `name`, `content` | `harness.save_skill` |
| `read_harness_item` | `kind` (memory \| skill), `name` | `read_memory` / `read_skill` |
| `manage_todos` | `action` (add \| complete \| remove \| list), `item` (optional for list) | todo functions |

`_execute_tool` in `brain.py` gains the four dispatch branches.

### System prompt addendum (`build_context()`, built once per session)

1. Memory index lines (from `MEMORY.md`)
2. Skill index lines
3. Current open todos
4. Behavior rules:
   - Save memories autonomously when learning something durable about the
     user, and say so aloud so the user can veto ("forget that").
   - Before performing a task, check whether a skill exists for it and read it.
   - Skills are written the same way: autonomously, announced aloud.

### Optional Apple Reminders sync

After any TODO mutation, if a shortcut named `Sync Jarvis Todos` exists in the
discovered shortcuts list, run it with the current `TODO.md` contents as
input text. If the shortcut does not exist, skip silently. No other coupling.

## Error handling

- All harness functions return user-speakable strings; errors are
  `"Error: ..."` strings, never exceptions escaping into the pipeline
  (matching the `hands.py` convention).
- Path escape attempts → `"Error: invalid name"`.
- Atomic writes prevent corruption of `TODO.md` / indexes on crash.

## Testing (`tests/test_harness.py`)

Against a tmp `JARVIS_HOME`:
- `init_harness` creates the structure; idempotent.
- save/read roundtrips for memories and skills.
- Index stays in sync after saves; slug collisions update instead of duplicate.
- Path traversal (`../../etc/passwd`, absolute paths, weird unicode) rejected.
- Todo add/complete/remove/list, including ambiguous-match behavior.
- `build_context()` renders all three sections plus behavior rules.
- File-size and index-count limits enforced.
- `brain.py` dispatch tests for the four new tools (mocked, same style as
  `test_brain.py`).
- Reminders sync: runs shortcut when present in the list, skips when absent
  (mocked subprocess).

## Out of scope (v1)

- Executable skills of any kind.
- Conversation journaling / session logs / daily notes (possible later layer).
- Semantic/embedding search over memories.
- Any write access outside `~/.jarvis`.
