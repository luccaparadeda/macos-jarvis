# Jarvis Harness (`~/.jarvis`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give Jarvis a persistent `~/.jarvis` folder with memories, a user TODO list, and self-written markdown skills — sandboxed so Jarvis can only write inside that folder.

**Architecture:** A new `src/jarvis/harness.py` module owns all `~/.jarvis` I/O behind a slug/sandbox helper, exposes 4 new tools (`save_memory`, `save_skill`, `read_harness_item`, `manage_todos`) wired into `brain.py`'s dispatch, and builds a per-session system-prompt addendum (`build_context()`). Spec: `docs/superpowers/specs/2026-06-06-jarvis-harness-design.md`.

**Tech Stack:** Python 3.11+, stdlib only (`pathlib`, `re`, `os`, `tempfile`), pytest + pytest-asyncio (asyncio_mode=auto is already set, but existing tests use explicit `@pytest.mark.asyncio` — follow that), `uv run pytest`.

**Conventions to follow (from existing code):**
- Tool schema builders return OpenAI-style dicts (`{"type": "function", "function": {...}}`) — see `src/jarvis/hands.py:16`.
- Tool functions return user-speakable strings; errors are `"Error: ..."` strings, never exceptions.
- Tests use `unittest.mock.patch`/`AsyncMock` — see `tests/test_hands.py`.
- Run tests with `uv run pytest tests/test_harness.py -v`.

---

### Task 1: Harness home + init

**Files:**
- Create: `src/jarvis/harness.py`
- Create: `tests/test_harness.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_harness.py`:

```python
import pytest

from jarvis import harness


@pytest.fixture(autouse=True)
def jarvis_home(tmp_path, monkeypatch):
    home = tmp_path / ".jarvis"
    monkeypatch.setenv("JARVIS_HOME", str(home))
    return home


class TestInit:
    def test_jarvis_home_uses_env_var(self, jarvis_home):
        assert harness.jarvis_home() == jarvis_home

    def test_init_creates_structure(self, jarvis_home):
        harness.init_harness()
        assert (jarvis_home / "memories").is_dir()
        assert (jarvis_home / "skills").is_dir()
        assert (jarvis_home / "MEMORY.md").is_file()
        assert (jarvis_home / "SKILLS.md").is_file()
        assert (jarvis_home / "TODO.md").is_file()

    def test_init_is_idempotent_and_preserves_content(self, jarvis_home):
        harness.init_harness()
        (jarvis_home / "TODO.md").write_text("- [ ] buy milk\n", encoding="utf-8")
        harness.init_harness()
        assert (jarvis_home / "TODO.md").read_text(encoding="utf-8") == "- [ ] buy milk\n"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_harness.py -v`
Expected: FAIL with `ImportError: cannot import name 'harness'` (or `AttributeError`)

- [ ] **Step 3: Write minimal implementation**

Create `src/jarvis/harness.py`:

```python
import os
from pathlib import Path

KINDS = {"memory": "memories", "skill": "skills"}
INDEX_FILES = {"memory": "MEMORY.md", "skill": "SKILLS.md"}


def jarvis_home() -> Path:
    return Path(os.environ.get("JARVIS_HOME", str(Path.home() / ".jarvis")))


def init_harness() -> Path:
    home = jarvis_home()
    for sub in KINDS.values():
        (home / sub).mkdir(parents=True, exist_ok=True)
    for name in (*INDEX_FILES.values(), "TODO.md"):
        path = home / name
        if not path.exists():
            path.write_text("", encoding="utf-8")
    return home
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_harness.py -v`
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add src/jarvis/harness.py tests/test_harness.py
git commit -m "feat: add harness module with ~/.jarvis init"
```

---

### Task 2: Slug + sandbox resolver

**Files:**
- Modify: `src/jarvis/harness.py`
- Modify: `tests/test_harness.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_harness.py`:

```python
class TestResolve:
    def test_simple_name(self, jarvis_home):
        harness.init_harness()
        path = harness._resolve("memory", "Coffee Preference")
        assert path == jarvis_home / "memories" / "coffee-preference.md"

    def test_traversal_is_neutralized(self, jarvis_home):
        harness.init_harness()
        path = harness._resolve("memory", "../../etc/passwd")
        assert path is not None
        assert path.parent == (jarvis_home / "memories").resolve()
        assert path.name == "etc-passwd.md"

    def test_absolute_path_is_neutralized(self, jarvis_home):
        harness.init_harness()
        path = harness._resolve("skill", "/usr/local/bin/evil")
        assert path is not None
        assert path.parent == (jarvis_home / "skills").resolve()

    def test_empty_and_symbol_only_names_rejected(self, jarvis_home):
        harness.init_harness()
        assert harness._resolve("memory", "") is None
        assert harness._resolve("memory", "  ") is None
        assert harness._resolve("memory", "../..") is None
        assert harness._resolve("memory", "!!!") is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_harness.py::TestResolve -v`
Expected: FAIL with `AttributeError: module 'jarvis.harness' has no attribute '_resolve'`

- [ ] **Step 3: Write minimal implementation**

Add to `src/jarvis/harness.py` (after the imports, add `import re`):

```python
def _slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def _resolve(kind: str, name: str) -> Path | None:
    slug = _slugify(name)
    if not slug:
        return None
    folder = (jarvis_home() / KINDS[kind]).resolve()
    path = (folder / f"{slug}.md").resolve()
    if path.parent != folder:
        return None
    return path
```

Note: slugification (only `[a-z0-9-]` survives) already makes traversal impossible — `../../etc/passwd` becomes `etc-passwd`. The `path.parent != folder` check is defense in depth.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_harness.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/jarvis/harness.py tests/test_harness.py
git commit -m "feat: add sandboxed name resolver to harness"
```

---

### Task 3: Save/read memories and skills with auto-maintained indexes

**Files:**
- Modify: `src/jarvis/harness.py`
- Modify: `tests/test_harness.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_harness.py`:

```python
class TestMemoriesAndSkills:
    def test_save_and_read_memory_roundtrip(self, jarvis_home):
        harness.init_harness()
        result = harness.save_memory("Coffee Preference", "# Coffee\nUser likes oat-milk flat whites.")
        assert result == "Saved memory 'coffee-preference'"
        assert harness.read_item("memory", "coffee preference") == "# Coffee\nUser likes oat-milk flat whites."

    def test_save_and_read_skill_roundtrip(self, jarvis_home):
        harness.init_harness()
        harness.save_skill("Morning Routine", "Open Spotify, then read calendar.")
        assert harness.read_item("skill", "morning routine") == "Open Spotify, then read calendar."

    def test_index_updated_on_save(self, jarvis_home):
        harness.init_harness()
        harness.save_memory("music", "# Music\nPrefers Spotify over Apple Music.")
        index = (jarvis_home / "MEMORY.md").read_text(encoding="utf-8")
        assert "- music: Music" in index

    def test_index_summary_uses_first_nonempty_line(self, jarvis_home):
        harness.init_harness()
        harness.save_skill("greet", "\n\n# Greeting style\nBe brief.")
        index = (jarvis_home / "SKILLS.md").read_text(encoding="utf-8")
        assert "- greet: Greeting style" in index

    def test_same_slug_updates_instead_of_duplicating(self, jarvis_home):
        harness.init_harness()
        harness.save_memory("music", "old")
        harness.save_memory("Music!", "new")
        assert harness.read_item("memory", "music") == "new"
        index = (jarvis_home / "MEMORY.md").read_text(encoding="utf-8")
        assert index.count("- music:") == 1

    def test_read_missing_returns_error(self, jarvis_home):
        harness.init_harness()
        assert harness.read_item("memory", "nope") == "Error: no memory named 'nope'"

    def test_read_invalid_kind_returns_error(self, jarvis_home):
        harness.init_harness()
        assert harness.read_item("recipe", "x") == "Error: invalid kind"

    def test_invalid_name_returns_error(self, jarvis_home):
        harness.init_harness()
        assert harness.save_memory("!!!", "content") == "Error: invalid name"

    def test_content_too_long_rejected(self, jarvis_home):
        harness.init_harness()
        result = harness.save_memory("big", "x" * 10_001)
        assert result == "Error: content too long, please summarize it"
        assert harness.read_item("memory", "big").startswith("Error:")

    def test_index_cap_rejects_new_but_allows_updates(self, jarvis_home, monkeypatch):
        harness.init_harness()
        monkeypatch.setattr(harness, "MAX_INDEX_ENTRIES", 2)
        harness.save_memory("one", "1")
        harness.save_memory("two", "2")
        result = harness.save_memory("three", "3")
        assert result == "Error: memory storage full, consider cleaning up old entries"
        assert harness.save_memory("one", "updated") == "Saved memory 'one'"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_harness.py::TestMemoriesAndSkills -v`
Expected: FAIL with `AttributeError: ... no attribute 'save_memory'`

- [ ] **Step 3: Write minimal implementation**

Add to `src/jarvis/harness.py` (add `import tempfile` to imports):

```python
MAX_CONTENT_CHARS = 10_000
MAX_INDEX_ENTRIES = 200


def _atomic_write(path: Path, content: str) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(tmp, path)
    except BaseException:
        os.unlink(tmp)
        raise


def _first_line(content: str) -> str:
    for line in content.splitlines():
        cleaned = line.strip().lstrip("#").strip()
        if cleaned:
            return cleaned[:80]
    return ""


def _rebuild_index(kind: str) -> None:
    folder = jarvis_home() / KINDS[kind]
    lines = [
        f"- {f.stem}: {_first_line(f.read_text(encoding='utf-8'))}"
        for f in sorted(folder.glob("*.md"))
    ]
    _atomic_write(jarvis_home() / INDEX_FILES[kind], "\n".join(lines) + ("\n" if lines else ""))


def _save_item(kind: str, name: str, content: str) -> str:
    path = _resolve(kind, name)
    if path is None:
        return "Error: invalid name"
    if len(content) > MAX_CONTENT_CHARS:
        return "Error: content too long, please summarize it"
    folder = jarvis_home() / KINDS[kind]
    if not path.exists() and len(list(folder.glob("*.md"))) >= MAX_INDEX_ENTRIES:
        return f"Error: {kind} storage full, consider cleaning up old entries"
    _atomic_write(path, content)
    _rebuild_index(kind)
    return f"Saved {kind} '{path.stem}'"


def save_memory(name: str, content: str) -> str:
    return _save_item("memory", name, content)


def save_skill(name: str, content: str) -> str:
    return _save_item("skill", name, content)


def read_item(kind: str, name: str) -> str:
    if kind not in KINDS:
        return "Error: invalid kind"
    path = _resolve(kind, name)
    if path is None or not path.exists():
        return f"Error: no {kind} named '{name}'"
    return path.read_text(encoding="utf-8")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_harness.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/jarvis/harness.py tests/test_harness.py
git commit -m "feat: add memory/skill storage with auto-maintained indexes"
```

---

### Task 4: TODO management

**Files:**
- Modify: `src/jarvis/harness.py`
- Modify: `tests/test_harness.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_harness.py`:

```python
class TestTodos:
    def test_add_todo(self, jarvis_home):
        harness.init_harness()
        result = harness.add_todo("buy milk")
        assert result == "Added todo: buy milk"
        todo = (jarvis_home / "TODO.md").read_text(encoding="utf-8")
        assert "- [ ] buy milk" in todo

    def test_add_empty_todo_rejected(self, jarvis_home):
        harness.init_harness()
        assert harness.add_todo("  ") == "Error: empty todo"

    def test_list_todos(self, jarvis_home):
        harness.init_harness()
        harness.add_todo("buy milk")
        harness.add_todo("call mom")
        result = harness.list_todos()
        assert "- [ ] buy milk" in result
        assert "- [ ] call mom" in result

    def test_list_empty(self, jarvis_home):
        harness.init_harness()
        assert harness.list_todos() == "No todos."

    def test_complete_todo_substring_case_insensitive(self, jarvis_home):
        harness.init_harness()
        harness.add_todo("Buy milk at the store")
        result = harness.complete_todo("MILK")
        assert result == "Completed: Buy milk at the store"
        todo = (jarvis_home / "TODO.md").read_text(encoding="utf-8")
        assert "- [x] Buy milk at the store" in todo

    def test_complete_no_match(self, jarvis_home):
        harness.init_harness()
        harness.add_todo("buy milk")
        assert harness.complete_todo("dentist") == "Error: no open todo matching 'dentist'"

    def test_complete_already_done_not_matched(self, jarvis_home):
        harness.init_harness()
        harness.add_todo("buy milk")
        harness.complete_todo("milk")
        assert harness.complete_todo("milk") == "Error: no open todo matching 'milk'"

    def test_complete_ambiguous_lists_candidates(self, jarvis_home):
        harness.init_harness()
        harness.add_todo("buy milk")
        harness.add_todo("buy milkshake mix")
        result = harness.complete_todo("milk")
        assert result.startswith("Ambiguous, matches: ")
        assert "buy milk" in result
        assert "buy milkshake mix" in result

    def test_remove_todo(self, jarvis_home):
        harness.init_harness()
        harness.add_todo("buy milk")
        harness.add_todo("call mom")
        result = harness.remove_todo("milk")
        assert result == "Removed: buy milk"
        todo = (jarvis_home / "TODO.md").read_text(encoding="utf-8")
        assert "milk" not in todo
        assert "call mom" in todo

    def test_remove_matches_completed_items_too(self, jarvis_home):
        harness.init_harness()
        harness.add_todo("buy milk")
        harness.complete_todo("milk")
        assert harness.remove_todo("milk") == "Removed: buy milk"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_harness.py::TestTodos -v`
Expected: FAIL with `AttributeError: ... no attribute 'add_todo'`

- [ ] **Step 3: Write minimal implementation**

Add to `src/jarvis/harness.py`:

```python
def _todo_path() -> Path:
    return jarvis_home() / "TODO.md"


def _read_todo_lines() -> list[str]:
    if not _todo_path().exists():
        return []
    return [l for l in _todo_path().read_text(encoding="utf-8").splitlines() if l.strip()]


def _write_todo_lines(lines: list[str]) -> None:
    _atomic_write(_todo_path(), "\n".join(lines) + ("\n" if lines else ""))


def _todo_text(line: str) -> str:
    return line.removeprefix("- [ ]").removeprefix("- [x]").strip()


def add_todo(item: str) -> str:
    item = item.strip()
    if not item:
        return "Error: empty todo"
    lines = _read_todo_lines()
    lines.append(f"- [ ] {item}")
    _write_todo_lines(lines)
    return f"Added todo: {item}"


def list_todos() -> str:
    lines = _read_todo_lines()
    return "\n".join(lines) if lines else "No todos."


def _match_indexes(lines: list[str], query: str, open_only: bool) -> list[int]:
    q = query.lower().strip()
    return [
        i for i, line in enumerate(lines)
        if q in _todo_text(line).lower() and (not open_only or line.startswith("- [ ]"))
    ]


def complete_todo(item: str) -> str:
    lines = _read_todo_lines()
    matches = _match_indexes(lines, item, open_only=True)
    if not matches:
        return f"Error: no open todo matching '{item}'"
    if len(matches) > 1:
        return "Ambiguous, matches: " + "; ".join(_todo_text(lines[i]) for i in matches)
    text = _todo_text(lines[matches[0]])
    lines[matches[0]] = f"- [x] {text}"
    _write_todo_lines(lines)
    return f"Completed: {text}"


def remove_todo(item: str) -> str:
    lines = _read_todo_lines()
    matches = _match_indexes(lines, item, open_only=False)
    if not matches:
        return f"Error: no todo matching '{item}'"
    if len(matches) > 1:
        return "Ambiguous, matches: " + "; ".join(_todo_text(lines[i]) for i in matches)
    text = _todo_text(lines.pop(matches[0]))
    _write_todo_lines(lines)
    return f"Removed: {text}"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_harness.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/jarvis/harness.py tests/test_harness.py
git commit -m "feat: add TODO.md management to harness"
```

---

### Task 5: manage_todos dispatcher + optional Reminders sync

**Files:**
- Modify: `src/jarvis/harness.py`
- Modify: `tests/test_harness.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_harness.py` (add `from unittest.mock import AsyncMock, patch` to the file's imports):

```python
class TestManageTodosAndSync:
    @pytest.fixture(autouse=True)
    def reset_shortcuts(self):
        harness.set_available_shortcuts([])
        yield
        harness.set_available_shortcuts([])

    @pytest.mark.asyncio
    async def test_manage_todos_dispatch(self, jarvis_home):
        harness.init_harness()
        assert (await harness.manage_todos("add", "buy milk")) == "Added todo: buy milk"
        assert "buy milk" in (await harness.manage_todos("list"))
        assert (await harness.manage_todos("complete", "milk")) == "Completed: buy milk"
        assert (await harness.manage_todos("remove", "milk")) == "Removed: buy milk"

    @pytest.mark.asyncio
    async def test_manage_todos_unknown_action(self, jarvis_home):
        harness.init_harness()
        assert (await harness.manage_todos("explode", "x")) == "Error: unknown action 'explode'"

    @pytest.mark.asyncio
    async def test_manage_todos_missing_item(self, jarvis_home):
        harness.init_harness()
        assert (await harness.manage_todos("add", None)) == "Error: item required"

    @pytest.mark.asyncio
    async def test_sync_runs_when_shortcut_available(self, jarvis_home):
        harness.init_harness()
        harness.set_available_shortcuts(["Sync Jarvis Todos", "Other"])
        with patch("jarvis.hands.run_shortcut", new_callable=AsyncMock) as mock_run:
            await harness.manage_todos("add", "buy milk")
        mock_run.assert_called_once()
        args, kwargs = mock_run.call_args
        assert args[0] == "Sync Jarvis Todos"
        assert "- [ ] buy milk" in kwargs["input_text"]

    @pytest.mark.asyncio
    async def test_no_sync_when_shortcut_absent(self, jarvis_home):
        harness.init_harness()
        with patch("jarvis.hands.run_shortcut", new_callable=AsyncMock) as mock_run:
            await harness.manage_todos("add", "buy milk")
        mock_run.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_sync_on_list_or_error(self, jarvis_home):
        harness.init_harness()
        harness.set_available_shortcuts(["Sync Jarvis Todos"])
        with patch("jarvis.hands.run_shortcut", new_callable=AsyncMock) as mock_run:
            await harness.manage_todos("list")
            await harness.manage_todos("complete", "nothing-matches")
        mock_run.assert_not_called()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_harness.py::TestManageTodosAndSync -v`
Expected: FAIL with `AttributeError: ... no attribute 'set_available_shortcuts'`

- [ ] **Step 3: Write minimal implementation**

Add to `src/jarvis/harness.py`:

```python
SYNC_SHORTCUT = "Sync Jarvis Todos"

_available_shortcuts: set[str] = set()


def set_available_shortcuts(names: list[str]) -> None:
    global _available_shortcuts
    _available_shortcuts = set(names)


async def _maybe_sync_todos() -> None:
    if SYNC_SHORTCUT not in _available_shortcuts:
        return
    from jarvis import hands
    contents = _todo_path().read_text(encoding="utf-8")
    await hands.run_shortcut(SYNC_SHORTCUT, input_text=contents)


async def manage_todos(action: str, item: str | None = None) -> str:
    if action == "list":
        return list_todos()
    if not item:
        return "Error: item required"
    if action == "add":
        result = add_todo(item)
    elif action == "complete":
        result = complete_todo(item)
    elif action == "remove":
        result = remove_todo(item)
    else:
        return f"Error: unknown action '{action}'"
    if not result.startswith(("Error", "Ambiguous")):
        await _maybe_sync_todos()
    return result
```

Note: `hands` is imported inside `_maybe_sync_todos` to keep module import light and avoid any future circularity; patching `jarvis.hands.run_shortcut` works either way.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_harness.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/jarvis/harness.py tests/test_harness.py
git commit -m "feat: add manage_todos dispatcher with optional Reminders sync"
```

---

### Task 6: build_context() system-prompt addendum

**Files:**
- Modify: `src/jarvis/harness.py`
- Modify: `tests/test_harness.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_harness.py`:

```python
class TestBuildContext:
    def test_renders_all_sections(self, jarvis_home):
        harness.init_harness()
        harness.save_memory("music", "Prefers Spotify.")
        harness.save_skill("greet", "Be brief.")
        harness.add_todo("buy milk")
        ctx = harness.build_context()
        assert "## Your memories" in ctx
        assert "- music: Prefers Spotify." in ctx
        assert "## Your skills" in ctx
        assert "- greet: Be brief." in ctx
        assert "## User's open todos" in ctx
        assert "- [ ] buy milk" in ctx
        assert "save_memory" in ctx  # behavior rules present

    def test_empty_sections_render_none(self, jarvis_home):
        harness.init_harness()
        ctx = harness.build_context()
        assert ctx.count("(none)") == 3

    def test_completed_todos_excluded(self, jarvis_home):
        harness.init_harness()
        harness.add_todo("buy milk")
        harness.complete_todo("milk")
        ctx = harness.build_context()
        assert "buy milk" not in ctx
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_harness.py::TestBuildContext -v`
Expected: FAIL with `AttributeError: ... no attribute 'build_context'`

- [ ] **Step 3: Write minimal implementation**

Add to `src/jarvis/harness.py`:

```python
HARNESS_RULES = (
    "You have a persistent home folder of memories, skills, and the user's todos. "
    "The indexes below list what you have; use read_harness_item to read one in full when relevant. "
    "Save a memory with save_memory whenever you learn something durable about the user, "
    "and say aloud that you did so the user can veto it. "
    "When you work out a reusable way to do a task, save it with save_skill and announce it. "
    "Before a multi-step task, check whether a matching skill exists and read it first. "
    "Manage the user's todo list with manage_todos."
)


def _read_or_none(path: Path) -> str:
    text = path.read_text(encoding="utf-8").strip() if path.exists() else ""
    return text or "(none)"


def build_context() -> str:
    home = jarvis_home()
    open_todos = [l for l in _read_todo_lines() if l.startswith("- [ ]")]
    todo_block = "\n".join(open_todos) if open_todos else "(none)"
    return (
        f"{HARNESS_RULES}\n\n"
        f"## Your memories\n{_read_or_none(home / 'MEMORY.md')}\n\n"
        f"## Your skills\n{_read_or_none(home / 'SKILLS.md')}\n\n"
        f"## User's open todos\n{todo_block}"
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_harness.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/jarvis/harness.py tests/test_harness.py
git commit -m "feat: add build_context system-prompt addendum"
```

---

### Task 7: Tool schemas

**Files:**
- Modify: `src/jarvis/harness.py`
- Modify: `tests/test_harness.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_harness.py`:

```python
class TestToolSchemas:
    def test_build_harness_tool_schemas(self):
        schemas = harness.build_harness_tool_schemas()
        names = [s["function"]["name"] for s in schemas]
        assert names == ["save_memory", "save_skill", "read_harness_item", "manage_todos"]
        for s in schemas:
            assert s["type"] == "function"
            assert "description" in s["function"]
            assert s["function"]["parameters"]["type"] == "object"

    def test_manage_todos_schema_actions(self):
        schemas = {s["function"]["name"]: s for s in harness.build_harness_tool_schemas()}
        actions = schemas["manage_todos"]["function"]["parameters"]["properties"]["action"]["enum"]
        assert actions == ["add", "complete", "remove", "list"]
        kinds = schemas["read_harness_item"]["function"]["parameters"]["properties"]["kind"]["enum"]
        assert kinds == ["memory", "skill"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_harness.py::TestToolSchemas -v`
Expected: FAIL with `AttributeError: ... no attribute 'build_harness_tool_schemas'`

- [ ] **Step 3: Write minimal implementation**

Add to `src/jarvis/harness.py`:

```python
def build_harness_tool_schemas() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": "save_memory",
                "description": (
                    "Save a durable memory about the user (preference, fact, habit) to your "
                    "persistent home folder. Use a short descriptive name; saving to an existing "
                    "name updates it. Announce aloud when you save one."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Short descriptive name, e.g. 'music preference'"},
                        "content": {"type": "string", "description": "The memory in markdown, one fact per memory"},
                    },
                    "required": ["name", "content"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "save_skill",
                "description": (
                    "Save a reusable skill — a markdown recipe describing how to perform a task "
                    "(which shortcuts to chain, user preferences to respect). Announce aloud when you save one."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Short descriptive name, e.g. 'morning routine'"},
                        "content": {"type": "string", "description": "Step-by-step instructions in markdown"},
                    },
                    "required": ["name", "content"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_harness_item",
                "description": "Read the full content of one of your saved memories or skills by name.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "kind": {"type": "string", "enum": ["memory", "skill"]},
                        "name": {"type": "string", "description": "Name as listed in your index"},
                    },
                    "required": ["kind", "name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "manage_todos",
                "description": (
                    "Manage the user's todo list. Actions: add a new item, complete or remove an "
                    "existing item (matched by substring), or list all items."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string", "enum": ["add", "complete", "remove", "list"]},
                        "item": {"type": "string", "description": "Todo text (for add) or match text (for complete/remove). Omit for list."},
                    },
                    "required": ["action"],
                },
            },
        },
    ]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_harness.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/jarvis/harness.py tests/test_harness.py
git commit -m "feat: add harness tool schemas"
```

---

### Task 8: Brain dispatch + system prompt addendum

**Files:**
- Modify: `src/jarvis/brain.py` (`_execute_tool` at line 45, `think_and_act` at line 63)
- Modify: `tests/test_brain.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_brain.py`:

```python
class TestHarnessDispatch:
    @pytest.mark.asyncio
    async def test_save_memory_dispatch(self):
        from jarvis.brain import _execute_tool
        with patch("jarvis.harness.save_memory", return_value="Saved memory 'music'") as mock_save:
            result = await _execute_tool("save_memory", {"name": "music", "content": "Prefers Spotify"})
        assert result == "Saved memory 'music'"
        mock_save.assert_called_once_with("music", "Prefers Spotify")

    @pytest.mark.asyncio
    async def test_save_skill_dispatch(self):
        from jarvis.brain import _execute_tool
        with patch("jarvis.harness.save_skill", return_value="Saved skill 'greet'") as mock_save:
            result = await _execute_tool("save_skill", {"name": "greet", "content": "Be brief"})
        assert result == "Saved skill 'greet'"
        mock_save.assert_called_once_with("greet", "Be brief")

    @pytest.mark.asyncio
    async def test_read_harness_item_dispatch(self):
        from jarvis.brain import _execute_tool
        with patch("jarvis.harness.read_item", return_value="Prefers Spotify") as mock_read:
            result = await _execute_tool("read_harness_item", {"kind": "memory", "name": "music"})
        assert result == "Prefers Spotify"
        mock_read.assert_called_once_with("memory", "music")

    @pytest.mark.asyncio
    async def test_manage_todos_dispatch(self):
        from jarvis.brain import _execute_tool
        with patch("jarvis.harness.manage_todos", new_callable=AsyncMock, return_value="Added todo: x") as mock_mt:
            result = await _execute_tool("manage_todos", {"action": "add", "item": "x"})
        assert result == "Added todo: x"
        mock_mt.assert_called_once_with("add", "x")


class TestSystemExtra:
    @pytest.mark.asyncio
    async def test_system_extra_appended_to_system_prompt(self):
        settings = _make_settings()
        interrupt = asyncio.Event()

        mock_text_block = MagicMock()
        mock_text_block.type = "text"
        mock_text_block.text = "Hi."

        mock_response = MagicMock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [mock_text_block]

        with patch("jarvis.brain._get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.messages.create = MagicMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            await think_and_act(
                "hello", None, interrupt, [], [], settings,
                system_extra="## Your memories\n- music: Prefers Spotify",
            )

        system = mock_client.messages.create.call_args[1]["system"]
        assert system.endswith("## Your memories\n- music: Prefers Spotify")
        assert system.startswith("You are Jarvis")

    @pytest.mark.asyncio
    async def test_no_system_extra_keeps_prompt_unchanged(self):
        from jarvis.brain import SYSTEM_PROMPT
        settings = _make_settings()
        interrupt = asyncio.Event()

        mock_text_block = MagicMock()
        mock_text_block.type = "text"
        mock_text_block.text = "Hi."

        mock_response = MagicMock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [mock_text_block]

        with patch("jarvis.brain._get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.messages.create = MagicMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            await think_and_act("hello", None, interrupt, [], [], settings)

        assert mock_client.messages.create.call_args[1]["system"] == SYSTEM_PROMPT
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_brain.py -v`
Expected: new tests FAIL — `Unknown tool: save_memory` assertion mismatch / `TypeError: think_and_act() got an unexpected keyword argument 'system_extra'`

- [ ] **Step 3: Write minimal implementation**

In `src/jarvis/brain.py`:

1. Change the import line `from jarvis import hands` to:

```python
from jarvis import hands, harness
```

2. Add the four branches to `_execute_tool` (before the final `return f"Unknown tool: {name}"`):

```python
    elif name == "save_memory":
        return harness.save_memory(args["name"], args["content"])
    elif name == "save_skill":
        return harness.save_skill(args["name"], args["content"])
    elif name == "read_harness_item":
        return harness.read_item(args["kind"], args["name"])
    elif name == "manage_todos":
        return await harness.manage_todos(args["action"], args.get("item"))
```

3. Add the `system_extra` keyword parameter to `think_and_act` and use it. The signature becomes:

```python
async def think_and_act(
    text: str,
    image: str | None,
    interrupt: asyncio.Event,
    tools: list[dict],
    conversation: list[dict],
    settings: Settings,
    system_extra: str = "",
) -> str:
```

and inside the loop, replace `"system": SYSTEM_PROMPT,` with:

```python
            "system": SYSTEM_PROMPT + ("\n\n" + system_extra if system_extra else ""),
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_brain.py tests/test_harness.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/jarvis/brain.py tests/test_brain.py
git commit -m "feat: wire harness tools and context into brain"
```

---

### Task 9: main.py wiring

**Files:**
- Modify: `src/jarvis/main.py` (imports at top, `pipeline_iteration` at line 23, `main` at line 101)
- Modify: `tests/test_main.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_main.py` (the file already imports `asyncio`, `numpy as np`, `pytest`, `AsyncMock`/`patch`, `pipeline_iteration`, `Settings`, and defines `_make_settings` — reuse them):

```python
@pytest.mark.asyncio
async def test_pipeline_passes_system_extra_to_brain():
    settings = _make_settings()
    interrupt = asyncio.Event()
    conversation: list[dict] = []
    tools = []
    audio_buf = np.random.randn(16000).astype(np.float32)

    with patch("jarvis.main.record_until_silence", new_callable=AsyncMock, return_value=audio_buf):
        with patch("jarvis.main.transcribe", new_callable=AsyncMock, return_value="add milk to my todos"):
            with patch("jarvis.main.needs_vision", return_value=False):
                with patch("jarvis.main.think_and_act", new_callable=AsyncMock, return_value="Done.") as mock_think:
                    with patch("jarvis.main.speak", new_callable=AsyncMock):
                        await pipeline_iteration(
                            interrupt, tools, conversation, settings,
                            system_extra="## Your memories\n(none)",
                        )

    assert mock_think.call_args[1]["system_extra"] == "## Your memories\n(none)"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_main.py -v`
Expected: new test FAILS with `TypeError: pipeline_iteration() got an unexpected keyword argument 'system_extra'`; the 4 existing tests still PASS

- [ ] **Step 3: Write minimal implementation**

In `src/jarvis/main.py`:

1. Add to imports:

```python
from jarvis.harness import build_context, build_harness_tool_schemas, init_harness, set_available_shortcuts
```

2. `pipeline_iteration` gains a keyword param and forwards it. Signature becomes:

```python
async def pipeline_iteration(
    interrupt: asyncio.Event,
    tools: list[dict],
    conversation: list[dict],
    settings: Settings,
    listener=None,
    system_extra: str = "",
) -> None:
```

and the `think_and_act` call becomes:

```python
        response = await think_and_act(
            text, image, interrupt, tools, conversation, settings,
            system_extra=system_extra,
        )
```

3. In `main()`, after `shortcut_names = await discover_shortcuts()`:

```python
    init_harness()
    set_available_shortcuts(shortcut_names)
```

then extend the tools list (both branches of the existing conditional) by appending:

```python
    tools.extend(build_harness_tool_schemas())
```

(place it right after the existing `tools = [...] if shortcut_names else [...]` expression), and before the wake loop:

```python
    system_extra = build_context()
```

finally pass it in the loop:

```python
            await pipeline_iteration(interrupt, tools, conversation, settings, listener, system_extra=system_extra)
```

Also add a startup log line after `init_harness()`:

```python
    print(f"[Jarvis] Harness ready at {init_harness()}")
```

(call `init_harness()` once and reuse its return value: `home = init_harness()` then `print(f"[Jarvis] Harness ready at {home}")`.)

- [ ] **Step 4: Run the full suite**

Run: `uv run pytest -v`
Expected: all PASS (45 existing + all new)

- [ ] **Step 5: Commit**

```bash
git add src/jarvis/main.py tests/test_main.py
git commit -m "feat: wire harness into startup and pipeline"
```

---

### Task 10: Full verification

- [ ] **Step 1: Run the complete test suite**

Run: `uv run pytest -v`
Expected: all PASS

- [ ] **Step 2: Verify the package still builds**

Run: `uv build`
Expected: sdist + wheel build successfully

- [ ] **Step 3: Smoke-test init manually**

Run: `JARVIS_HOME=/tmp/jarvis-smoke uv run python -c "from jarvis.harness import init_harness, build_context; init_harness(); print(build_context())"`
Expected: prints the rules + three `(none)` sections; `/tmp/jarvis-smoke` contains the folder structure.

Clean up: `rm -rf /tmp/jarvis-smoke`

- [ ] **Step 4: Commit any remaining changes and verify clean tree**

```bash
git status
```

Expected: clean working tree (everything committed in earlier tasks).
