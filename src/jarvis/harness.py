import os
import re
import tempfile
from pathlib import Path

KINDS = {"memory": "memories", "skill": "skills"}
INDEX_FILES = {"memory": "MEMORY.md", "skill": "SKILLS.md"}

MAX_CONTENT_CHARS = 10_000
MAX_INDEX_ENTRIES = 200

SYNC_SHORTCUT = "Sync Jarvis Todos"

_available_shortcuts: set[str] = set()


def jarvis_home() -> Path:
    return Path(os.environ.get("JARVIS_HOME") or (Path.home() / ".jarvis"))


def init_harness() -> Path:
    home = jarvis_home()
    for sub in KINDS.values():
        (home / sub).mkdir(parents=True, exist_ok=True)
    for name in (*INDEX_FILES.values(), "TODO.md"):
        path = home / name
        if not path.exists():
            path.write_text("", encoding="utf-8")
    return home


def _slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def _resolve(kind: str, name: str) -> Path | None:
    if kind not in KINDS:
        return None
    slug = _slugify(name)
    if not slug or len(slug) + len(".md") > 255:
        return None
    folder = (jarvis_home() / KINDS[kind]).resolve()
    path = (folder / f"{slug}.md").resolve()
    if path.parent != folder:
        return None
    return path


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


def _todo_path() -> Path:
    return jarvis_home() / "TODO.md"


def _read_todo_lines() -> list[str]:
    if not _todo_path().exists():
        return []
    return [line for line in _todo_path().read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_todo_lines(lines: list[str]) -> None:
    _atomic_write(_todo_path(), "\n".join(lines) + ("\n" if lines else ""))


def _todo_text(line: str) -> str:
    return re.sub(r"^- \[[xX ]\] ?", "", line).strip()


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


def set_available_shortcuts(names: list[str]) -> None:
    global _available_shortcuts
    _available_shortcuts = set(names)


async def _maybe_sync_todos() -> None:
    if SYNC_SHORTCUT not in _available_shortcuts:
        return
    from jarvis import hands
    try:
        contents = _todo_path().read_text(encoding="utf-8")
        await hands.run_shortcut(SYNC_SHORTCUT, input_text=contents)
    except Exception as e:
        print(f"  [Harness] Todo sync failed: {e}")


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
    open_todos = [line for line in _read_todo_lines() if line.startswith("- [ ]")]
    todo_block = "\n".join(open_todos) if open_todos else "(none)"
    return (
        f"{HARNESS_RULES}\n\n"
        f"## Your memories\n{_read_or_none(home / 'MEMORY.md')}\n\n"
        f"## Your skills\n{_read_or_none(home / 'SKILLS.md')}\n\n"
        f"## User's open todos\n{todo_block}"
    )


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
