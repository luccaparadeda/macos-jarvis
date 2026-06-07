import os
import re
import tempfile
from pathlib import Path

KINDS = {"memory": "memories", "skill": "skills"}
INDEX_FILES = {"memory": "MEMORY.md", "skill": "SKILLS.md"}

MAX_CONTENT_CHARS = 10_000
MAX_INDEX_ENTRIES = 200


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
