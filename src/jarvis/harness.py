import os
import re
from pathlib import Path

KINDS = {"memory": "memories", "skill": "skills"}
INDEX_FILES = {"memory": "MEMORY.md", "skill": "SKILLS.md"}


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
