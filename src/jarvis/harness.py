import os
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
