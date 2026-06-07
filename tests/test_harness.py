from pathlib import Path

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

    def test_empty_env_var_falls_back_to_default(self, jarvis_home, monkeypatch):
        monkeypatch.setenv("JARVIS_HOME", "")
        assert harness.jarvis_home() == Path.home() / ".jarvis"


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
