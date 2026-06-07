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
