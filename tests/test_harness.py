from pathlib import Path
from unittest.mock import AsyncMock, patch

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
        assert path == (jarvis_home / "memories" / "coffee-preference.md").resolve()

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

    def test_unknown_kind_rejected(self, jarvis_home):
        harness.init_harness()
        assert harness._resolve("config", "x") is None

    def test_overlong_name_rejected(self, jarvis_home):
        harness.init_harness()
        assert harness._resolve("memory", "a" * 300) is None


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

    def test_remove_handles_hand_edited_uppercase_x(self, jarvis_home):
        harness.init_harness()
        (jarvis_home / "TODO.md").write_text("- [X] buy milk\n", encoding="utf-8")
        assert harness.remove_todo("milk") == "Removed: buy milk"


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
