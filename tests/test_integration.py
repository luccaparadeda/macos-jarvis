"""End-to-end integration tests.

These exercise the REAL component seams: the actual harness writing to an
actual (tmp) filesystem, the actual brain tool-dispatch loop, and the actual
context rebuild across "sessions". Only the unavoidable external boundary is
faked: the Anthropic HTTP API (no network in CI) — and nothing else.
"""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from jarvis import harness
from jarvis.brain import _execute_tool, think_and_act
from jarvis.config import Settings
from jarvis.harness import build_harness_tool_schemas


@pytest.fixture(autouse=True)
def jarvis_home(tmp_path, monkeypatch):
    home = tmp_path / ".jarvis"
    monkeypatch.setenv("JARVIS_HOME", str(home))
    harness.init_harness()
    return home


@pytest.fixture(autouse=True)
def reset_brain_client():
    import jarvis.brain

    jarvis.brain._client = None
    yield
    jarvis.brain._client = None


def _make_settings() -> Settings:
    return Settings(anthropic_api_key="test-key")


def _tool_use(name: str, args: dict, tool_id: str = "toolu_1") -> MagicMock:
    block = MagicMock()
    block.type = "tool_use"
    block.id = tool_id
    block.name = name
    block.input = args
    return block


def _text(content: str) -> MagicMock:
    block = MagicMock()
    block.type = "text"
    block.text = content
    return block


def _response(stop_reason: str, content: list) -> MagicMock:
    response = MagicMock()
    response.stop_reason = stop_reason
    response.content = content
    return response


def _fake_client(responses: list) -> MagicMock:
    client = MagicMock()
    client.messages.create = MagicMock(side_effect=responses)
    return client


class TestVoiceCommandToDisk:
    @pytest.mark.asyncio
    async def test_add_todo_lands_on_disk_and_result_flows_back(self, jarvis_home):
        """Model asks for manage_todos(add) → REAL harness writes TODO.md →
        the real tool result is what gets sent back to the model."""
        conversation: list[dict] = []
        responses = [
            _response("tool_use", [_tool_use("manage_todos", {"action": "add", "item": "buy milk"})]),
            _response("end_turn", [_text("Added buy milk to your list.")]),
        ]

        with patch("jarvis.brain._get_client", return_value=_fake_client(responses)):
            reply = await think_and_act(
                "add buy milk to my todos",
                None,
                asyncio.Event(),
                build_harness_tool_schemas(),
                conversation,
                _make_settings(),
            )

        assert reply == "Added buy milk to your list."
        assert "- [ ] buy milk" in (jarvis_home / "TODO.md").read_text(encoding="utf-8")
        # the tool_result that went back to the model came from the real harness
        tool_result_msg = conversation[2]
        assert tool_result_msg["role"] == "user"
        assert tool_result_msg["content"][0]["content"] == "Added todo: buy milk"

    @pytest.mark.asyncio
    async def test_memory_survives_a_session_restart(self, jarvis_home):
        """Turn 1 saves a memory; a fresh 'session' rebuilds context from disk
        and reads the memory back through the real dispatcher."""
        responses = [
            _response("tool_use", [_tool_use("save_memory", {"name": "coffee", "content": "Oat milk flat whites."})]),
            _response("end_turn", [_text("I'll remember that.")]),
        ]
        with patch("jarvis.brain._get_client", return_value=_fake_client(responses)):
            reply = await think_and_act(
                "remember I like oat milk flat whites",
                None,
                asyncio.Event(),
                build_harness_tool_schemas(),
                [],
                _make_settings(),
            )
        assert reply == "I'll remember that."
        assert (jarvis_home / "memories" / "coffee.md").read_text(encoding="utf-8") == "Oat milk flat whites."
        assert "- coffee: Oat milk flat whites." in (jarvis_home / "MEMORY.md").read_text(encoding="utf-8")

        # "restart": context is rebuilt from disk, then the memory is read back
        context = harness.build_context()
        assert "- coffee: Oat milk flat whites." in context

        conversation2: list[dict] = []
        responses2 = [
            _response("tool_use", [_tool_use("read_harness_item", {"kind": "memory", "name": "coffee"})]),
            _response("end_turn", [_text("You like oat milk flat whites.")]),
        ]
        with patch("jarvis.brain._get_client", return_value=_fake_client(responses2)):
            reply2 = await think_and_act(
                "how do I take my coffee?",
                None,
                asyncio.Event(),
                build_harness_tool_schemas(),
                conversation2,
                _make_settings(),
                system_extra=context,
            )
        assert reply2 == "You like oat milk flat whites."
        assert conversation2[2]["content"][0]["content"] == "Oat milk flat whites."


class TestTodoLifecycle:
    @pytest.mark.asyncio
    async def test_full_lifecycle_through_real_dispatch(self, jarvis_home):
        """add → list → complete → remove, entirely through the brain's
        dispatcher and the real files."""
        assert await _execute_tool("manage_todos", {"action": "add", "item": "call mom"}) == "Added todo: call mom"
        assert await _execute_tool("manage_todos", {"action": "add", "item": "buy milk"}) == "Added todo: buy milk"

        listing = await _execute_tool("manage_todos", {"action": "list"})
        assert "- [ ] call mom" in listing
        assert "- [ ] buy milk" in listing

        assert await _execute_tool("manage_todos", {"action": "complete", "item": "milk"}) == "Completed: buy milk"
        assert "- [x] buy milk" in (jarvis_home / "TODO.md").read_text(encoding="utf-8")

        assert await _execute_tool("manage_todos", {"action": "remove", "item": "call mom"}) == "Removed: call mom"
        todo = (jarvis_home / "TODO.md").read_text(encoding="utf-8")
        assert "call mom" not in todo
        assert "- [x] buy milk" in todo


class TestSandboxIntegrity:
    @pytest.mark.asyncio
    async def test_hostile_names_through_real_dispatch_stay_inside_home(self, jarvis_home, tmp_path):
        """Adversarial tool args travel the REAL dispatch path; nothing may
        land outside JARVIS_HOME."""
        await _execute_tool("save_memory", {"name": "../../../../tmp/evil", "content": "x"})
        await _execute_tool("save_skill", {"name": "/etc/cron.d/pwn", "content": "x"})
        await _execute_tool("save_memory", {"name": "..", "content": "x"})

        outside = [
            p for p in tmp_path.rglob("*") if p.is_file() and jarvis_home not in p.parents and jarvis_home != p.parent
        ]
        assert outside == [], f"files escaped the sandbox: {outside}"

    @pytest.mark.asyncio
    async def test_missing_args_become_error_results_not_crashes(self, jarvis_home):
        """A malformed model call must not raise through the pipeline (the
        dangling-tool_use session-brick bug)."""
        conversation: list[dict] = []
        responses = [
            _response("tool_use", [_tool_use("save_memory", {"name": "x"})]),  # missing content
            _response("end_turn", [_text("Sorry.")]),
        ]
        with patch("jarvis.brain._get_client", return_value=_fake_client(responses)):
            reply = await think_and_act(
                "remember x",
                None,
                asyncio.Event(),
                build_harness_tool_schemas(),
                conversation,
                _make_settings(),
            )
        assert reply == "Sorry."
        assert conversation[2]["content"][0]["content"].startswith("Error:")
