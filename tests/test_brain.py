import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from jarvis.brain import needs_vision, think_and_act, _convert_tools_for_anthropic
from jarvis.config import Settings


def _make_settings(**kwargs) -> Settings:
    defaults = {"anthropic_api_key": "test-key"}
    defaults.update(kwargs)
    return Settings(**defaults)


@pytest.fixture(autouse=True)
def reset_client():
    import jarvis.brain
    jarvis.brain._client = None
    yield
    jarvis.brain._client = None


class TestNeedsVision:
    def test_look_keyword(self):
        s = _make_settings()
        assert needs_vision("look at this document", s) is True

    def test_see_keyword(self):
        s = _make_settings()
        assert needs_vision("can you see what's on my desk", s) is True

    def test_what_is_keyword(self):
        s = _make_settings()
        assert needs_vision("what is this thing", s) is True

    def test_no_vision_keyword(self):
        s = _make_settings()
        assert needs_vision("set a timer for 5 minutes", s) is False

    def test_case_insensitive(self):
        s = _make_settings()
        assert needs_vision("LOOK at the Screen", s) is True


class TestConvertTools:
    def test_converts_openai_format_to_anthropic(self):
        tools = [{
            "type": "function",
            "function": {
                "name": "search_files",
                "description": "Search files",
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
            },
        }]
        result = _convert_tools_for_anthropic(tools)
        assert len(result) == 1
        assert result[0]["name"] == "search_files"
        assert result[0]["description"] == "Search files"
        assert "input_schema" in result[0]


class TestThinkAndAct:
    @pytest.mark.asyncio
    async def test_simple_text_response(self):
        settings = _make_settings()
        interrupt = asyncio.Event()
        conversation: list[dict] = []
        tools = [{"type": "function", "function": {"name": "run_apple_shortcut", "description": "x", "parameters": {}}}]

        mock_text_block = MagicMock()
        mock_text_block.type = "text"
        mock_text_block.text = "Hello! How can I help?"

        mock_response = MagicMock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [mock_text_block]

        with patch("jarvis.brain._get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.messages.create = MagicMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            result = await think_and_act("hello", None, interrupt, tools, conversation, settings)

        assert result == "Hello! How can I help?"

    @pytest.mark.asyncio
    async def test_tool_call_then_response(self):
        settings = _make_settings()
        interrupt = asyncio.Event()
        conversation: list[dict] = []
        tools = [{"type": "function", "function": {"name": "run_apple_shortcut", "description": "x", "parameters": {}}}]

        mock_tool_block = MagicMock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.id = "toolu_123"
        mock_tool_block.name = "run_apple_shortcut"
        mock_tool_block.input = {"shortcut_name": "What's on today?"}

        mock_resp1 = MagicMock()
        mock_resp1.stop_reason = "tool_use"
        mock_resp1.content = [mock_tool_block]

        mock_text_block = MagicMock()
        mock_text_block.type = "text"
        mock_text_block.text = "You have 3 meetings today."

        mock_resp2 = MagicMock()
        mock_resp2.stop_reason = "end_turn"
        mock_resp2.content = [mock_text_block]

        with patch("jarvis.brain._get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.messages.create = MagicMock(side_effect=[mock_resp1, mock_resp2])
            mock_get_client.return_value = mock_client

            with patch("jarvis.hands.run_shortcut", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = "Meeting 1, Meeting 2, Meeting 3"
                result = await think_and_act(
                    "what's on my calendar", None, interrupt, tools, conversation, settings,
                )

        assert result == "You have 3 meetings today."
        mock_run.assert_called_once_with("What's on today?", input_text=None)

    @pytest.mark.asyncio
    async def test_interrupt_stops_tool_loop(self):
        settings = _make_settings()
        interrupt = asyncio.Event()
        conversation: list[dict] = []
        tools = []
        interrupt.set()

        with patch("jarvis.brain._get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_get_client.return_value = mock_client

            result = await think_and_act("hello", None, interrupt, tools, conversation, settings)

        assert result == ""
        mock_client.messages.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_image_included_in_message(self):
        settings = _make_settings()
        interrupt = asyncio.Event()
        conversation: list[dict] = []
        tools = []

        mock_text_block = MagicMock()
        mock_text_block.type = "text"
        mock_text_block.text = "I see a laptop on the desk."

        mock_response = MagicMock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [mock_text_block]

        with patch("jarvis.brain._get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.messages.create = MagicMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            result = await think_and_act(
                "look at my desk", "base64imgdata", interrupt, tools, conversation, settings,
            )

        assert result == "I see a laptop on the desk."
        call_args = mock_client.messages.create.call_args
        messages = call_args[1]["messages"]
        user_msg = messages[-1]
        assert isinstance(user_msg["content"], list)
        assert user_msg["content"][1]["type"] == "image"
        assert user_msg["content"][1]["source"]["data"] == "base64imgdata"


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
