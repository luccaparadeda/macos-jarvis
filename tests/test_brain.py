import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from jarvis.brain import needs_vision, think_and_act
from jarvis.config import Settings


def _make_settings(**kwargs) -> Settings:
    defaults = {"deepseek_api_key": "test-key"}
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


class TestThinkAndAct:
    @pytest.mark.asyncio
    async def test_simple_text_response(self):
        settings = _make_settings()
        interrupt = asyncio.Event()
        conversation: list[dict] = []
        tools = [{"type": "function", "function": {"name": "run_apple_shortcut"}}]

        mock_message = MagicMock()
        mock_message.content = "Hello! How can I help?"
        mock_message.tool_calls = None
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        with patch("jarvis.brain._get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.chat.completions.create = MagicMock(return_value=mock_response)
            mock_get_client.return_value = mock_client
            result = await think_and_act("hello", None, interrupt, tools, conversation, settings)

        assert result == "Hello! How can I help?"
        assert len(conversation) == 3  # system + user + assistant

    @pytest.mark.asyncio
    async def test_tool_call_then_response(self):
        settings = _make_settings()
        interrupt = asyncio.Event()
        conversation: list[dict] = []
        tools = [{"type": "function", "function": {"name": "run_apple_shortcut"}}]

        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_123"
        mock_tool_call.function.name = "run_apple_shortcut"
        mock_tool_call.function.arguments = json.dumps({"shortcut_name": "What's on today?"})

        mock_msg1 = MagicMock()
        mock_msg1.content = None
        mock_msg1.tool_calls = [mock_tool_call]
        mock_choice1 = MagicMock()
        mock_choice1.message = mock_msg1
        mock_resp1 = MagicMock()
        mock_resp1.choices = [mock_choice1]

        mock_msg2 = MagicMock()
        mock_msg2.content = "You have 3 meetings today."
        mock_msg2.tool_calls = None
        mock_choice2 = MagicMock()
        mock_choice2.message = mock_msg2
        mock_resp2 = MagicMock()
        mock_resp2.choices = [mock_choice2]

        with patch("jarvis.brain._get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.chat.completions.create = MagicMock(side_effect=[mock_resp1, mock_resp2])
            mock_get_client.return_value = mock_client
            with patch("jarvis.hands.run_shortcut", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = "Meeting 1, Meeting 2, Meeting 3"
                result = await think_and_act("what's on my calendar", None, interrupt, tools, conversation, settings)

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
        mock_client.chat.completions.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_image_included_in_message(self):
        settings = _make_settings()
        interrupt = asyncio.Event()
        conversation: list[dict] = []
        tools = []

        mock_message = MagicMock()
        mock_message.content = "I see a laptop on the desk."
        mock_message.tool_calls = None
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        with patch("jarvis.brain._get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.chat.completions.create = MagicMock(return_value=mock_response)
            mock_get_client.return_value = mock_client
            result = await think_and_act("look at my desk", "base64imgdata", interrupt, tools, conversation, settings)

        assert result == "I see a laptop on the desk."
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args[1]["messages"]
        user_msg = messages[-1]
        assert isinstance(user_msg["content"], list)
        assert user_msg["content"][1]["type"] == "image_url"
