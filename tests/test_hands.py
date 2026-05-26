import json
import pytest
from unittest.mock import AsyncMock, patch

from jarvis.hands import discover_shortcuts, run_shortcut, build_tool_schema


@pytest.mark.asyncio
async def test_discover_shortcuts_parses_output():
    mock_process = AsyncMock()
    mock_process.communicate.return_value = (
        b"Add new event\nWhat's on today?\nTake a Break\n",
        b"",
    )
    mock_process.returncode = 0

    with patch("asyncio.create_subprocess_exec", return_value=mock_process):
        names = await discover_shortcuts()

    assert names == ["Add new event", "What's on today?", "Take a Break"]


@pytest.mark.asyncio
async def test_discover_shortcuts_skips_empty_lines():
    mock_process = AsyncMock()
    mock_process.communicate.return_value = (
        b"Shortcut A\n\nShortcut B\n\n",
        b"",
    )
    mock_process.returncode = 0

    with patch("asyncio.create_subprocess_exec", return_value=mock_process):
        names = await discover_shortcuts()

    assert names == ["Shortcut A", "Shortcut B"]


def test_build_tool_schema():
    names = ["Add new event", "Take a Break"]
    schema = build_tool_schema(names)
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "run_apple_shortcut"
    params = schema["function"]["parameters"]
    assert params["properties"]["shortcut_name"]["enum"] == names
    assert "input_text" in params["properties"]
    assert params["required"] == ["shortcut_name"]


@pytest.mark.asyncio
async def test_run_shortcut_success():
    mock_process = AsyncMock()
    mock_process.communicate.return_value = (b"Event created", b"")
    mock_process.returncode = 0

    with patch("asyncio.create_subprocess_exec", return_value=mock_process) as mock_exec:
        result = await run_shortcut("Add new event", input_text="Meeting at 3pm")

    assert result == "Event created"
    mock_exec.assert_called_once_with(
        "shortcuts", "run", "Add new event", "--input-text", "Meeting at 3pm",
        stdout=-1, stderr=-1,
    )


@pytest.mark.asyncio
async def test_run_shortcut_no_input():
    mock_process = AsyncMock()
    mock_process.communicate.return_value = (b"Done", b"")
    mock_process.returncode = 0

    with patch("asyncio.create_subprocess_exec", return_value=mock_process) as mock_exec:
        result = await run_shortcut("Take a Break")

    assert result == "Done"
    mock_exec.assert_called_once_with(
        "shortcuts", "run", "Take a Break",
        stdout=-1, stderr=-1,
    )


@pytest.mark.asyncio
async def test_run_shortcut_failure_returns_stderr():
    mock_process = AsyncMock()
    mock_process.communicate.return_value = (b"", b"Shortcut not found")
    mock_process.returncode = 1

    with patch("asyncio.create_subprocess_exec", return_value=mock_process):
        result = await run_shortcut("Nonexistent")

    assert result == "Error: Shortcut not found"
