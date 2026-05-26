import json
import pytest
from unittest.mock import AsyncMock, patch

from jarvis.hands import discover_shortcuts, run_shortcut, build_tool_schema
from jarvis.hands import (
    open_item, search_files, system_maintenance,
    build_open_tool_schema, build_search_tool_schema, build_maintenance_tool_schema,
)


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


@pytest.mark.asyncio
async def test_open_item_app():
    mock_process = AsyncMock()
    mock_process.communicate.return_value = (b"", b"")
    mock_process.returncode = 0
    with patch("asyncio.create_subprocess_exec", return_value=mock_process) as mock_exec:
        result = await open_item("Spotify")
    assert "opened" in result.lower()
    mock_exec.assert_called_once_with("open", "-a", "Spotify", stdout=-1, stderr=-1)

@pytest.mark.asyncio
async def test_open_item_file_with_app():
    mock_process = AsyncMock()
    mock_process.communicate.return_value = (b"", b"")
    mock_process.returncode = 0
    with patch("asyncio.create_subprocess_exec", return_value=mock_process) as mock_exec:
        result = await open_item("/Users/test/doc.numbers", with_app="Numbers")
    mock_exec.assert_called_once_with("open", "-a", "Numbers", "/Users/test/doc.numbers", stdout=-1, stderr=-1)

@pytest.mark.asyncio
async def test_open_item_failure():
    mock_process = AsyncMock()
    mock_process.communicate.return_value = (b"", b"The file does not exist.")
    mock_process.returncode = 1
    with patch("asyncio.create_subprocess_exec", return_value=mock_process):
        result = await open_item("/nonexistent/file.txt")
    assert result.startswith("Error:")

@pytest.mark.asyncio
async def test_search_files():
    mock_process = AsyncMock()
    mock_process.communicate.return_value = (b"/Users/test/invoice.pdf\n/Users/test/Downloads/invoice2.pdf\n", b"")
    mock_process.returncode = 0
    with patch("asyncio.create_subprocess_exec", return_value=mock_process):
        result = await search_files("name:invoice.pdf")
    assert "/Users/test/invoice.pdf" in result
    assert "/Users/test/Downloads/invoice2.pdf" in result

@pytest.mark.asyncio
async def test_search_files_no_results():
    mock_process = AsyncMock()
    mock_process.communicate.return_value = (b"", b"")
    mock_process.returncode = 0
    with patch("asyncio.create_subprocess_exec", return_value=mock_process):
        result = await search_files("name:nonexistent.pdf")
    assert "no files found" in result.lower()

@pytest.mark.asyncio
async def test_system_maintenance_dry_run():
    mock_process = AsyncMock()
    mock_process.communicate.return_value = (b"Would remove 2.3GB of cache files", b"")
    mock_process.returncode = 0
    with patch("asyncio.create_subprocess_exec", return_value=mock_process) as mock_exec:
        result = await system_maintenance("clean", dry_run=True)
    assert "2.3GB" in result
    args = mock_exec.call_args[0]
    assert "--dry-run" in args

@pytest.mark.asyncio
async def test_system_maintenance_status():
    mock_process = AsyncMock()
    mock_process.communicate.return_value = (b"CPU: 12% | Memory: 8.2GB/16GB", b"")
    mock_process.returncode = 0
    with patch("asyncio.create_subprocess_exec", return_value=mock_process) as mock_exec:
        result = await system_maintenance("status")
    assert "CPU" in result
    args = mock_exec.call_args[0]
    assert "--dry-run" not in args

def test_build_open_tool_schema():
    schema = build_open_tool_schema()
    assert schema["function"]["name"] == "open_item"
    assert "path_or_app" in schema["function"]["parameters"]["properties"]

def test_build_search_tool_schema():
    schema = build_search_tool_schema()
    assert schema["function"]["name"] == "search_files"
    assert "query" in schema["function"]["parameters"]["properties"]

def test_build_maintenance_tool_schema():
    schema = build_maintenance_tool_schema()
    assert schema["function"]["name"] == "system_maintenance"
    props = schema["function"]["parameters"]["properties"]
    assert "action" in props
    assert "dry_run" in props
    assert props["action"]["enum"] == ["clean", "analyze", "status", "purge", "optimize"]
