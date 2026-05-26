import asyncio
import subprocess


async def discover_shortcuts() -> list[str]:
    proc = await asyncio.create_subprocess_exec(
        "shortcuts", "list",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout, _ = await proc.communicate()
    lines = stdout.decode().strip().splitlines()
    return [line.strip() for line in lines if line.strip()]


def build_tool_schema(shortcut_names: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": "run_apple_shortcut",
            "description": (
                "Execute a macOS Apple Shortcut by name. "
                "Use this to control apps, send messages, manage calendar, "
                "take notes, and automate the Mac."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "shortcut_name": {
                        "type": "string",
                        "enum": shortcut_names,
                    },
                    "input_text": {
                        "type": "string",
                        "description": "Optional text input to pass to the shortcut",
                    },
                },
                "required": ["shortcut_name"],
            },
        },
    }


async def run_shortcut(name: str, input_text: str | None = None) -> str:
    cmd = ["shortcuts", "run", name]
    if input_text is not None:
        cmd.extend(["--input-text", input_text])

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()

    if proc.returncode != 0:
        return f"Error: {stderr.decode().strip()}"
    return stdout.decode().strip()


DESTRUCTIVE_ACTIONS = {"clean", "purge", "optimize"}


def build_open_tool_schema() -> dict:
    return {
        "type": "function",
        "function": {
            "name": "open_item",
            "description": "Open a file, folder, or application on macOS. Like double-clicking in Finder.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path_or_app": {"type": "string", "description": "File path, folder path, or application name to open"},
                    "with_app": {"type": "string", "description": "Optional: open the file with a specific application"},
                },
                "required": ["path_or_app"],
            },
        },
    }


def build_search_tool_schema() -> dict:
    return {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Search for files on macOS using Spotlight (mdfind). Searches file names and contents instantly.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query — file name, content keywords, or metadata filter (e.g. 'name:invoice.pdf')"},
                },
                "required": ["query"],
            },
        },
    }


def build_maintenance_tool_schema() -> dict:
    return {
        "type": "function",
        "function": {
            "name": "system_maintenance",
            "description": "Run Mac system maintenance using Mole (mo). Clean caches, analyze disk usage, check system status, or purge build artifacts. Destructive commands default to dry-run mode for safety.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["clean", "analyze", "status", "purge", "optimize"]},
                    "dry_run": {"type": "boolean", "description": "If true (default), show what would be done without doing it."},
                },
                "required": ["action"],
            },
        },
    }


async def open_item(path_or_app: str, with_app: str | None = None) -> str:
    if with_app:
        cmd = ["open", "-a", with_app, path_or_app]
    else:
        cmd = ["open", "-a", path_or_app]
    proc = await asyncio.create_subprocess_exec(*cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        return f"Error: {stderr.decode().strip()}"
    return f"Opened {path_or_app}"


async def search_files(query: str) -> str:
    proc = await asyncio.create_subprocess_exec("mdfind", query, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = await proc.communicate()
    output = stdout.decode().strip()
    if not output:
        return "No files found."
    lines = output.splitlines()[:20]
    return "\n".join(lines)


async def system_maintenance(action: str, dry_run: bool = True) -> str:
    cmd = ["mo", action]
    if action in DESTRUCTIVE_ACTIONS and dry_run:
        cmd.append("--dry-run")
    proc = await asyncio.create_subprocess_exec(*cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        return f"Error: {stderr.decode().strip()}"
    return stdout.decode().strip()
