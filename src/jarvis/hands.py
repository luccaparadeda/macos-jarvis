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
