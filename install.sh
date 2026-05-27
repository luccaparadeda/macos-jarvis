#!/usr/bin/env bash
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
NC='\033[0m'

info()  { echo -e "${GREEN}==>${NC} ${BOLD}$*${NC}"; }
warn()  { echo -e "${YELLOW}==>${NC} ${BOLD}$*${NC}"; }
error() { echo -e "${RED}==>${NC} ${BOLD}$*${NC}" >&2; }

# --- Pre-flight checks ---

if [[ "$(uname)" != "Darwin" ]]; then
    error "Jarvis only runs on macOS."
    exit 1
fi

ARCH="$(uname -m)"
if [[ "$ARCH" != "arm64" ]]; then
    error "Jarvis requires Apple Silicon (M1+). Detected: $ARCH"
    exit 1
fi

MACOS_VERSION="$(sw_vers -productVersion)"
MAJOR="$(echo "$MACOS_VERSION" | cut -d. -f1)"
if (( MAJOR < 13 )); then
    error "Jarvis requires macOS 13.5 or later. Detected: $MACOS_VERSION"
    exit 1
fi

info "macOS $MACOS_VERSION on Apple Silicon — looking good."

# --- Find or install Python ---

PYTHON=""
for candidate in python3.13 python3.12 python3.11 python3; do
    if command -v "$candidate" &>/dev/null; then
        version="$("$candidate" --version 2>&1 | awk '{print $2}')"
        minor="$(echo "$version" | cut -d. -f2)"
        if (( minor >= 11 )); then
            PYTHON="$candidate"
            break
        fi
    fi
done

if [[ -z "$PYTHON" ]]; then
    error "Python 3.11+ is required but not found."
    echo "  Install it with:  brew install python@3.12"
    echo "  Or from:          https://www.python.org/downloads/"
    exit 1
fi

info "Using $($PYTHON --version)"

# --- Install via uv or pip ---

if command -v uv &>/dev/null; then
    info "Installing macos-jarvis with uv..."
    uv tool install macos-jarvis
elif command -v pipx &>/dev/null; then
    info "Installing macos-jarvis with pipx..."
    pipx install macos-jarvis
else
    info "Installing macos-jarvis with pip..."
    $PYTHON -m pip install --user macos-jarvis
fi

# --- Verify installation ---

if ! command -v jarvis &>/dev/null; then
    warn "The 'jarvis' command was installed but isn't on your PATH."
    echo "  You may need to add ~/.local/bin to your PATH:"
    echo "    export PATH=\"\$HOME/.local/bin:\$PATH\""
    echo "  Then restart your terminal."
    exit 1
fi

info "Installed $(jarvis --version 2>/dev/null || echo 'macos-jarvis')."

# --- API key setup ---

ENV_FILE="${HOME}/.config/jarvis/.env"
if [[ -f "$ENV_FILE" ]] && grep -q "ANTHROPIC_API_KEY" "$ENV_FILE"; then
    info "API key already configured at $ENV_FILE"
else
    echo ""
    warn "Jarvis needs an Anthropic API key to work."
    echo "  Get one at: https://console.anthropic.com/"
    echo ""
    read -rp "  Paste your API key (or press Enter to skip): " api_key
    if [[ -n "$api_key" ]]; then
        mkdir -p "$(dirname "$ENV_FILE")"
        echo "ANTHROPIC_API_KEY=$api_key" > "$ENV_FILE"
        chmod 600 "$ENV_FILE"
        info "Saved to $ENV_FILE"
    else
        warn "Skipped. Set ANTHROPIC_API_KEY in your environment or .env file before running jarvis."
    fi
fi

# --- Done ---

echo ""
info "You're all set! Run 'jarvis' to start."
echo "  Say 'Hey Jarvis' to activate."
echo ""
echo "  Docs:   https://github.com/luccaparadeda/macos-jarvis"
echo "  Issues: https://github.com/luccaparadeda/macos-jarvis/issues"
