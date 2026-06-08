# Security Policy

## Reporting a Vulnerability

Jarvis runs on your Mac with access to Apple Shortcuts, the filesystem
(read via Spotlight, writes confined to `~/.jarvis`), your microphone and
camera, and your Anthropic API key. Security reports are taken seriously.

**Please do not open a public issue for security vulnerabilities.**

Instead, report privately via
[GitHub Security Advisories](https://github.com/luccaparadeda/macos-jarvis/security/advisories/new)
or email luccaparadedaprofessional@gmail.com.

You can expect an acknowledgement within a few days. Please include steps
to reproduce and the impact you believe the issue has.

## Supported Versions

Only the latest release receives security fixes.

## Scope notes

- Jarvis's write access is sandboxed to `~/.jarvis` by design
  (`src/jarvis/harness.py`). Any input that escapes that directory is a
  vulnerability — reports welcome.
- Shortcut execution is limited to shortcuts the user has created, exposed
  by name to the model. Destructive `mo` maintenance actions default to
  dry-run.
