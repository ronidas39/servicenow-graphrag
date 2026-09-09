"""Find .env.local without assuming whose laptop this is.

⛔ FOUR SCRIPTS HARDCODED THE AUTHOR'S OWN FILESYSTEM PATH. Every one of them opened
`~/Projects/AI-RESEARCH/.env.local`, so every reader who cloned the repository and
followed Part 2 got a FileNotFoundError from a path belonging to somebody else. An
article cannot ship code that only runs on the machine it was written on, and that is
not an assumption about knowledge, it is an assumption about identity.

The search order is the one a reader would expect: an explicit override first, then the
project directory the code lives in, then the directory they are standing in, then the
home directory as a last resort so the author's own setup keeps working.

Author: Roni Das
Created: 2026-09-09
"""

from __future__ import annotations

import os
import pathlib
import re

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent


def env_path() -> pathlib.Path:
    """Where .env.local actually is, or a clear error naming everywhere we looked."""
    override = os.environ.get("SERVICENOW_GRAPHRAG_ENV")
    candidates = [pathlib.Path(override)] if override else []
    candidates += [
        PROJECT_ROOT / ".env.local",
        pathlib.Path.cwd() / ".env.local",
        pathlib.Path.home() / "Projects/AI-RESEARCH/.env.local",
    ]
    for path in candidates:
        if path.is_file():
            return path
    looked = "\n    ".join(str(c) for c in candidates)
    raise SystemExit(
        f"\n  No .env.local found. Looked in:\n    {looked}\n\n"
        f"  Create it in {PROJECT_ROOT}, or set SERVICENOW_GRAPHRAG_ENV to its path.\n"
        f"  Part 1 section 21 lists what goes in it.")


def read_env() -> dict[str, str]:
    """Parse it into a plain dict. Values may be quoted; keys are UPPER_SNAKE."""
    env: dict[str, str] = {}
    for line in env_path().read_text().splitlines():
        m = re.match(r"^([A-Z0-9_]+)=(.*)$", line.strip())
        if m:
            env[m.group(1)] = m.group(2).strip().strip('"').strip("'")
    return env
