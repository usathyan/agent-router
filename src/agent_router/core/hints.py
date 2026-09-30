"""Templated hint and deny text.

Security property: every field comes from the catalog entry (plus the host's own tool
name). No decider- or model-produced text is ever interpolated.
"""

from __future__ import annotations

import re

from agent_router.core.catalog import CatalogEntry
from agent_router.core.types import HookPoint

MAX_HINT = 400
_WS = re.compile(r"\s+")


def _clean(text: str) -> str:
    return _WS.sub(" ", text).strip()


def _cap(text: str) -> str:
    text = _clean(text)
    return text if len(text) <= MAX_HINT else text[: MAX_HINT - 1] + "…"


PLUGIN_MCP_PREFIX = "mcp__plugin_"
"""Tools a Claude Code plugin serves; the host defers them until loaded via ToolSearch."""


def _how(entry: CatalogEntry) -> str:
    target = _clean(entry.target)
    if entry.kind == "skill":
        return f'Invoke the Skill tool with skill="{target}".'
    if entry.kind == "agent":
        return f'Delegate it with the Agent tool (subagent_type="{target}").'
    if target.startswith(PLUGIN_MCP_PREFIX):  # named once: plugin tool names are long
        return f'Load it with ToolSearch "select:{target}" if deferred, then call it.'
    return f"Call tool {target}."


def render_hint(entry: CatalogEntry, point: HookPoint, prob: float) -> str:
    """Advisory hint. ``point`` and ``prob`` are accepted for future templates; unused.

    Over ``MAX_HINT``, the description is shortened first so the call instruction survives.

    An ``agent`` entry gets a scope statement without the description or the "optional"
    clause: measured live (Claude Code 2.1.283, Opus 5.5), the generic template made the main
    session answer inline, using the description as a recipe, on 3 of 3 prompts it delegated
    without any hint; this wording delegated 4 of 4
    (``integrations/first-principles/README.md``).
    """
    if entry.kind == "agent":
        target = _clean(entry.target)
        return _cap(
            f"[agent-router] Routing check: this request is in scope for the {target} agent "
            f"({_clean(entry.name)}, {_clean(entry.project)}, MIT). Delegate it with the Agent "
            f'tool (subagent_type="{target}") instead of answering inline.'
        )
    head = (
        "[agent-router] An MIT-licensed alternative may fit this step: "
        f"{_clean(entry.name)} ({_clean(entry.project)}, MIT) — "
    )
    tail = f" {_how(entry)} Optional: ignore it if your current approach is better."
    what = _clean(entry.what)
    room = MAX_HINT - len(head) - len(tail)
    if len(what) > room > 20:
        what = what[: room - 1].rstrip() + "…"
    return _cap(head + what + tail)


def render_deny(entry: CatalogEntry, tool_name: str) -> str:
    return _cap(
        f"[agent-router] Blocked {_clean(tool_name)}: {_clean(entry.name)} "
        f"({_clean(entry.project)}, MIT) fits this step. {_how(entry)}"
    )
