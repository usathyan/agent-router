"""Claude Code command-hook adapter: hook JSON on stdin -> hook JSON on stdout.

``agent-router hook`` runs this once per hook call, in a fresh process, so the per-session
state the SDK adapter keeps in memory (recent prompts, the once-per-turn hint set) lives in
a small JSON file per session instead. Pure JSON in and out: no SDK import.

Input (verified on Claude Code 2.1.283): every payload has ``hook_event_name`` and
``session_id``, plus ``prompt_id`` (one per user prompt; subagent calls carry their parent
turn's). ``UserPromptSubmit`` adds ``prompt``; ``PreToolUse`` adds ``tool_name`` and
``tool_input``, and, for a call made inside a subagent, ``agent_type``.

Mapping:

- ``UserPromptSubmit`` -> ``point=prompt``, ``text`` = the prompt.
- ``PreToolUse`` -> ``point=tool`` (``skill`` for ``Skill``). On the main thread ``text`` is
  the turn's prompt; inside a subagent it is empty: the subagent works on its own task, not
  the user's words, so only the pending call is routed.
- ``turn_id`` = CRC32 of ``prompt_id`` (the prompt count when a host omits it).
- SUGGEST -> ``additionalContext``; ENFORCE -> ``permissionDecision: deny``; else ``{}``.

State and audit default to ``$AGENT_ROUTER_STATE_DIR``, else
``$XDG_STATE_HOME/agent-router``, else ``~/.local/state/agent-router``:
``sessions/<session>.json`` and ``audit/<session>.jsonl`` (``AGENT_ROUTER_AUDIT`` wins).
Every failure fails open: the caller prints ``{}``.
"""

from __future__ import annotations

import json
import logging
import os
import re
import zlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

from agent_router.core.router import MAX_RECENT, Router
from agent_router.core.types import Action, Decision, HookPoint, RouterEvent

log = logging.getLogger(__name__)

POINTS = {"UserPromptSubmit": HookPoint.PROMPT, "PreToolUse": HookPoint.TOOL}
SKILL_TOOL = "Skill"
MAX_KEPT_PROMPTS = MAX_RECENT + 1
_UNSAFE = re.compile(r"[^A-Za-z0-9_.-]")

RouterFactory = Callable[[Path | None, set[tuple[str, int, str]]], Router]
"""Builds the router for one hook call from (audit path or None, suggested-key seed)."""


def state_root() -> Path:
    env = os.environ
    if env.get("AGENT_ROUTER_STATE_DIR", "").strip():
        return Path(env["AGENT_ROUTER_STATE_DIR"].strip())
    base = env.get("XDG_STATE_HOME", "").strip() or str(Path.home() / ".local" / "state")
    return Path(base) / "agent-router"


def _safe_name(session: str) -> str:
    return _UNSAFE.sub("_", session)[:128] or "unknown"


def load_state(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except (OSError, ValueError):
        pass
    return {}


def save_state(path: Path, data: dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(data), encoding="utf-8")
        tmp.replace(path)
    except OSError:
        log.exception("cannot save hook state %s", path)


def _str_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def turn_id(payload: dict[str, Any], prompt_count: int) -> int:
    prompt_id = payload.get("prompt_id")
    if isinstance(prompt_id, str) and prompt_id:
        return zlib.crc32(prompt_id.encode("utf-8"))
    return prompt_count


def to_event(payload: dict[str, Any], state: dict[str, Any]) -> RouterEvent | None:
    """The router event for one hook payload (updates ``state``), or None to ignore it."""
    point = POINTS.get(str(payload.get("hook_event_name") or ""))
    if point is None:
        return None
    session = str(payload.get("session_id") or "")
    prompts = [p for p in state.get("prompts", []) if isinstance(p, str)]
    agent_type = payload.get("agent_type")
    agent_type = agent_type if isinstance(agent_type, str) and agent_type else None
    if point == HookPoint.PROMPT:
        prompt = str(payload.get("prompt") or "")
        event = RouterEvent(
            point=point,
            session_id=session,
            turn_id=turn_id(payload, len(prompts) + 1),
            text=prompt,
            recent=tuple(prompts[-MAX_RECENT:]),
        )
        state["prompts"] = [*prompts, prompt][-MAX_KEPT_PROMPTS:]
        return event
    tool_name = payload.get("tool_name")
    if not isinstance(tool_name, str) or not tool_name:
        return None
    tool_input = payload.get("tool_input")
    main = agent_type is None
    return RouterEvent(
        point=HookPoint.SKILL if tool_name == SKILL_TOOL else HookPoint.TOOL,
        session_id=session,
        turn_id=turn_id(payload, len(prompts)),
        text=(prompts[-1] if prompts else "") if main else "",
        tool_name=tool_name,
        tool_input=tool_input if isinstance(tool_input, dict) else {},
        recent=tuple(prompts[:-1][-MAX_RECENT:]) if main else (),
        agent_type=agent_type,
        tool_use_id=_str_or_none(payload.get("tool_use_id")),
        agent_id=_str_or_none(payload.get("agent_id")),
    )


def to_output(decision: Decision, hook_event: str) -> dict[str, Any]:
    if not decision.hint:
        return {}
    if decision.action == Action.ENFORCE and hook_event == "PreToolUse":
        spec = {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": decision.hint,
        }
        return {"hookSpecificOutput": spec}
    if decision.action == Action.SUGGEST:
        return {
            "hookSpecificOutput": {"hookEventName": hook_event, "additionalContext": decision.hint}
        }
    return {}


def handle(payload: dict[str, Any], make_router: RouterFactory, root: Path | None = None) -> dict:
    """Route one hook payload; returns the hook output (``{}`` when nothing applies)."""
    root = root if root is not None else state_root()
    session = _safe_name(str(payload.get("session_id") or ""))
    state_path = root / "sessions" / f"{session}.json"
    state = load_state(state_path)
    event = to_event(payload, state)
    if event is None:
        return {}
    seed = {
        (event.session_id, int(t), str(e))
        for t, e in state.get("suggested", [])
        if isinstance(t, int) and isinstance(e, str)
    }
    audit_env = os.environ.get("AGENT_ROUTER_AUDIT", "").strip()
    audit = Path(audit_env) if audit_env else root / "audit" / f"{session}.jsonl"
    if not audit_env:
        audit.parent.mkdir(parents=True, exist_ok=True)
    router = make_router(audit, seed)
    decision = router.route(event)
    # hints are once per turn, so only the current turn's keys are worth keeping
    state["suggested"] = sorted(
        [t, e] for s, t, e in router.suggested() if s == event.session_id and t == event.turn_id
    )
    save_state(state_path, state)
    return to_output(decision, str(payload.get("hook_event_name")))
