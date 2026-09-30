"""Claude Agent SDK adapter: UserPromptSubmit / PreToolUse hooks that call the router.

``ClaudeRouterHooks(router).hooks()`` plugs into ``ClaudeAgentOptions.hooks``. Each hook
translates the SDK input into a ``RouterEvent``, routes it (in a worker thread, so network
deciders never block the event loop) and translates the ``Decision`` back:

- SUGGEST -> ``hookSpecificOutput.additionalContext = hint`` (advisory; the model decides)
- ENFORCE -> ``permissionDecision = "deny"``, ``permissionDecisionReason = hint``
- anything else -> ``{}``

Only ``Decision.hint`` (catalog-templated text) ever reaches the agent; ``Decision.reason``
may contain exception text and is kept for the audit log only. Hooks never raise into the
SDK: any failure is logged and the hook returns ``{}`` (fail open).

Timeline event schema
---------------------
Listeners registered with ``on_decision`` (and the ``on_event`` callback of
``agent_router.agent.run_agent``) receive plain JSON-serialisable dicts. Every dict has a
``type`` from ``EVENT_TYPES`` and a ``ts`` (float, ``time.time()``):

``prompt``       ``{"prompt": str}`` — emitted by ``run_agent`` before the session starts.
``decision``     one per routed hook call::

                     {"point": "prompt"|"tool"|"skill", "tool_name": str|None,
                      "session_id": str, "turn_id": int, "action":
                      "suggest"|"enforce"|"native"|"skipped", "entry_id": str|None,
                      "applied_threshold": float|None, "hint": str|None, "choice": str|None,
                      "probabilities": {option_id: float}, "confidence": float|None,
                      "backend": str|None, "latency_ms": float|None,
                      "stages": [stage, ...]}

                 (``reason`` is deliberately absent.) ``stages`` is ``[]`` except for the
                 cascade decider: one dict per stage asked, in order::

                     {"role": "primary"|"confirm", "backend": str, "choice": str|None,
                      "probabilities": {option_id: float}, "confidence": float|None,
                      "latency_ms": float, "failed": bool, "error_type": str|None,
                      "skipped": "circuit-open"|None}

                 A failed stage carries only its exception class name; the exception text
                 is kept (truncated) in the audit record only.
``hook``         what the hook returned to the SDK, right after its ``decision``::

                     {"point", "tool_name", "session_id", "turn_id",
                      "output": "additionalContext"|"deny"|"none"}

``assistant``    ``{"text": str}`` — one per assistant ``TextBlock``.
``tool_use``     ``{"id": str, "name": str, "input": dict}`` — assistant ``ToolUseBlock``.
``tool_result``  ``{"tool_use_id": str, "content": str, "is_error": bool}`` — content is
                 flattened to text and truncated to ``MAX_EVENT_TEXT`` characters.
``result``       ``{"result": str|None, "is_error": bool, "num_turns": int,
                 "total_cost_usd": float|None, "duration_ms": int}`` — the final message.
``error``        ``{"message": str}`` — the run failed (``run_agent`` re-raises after it).
"""

from __future__ import annotations

import logging
import time
from collections import deque
from collections.abc import Callable
from typing import Any

import anyio
from claude_agent_sdk import HookMatcher

from agent_router.core.router import MAX_RECENT, Router
from agent_router.core.types import Action, Decision, HookPoint, RouterEvent, public_stages

log = logging.getLogger(__name__)

EVENT_TYPES = frozenset(
    {"prompt", "hook", "decision", "assistant", "tool_use", "tool_result", "result", "error"}
)
MAX_EVENT_TEXT = 2000
SKILL_TOOL = "Skill"
Listener = Callable[[dict[str, Any]], None]


def emit(listeners: list[Listener], type_: str, **payload: Any) -> None:
    """Deliver one timeline event; listener exceptions are logged and swallowed."""
    event = {"type": type_, "ts": time.time(), **payload}
    for cb in list(listeners):
        try:
            cb(event)
        except Exception:
            log.exception("timeline listener failed")


def decision_payload(event: RouterEvent, decision: Decision) -> dict[str, Any]:
    """The ``decision`` event body. Never includes ``Decision.reason``."""
    res = decision.result
    return {
        "point": str(event.point),
        "tool_name": event.tool_name,
        "session_id": event.session_id,
        "turn_id": event.turn_id,
        "action": str(decision.action),
        "entry_id": decision.entry_id,
        "applied_threshold": decision.threshold,
        "hint": decision.hint,
        "choice": res.choice if res else None,
        "probabilities": {k: float(v) for k, v in res.probabilities.items()} if res else {},
        "confidence": float(res.confidence) if res else None,
        "backend": res.backend if res else None,
        "latency_ms": float(res.latency_ms) if res else None,
        "stages": public_stages(res.stages) if res else [],
    }


class ClaudeRouterHooks:
    """Per-session turn tracking plus the two SDK hook callbacks."""

    def __init__(self, router: Router) -> None:
        self.router = router
        self._turns: dict[str, int] = {}
        self._last_prompt: dict[str, str] = {}
        self._recent: dict[str, deque[str]] = {}
        self._listeners: list[Listener] = []

    def on_decision(self, callback: Listener) -> None:
        """Receive ``decision`` and ``hook`` timeline events (see module docstring)."""
        self._listeners.append(callback)

    def hooks(self) -> dict[str, list[HookMatcher]]:
        return {
            "UserPromptSubmit": [HookMatcher(hooks=[self.on_user_prompt])],
            "PreToolUse": [HookMatcher(matcher=None, hooks=[self.on_pre_tool_use])],
        }

    # -- SDK callbacks -------------------------------------------------------

    async def on_user_prompt(
        self, input_data: dict[str, Any], tool_use_id: str | None, context: Any
    ) -> dict[str, Any]:
        try:
            session = str(input_data.get("session_id") or "")
            prompt = str(input_data.get("prompt") or "")
            turn = self._turns.get(session, 0) + 1
            self._turns[session] = turn
            recent = self._recent.setdefault(session, deque(maxlen=MAX_RECENT))
            event = RouterEvent(
                point=HookPoint.PROMPT,
                session_id=session,
                turn_id=turn,
                text=prompt,
                recent=tuple(recent),
            )
            recent.append(prompt)
            self._last_prompt[session] = prompt
            return await self._route(event, "UserPromptSubmit")
        except Exception:
            log.exception("agent-router UserPromptSubmit hook failed; failing open")
            return {}

    async def on_pre_tool_use(
        self, input_data: dict[str, Any], tool_use_id: str | None, context: Any
    ) -> dict[str, Any]:
        try:
            session = str(input_data.get("session_id") or "")
            tool_name = input_data.get("tool_name")
            if not isinstance(tool_name, str) or not tool_name:
                return {}
            tool_in = input_data.get("tool_input")
            tool_in = tool_in if isinstance(tool_in, dict) else {}
            point = HookPoint.SKILL if tool_name == SKILL_TOOL else HookPoint.TOOL
            prior = self._recent.get(session, ())
            event = RouterEvent(
                point=point,
                session_id=session,
                turn_id=self._turns.get(session, 0),
                text=self._last_prompt.get(session, ""),
                tool_name=tool_name,
                tool_input=tool_in,
                recent=tuple(prior)[:-1],  # the last prompt is already ``text``
                tool_use_id=tool_use_id,
            )
            return await self._route(event, "PreToolUse")
        except Exception:
            log.exception("agent-router PreToolUse hook failed; failing open")
            return {}

    # -- internals -----------------------------------------------------------

    async def _route(self, event: RouterEvent, hook_event: str) -> dict[str, Any]:
        decision = await anyio.to_thread.run_sync(self.router.route, event)
        emit(self._listeners, "decision", **decision_payload(event, decision))
        output, kind = self._output(decision, hook_event)
        emit(
            self._listeners,
            "hook",
            point=str(event.point),
            tool_name=event.tool_name,
            session_id=event.session_id,
            turn_id=event.turn_id,
            output=kind,
        )
        return output

    @staticmethod
    def _output(decision: Decision, hook_event: str) -> tuple[dict[str, Any], str]:
        if not decision.hint:
            return {}, "none"
        if decision.action == Action.ENFORCE and hook_event == "PreToolUse":
            spec = {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": decision.hint,
            }
            return {"hookSpecificOutput": spec}, "deny"
        if decision.action == Action.SUGGEST:
            spec = {"hookEventName": hook_event, "additionalContext": decision.hint}
            return {"hookSpecificOutput": spec}, "additionalContext"
        return {}, "none"
