"""The host-agnostic router: gates, Jev ``choice`` decision, templated hint, audit.

Rules (in order):
1. disabled, or hook point not enabled                    -> SKIPPED
2. pending call already targets the catalog (loop guard)  -> SKIPPED
   tool/skill input matches ``config.skip_input``          -> SKIPPED
   prompt text matches ``config.skip_prompt``              -> SKIPPED
3. no catalog entry eligible at this point / tool / agent -> SKIPPED
   entries whose fit check (``fits``) says they cannot run the pending call are dropped
   here, before the decider, so a call they cannot do never spends their once-per-turn hint;
   if that leaves none                                     -> SKIPPED ("does not fit")
4. build the decider state from the event
5. ask the decider; any exception                         -> NATIVE (fail open)
6. choice outside the offered options                     -> NATIVE, recorded as ``none``
7. ``none`` or p(choice) < threshold                      -> NATIVE
   (the entry's own ``threshold`` when the catalog sets one, else the config's)
8. enforce mode at TOOL -> ENFORCE (deny) every time; marks the entry as suggested
   (never for an ``agent`` entry: denying the Agent call would drop the delegation)
9. entry already suggested this (session, turn) -> SKIPPED; else SUGGEST (hint)
Every call writes exactly one audit record.
"""

from __future__ import annotations

import json
import logging
import math
import re
import threading
from collections.abc import Iterable
from dataclasses import replace

from agent_router.core.audit import AuditLog
from agent_router.core.catalog import Catalog
from agent_router.core.config import RouterConfig
from agent_router.core.fit import FITS
from agent_router.core.hints import render_deny, render_hint
from agent_router.core.types import (
    NONE_ID,
    RECENT_PREFIX,
    Action,
    ChoiceResult,
    Decision,
    HookPoint,
    OptionSpec,
    RouterEvent,
)
from agent_router.deciders.base import Decider

log = logging.getLogger(__name__)

OWN_TOOL_PREFIX = "mcp__agent_router__"
OWN_PLUGIN_TOOL = re.compile(r"^mcp__plugin_.+_agent_router__")
"""The same tools served by a Claude Code plugin (``mcp__plugin_<plugin>_agent_router__x``)."""
NONE_OPTION = OptionSpec("the agent's own tools are enough")
MAX_TOOL_INPUT = 500
MAX_RECENT = 3


def build_state(event: RouterEvent) -> str:
    """Decider input: prompt text, pending call (compact JSON) and recent context."""
    parts = [event.text]
    if event.point in (HookPoint.TOOL, HookPoint.SKILL):
        try:
            payload = json.dumps(
                event.tool_input or {}, separators=(",", ":"), ensure_ascii=False, default=str
            )
        except (TypeError, ValueError):  # non-str keys, cycles: fall back, never fail
            payload = repr(event.tool_input)
        parts.append(f"pending {event.tool_name}: {payload[:MAX_TOOL_INPUT]}")
    recent = event.recent[-MAX_RECENT:] if event.recent else ()
    # one "previous:" line per prompt: a multi-line prompt is flattened so that
    # ``current_step`` can drop all of it
    parts.extend(f"{RECENT_PREFIX}{' '.join(str(p).split())}" for p in recent)
    return "\n".join(parts)


def _as_none(result: ChoiceResult, option_ids: tuple[str, ...]) -> ChoiceResult:
    """``result`` recorded as ``none``: only the offered options are kept (any other key's
    mass moves to ``none``), so the unknown text never reaches the audit, timeline or UI."""

    def mass(value: object) -> float:
        try:
            v = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return 0.0
        return v if math.isfinite(v) and v > 0 else 0.0

    raw = result.probabilities if isinstance(result.probabilities, dict) else {}
    probs = {oid: mass(raw.get(oid, 0.0)) for oid in option_ids}
    probs[NONE_ID] += sum(mass(v) for k, v in raw.items() if k not in probs)
    total = sum(probs.values())
    if total > 0:
        probs = {oid: v / total for oid, v in probs.items()}
    else:
        probs = {oid: (1.0 if oid == NONE_ID else 0.0) for oid in option_ids}
    stages = tuple(
        {
            **st,
            "choice": st.get("choice") if st.get("choice") in (None, *option_ids) else NONE_ID,
            "probabilities": {
                k: v for k, v in (st.get("probabilities") or {}).items() if k in option_ids
            },
        }
        for st in result.stages
    )
    return replace(result, choice=NONE_ID, probabilities=probs, stages=stages)


class Router:
    def __init__(
        self,
        catalog: Catalog,
        decider: Decider,
        config: RouterConfig | None = None,
        audit: AuditLog | None = None,
        suggested: Iterable[tuple[str, int, str]] = (),
    ) -> None:
        """``suggested`` seeds the once-per-(session, turn, entry) hint set, for hosts that
        run each hook in a fresh process (see ``adapters.claude_code``)."""
        self.catalog = catalog
        self.decider = decider
        self.config = config if config is not None else RouterConfig()
        self.audit = audit if audit is not None else AuditLog(self.config.audit_path)
        self._suggested: set[tuple[str, int, str]] = set(suggested)
        self._lock = threading.Lock()

    def suggested(self) -> set[tuple[str, int, str]]:
        """The (session, turn, entry) keys already hinted or enforced."""
        with self._lock:
            return set(self._suggested)

    def route(self, event: RouterEvent) -> Decision:
        cfg = self.config
        state: str | None = None
        # 1. enabled gates (before any state building)
        if not cfg.enabled or event.point not in cfg.points:
            decision = Decision(Action.SKIPPED, "disabled")
        else:
            state = build_state(event)
            decision = self._decide(event, state)
        try:
            self.audit.record(event, decision, self.catalog.version, cfg, state=state)
        except Exception:  # auditing must never block the host's hook
            log.exception("audit record failed")
        return decision

    def _decide(self, event: RouterEvent, state: str) -> Decision:
        cfg = self.config
        # 2. loop guard
        tool_name = event.tool_name
        tool_input = event.tool_input if isinstance(event.tool_input, dict) else {}
        skill = tool_input.get("skill")
        subagent = tool_input.get("subagent_type")
        if self.catalog.owns_target(
            tool_name,
            skill=skill if isinstance(skill, str) else None,
            subagent=subagent if isinstance(subagent, str) else None,
        ) or (
            tool_name is not None
            and (tool_name.startswith(OWN_TOOL_PREFIX) or OWN_PLUGIN_TOOL.match(tool_name))
        ):
            return Decision(Action.SKIPPED, "own tool")
        if cfg.skip_input and event.point in (HookPoint.TOOL, HookPoint.SKILL):
            try:
                if re.search(cfg.skip_input, build_state(replace(event, text="", recent=()))):
                    return Decision(Action.SKIPPED, "skip pattern")
            except re.error:  # a bad pattern disables the skip rule, never the hook
                log.warning("invalid skip_input pattern %r", cfg.skip_input)
        if cfg.skip_prompt and event.point == HookPoint.PROMPT:
            try:
                if re.search(cfg.skip_prompt, event.text or ""):
                    return Decision(Action.SKIPPED, "skip pattern")
            except re.error:
                log.warning("invalid skip_prompt pattern %r", cfg.skip_prompt)
        # 3. structural eligibility
        eligible = self.catalog.eligible(event.point, tool_name, event.agent_type)
        if not eligible:
            return Decision(Action.SKIPPED, "no eligible entries")
        unfit = [e.id for e in eligible if e.fits and not FITS[e.fits](tool_input)]
        eligible = [e for e in eligible if e.id not in unfit]
        if not eligible:
            return Decision(Action.SKIPPED, f"does not fit: {', '.join(unfit)}")
        options = {e.id: e.option() for e in eligible} | {NONE_ID: NONE_OPTION}
        option_ids = tuple(options)
        # 5. ask the decider, failing open
        try:
            result = self.decider.decide(state, options)
        except Exception as exc:  # any backend failure falls back to native
            return Decision(
                Action.NATIVE, f"decider error: {type(exc).__name__}: {exc}", options=option_ids
            )
        # 6. defensive: an unknown choice is treated as none (its text is never echoed)
        if not isinstance(result.choice, str) or result.choice not in options:
            return Decision(
                Action.NATIVE,
                "decider returned an unknown option",
                result=_as_none(result, option_ids),
                options=option_ids,
            )
        choice = result.choice
        # 7. abstain / threshold
        if choice == NONE_ID:
            return Decision(Action.NATIVE, "decider chose none", result=result, options=option_ids)
        entry = self.catalog.get(choice)
        assert entry is not None  # eligible entries come from the catalog
        threshold = entry.threshold if entry.threshold is not None else cfg.threshold
        prob = float(result.probabilities.get(choice, 0.0))
        if prob < threshold:
            return Decision(
                Action.NATIVE,
                f"p={prob:.3f} below threshold {threshold:.3f}",
                entry_id=choice,
                result=result,
                options=option_ids,
                threshold=threshold,
            )
        key = (event.session_id, event.turn_id, entry.id)
        # 8. enforce at TOOL denies every matching call; it marks but never consults the set
        if cfg.mode == "enforce" and event.point == HookPoint.TOOL and entry.kind != "agent":
            with self._lock:
                self._suggested.add(key)
            return Decision(
                Action.ENFORCE,
                f"p={prob:.3f} at or above threshold {threshold:.3f}, enforce mode",
                entry_id=entry.id,
                hint=render_deny(entry, tool_name or ""),
                result=result,
                options=option_ids,
                threshold=threshold,
            )
        # 9. advisory hints: once per (session, turn, entry)
        with self._lock:
            if key in self._suggested:
                return Decision(
                    Action.SKIPPED,
                    "already suggested this turn",
                    entry_id=entry.id,
                    result=result,
                    options=option_ids,
                    threshold=threshold,
                )
            self._suggested.add(key)
        return Decision(
            Action.SUGGEST,
            f"p={prob:.3f} at or above threshold {threshold:.3f}",
            entry_id=entry.id,
            hint=render_hint(entry, event.point, prob),
            result=result,
            options=option_ids,
            threshold=threshold,
        )
