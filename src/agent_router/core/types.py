"""Core value types shared by the router, deciders and adapters.

Nothing in ``agent_router.core`` may import a host SDK. Adapters translate host events
into ``RouterEvent`` and ``Decision`` back into host hook output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

NONE_ID = "none"
"""Reserved option id: no catalog entry fits, the agent's native path is enough."""

RECENT_PREFIX = "previous: "
"""Prefix of the recent-context lines ``router.build_state`` appends to the decider state."""


def current_step(state: str) -> str:
    """The decider state without its recent-context lines: the step being routed.

    Context-free deciders (the local classifiers) score this, so an earlier prompt in the
    session cannot pull the current step toward its entry. Context-capable deciders (Jev)
    get the full state.
    """
    return "\n".join(line for line in state.split("\n") if not line.startswith(RECENT_PREFIX))


class HookPoint(StrEnum):
    PROMPT = "prompt"  # the user submitted a prompt
    TOOL = "tool"  # the agent is about to call a native tool
    SKILL = "skill"  # the agent is about to invoke a skill


class Action(StrEnum):
    SUGGEST = "suggest"  # advisory hint injected; the agent decides
    ENFORCE = "enforce"  # native call denied with a templated reason
    NATIVE = "native"  # decider chose none, or confidence below threshold
    SKIPPED = "skipped"  # a gate stopped routing before the decider was asked


@dataclass(frozen=True)
class OptionSpec:
    """One option of a Jev ``choice`` question, in Jev's structured-criteria form."""

    what: str
    not_for: tuple[str, ...] = ()
    examples: tuple[str, ...] = ()


@dataclass(frozen=True)
class ChoiceResult:
    """The Jev ``choice`` answer: the most likely option and the full distribution."""

    choice: str
    probabilities: dict[str, float]
    confidence: float
    backend: str = ""
    latency_ms: float = 0.0
    stages: tuple[dict[str, Any], ...] = ()
    """Per-stage answers of a multi-stage decider (the cascade), plain JSON-able dicts."""


@dataclass(frozen=True)
class RouterEvent:
    point: HookPoint
    session_id: str
    turn_id: int
    text: str
    tool_name: str | None = None
    tool_input: dict[str, Any] | None = None
    recent: tuple[str, ...] = ()
    agent_type: str | None = None
    """The subagent making the call (host-reported), or None on the main thread."""
    tool_use_id: str | None = None
    """The host's id for the pending call: joins a decision to what the call did."""
    agent_id: str | None = None
    """The host's id for the subagent run making the call (None on the main thread)."""


@dataclass(frozen=True)
class Decision:
    action: Action
    reason: str
    entry_id: str | None = None
    hint: str | None = None
    result: ChoiceResult | None = None
    options: tuple[str, ...] = field(default_factory=tuple)
    threshold: float | None = None
    """The bar the chosen entry was held to (its own, else the router's); None when none chosen."""


PRIVATE_STAGE_KEYS = frozenset({"error"})
"""Stage keys that may carry exception / remote text: audit log only, never timeline or UI."""


def public_stages(
    stages: tuple[dict[str, Any], ...] | list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Stage records safe to show outside the audit log (no exception or remote text)."""
    return [{k: v for k, v in st.items() if k not in PRIVATE_STAGE_KEYS} for st in stages]
