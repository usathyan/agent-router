"""Router configuration (defaults, environment overrides)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from agent_router.core.types import HookPoint

Mode = Literal["advisory", "enforce"]
MODES: tuple[Mode, ...] = ("advisory", "enforce")
_TRUTHY = frozenset({"1", "true", "yes", "on"})


@dataclass(frozen=True)
class RouterConfig:
    mode: Mode = "advisory"
    threshold: float = 0.5
    enabled: bool = True
    audit_path: Path | None = None
    points: tuple[HookPoint, ...] = tuple(HookPoint)
    skip_input: str | None = None
    """Regex over the pending tool/skill call (``pending <tool>: <json input>``); a match
    is SKIPPED before the decider. For calls a host agent must always make natively."""
    skip_prompt: str | None = None
    """Regex over the prompt text at the PROMPT point; a match is SKIPPED before the decider.
    For prompts that already name their route, e.g. a slash command launching the target."""

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {self.threshold}")

    @classmethod
    def from_env(cls, recommended_threshold: float | None = None) -> RouterConfig:
        """Read ``AGENT_ROUTER_MODE``, ``_THRESHOLD``, ``_DISABLED``, ``_AUDIT`` and
        ``_SKIP_INPUT`` and ``_SKIP_PROMPT``.

        The threshold is ``AGENT_ROUTER_THRESHOLD`` when set, else ``recommended_threshold``
        (the decider's calibrated one) when given, else the default.
        """
        env = os.environ
        audit = env.get("AGENT_ROUTER_AUDIT", "").strip()
        raw_thr = env.get("AGENT_ROUTER_THRESHOLD", "").strip()
        if raw_thr:
            threshold = float(raw_thr)
        elif recommended_threshold is not None:
            threshold = float(recommended_threshold)
        else:
            threshold = cls.threshold
        return cls(
            mode=env.get("AGENT_ROUTER_MODE", "advisory").strip().lower(),  # type: ignore[arg-type]
            threshold=threshold,
            enabled=env.get("AGENT_ROUTER_DISABLED", "").strip().lower() not in _TRUTHY,
            audit_path=Path(audit) if audit else None,
            skip_input=env.get("AGENT_ROUTER_SKIP_INPUT", "").strip() or None,
            skip_prompt=env.get("AGENT_ROUTER_SKIP_PROMPT", "").strip() or None,
        )
