"""Append-only JSONL audit log: one record per routing decision."""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agent_router.core.config import RouterConfig
from agent_router.core.types import Decision, RouterEvent

log = logging.getLogger(__name__)

MAX_TEXT = 300
Subscriber = Callable[[dict[str, Any]], None]


def _trunc(text: str | None) -> str | None:
    return None if text is None else text[:MAX_TEXT]


def _stage(stage: dict[str, Any]) -> dict[str, Any]:
    out = dict(stage)
    if out.get("error") is not None:
        out["error"] = _trunc(str(out["error"]))
    return out


class AuditLog:
    """Writes records to ``path`` (JSONL) or, when ``path`` is None, keeps them in memory.

    Subscribers are notified of every record either way; their exceptions are logged
    and swallowed so they can never break routing.
    """

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path is not None else None
        self.records: list[dict[str, Any]] = []
        self._subscribers: list[Subscriber] = []
        self._lock = threading.Lock()
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def subscribe(self, callback: Subscriber) -> Callable[[], None]:
        """Register a live listener; returns a function that unsubscribes it."""
        with self._lock:
            self._subscribers.append(callback)

        def unsubscribe() -> None:
            with self._lock:
                if callback in self._subscribers:
                    self._subscribers.remove(callback)

        return unsubscribe

    def record(
        self,
        event: RouterEvent,
        decision: Decision,
        catalog_version: str,
        config: RouterConfig,
        *,
        state: str | None = None,
    ) -> dict[str, Any]:
        res = decision.result
        hashed = state if state is not None else event.text
        rec: dict[str, Any] = {
            "ts": datetime.now(UTC).isoformat(),
            "session": event.session_id,
            "turn": event.turn_id,
            "point": str(event.point),
            "tool_name": event.tool_name,
            "agent_type": event.agent_type,
            "tool_use_id": event.tool_use_id,
            "agent_id": event.agent_id,
            "state_sha256": hashlib.sha256(hashed.encode("utf-8")).hexdigest(),
            "text": _trunc(event.text),
            "options": list(decision.options),
            "probabilities": {k: float(v) for k, v in res.probabilities.items()} if res else {},
            "choice": res.choice if res else None,
            "confidence": float(res.confidence) if res else None,
            "action": str(decision.action),
            "reason": decision.reason,
            "entry_id": decision.entry_id,
            "applied_threshold": decision.threshold,
            "hint": _trunc(decision.hint),
            "backend": res.backend if res else None,
            "latency_ms": float(res.latency_ms) if res else None,
            "stages": [_stage(st) for st in res.stages] if res else [],
            "catalog_version": catalog_version,
            "thresholds": {"threshold": config.threshold, "mode": config.mode},
        }
        line = json.dumps(rec, ensure_ascii=False, default=str)
        with self._lock:
            if self.path is None:
                self.records.append(rec)
            else:
                try:
                    with self.path.open("a", encoding="utf-8") as fh:
                        fh.write(line + "\n")
                except OSError:  # disk full, permissions, path is a directory: fail open
                    log.exception("audit write failed: %s", self.path)
            subscribers = list(self._subscribers)
        for cb in subscribers:
            try:
                cb(rec)
            except Exception:  # a listener must never break routing
                log.exception("audit subscriber failed")
        return rec


def read_audit(path: Path | str) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]
