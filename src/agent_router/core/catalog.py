"""The fixed catalog of MIT-licensed alternatives the router may suggest."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

from agent_router.core.fit import FITS
from agent_router.core.types import NONE_ID, HookPoint, OptionSpec

ALLOWED_LICENSE = "MIT"
DEFAULT_CATALOG = Path(__file__).resolve().parent.parent / "catalog.yaml"
KINDS = ("tool", "skill", "agent")
MAIN_AGENT = "main"
"""``agents:`` value for the host's main thread (a call that carries no subagent type)."""


class CatalogError(ValueError):
    pass


@dataclass(frozen=True)
class CatalogEntry:
    id: str
    kind: Literal["tool", "skill", "agent"]
    name: str
    project: str
    license: str
    url: str
    what: str
    target: str  # MCP tool name (mcp__agent_router__x), skill name, or subagent type
    points: tuple[HookPoint, ...]
    replaces: tuple[str, ...] = ()  # native tool names this can stand in for at TOOL/SKILL
    not_for: tuple[str, ...] = ()
    examples: tuple[str, ...] = ()
    agents: tuple[str, ...] = ()  # agent contexts (``main``, subagent types); () = all
    threshold: float | None = None  # own acceptance bar; None = the router's threshold
    fits: str | None = None  # named check (core.fit.FITS) that the entry can run the call

    def applies_to(self, agent_type: str | None) -> bool:
        return not self.agents or (agent_type or MAIN_AGENT) in self.agents

    def option(self) -> OptionSpec:
        return OptionSpec(what=self.what, not_for=self.not_for, examples=self.examples)


@dataclass(frozen=True)
class Catalog:
    version: str
    entries: tuple[CatalogEntry, ...]
    native_examples: tuple[str, ...] = ()  # exemplars where the agent's own tools suffice

    def get(self, entry_id: str) -> CatalogEntry | None:
        return next((e for e in self.entries if e.id == entry_id), None)

    def eligible(
        self, point: HookPoint, tool_name: str | None = None, agent_type: str | None = None
    ) -> list[CatalogEntry]:
        """Entries that may answer at this hook point (structural gate, before the decider)."""
        out = []
        for e in self.entries:
            if point not in e.points or not e.applies_to(agent_type):
                continue
            if point in (HookPoint.TOOL, HookPoint.SKILL) and tool_name not in e.replaces:
                continue
            out.append(e)
        return out

    def owns_target(
        self, tool_name: str | None, skill: str | None = None, subagent: str | None = None
    ) -> bool:
        """True when the pending call already targets a catalog entry (loop guard)."""
        targets = {e.target for e in self.entries}
        return (
            (tool_name in targets)
            or (skill is not None and skill in targets)
            or (subagent is not None and subagent in targets)
        )


_REQUIRED = ("id", "kind", "name", "project", "license", "url", "what", "target", "points")


def load_catalog(path: str | Path = DEFAULT_CATALOG) -> Catalog:
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict) or "entries" not in data:
        raise CatalogError("catalog must be a mapping with an 'entries' list")
    entries: list[CatalogEntry] = []
    seen: set[str] = set()
    for raw in data["entries"]:
        missing = [k for k in _REQUIRED if not raw.get(k)]
        if missing:
            raise CatalogError(f"entry {raw.get('id', '?')}: missing {missing}")
        if raw["id"] == NONE_ID:
            raise CatalogError(f"'{NONE_ID}' is a reserved option id")
        if raw["id"] in seen:
            raise CatalogError(f"duplicate id {raw['id']}")
        if raw["license"] != ALLOWED_LICENSE:
            raise CatalogError(
                f"entry {raw['id']}: license {raw['license']!r} rejected; only MIT is allowed"
            )
        if raw["kind"] not in KINDS:
            raise CatalogError(f"entry {raw['id']}: kind must be one of {', '.join(KINDS)}")
        threshold = raw.get("threshold")
        if threshold is not None and not (
            isinstance(threshold, int | float) and 0.0 <= threshold <= 1.0
        ):
            raise CatalogError(f"entry {raw['id']}: threshold must be a number in [0, 1]")
        fits = raw.get("fits")
        if fits is not None and fits not in FITS:
            raise CatalogError(
                f"entry {raw['id']}: unknown fit check {fits!r}; known: {', '.join(sorted(FITS))}"
            )
        seen.add(raw["id"])
        entries.append(
            CatalogEntry(
                id=raw["id"],
                kind=raw["kind"],
                name=raw["name"],
                project=raw["project"],
                license=raw["license"],
                url=raw["url"],
                what=raw["what"].strip(),
                target=raw["target"],
                points=tuple(HookPoint(p) for p in raw["points"]),
                replaces=tuple(raw.get("replaces", ())),
                not_for=tuple(raw.get("not_for", ())),
                examples=tuple(raw.get("examples", ())),
                agents=tuple(raw.get("agents", ())),
                threshold=float(threshold) if threshold is not None else None,
                fits=fits,
            )
        )
    return Catalog(
        version=str(data.get("version", "0")),
        entries=tuple(entries),
        native_examples=tuple(data.get("native_examples", ())),
    )
