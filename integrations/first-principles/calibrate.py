#!/usr/bin/env python3
"""Calibrate the first-principles integration on its ``cal`` split (never the holdout).

1. Shared local-classifier parameters on every ``cal`` case (what ``agent-router calibrate``
   does), written to ``calibration.json``.
2. One threshold per catalog entry, fit with those parameters fixed on the ``cal`` cases of
   the entry's own agent context. The entries never compete (the delegation nudge is
   main-thread only, the calculator is first-principles-agent only), so one shared bar would
   trade one entry's recall for the other's false positives.

Prints the ``threshold:`` value to set on each entry in ``catalog.yaml``.
"""

from __future__ import annotations

from pathlib import Path

from agent_router.core.catalog import MAIN_AGENT, load_catalog
from agent_router.deciders.embedders import Model2VecEmbedder
from agent_router.evaluate import DEFAULT_GRID, grid_search, load_cases, write_calibration

HERE = Path(__file__).resolve().parent


def main() -> int:
    catalog = load_catalog(HERE / "catalog.yaml")
    cases = load_cases(HERE / "eval_set.yaml", split="cal")
    embedder = Model2VecEmbedder()
    embedder.load()
    shared = grid_search(cases, DEFAULT_GRID, embedder=embedder, catalog=catalog)
    write_calibration({"model2vec": shared}, HERE / "calibration.json")
    p = shared.params
    print(
        f"shared: none_floor={p.none_floor} temperature={p.temperature} "
        f"not_for_penalty={p.not_for_penalty} (feasible={shared.feasible})"
    )
    for entry in catalog.entries:
        contexts = set(entry.agents)
        mine = [c for c in cases if (c.agent_type or MAIN_AGENT) in contexts or not contexts]
        res = grid_search(
            mine,
            {"threshold": DEFAULT_GRID["threshold"]},
            embedder=embedder,
            catalog=catalog,
            base=p,
        )
        print(
            f"{entry.id}: threshold: {res.threshold}  ({len(mine)} cal cases, "
            f"accuracy={res.accuracy:.3f} FP={res.false_positives} "
            f"cv_accuracy={res.cv_accuracy:.3f} cv_fpr={res.cv_fpr:.3f} "
            f"feasible={res.feasible})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
