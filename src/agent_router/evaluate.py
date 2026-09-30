"""Labelled evaluation and calibration of routing decisions.

- ``load_cases`` reads ``evals/eval_set.yaml`` (``split: cal | test``).
- ``run_eval(router_factory, cases)`` routes every case through a real ``Router`` and scores
  the outcome: a SUGGEST / ENFORCE counts as predicting its entry; anything else (NATIVE,
  SKIPPED, below threshold, decider error) counts as ``none``.
- ``grid_search`` / ``calibrate`` tune the local decider's ``none_floor``, ``temperature``,
  ``not_for_penalty`` and the router threshold on the ``cal`` split: best accuracy with at
  most one false positive and non-saturated probabilities, selected by stratified k-fold
  agreement, ties broken toward the conservative / softer setting (see ``grid_search``).
- ``write_calibration`` merges results per embedder into ``calibration.json``, tagged with
  the catalog version and embedder id they are valid for.

Metrics: ``accuracy`` over all cases; ``fpr`` = expected-none cases routed to an entry /
expected-none cases; ``misroute_rate`` = positives routed to a *different* entry / positives;
``errors`` = decider failures (fail-open NATIVE, scored as ``none`` but counted separately).
"""

from __future__ import annotations

import itertools
import json
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from agent_router.core.audit import AuditLog
from agent_router.core.catalog import Catalog
from agent_router.core.config import RouterConfig
from agent_router.core.router import Router
from agent_router.core.types import NONE_ID, Action, Decision, HookPoint, RouterEvent
from agent_router.deciders import local
from agent_router.deciders.base import Decider, recommended_threshold
from agent_router.deciders.embedders import Embedder
from agent_router.deciders.local import LocalJevDecider, LocalParams, embedder_id

DEFAULT_EVAL_SET = Path(__file__).resolve().parents[2] / "evals" / "eval_set.yaml"
SPLITS = ("cal", "test")
DECIDER_ERROR = "decider error"

TARGET_ACCURACY = 0.85
"""Holdout targets for the default decision path (asserted on the local -> Jev cascade)."""
TARGET_FPR = 0.10

# alpha stays at its default: the grid is kept small because ~60 calibration cases cannot
# support many free parameters. temperature starts at 0.08 so probabilities stay
# informative (below that model2vec saturates to ~1.0 / 0.0 on most cases).
DEFAULT_GRID: dict[str, list[float]] = {
    "none_floor": [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6],
    "temperature": [0.08, 0.1, 0.12, 0.15, 0.2, 0.25, 0.3],
    "not_for_penalty": [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0],
    "threshold": [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85],
}
QUICK_GRID: dict[str, list[float]] = {
    "none_floor": [0.3, 0.35, 0.45],
    "temperature": [0.08, 0.12],
    "threshold": [0.4, 0.5, 0.6],
}


# --- cases ----------------------------------------------------------------------


@dataclass(frozen=True)
class EvalCase:
    id: str
    text: str
    point: HookPoint
    expected: str
    tool_name: str | None = None
    tool_input: dict[str, Any] | None = None
    split: str = "test"
    decoy: bool = False
    recent: tuple[str, ...] = ()  # earlier prompts of the session (multi-turn context)
    agent_type: str | None = None  # the subagent making the call (None: main thread)

    def event(self, turn_id: int, session_id: str = "eval") -> RouterEvent:
        return RouterEvent(
            point=self.point,
            session_id=session_id,
            turn_id=turn_id,
            text=self.text,
            tool_name=self.tool_name,
            tool_input=self.tool_input,
            recent=self.recent,
            agent_type=self.agent_type,
        )


def load_cases(path: str | Path = DEFAULT_EVAL_SET, split: str | None = None) -> list[EvalCase]:
    """Cases from ``path``; ``split`` = ``cal`` | ``test`` | ``all`` / None (every case)."""
    if split not in (None, "all", *SPLITS):
        raise ValueError(f"split must be one of cal, test, all; got {split!r}")
    data = yaml.safe_load(Path(path).read_text())
    out = []
    for raw in data["cases"]:
        case = EvalCase(
            id=str(raw["id"]),
            text=str(raw.get("text") or ""),
            point=HookPoint(raw["point"]),
            expected=str(raw["expected"]),
            tool_name=raw.get("tool_name"),
            tool_input=raw.get("tool_input"),
            split=str(raw.get("split", "test")),
            decoy=bool(raw.get("decoy", False)),
            recent=tuple(str(r) for r in raw.get("recent") or ()),
            agent_type=raw.get("agent_type"),
        )
        if case.split not in SPLITS:
            raise ValueError(f"case {case.id}: split must be cal or test")
        if split in (None, "all") or case.split == split:
            out.append(case)
    return out


# --- scoring ----------------------------------------------------------------------


def predicted_label(decision: Decision) -> str:
    """The entry the agent would be nudged toward, or ``none``."""
    if decision.action in (Action.SUGGEST, Action.ENFORCE) and decision.entry_id:
        return decision.entry_id
    return NONE_ID


@dataclass(frozen=True)
class CaseResult:
    id: str
    expected: str
    predicted: str
    action: str
    reason: str
    prob: float | None = None
    decoy: bool = False
    error: bool = False
    latency_ms: float | None = None
    backend: str | None = None


@dataclass
class EvalReport:
    n: int
    accuracy: float
    fpr: float
    misroute_rate: float
    errors: int
    per_entry: dict[str, dict[str, float]]
    confusions: Counter[tuple[str, str]]
    results: list[CaseResult] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"n={self.n} accuracy={self.accuracy:.3f} FPR={self.fpr:.3f} "
            f"misroute={self.misroute_rate:.3f} errors={self.errors}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "accuracy": self.accuracy,
            "fpr": self.fpr,
            "misroute_rate": self.misroute_rate,
            "errors": self.errors,
            "per_entry": self.per_entry,
            "confusions": [
                {"expected": e, "predicted": p, "count": c}
                for (e, p), c in sorted(self.confusions.items())
            ],
            "cases": [asdict(r) for r in self.results],
        }


def score(results: Sequence[CaseResult]) -> EvalReport:
    n = len(results)
    correct = sum(r.predicted == r.expected for r in results)
    negatives = [r for r in results if r.expected == NONE_ID]
    positives = [r for r in results if r.expected != NONE_ID]
    fp = sum(r.predicted != NONE_ID for r in negatives)
    misroutes = sum(r.predicted not in (NONE_ID, r.expected) for r in positives)
    confusions: Counter[tuple[str, str]] = Counter(
        (r.expected, r.predicted) for r in results if r.predicted != r.expected
    )
    labels = sorted({r.expected for r in results} | {r.predicted for r in results})
    per_entry: dict[str, dict[str, float]] = {}
    for label in labels:
        tp = sum(r.expected == label and r.predicted == label for r in results)
        support = sum(r.expected == label for r in results)
        predicted = sum(r.predicted == label for r in results)
        per_entry[label] = {
            "support": support,
            "predicted": predicted,
            "precision": tp / predicted if predicted else 0.0,
            "recall": tp / support if support else 0.0,
        }
    return EvalReport(
        n=n,
        accuracy=correct / n if n else 0.0,
        fpr=fp / len(negatives) if negatives else 0.0,
        misroute_rate=misroutes / len(positives) if positives else 0.0,
        errors=sum(r.error for r in results),
        per_entry=per_entry,
        confusions=confusions,
        results=list(results),
    )


def _case_result(case: EvalCase, decision: Decision, predicted: str | None = None) -> CaseResult:
    res = decision.result
    prob = None
    if res is not None and res.choice in res.probabilities:
        prob = float(res.probabilities[res.choice])
    return CaseResult(
        id=case.id,
        expected=case.expected,
        predicted=predicted if predicted is not None else predicted_label(decision),
        action=str(decision.action),
        reason=decision.reason,
        prob=prob,
        decoy=case.decoy,
        error=decision.reason.startswith(DECIDER_ERROR),
        latency_ms=res.latency_ms if res is not None else None,
        backend=res.backend if res is not None else None,
    )


def routing_stats(report: EvalReport) -> dict[str, Any]:
    """Cascade escalation and latency over a report's cases.

    ``escalated``: cases the cascade sent to its confirm stage (any ``cascade:*`` backend but
    ``cascade:local``); ``fallbacks``: of those, the ones where the confirm stage failed and the
    local answer was used (``cascade:local-fallback``). Latency is the decider's, per case
    that reached it.
    """
    backends = [r.backend for r in report.results]
    escalated = sum(
        b is not None and b.startswith("cascade:") and b != "cascade:local" for b in backends
    )
    fallbacks = sum(b == "cascade:local-fallback" for b in backends)
    lat = [r.latency_ms for r in report.results if r.latency_ms is not None]
    n = len(report.results)
    return {
        "n": n,
        "escalated": escalated,
        "escalation_rate": escalated / n if n else 0.0,
        "fallbacks": fallbacks,
        "mean_latency_ms": float(np.mean(lat)) if lat else 0.0,
        "p95_latency_ms": float(np.percentile(lat, 95)) if lat else 0.0,
        "max_latency_ms": float(max(lat)) if lat else 0.0,
    }


def run_eval(router_factory: Callable[[], Router], cases: Iterable[EvalCase]) -> EvalReport:
    """Route every case through one router (a unique turn per case) and score it."""
    router = router_factory()
    results = [
        _case_result(case, router.route(case.event(turn_id=i)))
        for i, case in enumerate(cases, start=1)
    ]
    return score(results)


# --- routers ------------------------------------------------------------------------


def make_router(
    backend: str | Decider,
    catalog: Catalog,
    *,
    embedder: Embedder | None = None,
    threshold: float | None = None,
    timeout: float | None = None,
    mode: str = "advisory",
) -> Router:
    """A router with an in-memory audit (advisory by default) for evaluation and the CLI.

    ``threshold`` defaults to the decider's calibrated ``recommended_threshold`` when it has
    one, else ``RouterConfig``'s default. ``embedder`` applies to the local backend only.
    """
    if isinstance(backend, str):
        if backend == "local" and embedder is not None:
            decider: Decider = LocalJevDecider(
                embedder=embedder,
                native_examples=catalog.native_examples,
                catalog_version=catalog.version,
            )
        else:
            from agent_router.deciders.registry import make_decider

            decider = make_decider(backend, catalog)
    else:
        decider = backend
    if timeout is not None and hasattr(decider, "timeout"):
        decider.timeout = timeout  # type: ignore[attr-defined]
    if threshold is None:
        threshold = recommended_threshold(decider)
    if threshold is None:
        threshold = RouterConfig().threshold
    return Router(
        catalog,
        decider,
        RouterConfig(mode=mode, threshold=float(threshold)),  # type: ignore[arg-type]
        AuditLog(None),
    )


# --- calibration --------------------------------------------------------------------

TIEBREAK = ("threshold", "none_floor", "temperature", "not_for_penalty")
"""Among equally scored candidates prefer, in order, the higher value of each of these:
the more conservative (threshold, none_floor, not_for_penalty) and the softer (temperature)
setting."""


@dataclass(frozen=True)
class CalibrationResult:
    params: LocalParams
    threshold: float
    accuracy: float  # on all calibration cases, at the chosen point
    fpr: float
    misroute_rate: float
    false_positives: int
    feasible: bool  # FP, saturation and cross-validated FPR bounds all met; else not loaded
    n_cases: int
    cv_accuracy: float  # out-of-fold estimate of the whole selection procedure
    cv_fpr: float
    cv_agreement: float  # share of folds that selected the final candidate
    saturated: float  # share of routed cal cases whose top probability >= SATURATED
    edge_params: tuple[str, ...] = ()  # chosen values on the edge of their grid range
    catalog_version: str = ""
    embedder: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self.params),
            "threshold": self.threshold,
            "cal_accuracy": round(self.accuracy, 4),
            "cal_fpr": round(self.fpr, 4),
            "cal_misroute_rate": round(self.misroute_rate, 4),
            "cal_false_positives": self.false_positives,
            "cv_accuracy": round(self.cv_accuracy, 4),
            "cv_fpr": round(self.cv_fpr, 4),
            "cv_agreement": round(self.cv_agreement, 4),
            "saturated": round(self.saturated, 4),
            "edge_params": list(self.edge_params),
            "feasible": self.feasible,
            "n_cases": self.n_cases,
            "catalog_version": self.catalog_version,
            "embedder": self.embedder,
        }


SATURATED = 0.99


def _folds(cases: Sequence[EvalCase], k: int) -> np.ndarray:
    """Deterministic stratified fold index per case (round-robin inside each label)."""
    fold = np.zeros(len(cases), dtype=int)
    counters: Counter[str] = Counter()
    for i in sorted(range(len(cases)), key=lambda j: (cases[j].expected, cases[j].id)):
        fold[i] = counters[cases[i].expected] % k
        counters[cases[i].expected] += 1
    return fold


def rank_candidates(
    correct: np.ndarray,
    false_pos: np.ndarray,
    allowed: np.ndarray,
    tiebreak: np.ndarray,
    max_fp: int,
) -> np.ndarray:
    """Candidate indices, best first.

    ``correct`` / ``false_pos``: (candidates, cases) booleans over the cases to select on;
    ``allowed``: (candidates,) extra feasibility (e.g. not saturated); ``tiebreak``:
    (candidates, t) values where higher is preferred. Feasible (FP count <= ``max_fp`` and
    allowed) candidates rank first by accuracy, then fewer FPs, then ``tiebreak``; when none
    is feasible, the fewest-FP candidates come first.
    """
    acc = correct.mean(axis=1) if correct.shape[1] else np.zeros(len(correct))
    fps = false_pos.sum(axis=1)
    feasible = (fps <= max_fp) & allowed
    # np.lexsort sorts by the LAST key first, ascending: negate what should be descending.
    keys = [-tiebreak[:, j] for j in reversed(range(tiebreak.shape[1]))]
    keys += [fps, -acc] if feasible.any() else [-acc, fps]
    keys.append(~feasible)
    return np.lexsort(keys)


def grid_search(
    cases: Sequence[EvalCase],
    grid: Mapping[str, Sequence[float]] | None = None,
    *,
    embedder: Embedder,
    catalog: Catalog,
    base: LocalParams | None = None,
    max_fp: int = 1,
    k: int = 5,
    max_saturated: float = 0.5,
    max_cv_fpr: float = TARGET_FPR,
) -> CalibrationResult:
    """Conservative, cross-validated grid search of ``LocalParams`` x router threshold.

    ``grid`` maps ``threshold`` and ``LocalParams`` fields to candidate values (fields not
    in the grid keep ``base``). Every params combination is routed once at threshold 0 and
    each threshold is applied to p(choice) as the router does.

    A candidate is feasible when it has at most ``max_fp`` false positives on the cases it
    is selected on (FPR <= 0.10 is too loose a bound to trust on ~25 negatives) and at most
    ``max_saturated`` of routed cases get a top probability >= ``SATURATED`` (so the
    probabilities stay informative). Selection is stratified ``k``-fold: each fold picks
    the best candidate on the other folds and is scored on its own (``cv_*``, an honest
    estimate of the procedure). The final candidate is the one most folds agreed on (ties
    and no agreement resolved by the ranking on all cases), provided it is feasible on all
    cases; otherwise the best on all cases. The result is only ``feasible`` (and only then
    loaded by the decider) if that candidate is feasible on all cases AND the cross-validated
    FPR is at most ``max_cv_fpr``: a selection procedure whose out-of-fold FPR already
    exceeds the target is not trusted to ship.
    """
    grid = grid if grid is not None else DEFAULT_GRID
    base = base if base is not None else LocalParams()
    keys = [f.name for f in fields(LocalParams) if f.name in grid]
    thresholds = [float(t) for t in grid["threshold"]]
    if not cases or not thresholds:
        raise ValueError("calibration needs cases and a non-empty grid")
    decider = LocalJevDecider(
        embedder=embedder, params=base, native_examples=catalog.native_examples
    )
    expected = [c.expected for c in cases]
    negative = np.array([e == NONE_ID for e in expected])
    cands: list[tuple[LocalParams, float]] = []
    correct_rows, fp_rows, mis_rows, allowed = [], [], [], []
    sat_rows: list[float] = []
    for combo in itertools.product(*(grid[k_] for k_ in keys)):
        params = replace(base, **{k_: float(v) for k_, v in zip(keys, combo, strict=True)})
        decider.params = params
        router = make_router(decider, catalog, threshold=0.0)
        decisions = [router.route(c.event(turn_id=i)) for i, c in enumerate(cases, start=1)]
        labels = [predicted_label(d) for d in decisions]
        probs = np.array([_prob(d) for d in decisions])
        tops = [max(d.result.probabilities.values()) for d in decisions if d.result is not None]
        saturated = float(np.mean([t >= SATURATED for t in tops])) if tops else 0.0
        for thr in thresholds:
            pred = [
                lab if lab != NONE_ID and p >= thr else NONE_ID
                for lab, p in zip(labels, probs, strict=True)
            ]
            ok = np.array([a == b for a, b in zip(pred, expected, strict=True)])
            routed_to_entry = np.array([lab != NONE_ID for lab in pred])
            cands.append((params, thr))
            correct_rows.append(ok)
            fp_rows.append(negative & routed_to_entry)
            mis_rows.append(~negative & routed_to_entry & ~ok)
            allowed.append(saturated <= max_saturated)
            sat_rows.append(saturated)
    correct, false_pos, misroute = np.array(correct_rows), np.array(fp_rows), np.array(mis_rows)
    allowed_arr = np.array(allowed)
    tiebreak = np.array(
        [
            [thr if name == "threshold" else getattr(p, name) for name in TIEBREAK]
            for p, thr in cands
        ]
    )

    def rank(idx: np.ndarray) -> np.ndarray:
        return rank_candidates(correct[:, idx], false_pos[:, idx], allowed_arr, tiebreak, max_fp)

    everything = np.arange(len(cases))
    fold = _folds(cases, k)
    picks: list[int] = []
    oof_correct = np.zeros(len(cases), dtype=bool)
    oof_fp = np.zeros(len(cases), dtype=bool)
    for f in range(k):
        test_idx = everything[fold == f]
        if not len(test_idx):
            continue
        pick = int(rank(everything[fold != f])[0])
        picks.append(pick)
        oof_correct[test_idx] = correct[pick, test_idx]
        oof_fp[test_idx] = false_pos[pick, test_idx]

    order = rank(everything)
    position = {int(c): i for i, c in enumerate(order)}
    full_fp = false_pos.sum(axis=1)
    feasible_all = (full_fp <= max_fp) & allowed_arr
    votes = Counter(p for p in picks if feasible_all[p])
    if votes and max(votes.values()) >= 2:
        top = max(votes.values())
        final = min((c for c, v in votes.items() if v == top), key=position.__getitem__)
    else:
        final = int(order[0])

    params, thr = cands[final]
    edges = []
    for name in (*keys, "threshold"):
        values = sorted({float(v) for v in grid[name]})
        chosen = thr if name == "threshold" else getattr(params, name)
        if len(values) > 1 and chosen in (values[0], values[-1]):
            edges.append(name)
    n_neg = int(negative.sum())
    n_pos = len(cases) - n_neg
    cv_fpr = float(oof_fp.sum() / n_neg) if n_neg else 0.0
    return CalibrationResult(
        params=params,
        threshold=thr,
        accuracy=float(correct[final].mean()),
        fpr=float(full_fp[final] / n_neg) if n_neg else 0.0,
        misroute_rate=float(misroute[final].sum() / n_pos) if n_pos else 0.0,
        false_positives=int(full_fp[final]),
        feasible=bool(feasible_all[final]) and cv_fpr <= max_cv_fpr,
        n_cases=len(cases),
        cv_accuracy=float(oof_correct.mean()),
        cv_fpr=cv_fpr,
        cv_agreement=sum(p == final for p in picks) / len(picks) if picks else 0.0,
        saturated=sat_rows[final],
        edge_params=tuple(edges),
        catalog_version=catalog.version,
        embedder=embedder_id(embedder) or "",
    )


def _prob(decision: Decision) -> float:
    res = decision.result
    if res is None or decision.entry_id is None:
        return 0.0
    return float(res.probabilities.get(decision.entry_id, 0.0))


def calibrate(
    cases: Sequence[EvalCase],
    grid: Mapping[str, Sequence[float]] | None = None,
    *,
    embedder: Embedder,
    catalog: Catalog,
) -> tuple[LocalParams, float]:
    res = grid_search(cases, grid, embedder=embedder, catalog=catalog)
    return res.params, res.threshold


def write_calibration(
    results: Mapping[str, CalibrationResult | CascadeCalibration], path: str | Path | None = None
) -> Path:
    """Merge ``{key: result}`` (``model2vec`` / ``hashing`` / ``cascade``) into the calibration
    JSON (default: the active one)."""
    path = Path(path) if path is not None else local.CALIBRATION_PATH
    data: dict[str, Any] = {}
    if path.is_file():
        try:
            data = json.loads(path.read_text())
        except ValueError:
            data = {}
    for key, res in results.items():
        data[key] = res.to_dict()
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    return path


# --- cascade native_gate calibration ------------------------------------------------------

GATE_GRID: tuple[float, ...] = (
    0.5,
    0.6,
    0.7,
    0.8,
    0.85,
    0.9,
    0.93,
    0.95,
    0.97,
    0.98,
    0.99,
    0.995,
    0.999,
    0.9999,
    1.0,
    1.01,
)
"""Candidate ``native_gate`` values. p(none) can saturate to exactly 1.0, so 1.0 still keeps
those cases local; 1.01 is the "always escalate" sentinel (= Jev on every step)."""


class _Memo:
    """Caches a stage decider's answer per (state, options): one real call per case.

    Exceptions are recorded and re-raised; ``calibrate_cascade`` refuses to calibrate on them
    (a failed Jev call would otherwise be scored as the cascade's local fallback)."""

    def __init__(self, decider: Decider) -> None:
        self.decider = decider
        self.name = getattr(decider, "name", "")
        self.cache: dict[tuple[str, tuple[str, ...]], Any] = {}
        self.errors: list[str] = []

    def decide(self, state: str, options: dict[str, Any]) -> Any:
        key = (state, tuple(options))
        if key not in self.cache:
            try:
                self.cache[key] = self.decider.decide(state, options)
            except Exception as exc:
                self.errors.append(f"{type(exc).__name__}: {exc}")
                raise
        return self.cache[key]


@dataclass(frozen=True)
class CascadeCalibration:
    native_gate: float
    threshold: float
    accuracy: float  # cascade on the cal cases at the chosen gate
    fpr: float
    escalation_rate: float
    jev_accuracy: float  # confirm stage alone on the same cases, same answers
    jev_fpr: float
    n_cases: int
    local_errors: int  # cases the cascade answered locally and got wrong
    sweep: tuple[dict[str, float], ...]
    catalog_version: str = ""
    embedder: str = ""
    local_params: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "native_gate": self.native_gate,
            "threshold": self.threshold,
            "cal_accuracy": round(self.accuracy, 4),
            "cal_fpr": round(self.fpr, 4),
            "cal_escalation_rate": round(self.escalation_rate, 4),
            "cal_jev_accuracy": round(self.jev_accuracy, 4),
            "cal_jev_fpr": round(self.jev_fpr, 4),
            "cal_local_errors": self.local_errors,
            "n_cases": self.n_cases,
            "feasible": True,  # the always-escalate sentinel guarantees a feasible gate
            "sweep": list(self.sweep),
            "catalog_version": self.catalog_version,
            "embedder": self.embedder,
            "local_params": dict(self.local_params),
        }


def calibrate_cascade(
    cases: Sequence[EvalCase],
    catalog: Catalog,
    primary: Decider,
    confirm: Decider,
    *,
    gates: Sequence[float] = GATE_GRID,
    threshold: float | None = None,
) -> CascadeCalibration:
    """Pick the cascade's ``native_gate`` on ``cases`` (the cal split).

    Objective: the lowest escalation rate among gates whose cal accuracy is not below and
    whose cal FPR is not above the confirm stage alone (Jev-only); ties go to the higher,
    more cautious gate. Both stages are memoised, so Jev is asked at most once per case and
    every gate (and the Jev-only baseline) is scored on the same answers. ``threshold`` is
    the router threshold for both (default: the confirm stage's recommended one, else the
    ``RouterConfig`` default) and is stored with the gate. Raises ``RuntimeError`` if any
    confirm call failed.
    """
    from agent_router.deciders.cascade import CascadeDecider

    if not cases or not gates:
        raise ValueError("cascade calibration needs cases and gates")
    thr = threshold if threshold is not None else recommended_threshold(confirm)
    thr = float(thr if thr is not None else RouterConfig().threshold)
    first, second = _Memo(primary), _Memo(confirm)

    def check() -> None:
        if second.errors:
            raise RuntimeError(
                f"confirm stage failed on {len(second.errors)} case(s) during cascade "
                f"calibration (first: {second.errors[0]}); not calibrating on fallbacks"
            )

    jev = run_eval(lambda: make_router(second, catalog, threshold=thr), cases)
    check()
    rows = []
    for gate in sorted({float(g) for g in gates}):
        dec = CascadeDecider(first, second, gate)
        rep = run_eval(lambda d=dec: make_router(d, catalog, threshold=thr), cases)
        check()
        stats = routing_stats(rep)
        local_errors = sum(
            r.backend == "cascade:local" and r.predicted != r.expected for r in rep.results
        )
        rows.append((gate, rep, stats["escalation_rate"], local_errors))
    eps = 1e-9
    feasible = [
        row for row in rows if row[1].accuracy >= jev.accuracy - eps and row[1].fpr <= jev.fpr + eps
    ]
    if not feasible:  # only possible without an always-escalate gate in ``gates``
        raise RuntimeError("no native_gate matches Jev-only on the cal cases; add a gate > 1")
    gate, rep, esc, local_errors = min(feasible, key=lambda row: (row[2], -row[0]))
    emb = getattr(primary, "embedder", None)
    params = getattr(primary, "params", None)
    return CascadeCalibration(
        native_gate=gate,
        threshold=thr,
        accuracy=rep.accuracy,
        fpr=rep.fpr,
        escalation_rate=esc,
        jev_accuracy=jev.accuracy,
        jev_fpr=jev.fpr,
        n_cases=len(cases),
        local_errors=local_errors,
        sweep=tuple(
            {
                "gate": g,
                "accuracy": round(r.accuracy, 4),
                "fpr": round(r.fpr, 4),
                "escalation_rate": round(e, 4),
            }
            for g, r, e, _ in rows
        ),
        catalog_version=catalog.version,
        embedder=(embedder_id(emb) or "") if emb is not None else "",
        local_params=asdict(params) if isinstance(params, LocalParams) else {},
    )
