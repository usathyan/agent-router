"""``agent-router`` command line: route, eval, calibrate, calibrate-cascade, demo, run, hook,
mcp.

``--backend`` defaults to ``default_backend()``: the local -> Jev cascade when a Jev API key
is set, else the offline local classifier (``hook`` defaults to local; see ``cmd_hook``).

``--catalog`` (route, eval, calibrate, calibrate-cascade, hook) defaults to
``AGENT_ROUTER_CATALOG``, else the packaged catalog. ``AGENT_ROUTER_CALIBRATION`` points the
local decider at another calibration file (e.g. one fit for that catalog).

Heavy or optional modules (the demo server, the live agent, model weights) are imported
inside the command that needs them, so ``route`` / ``eval`` work without them.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from agent_router.core.types import HookPoint, RouterEvent

EMBEDDERS = ("model2vec", "hashing")


class CliError(Exception):
    """A user-facing error: printed to stderr, exit status 2."""


# --- helpers ---------------------------------------------------------------------------


def _set_embedder(name: str | None) -> None:
    if name:
        os.environ["AGENT_ROUTER_EMBEDDER"] = name


def _catalog(args: argparse.Namespace):
    """``--catalog``, else ``AGENT_ROUTER_CATALOG``, else the packaged catalog."""
    from agent_router.core.catalog import DEFAULT_CATALOG, CatalogError, load_catalog

    path = getattr(args, "catalog", None) or os.environ.get("AGENT_ROUTER_CATALOG", "").strip()
    try:
        return load_catalog(path or DEFAULT_CATALOG)
    except (OSError, CatalogError) as exc:
        raise CliError(f"cannot load catalog {path or DEFAULT_CATALOG}: {exc}") from exc


def _apply_calibration_env() -> None:
    raw = os.environ.get("AGENT_ROUTER_CALIBRATION", "").strip()
    if raw:
        from agent_router.deciders import local

        local.CALIBRATION_PATH = Path(raw)


def _check_backend(name: str) -> None:
    from agent_router.deciders.registry import BACKENDS, available_backends

    if name not in BACKENDS:
        raise CliError(f"unknown backend {name!r}; expected one of {', '.join(BACKENDS)}")
    if not available_backends().get(name, False):
        raise CliError(f"backend {name!r} is not available here (missing package or API key)")


def _resolve_backend(args: argparse.Namespace) -> str:
    """``--backend``, else the default path (``cascade`` with a Jev key, else ``local``)."""
    from agent_router.deciders.registry import default_backend

    if not getattr(args, "backend", None):
        args.backend = default_backend()
    return args.backend


def _router(
    args: argparse.Namespace,
    *,
    env: bool = False,
    audit_path: Path | None = None,
    suggested: Any = (),
):
    """The router for a command.

    ``env=True`` (``route``, ``run``): the ``AGENT_ROUTER_*`` environment config applies
    (mode, threshold, disabled, audit path), with the decider's calibrated threshold unless
    ``AGENT_ROUTER_THRESHOLD`` is set; an explicit ``--mode`` / ``--threshold`` beats it.
    ``env=False`` (``eval``): an explicit advisory config and an in-memory audit, so scores
    never depend on the shell.
    """
    from dataclasses import replace

    from agent_router.core.audit import AuditLog
    from agent_router.core.config import RouterConfig
    from agent_router.core.router import Router
    from agent_router.deciders.base import recommended_threshold
    from agent_router.evaluate import make_router

    _check_backend(_resolve_backend(args))
    _set_embedder(getattr(args, "embedder", None))
    catalog = _catalog(args)
    mode = getattr(args, "mode", None)
    threshold = getattr(args, "threshold", None)
    router = make_router(
        args.backend,
        catalog,
        threshold=threshold,
        timeout=getattr(args, "timeout", None),
        mode=mode or "advisory",
    )
    if not env:
        return router
    try:
        config = RouterConfig.from_env(recommended_threshold=recommended_threshold(router.decider))
        if mode is not None:
            config = replace(config, mode=mode)
        if threshold is not None:
            config = replace(config, threshold=threshold)
        if audit_path is not None:
            config = replace(config, audit_path=audit_path)
    except ValueError as exc:
        raise CliError(f"invalid AGENT_ROUTER_* setting or flag: {exc}") from exc
    return Router(catalog, router.decider, config, AuditLog(config.audit_path), suggested)


def _print_json(obj: Any) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=False, default=str))


# --- route -------------------------------------------------------------------------------


def cmd_route(args: argparse.Namespace) -> int:
    point = HookPoint(args.point)
    tool_name = args.tool
    if point == HookPoint.SKILL and tool_name is None:
        tool_name = "Skill"
    router = _router(args, env=True)
    event = RouterEvent(
        point=point,
        session_id="cli",
        turn_id=1,
        text=args.text,
        tool_name=tool_name,
        tool_input=args.input,
    )
    d = router.route(event)
    res = d.result
    out = {
        "action": str(d.action),
        "choice": res.choice if res else None,
        "entry_id": d.entry_id,
        "probabilities": res.probabilities if res else {},
        "confidence": res.confidence if res else None,
        "reason": d.reason,
        "hint": d.hint,
        "options": list(d.options),
        "backend": res.backend if res else args.backend,
        "latency_ms": res.latency_ms if res else None,
        "stages": list(res.stages) if res else [],
        "threshold": router.config.threshold,
        "applied_threshold": d.threshold,
    }
    if args.json:
        _print_json(out)
        return 0
    print(f"action     {out['action']}  ({d.reason})")
    print(f"choice     {out['choice']}")
    if res:
        print(f"confidence {res.confidence:.3f}   backend {res.backend}   {res.latency_ms:.1f} ms")
        if res.stages:
            print(f"cascade    {_cascade_note(res.backend)}")
            for st in res.stages:
                if st.get("skipped"):
                    print(f"  [{st['role']} {st['backend']}] skipped: {st['skipped']}")
                    continue
                if st.get("error"):  # the operator's own terminal, like the audit log
                    print(f"  [{st['role']} {st['backend']}] failed: {st['error']}")
                    continue
                print(f"  [{st['role']} {st['backend']}] choice {st['choice']}")
                _print_bars(st["probabilities"], indent="    ")
        else:
            _print_bars(res.probabilities)
    if d.hint:
        print(f"hint       {d.hint}")
    return 0


def _cascade_note(backend: str) -> str:
    if backend == "cascade:local":
        return "answered locally (confident none)"
    if backend == "cascade:local-fallback":
        return "escalated, but the confirm stage failed: local answer, biased to none"
    return f"escalated to {backend.removeprefix('cascade:')}"


def _print_bars(probs: dict[str, float], indent: str = "  ") -> None:
    for oid, p in sorted(probs.items(), key=lambda kv: -kv[1]):
        print(f"{indent}{oid:<18} {p:6.3f} {'#' * round(p * 40)}")


# --- eval --------------------------------------------------------------------------------


def cmd_eval(args: argparse.Namespace) -> int:
    from agent_router.deciders.local import embedder_key
    from agent_router.evaluate import DEFAULT_EVAL_SET, load_cases, routing_stats, run_eval

    cases = load_cases(args.cases or DEFAULT_EVAL_SET, split=args.split)
    router = _router(args)
    if args.skip_input or args.skip_prompt:
        from dataclasses import replace

        router.config = replace(
            router.config,
            skip_input=args.skip_input or router.config.skip_input,
            skip_prompt=args.skip_prompt or router.config.skip_prompt,
        )
    report = run_eval(lambda: router, cases)
    stats = routing_stats(report)
    stage = getattr(router.decider, "primary", router.decider)
    if args.json:
        _print_json(
            {
                "backend": args.backend,
                "embedder": embedder_key(getattr(stage, "embedder", None)),
                "split": args.split,
                "threshold": router.config.threshold,
                "routing": stats,
                **report.to_dict(),
            }
        )
        return 0
    print(f"backend {args.backend}  split {args.split}  threshold {router.config.threshold:.2f}")
    print(
        f"n {report.n}  accuracy {report.accuracy:.3f}  FPR {report.fpr:.3f}  "
        f"misroute {report.misroute_rate:.3f}  errors {report.errors}"
    )
    if args.backend == "cascade":
        print(
            f"escalation {stats['escalation_rate']:.3f} ({stats['escalated']}/{stats['n']})  "
            f"fallbacks {stats['fallbacks']}"
        )
    print(
        f"latency mean {stats['mean_latency_ms']:.1f} ms  p95 {stats['p95_latency_ms']:.1f} ms  "
        f"max {stats['max_latency_ms']:.1f} ms"
    )
    print("per entry (precision / recall / support):")
    for label, m in report.per_entry.items():
        print(f"  {label:<18} {m['precision']:.2f} / {m['recall']:.2f} / {int(m['support'])}")
    if report.confusions:
        print("confusions (expected -> predicted):")
        for (exp, pred), n in report.confusions.most_common():
            print(f"  {exp} -> {pred}: {n}")
    if args.verbose:
        for r in report.results:
            mark = "ok " if r.predicted == r.expected else "BAD"
            prob = f"{r.prob:.2f}" if r.prob is not None else "  - "
            print(f"  {mark} {r.id:<12} exp={r.expected:<17} got={r.predicted:<17} p={prob}")
    return 0


# --- calibrate -----------------------------------------------------------------------------


def cmd_calibrate(args: argparse.Namespace) -> int:
    from agent_router.deciders.embedders import HashingEmbedder, Model2VecEmbedder
    from agent_router.evaluate import (
        DEFAULT_EVAL_SET,
        DEFAULT_GRID,
        QUICK_GRID,
        grid_search,
        load_cases,
        write_calibration,
    )

    catalog = _catalog(args)
    cases = load_cases(args.cases or DEFAULT_EVAL_SET, split="cal")
    grid = QUICK_GRID if args.quick else DEFAULT_GRID
    names = EMBEDDERS if args.embedder == "all" else (args.embedder,)
    results = {}
    for name in names:
        if name == "hashing":
            emb: Any = HashingEmbedder()
        else:
            emb = Model2VecEmbedder()
            try:
                emb.load()
            except Exception as exc:  # network, package or file problems
                raise CliError(f"cannot load the model2vec embedder: {exc}") from exc
        res = grid_search(cases, grid, embedder=emb, catalog=catalog)
        results[name] = res
        p = res.params
        print(
            f"{name}: none_floor={p.none_floor} temperature={p.temperature} "
            f"not_for_penalty={p.not_for_penalty} threshold={res.threshold}\n"
            f"  cal accuracy={res.accuracy:.3f} FPR={res.fpr:.3f} FP={res.false_positives} "
            f"| {len(cases)}-case CV accuracy={res.cv_accuracy:.3f} FPR={res.cv_fpr:.3f} "
            f"fold agreement={res.cv_agreement:.2f} | saturated={res.saturated:.2f}"
        )
        if not res.feasible:
            print(f"  WARNING: {name}: not feasible (FP, saturation or CV FPR bound); not loaded")
        if res.edge_params:
            print(f"  WARNING: {name}: on the grid edge: {', '.join(res.edge_params)}")
    path = write_calibration(results, args.out)
    print(f"wrote {path}")
    return 0


def cmd_calibrate_cascade(args: argparse.Namespace) -> int:
    from agent_router.deciders import registry
    from agent_router.evaluate import (
        DEFAULT_EVAL_SET,
        calibrate_cascade,
        load_cases,
        write_calibration,
    )

    _check_backend("cascade")
    _set_embedder(args.embedder)
    catalog = _catalog(args)
    cases = load_cases(args.cases or DEFAULT_EVAL_SET, split="cal")
    primary = registry.make_decider("local", catalog)
    confirm = registry.make_decider("jev", catalog)
    if args.timeout is not None and hasattr(confirm, "timeout"):
        confirm.timeout = args.timeout  # type: ignore[attr-defined]
    try:
        res = calibrate_cascade(cases, catalog, primary, confirm)
    except RuntimeError as exc:
        raise CliError(str(exc)) from exc
    print(
        f"cascade: native_gate={res.native_gate} threshold={res.threshold} "
        f"on {res.n_cases} cal cases ({res.embedder or 'unknown embedder'})\n"
        f"  cascade accuracy={res.accuracy:.3f} FPR={res.fpr:.3f} "
        f"escalation={res.escalation_rate:.3f} | jev-only accuracy={res.jev_accuracy:.3f} "
        f"FPR={res.jev_fpr:.3f}"
    )
    path = write_calibration({"cascade": res}, args.out)
    print(f"wrote {path}")
    return 0


# --- demo / run (optional modules) -------------------------------------------------------------


def cmd_demo(args: argparse.Namespace) -> int:
    try:
        import agent_router.demo.server as server
    except ModuleNotFoundError as exc:
        print(
            f"demo server is not available ({exc}); agent_router.demo.server is missing",
            file=sys.stderr,
        )
        return 1
    catalog = _catalog(args) if (args.catalog or os.environ.get("AGENT_ROUTER_CATALOG")) else None
    audit_dir = Path(args.audit_dir) if args.audit_dir else None
    trace_dir = Path(args.trace_dir) if args.trace_dir else None
    server.main(  # loopback only
        port=args.port,
        allow_shell=args.allow_shell,
        catalog=catalog,
        audit_dir=audit_dir,
        trace_dir=trace_dir,
    )
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    try:
        from agent_router.agent import run_agent
    except ModuleNotFoundError as exc:
        print(
            f"live agent is not available ({exc}); agent_router.agent is missing", file=sys.stderr
        )
        return 1
    import asyncio
    import time

    router = _router(args, env=True)
    start = time.perf_counter()

    def on_event(event: Any) -> None:
        stamp = f"[{time.perf_counter() - start:6.2f}s]"
        if isinstance(event, dict):
            kind = event.get("type") or event.get("event") or "event"
            body = {k: v for k, v in event.items() if k not in ("type", "event")}
            print(f"{stamp} {kind:<10} {json.dumps(body, ensure_ascii=False, default=str)[:300]}")
        else:
            print(f"{stamp} {event}")

    workspace = Path(args.workspace).resolve()
    result = asyncio.run(
        run_agent(args.prompt, router, workspace, on_event, allow_shell=args.allow_shell)
    )
    print(result)
    return 0


# --- parser ----------------------------------------------------------------------------------


# --- hook / mcp (Claude Code plugin) ---------------------------------------------------------


def cmd_hook(args: argparse.Namespace) -> int:
    """One Claude Code command hook: hook JSON on stdin, hook JSON on stdout, always exit 0.

    The backend is ``--backend``, else ``AGENT_ROUTER_BACKEND``, else ``local`` (offline and
    deterministic: a hook should not spend money or depend on the network by default).
    Any failure prints ``{}`` (fail open).
    """
    import logging

    from agent_router.adapters.claude_code import handle

    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    out: dict[str, Any] = {}
    try:
        if not args.backend:
            args.backend = os.environ.get("AGENT_ROUTER_BACKEND", "").strip() or "local"
        payload = json.load(sys.stdin)
        if isinstance(payload, dict):
            out = handle(
                payload,
                lambda audit, seed: _router(args, env=True, audit_path=audit, suggested=seed),
            )
    except Exception as exc:  # a hook must never break the host session
        print(f"agent-router hook: failing open: {type(exc).__name__}: {exc}", file=sys.stderr)
        out = {}
    print(json.dumps(out, ensure_ascii=False))
    return 0


def cmd_mcp(args: argparse.Namespace) -> int:
    from agent_router.adapters.mcp_stdio import TOOLS, build_server

    tools = args.tools.split(",") if args.tools else list(TOOLS)
    try:
        server = build_server(Path(args.root or os.getcwd()), [t.strip() for t in tools])
    except ValueError as exc:
        raise CliError(str(exc)) from exc
    server.run()
    return 0


def _json_obj(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"--input must be JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise argparse.ArgumentTypeError("--input must be a JSON object")
    return value


def _add_catalog(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--catalog", help="catalog YAML (default: AGENT_ROUTER_CATALOG, else the packaged one)"
    )


def _add_backend(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--backend",
        help="decider backend (default: cascade when a Jev API key is set, else local)",
    )
    p.add_argument("--embedder", choices=EMBEDDERS, help="local backend embedder")
    p.add_argument(
        "--threshold",
        type=float,
        help="router threshold (default: AGENT_ROUTER_THRESHOLD for route/run, else calibrated)",
    )
    p.add_argument("--timeout", type=float, help="hosted backend request timeout (s)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent-router",
        description="Route agent steps to MIT catalog alternatives with a Jev-spec decider.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("route", help="route one step and show the decision")
    p.add_argument("text", help="prompt text (at tool/skill points: the turn's prompt)")
    p.add_argument("--point", choices=[h.value for h in HookPoint], default="prompt")
    p.add_argument("--tool", help="native tool about to run (Bash, Read, Glob, Skill, ...)")
    p.add_argument("--input", type=_json_obj, help="tool input as a JSON object")
    p.add_argument("--json", action="store_true", help="print JSON")
    _add_backend(p)
    _add_catalog(p)
    p.set_defaults(func=cmd_route)

    p = sub.add_parser("eval", help="score a backend on the labelled eval set")
    p.add_argument("--split", choices=["test", "cal", "all"], default="test")
    p.add_argument("--cases", help="eval set YAML (default: evals/eval_set.yaml)")
    p.add_argument("--json", action="store_true", help="print JSON")
    p.add_argument("-v", "--verbose", action="store_true", help="list every case")
    p.add_argument(
        "--skip-input", help="skip pattern to apply, as the deployed AGENT_ROUTER_SKIP_INPUT"
    )
    p.add_argument(
        "--skip-prompt", help="prompt skip pattern, as the deployed AGENT_ROUTER_SKIP_PROMPT"
    )
    _add_backend(p)
    _add_catalog(p)
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("calibrate", help="grid-search local decider params on the cal split")
    p.add_argument("--embedder", choices=[*EMBEDDERS, "all"], default="all")
    p.add_argument("--cases", help="eval set YAML (default: evals/eval_set.yaml)")
    p.add_argument("--out", help="calibration JSON (default: the packaged calibration.json)")
    p.add_argument("--quick", action="store_true", help="small grid (smoke test)")
    _add_catalog(p)
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser(
        "calibrate-cascade",
        help="fit the cascade's native_gate on the cal split (asks Jev once per cal case)",
    )
    p.add_argument("--embedder", choices=EMBEDDERS, help="local stage embedder")
    p.add_argument("--cases", help="eval set YAML (default: evals/eval_set.yaml)")
    p.add_argument("--out", help="calibration JSON (default: the packaged calibration.json)")
    p.add_argument("--timeout", type=float, default=15.0, help="Jev request timeout (s)")
    _add_catalog(p)
    p.set_defaults(func=cmd_calibrate_cascade)

    p = sub.add_parser(
        "demo", help="start the visual demo server on http://127.0.0.1 (loopback only)"
    )
    p.add_argument("--port", type=int, default=8765)
    p.add_argument(
        "--allow-shell",
        action="store_true",
        help="auto-approve Bash/WebFetch in live agent runs (default: off)",
    )
    p.add_argument(
        "--audit-dir", help="audit JSONL folder to replay (default: .agent-router/audit)"
    )
    p.add_argument(
        "--trace-dir",
        help="decision-trace JSONL folder for the Trace tab (default: trace/ beside --audit-dir)",
    )
    _add_catalog(p)
    p.set_defaults(func=cmd_demo)

    p = sub.add_parser("run", help="run the live agent and print its timeline")
    p.add_argument("prompt")
    p.add_argument(
        "--mode",
        choices=["advisory", "enforce"],
        help="router mode (default: AGENT_ROUTER_MODE, else advisory)",
    )
    p.add_argument("--workspace", default=".")
    p.add_argument(
        "--allow-shell",
        action="store_true",
        help="auto-approve Bash/WebFetch for the agent (default: off)",
    )
    _add_backend(p)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser(
        "hook", help="Claude Code command hook: hook JSON on stdin, hook JSON on stdout"
    )
    _add_backend(p)
    _add_catalog(p)
    p.set_defaults(func=cmd_hook)

    p = sub.add_parser("mcp", help="serve the catalog tools as a stdio MCP server")
    p.add_argument("--tools", help="comma-separated subset (default: all)")
    p.add_argument("--root", help="directory the file tools are confined to (default: cwd)")
    p.set_defaults(func=cmd_mcp)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _apply_calibration_env()
    try:
        return int(args.func(args))
    except CliError as exc:
        print(f"agent-router: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
