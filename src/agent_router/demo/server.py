"""Demo server: Playground (route one step), Live agent (SSE timeline) and audit Replay.

``create_app(audit_dir)`` builds the FastAPI app; ``main(port)`` serves it on 127.0.0.1.

API
---
``GET  /api/catalog``            catalog entries, the ``none`` option, default threshold/mode
``GET  /api/backends``           ``available_backends()`` as a list (read per request) and the
                                 ``default_backend()`` (cascade with a Jev key, else local)
``POST /api/route``              ``{text, point, tool_name?, tool_input?, backend?, mode?,
                                 threshold?}`` -> ``{decision, result, options, hint,
                                 latency_ms, state, threshold, mode, backend}``; stateless.
                                 ``result.stages`` lists the cascade's per-stage answers.
                                 Without ``threshold``: ``AGENT_ROUTER_THRESHOLD`` if set,
                                 else the backend's calibrated threshold, else 0.5
``POST /api/run``                ``{prompt, mode, backend}`` -> ``{token, stream}``: a random
                                 single-use token valid for ``RUN_TOKEN_TTL`` seconds
``GET  /api/run/stream``         ``?token=`` -> ``text/event-stream`` of the timeline events
                                 (see ``adapters.claude_sdk``), framed by ``session`` (first)
                                 and ``done`` (last); one live run at a time (else 409)
``GET  /api/audit/sessions``     audit JSONL files, newest first
``GET  /api/audit/{session}``    one session's records (bad lines skipped)

Deciders are built once per backend name and serialised with a lock; every route call
runs in the threadpool so network-backed deciders never block the event loop.

Live runs start a real agent, so they are locked down: only ``127.0.0.1`` / ``localhost``
Host headers are served (defeats DNS rebinding), run requests must be same-origin
(``Sec-Fetch-Site`` and ``Origin`` checked when present), the run is started by a POST that
mints a one-time token, and ``Bash`` / ``WebFetch`` are not auto-approved unless the server
was started with ``allow_shell=True``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import secrets
import shutil
import threading
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import anyio
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from agent_router.core.audit import AuditLog
from agent_router.core.catalog import Catalog, CatalogEntry, load_catalog
from agent_router.core.config import MODES, RouterConfig
from agent_router.core.hints import render_deny, render_hint
from agent_router.core.router import NONE_OPTION, Router, build_state
from agent_router.core.types import (
    NONE_ID,
    ChoiceResult,
    HookPoint,
    OptionSpec,
    RouterEvent,
    public_stages,
)
from agent_router.deciders import registry
from agent_router.deciders.base import recommended_threshold

log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_AUDIT_DIR = Path(".agent-router/audit")
SAMPLE_AUDIT_DIR = REPO_ROOT / "audit"
PLAYGROUND_SESSION = "playground"
SKILL_TOOL = "Skill"
ALLOWED_HOSTS = ("127.0.0.1", "localhost")
RUN_TOKEN_TTL = 60.0
WORKSPACE_WAIT = 30.0  # seconds to let a cancelled run wind down before cleanup
_SESSION_RE = re.compile(r"^[A-Za-z0-9._-]{1,128}$")

Listener = Callable[[dict[str, Any]], None]
Runner = Callable[[str, Router, Path, Listener], Awaitable[Any]]


class RouteRequest(BaseModel):
    text: str = Field(default="", max_length=20_000)
    point: str = "prompt"
    tool_name: str | None = None
    tool_input: dict[str, Any] | None = None
    backend: str | None = None  # None: ``default_backend()``
    mode: str | None = None
    threshold: float | None = None  # None: the backend's calibrated threshold (or env/default)


class RunRequest(BaseModel):
    prompt: str = Field(max_length=20_000)
    mode: str = "advisory"
    backend: str | None = None  # None: ``default_backend()``


class _LockedDecider:
    """Serialises ``decide`` on one shared decider (caches inside are not thread-safe)."""

    def __init__(self, decider: Any) -> None:
        self._decider = decider
        self._lock = threading.Lock()
        self.name = getattr(decider, "name", "")
        self.recommended_threshold = recommended_threshold(decider)

    def decide(self, state: str, options: dict[str, OptionSpec]) -> ChoiceResult:
        with self._lock:
            return self._decider.decide(state, options)


def _entry_json(e: CatalogEntry) -> dict[str, Any]:
    return {
        "id": e.id,
        "kind": e.kind,
        "name": e.name,
        "project": e.project,
        "license": e.license,
        "url": e.url,
        "what": e.what,
        "target": e.target,
        "points": [str(p) for p in e.points],
        "replaces": list(e.replaces),
        "not_for": list(e.not_for),
        "examples": list(e.examples),
        "agents": list(e.agents),
        "threshold": e.threshold,
    }


NONE_JSON = {
    "id": NONE_ID,
    "kind": "none",
    "name": "Native path",
    "project": "the agent's own tools",
    "what": NONE_OPTION.what,
}


def _option_json(e: CatalogEntry) -> dict[str, Any]:
    return {
        k: v for k, v in _entry_json(e).items() if k in ("id", "kind", "name", "project", "what")
    }


def _default_runner(allow_shell: bool) -> Runner:
    from agent_router.agent import run_agent

    async def run(prompt: str, router: Router, workspace: Path, on_event: Listener) -> Any:
        return await run_agent(prompt, router, workspace, on_event, allow_shell=allow_shell)

    return run


DECIDER_ERROR = "decider error"  # the router's fail-open reason prefix


def _public_reason(reason: str) -> str:
    """A decider error keeps only its exception type: the rest may be a provider's body."""
    if reason.startswith(DECIDER_ERROR):
        kind = reason[len(DECIDER_ERROR) :].lstrip(": ").split(":", 1)[0].strip()
        return f"{DECIDER_ERROR}: {kind}" if kind else DECIDER_ERROR
    return reason


def _load_base_config() -> RouterConfig:
    """Environment config, validated once at startup with a readable error."""
    try:
        return RouterConfig.from_env()
    except ValueError as exc:
        raise ValueError(f"invalid AGENT_ROUTER_* environment for the demo server: {exc}") from exc


def _prepare_workspace() -> Path:
    from agent_router.agent import prepare_workspace

    return prepare_workspace()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:  # a line still being written during a live run
                continue
            if isinstance(rec, dict):
                out.append(rec)
    return out


def create_app(
    audit_dir: Path | None = None,
    *,
    runner: Runner | None = None,
    catalog: Catalog | None = None,
    allow_shell: bool = False,
    trace_dir: Path | None = None,
) -> FastAPI:
    """The demo app. ``runner`` defaults to ``agent_router.agent.run_agent`` (tests inject one);
    ``allow_shell`` lets live runs use Bash/WebFetch without the SDK refusing them.
    ``trace_dir`` holds decision traces (``<session>.jsonl``, written by an integration's
    record-only hooks, e.g. integrations/first-principles/trace.py); it defaults to the
    ``trace`` folder next to the audit folder, where those hooks put it."""
    audit_root = Path(audit_dir) if audit_dir is not None else DEFAULT_AUDIT_DIR
    trace_root = Path(trace_dir) if trace_dir is not None else audit_root.parent / "trace"
    # With the default location, the committed sample session is offered for replay too.
    read_dirs = [audit_root] + ([SAMPLE_AUDIT_DIR] if audit_dir is None else [])
    cat = catalog if catalog is not None else load_catalog()
    base_config = _load_base_config()
    env_threshold = bool(os.environ.get("AGENT_ROUTER_THRESHOLD", "").strip())
    run_fn = runner if runner is not None else _default_runner(allow_shell)
    run_slot = asyncio.Semaphore(1)  # one paid live run at a time
    tokens: dict[str, tuple[float, dict[str, Any]]] = {}
    deciders: dict[str, _LockedDecider] = {}
    deciders_lock = threading.Lock()

    app = FastAPI(title="agent-router demo")
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(ALLOWED_HOSTS))

    @app.middleware("http")
    async def revalidate_assets(request: Request, call_next: Callable[[Request], Awaitable[Any]]):
        # Without Cache-Control a browser may reuse a cached app.js next to a newer index.html
        # (heuristic freshness), and a tab the old script does not know silently does nothing.
        # no-cache = ask every time; unchanged files still come back as 304 via their ETag.
        response = await call_next(request)
        if request.url.path == "/" or request.url.path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-cache"
        return response

    app.state.run_slot = run_slot

    # -- helpers --------------------------------------------------------------

    def check_backend(name: str) -> None:
        available = registry.available_backends()
        if name not in available:
            raise HTTPException(400, f"unknown backend {name!r}; known: {sorted(available)}")
        if not available[name]:
            raise HTTPException(
                400, f"backend {name!r} is not available here (missing package or API key)"
            )

    def decider_for(name: str) -> _LockedDecider:
        with deciders_lock:
            if name not in deciders:
                deciders[name] = _LockedDecider(registry.make_decider(name, cat))
            return deciders[name]

    def config_for(mode: str | None, threshold: float | None, decider: Any = None) -> RouterConfig:
        """Request threshold, else ``AGENT_ROUTER_THRESHOLD``, else the decider's calibrated
        one, else the default."""
        mode = mode if mode is not None else base_config.mode
        if mode not in MODES:
            raise HTTPException(400, f"mode must be one of {MODES}")
        thr = threshold
        if thr is None and not env_threshold:
            thr = recommended_threshold(decider)
        if thr is None:
            thr = base_config.threshold
        if not 0.0 <= thr <= 1.0:
            raise HTTPException(400, "threshold must be in [0, 1]")
        return RouterConfig(
            mode=mode,  # type: ignore[arg-type]
            threshold=thr,
            enabled=base_config.enabled,  # AGENT_ROUTER_DISABLED
        )

    def check_same_origin(request: Request) -> None:
        """Refuse run requests another site (or another localhost port) triggered."""
        fetch_site = request.headers.get("sec-fetch-site")
        if fetch_site is not None and fetch_site not in ("same-origin", "none"):
            raise HTTPException(403, "live runs must be started from the demo page itself")
        origin = request.headers.get("origin")
        if origin is not None and origin != f"{request.url.scheme}://{request.headers.get('host')}":
            raise HTTPException(403, "live runs must be started from the demo page itself")

    def point_of(raw: str) -> HookPoint:
        try:
            return HookPoint(raw)
        except ValueError:
            raise HTTPException(
                400, f"point must be one of {[str(p) for p in HookPoint]}"
            ) from None

    def find_session(name: str) -> Path:
        if not _SESSION_RE.match(name):
            raise HTTPException(400, "invalid session name")
        for d in read_dirs:
            path = d / f"{name}.jsonl"
            if path.is_file():
                return path
        raise HTTPException(404, f"no audit session {name!r}")

    def restore_hint(rec: dict[str, Any]) -> dict[str, Any]:
        """The audit keeps 300 chars of the hint; re-render the full templated text."""
        rec = dict(rec)
        if isinstance(rec.get("reason"), str):  # the replay UI shows it, like the playground
            rec["reason"] = _public_reason(rec["reason"])
        rec["hint_restored"] = False
        hint, entry_id = rec.get("hint"), rec.get("entry_id")
        entry = cat.get(entry_id) if isinstance(entry_id, str) else None
        if not hint or entry is None:
            return rec
        try:
            point = HookPoint(rec.get("point"))
        except ValueError:
            return rec
        if rec.get("action") == "enforce":
            full = render_deny(entry, str(rec.get("tool_name") or ""))
        else:
            prob = float((rec.get("probabilities") or {}).get(entry.id, 0.0))
            full = render_hint(entry, point, prob)
        if full != hint and full.startswith(hint):
            rec["hint"] = full
            rec["hint_restored"] = True
        return rec

    # -- routes ---------------------------------------------------------------

    @app.get("/api/catalog")
    def get_catalog() -> dict[str, Any]:
        return {
            "version": cat.version,
            "entries": [_entry_json(e) for e in cat.entries],
            "none": NONE_JSON,
            "native_examples": list(cat.native_examples),
            "threshold": base_config.threshold,
            "mode": base_config.mode,
            "live": {"allow_shell": allow_shell},
        }

    @app.get("/api/backends")
    def get_backends() -> dict[str, Any]:
        available = registry.available_backends()
        return {
            "backends": [{"name": k, "available": bool(v)} for k, v in available.items()],
            "default": registry.default_backend(),
        }

    @app.post("/api/route")
    def post_route(req: RouteRequest) -> dict[str, Any]:
        # sync def: FastAPI runs it in the threadpool, so jev/openrouter never block the loop
        point = point_of(req.point)
        config_for(req.mode, req.threshold)  # validate before building a backend
        backend = req.backend or registry.default_backend()
        tool_name = req.tool_name or None
        if point == HookPoint.SKILL and not tool_name:
            tool_name = SKILL_TOOL
        if point == HookPoint.TOOL and not tool_name:
            raise HTTPException(400, "tool_name is required at the tool hook point")
        check_backend(backend)
        try:
            decider = decider_for(backend)
        except Exception as exc:
            raise HTTPException(503, f"could not start backend {backend!r}: {exc}") from exc
        config = config_for(req.mode, req.threshold, decider)
        event = RouterEvent(
            point=point,
            session_id=PLAYGROUND_SESSION,
            turn_id=1,
            text=req.text,
            tool_name=tool_name if point != HookPoint.PROMPT else None,
            tool_input=req.tool_input if point != HookPoint.PROMPT else None,
        )
        router = Router(cat, decider, config, AuditLog(None))  # fresh: no dedupe across calls
        start = time.perf_counter()
        decision = router.route(event)
        latency = (time.perf_counter() - start) * 1000.0
        eligible = cat.eligible(point, event.tool_name)
        res = decision.result
        return {
            "decision": {
                "action": str(decision.action),
                "reason": _public_reason(decision.reason),
                "entry_id": decision.entry_id,
                "applied_threshold": decision.threshold,
            },
            "result": None
            if res is None
            else {
                "choice": res.choice,
                "probabilities": {k: float(v) for k, v in res.probabilities.items()},
                "confidence": float(res.confidence),
                "backend": res.backend,
                "latency_ms": float(res.latency_ms),
                "stages": public_stages(res.stages),  # no exception text
            },
            "options": [_option_json(e) for e in eligible] + [NONE_JSON],
            "hint": decision.hint,
            "latency_ms": latency,
            "state": build_state(event),
            "point": str(point),
            "tool_name": event.tool_name,
            "threshold": config.threshold,
            "mode": config.mode,
            "backend": backend,
        }

    @app.post("/api/run")
    async def post_run(req: RunRequest, request: Request) -> dict[str, Any]:
        check_same_origin(request)
        if not req.prompt.strip():
            raise HTTPException(400, "prompt is empty")
        config_for(req.mode, None)
        req.backend = req.backend or registry.default_backend()
        check_backend(req.backend)
        if run_slot.locked():
            raise HTTPException(409, "another live run is in progress; wait for it to finish")
        now = time.monotonic()
        for tok in [t for t, (exp, _) in tokens.items() if exp < now]:
            del tokens[tok]
        token = secrets.token_urlsafe(24)
        tokens[token] = (now + RUN_TOKEN_TTL, req.model_dump())
        return {"token": token, "stream": f"/api/run/stream?token={token}"}

    @app.get("/api/run/stream")
    async def get_run_stream(token: str, request: Request) -> StreamingResponse:
        check_same_origin(request)
        entry = tokens.pop(token, None)  # single use
        if entry is None or entry[0] < time.monotonic():
            raise HTTPException(403, "unknown or expired run token; start the run again")
        params = entry[1]
        if run_slot.locked():
            raise HTTPException(409, "another live run is in progress; wait for it to finish")
        backend = params["backend"]
        try:
            decider = await run_in_threadpool(decider_for, backend)
        except Exception as exc:
            raise HTTPException(503, f"could not start backend {backend!r}: {exc}") from exc
        config = config_for(params["mode"], None, decider)
        session = f"{datetime.now():%Y%m%d-%H%M%S}-{secrets.token_hex(3)}"
        router = Router(cat, decider, config, AuditLog(audit_root / f"{session}.jsonl"))
        return StreamingResponse(
            _stream(run_fn, params["prompt"], router, session, config, backend, run_slot),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/audit/sessions")
    def get_sessions() -> dict[str, Any]:
        seen: set[str] = set()
        rows: list[dict[str, Any]] = []
        for d in read_dirs:
            if not d.is_dir():
                continue
            for path in d.glob("*.jsonl"):
                if path.stem in seen or not _SESSION_RE.match(path.stem):
                    continue
                seen.add(path.stem)
                try:
                    records = _read_jsonl(path)
                except OSError:
                    continue
                first = records[0] if records else {}
                rows.append(
                    {
                        "session": path.stem,
                        "records": len(records),
                        "mtime": path.stat().st_mtime,
                        "first_text": first.get("text"),
                        "mode": (first.get("thresholds") or {}).get("mode"),
                        "backend": first.get("backend"),
                        "sample": d != audit_root,
                    }
                )
        rows.sort(key=lambda r: r["mtime"], reverse=True)
        return {"sessions": rows}

    @app.get("/api/audit/{session}")
    def get_session(session: str) -> dict[str, Any]:
        path = find_session(session)
        return {"session": session, "records": [restore_hint(r) for r in _read_jsonl(path)]}

    # -- decision traces (read-only) --------------------------------------------

    def trace_path(name: str) -> Path:
        if not _SESSION_RE.match(name):
            raise HTTPException(400, "invalid session name")
        path = trace_root / f"{name}.jsonl"
        if not path.is_file():
            raise HTTPException(404, f"no trace for session {name!r}")
        return path

    @app.get("/api/trace/sessions")
    def get_trace_sessions() -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        if trace_root.is_dir():
            for path in trace_root.glob("*.jsonl"):
                if not _SESSION_RE.match(path.stem):
                    continue
                try:
                    records = _read_jsonl(path)
                except OSError:
                    continue
                title = next(
                    (
                        r["headings"][0]
                        for r in records
                        if r.get("kind") == "section_written" and r.get("headings")
                    ),
                    None,
                )
                rows.append(
                    {
                        "session": path.stem,
                        "records": len(records),
                        "runs": sum(r.get("kind") == "run_start" for r in records),
                        "finished": sum(r.get("kind") == "run_end" for r in records),
                        "title": title,
                        "mtime": path.stat().st_mtime,
                    }
                )
        rows.sort(key=lambda r: r["mtime"], reverse=True)
        return {"sessions": rows, "trace_dir": str(trace_root)}

    @app.get("/api/trace/{session}")
    def get_trace(session: str) -> dict[str, Any]:
        records = _read_jsonl(trace_path(session))
        audit = next(
            (d / f"{session}.jsonl" for d in read_dirs if (d / f"{session}.jsonl").is_file()), None
        )
        return {
            "session": session,
            "records": records,
            # the router's own decisions in the same session, to interleave with the trace
            "audit": [restore_hint(r) for r in _read_jsonl(audit)] if audit else [],
        }

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"


async def _stream(
    run: Runner,
    prompt: str,
    router: Router,
    session: str,
    config: RouterConfig,
    backend: str,
    slot: asyncio.Semaphore,
) -> AsyncIterator[str]:
    if slot.locked():  # lost a race with another stream
        yield _sse("error", {"type": "error", "ts": time.time(), "message": "another run is live"})
        yield _sse("done", {"session": None, "ts": time.time()})
        return
    async with slot:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        workspace: Path | None = None
        task: asyncio.Task[None] | None = None

        def on_event(event: dict[str, Any]) -> None:
            # safe from the loop thread and from worker threads alike
            loop.call_soon_threadsafe(queue.put_nowait, event)

        async def drive(ws: Path) -> None:
            try:
                await run(prompt, router, ws, on_event)
            except Exception as exc:  # run_agent emits ``error`` then re-raises
                log.warning("demo run %s failed: %s", session, exc)
                on_event({"type": "error", "ts": time.time(), "message": type(exc).__name__})
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        try:
            yield _sse(
                "session",
                {
                    "session": session,
                    "mode": config.mode,
                    "threshold": config.threshold,
                    "backend": backend,
                    "ts": time.time(),
                },
            )
            workspace = await run_in_threadpool(_prepare_workspace)
            task = asyncio.create_task(drive(workspace))
            errored = False
            while True:
                event = await queue.get()
                if event is None:
                    break
                kind = str(event.get("type", "message"))
                if kind == "error":
                    if errored:
                        continue  # the runner's own error event already went out
                    errored = True
                yield _sse(kind, event)
            yield _sse("done", {"session": session, "ts": time.time()})
        finally:
            # We may be here because the client disconnected (cancellation): shield cleanup.
            with anyio.CancelScope(shield=True):
                await _finish(task, workspace, session)


async def _finish(task: asyncio.Task[None] | None, workspace: Path | None, session: str) -> None:
    """Cancel the run, wait for it to stop, then remove its workspace (never under it)."""
    if task is not None and not task.done():
        task.cancel()
        done, _ = await asyncio.wait([task], timeout=WORKSPACE_WAIT)
        if not done:
            log.warning("demo run %s did not stop within %.0fs", session, WORKSPACE_WAIT)
            if workspace is not None:
                ws_root = workspace.parent
                task.add_done_callback(lambda _t: shutil.rmtree(ws_root, ignore_errors=True))
            return
    if workspace is not None:
        shutil.rmtree(workspace.parent, ignore_errors=True)


def main(
    port: int = 8765,
    allow_shell: bool = False,
    *,
    catalog: Catalog | None = None,
    audit_dir: Path | None = None,
    trace_dir: Path | None = None,
) -> None:
    """Serve the demo on http://127.0.0.1:<port> (loopback only).

    ``catalog`` / ``audit_dir`` point the demo at another catalog and another audit folder,
    e.g. an integration's catalog and the audit logs its Claude Code hooks wrote.
    """
    import uvicorn

    note = " (live runs may use Bash/WebFetch)" if allow_shell else ""
    print(f"agent-router demo: http://127.0.0.1:{port}{note}")
    if catalog is not None:
        print(f"  catalog {catalog.version} ({len(catalog.entries)} entries)")
    if audit_dir is not None:
        print(f"  replaying audit logs from {audit_dir}")
    if trace_dir is not None:
        print(f"  decision traces from {trace_dir}")
    app = create_app(audit_dir, allow_shell=allow_shell, catalog=catalog, trace_dir=trace_dir)
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="agent-router visual demo")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--allow-shell", action="store_true", help="auto-approve Bash/WebFetch in live runs"
    )
    ns = parser.parse_args()
    main(port=ns.port, allow_shell=ns.allow_shell)
