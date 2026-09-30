"""Demo server API tests (FastAPI TestClient, hashing embedder, no network, no SDK)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from agent_router.core.catalog import load_catalog
from agent_router.core.hints import render_hint
from agent_router.core.types import HookPoint
from agent_router.demo import server


@pytest.fixture(autouse=True)
def _hashing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_ROUTER_EMBEDDER", "hashing")


@pytest.fixture
def audit_dir(tmp_path: Path) -> Path:
    d = tmp_path / "audit"
    d.mkdir()
    return d


@pytest.fixture
def client(audit_dir: Path) -> TestClient:
    return _client(server.create_app(audit_dir=audit_dir))


BASE = "http://127.0.0.1:8765"


def _client(app) -> TestClient:
    """The server only answers loopback Host headers."""
    return TestClient(app, base_url=BASE)


# -- catalog / backends -------------------------------------------------------


def test_catalog_lists_entries_none_and_config(client: TestClient) -> None:
    body = client.get("/api/catalog").json()
    ids = [e["id"] for e in body["entries"]]
    assert ids == [e.id for e in load_catalog().entries]
    first = body["entries"][0]
    assert {"id", "name", "project", "license", "what", "kind", "points", "replaces"} <= set(first)
    assert body["none"]["id"] == "none"
    assert 0.0 <= body["threshold"] <= 1.0
    assert body["mode"] in ("advisory", "enforce")
    assert body["version"]


def test_backends_follow_registry(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        server.registry, "available_backends", lambda: {"local": True, "extra": False}
    )
    body = client.get("/api/backends").json()
    assert body["default"] == "local"
    assert body["backends"] == [
        {"name": "local", "available": True},
        {"name": "extra", "available": False},
    ]


# -- route --------------------------------------------------------------------


def _route(client: TestClient, **payload):
    payload.setdefault("backend", "local")
    return client.post("/api/route", json=payload)


def test_route_prompt_returns_full_distribution(client: TestClient) -> None:
    res = _route(client, text="What is 17% of 2,340 exactly?", point="prompt")
    assert res.status_code == 200, res.text
    body = res.json()
    assert {"decision", "result", "options", "hint", "latency_ms", "state"} <= set(body)
    probs = body["result"]["probabilities"]
    option_ids = [o["id"] for o in body["options"]]
    assert option_ids[-1] == "none"
    assert set(probs) == set(option_ids)
    assert sum(probs.values()) == pytest.approx(1.0, abs=1e-6)
    assert body["result"]["choice"] in option_ids
    assert body["decision"]["action"] in ("suggest", "native", "enforce", "skipped")
    assert "reason" in body["decision"]
    assert body["latency_ms"] >= 0
    if body["decision"]["action"] == "suggest":
        assert body["hint"].startswith("[agent-router]")


def test_route_is_stateless_between_calls(client: TestClient) -> None:
    """Same input twice gives the same action (no 'already suggested this turn')."""
    a = _route(client, text="compute 2**200 exactly", point="prompt", threshold=0.0).json()
    b = _route(client, text="compute 2**200 exactly", point="prompt", threshold=0.0).json()
    assert a["decision"]["action"] == b["decision"]["action"] == "suggest"


class _StubDecider:
    name = "stub"

    def decide(self, state, options):
        from agent_router.core.types import ChoiceResult

        rest = (1.0 - 0.6) / (len(options) - 1)
        probs = {k: (0.6 if k == "exact-calc" else rest) for k in options}
        return ChoiceResult("exact-calc", probs, 0.4, backend="stub", latency_ms=1.0)


def test_route_below_threshold_is_native_but_keeps_entry(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(server.registry, "make_decider", lambda name, catalog: _StubDecider())
    body = _route(client, text="compute 2**200 exactly", point="prompt", threshold=0.7).json()
    assert body["decision"]["action"] == "native"
    assert body["decision"]["entry_id"] == "exact-calc"  # pointed, but gated
    assert body["hint"] is None
    assert body["threshold"] == 0.7
    low = _route(client, text="compute 2**200 exactly", point="prompt", threshold=0.5).json()
    assert low["decision"]["action"] == "suggest"


def test_route_tool_enforce_denies(client: TestClient) -> None:
    body = _route(
        client,
        text="compute 2**200 exactly",
        point="tool",
        tool_name="Bash",
        tool_input={"command": 'python3 -c "print(2**200)"'},
        mode="enforce",
        threshold=0.0,
    ).json()
    assert body["decision"]["action"] in ("enforce", "native")
    if body["decision"]["action"] == "enforce":
        assert body["hint"].startswith("[agent-router] Blocked Bash")
    # Bash-eligible entries only, plus none
    option_ids = {o["id"] for o in body["options"]}
    assert "commit-writer" not in option_ids
    assert "none" in option_ids


def test_route_skill_defaults_tool_name(client: TestClient) -> None:
    body = _route(
        client,
        text="write a commit message for adding retry logic",
        point="skill",
        tool_input={"skill": "release-notes"},
    ).json()
    assert body["decision"]["action"] != "skipped"
    assert [o["id"] for o in body["options"]] == ["commit-writer", "none"]


def test_route_ineligible_tool_is_skipped_without_result(client: TestClient) -> None:
    body = _route(client, text="edit file", point="tool", tool_name="Edit").json()
    assert body["decision"]["action"] == "skipped"
    assert body["result"] is None
    assert [o["id"] for o in body["options"]] == ["none"]


@pytest.mark.parametrize(
    "payload",
    [
        {"text": "x", "point": "nowhere"},
        {"text": "x", "point": "prompt", "mode": "loud"},
        {"text": "x", "point": "prompt", "threshold": 1.5},
        {"text": "x", "point": "prompt", "backend": "no-such-backend"},
        {"text": "x", "point": "tool"},
    ],
)
def test_route_rejects_bad_input(client: TestClient, payload: dict) -> None:
    res = client.post("/api/route", json={"backend": "local", **payload})
    assert 400 <= res.status_code < 500


def test_route_unavailable_backend_is_rejected(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        server.registry, "available_backends", lambda: {"local": True, "jev": False}
    )
    res = _route(client, text="x", point="prompt", backend="jev")
    assert res.status_code == 400
    assert "jev" in res.json()["detail"]


def test_decider_is_cached_per_backend(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    real = server.registry.make_decider

    def counting(name, catalog):
        calls.append(name)
        return real(name, catalog)

    monkeypatch.setattr(server.registry, "make_decider", counting)
    _route(client, text="a", point="prompt")
    _route(client, text="b", point="prompt")
    assert calls == ["local"]


# -- audit ----------------------------------------------------------------------


def _write_audit(audit_dir: Path, name: str, records: list[dict], junk: bool = False) -> Path:
    path = audit_dir / f"{name}.jsonl"
    lines = [json.dumps(r) for r in records]
    if junk:
        lines.append('{"half written')
    path.write_text("\n".join(lines) + "\n")
    return path


def _rec(**kw) -> dict:
    base = {
        "ts": "2026-09-26T10:00:00+00:00",
        "session": "sdk-1",
        "turn": 1,
        "point": "prompt",
        "tool_name": None,
        "text": "What is 17% of 2,340 exactly?",
        "options": ["exact-calc", "none"],
        "probabilities": {"exact-calc": 0.9, "none": 0.1},
        "choice": "exact-calc",
        "confidence": 0.5,
        "action": "native",
        "reason": "x",
        "entry_id": None,
        "hint": None,
        "backend": "local",
        "latency_ms": 3.0,
        "catalog_version": "v",
        "thresholds": {"threshold": 0.5, "mode": "advisory"},
    }
    return base | kw


def test_audit_sessions_and_records(client: TestClient, audit_dir: Path) -> None:
    _write_audit(audit_dir, "s1", [_rec(), _rec(turn=2)], junk=True)
    sessions = client.get("/api/audit/sessions").json()["sessions"]
    assert [s["session"] for s in sessions] == ["s1"]
    assert sessions[0]["records"] == 2
    assert sessions[0]["first_text"] == "What is 17% of 2,340 exactly?"

    records = client.get("/api/audit/s1").json()["records"]
    assert [r["turn"] for r in records] == [1, 2]


def test_audit_restores_truncated_hint(client: TestClient, audit_dir: Path) -> None:
    entry = load_catalog().get("exact-calc")
    full = render_hint(entry, HookPoint.PROMPT, 0.9)
    assert len(full) > 300
    _write_audit(
        audit_dir,
        "s2",
        [_rec(action="suggest", entry_id="exact-calc", hint=full[:300])],
    )
    rec = client.get("/api/audit/s2").json()["records"][0]
    assert rec["hint"] == full
    assert rec["hint_restored"] is True


@pytest.mark.parametrize("name", ["..%2Fsecrets", "a b", "nope"])
def test_audit_unknown_or_bad_session_404(client: TestClient, name: str) -> None:
    assert client.get(f"/api/audit/{name}").status_code in (400, 404)


def test_page_and_assets_are_always_revalidated(client: TestClient) -> None:
    """A stale cached app.js next to a new index.html left the Trace tab dead on click."""
    for path in ("/", "/static/app.js", "/static/app.css"):
        res = client.get(path)
        assert res.status_code == 200 and res.headers["cache-control"] == "no-cache"
    etag = client.get("/static/app.js").headers["etag"]
    assert client.get("/static/app.js", headers={"If-None-Match": etag}).status_code == 304
    assert "cache-control" not in client.get("/api/catalog").headers


# -- decision traces ----------------------------------------------------------------


def _trace(kind: str, **kw) -> dict:
    return {"ts": "2026-09-29T10:00:00+00:00", "session": "s1", "agent_id": "a1", "kind": kind} | kw


def test_trace_sessions_default_to_the_folder_beside_the_audit_log(
    client: TestClient, audit_dir: Path
) -> None:
    trace_dir = audit_dir.parent / "trace"
    trace_dir.mkdir()
    records = [
        _trace("run_start"),
        _trace("section_written", headings=["First-Principles Analysis — test", "1. X"]),
        _trace("run_end", duration_s=12.5),
    ]
    _write_audit(trace_dir, "s1", records, junk=True)
    _write_audit(trace_dir, "bad name", records)
    body = client.get("/api/trace/sessions").json()
    assert body["trace_dir"] == str(trace_dir)
    (row,) = body["sessions"]
    assert row["session"] == "s1" and row["records"] == 3
    assert row["runs"] == 1 and row["finished"] == 1
    assert row["title"] == "First-Principles Analysis — test"


def test_trace_session_includes_the_routers_decisions(client: TestClient, audit_dir: Path) -> None:
    trace_dir = audit_dir.parent / "trace"
    trace_dir.mkdir()
    _write_audit(trace_dir, "s1", [_trace("run_start"), _trace("run_end")])
    _write_audit(audit_dir, "s1", [_rec(), _rec(turn=2)])
    body = client.get("/api/trace/s1").json()
    assert [r["kind"] for r in body["records"]] == ["run_start", "run_end"]
    assert len(body["audit"]) == 2
    _write_audit(trace_dir, "s2", [_trace("run_start")])
    assert client.get("/api/trace/s2").json()["audit"] == []  # a trace without an audit log


def test_trace_dir_can_be_set_and_missing_sessions_404(tmp_path: Path) -> None:
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    _write_audit(elsewhere, "t", [_trace("run_start")])
    c = _client(server.create_app(audit_dir=tmp_path / "audit", trace_dir=elsewhere))
    assert [r["session"] for r in c.get("/api/trace/sessions").json()["sessions"]] == ["t"]
    assert c.get("/api/trace/nope").status_code == 404
    assert c.get("/api/trace/..%2Fsecrets").status_code in (400, 404)
    empty = _client(server.create_app(audit_dir=tmp_path / "audit", trace_dir=tmp_path / "none"))
    assert empty.get("/api/trace/sessions").json()["sessions"] == []


# -- live run (SSE, fake runner) ------------------------------------------------


def _sse_events(text: str) -> list[tuple[str, dict]]:
    out = []
    for block in text.strip().split("\n\n"):
        ev, data = "message", ""
        for line in block.splitlines():
            if line.startswith("event: "):
                ev = line[7:]
            elif line.startswith("data: "):
                data += line[6:]
        out.append((ev, json.loads(data) if data else {}))
    return out


def _start(c: TestClient, **body):
    body.setdefault("prompt", "hi")
    return c.post("/api/run", json=body)


def _run(c: TestClient, **body):
    res = _start(c, **body)
    assert res.status_code == 200, res.text
    return c.get(res.json()["stream"])


def test_run_streams_events_and_writes_audit(audit_dir: Path) -> None:
    seen: dict = {}

    async def fake_runner(prompt, router, workspace, on_event):
        seen["workspace"] = workspace
        seen["mode"] = router.config.mode
        on_event({"type": "prompt", "ts": 1.0, "prompt": prompt})
        on_event({"type": "tool_use", "ts": 2.0, "id": "t1", "name": "Bash", "input": {}})
        from agent_router.core.types import HookPoint, RouterEvent

        router.route(RouterEvent(HookPoint.PROMPT, "sdk", 1, prompt))
        on_event({"type": "result", "ts": 3.0, "result": "42", "is_error": False})
        return "42"

    app = server.create_app(audit_dir=audit_dir, runner=fake_runner)
    with _client(app) as c:
        res = _run(c, prompt="hi there", mode="enforce")
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("text/event-stream")
    events = _sse_events(res.text)
    types = [e for e, _ in events]
    assert types[0] == "session"
    assert types[-1] == "done"
    assert "tool_use" in types and "result" in types
    session = events[0][1]["session"]
    assert seen["mode"] == "enforce"
    assert not seen["workspace"].exists()  # temp workspace cleaned up
    audit_file = audit_dir / f"{session}.jsonl"
    assert audit_file.exists()
    assert len(audit_file.read_text().splitlines()) == 1


def test_run_error_still_sends_done(audit_dir: Path) -> None:
    async def failing(prompt, router, workspace, on_event):
        on_event({"type": "error", "ts": 1.0, "message": "RuntimeError: boom"})
        raise RuntimeError("boom")

    app = server.create_app(audit_dir=audit_dir, runner=failing)
    with _client(app) as c:
        res = _run(c)
    types = [e for e, _ in _sse_events(res.text)]
    assert "error" in types
    assert types[-1] == "done"


@pytest.mark.parametrize(
    "body", [{"mode": "loud"}, {"prompt": "  "}, {"backend": "no-such-backend"}]
)
def test_run_rejects_bad_input(client: TestClient, body: dict) -> None:
    assert _start(client, **body).status_code == 400


def test_index_served(client: TestClient) -> None:
    res = client.get("/")
    assert res.status_code == 200
    assert "agent-router" in res.text


# -- live-run lockdown -----------------------------------------------------------


async def _never(prompt, router, workspace, on_event):  # pragma: no cover
    raise AssertionError("must not run")


def test_foreign_host_is_rejected(audit_dir: Path) -> None:
    """DNS rebinding: evil.example resolving to 127.0.0.1 still sends its own Host."""
    app = server.create_app(audit_dir=audit_dir, runner=_never)
    c = TestClient(app, base_url="http://evil.example:8765")
    assert c.post("/api/run", json={"prompt": "hi"}).status_code == 400
    assert c.get("/api/audit/sessions").status_code == 400
    assert TestClient(app, base_url="http://localhost:8765").get("/api/catalog").status_code == 200


@pytest.mark.parametrize(
    "headers",
    [
        {"Sec-Fetch-Site": "cross-site"},
        {"Sec-Fetch-Site": "same-site"},  # another localhost port
        {"Origin": "http://localhost:3000"},
        {"Origin": "http://evil.example"},
    ],
)
def test_run_refuses_other_origins(audit_dir: Path, headers: dict) -> None:
    c = _client(server.create_app(audit_dir=audit_dir, runner=_never))
    assert c.post("/api/run", json={"prompt": "hi"}, headers=headers).status_code == 403


def test_run_accepts_same_origin_headers(client: TestClient) -> None:
    headers = {"Sec-Fetch-Site": "same-origin", "Origin": BASE}
    assert client.post("/api/run", json={"prompt": "hi"}, headers=headers).status_code == 200


def test_run_token_is_single_use(audit_dir: Path) -> None:
    async def quick(prompt, router, workspace, on_event):
        on_event({"type": "result", "ts": 1.0, "result": "ok", "is_error": False})

    app = server.create_app(audit_dir=audit_dir, runner=quick)
    with _client(app) as c:
        stream = _start(c).json()["stream"]
        assert c.get(stream).status_code == 200
        assert c.get(stream).status_code == 403
        assert c.get("/api/run/stream", params={"token": "forged"}).status_code == 403
        assert c.get("/api/run", params={"prompt": "hi"}).status_code == 405  # no GET start


def test_run_token_expires(audit_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    c = _client(server.create_app(audit_dir=audit_dir, runner=_never))
    stream = _start(c).json()["stream"]
    real = server.time.monotonic
    monkeypatch.setattr(server.time, "monotonic", lambda: real() + server.RUN_TOKEN_TTL + 1)
    assert c.get(stream).status_code == 403


def test_bad_threshold_env_fails_at_startup(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_ROUTER_THRESHOLD", "2")
    with pytest.raises(ValueError, match="AGENT_ROUTER_"):
        server.create_app()


def test_catalog_reports_shell_setting(audit_dir: Path) -> None:
    off = _client(server.create_app(audit_dir=audit_dir)).get("/api/catalog").json()
    on = _client(server.create_app(audit_dir=audit_dir, allow_shell=True)).get("/api/catalog")
    assert off["live"] == {"allow_shell": False}
    assert on.json()["live"] == {"allow_shell": True}


def test_default_runner_passes_allow_shell(monkeypatch: pytest.MonkeyPatch) -> None:
    import asyncio

    import agent_router.agent as agent_mod

    seen = {}

    async def fake_run_agent(prompt, router, workspace, on_event, *, allow_shell=False):
        seen["allow_shell"] = allow_shell

    monkeypatch.setattr(agent_mod, "run_agent", fake_run_agent)
    for flag in (False, True):
        asyncio.run(server._default_runner(flag)("p", None, Path("."), lambda e: None))
        assert seen["allow_shell"] is flag


# -- stream lifecycle (driven directly: disconnect == closing the generator) --------


def _router(audit_dir: Path):
    from agent_router.core.audit import AuditLog
    from agent_router.core.config import RouterConfig
    from agent_router.core.router import Router

    return Router(load_catalog(), _StubDecider(), RouterConfig(), AuditLog(None))


async def test_disconnect_mid_stream_cancels_run_and_removes_workspace(audit_dir: Path) -> None:
    import asyncio

    seen: dict = {}

    async def hanging(prompt, router, workspace, on_event):
        seen["workspace"] = workspace
        on_event({"type": "prompt", "ts": 1.0, "prompt": prompt})
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            seen["cancelled"] = True
            raise

    slot = asyncio.Semaphore(1)
    cfg = _router(audit_dir).config
    gen = server._stream(hanging, "hi", _router(audit_dir), "s1", cfg, "local", slot)
    assert (await gen.__anext__()).startswith("event: session")
    assert (await gen.__anext__()).startswith("event: prompt")
    assert slot.locked()
    await gen.aclose()  # what Starlette does when the client goes away
    assert seen["cancelled"] is True
    assert not seen["workspace"].parent.exists()
    assert not slot.locked()


async def test_disconnect_before_workspace_leaks_nothing(
    audit_dir: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import asyncio

    made: list[Path] = []
    real = server._prepare_workspace

    def tracking():
        ws = real()
        made.append(ws)
        return ws

    monkeypatch.setattr(server, "_prepare_workspace", tracking)
    slot = asyncio.Semaphore(1)
    gen = server._stream(
        _never, "hi", _router(audit_dir), "s2", None or _router(audit_dir).config, "local", slot
    )
    await gen.__anext__()  # session event, workspace not created yet
    await gen.aclose()
    assert made == []
    assert not slot.locked()


async def test_second_concurrent_run_is_refused(audit_dir: Path) -> None:
    import asyncio

    slot = asyncio.Semaphore(1)
    await slot.acquire()
    gen = server._stream(
        _never, "hi", _router(audit_dir), "s3", _router(audit_dir).config, "local", slot
    )
    chunks = [c async for c in gen]
    assert chunks[0].startswith("event: error") and chunks[-1].startswith("event: done")
    slot.release()


def test_run_while_busy_is_409(audit_dir: Path) -> None:
    import asyncio

    app = server.create_app(audit_dir=audit_dir, runner=_never)
    c = _client(app)
    stream = _start(c).json()["stream"]  # minted while idle
    asyncio.run(app.state.run_slot.acquire())  # a live run holds the slot
    try:
        assert _start(c).status_code == 409
        assert c.get(stream).status_code == 409
    finally:
        app.state.run_slot.release()
    assert _start(c).status_code == 200


# -- default backend, calibrated threshold, cascade stages --------------------------------


class _CalibratedStub(_StubDecider):
    name = "calibrated"
    recommended_threshold = 0.65


def test_backends_default_is_cascade_when_available(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        server.registry, "available_backends", lambda: {"cascade": True, "local": True}
    )
    assert client.get("/api/backends").json()["default"] == "cascade"


def test_route_without_backend_uses_default(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    built = []
    monkeypatch.setattr(server.registry, "default_backend", lambda: "local")
    monkeypatch.setattr(
        server.registry, "make_decider", lambda name, catalog: built.append(name) or _StubDecider()
    )
    body = client.post("/api/route", json={"text": "compute 2**200", "point": "prompt"}).json()
    assert body["backend"] == "local" and built == ["local"]


def test_route_without_threshold_uses_decider_calibration(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(server.registry, "make_decider", lambda name, catalog: _CalibratedStub())
    body = _route(client, text="compute 2**200 exactly", point="prompt", threshold=None).json()
    assert body["threshold"] == 0.65
    assert body["decision"]["action"] == "native"  # p=0.6 < 0.65
    explicit = _route(client, text="compute 2**200 exactly", point="prompt", threshold=0.5).json()
    assert explicit["threshold"] == 0.5


def test_env_threshold_beats_decider_calibration(
    audit_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AGENT_ROUTER_THRESHOLD", "0.3")
    monkeypatch.setattr(server.registry, "make_decider", lambda name, catalog: _CalibratedStub())
    c = _client(server.create_app(audit_dir=audit_dir))
    body = _route(c, text="compute 2**200 exactly", point="prompt").json()
    assert body["threshold"] == 0.3


def test_live_run_uses_decider_calibration(audit_dir: Path, monkeypatch) -> None:
    monkeypatch.setattr(server.registry, "make_decider", lambda name, catalog: _CalibratedStub())
    seen = {}

    async def runner(prompt, router, workspace, on_event):
        seen["threshold"] = router.config.threshold

    with _client(server.create_app(audit_dir=audit_dir, runner=runner)) as c:
        res = _run(c, prompt="hi", backend="local")
    events = _sse_events(res.text)
    assert events[0][1]["threshold"] == 0.65 and seen["threshold"] == 0.65


def test_route_returns_cascade_stages(client: TestClient, monkeypatch) -> None:
    from agent_router.deciders.cascade import CascadeDecider

    class Unsure(_StubDecider):
        name = "local"

    monkeypatch.setattr(
        server.registry,
        "make_decider",
        lambda name, catalog: CascadeDecider(Unsure(), _StubDecider()),
    )
    body = _route(client, text="compute 2**200 exactly", point="prompt", threshold=0.5).json()
    assert body["result"]["backend"] == "cascade:stub"
    stages = body["result"]["stages"]
    assert [s["role"] for s in stages] == ["primary", "confirm"]
    assert set(stages[0]["probabilities"]) == {o["id"] for o in body["options"]}


def test_route_stages_carry_no_error_text(client: TestClient, monkeypatch) -> None:
    from agent_router.deciders.base import DeciderError
    from agent_router.deciders.cascade import CascadeDecider

    class Down:
        name = "jev"

        def decide(self, state, options):
            raise DeciderError("Jev returned HTTP 500: <remote body SECRET>")

    monkeypatch.setattr(
        server.registry,
        "make_decider",
        lambda name, catalog: CascadeDecider(_StubDecider(), Down()),
    )
    res = _route(client, text="compute 2**200 exactly", point="prompt", threshold=0.5)
    assert "SECRET" not in res.text
    stage = res.json()["result"]["stages"][-1]
    assert stage["failed"] is True and stage["error_type"] == "DeciderError"


def test_env_disabled_skips_playground_and_live_runs(
    audit_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AGENT_ROUTER_DISABLED", "1")
    seen = {}

    async def runner(prompt, router, workspace, on_event):
        seen["enabled"] = router.config.enabled

    with _client(server.create_app(audit_dir=audit_dir, runner=runner)) as c:
        body = _route(c, text="compute 2**200 exactly", point="prompt").json()
        assert (body["decision"]["action"], body["decision"]["reason"]) == ("skipped", "disabled")
        _run(c, prompt="hi", backend="local")
    assert seen["enabled"] is False


class _FailingDecider:
    name = "failing"

    def decide(self, state, options):
        from agent_router.deciders.base import DeciderError

        raise DeciderError("Jev returned HTTP 500: <html>SECRET provider body</html>")


def test_route_decider_error_reason_hides_the_provider_body(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(server.registry, "make_decider", lambda name, catalog: _FailingDecider())
    body = _route(client, text="compute 2**200 exactly", point="prompt").json()
    assert body["decision"]["action"] == "native"
    assert body["decision"]["reason"] == "decider error: DeciderError"
    assert "SECRET" not in json.dumps(body)


def test_replay_decider_error_reason_hides_the_provider_body(
    client: TestClient, audit_dir: Path
) -> None:
    rec = {
        "session": "s-err",
        "turn": 1,
        "point": "prompt",
        "action": "native",
        "reason": "decider error: DeciderError: Jev returned HTTP 500: SECRET body",
    }
    (audit_dir / "s-err.jsonl").write_text(json.dumps(rec) + "\n")
    body = client.get("/api/audit/s-err").json()
    assert body["records"][0]["reason"] == "decider error: DeciderError"
