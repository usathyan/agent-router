import json

import pytest

from agent_router.core.audit import AuditLog, read_audit
from agent_router.core.config import RouterConfig
from agent_router.core.types import (
    Action,
    ChoiceResult,
    Decision,
    HookPoint,
    RouterEvent,
)

EVENT = RouterEvent(
    point=HookPoint.TOOL,
    session_id="s1",
    turn_id=3,
    text="y" * 1000,
    tool_name="Bash",
    tool_input={"command": "bc"},
)
RESULT = ChoiceResult(
    choice="exact-calc",
    probabilities={"exact-calc": 0.8, "none": 0.2},
    confidence=0.4,
    backend="fake",
    latency_ms=1.5,
)
DECISION = Decision(
    action=Action.SUGGEST,
    reason="match",
    entry_id="exact-calc",
    hint="h",
    result=RESULT,
    options=("exact-calc", "none"),
)

KEYS = {
    "ts",
    "session",
    "turn",
    "point",
    "tool_name",
    "agent_type",
    "tool_use_id",
    "agent_id",
    "state_sha256",
    "text",
    "options",
    "probabilities",
    "choice",
    "confidence",
    "action",
    "reason",
    "entry_id",
    "applied_threshold",
    "hint",
    "backend",
    "latency_ms",
    "stages",
    "catalog_version",
    "thresholds",
}


def test_record_appends_json_line(tmp_path):
    path = tmp_path / "sub" / "audit.jsonl"
    log = AuditLog(path)
    rec = log.record(EVENT, DECISION, "7", RouterConfig(), state="the state")
    assert set(rec) == KEYS
    assert rec["ts"].endswith("+00:00")
    assert len(rec["text"]) == 300
    assert rec["action"] == "suggest" and rec["point"] == "tool"
    assert rec["probabilities"] == {"exact-calc": 0.8, "none": 0.2}
    assert rec["thresholds"] == {"threshold": 0.5, "mode": "advisory"}
    assert len(rec["state_sha256"]) == 64
    lines = path.read_text().splitlines()
    assert len(lines) == 1 and json.loads(lines[0]) == rec
    assert read_audit(path) == [rec]


def test_skipped_record_has_same_schema():
    log = AuditLog(None)
    rec = log.record(EVENT, Decision(Action.SKIPPED, "disabled"), "7", RouterConfig())
    assert set(rec) == KEYS
    assert rec["probabilities"] == {} and rec["choice"] is None
    assert rec["stages"] == []
    json.dumps(rec)


def test_memory_log_keeps_records_and_notifies():
    log = AuditLog(None)
    seen = []

    def boom(rec):
        raise RuntimeError("subscriber failure")

    log.subscribe(boom)
    log.subscribe(seen.append)
    log.record(EVENT, DECISION, "7", RouterConfig())
    log.record(EVENT, DECISION, "7", RouterConfig())
    assert len(log.records) == 2
    assert len(seen) == 2


def test_read_audit_skips_blank_lines(tmp_path):
    p = tmp_path / "a.jsonl"
    p.write_text('{"a": 1}\n\n{"a": 2}\n')
    assert read_audit(p) == [{"a": 1}, {"a": 2}]


def test_config_from_env(monkeypatch):
    for k in (
        "AGENT_ROUTER_MODE",
        "AGENT_ROUTER_THRESHOLD",
        "AGENT_ROUTER_DISABLED",
        "AGENT_ROUTER_AUDIT",
    ):
        monkeypatch.delenv(k, raising=False)
    cfg = RouterConfig.from_env()
    assert cfg.mode == "advisory" and cfg.threshold == 0.5 and cfg.enabled
    assert cfg.audit_path is None and set(cfg.points) == set(HookPoint)

    monkeypatch.setenv("AGENT_ROUTER_MODE", "enforce")
    monkeypatch.setenv("AGENT_ROUTER_THRESHOLD", "0.7")
    monkeypatch.setenv("AGENT_ROUTER_DISABLED", "true")
    monkeypatch.setenv("AGENT_ROUTER_AUDIT", "/tmp/x.jsonl")
    cfg = RouterConfig.from_env()
    assert cfg.mode == "enforce" and cfg.threshold == 0.7 and not cfg.enabled
    assert str(cfg.audit_path) == "/tmp/x.jsonl"

    monkeypatch.setenv("AGENT_ROUTER_DISABLED", "0")
    monkeypatch.setenv("AGENT_ROUTER_AUDIT", "")
    cfg = RouterConfig.from_env()
    assert cfg.enabled and cfg.audit_path is None


@pytest.mark.parametrize(
    ("var", "value"),
    [("AGENT_ROUTER_MODE", "strict"), ("AGENT_ROUTER_THRESHOLD", "abc")],
)
def test_config_from_env_rejects_bad_values(monkeypatch, var, value):
    monkeypatch.setenv(var, value)
    with pytest.raises(ValueError):
        RouterConfig.from_env()


def test_config_rejects_bad_threshold():
    with pytest.raises(ValueError):
        RouterConfig(threshold=1.5)


def test_record_write_error_is_logged_not_raised(tmp_path, caplog):
    log = AuditLog(tmp_path)  # path is a directory: open() raises IsADirectoryError
    seen = []
    log.subscribe(seen.append)
    rec = log.record(EVENT, DECISION, "7", RouterConfig())
    assert rec["action"] == "suggest" and seen == [rec]
    assert "audit write failed" in caplog.text
