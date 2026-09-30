import json
import subprocess
import sys

import pytest

from agent_router import cli


@pytest.fixture(autouse=True)
def hashing(monkeypatch):
    monkeypatch.setenv("AGENT_ROUTER_EMBEDDER", "hashing")
    for name in ("MODE", "THRESHOLD", "DISABLED", "AUDIT"):
        monkeypatch.delenv(f"AGENT_ROUTER_{name}", raising=False)


def test_route_json_prompt(capsys):
    rc = cli.main(
        ["route", "compute 3**80 exactly", "--backend", "local", "--embedder", "hashing", "--json"]
    )
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["choice"] in {"exact-calc", "none"}
    assert out["action"] in {"suggest", "native"}
    assert set(out["probabilities"]) >= {"exact-calc", "none"}
    assert "threshold" in out and "backend" in out


def test_route_json_tool_point(capsys):
    rc = cli.main(
        [
            "route",
            "run the tests",
            "--point",
            "tool",
            "--tool",
            "Bash",
            "--input",
            '{"command": "pytest -q"}',
            "--embedder",
            "hashing",
            "--json",
        ]
    )
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert "choice" in out
    assert set(out["options"]) == {
        "exact-calc",
        "json-query",
        "html-to-markdown",
        "repo-stats",
        "none",
    }


def test_route_skipped_has_null_choice(capsys):
    rc = cli.main(
        ["route", "x", "--point", "tool", "--tool", "Edit", "--embedder", "hashing", "--json"]
    )
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["action"] == "skipped"
    assert out["choice"] is None


def test_route_text_output(capsys):
    assert cli.main(["route", "git status", "--embedder", "hashing"]) == 0
    assert "action" in capsys.readouterr().out


def test_route_bad_input_json(capsys):
    with pytest.raises(SystemExit):
        cli.main(["route", "x", "--point", "tool", "--tool", "Bash", "--input", "{bad"])


def test_unknown_backend_errors(capsys):
    assert cli.main(["route", "x", "--backend", "nope", "--embedder", "hashing"]) == 2
    assert "unknown backend" in capsys.readouterr().err


def test_unavailable_backend_errors(capsys, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    assert cli.main(["eval", "--backend", "jev"]) == 2
    assert "not available" in capsys.readouterr().err


def test_eval_json(capsys):
    rc = cli.main(["eval", "--backend", "local", "--embedder", "hashing", "--json"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["split"] == "test"
    assert {"accuracy", "fpr", "misroute_rate", "per_entry", "confusions", "cases"} <= set(out)
    assert out["n"] == len([c for c in out["cases"]])


def test_eval_text(capsys):
    assert cli.main(["eval", "--embedder", "hashing", "--split", "cal"]) == 0
    out = capsys.readouterr().out
    assert "accuracy" in out and "FPR" in out


def test_calibrate_writes_file(tmp_path, capsys):
    out = tmp_path / "cal.json"
    rc = cli.main(["calibrate", "--embedder", "hashing", "--out", str(out), "--quick"])
    assert rc == 0
    data = json.loads(out.read_text())
    assert "hashing" in data and "threshold" in data["hashing"]


def test_demo_and_run_missing_modules_are_reported(capsys, monkeypatch):
    monkeypatch.setitem(sys.modules, "agent_router.demo.server", None)
    monkeypatch.setitem(sys.modules, "agent_router.agent", None)
    assert cli.main(["demo", "--port", "8799"]) == 1
    assert "demo" in capsys.readouterr().err.lower()
    assert cli.main(["run", "hello"]) == 1
    assert "agent" in capsys.readouterr().err.lower()


def test_cli_import_is_light():
    code = (
        "import sys, agent_router.cli; "
        "heavy = ('claude_agent_sdk', 'fastapi', 'uvicorn', 'model2vec'); "
        "bad = [m for m in heavy if m in sys.modules]; "
        "print(bad)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"


def test_help_lists_subcommands(capsys):
    with pytest.raises(SystemExit):
        cli.main(["--help"])
    out = capsys.readouterr().out
    for sub in ("route", "eval", "calibrate", "demo", "run"):
        assert sub in out


# -- default backend, cascade, shell flags --------------------------------------------------


class _Always:
    def __init__(self, name, choice="none"):
        self.name = name
        self.choice = choice

    def decide(self, state, options):
        from agent_router.core.types import ChoiceResult

        probs = {o: (1.0 if o == self.choice else 0.0) for o in options}
        return ChoiceResult(self.choice, probs, 1.0, backend=self.name, latency_ms=1.0)


def _fake_cascade(monkeypatch):
    from agent_router.deciders import registry
    from agent_router.deciders.cascade import CascadeDecider

    monkeypatch.setenv("TYPESAFE_API_KEY", "ts-test")

    def make(name, catalog):
        if name == "cascade":  # local unsure -> escalates to a fake jev
            return CascadeDecider(_Always("local", "exact-calc"), _Always("jev", "exact-calc"))
        return _Always(name)

    monkeypatch.setattr(registry, "make_decider", make)


def test_backend_defaults_to_local_without_jev_key(capsys):
    assert cli.main(["route", "compute 3**80 exactly", "--embedder", "hashing", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["backend"] == "local"


def test_backend_defaults_to_cascade_with_jev_key(capsys, monkeypatch):
    _fake_cascade(monkeypatch)
    assert cli.main(["route", "compute 3**80 exactly", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["backend"] == "cascade:jev"
    assert [s["role"] for s in out["stages"]] == ["primary", "confirm"]
    assert cli.main(["route", "compute 3**80 exactly"]) == 0
    assert "escalated to jev" in capsys.readouterr().out


def test_eval_reports_escalation_for_cascade(capsys, monkeypatch):
    _fake_cascade(monkeypatch)
    assert cli.main(["eval", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["backend"] == "cascade"
    assert out["routing"]["escalation_rate"] == 1.0
    assert cli.main(["eval"]) == 0
    assert "escalation" in capsys.readouterr().out


def test_calibrate_cascade_writes_block(tmp_path, capsys, monkeypatch):
    _fake_cascade(monkeypatch)
    out = tmp_path / "cal.json"
    assert cli.main(["calibrate-cascade", "--out", str(out)]) == 0
    block = json.loads(out.read_text())["cascade"]
    assert block["native_gate"] == 1.0  # local always sure of none: never needs jev
    assert "native_gate" in capsys.readouterr().out


def test_calibrate_cascade_needs_jev(capsys):
    assert cli.main(["calibrate-cascade"]) == 2
    assert "not available" in capsys.readouterr().err


def test_demo_has_allow_shell_and_no_host(monkeypatch):
    import agent_router.demo.server as server

    seen = {}
    monkeypatch.setattr(server, "main", lambda **kw: seen.update(kw))
    assert cli.main(["demo", "--port", "8799"]) == 0
    assert seen == {
        "port": 8799,
        "allow_shell": False,
        "catalog": None,
        "audit_dir": None,
        "trace_dir": None,
    }
    assert cli.main(["demo", "--allow-shell"]) == 0
    assert seen["allow_shell"] is True
    with pytest.raises(SystemExit):
        cli.main(["demo", "--host", "0.0.0.0"])


def test_run_passes_allow_shell(monkeypatch, capsys):
    import agent_router.agent as agent

    seen = {}

    async def fake_run(prompt, router, workspace, on_event, **kw):
        seen.update(kw, backend=router.decider.name)
        return "ok"

    monkeypatch.setattr(agent, "run_agent", fake_run)
    assert cli.main(["run", "hi", "--embedder", "hashing"]) == 0
    assert seen == {"allow_shell": False, "backend": "local"}
    assert cli.main(["run", "hi", "--embedder", "hashing", "--allow-shell"]) == 0
    assert seen["allow_shell"] is True


# -- AGENT_ROUTER_* environment (route and run honour it; eval scores explicitly) ------------

CALC_PROMPT = ["route", "compute 3**80 exactly", "--backend", "local", "--json"]


def _route_json(capsys, *extra):
    assert cli.main([*CALC_PROMPT, *extra]) == 0
    return json.loads(capsys.readouterr().out)


def test_route_honours_agent_router_disabled(capsys, monkeypatch):
    monkeypatch.setenv("AGENT_ROUTER_DISABLED", "1")
    out = _route_json(capsys)
    assert (out["action"], out["reason"]) == ("skipped", "disabled")


def test_route_honours_agent_router_threshold(capsys, monkeypatch):
    monkeypatch.setenv("AGENT_ROUTER_THRESHOLD", "0.99")
    assert _route_json(capsys)["threshold"] == 0.99
    assert _route_json(capsys, "--threshold", "0.2")["threshold"] == 0.2  # flag beats env


def test_route_honours_agent_router_mode(capsys, monkeypatch):
    from agent_router.deciders import registry

    monkeypatch.setattr(registry, "make_decider", lambda name, cat: _Always("local", "exact-calc"))
    monkeypatch.setenv("AGENT_ROUTER_MODE", "enforce")
    tool = ["--point", "tool", "--tool", "Bash", "--input", '{"command": "bc"}']
    assert _route_json(capsys, *tool)["action"] == "enforce"
    monkeypatch.delenv("AGENT_ROUTER_MODE")
    assert _route_json(capsys, *tool)["action"] == "suggest"


def test_route_writes_agent_router_audit(capsys, monkeypatch, tmp_path):
    path = tmp_path / "audit" / "cli.jsonl"
    monkeypatch.setenv("AGENT_ROUTER_AUDIT", str(path))
    _route_json(capsys)
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(records) == 1 and records[0]["text"] == "compute 3**80 exactly"


def test_invalid_env_is_a_cli_error(capsys, monkeypatch):
    monkeypatch.setenv("AGENT_ROUTER_MODE", "loud")
    assert cli.main(CALC_PROMPT) == 2
    assert "AGENT_ROUTER" in capsys.readouterr().err
    monkeypatch.setenv("AGENT_ROUTER_MODE", "advisory")
    monkeypatch.setenv("AGENT_ROUTER_THRESHOLD", "high")
    assert cli.main(CALC_PROMPT) == 2


def test_run_honours_env_config_and_audit(monkeypatch, tmp_path):
    import agent_router.agent as agent
    from agent_router.core.types import HookPoint, RouterEvent

    path = tmp_path / "run.jsonl"
    monkeypatch.setenv("AGENT_ROUTER_AUDIT", str(path))
    monkeypatch.setenv("AGENT_ROUTER_MODE", "enforce")
    monkeypatch.setenv("AGENT_ROUTER_THRESHOLD", "0.42")
    seen = {}

    async def fake_run(prompt, router, workspace, on_event, **kw):
        seen["config"] = router.config
        seen["workspace"] = workspace
        router.route(RouterEvent(HookPoint.PROMPT, "run", 1, prompt))
        return "ok"

    monkeypatch.setattr(agent, "run_agent", fake_run)
    ws = tmp_path / "ws"
    ws.mkdir()
    assert cli.main(["run", "hi", "--backend", "local", "--workspace", str(ws)]) == 0
    cfg = seen["config"]
    assert (cfg.mode, cfg.threshold, cfg.enabled) == ("enforce", 0.42, True)
    assert seen["workspace"] == ws.resolve()
    assert json.loads(path.read_text().splitlines()[0])["text"] == "hi"
    # explicit flags beat the environment
    assert (
        cli.main(["run", "hi", "--backend", "local", "--mode", "advisory", "--threshold", "0.3"])
        == 0
    )
    assert (seen["config"].mode, seen["config"].threshold) == ("advisory", 0.3)
    monkeypatch.setenv("AGENT_ROUTER_DISABLED", "yes")
    assert cli.main(["run", "hi", "--backend", "local"]) == 0
    assert seen["config"].enabled is False


def test_eval_scores_explicitly_and_ignores_router_env(capsys, monkeypatch):
    monkeypatch.setenv("AGENT_ROUTER_DISABLED", "1")
    monkeypatch.setenv("AGENT_ROUTER_THRESHOLD", "0.99")
    assert cli.main(["eval", "--backend", "local", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["threshold"] != 0.99
    assert all(c["reason"] != "disabled" for c in out["cases"])


def test_demo_catalog_and_audit_dir(monkeypatch, tmp_path):
    import agent_router.demo.server as server

    path = tmp_path / "c.yaml"
    path.write_text(
        "version: v9\nentries:\n  - {id: a, kind: agent, name: n, project: p, license: MIT,"
        " url: u, what: w, target: t, points: [prompt], agents: [main]}\n"
    )
    seen = {}
    monkeypatch.setattr(server, "main", lambda **kw: seen.update(kw))
    argv = ["demo", "--catalog", str(path), "--audit-dir", str(tmp_path)]
    assert cli.main(argv) == 0
    assert seen["catalog"].version == "v9" and seen["audit_dir"] == tmp_path
