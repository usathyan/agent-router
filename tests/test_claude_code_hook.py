"""Claude Code command-hook adapter, ``agent-router hook`` and ``agent-router mcp``."""

import asyncio
import io
import json
import zlib

import pytest

from agent_router import cli
from agent_router.adapters import claude_code
from agent_router.adapters.mcp_stdio import build_server
from agent_router.core.catalog import Catalog, CatalogEntry
from agent_router.core.config import RouterConfig
from agent_router.core.router import Router
from agent_router.core.types import ChoiceResult, HookPoint

FP = "first-principles:first-principles"
DELEGATE = CatalogEntry(
    id="fp-agent",
    kind="agent",
    name="First-principles analysis agent",
    project="first-principles-skill",
    license="MIT",
    url="https://example.invalid",
    what="Decompose a claim into ground truths.",
    target=FP,
    points=(HookPoint.PROMPT, HookPoint.TOOL),
    replaces=("Agent",),
    agents=("main",),
)
CALC = CatalogEntry(
    id="exact-calc",
    kind="tool",
    name="Exact calculator",
    project="agent-router (own code)",
    license="MIT",
    url="https://example.invalid",
    what="Exact arithmetic.",
    target="mcp__plugin_x_agent_router__calc",
    points=(HookPoint.TOOL,),
    replaces=("Bash",),
    agents=(FP,),
)
CATALOG = Catalog(version="t", entries=(DELEGATE, CALC))


class Pick:
    name = "fake"

    def __init__(self):
        self.states = []

    def decide(self, state, options):
        self.states.append(state)
        choice = next(k for k in options if k != "none")
        probs = {k: (0.9 if k == choice else 0.1 / (len(options) - 1)) for k in options}
        return ChoiceResult(choice=choice, probabilities=probs, confidence=0.8)


@pytest.fixture
def factory():
    decider = Pick()

    def make(audit, seed):
        return Router(CATALOG, decider, RouterConfig(audit_path=audit), suggested=seed)

    make.decider = decider
    return make


def prompt(text="challenge the assumptions behind our pricing", pid="p1"):
    return {
        "hook_event_name": "UserPromptSubmit",
        "session_id": "s/1",
        "prompt_id": pid,
        "prompt": text,
    }


def tool(name, tool_input, agent_type=None, pid="p1"):
    out = {
        "hook_event_name": "PreToolUse",
        "session_id": "s/1",
        "prompt_id": pid,
        "tool_name": name,
        "tool_input": tool_input,
    }
    if agent_type:
        out["agent_type"] = agent_type
    return out


def test_prompt_hint_is_additional_context(tmp_path, factory):
    out = claude_code.handle(prompt(), factory, tmp_path)
    spec = out["hookSpecificOutput"]
    assert spec["hookEventName"] == "UserPromptSubmit"
    assert f'subagent_type="{FP}"' in spec["additionalContext"]
    # state and audit are per session, with a filesystem-safe name
    assert (tmp_path / "sessions" / "s_1.json").is_file()
    (line,) = (tmp_path / "audit" / "s_1.jsonl").read_text().splitlines()
    rec = json.loads(line)
    assert rec["turn"] == zlib.crc32(b"p1") and rec["agent_type"] is None


def test_hint_once_per_turn_across_processes(tmp_path, factory):
    assert claude_code.handle(prompt(), factory, tmp_path)
    call = tool("Agent", {"subagent_type": "general-purpose", "prompt": "x"})
    assert claude_code.handle(call, factory, tmp_path) == {}  # same turn: deduped
    assert claude_code.handle(prompt(pid="p2"), factory, tmp_path)  # new turn


def test_subagent_tool_call_routes_only_the_pending_call(tmp_path, factory):
    claude_code.handle(prompt("what is 2**10"), factory, tmp_path)
    out = claude_code.handle(
        tool("Bash", {"command": "python3 -c 'print(2**10)'"}, FP), factory, tmp_path
    )
    assert "ToolSearch" in out["hookSpecificOutput"]["additionalContext"]
    state = factory.decider.states[-1]
    assert state.startswith("\npending Bash:") and "what is 2**10" not in state


def test_main_thread_bash_has_no_eligible_entry(tmp_path, factory):
    claude_code.handle(prompt(), factory, tmp_path)
    assert claude_code.handle(tool("Bash", {"command": "ls"}), factory, tmp_path) == {}


def test_unknown_events_are_ignored(tmp_path, factory):
    assert claude_code.handle({"hook_event_name": "Stop"}, factory, tmp_path) == {}


def test_state_root_env(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_ROUTER_STATE_DIR", str(tmp_path))
    assert claude_code.state_root() == tmp_path
    monkeypatch.delenv("AGENT_ROUTER_STATE_DIR")
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    assert claude_code.state_root() == tmp_path / "agent-router"


def _run_hook(monkeypatch, capsys, payload, *argv):
    monkeypatch.setattr("sys.stdin", io.StringIO(payload))
    assert cli.main(["hook", *argv]) == 0
    return json.loads(capsys.readouterr().out)


def test_cli_hook_fails_open_on_bad_input(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("AGENT_ROUTER_STATE_DIR", str(tmp_path))
    assert _run_hook(monkeypatch, capsys, "not json") == {}


def test_cli_hook_fails_open_on_bad_catalog(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("AGENT_ROUTER_STATE_DIR", str(tmp_path))
    out = _run_hook(
        monkeypatch, capsys, json.dumps(prompt()), "--catalog", str(tmp_path / "missing.yaml")
    )
    assert out == {}


def test_cli_hook_routes_with_hashing_embedder(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("AGENT_ROUTER_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("AGENT_ROUTER_EMBEDDER", "hashing")
    out = _run_hook(
        monkeypatch,
        capsys,
        json.dumps(prompt("what is 17% of 2,340 exactly")),
        "--backend",
        "local",
    )
    assert "Exact calculator" in out["hookSpecificOutput"]["additionalContext"]


def test_cli_catalog_env(monkeypatch, tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text(
        "version: v9\nentries:\n  - {id: a, kind: agent, name: n, project: p, license: MIT,"
        " url: u, what: w, target: t, points: [prompt]}\n"
    )
    monkeypatch.setenv("AGENT_ROUTER_CATALOG", str(path))
    assert cli._catalog(cli.build_parser().parse_args(["route", "x"])).version == "v9"


def test_calibration_env(monkeypatch, tmp_path):
    from agent_router.deciders import local

    monkeypatch.setattr(local, "CALIBRATION_PATH", local.CALIBRATION_PATH)
    monkeypatch.setenv("AGENT_ROUTER_CALIBRATION", str(tmp_path / "cal.json"))
    cli._apply_calibration_env()
    assert tmp_path / "cal.json" == local.CALIBRATION_PATH


def test_mcp_server_serves_chosen_tools(tmp_path):
    server = build_server(tmp_path, ["calc"])
    assert [t.name for t in asyncio.run(server.list_tools())] == ["calc"]
    res = asyncio.run(server.call_tool("calc", {"expression": "17% of 2340"}))
    assert "1989/5" in json.dumps(res, default=str)
    with pytest.raises(ValueError):
        build_server(tmp_path, ["rm"])


@pytest.mark.parametrize(
    ("tool", "args", "message"),
    [
        ("calc", {"expression": "1/0"}, "division by zero"),
        ("calc", {"expression": "log(2)"}, "unknown function 'log'"),
        ("calc", {"expression": "[1, [2]]"}, "unsupported syntax: List"),
        ("json_query", {"expression": "a", "text": "{"}, "invalid JSON"),
        ("repo_stats", {"path": "../"}, "outside the workspace"),
    ],
)
def test_mcp_tool_errors_reach_the_model(tmp_path, tool, args, message):
    """A tool's ValueError comes back as its message, not the SDK's bare crash text."""
    server = build_server(tmp_path)
    with pytest.raises(Exception) as err:
        asyncio.run(server.call_tool(tool, args))
    assert message in str(err.value)
    assert type(err.value).__name__ == "ToolError"  # anticipated, not UnexpectedToolError


def test_mcp_tool_schemas_keep_their_parameters(tmp_path):
    tools = {t.name: t for t in asyncio.run(build_server(tmp_path).list_tools())}
    assert list(tools["calc"].input_schema["properties"]) == ["expression"]
    assert set(tools["json_query"].input_schema["properties"]) == {"expression", "path", "text"}


def test_audit_carries_the_hosts_call_and_run_ids(tmp_path, factory):
    claude_code.handle(prompt(), factory, tmp_path)
    call = tool("Bash", {"command": "python3 -c 'print(2**10)'"}, FP)
    claude_code.handle({**call, "tool_use_id": "toolu_9", "agent_id": "a1"}, factory, tmp_path)
    first, last = [
        json.loads(x) for x in (tmp_path / "audit" / "s_1.jsonl").read_text().splitlines()
    ]
    assert (first["tool_use_id"], first["agent_id"]) == (None, None)  # the prompt
    assert (last["tool_use_id"], last["agent_id"]) == ("toolu_9", "a1")
