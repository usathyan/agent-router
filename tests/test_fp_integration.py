"""integrations/first-principles: catalog, eval set, plugin files and battery helpers."""

import importlib.util
import json
import os
from pathlib import Path

import pytest

from agent_router.core.catalog import load_catalog
from agent_router.core.types import HookPoint
from agent_router.evaluate import load_cases

INTEG = Path(__file__).resolve().parents[1] / "integrations" / "first-principles"
FP = "first-principles:first-principles"


def _battery():
    spec = importlib.util.spec_from_file_location("fp_battery", INTEG / "battery.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_catalog_scopes_each_entry():
    cat = load_catalog(INTEG / "catalog.yaml")
    delegate = cat.get("first-principles-agent")
    assert delegate.kind == "agent" and delegate.target == FP and delegate.agents == ("main",)
    # no entry routes to an individual first-principles technique (that is Step 0's job)
    assert not any(
        e.target.startswith("first-principles:") and e.kind == "skill" for e in cat.entries
    )
    assert cat.eligible(HookPoint.PROMPT, agent_type=FP) == []


def test_eval_set_ids_unique_and_labels_known():
    cat = load_catalog(INTEG / "catalog.yaml")
    cases = load_cases(INTEG / "eval_set.yaml")
    assert len({c.id for c in cases}) == len(cases)
    known = {e.id for e in cat.entries} | {"none"}
    assert {c.expected for c in cases} <= known
    holdout = [c for c in cases if c.id.startswith("fp-")]
    assert len(holdout) == 33 and all(c.split == "test" for c in holdout)


def test_calibration_is_tagged_with_the_integration_catalog():
    cal = json.loads((INTEG / "calibration.json").read_text())
    version = load_catalog(INTEG / "catalog.yaml").version
    assert cal["model2vec"]["catalog_version"] == version


def test_plugin_files():
    plugin = INTEG / "plugin"
    manifest = json.loads((plugin / ".claude-plugin" / "plugin.json").read_text())
    assert manifest["name"] == "agent-router-fp"
    hooks = json.loads((plugin / "hooks" / "hooks.json").read_text())["hooks"]
    assert set(hooks) == {
        "UserPromptSubmit",
        "PreToolUse",
        "SubagentStart",
        "SubagentStop",
        "PostToolUse",
        "PostToolUseFailure",
    }
    for event in ("SubagentStart", "SubagentStop"):
        assert hooks[event][0]["matcher"] == FP
    assert os.access(plugin / "bin" / "hook", os.X_OK)
    assert os.access(plugin / "bin" / "trace", os.X_OK)


def test_delegated_call_detection():
    battery = _battery()
    call = {
        "type": "assistant",
        "message": {
            "content": [{"type": "tool_use", "name": "Agent", "input": {"subagent_type": FP}}]
        },
    }
    assert battery.delegated_call(call)
    other = json.loads(json.dumps(call).replace(FP, "general-purpose"))
    assert not battery.delegated_call(other)
    assert not battery.delegated_call({"type": "result"})


@pytest.mark.model
def test_holdout_meets_first_principles_thresholds(monkeypatch):
    """The nudge alone clears the bar first-principles sets for its own routing."""
    from agent_router.deciders import local
    from agent_router.evaluate import make_router, run_eval

    monkeypatch.setattr(local, "CALIBRATION_PATH", INTEG / "calibration.json")
    cat = load_catalog(INTEG / "catalog.yaml")
    router = make_router("local", cat)
    cases = [c for c in load_cases(INTEG / "eval_set.yaml") if c.id.startswith("fp-")]
    report = run_eval(lambda: router, cases)
    got = {r.id: r.predicted for r in report.results}
    p_ok = sum(got[c.id] == c.expected for c in cases if c.expected != "none")
    n_ok = sum(got[c.id] == "none" for c in cases if c.expected == "none")
    assert p_ok >= 11 and n_ok >= 18


def _hook(payload, tmp_path):
    import subprocess

    env = {
        **os.environ,
        "AGENT_ROUTER_STATE_DIR": str(tmp_path),
        "AGENT_ROUTER_EMBEDDER": "hashing",  # offline: no model download
    }
    out = subprocess.run(
        [str(INTEG / "plugin" / "bin" / "hook")],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
        check=True,
    )
    return json.loads(out.stdout)


def test_plugin_serves_only_the_calculator():
    mcp = json.loads((INTEG / "plugin" / ".mcp.json").read_text())["mcpServers"]
    assert list(mcp) == ["agent_router"]
    script = (INTEG / "plugin" / "bin" / "mcp").read_text()
    assert "mcp --tools calc" in script
    target = load_catalog(INTEG / "catalog.yaml").get("exact-calc").target
    assert target == "mcp__plugin_agent-router-fp_agent_router__calc"


def test_hook_fast_path_for_main_thread_bash(tmp_path):
    payload = {
        "hook_event_name": "PreToolUse",
        "session_id": "s",
        "tool_name": "Bash",
        "tool_input": {"command": "python3 -c 'print(2**64)'"},
    }
    assert _hook(payload, tmp_path) == {}
    assert not (tmp_path / "audit").exists()  # answered by the shell, not the router


def test_hook_skips_the_agents_report_writes(tmp_path):
    payload = {
        "hook_event_name": "PreToolUse",
        "session_id": "s",
        "prompt_id": "p",
        "agent_type": FP,
        "tool_name": "Bash",
        "tool_input": {"command": "cat >> \".first-principles/a.md\" <<'X'\n12 x 1500 = 18000\nX"},
    }
    assert _hook(payload, tmp_path) == {}
    (line,) = (tmp_path / "audit" / "s.jsonl").read_text().splitlines()
    assert json.loads(line)["reason"] == "skip pattern"


# --- decision trace (trace.py, plugin/bin/trace) -----------------------------------------------


def _trace():
    spec = importlib.util.spec_from_file_location("fp_trace", INTEG / "trace.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ANALYSIS = """# First-Principles Analysis — test

Run mode: `full-composer`

**Re-entry disclosure.** One edge fired: the Self-Audit Gate's Fix/Repeat. It re-scored C2.

## Techniques not applied (process output)

- fishbone (Phase 2) — not applicable — one cause
- trade-off (Phase 4) — not applicable — one option

## Self-Audit Gate (process output)

**Criterion 1: Identify Essence**
Band: **Rigorous**

**Criterion 2: Challenge Assumptions**
Band: **Hand-wavy**

**Criterion 2: Challenge Assumptions**
Band: **Sound**

Gate result: no criterion Absent; Fix/Repeat fired once.

## 1. Problem Essence

**Core problem:** x

## 2. Assumptions Table

| Assumption | Type | Treatment | Verdict | Verification |
|---|---|---|---|---|
| (A1) Demand grows | untested belief | Verify | Challenge — no data | unverified — flagged |
| (A2) Energy is conserved | physical law | Accept | Accept — law | textbook |
| (A3) Copy peers | convention — analogy-as-evidence | Challenge | **Discard** — analogy | n/a |

## 3. Ground Truths

- **GT-1** Energy is conserved — source: physics
- **GT-2?** Demand is 10/day — unverified: no telemetry

## 4. Derivation Chains

### Conclusion C1: The load fits

GT-1 + GT-2? -> fits

**Confidence:** MEDIUM — GT-2? unverified

### Conclusion C2: [Illustrative] Cost is low

**Confidence:** LOW

### Conclusion C3 [Speculative]: Queues shrink

**Confidence:** LOW

## 5. Abandoned Reasoning

### Dead End: copy the competitor
### Dead End 2: "scale first"

## 6. Conclusion

**Recommended approach:** Measure demand first (chain C1).

**Confidence:** MEDIUM — C1 rests on GT-2?
"""


def test_parse_analysis_reads_every_decision():
    d = _trace().parse_analysis(ANALYSIS)
    assert d["run_mode"] == "full-composer"
    assert d["re_entry"] == {
        "fired": True,
        "disclosure": "Re-entry disclosure. One edge fired: the Self-Audit Gate's Fix/Repeat",
    }
    assert d["sections"][-6:] == [
        "1. Problem Essence",
        "2. Assumptions Table",
        "3. Ground Truths",
        "4. Derivation Chains",
        "5. Abandoned Reasoning",
        "6. Conclusion",
    ]
    a = d["assumptions"]
    assert a["count"] == 3
    assert a["by_type"] == {"untested belief": 1, "physical law": 1, "convention": 1}
    assert a["by_verdict"] == {"Challenge": 1, "Accept": 1, "Discard": 1}
    assert d["ground_truths"] == {"count": 2, "unverified": ["GT-2?"]}
    chains = [(c["id"], c["tag"], c["confidence"]) for c in d["chains"]]
    assert chains == [("C1", None, "MEDIUM"), ("C2", None, "LOW"), ("C3", "[Speculative]", "LOW")]
    assert d["dead_ends"] == ["copy the competitor", '"scale first"']
    assert [t["technique"] for t in d["techniques_not_applied"]] == ["fishbone", "trade-off"]
    gate = d["gate"]
    assert gate["passes"] == 2 and gate["fix_repeat"] is True
    assert gate["bands"] == {"2": "Sound"}  # the last pass
    assert gate["criteria"] == {"1": "Identify Essence", "2": "Challenge Assumptions"}
    assert gate["result"].startswith("no criterion Absent")
    assert d["conclusion"]["confidence"] == "MEDIUM"
    assert d["conclusion"]["recommended"].startswith("Measure demand first")


def test_parse_analysis_tolerates_a_partial_file():
    d = _trace().parse_analysis("## 1. Problem Essence\n\nhalf written")
    assert d["chains"] == [] and d["gate"]["passes"] == 0
    assert d["conclusion"] == {"recommended": None, "confidence": None}


def _audit(path, *recs):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in recs))


def test_trace_records_a_run_and_links_the_routers_decisions(tmp_path):
    tr = _trace()
    work = tmp_path / "work"
    (work / ".first-principles").mkdir(parents=True)
    report = work / ".first-principles" / "analysis-20260929T000000Z.md"
    report.write_text(ANALYSIS)
    state = tmp_path / "state"
    _audit(
        state / "audit" / "s.jsonl",
        {
            "ts": "2026-09-29T10:00:00+00:00",
            "point": "prompt",
            "action": "suggest",
            "entry_id": "first-principles-agent",
            "applied_threshold": 0.55,
            "probabilities": {"first-principles-agent": 0.8},
        },
        {
            "ts": "2026-09-29T10:01:15+00:00",
            "point": "tool",
            "agent_type": FP,
            "tool_use_id": "t-script",
            "action": "skipped",
            "reason": "does not fit: exact-calc",
            "options": [],
        },
        {
            "ts": "2026-09-29T10:02:00+00:00",
            "point": "tool",
            "agent_type": FP,
            "tool_use_id": "t-note",
            "action": "suggest",
            "entry_id": "exact-calc",
            "options": ["exact-calc", "none"],
        },
    )
    base = {"session_id": "s", "agent_type": FP, "agent_id": "a1", "cwd": str(work)}
    handoff = {
        "session_id": "s",
        "hook_event_name": "PreToolUse",
        "tool_name": "Agent",
        "tool_input": {
            "subagent_type": FP,
            "prompt": "Recompute every figure with the calculator.",
        },
    }
    tr.handle(handoff, state, now="2026-09-29T10:00:30Z")
    steps = [
        ("2026-09-29T10:01:00Z", {"hook_event_name": "SubagentStart"}),
        (
            "2026-09-29T10:01:05Z",
            {
                "hook_event_name": "PostToolUse",
                "tool_name": "ToolSearch",
                "tool_input": {"query": f"select:{tr.CALC_TOOL}"},
                "tool_response": "loaded",
            },
        ),
        (
            "2026-09-29T10:01:10Z",
            {
                "hook_event_name": "PostToolUse",
                "tool_name": "Read",
                "tool_input": {"file_path": "/x/agents/references/pre-mortem.md"},
            },
        ),
        (
            "2026-09-29T10:01:20Z",
            {
                "hook_event_name": "PostToolUse",
                "tool_name": "Bash",
                "tool_use_id": "t-script",
                "tool_input": {"command": "python3 -c 'x=3\nprint(1/x)'"},
                "tool_response": {"stdout": "0.333"},
            },
        ),
        (
            "2026-09-29T10:02:05Z",
            {
                "hook_event_name": "PostToolUse",
                "tool_name": "Bash",
                "tool_use_id": "t-note",
                "tool_input": {"command": "python3 -c 'print(2/3)'"},
                "tool_response": {"stdout": "0.667"},
            },
        ),
        (
            "2026-09-29T10:02:30Z",
            {
                "hook_event_name": "PostToolUse",
                "tool_name": tr.CALC_TOOL,
                "tool_input": {"expression": "1/3"},
                "tool_response": [{"type": "text", "text": "1/3"}],
            },
        ),
        (
            "2026-09-29T10:02:40Z",
            {
                "hook_event_name": "PostToolUseFailure",
                "tool_name": tr.CALC_TOOL,
                "tool_input": {"expression": "1/0"},
                "error": "division by zero",
            },
        ),
        (
            "2026-09-29T10:03:00Z",
            {
                "hook_event_name": "PostToolUse",
                "tool_name": "Bash",
                "tool_input": {
                    "command": 'cat >> ".first-principles/analysis-'
                    "20260929T000000Z.md\" <<'FP_EOF'\n## 6. Conclusion"
                    "\n### Conclusion C1: x\nbody\nFP_EOF"
                },
                "tool_response": {"stdout": ""},
            },
        ),
        (
            "2026-09-29T10:03:30Z",
            {
                "hook_event_name": "PostToolUse",
                "tool_name": "Bash",
                "tool_input": {
                    "command": "cd .first-principles && python3 - <<'PY'\n"
                    "s=open(p).read()\nopen(p,'w').write(s.replace(a,b))\nPY"
                },
                "tool_response": {"stdout": ""},
            },
        ),
        (
            "2026-09-29T10:04:00Z",
            {"hook_event_name": "SubagentStop", "last_assistant_message": "Analysis written."},
        ),
    ]
    for ts, extra in steps:
        tr.handle({**base, **extra}, state, now=ts)
    recs = [json.loads(x) for x in (state / "trace" / "s.jsonl").read_text().splitlines()]
    assert [r["kind"] for r in recs] == [
        "delegation",
        "run_start",
        "tool_loaded",
        "reference_read",
        "shell",
        "shell",
        "calc",
        "tool_failed",
        "section_written",
        "report_op",
        "run_end",
    ]
    recs = recs[2:]  # the handoff and run_start, checked below
    assert recs[0]["calc"] is True
    recs = recs[1:]
    assert recs[0]["meaning"] == "adversarial pass on a plan (Phase 5)"
    assert recs[1]["interpreter"] is True and recs[1]["tool_use_id"] == "t-script"
    assert recs[3] == {**recs[3], "expression": "1/3", "result": "1/3"}
    assert recs[4]["was"] == "calc" and recs[4]["expression"] == "1/0"
    assert recs[5]["headings"] == ["6. Conclusion", "Conclusion C1: x"]
    end = recs[-1]
    assert end["duration_s"] == 180.0
    assert end["analysis"] == str(report)
    assert end["references_read"] == ["pre-mortem.md"]
    assert recs[6]["op"] == "revise"
    assert end["sections_written"] == 1 and end["tool_failures"] == 1
    assert end["report_revisions"] == 1
    assert end["decisions"]["gate"]["fix_repeat"] is True
    assert end["routing"] == {
        "delegation": {
            "action": "suggest",
            "entry_id": "first-principles-agent",
            "p": 0.8,
            "applied_threshold": 0.55,
        },
        "delegation_prompt_names_calc": True,
        "calc_loaded_at": "2026-09-29T10:01:05Z",
        "calc_loaded_before_first_note": True,
        "calc_notes": 1,
        "note_calls": [
            {
                "ts": "2026-09-29T10:02:00+00:00",
                "tool_use_id": "t-note",
                "command": "python3 -c 'print(2/3)'",
            }
        ],
        "calc_calls": 2,
        "calc_failures": 1,
        "calc_calls_before_note": 0,
        "calc_calls_after_note": 2,
        "interpreter_calls": 2,
        "interpreter_calls_calc_could_run": 1,
    }


def test_trace_ignores_everything_outside_the_agent(tmp_path):
    tr = _trace()
    main = {
        "session_id": "s",
        "hook_event_name": "PostToolUse",
        "tool_name": "Bash",
        "tool_input": {"command": "ls"},
    }
    other = {**main, "agent_type": "general-purpose", "agent_id": "x"}
    unrelated = {
        **main,
        "agent_type": FP,
        "agent_id": "a",
        "tool_name": "Grep",
        "tool_input": {"pattern": "x"},
    }
    plain_read = {**unrelated, "tool_name": "Read", "tool_input": {"file_path": "/src/a.py"}}
    for payload in (main, other, unrelated, plain_read):
        assert tr.handle(payload, tmp_path) is None
    assert not (tmp_path / "trace").exists()


def _run_trace(payload, tmp_path):
    import subprocess

    env = {**os.environ, "AGENT_ROUTER_STATE_DIR": str(tmp_path)}
    env.pop("AGENT_ROUTER_DISABLED", None)
    out = subprocess.run(
        [str(INTEG / "plugin" / "bin" / "trace")],
        input=payload if isinstance(payload, str) else json.dumps(payload),
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
        check=True,
    )
    return json.loads(out.stdout)


def test_trace_hook_is_record_only_and_fails_open(tmp_path):
    start = {
        "hook_event_name": "SubagentStart",
        "session_id": "s",
        "agent_type": FP,
        "agent_id": "a",
    }
    assert _run_trace(start, tmp_path) == {}
    assert json.loads((tmp_path / "trace" / "s.jsonl").read_text())["kind"] == "run_start"
    main = {
        "hook_event_name": "PostToolUse",
        "session_id": "m",
        "tool_name": "Bash",
        "tool_input": {"command": "ls"},
    }
    assert _run_trace(main, tmp_path) == {}
    assert not (tmp_path / "trace" / "m.jsonl").exists()  # answered by the shell
    garbage = '{"agent_type": "first-principles:first-principles", not json'
    assert _run_trace(garbage, tmp_path) == {}


def test_trace_replays_a_capture(tmp_path):
    tr = _trace()
    work = tmp_path / "w"
    (work / ".first-principles").mkdir(parents=True)
    (work / ".first-principles" / "analysis-1.md").write_text(ANALYSIS)
    call = {"type": "tool_use", "id": "ag", "name": "Agent", "input": {"subagent_type": FP}}
    read = {
        "type": "tool_use",
        "id": "r1",
        "name": "Read",
        "input": {"file_path": "/p/references/trade-off.md"},
    }
    capture = [
        {"type": "system", "subtype": "init", "session_id": "s", "cwd": str(work)},
        {"type": "assistant", "timestamp": "2026-09-29T10:00:00Z", "message": {"content": [call]}},
        {
            "type": "assistant",
            "timestamp": "2026-09-29T10:00:05Z",
            "parent_tool_use_id": "ag",
            "message": {"content": [read]},
        },
        {
            "type": "user",
            "timestamp": "2026-09-29T10:00:06Z",
            "parent_tool_use_id": "ag",
            "message": {"content": [{"type": "tool_result", "tool_use_id": "r1", "content": "x"}]},
        },
        {
            "type": "user",
            "timestamp": "2026-09-29T10:05:00Z",
            "message": {
                "content": [{"type": "tool_result", "tool_use_id": "ag", "content": "done"}]
            },
        },
    ]
    path = tmp_path / "run.jsonl"
    path.write_text("".join(json.dumps(e) + "\n" for e in capture))
    for payload, ts in tr.replay_payloads(path):
        tr.handle(payload, tmp_path / "state", now=ts)
    recs = [
        json.loads(x) for x in (tmp_path / "state" / "trace" / "s.jsonl").read_text().splitlines()
    ]
    assert [r["kind"] for r in recs] == ["delegation", "run_start", "reference_read", "run_end"]
    assert recs[0]["prompt_names_calc"] is False
    assert recs[2]["tool_use_id"] == "r1"  # joins the record to the router's audit
    recs = recs[1:]
    assert recs[-1]["duration_s"] == 300.0
    assert recs[-1]["analysis"].endswith("analysis-1.md")  # found in cwd: no append recorded


def test_trace_report_has_one_row_per_finished_run(tmp_path):
    tr = _trace()
    work = tmp_path / "w"
    (work / ".first-principles").mkdir(parents=True)
    (work / ".first-principles" / "analysis-1.md").write_text(ANALYSIS)
    base = {"session_id": "s", "agent_type": FP, "cwd": str(work)}
    for agent in ("a1", "a2"):
        tr.handle({**base, "agent_id": agent, "hook_event_name": "SubagentStart"}, tmp_path)
    tr.handle({**base, "agent_id": "a1", "hook_event_name": "SubagentStop"}, tmp_path)
    rows = tr.report(tmp_path / "trace").splitlines()[2:]
    assert len(rows) == 1  # a2 never finished
    assert "MEDIUM" in rows[0] and "| yes |" in rows[0]  # confidence, Fix/Repeat


def test_hook_skips_in_place_report_revisions(tmp_path):
    """Revisions cd into .first-principles without naming it with a slash (seen live)."""
    payload = {
        "hook_event_name": "PreToolUse",
        "session_id": "s",
        "prompt_id": "p",
        "agent_type": FP,
        "tool_name": "Bash",
        "tool_input": {
            "command": "cd /w/.first-principles && python3 - <<'PY'\n"
            "s=open('analysis-1.md').read()\nprint(12 * 1500)\nPY"
        },
    }
    assert _hook(payload, tmp_path) == {}
    (line,) = (tmp_path / "audit" / "s.jsonl").read_text().splitlines()
    assert json.loads(line)["reason"] == "skip pattern"


def test_trace_flags_revisions_and_restarts_of_the_report():
    tr = _trace()

    def bash(cmd):
        return tr._tool_record({"tool_name": "Bash", "tool_input": {"command": cmd}})

    restart = bash('P=/w/.first-principles/analysis-1.md\n: > "$P"\ncat >> "$P" <<E\n# A\nE')
    assert restart["kind"] == "section_written" and restart["restarts"] and not restart["revises"]
    plain = bash('cat >> "/w/.first-principles/analysis-1.md" <<E\nx : > y\nE')
    assert not plain["restarts"]
    fix = bash(
        'F=/w/.first-principles/a.md; sed -i \'s/six/seven/\' "$F"; cat >> "$F" <<E\n## 1. X\nE'
    )
    assert fix["kind"] == "section_written" and fix["revises"]
    assert bash("cd /w/.first-principles && grep -c '^#' analysis-1.md")["op"] == "check"
    assert tr.parse_analysis("- **Re-entry edges fired:** none.")["re_entry"]["fired"] is False
    assert tr.parse_analysis("no disclosure")["re_entry"] is None


def _replay_capture(tmp_path, agent_calls, main_calls=()):
    """A minimal ``claude -p`` capture: one first-principles run with the given tool calls
    (``(name, input, result)``) inside it, then the given main-session calls after it."""
    events = [
        {"type": "system", "subtype": "init", "session_id": "s", "cwd": str(tmp_path / "gone")},
        {
            "type": "assistant",
            "timestamp": "2026-09-29T10:00:00Z",
            "message": {
                "content": [
                    {
                        "type": "tool_use",
                        "id": "ag",
                        "name": "Agent",
                        "input": {"subagent_type": FP},
                    }
                ]
            },
        },
    ]
    for i, (name, inp, out) in enumerate(agent_calls):
        use = {"type": "tool_use", "id": f"a{i}", "name": name, "input": inp}
        res = {"type": "tool_result", "tool_use_id": f"a{i}", "content": out}
        events += [
            {"type": "assistant", "parent_tool_use_id": "ag", "message": {"content": [use]}},
            {"type": "user", "parent_tool_use_id": "ag", "message": {"content": [res]}},
        ]
    done = {"type": "tool_result", "tool_use_id": "ag", "content": "done"}
    events.append(
        {"type": "user", "timestamp": "2026-09-29T10:05:00Z", "message": {"content": [done]}}
    )
    for i, (name, inp, out) in enumerate(main_calls):
        use = {"type": "tool_use", "id": f"m{i}", "name": name, "input": inp}
        res = {"type": "tool_result", "tool_use_id": f"m{i}", "content": out}
        events += [
            {"type": "assistant", "message": {"content": [use]}},
            {"type": "user", "message": {"content": [res]}},
        ]
    path = tmp_path / "run.jsonl"
    path.write_text("".join(json.dumps(e) + "\n" for e in events))
    tr = _trace()
    for payload, ts in tr.replay_payloads(path):
        tr.handle(payload, tmp_path / "state", now=ts)
    return [
        json.loads(x) for x in (tmp_path / "state" / "trace" / "s.jsonl").read_text().splitlines()
    ]


REPORT = ".first-principles/analysis-1.md"


def _append(text):
    return ("Bash", {"command": f"cat >> {REPORT} <<'FP_EOF'\n{text}\nFP_EOF"}, "")


def test_trace_rebuilds_the_report_from_its_appends_when_the_file_is_gone(tmp_path):
    half = ANALYSIS.index("## 4.")
    recs = _replay_capture(
        tmp_path,
        [
            ("Bash", {"command": f"mkdir -p .first-principles && : > {REPORT}"}, ""),
            _append(ANALYSIS[:half][:-1]),  # a heredoc adds back one newline per append
            _append(ANALYSIS[half:].removesuffix("\n")),
        ],
    )
    sections = [r for r in recs if r["kind"] == "section_written"]
    assert sections[0]["text"].startswith("# First-Principles Analysis")
    end = recs[-1]
    assert end["analysis_source"] == "appends" and end["analysis_exact"] is True
    assert end["decisions"]["gate"].pop("passes_written") == 2  # both passes were appended
    assert end["decisions"] == _trace().parse_analysis(ANALYSIS)
    assert end["analysis_text"] == ANALYSIS.removesuffix("\n") + "\n"
    assert len(end["analysis_sha256"]) == 64


def test_trace_marks_a_rebuilt_report_inexact_after_an_in_place_revision(tmp_path):
    recs = _replay_capture(
        tmp_path,
        [_append(ANALYSIS), ("Bash", {"command": f"sed -i 's/HIGH/LOW/' {REPORT}"}, "")],
    )
    assert recs[-1]["analysis_source"] == "appends" and recs[-1]["analysis_exact"] is False


def test_trace_restart_discards_what_was_appended_before(tmp_path):
    recs = _replay_capture(
        tmp_path,
        [
            _append("# stale draft"),
            (
                "Bash",
                {"command": f": > {REPORT} && cat >> {REPORT} <<'FP_EOF'\n{ANALYSIS}\nFP_EOF"},
                "",
            ),
        ],
    )
    assert "stale draft" not in recs[-1]["analysis_text"]
    assert recs[-1]["decisions"]["sections"] == _trace().parse_analysis(ANALYSIS)["sections"]


def test_trace_replay_prefers_the_main_sessions_full_read_of_the_report(tmp_path):
    full = str(tmp_path / "gone" / REPORT)
    numbered = "".join(f"{i}\t{line}\n" for i, line in enumerate(ANALYSIS.splitlines(), 1))
    recs = _replay_capture(
        tmp_path,
        [_append("# only a draft"), ("Bash", {"command": f"sed -i 's/a/b/' {REPORT}"}, "")],
        main_calls=[
            ("Read", {"file_path": full}, numbered),
            ("Read", {"file_path": full, "offset": 1, "limit": 2}, "1\t# partial"),
        ],
    )
    end = recs[-1]
    assert end["analysis_source"] == "capture" and end["analysis_exact"] is True
    assert end["decisions"]["gate"].pop("passes_written") == 0  # the appends had no gate
    assert end["decisions"] == _trace().parse_analysis(ANALYSIS)


@pytest.mark.parametrize(
    ("prompt", "names"),
    [
        ("Recompute every figure with the calculator.", True),
        ("recompute it (use the calc tool if available)", True),
        ("use mcp__plugin_agent-router-fp_agent_router__calc", True),
        ("Recalculate the totals from first principles.", False),
        ("Analyse this from first principles.", False),
    ],
)
def test_trace_records_whether_the_handoff_names_the_calculator(tmp_path, prompt, names):
    rec = _trace().handle(
        {
            "session_id": "s",
            "hook_event_name": "PreToolUse",
            "tool_name": "Agent",
            "tool_input": {"subagent_type": FP, "prompt": prompt},
        },
        tmp_path,
    )
    assert rec["kind"] == "delegation" and rec["prompt_names_calc"] is names


def test_trace_ignores_a_handoff_to_another_agent(tmp_path):
    call = {"subagent_type": "general-purpose", "prompt": "use the calculator"}
    payload = {"session_id": "s", "hook_event_name": "PreToolUse", "tool_name": "Agent"}
    assert _trace().handle({**payload, "tool_input": call}, tmp_path) is None


@pytest.mark.parametrize(
    ("line", "mode"),
    [
        ("Run mode: `full-composer` (no trigger fired).", "full-composer"),
        ("Mode: full-composer. No technique-specific trigger fired.", "full-composer"),
        ("- **Mode:** `focused-five-whys` (causal mode).", "focused-five-whys"),
        ("Run disclosures: MODE = full-composer (no trigger).", "full-composer"),
        ("Step 0 selected `MODE = full-composer`, because no trigger fired.", "full-composer"),
        ("The mode of failure: fatigue.", None),
    ],
)
def test_parse_analysis_reads_the_run_mode_however_it_is_stated(line, mode):
    assert _trace().parse_analysis(line + "\n")["run_mode"] == mode


def test_parse_analysis_takes_the_last_gate_result_bold_or_not():
    text = (
        "**Gate result:** passes, but Criterion 3 fails, so the Fix step runs.\n"
        "Gate result after re-score: cleared.\n"
    )
    assert _trace().parse_analysis(text)["gate"]["result"] == "cleared."


def test_parse_analysis_flags_a_re_entry_disclosure_the_gate_contradicts():
    base = "No re-entry edge fired in this run.\n"
    blocks = "".join(
        f"**Criterion 1: Identify Essence**\nBand: **{b}**\n" for b in ("Hand-wavy", "Rigorous")
    )
    d = _trace().parse_analysis(base + blocks)
    assert d["gate"]["passes"] == 2 and d["gate"]["cleared"] is True
    assert d["contradictions"] == [
        "the report says no re-entry edge fired, but the gate was scored 2 times"
    ]
    assert _trace().parse_analysis(base + blocks[: len(blocks) // 2])["contradictions"] == []


def test_trace_counts_gate_passes_a_rewrite_dropped(tmp_path):
    first = "**Criterion 1: Identify Essence**\nBand: **Hand-wavy**"
    second = "**Criterion 1: Identify Essence**\nBand: **Rigorous**"
    recs = _replay_capture(
        tmp_path,
        [
            _append("## 7. Gate\n" + first),
            (
                "Bash",
                {"command": f": > {REPORT} && cat >> {REPORT} <<'FP_EOF'\n{second}\nFP_EOF"},
                "",
            ),
        ],
    )
    gate = recs[-1]["decisions"]["gate"]
    assert (gate["passes"], gate["passes_written"]) == (1, 2)
    assert _trace()._fix_repeat(recs[-1]["decisions"]) == "yes (rewritten)"


def _run_examples():
    spec = importlib.util.spec_from_file_location("fp_run_examples", INTEG / "run_examples.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _events(*, result=None, returned=True):
    agent = {"type": "tool_use", "id": "ag", "name": "Agent", "input": {"subagent_type": FP}}
    out = [{"type": "assistant", "message": {"content": [agent]}}]
    if returned:
        done = {"type": "tool_result", "tool_use_id": "ag", "content": "done"}
        out.append({"type": "user", "message": {"content": [done]}})
    if result is not None:
        out.append({"type": "result", **result})
    return out


def _report(tmp_path, text):
    path = tmp_path / "analysis-1.md"
    path.write_text(text)
    return [path]


def test_run_examples_marks_a_finished_run_complete(tmp_path):
    ok = _events(result={"is_error": False, "result": "done"})
    assert _run_examples().completeness(ok, _report(tmp_path, ANALYSIS)) == ("complete", [])


def test_run_examples_marks_a_run_cut_short_partial(tmp_path):
    """The 2026-09-29 rerun: the spend limit stopped one agent after sections 1-3."""
    limit = {"is_error": True, "result": "You've hit your monthly spend limit"}
    head = ANALYSIS[: ANALYSIS.index("## 4.")]
    status, problems = _run_examples().completeness(_events(result=limit), _report(tmp_path, head))
    assert status == "partial"
    assert problems == [
        "error: You've hit your monthly spend limit",
        "analysis missing sections 4, 5, 6",  # its gate blocks come first, so they survive
    ]


def test_run_examples_marks_a_killed_run_failed(tmp_path):
    status, problems = _run_examples().completeness(_events(returned=False), [])
    assert status == "failed"
    assert problems == [
        "no result (killed or timed out)",
        "the agent did not hand back",
        "no analysis file",
    ]


def test_hook_skips_prompts_that_already_launch_first_principles(tmp_path):
    """The launcher already delegates; a note there can change nothing (10 of 28 runs)."""
    out = _hook(
        {
            "hook_event_name": "UserPromptSubmit",
            "session_id": "s",
            "prompt": "/first-principles:first-principles-analysis Is a four-day week better?",
        },
        tmp_path,
    )
    assert out == {}
    (rec,) = [json.loads(x) for x in (tmp_path / "audit" / "s.jsonl").read_text().splitlines()]
    assert (rec["action"], rec["reason"]) == ("skipped", "skip pattern")


def test_trace_records_source_checks_and_the_hosts_that_failed(tmp_path):
    fetch = "WebFetch"
    recs = _replay_capture(
        tmp_path,
        [
            (fetch, {"url": "https://www.nlr.gov/atb", "prompt": "q"}, "Capex is $1,200/kW."),
            (fetch, {"url": "https://www.osti.gov/x.pdf", "prompt": "q"}, "I cannot locate it."),
            ("WebSearch", {"query": "molten salt price"}, "Links: [...]"),
        ],
    )
    sources = [r for r in recs if r["kind"] == "source"]
    assert [(r["host"], r["outcome"]) for r in sources] == [
        ("www.nlr.gov", "ok"),
        ("www.osti.gov", "reported_missing"),
        (None, "ok"),
    ]
    assert sources[2]["target"] == "molten salt price"
    tr = _trace()
    failed = tr.handle(
        {
            "session_id": "s",
            "agent_type": FP,
            "agent_id": "replay-1",
            "hook_event_name": "PostToolUseFailure",
            "tool_name": fetch,
            "tool_input": {"url": "https://atb.nrel.gov/x"},
            "error": "getaddrinfo ENOTFOUND atb.nrel.gov",
        },
        tmp_path / "state",
    )
    assert failed["was"] == "source" and failed["host"] == "atb.nrel.gov"
    assert "outcome" not in failed
    run = sources + [failed]
    assert tr._sources(run) == {
        "checks": 4,
        "ok": 2,
        "reported_missing": 1,
        "failed": 1,
        "failed_hosts": {"atb.nrel.gov": 1},
    }


@pytest.mark.parametrize(
    ("call", "own"),
    [
        (("Read", {"file_path": "/fp/agents/references/examples/estimate-fermi.md"}), True),
        (("Bash", {"command": "head -60 /fp/agents/references/examples/estimate-fermi.md"}), True),
        (("Read", {"file_path": "/fp/agents/references/examples/estimate-fermi-2.md"}), False),
        (("Read", {"file_path": "/fp/agents/references/examples/product-business.md"}), False),
    ],
)
def test_run_examples_flags_a_run_that_read_its_own_answer_key(call, own):
    name, inp = call
    use = {"type": "tool_use", "id": "x", "name": name, "input": inp}
    events = [{"type": "assistant", "parent_tool_use_id": "ag", "message": {"content": [use]}}]
    assert _run_examples().read_own_example(events, "estimate-fermi") is own


def _framed(report):
    """How the host hands a subagent's final report back: a frame line, then two-space indent."""
    body = "".join(f"  {line}\n" for line in report.splitlines())
    return "[Subagent hand-back] The text below is the final report...\n" + body


def test_trace_reads_a_report_returned_in_the_handback(tmp_path):
    """Asked for markdown back, the agent returns the report instead of writing a file."""
    tr = _trace()
    base = {"session_id": "s", "agent_type": FP, "agent_id": "a1", "cwd": str(tmp_path / "gone")}
    tr.handle({**base, "hook_event_name": "SubagentStart"}, tmp_path, now="2026-09-30T10:00:00Z")
    end = tr.handle(
        {**base, "hook_event_name": "SubagentStop", "last_assistant_message": _framed(ANALYSIS)},
        tmp_path,
        now="2026-09-30T10:05:00Z",
    )
    assert end["analysis_source"] == "handback" and end["analysis_exact"] is True
    assert end["decisions"]["sections"] == tr.parse_analysis(ANALYSIS)["sections"]


def test_a_pointer_message_is_not_a_report():
    assert _trace().handback_report("The full analysis is in `.first-principles/a.md`.") is None


def test_run_examples_counts_a_report_returned_inline_as_complete():
    done = {"type": "tool_result", "tool_use_id": "ag", "content": _framed(ANALYSIS)}
    agent = {"type": "tool_use", "id": "ag", "name": "Agent", "input": {"subagent_type": FP}}
    events = [
        {"type": "assistant", "message": {"content": [agent]}},
        {"type": "user", "message": {"content": [done]}},
        {"type": "result", "is_error": False, "result": "ok"},
    ]
    re_ = _run_examples()
    assert re_.completeness(events, []) == ("complete", [])
    assert re_.inline_report(events).startswith("# First-Principles Analysis")


@pytest.mark.parametrize(
    ("line", "fired"),
    [
        ("No re-entry edge fired in this run.", False),
        ("No bounded re-entry edge fired.", False),
        ("**Re-entry edges fired:** none.", False),
        ("**Re-entry edges:** none fired.", False),
        ("Run disclosures: no re-entry edge fired; the gate ran.", False),
        # a "no" elsewhere in the sentence is not a denial (2026-09-30 heat-pump run)
        (
            "Disclosed: One bounded re-entry edge fired: the Self-Audit Gate's Fix/Repeat loop. "
            "It was triggered because unsuffixed ground truths fed no HIGH chain.",
            True,
        ),
        ("Re-entry edge fired: the Self-Audit Gate's Fix/Repeat loop, once.", True),
    ],
)
def test_parse_analysis_reads_the_re_entry_disclosure(line, fired):
    assert _trace().parse_analysis(line + "\n")["re_entry"]["fired"] is fired
