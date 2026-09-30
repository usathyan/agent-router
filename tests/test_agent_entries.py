"""Agent-kind entries, per-agent scoping and the skip pattern (first-principles integration)."""

from dataclasses import replace

from agent_router.core.catalog import MAIN_AGENT, Catalog, CatalogEntry, load_catalog
from agent_router.core.config import RouterConfig
from agent_router.core.hints import render_hint
from agent_router.core.router import Router
from agent_router.core.types import Action, ChoiceResult, HookPoint, RouterEvent

FP_AGENT = "first-principles:first-principles"
DELEGATE = CatalogEntry(
    id="fp-agent",
    kind="agent",
    name="First-principles analysis agent",
    project="first-principles-skill",
    license="MIT",
    url="https://example.invalid",
    what="Decompose a claim or design into ground truths and reason upward.",
    target=FP_AGENT,
    points=(HookPoint.PROMPT, HookPoint.TOOL),
    replaces=("Agent",),
    agents=(MAIN_AGENT,),
)
CALC = CatalogEntry(
    id="exact-calc",
    kind="tool",
    name="Exact calculator",
    project="agent-router (own code)",
    license="MIT",
    url="https://example.invalid",
    what="Exact arithmetic.",
    target="mcp__plugin_agent-router-fp_agent_router__calc",
    points=(HookPoint.TOOL,),
    replaces=("Bash",),
    agents=(FP_AGENT,),
)
CATALOG = Catalog(version="fp-test", entries=(DELEGATE, CALC))


class Always:
    """A decider that always picks ``choice`` with p=0.9."""

    name = "fake"

    def __init__(self, choice):
        self.choice = choice
        self.calls = 0

    def decide(self, state, options):
        self.calls += 1
        probs = {k: (0.9 if k == self.choice else 0.1 / (len(options) - 1)) for k in options}
        return ChoiceResult(choice=self.choice, probabilities=probs, confidence=0.8)


def ev(point=HookPoint.PROMPT, tool_name=None, tool_input=None, agent_type=None, turn=1):
    return RouterEvent(
        point=point,
        session_id="s",
        turn_id=turn,
        text="challenge the assumptions behind our pricing",
        tool_name=tool_name,
        tool_input=tool_input,
        agent_type=agent_type,
    )


def test_eligible_scopes_entries_by_agent_context():
    assert CATALOG.eligible(HookPoint.PROMPT) == [DELEGATE]
    assert CATALOG.eligible(HookPoint.TOOL, "Agent") == [DELEGATE]
    assert CATALOG.eligible(HookPoint.TOOL, "Bash") == []  # calc is FP-agent only
    assert CATALOG.eligible(HookPoint.TOOL, "Bash", FP_AGENT) == [CALC]
    assert CATALOG.eligible(HookPoint.TOOL, "Agent", FP_AGENT) == []  # no nested nudge


def test_unscoped_entry_applies_everywhere():
    free = replace(CALC, agents=())
    assert free.applies_to(None) and free.applies_to("anything")


def test_agent_hint_says_how_to_delegate():
    hint = render_hint(DELEGATE, HookPoint.PROMPT, 0.9)
    assert f'Delegate it with the Agent tool (subagent_type="{FP_AGENT}")' in hint


def test_plugin_tool_hint_says_how_to_load_deferred_tool():
    hint = render_hint(CALC, HookPoint.TOOL, 0.9)
    assert f'Load it with ToolSearch "select:{CALC.target}" if deferred, then call it.' in hint


def test_prompt_suggests_delegation():
    d = Router(CATALOG, Always("fp-agent")).route(ev())
    assert d.action == Action.SUGGEST and d.entry_id == "fp-agent"


def test_agent_entry_is_never_enforced():
    router = Router(CATALOG, Always("fp-agent"), RouterConfig(mode="enforce"))
    d = router.route(ev(HookPoint.TOOL, "Agent", {"subagent_type": "general-purpose"}))
    assert d.action == Action.SUGGEST


def test_delegating_to_the_target_agent_is_the_loop_guard():
    decider = Always("fp-agent")
    d = Router(CATALOG, decider).route(ev(HookPoint.TOOL, "Agent", {"subagent_type": FP_AGENT}))
    assert d.action == Action.SKIPPED and d.reason == "own tool" and decider.calls == 0


def test_plugin_served_own_tool_is_skipped():
    decider = Always("exact-calc")
    name = "mcp__plugin_whatever_agent_router__json_query"
    d = Router(CATALOG, decider).route(ev(HookPoint.TOOL, name, {}, agent_type=FP_AGENT))
    assert d.action == Action.SKIPPED and decider.calls == 0


def test_skip_pattern_stops_matching_tool_calls_before_the_decider():
    decider = Always("exact-calc")
    router = Router(CATALOG, decider, RouterConfig(skip_input=r"\.first-principles/"))
    write = {"command": 'cat >> ".first-principles/analysis-1.md" <<EOF\n2**10 = 1024\nEOF'}
    d = router.route(ev(HookPoint.TOOL, "Bash", write, agent_type=FP_AGENT))
    assert d.action == Action.SKIPPED and d.reason == "skip pattern" and decider.calls == 0
    d = router.route(
        ev(HookPoint.TOOL, "Bash", {"command": "python3 -c 'print(2**10)'"}, agent_type=FP_AGENT)
    )
    assert d.action == Action.SUGGEST and d.entry_id == "exact-calc"


def test_skip_pattern_ignores_the_prompt_text():
    router = Router(CATALOG, Always("fp-agent"), RouterConfig(skip_input="assumptions"))
    assert router.route(ev()).action == Action.SUGGEST


def test_invalid_skip_pattern_fails_open():
    router = Router(CATALOG, Always("exact-calc"), RouterConfig(skip_input="("))
    d = router.route(ev(HookPoint.TOOL, "Bash", {"command": "bc"}, agent_type=FP_AGENT))
    assert d.action == Action.SUGGEST


def test_skip_input_from_env(monkeypatch):
    monkeypatch.setenv("AGENT_ROUTER_SKIP_INPUT", r"\.first-principles/")
    assert RouterConfig.from_env().skip_input == r"\.first-principles/"


def test_catalog_loads_agent_kind_and_agents(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text(
        """
version: t
entries:
  - {id: fp, kind: agent, name: n, project: p, license: MIT, url: u, what: w,
     target: first-principles:first-principles, points: [prompt], agents: [main]}
"""
    )
    (entry,) = load_catalog(path).entries
    assert entry.kind == "agent" and entry.agents == ("main",)


def test_entry_threshold_overrides_the_router_threshold():
    strict = replace(DELEGATE, threshold=0.95)
    router = Router(Catalog(version="t", entries=(strict, CALC)), Always("fp-agent"))
    d = router.route(ev())
    assert d.action == Action.NATIVE and "below threshold 0.950" in d.reason
    assert d.threshold == 0.95  # the entry's bar, not the router's, travels with the decision
    loose = replace(DELEGATE, threshold=0.1)
    router = Router(
        Catalog(version="t", entries=(loose, CALC)),
        Always("fp-agent"),
        RouterConfig(threshold=0.99),
    )
    d = router.route(ev())
    assert d.action == Action.SUGGEST and d.threshold == 0.1
    assert "at or above threshold 0.100" in d.reason


def test_catalog_threshold_is_validated(tmp_path):
    import pytest

    from agent_router.core.catalog import CatalogError

    path = tmp_path / "c.yaml"
    path.write_text(
        "version: t\nentries:\n  - {id: a, kind: agent, name: n, project: p, license: MIT,"
        " url: u, what: w, target: t, points: [prompt], threshold: 1.5}\n"
    )
    with pytest.raises(CatalogError, match="threshold"):
        load_catalog(path)
