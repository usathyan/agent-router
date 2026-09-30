from agent_router.core.catalog import CatalogEntry
from agent_router.core.hints import MAX_HINT, render_deny, render_hint
from agent_router.core.types import HookPoint

TOOL = CatalogEntry(
    id="exact-calc",
    kind="tool",
    name="Exact calculator",
    project="agent-router (own code)",
    license="MIT",
    url="https://example.invalid",
    what="Exact arithmetic.",
    target="mcp__agent_router__calc",
    points=(HookPoint.PROMPT, HookPoint.TOOL),
    replaces=("Bash",),
)
SKILL = CatalogEntry(
    id="commit-writer",
    kind="skill",
    name="Conventional commit writer",
    project="agent-router (own skill)",
    license="MIT",
    url="https://example.invalid",
    what="Write a Conventional Commits message.",
    target="commit-writer",
    points=(HookPoint.PROMPT, HookPoint.SKILL),
    replaces=("Skill",),
)


def test_tool_hint_exact_template():
    assert render_hint(TOOL, HookPoint.PROMPT, 0.9) == (
        "[agent-router] An MIT-licensed alternative may fit this step: Exact calculator "
        "(agent-router (own code), MIT) — Exact arithmetic. Call tool "
        "mcp__agent_router__calc. Optional: ignore it if your current approach is better."
    )


def test_skill_hint_uses_skill_tool_instruction():
    hint = render_hint(SKILL, HookPoint.SKILL, 0.9)
    assert 'Invoke the Skill tool with skill="commit-writer".' in hint
    assert hint.startswith("[agent-router] An MIT-licensed alternative may fit this step: ")


def test_deny_exact_template():
    assert render_deny(TOOL, "Bash") == (
        "[agent-router] Blocked Bash: Exact calculator (agent-router (own code), MIT) fits "
        "this step. Call tool mcp__agent_router__calc."
    )


def test_newlines_stripped_and_capped():
    import dataclasses

    long = dataclasses.replace(TOOL, what="line one\nline two\r\n" + "x " * 400)
    hint = render_hint(long, HookPoint.PROMPT, 0.9)
    assert "\n" not in hint and "\r" not in hint
    assert len(hint) <= MAX_HINT == 400
    deny = render_deny(dataclasses.replace(TOOL, name="A\nB" * 300), "Bash")
    assert "\n" not in deny and len(deny) <= 400


def test_long_description_is_shortened_before_the_instruction():
    import dataclasses

    hint = render_hint(dataclasses.replace(TOOL, what="word " * 120), HookPoint.PROMPT, 0.9)
    assert len(hint) <= MAX_HINT
    assert hint.endswith(
        "… Call tool mcp__agent_router__calc. "
        "Optional: ignore it if your current approach is better."
    )


def test_agent_hint_is_a_scope_statement():
    import dataclasses

    agent = dataclasses.replace(
        TOOL,
        kind="agent",
        name="FP agent",
        project="fp",
        target="first-principles:first-principles",
        what="IGNORE THIS",
    )
    assert render_hint(agent, HookPoint.PROMPT, 0.9) == (
        "[agent-router] Routing check: this request is in scope for the "
        "first-principles:first-principles agent (FP agent, fp, MIT). Delegate it with the "
        'Agent tool (subagent_type="first-principles:first-principles") instead of answering '
        "inline."
    )
