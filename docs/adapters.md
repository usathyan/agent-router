# Host adapters

The router core never imports a host SDK. An adapter does two translations:

1. host hook input → `RouterEvent(point, session_id, turn_id, text, tool_name, tool_input, recent)`
2. `Decision(action, reason, entry_id, hint, ...)` → host hook output

| `Decision.action` | Meaning | Host output |
|---|---|---|
| `suggest` | advisory hint; the agent decides | inject `hint` as model-visible context |
| `enforce` | native call denied | deny the tool call with `hint` (the templated deny text) |
| `native` / `skipped` | do nothing | empty output (exit 0) |

Enforcement only applies at the `tool` point. At `prompt` and `skill` the router only suggests,
even in enforce mode: denying a prompt would discard the user's message on every host.
Never forward `Decision.reason` to the model: it can hold exception text from a backend. Only
`hint` is built from catalog fields.

## Mapping

| RouterEvent | Claude Agent SDK (Python) | Codex CLI hooks | Gemini CLI hooks |
|---|---|---|---|
| `point=prompt` | `UserPromptSubmit` | `UserPromptSubmit` | `BeforeAgent` |
| `point=tool` | `PreToolUse` | `PreToolUse` | `BeforeTool` |
| `point=skill` | `PreToolUse` with `tool_name == "Skill"` | no skill tool event (n/a) | no skill tool event (n/a) |
| `text` (prompt) | `input["prompt"]` | stdin `prompt` | stdin `prompt` |
| `tool_name` | `input["tool_name"]` | stdin `tool_name` (`Bash`, `apply_patch`, MCP names) | stdin `tool_name` |
| `tool_input` | `input["tool_input"]`; for Skill `{"skill": ..., "args": ...}` | stdin `tool_input` | stdin `tool_input` |
| `session_id` | `input["session_id"]` | stdin `session_id` | stdin `session_id` |
| `turn_id` | adapter counter, bumped on each `UserPromptSubmit` | stdin `turn_id` | adapter counter, bumped on each `BeforeAgent` |
| suggest (prompt) | `hookSpecificOutput: {hookEventName: "UserPromptSubmit", additionalContext}` | stdout `{"hookSpecificOutput": {"hookEventName": "UserPromptSubmit", "additionalContext": ...}}` | stdout `{"hookSpecificOutput": {"additionalContext": ...}}` (appended to the prompt for this turn) |
| suggest (tool) | `hookSpecificOutput: {hookEventName: "PreToolUse", additionalContext}` | stdout `hookSpecificOutput.additionalContext` | no model-visible context on `BeforeTool`; use `systemMessage` (user-facing only) or skip |
| enforce (tool) | `hookSpecificOutput: {hookEventName: "PreToolUse", permissionDecision: "deny", permissionDecisionReason}` | stdout `{"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny", "permissionDecisionReason": ...}}` (or exit 2 + stderr) | stdout `{"decision": "deny", "reason": ...}` (reason returns to the agent as a tool error; or exit 2 + stderr) |
| transport | in-process Python callback (`HookMatcher`) | command hook: JSON on stdin, JSON on stdout | command hook: JSON on stdin, JSON on stdout |
| config | `ClaudeAgentOptions(hooks={...})` | `~/.codex/hooks.json` or `<repo>/.codex/hooks.json` (also `[hooks]` in `config.toml`): event → matcher → `hooks[]{type, command, timeout}` | `settings.json` `"hooks"`: event → `[{matcher, hooks: [{type: "command", command, timeout}]}]` |

### Verified vs. assumed

- **Claude Agent SDK** — verified on SDK 0.2.160 / CLI 2.1.283: `UserPromptSubmit` input has
  `prompt`; `PreToolUse` input has `tool_name`, `tool_input`, `tool_use_id`; Skill calls arrive
  as `tool_name="Skill"`, `tool_input={"skill": ..., "args": ...}`; `additionalContext` reaches
  the model for both hooks; `permissionDecision: "deny"` + `permissionDecisionReason` blocks.
- **Codex CLI** — read from the official hooks page on 2026-09-26
  (developers.openai.com/codex/hooks, which now redirects to learn.chatgpt.com/docs/hooks):
  event names, config locations, stdin fields (`session_id`, `transcript_path`, `cwd`,
  `hook_event_name`, `model`, `turn_id`, `permission_mode`, plus `prompt` /
  `tool_name`, `tool_use_id`, `tool_input`), the output shapes above, exit-code semantics
  (0 ok, 2 blocks with stderr as reason), `additionalContext` on `PreToolUse`, and that
  `PreToolUse` covers `Bash`, `apply_patch`, MCP and local function tools but not hosted
  tools such as web search. `permissionDecision: "ask"` is documented as parsed but not
  supported. Hooks are on by default (`[features] hooks = false` disables them).
  **Not run** against a Codex binary here.
- **Gemini CLI** — read from geminicli.com/docs/hooks/reference on 2026-09-26: event names,
  `settings.json` shape, `BeforeAgent` input `prompt`, `BeforeTool` input `tool_name`,
  `tool_input` (plus `mcp_context`, `original_request_name`), common output fields
  (`decision`, `reason`, `continue`, `systemMessage`, `hookSpecificOutput`),
  `BeforeAgent` `hookSpecificOutput.additionalContext` is appended to the prompt,
  `BeforeTool` `hookSpecificOutput` accepts `tool_input` (argument rewrite), and exit codes
  (0 parse stdout, 2 block with stderr, other = warning). **Assumed:** no model-visible
  context channel on `BeforeTool` (none is documented) and that `session_id` is stable
  across a session. **Not run** against a Gemini binary here.

## Claude Code command hooks (implemented)

`agent-router hook` is the command-hook adapter for Claude Code itself
(`adapters/claude_code.py`): hook JSON on stdin, the same outputs as the SDK column above on
stdout, always exit 0. Each hook call is a fresh process, so recent prompts and the
once-per-turn hint set live in a per-session state file. `prompt_id` gives the turn, and
`agent_type` (present on calls made inside a subagent) scopes entries with `agents:`.
`agent-router mcp` serves the catalog tools over stdio for a plugin's `.mcp.json`. They
surface as `mcp__plugin_<plugin>_agent_router__<tool>` and are deferred until loaded with
ToolSearch. Worked example: `integrations/first-principles/`.

## stdin/stdout shim sketch (Codex / Gemini command hooks)

```python
#!/usr/bin/env python3
"""agent-router command hook: reads host JSON on stdin, writes host JSON on stdout."""
import json, sys, zlib
from agent_router.core.catalog import load_catalog
from agent_router.core.router import Router
from agent_router.core.types import Action, HookPoint, RouterEvent
from agent_router.deciders.registry import make_decider

HOST = sys.argv[1]  # "codex" | "gemini"
POINTS = {"UserPromptSubmit": HookPoint.PROMPT, "BeforeAgent": HookPoint.PROMPT,
          "PreToolUse": HookPoint.TOOL, "BeforeTool": HookPoint.TOOL}
inp = json.load(sys.stdin)
event_name = inp["hook_event_name"]
point = POINTS[event_name]
turn = str(inp.get("turn_id", "0"))  # Codex sends one; Gemini: keep a per-session counter
ev = RouterEvent(point=point, session_id=inp.get("session_id", ""),
                 turn_id=int(turn) if turn.isdigit() else zlib.crc32(turn.encode()),
                 text=inp.get("prompt", ""), tool_name=inp.get("tool_name"),
                 tool_input=inp.get("tool_input"))
catalog = load_catalog()
d = Router(catalog, make_decider("local", catalog)).route(ev)
out = {}
if d.action is Action.ENFORCE and point is HookPoint.TOOL:
    out = ({"decision": "deny", "reason": d.hint} if HOST == "gemini" else
           {"hookSpecificOutput": {"hookEventName": event_name, "permissionDecision": "deny",
                                   "permissionDecisionReason": d.hint}})
elif d.action is Action.SUGGEST and d.hint and not (HOST == "gemini" and point is HookPoint.TOOL):
    out = {"hookSpecificOutput": {"hookEventName": event_name, "additionalContext": d.hint}}
print(json.dumps(out))  # exit 0; any router failure should also print {} and exit 0 (fail open)
```

**Assumed:** Gemini ignores the extra `hookEventName` key (drop it for Gemini if not).
A command hook is a fresh process per event, so the router's per-turn suggestion dedupe and
any turn counter must be persisted by the shim (e.g. a small file keyed by `session_id`);
the in-process Claude Agent SDK adapter does not need this. Wrap the body in `try/except` and print `{}` on
any error so a router failure never blocks the host (fail open to native).
