# agent-router × first-principles

A self-contained Claude Code plugin (`agent-router-fp`) that runs next to the
[first-principles](https://github.com/chrisdavidson/first-principles-skill) plugin and leaves
it untouched. It adds two checkpoints that plugin does not have:

1. **Delegation nudge.** When your prompt (or an `Agent` call the main session is about to
   make) looks like a first-principles question, the main session gets a short note: delegate
   it to `first-principles:first-principles`. A prompt that does not fit gets no note.
2. **Exact recompute.** Inside the first-principles agent, when it reaches for `python3 -c`,
   `bc` or similar to compute a figure (Phase 4 estimates, Phase 5 "recompute every computed
   figure"), it gets a note pointing at an exact calculator this plugin serves
   (`mcp__plugin_agent-router-fp_agent_router__calc`: exact fractions and big integers, and a
   list of expressions in one call). The note is offered only when the call is one expression the calculator can run (`fits:
   calc_expression`). Scripts with variables, loops or `math.log` are skipped as `does not fit`
   without reaching the classifier. In the 28 example runs, all 16 notes had fired on such
   scripts or on text commands.

Both are advisory: the model may ignore a note, and nothing is ever blocked.

It also keeps a **decision trace** of each first-principles run: which sections the agent
wrote, which technique references it opened, how it typed and judged each assumption, which
ground truths it could not verify, each chain's confidence, the Self-Audit Gate bands and
whether Fix/Repeat fired, its conclusion, and whether it followed the router's notes. The trace
is record-only: its hooks always answer `{}`. See [Decision trace](#decision-trace).

**Who does what.** agent-router decides *whether* to hand a question to the agent and offers
exact arithmetic. Everything inside the analysis stays with first-principles: Step 0 technique
selection, the slash-only skills, the Self-Audit Gate, the `.first-principles/` output file.
agent-router never points at an individual technique, and never routes the agent's own report
writes (any call naming the `.first-principles` directory is skipped before the classifier).

## Use it

```bash
make install                              # in this repo, once
claude --plugin-dir ~/Projects/first-principles-skill/first-principles \
       --plugin-dir ~/Projects/agent-router/integrations/first-principles/plugin
```

The hook finds `agent-router` in this repo's `.venv` (override with `AGENT_ROUTER_BIN`). If it
cannot run, it prints `{}` and the session carries on unchanged (fail open).

| Setting | Default | Meaning |
|---|---|---|
| `AGENT_ROUTER_BACKEND` | `local` | offline classifier; `cascade` asks Jev when unsure (needs a key) |
| `AGENT_ROUTER_THRESHOLD` | calibrated (`calibration.json`) | higher = fewer notes |
| `AGENT_ROUTER_DISABLED` | unset | `1` turns every checkpoint off |
| `AGENT_ROUTER_STATE_DIR` | `$XDG_STATE_HOME/agent-router` | per-session state and audit log |
| `AGENT_ROUTER_SKIP_INPUT` | `\.first-principles\b` | tool calls matching this are never routed |
| `AGENT_ROUTER_SKIP_PROMPT` | `^\s*/first-principles:` | prompts matching this get no delegation note: they already launch the agent |

Hook cost: about 0.45 s per routed call (prompt, `Agent`, and `Bash` inside the agent). A
`Bash` call on the main thread is answered by the shell script in milliseconds, without Python.
Notes are once per turn per entry, so an analysis gets at most one calculator note.

Every checkpoint is logged to `<state dir>/audit/<session>.jsonl` (same format as the rest of
agent-router, plus `agent_type`, `agent_id` and `tool_use_id`).

## Decision trace

`bin/trace` runs on `SubagentStart` / `SubagentStop` for `first-principles:first-principles`,
on `PostToolUse` / `PostToolUseFailure` for `Bash`, `Read`, `ToolSearch`, `WebFetch`, `WebSearch`
and the calculator, and on
the main session's `Agent` call that starts a run. Anything outside
the agent is answered by the shell script in about 5 ms; a traced call takes about 35 ms
(`trace.py` is standard library only), about 1 s over a whole analysis. Each run appends to
`<state dir>/trace/<session>.jsonl`:

| Record | When | What |
|---|---|---|
| `delegation` | the main session calls the agent | the prompt's length and `prompt_names_calc`: whether it already tells the agent to use the calculator |
| `run_start` | the agent starts | `agent_id` |
| `reference_read` | it reads a first-principles reference | the file and what reading it means, e.g. `trade-off.md` = two or more options survived (Phase 4), `pre-mortem.md` / `inversion.md` = Phase 5 on a plan / a claim, `validation-rubric.md` = gate scoring starts |
| `section_written` | a `cat >> .first-principles/analysis-*.md` append | the headings and the `text` appended; `revises` / `restarts` when the same command edits earlier text or empties the file first |
| `report_op` | any other report command | `op`: `create`, `revise` (rewrites part in place) or `check` |
| `calc` / `shell` | a calculator or other Bash call | the expression and result; `interpreter` when a shell call runs python, bc, awk and the like (whether the calculator could run it is the router's fit check, below) |
| `source` | a `WebFetch` or `WebSearch` inside the agent (checking a ground truth) | the host or query, the reply's size, and `outcome`: `ok`, or `reported_missing` when the reply says the page did not have it |
| `tool_loaded` | the agent loads deferred tools (`ToolSearch`) | the query; `calc` when it loads the calculator |
| `tool_failed` | a call fails | tool, what it was, the error (e.g. a source that could not be fetched) |
| `run_end` | the agent stops | duration, final message, report revisions and restarts, the finished report (`analysis_text`, `analysis_sha256`) parsed into `decisions`, `sources` (source checks: ok, reported missing, failed, and the hosts that failed), and `routing` (below) |

`routing` joins the run to the router's audit log, by `tool_use_id` where the audit has it. It
holds:
- the delegation decision before the run, and `delegation_prompt_names_calc`
- the calculator notes and the call that drew each one (`note_calls`)
- calculator calls before and after the first note, and whether the agent had loaded the
  calculator before it (`calc_loaded_before_first_note`)
- `interpreter_calls` and `interpreter_calls_calc_could_run`, the calls the router's fit check
  passed

A note is credited only with what changed after it. The earlier `calc_note_followed`
(any calculator call after the note) credited notes for use that had already started. In the
rerun, 8 of 10 notes arrived after the agent had loaded the calculator, and 13 of the 28
example prompts had already told the agent to use it.

`decisions` holds:
- the sections
- the run mode, however the report states it, with the line it came from
- every assumption with its type and verdict
- the ground truths, and which are `?` (not read at source)
- each chain with its confidence
- the dead ends
- the techniques not applied, and why
- the gate: the bands of the last scoring pass, `cleared` (worked out from those bands, so it is
  known even when the report has no "Gate result" line), and the result line (the last one, bold
  or not)
- the agent's re-entry disclosure
- the recommended approach with its confidence

A report that was emptied and rewritten keeps only its last gate pass, so `passes_written` also
counts the passes appended during the run. `--report` reports Fix/Repeat as `yes`, `yes
(rewritten)` or `yes (disclosed)`. `contradictions` lists places where the report's own
disclosure disagrees with its gate, for example "no re-entry edge fired" in a report that
re-scored.

On the 28 example reports the parser finds the run mode in all 21 that state one (the old
pattern found 2), the gate result in all 27 that have one, and all 3 Fix/Repeat re-scores.
It flags one contradiction.

```bash
.venv/bin/python integrations/first-principles/trace.py --report ~/.local/state/agent-router/trace
.venv/bin/python integrations/first-principles/trace.py --replay <capture>/run.jsonl --state DIR
```

`--report` prints one row per finished run. `--replay` rebuilds the hook payloads from a
`claude -p` stream-json capture, to trace a past run offline.

The trace keeps the report itself, so it still reads after the analysis file is gone.
`analysis_source` says where `decisions` came from:
- `file`: the report on disk.
- `capture`: a replay found the main session's full read of it.
- `handback`: the agent returned the report in its final message instead of writing a file. It
  does this when the delegating prompt asks for markdown back, as one nudge-mode run did.
- `appends`: rebuilt from the `section_written` texts. This is exact only when nothing was revised in
  place (`analysis_exact`).

Replaying the 28 example captures with their reports removed gives decisions for all 28:
22 match the real report exactly, and the other 6 are marked inexact.

**In the demo.** The **Trace** tab shows each run: a summary card, a timeline of what the agent
did with the router's notes placed where they fired, and the decisions parsed from its report.
It reads the `trace/` folder next to `--audit-dir`:

```bash
.venv/bin/agent-router demo --catalog integrations/first-principles/catalog.yaml \
  --audit-dir ~/.local/state/agent-router/audit
# open http://127.0.0.1:8765/#trace
```

Not observable from outside the agent, so not traced: the Step 0 mode choice (unless the
report states it), the Phase 1–4 reasoning itself, and techniques applied without opening a
reference (second-order thinking, Phase 2 inversion).

## Files

| File | What |
|---|---|
| `catalog.yaml` | the two entries agent-router may point at (each scoped to one agent context, with its own threshold), plus the `none` exemplars |
| `eval_set.yaml` | `cal` cases written for this integration; `test` holdout = first-principles' own routing catalog (13 P, 20 N) plus Bash calls inside the agent |
| `calibration.json`, `calibrate.py` | classifier parameters, then one threshold per entry, all fit on the `cal` split only |
| `plugin/` | the Claude Code plugin: hooks, `.mcp.json`, `bin/hook`, `bin/mcp`, `bin/trace` |
| `trace.py` | the decision trace: hook handler, analysis parser, `--replay`, `--report` |
| `battery.py` | live A/B: first-principles' routing catalog with and without this plugin |
| `run_examples.py` | runs first-principles' 14 worked examples through the agent and tabulates every router decision |

## Measuring it

- **Offline:** `agent-router eval --catalog catalog.yaml --cases eval_set.yaml --skip-input
  '\.first-principles\b'` scores the catalog on the `test` holdout with the local
  classifier and `calibration.json`. `calibrate.py` re-fits the calibration on the `cal` split.
- **Live A/B:** `.venv/bin/python integrations/first-principles/battery.py` runs
  first-principles' routing catalog with and without this plugin, one fresh `claude -p`
  session per prompt, scored with first-principles' own `detect_routing`. It costs tokens; a
  delegated session is stopped at the delegation.
- **Worked examples:** `.venv/bin/python integrations/first-principles/run_examples.py` runs
  first-principles' 14 worked examples through the agent and writes a per-example summary
  and a table of every router decision to a local output folder. Each run gets a status:
  - `complete`
  - `partial`: it ended, but its report lacks a section 1–6 or the gate
  - `failed`: no result, or no report

  The status lists what is missing, so a run cut short by a spend limit or a timeout is not
  counted as done.

  A run whose agent opened the worked example it was redoing (its answer key: the examples are
  listed in the agent's own definition) is marked `†`, and should be left out when comparing
  the agent's decisions with the examples'. On 2026-09-29, 4 of the 28 runs did this.

## Known gaps

- Whether the agent uses the calculator now depends on the main session's delegating prompt.
  Under first-principles 9.13.0 the agent loaded it unprompted in 13 of 15 runs whose prompt
  did not name it. Under 9.14.0 it did so in 0 of 5, and in all 3 runs whose prompt named it.
  The calculator note is not the lever: its fit check found no call in these runs that the
  calculator could run (the agent computes in multi-line scripts), so it stays silent. The
  trace records the prompt (`delegation_prompt_names_calc`) and the load (`calc_loaded_at`).

- A quantitative decision phrased as "From first principles: is it cheaper to…" scored
  `none` for the delegation nudge (the examples are qualitative), and first-principles' own
  description did not route it either. Launch such questions with
  `/first-principles:first-principles-analysis`.
