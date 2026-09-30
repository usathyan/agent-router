# agent-router

**A checkpoint for your AI agent.** Every time the agent is about to act, agent-router stops it for a
moment, looks at what it's about to do, and checks it against a short, fixed list of approved tools.

MIT licensed · works with the [Claude Agent SDK](https://github.com/anthropics/claude-agent-sdk-python) today ·
designed to also plug into Codex CLI and Gemini CLI.

![The demo: one agent step, judged by every classifier](docs/img/demo-compare.png)

## The idea, explained simply

Think of an AI agent as a very capable new employee who improvises. Ask it to add two numbers and it
might write a Python script, run it in a shell, and read the output. That usually works, but it's
different every time, and you can't easily predict or audit it.

agent-router puts a **gatekeeper at every door the agent walks through**:

1. **When you ask it something.** This is your prompt.
2. **When it's about to use a tool**, like running a shell command or reading a file.
3. **When it's about to use a skill**, which is a packaged set of instructions.

At each door, the gatekeeper asks one question: *"Is there an approved tool on our list for this?"*
It can only answer by **pointing at one item on the list**, or by saying **"none, carry on"**. It never
writes instructions or makes up commands. Pointing is all it can do.

- If an approved tool fits, the agent gets a short note that names it (for example, the exact
  calculator) and says how to call it. In **enforce** mode, the gatekeeper can also block the agent's
  improvised approach, such as a shell command.
- If nothing fits, the agent carries on as usual, and nothing changes.
- Every stop is written to a log, so you can replay exactly what was checked and why.

### Why: a more predictable inner agent

agent-router is meant to make the agent inside a harness (the "inner Claude") **more deterministic**.
Claude Code and the Agent SDK already have *hooks*: places where your code can run when the agent acts.
Hooks are the doors. agent-router is the **gatekeeper that stands at those doors**. It makes sure every
decision is inspected, and it steers the agent toward a predefined set of approved tools and
instructions, instead of whatever it would have improvised.

**What that means today, precisely:**

| | Today |
|---|---|
| Every prompt, tool call and skill call is inspected | ✅ Yes, and each one is logged |
| The gatekeeper can only pick from a fixed, approved list | ✅ Yes (every item must be MIT-licensed) |
| The same step gets the same routing answer | ✅ Yes for the offline classifier; near-identical with Jev |
| The agent is *told* the approved way | ✅ Yes (advisory mode) |
| The agent is *stopped* from using a shell or file tool when an approved tool fits | ✅ Yes, in enforce mode |
| The agent is *forced* to then use the approved tool | ❌ Not yet. It's blocked and pointed at it, but it chooses its next move |
| The agent is made to follow a predefined **graph** of steps (step A, then B, then C) | ❌ Not yet. Today each step is judged on its own. This is the planned next stage |
| If the gatekeeper breaks, the agent is blocked | ❌ No, on purpose: a failure lets the step through so the agent never gets stuck |

## Try the demo in 3 minutes

You need Python 3.11+, [uv](https://docs.astral.sh/uv/), and (for the live agent) a Claude Code login.

```bash
git clone https://github.com/usathyan/agent-router && cd agent-router
make install                          # one-time setup
export OPENROUTER_API_KEY=sk-or-...   # optional: lets the real Jev classifier double-check
make run                              # open http://127.0.0.1:8765
```

No API key? Everything still works, using a free classifier that runs on your machine.

The page has four tabs:

| Tab | What it is | Costs |
|---|---|---|
| **Playground** | Test the gatekeeper on one step. You describe the step, it shows its decision. No agent runs. | Free (a fraction of a cent if Jev is asked) |
| **Live agent** | Give a real Claude agent a task and watch every checkpoint happen, step by step. | A few cents per run |
| **Replay** | Step back through any earlier run, one checkpoint at a time. | Free |
| **Trace** | What an integrated agent did and decided in a run (for now, the first-principles agent): sections written, references opened, calculator use against the router's notes, the parsed conclusion, assumptions and Self-Audit Gate. | Free |

### How to read a decision

![A live run: the checkpoint, the decision bars, the note the agent receives](docs/img/demo-live.png)

- **The bars** show how strongly the gatekeeper leans toward each approved tool, and toward
  **none** (let the agent carry on). The dashed line is the bar a choice must clear before anyone acts.
- **"Escalated to Jev"** means the free local classifier wasn't sure it could say "none", so a second,
  more careful classifier (Jev) confirmed the answer. **"Answered locally"** means no second opinion
  was needed.
- **"What the agent sees"** is the exact note added to the agent's context. It always comes from the
  approved list, never from the classifier itself.
- **Badges:** **Hint injected** means the agent got the note. **Blocked** means enforce mode stopped the
  agent's own approach. **Native path** means nothing fit, so nothing changed.

## Test the scenarios

Each scenario is one click in the **Playground** (the buttons under *Examples*), then **Route this step**.

| # | Scenario | Click | What should happen |
|---|---|---|---|
| 1 | The user asks for exact math | **17% of 2,340** | Points to the **Exact calculator**; a note is added |
| 2 | The agent is about to do math in a shell | **python3 -c 2\*\*200** | Points to the calculator instead of running Python in a shell |
| 3 | The agent is about to read a saved web page | **Read release.html** | Points to **HTML to Markdown** |
| 4 | The user asks a question about a JSON file | **failed orders** | Points to **JSON query** |
| 5 | The agent is about to use a skill that isn't on the list | **release-notes skill** | Points to the approved **commit-writer** skill |
| 6 | An ordinary request | **run the test suite** | Says **none**. The agent is left alone, which is the point |

Then try these variations:

- **Block instead of suggest.** Pick scenario 2, switch **Mode** to **Enforce**, and route again. The
  badge becomes **Blocked**. The agent's shell command would be refused, with the reason shown.
- **Compare classifiers.** Pick any scenario and click **Compare classifiers**. Every available
  classifier answers the same question side by side, and you can see where they agree and disagree.
- **Be stricter or looser.** Move the **Threshold** slider. Higher means the gatekeeper speaks up less
  often; lower means more often.
- **Your own step.** Type any request in *Prompt*. For a tool step, pick a tool and edit its JSON input.
- **Watch a real agent.** In **Live agent**, ask *"What is 17% of 2,340 exactly?"* and click **Run
  agent**. You'll see the note go in, the agent pick the approved calculator, and the exact answer
  come back (1989/5 = 397.8). Then open **Replay** to step through it again.
- **Run a live task in Enforce mode.** Ask *"Use python to compute 2\*\*200"* with **Enforce** on. If
  the agent reaches for the shell, that call is blocked and the agent is pointed at the calculator.
  Often the prompt note alone is enough and it goes straight to the calculator. Either way, the
  timeline shows every checkpoint.

Prefer the terminal?

```bash
make route Q="What is 17% of 2,340 exactly?"                       # one decision
.venv/bin/agent-router run "What is 2**200 exactly?" --workspace demo_workspace   # a live run
make test                                                         # 405 automated checks, offline
```

## The approved list (today)

The list is deliberately small: a demo catalog of free, MIT-licensed tools that run on your machine.

| Approved tool | Used instead of |
|---|---|
| Exact calculator | Doing math in a shell (`python -c`, `bc`) |
| JSON query | Hand-written `jq` or Python scripts to pull fields out of JSON |
| HTML to Markdown | Reading or scraping raw HTML |
| Repository stats | `wc`/`find`/`cloc` pipelines to count lines of code |
| Conventional commit writer (skill) | Other, unapproved skills for writing commit messages |

Adding your own entries is described in the [technical guide](docs/technical-guide.md#catalog).

## How well does it choose?

We measured the gatekeeper on 64 test steps it had never seen:

- **Default (with a key):** it picks the right answer **89%** of the time and **never** points at a
  tool when it shouldn't (0 false alarms). It asks Jev only when the free classifier isn't sure;
  a decision takes about 0.2 s on average.
- **Free and offline (no key):** it's right **80%** of the time, with some false alarms (15%). That's
  good for trying things out, but not as reliable.

## Learn more

- **[Technical guide](docs/technical-guide.md)**: how it works, all classifiers and their measured
  accuracy, the catalog format, modes, configuration, security, and command-line reference.
- **[Architecture](docs/architecture.md)**: diagrams of the components and of one decision.
- **[Other agent harnesses](docs/adapters.md)**: how to connect Codex CLI and Gemini CLI.
- **[With the first-principles plugin](integrations/first-principles/README.md)**: a Claude Code
  plugin that nudges delegation to the first-principles agent and gives it an exact calculator.
- **[Research](docs/research.md)**: the papers and projects this design builds on, including TypeSafe's
  Jev and Tenjin.

## Roadmap

- **Decision graphs.** A predefined map of allowed steps (A, then B or C, then D) that the gatekeeper
  enforces, so the agent must follow a workflow and not just pick approved tools step by step.
- **Force the approved tool** after a block, not just point at it.
- Adapters for **Codex CLI** and **Gemini CLI** (the mappings are designed and documented).

## License

MIT. See [LICENSE](LICENSE). Two optional pieces are not MIT: the `jev` classifier calls TypeSafe's
hosted (proprietary) model, and the `anyjev` extra installs Apache-2.0 code. Neither is required.
