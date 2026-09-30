#!/usr/bin/env python3
"""Run first-principles' worked examples through the agent, with agent-router-fp loaded.

first-principles-skill ships 14 worked examples (``shared/examples/*.md``). Each is a finished
analysis; this script takes each example's *setup* (its ``**Scenario.**`` paragraph, else the
``**Claim under analysis.**`` / ``**Target quantity.**`` / ``**Target:**`` paragraph, else its
``**Core problem:**``), asks the agent to analyse it from
scratch, and records what the agent-router checkpoints did along the way. The
first-principles repository is only read.

For each example, in ``<out>/<name>/``:
  run.jsonl          the ``claude -p`` stream-json capture
  prompt.txt         the exact prompt sent
  .first-principles/ the agent's analysis file (the agent writes it into its cwd)
and ``<out>/summary.md`` / ``summary.json``: status (complete / partial / failed, with what is
missing), delegated?, analysis file and size, calculator
notes and calls, cost, turns, duration. ``<out>/state/audit/`` holds every checkpoint;
``<out>/decisions.md`` / ``decisions.json`` tabulate them per example.
Replay them in the demo UI with ``agent-router demo --catalog
integrations/first-principles/catalog.yaml --audit-dir <out>/state/audit``.

Entry modes:
  launcher  (default) ``/first-principles:first-principles-analysis <setup>``: the agent
            always runs, so every example exercises the exact-recompute checkpoint.
  nudge     the bare setup as a prompt: whether the agent runs depends on the delegation
            nudge and first-principles' own routing.

Costs real tokens: each example is a full analysis (up to the agent's 60 turns).

Usage:
    run_examples.py [--fp-repo PATH] [--only NAME,...] [--entry launcher|nudge]
                    [--model M] [--jobs N] [--out DIR] [--timeout S] [--dry-run]
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PLUGIN = HERE / "plugin"
CALC_TOOL = "mcp__plugin_agent-router-fp_agent_router__calc"
FP_AGENT = "first-principles:first-principles"
LAUNCHER = "/first-principles:first-principles-analysis"

# Where each worked example states its setup, in order of preference; the prefix is prepended
# so the agent knows what kind of input it is (a claim to test, a quantity to estimate, ...).
_SETUP_LABELS = (
    ("Scenario.", ""),
    ("Claim under analysis.", "Evaluate this claim: "),
    ("Target quantity.", "Estimate this quantity: "),
    ("Target:", "Find the law-permitted limit for this: "),
    ("Core problem:", ""),
)


def _flat(text: str) -> str:
    return " ".join(text.split())


def _labelled(md: str, label: str) -> str | None:
    m = re.search(rf"^\*\*{re.escape(label)}\*\*\s*(.+?)(?:\n\s*\n|\Z)", md, re.S | re.M)
    return _flat(m.group(1)) if m else None


def extract_setup(md: str) -> tuple[str, str]:
    """(setup text, the label it came from) for one worked example.

    Only the setup is used, never the analysis sections, so the agent starts from the same
    input the example did.
    """
    for label, prefix in _SETUP_LABELS:
        text = _labelled(md, label)
        if text:
            return prefix + text, label
    raise ValueError("no setup paragraph found (Scenario / Claim / Target / Core problem)")


def build_prompt(setup: str, entry: str) -> str:
    ask = (
        f"{setup}\n\nAnalyse this from first principles. Show the arithmetic for every "
        "computed figure and recompute it."
    )
    return f"{LAUNCHER} {ask}" if entry == "launcher" else ask


def _trace_module():
    """trace.py, next to this script (standard library only): its report parser."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("fp_trace", HERE / "trace.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


REQUIRED_SECTIONS = tuple(range(1, 7))  # the report's ## 1. to ## 6.


def inline_report(events: list[dict]) -> str | None:
    """The analysis the agent returned in its hand-back instead of a file (it does when the
    delegating prompt asks for markdown back, as the main session did in a nudge-mode run)."""
    for e in events:
        if e.get("type") != "user" or e.get("parent_tool_use_id"):
            continue
        for b in (e.get("message") or {}).get("content") or []:
            if isinstance(b, dict) and b.get("type") == "tool_result":
                body = b.get("content")
                if isinstance(body, list):
                    body = "".join(x.get("text", "") for x in body if isinstance(x, dict))
                report = _trace_module().handback_report(str(body or ""))
                if report:
                    return report
    return None


def completeness(events: list[dict], files: list[Path]) -> tuple[str, list[str]]:
    """``complete``, ``partial`` (the run ended but its analysis is unfinished) or ``failed``,
    with what is missing. A run cut short (a spend limit, a timeout) can still exit with a
    report on disk: in the 2026-09-29 rerun one agent stopped after 9 of its sections."""
    problems: list[str] = []
    result = next((e for e in reversed(events) if e.get("type") == "result"), None)
    if result is None:
        problems.append("no result (killed or timed out)")
    elif result.get("is_error"):
        problems.append("error: " + " ".join(str(result.get("result") or "").split())[:100])
    calls = {  # the main session's Agent calls to the first-principles agent
        b.get("id")
        for e in events
        if e.get("type") == "assistant" and not e.get("parent_tool_use_id")
        for b in (e.get("message") or {}).get("content") or []
        if b.get("type") == "tool_use"
        and b.get("name") in ("Agent", "Task")
        and FP_AGENT in json.dumps(b.get("input", {}))
    }
    returned = {  # its result, or a backgrounded agent's completion notification
        b.get("tool_use_id")
        for e in events
        if e.get("type") == "user" and not e.get("parent_tool_use_id")
        for b in (e.get("message") or {}).get("content") or []
        if isinstance(b, dict)
        and b.get("type") == "tool_result"
        and "async_launched" not in json.dumps(b.get("content"), default=str)
    } | {
        e.get("tool_use_id")
        for e in events
        if e.get("subtype") == "task_notification" and e.get("status") == "completed"
    }
    started = bool(calls)
    handed_back = bool(calls & returned)
    if started and not handed_back:
        problems.append("the agent did not hand back")
    inline = inline_report(events)
    if not files and not inline:
        problems.append("no analysis file")
    else:
        texts = [f.read_text(errors="replace") for f in files] or [inline or ""]
        text = max(texts, key=len)
        d = _trace_module().parse_analysis(text)
        have = {int(m) for m in re.findall(r"^## (\d)\.", text, re.M)}
        missing = [n for n in REQUIRED_SECTIONS if n not in have]
        if missing:
            problems.append("analysis missing sections " + ", ".join(map(str, missing)))
        if not d["gate"]["bands"]:
            problems.append("no Self-Audit Gate")
    if not problems:
        return "complete", []
    ran = files and not any(p.startswith(("no analysis", "no result")) for p in problems)
    return ("partial" if ran else "failed"), problems


def read_own_example(events: list[dict], name: str) -> bool:
    """Whether the agent opened the worked example it was asked to redo: its answer key. The
    examples are listed in the agent's own definition; 4 of the 28 runs on 2026-09-29 read
    theirs, so their decisions are not an independent test of the method."""
    target = f"/references/examples/{name}.md"
    return any(
        b.get("type") == "tool_use"
        and (
            str((b.get("input") or {}).get("file_path") or "").endswith(target)
            or target in str((b.get("input") or {}).get("command") or "")
        )
        for e in events
        if e.get("type") == "assistant" and e.get("parent_tool_use_id")
        for b in (e.get("message") or {}).get("content") or []
    )


def summarize(name: str, workdir: Path, capture: Path, state: Path) -> dict:
    events = []
    for line in capture.read_text(errors="replace").splitlines():
        try:
            events.append(json.loads(line))
        except ValueError:
            continue
    session = next((e.get("session_id") for e in events if e.get("session_id")), None)
    tool_uses = [
        b
        for e in events
        if e.get("type") == "assistant"
        for b in (e.get("message") or {}).get("content") or []
        if b.get("type") == "tool_use"
    ]
    delegated = any(
        b.get("name") in ("Agent", "Task") and FP_AGENT in json.dumps(b.get("input", {}))
        for b in tool_uses
    )
    calc_calls = sum(b.get("name") == CALC_TOOL for b in tool_uses)
    result = next((e for e in reversed(events) if e.get("type") == "result"), {})
    audit = state / "audit" / f"{session}.jsonl"
    records = (
        [json.loads(x) for x in audit.read_text().splitlines()]
        if session and audit.is_file()
        else []
    )
    files = sorted((workdir / ".first-principles").glob("analysis-*.md"))
    words = sum(len(f.read_text(errors="replace").split()) for f in files)
    inline = None if files else inline_report(events)
    if inline:
        words = len(inline.split())
    status, problems = completeness(events, files)
    return {
        "example": name,
        "status": status,
        "problems": problems,
        "read_own_example": read_own_example(events, name),
        "analysis_inline": bool(inline),  # returned in the hand-back, not written to a file
        "session": session,
        "delegated": delegated,
        "analysis_files": [str(f) for f in files],
        "analysis_words": words,
        "delegation_note": any(
            r["action"] == "suggest" and r["entry_id"] == "first-principles-agent" for r in records
        ),
        "calc_notes": sum(
            r["action"] == "suggest" and r["entry_id"] == "exact-calc" for r in records
        ),
        "report_writes_skipped": sum(r["reason"] == "skip pattern" for r in records),
        "calc_calls": calc_calls,
        "cost_usd": result.get("total_cost_usd"),
        "turns": result.get("num_turns"),
        "duration_s": round((result.get("duration_ms") or 0) / 1000, 1),
        "error": bool(result.get("is_error")) or not result,
    }


def run_example(name: str, prompt: str, out: Path, args) -> dict:
    workdir = out / name
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "prompt.txt").write_text(prompt + "\n")
    capture = workdir / "run.jsonl"
    state = out / "state"
    argv = [
        "claude",
        "-p",
        "--plugin-dir",
        str(args.fp_repo / "first-principles"),
        "--plugin-dir",
        str(PLUGIN),
        "--no-session-persistence",
        "--output-format",
        "stream-json",
        "--verbose",
        "--permission-mode",
        "bypassPermissions",
    ]
    if args.model:
        argv += ["--model", args.model]
    argv.append(prompt)
    env = {**os.environ, "AGENT_ROUTER_STATE_DIR": str(state)}
    start = time.monotonic()
    print(f"[start] {name}", flush=True)
    with capture.open("w") as fh:
        try:
            subprocess.run(
                argv,
                cwd=workdir,
                env=env,
                stdout=fh,
                stderr=subprocess.STDOUT,
                timeout=args.timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            fh.write(json.dumps({"type": "runner", "timeout": args.timeout}) + "\n")
    row = summarize(name, workdir, capture, state)
    print(
        f"[done]  {name}: delegated={row['delegated']} words={row['analysis_words']} "
        f"calc notes/calls={row['calc_notes']}/{row['calc_calls']} "
        f"cost=${row['cost_usd'] or 0:.2f} ({time.monotonic() - start:.0f}s) {row['status']}"
        + (f": {'; '.join(row['problems'])}" if row["problems"] else ""),
        flush=True,
    )
    return row


def write_summary(out: Path, rows: list[dict]) -> None:
    (out / "summary.json").write_text(json.dumps(rows, indent=2))
    lines = [
        "| example | status | delegated | analysis words | delegation note | calc notes "
        "| calc calls | report writes skipped | cost $ | turns | s |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['example']}{' †' if r['read_own_example'] else ''} | {r['status']} "
            f"| {'yes' if r['delegated'] else 'NO'} "
            f"| {r['analysis_words']} "
            f"| {'yes' if r['delegation_note'] else '-'} | {r['calc_notes']} | {r['calc_calls']} "
            f"| {r['report_writes_skipped']} | {r['cost_usd'] or 0:.2f} | {r['turns']} "
            f"| {r['duration_s']} |"
        )
    total = sum(r["cost_usd"] or 0 for r in rows)
    done = sum(r["status"] == "complete" for r in rows)
    lines += ["", f"{done}/{len(rows)} examples complete; total cost ${total:.2f}."]
    keyed = [r["example"] for r in rows if r["read_own_example"]]
    if keyed:
        lines.append(
            f"† {len(keyed)} run(s) read their own worked example (the answer key): "
            + ", ".join(keyed)
            + ". Leave them out when comparing the agent's decisions with the examples'."
        )
    for r in rows:
        if r["problems"]:
            lines.append(f"- {r['example']} ({r['status']}): {'; '.join(r['problems'])}")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


def _kind(rec: dict) -> str:
    """One checkpoint as a short label: where it happened and what the router did."""
    where = rec.get("point")
    if rec.get("tool_name"):
        where = rec["tool_name"] + (" (in agent)" if rec.get("agent_type") else " (main)")
    what = rec.get("action")
    if what == "suggest":
        what = f"note: {rec.get('entry_id')}"
    elif what == "skipped":
        what = f"skipped: {rec.get('reason')}"
    return f"{where} -> {what}"


def write_decisions(out: Path, rows: list[dict]) -> None:
    """Every router checkpoint of every run: counts per example, then the full log."""
    audit_dir = out / "state" / "audit"
    per_example: dict[str, list[dict]] = {}
    for r in rows:
        path = audit_dir / f"{r['session']}.jsonl" if r.get("session") else None
        recs = (
            [json.loads(x) for x in path.read_text().splitlines()]
            if path and path.is_file()
            else []
        )
        per_example[r["example"]] = recs
    (out / "decisions.json").write_text(json.dumps(per_example, indent=2))

    kinds = sorted({_kind(rec) for recs in per_example.values() for rec in recs})
    lines = [
        "# agent-router decisions",
        "",
        f"Every checkpoint the agent-router-fp hooks recorded, from `{audit_dir}`.",
        "`(main)` = the main session, `(in agent)` = inside first-principles:first-principles.",
        "",
        "## Counts per example",
        "",
        "| example | checkpoints | " + " | ".join(kinds) + " |",
        "|---|---|" + "---|" * len(kinds),
    ]
    totals = dict.fromkeys(kinds, 0)
    for name, recs in per_example.items():
        counts = {k: 0 for k in kinds}
        for rec in recs:
            counts[_kind(rec)] += 1
            totals[_kind(rec)] += 1
        lines.append(
            f"| {name} | {len(recs)} | " + " | ".join(str(counts[k] or "") for k in kinds) + " |"
        )
    n = sum(len(v) for v in per_example.values())
    lines.append(f"| **total** | **{n}** | " + " | ".join(f"**{totals[k]}**" for k in kinds) + " |")
    lines += ["", "## Every checkpoint", ""]
    for name, recs in per_example.items():
        lines += [
            f"### {name}",
            "",
            "| # | where | action | entry | p | reason / call |",
            "|---|---|---|---|---|---|",
        ]
        for i, rec in enumerate(recs, 1):
            where = (
                rec.get("point")
                if not rec.get("tool_name")
                else (f"{rec['tool_name']} ({'in agent' if rec.get('agent_type') else 'main'})")
            )
            p = (rec.get("probabilities") or {}).get(rec.get("entry_id") or rec.get("choice") or "")
            detail = rec.get("reason") or ""
            if rec.get("point") == "prompt":
                detail += f" · “{(rec.get('text') or '')[:60]}…”"
            lines.append(
                f"| {i} | {where} | {rec.get('action')} | {rec.get('entry_id') or ''} "
                f"| {'' if p is None else f'{p:.2f}'} | {detail.replace('|', '/')} |"
            )
        lines.append("")
    (out / "decisions.md").write_text("\n".join(lines) + "\n")
    print(f"\n{n} router checkpoints recorded -> {out / 'decisions.md'}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--fp-repo", type=Path, default=Path.home() / "Projects" / "first-principles-skill"
    )
    ap.add_argument("--only", help="comma-separated example names (file stems)")
    ap.add_argument("--entry", choices=["launcher", "nudge"], default="launcher")
    ap.add_argument("--model", help="claude --model (default: the CLI's default)")
    ap.add_argument("--jobs", type=int, default=1, help="examples run at once (default 1)")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--timeout", type=float, default=2400.0, help="seconds per example")
    ap.add_argument("--dry-run", action="store_true", help="print the prompts, run nothing")
    args = ap.parse_args()

    examples = sorted((args.fp_repo / "shared" / "examples").glob("*.md"))
    if args.only:
        wanted = set(args.only.split(","))
        examples = [e for e in examples if e.stem in wanted]
    if not examples:
        print("no examples found", file=sys.stderr)
        return 2
    prompts = {}
    for path in examples:
        setup, source = extract_setup(path.read_text())
        prompts[path.stem] = build_prompt(setup, args.entry)
        if args.dry_run:
            print(
                f"=== {path.stem}  [{source}, {len(setup.split())} words]\n{prompts[path.stem]}\n"
            )
    if args.dry_run:
        return 0

    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    out = (args.out or Path.cwd() / f"fp-example-runs-{stamp}").resolve()
    out.mkdir(parents=True, exist_ok=True)
    print(f"{len(prompts)} examples -> {out}  (entry={args.entry}, jobs={args.jobs})")
    with cf.ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        futures = {pool.submit(run_example, n, p, out, args): n for n, p in prompts.items()}
        rows = [f.result() for f in cf.as_completed(futures)]
    rows.sort(key=lambda r: r["example"])
    write_summary(out, rows)
    write_decisions(out, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
