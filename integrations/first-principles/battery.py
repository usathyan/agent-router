#!/usr/bin/env python3
"""Delegation battery: first-principles' routing catalog, with and without agent-router.

Runs every P (should delegate) and N (should not) prompt of a first-principles routing
catalog through a fresh ``claude -p`` session, once with only the first-principles plugin
(``baseline``) and once with this integration's plugin loaded next to it (``router``), and
reports how often the main session delegated to ``first-principles:first-principles``.

The first-principles repository is used read-only: its ``scripts/check-routing.py`` is
imported for the catalog parser and the DELEGATE / NO-DELEGATE verdict, so both plugins are
scored exactly the way that project scores itself. The transport is its flag set plus a
second ``--plugin-dir``.

A session is stopped as soon as the verdict is known (the first ``Agent``/``Task`` call to
first-principles, or the final result), so a delegated run does not pay for a full analysis.
With ``router`` the per-session audit log shows whether the nudge fired (``hint``).

Costs real tokens: one short Claude session per prompt x repeat x config.

Usage:
    battery.py [--fp-repo PATH] [--catalog PATH] [--configs baseline,router]
               [--router-plugin DIR] [--repeat N] [--model M] [--ids P1,N19]
               [--out DIR] [--timeout S]
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PLUGIN = HERE / "plugin"
FP_AGENT = "first-principles:first-principles"


def load_fp_router_check(fp_repo: Path):
    """Import first-principles' scripts/check-routing.py (read-only) as a module."""
    path = fp_repo / "scripts" / "check-routing.py"
    spec = importlib.util.spec_from_file_location("fp_check_routing", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["fp_check_routing"] = mod  # dataclasses resolve their module by name
    spec.loader.exec_module(mod)
    return mod


def delegated_call(event: dict) -> bool:
    """True for an assistant tool_use that hands the task to the first-principles agent."""
    if event.get("type") != "assistant":
        return False
    for block in (event.get("message") or {}).get("content") or []:
        if (
            block.get("type") == "tool_use"
            and block.get("name") in ("Agent", "Task")
            and FP_AGENT.split(":")[0] in json.dumps(block.get("input", {})).lower()
        ):
            return True
    return False


def run_one(text: str, plugins: list[Path], out: Path, state: Path, args) -> str | None:
    """One ``claude -p`` session captured to ``out``; returns its session id."""
    argv = ["claude", "-p"]
    for p in plugins:
        argv += ["--plugin-dir", str(p)]
    argv += [
        "--no-session-persistence",
        "--output-format",
        "stream-json",
        "--verbose",
        "--permission-mode",
        "bypassPermissions",
    ]
    if args.model:
        argv += ["--model", args.model]
    argv.append(text)
    env = {**os.environ, "AGENT_ROUTER_STATE_DIR": str(state)}
    session = None
    with tempfile.TemporaryDirectory() as cwd, out.open("w") as fh:
        proc = subprocess.Popen(
            argv, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        deadline = time.monotonic() + args.timeout
        assert proc.stdout is not None
        for line in proc.stdout:
            fh.write(line)
            try:
                event = json.loads(line)
            except ValueError:
                continue
            session = session or event.get("session_id")
            if delegated_call(event) or event.get("type") == "result":
                break
            if time.monotonic() > deadline:
                fh.write(json.dumps({"type": "battery", "timeout": args.timeout}) + "\n")
                break
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
    return session


def hint_fired(state: Path, session: str | None) -> bool:
    if not session:
        return False
    audit = state / "audit" / f"{session}.jsonl"
    if not audit.is_file():
        return False
    for line in audit.read_text().splitlines():
        rec = json.loads(line)
        if rec.get("action") == "suggest" and rec.get("entry_id") == "first-principles-agent":
            return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--fp-repo", type=Path, default=Path.home() / "Projects" / "first-principles-skill"
    )
    ap.add_argument("--catalog", type=Path, help="default: <fp-repo>/tests/routing-catalog.md")
    ap.add_argument("--configs", default="baseline,router")
    ap.add_argument("--router-plugin", type=Path, default=PLUGIN, help="plugin for `router`")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--model", help="claude --model (default: the CLI's default)")
    ap.add_argument("--ids", help="comma-separated prompt ids to run (default: all)")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--timeout", type=float, default=240.0, help="seconds per session")
    args = ap.parse_args()

    fp = load_fp_router_check(args.fp_repo)
    positives, negatives = fp.parse_catalog(
        args.catalog or args.fp_repo / "tests/routing-catalog.md"
    )
    prompts = [*positives, *negatives]
    if args.ids:
        wanted = set(args.ids.split(","))
        prompts = [p for p in prompts if p.id in wanted]
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.out or Path(tempfile.gettempdir()) / f"fp-battery-{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    fp_plugin = args.fp_repo / "first-principles"
    configs = {
        "baseline": [fp_plugin],
        "router": [fp_plugin, args.router_plugin],
    }
    chosen = [c.strip() for c in args.configs.split(",")]
    rows = []
    for prompt in prompts:
        for config in chosen:
            for n in range(1, args.repeat + 1):
                state = out_dir / config / "state"
                path = out_dir / config / f"{prompt.id}-run{n}.jsonl"
                path.parent.mkdir(parents=True, exist_ok=True)
                session = run_one(prompt.text, configs[config], path, state, args)
                verdict = fp.detect_routing(path)
                row = {
                    "id": prompt.id,
                    "expected": prompt.expected,
                    "config": config,
                    "run": n,
                    "verdict": verdict,
                    "hint": hint_fired(state, session) if config == "router" else None,
                }
                rows.append(row)
                print(json.dumps(row), flush=True)
    (out_dir / "results.json").write_text(json.dumps(rows, indent=2))

    print(f"\nresults: {out_dir / 'results.json'}")
    print(f"{'config':<9} {'P delegated':>12} {'N not delegated':>16} {'hints P/N':>10}")
    for config in chosen:
        mine = [r for r in rows if r["config"] == config]
        p = [r for r in mine if r["expected"] == "DELEGATE"]
        n = [r for r in mine if r["expected"] == "NO-DELEGATE"]
        p_ok = sum(r["verdict"] == "DELEGATE" for r in p)
        n_ok = sum(r["verdict"] == "NO-DELEGATE" for r in n)
        hints = (
            f"{sum(bool(r['hint']) for r in p)}/{sum(bool(r['hint']) for r in n)}"
            if config == "router"
            else "-"
        )
        print(f"{config:<9} {p_ok:>6}/{len(p):<5} {n_ok:>9}/{len(n):<6} {hints:>10}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
