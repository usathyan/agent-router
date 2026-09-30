"""Stdio MCP server ``agent_router`` for hosts that load MCP servers as processes.

``agent-router mcp`` serves the catalog tools to Claude Code (a plugin's ``.mcp.json``), where
they surface as ``mcp__plugin_<plugin>_agent_router__<tool>``. File tools are confined to the
working directory the host starts the server in. ``--tools`` limits what is served.

Works with the ``mcp`` package 1.x (``FastMCP``) and 2.x (``MCPServer``).
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from agent_router.tools import calc, html_to_markdown, json_query, repo_stats

SERVER_NAME = "agent_router"
TOOLS = ("calc", "json_query", "html_to_markdown", "repo_stats")


def _server_class() -> Callable[..., Any]:
    try:
        from mcp.server.mcpserver import MCPServer  # mcp >= 2

        return MCPServer
    except ImportError:
        from mcp.server.fastmcp import FastMCP  # mcp 1.x

        return FastMCP


def _tool_error_class() -> type[Exception]:
    try:
        from mcp.server.mcpserver.exceptions import ToolError  # mcp >= 2
    except ImportError:
        from mcp.server.fastmcp.exceptions import ToolError  # type: ignore  # mcp 1.x
    return ToolError


def _expected(fn: Callable[..., str]) -> Callable[..., str]:
    """Report a tool's ``ValueError`` as the SDK's ``ToolError``, so its message reaches the model.

    The SDK treats any other exception as a crash and sends only ``Error executing tool <name>``:
    measured in the first-principles runs, the agent retried 7 failed calculator calls without
    learning why (e.g. ``log()`` is not supported).
    """
    tool_error = _tool_error_class()

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> str:
        try:
            return fn(*args, **kwargs)
        except ValueError as e:
            raise tool_error(str(e)) from e

    return wrapper


def build_server(root: Path, tools: Iterable[str] = TOOLS) -> Any:
    """The ``agent_router`` MCP server with the chosen tools, files confined to ``root``."""
    root = Path(root).resolve()
    wanted = set(tools)
    unknown = wanted - set(TOOLS)
    if unknown:
        raise ValueError(f"unknown tools: {', '.join(sorted(unknown))}")
    server = _server_class()(SERVER_NAME)

    if "calc" in wanted:

        @server.tool(name="calc")
        @_expected
        def calc_tool(expression: str) -> str:
            """Exact calculator (agent-router, MIT). Evaluates arithmetic with exact fractions
            and big integers: + - * / // % **, '17% of 2340', sqrt, factorial, gcd, lcm, abs,
            round, comb, perm. Returns an integer, 'a/b (≈ decimal)', or '≈ decimal' if
            irrational. A list evaluates each item: '[2*3, 7/2]'. Example expressions:
            '3/7 + 5/11', '2**200', '10000*(1+5/100)**7'."""
            return calc.evaluate(expression)

    if "json_query" in wanted:

        @server.tool(name="json_query")
        @_expected
        def json_query_tool(
            expression: str, path: str | None = None, text: str | None = None
        ) -> str:
            """Query JSON with a JMESPath expression (jmespath.py, MIT). Give exactly one of
            `path` (a JSON file in the workspace) or `text` (inline JSON). Returns pretty JSON."""
            return json_query.query(expression, path=path, text=text, root=root)

    if "html_to_markdown" in wanted:

        @server.tool(name="html_to_markdown")
        @_expected
        def html_to_markdown_tool(path: str | None = None, html: str | None = None) -> str:
            """Convert HTML to clean Markdown (python-markdownify, MIT), keeping headings,
            links, lists and tables. Give exactly one of `path` or `html`."""
            return html_to_markdown.convert(path=path, html=html, root=root)

    if "repo_stats" in wanted:

        @server.tool(name="repo_stats")
        @_expected
        def repo_stats_tool(path: str = ".") -> str:
            """Count files and lines of code per language under a workspace directory
            (agent-router, MIT), skipping .git, virtualenvs and node_modules."""
            return repo_stats.stats(path or ".", root=root)

    return server
