from pathlib import Path

import pytest

from agent_router.core.catalog import CatalogError, load_catalog
from agent_router.core.types import HookPoint


def _entry(id_: str, license_: str = "MIT") -> str:
    return f"""
  - id: {id_}
    kind: tool
    name: Test {id_}
    project: test
    license: {license_}
    url: https://example.com/{id_}
    target: mcp__agent_router__{id_.replace("-", "_")}
    points: [prompt]
    what: does {id_}
"""


def _write(tmp_path: Path, *entries: str) -> Path:
    path = tmp_path / "catalog.yaml"
    path.write_text('version: "t"\nentries:\n' + "".join(entries))
    return path


def test_default_catalog_loads_five_entries():
    cat = load_catalog()
    assert [e.id for e in cat.entries] == [
        "exact-calc",
        "json-query",
        "html-to-markdown",
        "repo-stats",
        "commit-writer",
    ]
    assert all(e.license == "MIT" for e in cat.entries)
    assert cat.native_examples


def test_valid_custom_catalog_loads(tmp_path):
    cat = load_catalog(_write(tmp_path, _entry("a"), _entry("b")))
    assert [e.id for e in cat.entries] == ["a", "b"]


def test_non_mit_entry_rejected(tmp_path):
    with pytest.raises(CatalogError, match="only MIT"):
        load_catalog(_write(tmp_path, _entry("gpl-thing", license_="GPL-3.0")))


def test_reserved_none_id_rejected(tmp_path):
    with pytest.raises(CatalogError, match="reserved"):
        load_catalog(_write(tmp_path, _entry("none")))


def test_duplicate_id_rejected(tmp_path):
    with pytest.raises(CatalogError, match="duplicate"):
        load_catalog(_write(tmp_path, _entry("a"), _entry("a")))


def test_eligible_tool_bash_excludes_skill_entry():
    ids = [e.id for e in load_catalog().eligible(HookPoint.TOOL, "Bash")]
    assert "commit-writer" not in ids
    assert "exact-calc" in ids


def test_eligible_skill_is_commit_writer_only():
    ids = [e.id for e in load_catalog().eligible(HookPoint.SKILL, "Skill")]
    assert ids == ["commit-writer"]


def test_owns_target():
    cat = load_catalog()
    assert cat.owns_target("mcp__agent_router__calc")
    assert cat.owns_target("Skill", skill="commit-writer")
    assert not cat.owns_target("Bash")


def test_fit_check_is_loaded_and_unknown_ones_rejected(tmp_path):
    cat = load_catalog(_write(tmp_path, _entry("a") + "    fits: calc_expression\n"))
    assert cat.entries[0].fits == "calc_expression"
    with pytest.raises(CatalogError, match="unknown fit check 'eval'"):
        load_catalog(_write(tmp_path, _entry("b") + "    fits: eval\n"))
