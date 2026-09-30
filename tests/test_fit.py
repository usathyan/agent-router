import pytest

from agent_router.core.fit import calc_expression


@pytest.mark.parametrize(
    "command",
    [
        'python3 -c "print(0.17 * 2340)"',
        "python -c 'import math; print(math.sqrt(2) * 400)'",
        'python3 -c "import math; print(math.factorial(20) / math.factorial(17))"',
        'echo "scale=6; 1/3 + 2/7" | bc -l',
        'echo "2500*(1+0.042)^12" | bc -l',
        "awk 'BEGIN { print 1500 * 0.07 * 12 }'",
        'node -e "console.log(3.5e6 / 12 / 30)"',
        r"expr 365 \* 24 \* 60",
        "echo $((365 * 24))",
    ],
)
def test_one_expression_the_calculator_runs_fits(command):
    assert calc_expression({"command": command})


@pytest.mark.parametrize(
    "command",
    [
        # measured in the first-principles runs: scripts and text commands drew the note
        'python3 -c "\nimport math\nT0=298.15;Th=838.15\nprint((Th-T0)/math.log(Th/T0))\n"',
        "cat > /tmp/tes.py <<'EOF'\nE=5000.0\nprint(E)\nEOF",
        "awk 'BEGIN{cp=1443+0.172*427.5; q=cp*275/3600000; printf \"%.2f\", q}'",
        "cd references; wc -c inversion.md pre-mortem.md trade-off.md",
        'python3 -c "print(math.log(2))"',  # log is not a calculator function
        'python3 -c "print(sum([120, 85.5, 40.25]) / 3)"',  # lists are not supported
        "python3 -c \"import json; print(json.load(open('data.json'))['total'])\"",
        'python3 -c "import yaml; print(yaml.__version__)"',
        "grep -c '^#' README.md",
        "",
    ],
)
def test_anything_else_does_not_fit(command):
    assert not calc_expression({"command": command})
