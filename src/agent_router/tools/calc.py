"""Exact calculator: a whitelisted AST evaluator over Fractions (agent-router, MIT).

Never calls ``eval``/``exec``. Supports int/decimal literals (as exact Fractions),
``+ - * / // % **``, unary +/-, parentheses, postfix percent (``17% of 2340``, ``50%``) and
the functions in ``FUNCTIONS``. Irrational results (``sqrt(2)``, fractional powers) are
reported as ``≈ decimal``. A top-level list or tuple (``[2*3, 7/2]``) evaluates each item and
returns ``[6, 7/2 (≈ 3.5)]``. All failures raise ``ValueError``.
"""

import ast
import math
import operator
import re
from collections.abc import Callable
from decimal import Decimal, localcontext
from fractions import Fraction

MAX_EXPONENT = 10_000
# Output cap: stays under CPython's 4300-digit int->str limit.
MAX_DIGITS = 4_000
# Intermediate cap on numerator + denominator size (~2x the output cap).
MAX_RESULT_BITS = 32_000
# factorial/comb/perm/round arguments; factorial(1000) has 2568 digits.
MAX_INT_ARG = 1_000
MAX_EXPRESSION_CHARS = 1_000
MAX_DEPTH = 100
MAX_ITEMS = 50  # a top-level list: agents batch a table's rows (6 of ~600 calls in the fp runs)
DECIMAL_PLACES = 10
_APPROX_DIGITS = 50

_NUM = r"\d+(?:\.\d+)?"
_THOUSANDS = re.compile(r"(?<![\d.,])\d{1,3}(?:,\d{3})+(?![\d.,])")
_CALL = re.compile(r"[A-Za-z_]\w*\s*\(")
_PERCENT_OF = re.compile(rf"({_NUM})\s*%\s*of\s+(.+)$", re.IGNORECASE)
# "50%" / "50% * x": percent. "50% + 10" (no space before %) is percent; "10 % -3" is modulo.
_POSTFIX_PERCENT = re.compile(rf"({_NUM})(?:\s*%(?=\s*(?:$|[)*/,]))|%(?=\s*[+\-]))")


class _Inexact:
    """Mutable flag shared through one evaluation."""

    def __init__(self) -> None:
        self.flag = False


def _as_int(x: Fraction, fn: str) -> int:
    if x.denominator != 1:
        raise ValueError(f"{fn}() needs integer arguments")
    return x.numerator


def _small_int(x: Fraction, fn: str) -> int:
    n = _as_int(x, fn)
    if abs(n) > MAX_INT_ARG:
        raise ValueError(f"{fn}() argument too large (limit {MAX_INT_ARG})")
    return n


def percent_of(pct: Fraction, value: Fraction = Fraction(1)) -> Fraction:
    """``pct`` percent of ``value`` (``percent_of(17, 2340)`` is 17% of 2340)."""
    return Fraction(pct) / 100 * Fraction(value)


def _sqrt(x: Fraction, state: _Inexact) -> Fraction:
    if x < 0:
        raise ValueError("sqrt() of a negative number")
    rn, rd = math.isqrt(x.numerator), math.isqrt(x.denominator)
    if rn * rn == x.numerator and rd * rd == x.denominator:
        return Fraction(rn, rd)
    state.flag = True
    with localcontext() as ctx:
        ctx.prec = _APPROX_DIGITS
        return Fraction(Decimal(x.numerator).sqrt() / Decimal(x.denominator).sqrt())


def _pow(base: Fraction, exp: Fraction, state: _Inexact) -> Fraction:
    if abs(exp) > MAX_EXPONENT:
        raise ValueError(f"exponent too large (limit {MAX_EXPONENT})")
    if base != 0:
        # |log2(result)| ~ |exp| * |log2(base)|; reject before doing the work.
        log2_base = abs(base.numerator.bit_length() - base.denominator.bit_length()) + 1
        if exp.denominator == 1:
            log2_base = max(base.numerator.bit_length(), base.denominator.bit_length())
        if log2_base * abs(exp) > MAX_RESULT_BITS:
            raise ValueError("result too large")
    if exp.denominator == 1:
        e = exp.numerator
        if base == 0 and e < 0:
            raise ValueError("division by zero")
        return base**e
    if base < 0:
        raise ValueError("fractional power of a negative number")
    if base == 0:
        return Fraction(0)
    state.flag = True
    with localcontext() as ctx:
        ctx.prec = _APPROX_DIGITS
        d = Decimal(base.numerator) / Decimal(base.denominator)
        return Fraction(d ** (Decimal(exp.numerator) / Decimal(exp.denominator)))


def _round(x: Fraction, ndigits: Fraction | None = None) -> Fraction:
    if ndigits is None:
        return Fraction(round(x))
    return Fraction(round(x, _small_int(ndigits, "round")))


def _factorial(n: Fraction) -> Fraction:
    k = _small_int(n, "factorial")
    if k < 0:
        raise ValueError("factorial() of a negative number")
    return Fraction(math.factorial(k))


def _comb(n: Fraction, k: Fraction) -> Fraction:
    a, b = _small_int(n, "comb"), _small_int(k, "comb")
    if a < 0 or b < 0:
        raise ValueError("comb() needs non-negative integers")
    return Fraction(math.comb(a, b))


def _perm(n: Fraction, k: Fraction | None = None) -> Fraction:
    a = _small_int(n, "perm")
    b = None if k is None else _small_int(k, "perm")
    if a < 0 or (b is not None and b < 0):
        raise ValueError("perm() needs non-negative integers")
    return Fraction(math.perm(a, b))


def _gcd(*xs: Fraction) -> Fraction:
    return Fraction(math.gcd(*(_as_int(x, "gcd") for x in xs)))


def _lcm(*xs: Fraction) -> Fraction:
    ints = [_as_int(x, "lcm") for x in xs]
    if sum(i.bit_length() for i in ints) > MAX_RESULT_BITS:
        raise ValueError("result too large")
    return Fraction(math.lcm(*ints))


FUNCTIONS: dict[str, Callable[..., Fraction]] = {
    "abs": lambda x: abs(x),
    "round": _round,
    "factorial": _factorial,
    "comb": _comb,
    "perm": _perm,
    "gcd": _gcd,
    "lcm": _lcm,
    "percent_of": percent_of,
}
_STATEFUL = {"sqrt": _sqrt}

_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
}


def _bits(x: Fraction) -> int:
    return x.numerator.bit_length() + x.denominator.bit_length()


def _binop(op_node: ast.operator, left: Fraction, right: Fraction, state: _Inexact) -> Fraction:
    if isinstance(op_node, ast.Pow):
        return _checked(_pow(left, right, state))
    op = _BINOPS.get(type(op_node))
    if op is None:
        raise ValueError(f"unsupported operator {type(op_node).__name__}")
    if _bits(left) + _bits(right) > MAX_RESULT_BITS:
        raise ValueError("result too large")
    try:
        return op(left, right)
    except ZeroDivisionError as e:
        raise ValueError("division by zero") from e


def _checked(x: Fraction) -> Fraction:
    if _bits(x) > MAX_RESULT_BITS:
        raise ValueError("result too large")
    return x


def _eval(node: ast.AST, state: _Inexact, depth: int = 0) -> Fraction:
    if depth > MAX_DEPTH:
        raise ValueError(f"expression nested too deeply (limit {MAX_DEPTH})")
    if isinstance(node, ast.Expression):
        return _eval(node.body, state, depth + 1)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, int | float):
            raise ValueError(f"unsupported literal {node.value!r}")
        return Fraction(repr(node.value)) if isinstance(node.value, float) else Fraction(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd | ast.USub):
        v = _eval(node.operand, state, depth + 1)
        return -v if isinstance(node.op, ast.USub) else v
    if isinstance(node, ast.BinOp):
        # "a+b+c+..." parses as a left-leaning chain; walk it iteratively so long flat
        # sums/products don't count as nesting. Only right operands add depth.
        chain = [node]
        while isinstance(chain[-1].left, ast.BinOp):
            chain.append(chain[-1].left)
        acc = _eval(chain[-1].left, state, depth + 1)
        for link in reversed(chain):
            acc = _binop(link.op, acc, _eval(link.right, state, depth + 1), state)
        return acc
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.keywords:
            raise ValueError("only plain calls to supported functions are allowed")
        name = node.func.id
        args = [_eval(a, state, depth + 1) for a in node.args]
        try:
            if name in _STATEFUL:
                if len(args) != 1:
                    raise ValueError(f"{name}() takes one argument")
                return _checked(_STATEFUL[name](args[0], state))
            fn = FUNCTIONS.get(name)
            if fn is None:
                raise ValueError(f"unknown function {name!r}")
            return _checked(Fraction(fn(*args)))
        except TypeError as e:
            raise ValueError(f"bad arguments to {name}(): {e}") from e
    raise ValueError(f"unsupported syntax: {type(node).__name__}")


def _preprocess(expression: str) -> str:
    expr = expression.strip()
    if not _CALL.search(expr) and not expr.startswith("["):  # in a list, "5,100" is two items
        # "2,340" is a thousands separator only when no function call could use commas.
        expr = _THOUSANDS.sub(lambda m: m.group(0).replace(",", ""), expr)
    m = _PERCENT_OF.search(expr)
    if m:
        rest = _THOUSANDS.sub(lambda mm: mm.group(0).replace(",", ""), m.group(2))
        expr = f"{expr[: m.start()]}percent_of({m.group(1)}, ({rest}))"
    expr = _POSTFIX_PERCENT.sub(r"percent_of(\1)", expr)
    return expr.replace("^", "**").replace("×", "*").replace("÷", "/")


def _decimal(x: Fraction) -> str:
    with localcontext() as ctx:
        ctx.prec = max(len(str(abs(x.numerator) // x.denominator)), 1) + DECIMAL_PLACES + 5
        d = (Decimal(x.numerator) / Decimal(x.denominator)).quantize(
            Decimal(1).scaleb(-DECIMAL_PLACES)
        )
    s = f"{d:f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def _check_digits(x: Fraction) -> None:
    # bit_length * log10(2) over-estimates digits by < 1; avoids str() on huge ints.
    for part in (x.numerator, x.denominator):
        if part.bit_length() * 0.30103 > MAX_DIGITS:
            raise ValueError(f"result has more than {MAX_DIGITS} digits")


def format_result(x: Fraction, inexact: bool = False) -> str:
    _check_digits(x)
    if inexact:
        return f"≈ {_decimal(x)}"
    if x.denominator == 1:
        return str(x.numerator)
    return f"{x.numerator}/{x.denominator} (≈ {_decimal(x)})"


def evaluate(expression: str) -> str:
    """Evaluate ``expression`` exactly; return ``int``, ``a/b (≈ decimal)`` or ``≈ decimal``."""
    if not expression or not expression.strip():
        raise ValueError("empty expression")
    if len(expression) > MAX_EXPRESSION_CHARS:
        raise ValueError(f"expression too long (limit {MAX_EXPRESSION_CHARS} chars)")
    source = _preprocess(expression)
    try:
        tree = ast.parse(source, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"invalid expression: {e.msg}") from e
    except (RecursionError, MemoryError) as e:
        raise ValueError("expression nested too deeply") from e
    items = tree.body.elts if isinstance(tree.body, ast.List | ast.Tuple) else None
    if items is not None:
        if not items:
            raise ValueError("empty list")
        if len(items) > MAX_ITEMS:
            raise ValueError(f"too many items (limit {MAX_ITEMS})")
        return "[" + ", ".join(_one(ast.Expression(body=node)) for node in items) + "]"
    return _one(tree)


def _one(tree: ast.Expression) -> str:
    state = _Inexact()
    try:
        return format_result(_eval(tree, state), state.flag)
    except (ArithmeticError, RecursionError, MemoryError) as e:  # decimal/overflow edge cases
        raise ValueError(f"cannot evaluate: {type(e).__name__}") from e
