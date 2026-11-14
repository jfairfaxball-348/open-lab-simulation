"""math.evaluate: exact arithmetic on a restricted expression grammar.

The expression is parsed with ``ast`` and walked by hand. Nothing is ever
passed to ``eval``; any syntax outside numbers, parentheses and the listed
operators is refused. Run as a script it speaks the subprocess protocol:
argument object as JSON on stdin, result object as JSON on stdout.
"""

from __future__ import annotations

import ast
import json
import math
import re
import sys
from fractions import Fraction

MAX_EXPRESSION_LENGTH = 500
# Bounds that keep one call cheap whatever the input.
MAX_EXPONENT = 1000
MAX_DECIMAL_EXPONENT = 400
MAX_BITS = 20_000

_BINARY = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.Mod: lambda a, b: a % b,
}


class EvaluationError(ValueError):
    """The expression cannot be evaluated; the message is safe to show the caller."""


def _check_size(value: Fraction) -> Fraction:
    if max(value.numerator.bit_length(), value.denominator.bit_length()) > MAX_BITS:
        raise EvaluationError("result too large")
    return value


def _number(node: ast.Constant, source: str) -> Fraction:
    value = node.value
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise EvaluationError(f"only numbers are allowed, not {type(value).__name__}")
    if isinstance(value, int):
        return _check_size(Fraction(value))
    if not math.isfinite(value):
        raise EvaluationError("number too large")
    # Take the decimal as written, so 0.1 is exactly 1/10 and not the nearest float.
    literal = (ast.get_source_segment(source, node) or repr(value)).replace("_", "")
    exponent = re.search(r"[eE]([+-]?\d+)$", literal)
    if exponent and abs(int(exponent.group(1))) > MAX_DECIMAL_EXPONENT:
        raise EvaluationError("decimal exponent too large")
    try:
        return _check_size(Fraction(literal))
    except ValueError:
        return _check_size(Fraction(repr(value)))


def _power(base: Fraction, exponent: Fraction) -> Fraction:
    if exponent.denominator != 1:
        raise EvaluationError("exponent must be a whole number")
    power = exponent.numerator
    if abs(power) > MAX_EXPONENT:
        raise EvaluationError(f"exponent too large (limit {MAX_EXPONENT})")
    bits = max(base.numerator.bit_length(), base.denominator.bit_length())
    if bits * abs(power) > MAX_BITS:
        raise EvaluationError("result too large")
    return base**power


def _evaluate(node: ast.AST, source: str) -> Fraction:
    if isinstance(node, ast.Constant):
        return _number(node, source)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd | ast.USub):
        operand = _evaluate(node.operand, source)
        return -operand if isinstance(node.op, ast.USub) else operand
    if isinstance(node, ast.BinOp):
        left = _evaluate(node.left, source)
        right = _evaluate(node.right, source)
        if isinstance(node.op, ast.Pow):
            return _power(left, right)
        operation = _BINARY.get(type(node.op))
        if operation is None:
            raise EvaluationError(f"unsupported operator: {type(node.op).__name__}")
        return _check_size(operation(left, right))
    if isinstance(node, ast.Name):
        raise EvaluationError(f"names are not allowed: '{node.id}'")
    if isinstance(node, ast.Call):
        raise EvaluationError("function calls are not allowed")
    if isinstance(node, ast.Attribute):
        raise EvaluationError("attribute access is not allowed")
    raise EvaluationError(f"unsupported syntax: {type(node).__name__}")


def evaluate(expression: str) -> Fraction:
    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise EvaluationError("expression too long")
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        raise EvaluationError("not a valid arithmetic expression") from None
    try:
        return _evaluate(tree.body, expression.strip())
    except ZeroDivisionError:
        raise EvaluationError("division by zero") from None
    except RecursionError:
        raise EvaluationError("expression nested too deeply") from None


def run(arguments: dict) -> dict:
    result = evaluate(arguments["expression"])
    if result.denominator == 1 and abs(result.numerator) < 2**53:
        value: int | float = result.numerator
    else:
        try:
            value = float(result)
        except OverflowError:
            raise EvaluationError("result too large to give as a decimal number") from None
    return {"value": value, "exact": str(result)}


def main() -> int:
    # Bytes, not text streams: the console code page must not affect the protocol.
    arguments = json.loads(sys.stdin.buffer.read().decode("utf-8"))
    try:
        payload, status = run(arguments), 0
    except EvaluationError as error:
        payload, status = {"error": str(error)}, 1
    sys.stdout.buffer.write(json.dumps(payload).encode("utf-8"))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
