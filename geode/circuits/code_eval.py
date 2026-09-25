"""CRUXEval execution checks in short-lived, resource-limited subprocesses.

The official equality test is preserved, including alternative valid inputs and
expression inputs (some published references require lambdas or globals). Resource
limits and restricted builtins are defense in depth, not an OS security sandbox;
run only the hash-pinned benchmark functions, never arbitrary user programs.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass


@dataclass(frozen=True)
class ExecutionResult:
    correct: bool
    parse_failure: bool = False
    timed_out: bool = False
    prediction: str | None = None
    execution_failure: bool = False


def parse_crux_answer(generation: str, direction: str) -> str | None:
    """Parse a generated suffix/full assertion without splitting inside strings."""
    if direction not in {"input", "output"}:
        raise ValueError("direction must be input or output")
    text = generation.split("[ANSWER]", 1)[-1].split("[/ANSWER]", 1)[0].strip()
    if text.startswith("assert "):
        text = text[7:].strip()
    try:
        expr = ast.parse(text, mode="eval").body
        if isinstance(expr, ast.Compare):
            if len(expr.ops) != 1 or not isinstance(expr.ops[0], ast.Eq):
                return None
            expr = expr.left if direction == "input" else expr.comparators[0]
        if direction == "input":
            if not (
                isinstance(expr, ast.Call)
                and isinstance(expr.func, ast.Name)
                and expr.func.id == "f"
            ):
                return None
        for node in ast.walk(expr):
            if isinstance(node, ast.Name) and node.id.startswith("__"):
                return None
            if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
                return None
            if isinstance(node, (ast.NamedExpr, ast.Await, ast.Yield, ast.YieldFrom)):
                return None
        return ast.unparse(expr)
    except (SyntaxError, ValueError, TypeError, RecursionError):
        return None


_WORKER = r"""
import ast, builtins, json, resource, sys
payload = json.loads(sys.stdin.read())
memory = payload["memory_mb"] * 1024 * 1024
resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
resource.setrlimit(resource.RLIMIT_CPU, (payload["cpu_seconds"], payload["cpu_seconds"]))
resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
allowed = "abs all any ascii bin bool bytearray bytes chr complex dict divmod enumerate filter float format frozenset hash hex int isinstance issubclass iter len list map max min next oct ord pow range repr reversed round set slice sorted str sum tuple type zip ArithmeticError AssertionError Exception IndexError KeyError StopIteration TypeError ValueError ZeroDivisionError".split()
scope = {"__builtins__": {name: getattr(builtins, name) for name in allowed}}
try:
    tree = ast.parse(payload["code"])
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise ValueError("unsupported benchmark source")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise ValueError("unsupported benchmark attribute")
    exec(compile(tree, "<cruxeval>", "exec"), scope)
    expected = eval(compile(ast.parse(payload["output"], mode="eval"), "<reference>", "eval"), scope)
    actual = eval(compile(ast.parse(payload["prediction"], mode="eval"), "<prediction>", "eval"), scope)
    print(json.dumps({"correct": bool(expected == actual)}))
except BaseException:
    print(json.dumps({"correct": False, "execution_failure": True}))
"""


def evaluate_crux(
    code: str,
    reference_input: str,
    reference_output: str,
    generation: str,
    direction: str,
    *,
    timeout: float = 3.0,
    memory_mb: int = 256,
) -> ExecutionResult:
    """Check greedy pass@1, accepting any supported input producing the output."""
    if timeout <= 0 or memory_mb < 32:
        raise ValueError("positive timeout and memory_mb >= 32 required")
    prediction = parse_crux_answer(generation, direction)
    if prediction is None:
        return ExecutionResult(False, parse_failure=True)
    # Preserve the official verifier's exclusion of an output consisting of
    # the reference function call, instead of the predicted result.
    if direction == "output" and f"f({reference_input})" in generation.split("==")[-1]:
        return ExecutionResult(False, prediction=prediction)
    payload = {
        "code": code,
        "output": reference_output,
        "prediction": prediction,
        "direction": direction,
        "memory_mb": memory_mb,
        "cpu_seconds": max(1, int(timeout) + 1),
    }
    with tempfile.TemporaryDirectory(prefix="geode-crux-") as directory:
        try:
            result = subprocess.run(
                [sys.executable, "-I", "-c", _WORKER],
                input=json.dumps(payload),
                capture_output=True,
                text=True,
                cwd=directory,
                env={"OMP_NUM_THREADS": "1", "PYTHONHASHSEED": "0"},
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                False, timed_out=True, prediction=prediction, execution_failure=True
            )
    try:
        verdict = json.loads(result.stdout)
        correct = result.returncode == 0 and verdict["correct"] is True
        execution_failure = result.returncode != 0 or verdict.get("execution_failure", False)
    except (ValueError, KeyError):
        correct = False
        execution_failure = True
    return ExecutionResult(correct, prediction=prediction, execution_failure=execution_failure)
