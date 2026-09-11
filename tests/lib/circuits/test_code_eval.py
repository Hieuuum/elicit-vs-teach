"""Execution, ambiguity, parser and resource-bound properties; entirely CPU."""

import pytest

from geode.circuits.code_eval import evaluate_crux, parse_crux_answer


def test_alternative_valid_input_is_accepted_even_when_reference_string_differs():
    code = "def f(x):\n    return abs(x)"
    result = evaluate_crux(code, "3", "3", "assert f(-3) == 3\n[/ANSWER]", "input")
    assert result.correct
    assert not result.parse_failure


@pytest.mark.parametrize(
    "answer", ["3", "assert f(2) == 3", "[ANSWER]\nassert f(2) == 3\n[/ANSWER]"]
)
def test_output_parser_accepts_literal_or_published_full_assertion(answer):
    assert evaluate_crux("def f(x):\n    return x + 1", "2", "3", answer, "output").correct


def test_parser_does_not_split_equality_markers_inside_strings():
    assert parse_crux_answer("assert f('==') == '=='", "input") == "f('==')"
    assert parse_crux_answer("assert f('==') == '=='", "output") == "'=='"


def test_multiple_argument_and_zero_argument_calls_are_valid():
    assert evaluate_crux("def f(a,b):\n    return a+b", "1, 2", "3", "f(0, 3)", "input").correct
    assert evaluate_crux("def f():\n    return 5", "", "5", "f()", "input").correct


@pytest.mark.parametrize("candidate", ["__import__('os').getcwd()", "3; exit()"])
def test_output_rejects_code_injection(candidate):
    result = evaluate_crux("def f(x):\n    return x+1", "2", "3", candidate, "output")
    assert not result.correct
    assert result.parse_failure


@pytest.mark.parametrize("candidate", ["3", "g(3)", "f(__import__('os'))"])
def test_input_rejects_executable_argument_expressions(candidate):
    assert parse_crux_answer(candidate, "input") is None


def test_official_verifier_accepts_expressions_and_rejects_reference_call_shortcut():
    assert evaluate_crux("def f(x):\n    return x+1", "2", "3", "1+2", "output").correct
    assert not evaluate_crux("def f(x):\n    return x+1", "2", "3", "f(2)", "output").correct


def test_published_input_constructs_lambdas_globals_and_range_are_supported():
    cases = [
        ("def f(x):\n    return len(x)", "range(4)", "4"),
        ("xs = [1,2]\ndef f(x):\n    return sum(x)", "xs[:]", "3"),
        ("def f(x, op):\n    return op(x)", "2, lambda x: x+1", "3"),
        ("def f(x):\n    global y\n    y=x\n    return y", "3", "3"),
        ("def f(x):\n    return len(x)", "''.join(['A'] * 20)", "20"),
    ]
    for code, inp, out in cases:
        assert evaluate_crux(code, inp, out, f"f({inp})", "input").correct


def test_runtime_exception_is_incorrect_not_parse_failure():
    result = evaluate_crux("def f(x):\n    return 1/x", "1", "1.0", "f(0)", "input")
    assert not result.correct
    assert not result.parse_failure
    assert result.execution_failure


def test_wrong_value_is_distinguished_from_execution_failure():
    result = evaluate_crux("def f(x):\n    return x+1", "1", "2", "f(3)", "input")
    assert not result.correct
    assert not result.parse_failure
    assert not result.execution_failure


def test_nonterminating_function_is_bounded_by_wall_timeout():
    result = evaluate_crux(
        "def f(x):\n    while True:\n        pass", "0", "0", "f(0)", "input", timeout=0.1
    )
    assert not result.correct
    assert result.timed_out


def test_large_allocation_fails_under_memory_limit():
    result = evaluate_crux(
        "def f(x):\n    return [0] * (10**9)", "0", "0", "f(0)", "input", memory_mb=64
    )
    assert not result.correct


def test_benchmark_source_cannot_import_os_or_read_files():
    result = evaluate_crux(
        "import os\ndef f(x):\n    return os.getcwd()", "0", "'anything'", "f(0)", "input"
    )
    assert not result.correct


def test_dict_equality_and_numeric_equivalence_follow_python_execution():
    assert evaluate_crux(
        "def f(x):\n    return x", "0", "{'a': 1, 'b': 2}", "{'b': 2, 'a': 1}", "output"
    ).correct
    assert evaluate_crux("def f(x):\n    return x", "3", "3", "3.0", "output").correct
