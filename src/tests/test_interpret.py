import sys

import pytest

from language import *
from syntax import lex, parse
from vsa import FHRR, HRR


@pytest.fixture
def dim() -> int:
    return 1000


def test_interpret_check_atomic(dim: int) -> None:
    src = "nil"

    enc_env = EncodingEnvironment(
        vsa=HRR, dim=dim, integer_encoding_scheme=IntegerEncodingScheme.ListIntegers
    )
    encoded_rep = encode(parse(lex(src)), enc_env)

    assert is_approx_eq(
        check_atomic(encoded_rep, enc_env), enc_env.codebook["#t"], enc_env
    )


def test_interpret_check_function(dim: int) -> None:
    src = "(lambda (meow) meow)"

    enc_env = EncodingEnvironment(
        vsa=HRR, dim=dim, integer_encoding_scheme=IntegerEncodingScheme.ListIntegers
    )
    encoded_rep = encode(parse(lex(src)), enc_env)

    eval_env = EvalEnvironment(AssociativeMemory(enc_env.vsa, enc_env.dim), None)
    evaluated_rep = evaluate(encoded_rep, enc_env, eval_env)

    assert is_approx_eq(
        check_function(evaluated_rep, enc_env), enc_env.codebook["#t"], enc_env
    )


def test_interpret_cons(dim: int) -> None:
    src = "(cons nil nil)"

    vsa = FHRR
    enc_env = EncodingEnvironment(
        vsa=vsa, dim=dim, integer_encoding_scheme=IntegerEncodingScheme.ListIntegers
    )
    encoded_rep = encode(parse(lex(src)), enc_env)

    eval_env = EvalEnvironment(
        AssociativeMemory(vsa=enc_env.vsa, dim=enc_env.dim), None
    )
    evaluated_rep = evaluate(encoded_rep, enc_env, eval_env)

    assert is_approx_eq(
        check_atomic(evaluated_rep, enc_env), enc_env.codebook["#f"], enc_env
    )


def test_interpret_cons_structure(dim: int) -> None:
    src = "(cons #t #f)"

    vsa = FHRR
    enc_env = EncodingEnvironment(vsa=vsa, dim=dim)
    encoded_value = encode(parse(lex(src)), enc_env)

    eval_env = EvalEnvironment(
        AssociativeMemory(vsa=enc_env.vsa, dim=enc_env.dim), None
    )
    ptr = evaluate(encoded_value, enc_env, eval_env)

    tuple_chunk = enc_env.associative_memory.deref(ptr)
    assert tuple_chunk is not None

    lhs_slot = FHRR.unbind(tuple_chunk.data, enc_env.codebook["__lhs"].data)
    assert closest(FHRR(lhs_slot), enc_env) == "#t"

    rhs_slot = FHRR.unbind(tuple_chunk.data, enc_env.codebook["__rhs"].data)
    assert closest(FHRR(rhs_slot), enc_env) == "#f"


def test_interpret_car(dim: int) -> None:
    src = "(cons #t #f)"
    vsa = FHRR

    enc_env = EncodingEnvironment(vsa=vsa, dim=dim)
    eval_env = EvalEnvironment(AssociativeMemory(vsa=vsa, dim=dim), None)

    encoded_value = encode(parse(lex(src)), enc_env)
    ptr = evaluate(encoded_value, enc_env, eval_env)

    lhs_slot = car(ptr, enc_env, eval_env)
    assert closest(lhs_slot, enc_env) == "#t"
    assert is_approx_eq(lhs_slot, enc_env.codebook["#t"], enc_env)


def test_interpret_cdr(dim: int) -> None:
    src = "(cons #t #f)"
    vsa = FHRR

    enc_env = EncodingEnvironment(vsa=vsa, dim=dim)
    eval_env = EvalEnvironment(AssociativeMemory(vsa=vsa, dim=dim), None)

    encoded_value = encode(parse(lex(src)), enc_env)
    ptr = evaluate(encoded_value, enc_env, eval_env)

    rhs_slot = cdr(ptr, enc_env, eval_env)
    assert closest(rhs_slot, enc_env) == "#f"
    assert is_approx_eq(rhs_slot, enc_env.codebook["#f"], enc_env)


def test_embedded_cons(dim: int) -> None:
    src = "(car (cons #t #f))"
    vsa = FHRR

    enc_env = EncodingEnvironment(vsa=vsa, dim=dim)
    eval_env = EvalEnvironment(AssociativeMemory(vsa=vsa, dim=dim), None)

    encoded_value = encode(parse(lex(src)), enc_env)
    value = evaluate(encoded_value, enc_env, eval_env)

    assert closest(value, enc_env) == "#t"
    # assert is_approx_eq(check_atomic(value, enc_env), enc_env.codebook["#t"], enc_env)


def test_is_false(dim: int) -> None:
    src = "#f"
    vsa = FHRR

    enc_env = EncodingEnvironment(vsa=vsa, dim=dim)
    eval_env = EvalEnvironment(AssociativeMemory(vsa=vsa, dim=dim), None)

    encoded_value = encode(parse(lex(src)), enc_env)
    value = evaluate(encoded_value, enc_env, eval_env)

    assert is_false(value, enc_env)


def test_is_true(dim: int) -> None:
    src = "#t"
    vsa = FHRR

    enc_env = EncodingEnvironment(vsa=vsa, dim=dim)
    eval_env = EvalEnvironment(AssociativeMemory(vsa=vsa, dim=dim), None)

    encoded_value = encode(parse(lex(src)), enc_env)
    value = evaluate(encoded_value, enc_env, eval_env)

    assert is_true(value, enc_env)


def test_if_consequent(dim: int) -> None:
    src = "(if #t #t #t)"
    vsa = FHRR

    enc_env = EncodingEnvironment(vsa=vsa, dim=dim)
    eval_env = EvalEnvironment(AssociativeMemory(vsa=vsa, dim=dim), None)

    encoded_value = encode(parse(lex(src)), enc_env)
    value = evaluate(encoded_value, enc_env, eval_env)

    assert is_true(value, enc_env)


def test_if_alternate(dim: int) -> None:
    src = "(if #f #f #t)"
    vsa = FHRR

    enc_env = EncodingEnvironment(vsa=vsa, dim=dim)
    eval_env = EvalEnvironment(AssociativeMemory(vsa=vsa, dim=dim), None)

    encoded_value = encode(parse(lex(src)), enc_env)
    value = evaluate(encoded_value, enc_env, eval_env)

    assert is_true(value, enc_env)


def test_atom(dim: int) -> None:
    srcs = [
        "(atom? #f)",
        "(atom? #t)",
        "(atom? nil)",
        "(atom? cons)",
    ]
    vsa = FHRR

    enc_env = EncodingEnvironment(vsa=vsa, dim=dim)
    eval_env = EvalEnvironment(AssociativeMemory(vsa=vsa, dim=dim), None)

    for src in srcs:
        encoded_value = encode(parse(lex(src)), enc_env)
        value = evaluate(encoded_value, enc_env, eval_env)

        assert is_true(value, enc_env)

    src = "(atom? (cons nil nil))"
    encoded_value = encode(parse(lex(src)), enc_env)
    value = evaluate(encoded_value, enc_env, eval_env)
    assert is_false(value, enc_env)
