from language import (AssociativeMemory, EncodingEnvironment, EvalEnvironment,
                      IntegerEncodingScheme, check_atomic, check_function,
                      encode, evaluate, is_approx_eq)
from syntax import lex, parse
from vsa import FHRR, HRR


def test_interpret_check_atomic() -> None:
    dim = 100
    src = "nil"

    enc_env = EncodingEnvironment(
        vsa=HRR, dim=dim, integer_encoding_scheme=IntegerEncodingScheme.ListIntegers
    )
    encoded_rep = encode(parse(lex(src)), enc_env)

    assert is_approx_eq(
        check_atomic(encoded_rep, enc_env), enc_env.codebook["#t"], enc_env
    )


def test_interpret_check_function() -> None:
    dim = 100
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


def test_interpret_cons() -> None:
    dim = 100
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


def test_car() -> None:
    dim = 100
    src = "(car (cons #t #t))"

    vsa = HRR
    enc_env = EncodingEnvironment(vsa=vsa, dim=dim)
    encoded_rep = encode(parse(lex(src)), enc_env)

    eval_env = EvalEnvironment(
        AssociativeMemory(vsa=enc_env.vsa, dim=enc_env.dim), None
    )
    evaluated_rep = evaluate(encoded_rep, enc_env, eval_env)

    assert is_approx_eq(evaluated_rep, enc_env.codebook["#t"], enc_env)
