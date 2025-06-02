import sys

import pytest

from language.encoding import *
from language.interpreter import *
from vsa import FHRR, HRR


@pytest.fixture
def dim() -> int:
    return 1000


def test_encoding0() -> None:
    assert True


def test_number1(dim: int) -> None:
    vsa = FHRR
    src = "1"

    enc_env = EncodingEnvironment(vsa=vsa, dim=dim)
    eval_env = EvalEnvironment(AssociativeMemory(vsa=vsa, dim=dim), None)

    value = encode(parse(lex(src)), enc_env)

    print(decode(value, enc_env, eval_env), file=sys.stderr)
    assert is_nil(car(value, enc_env, eval_env), enc_env)
    assert is_nil(cdr(value, enc_env, eval_env), enc_env)


def test_all_numbers(dim: int) -> None:
    vsa = FHRR
    enc_env = EncodingEnvironment(vsa=vsa, dim=dim)
    eval_env = EvalEnvironment(AssociativeMemory(vsa=vsa, dim=dim), None)

    assert True
