import sys

import numpy as np

from vsa import RHC, resonator_decoding


def test_rhc_sim() -> None:
    dim = 1000
    left = RHC.encode(dim=dim, x=10)

    self_sim = RHC.similarity(left.data, left.data)

    assert 0.2 <= self_sim <= 1.0
    for i in range(10):
        right = RHC.encode(dim=dim, x=i)
        other_sim = RHC.similarity(left.data, right.data)
        assert -1.0 <= other_sim < 0.2


def test_rhc_bind() -> None:
    dim = 1000
    left = RHC.encode(dim=dim, x=10)
    right = RHC.encode(dim=dim, x=15)

    bound = RHC.bind(left.data, right.data)

    left_sim = RHC.similarity(left.data, bound)
    right_sim = RHC.similarity(right.data, bound)

    assert -1.0 <= left_sim < 0.2, "lhs of binding"
    assert -1.0 <= right_sim < 0.2, "rhs of binding"

    left_unbound = RHC.unbind(bound, left.data)
    right_unbound = RHC.unbind(bound, right.data)

    assert (
        1.1 >= RHC.similarity(left_unbound, right.data) >= 0.2
    ), "left unbound should be right"
    assert (
        1.1 >= RHC.similarity(right_unbound, left.data) >= 0.2
    ), "right unbound should be left"


def test_rhc_superpose() -> None:
    dim = 10000
    left = RHC.encode(dim=dim, x=10)
    right = RHC.encode(dim=dim, x=2)

    sup = RHC.bundle(left.data, right.data)

    assert 1.1 >= RHC.similarity(left.data, sup) >= 0.2, "left should be in bundle"
    assert 1.1 >= RHC.similarity(right.data, sup) >= 0.2, "right should be in bundle"


def test_rhc_hash() -> None:
    dim = 1000
    value = RHC.encode(dim=dim, x=10)

    d = {value: "test"}

    assert d[value] == "test"

    reverse = {v: k for (k, v) in d.items()}

    assert reverse["test"] == value


def test_rhc_resonate() -> None:
    dim = 1000
    value = RHC.encode(dim=dim, x=10)

    dec, codebooks = resonator_decoding(value)
    last_guess = dec[-1]
    finest = []
    for mod, est in last_guess.items():
        sims = []
        book = codebooks[mod]
        for i in range(mod):
            sims.append(RHC.similarity(est, book[:, i]))
        finest.append(np.argmax(sims))
    print(finest, file=sys.stderr)
    assert False
