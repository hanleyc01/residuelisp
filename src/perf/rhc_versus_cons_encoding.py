"""Module comparing the two integer encodings. Here, we are going to
completely handroll the different representations as both a pedagogical
tool (looking at a whole project and gleaning the encoding is annoying
for the casual reader), but also to make the tests very easy.

For more information about the encoding as it used in the interpreter,
see `language.encoding.make_tuple`.
"""

import time

import matplotlib.pyplot as plt
import numpy as np

from language import *
from syntax import *
from vsa import FHRR, RHC


def iterated_rhc_bind(iterations: int = 100, dim: int = 100) -> list[tuple[float, int]]:
    iters = []

    left = RHC.encode(dim, 10)
    right = RHC.encode(dim, 7)

    prod = left * right
    for i in range(iterations):
        begin_time = time.time()

        prod *= left

        end_time = time.time()
        iters.append((end_time - begin_time, i))

    return iters


def iterated_cons_add(iterations: int, dim: int) -> list[tuple[float, int]]:
    iters = []
    enc_env = EncodingEnvironment(FHRR, dim=dim)
    eval_env = EvalEnvironment(AssociativeMemory(dim=dim, vsa=FHRR), None)

    left_src = "10"
    right_src = "7"

    left_enc = encode(parse(lex(left_src)), enc_env)
    right_enc = encode(parse(lex(right_src)), enc_env)

    rand = make_cons(left_enc, right_enc, enc_env)
    rand = add(rand, enc_env, eval_env)
    for i in range(iterations):
        begin_time = time.time()

        args = make_cons(left_enc, rand, enc_env)
        rand = add(args, enc_env, eval_env)

        end_time = time.time()
        iters.append((end_time - begin_time, i))

    return iters


def perf() -> None:
    dim = 1000
    num_iterations = 1000
    rhc_bind: list[tuple[float, int]] = iterated_rhc_bind(
        dim=dim, iterations=num_iterations
    )
    times = [time * 1000 for time, _ in rhc_bind]
    iterations = [iteration for _, iteration in rhc_bind]
    plt.rcParams["font.family"] = "Times New Roman"
    plt.plot(iterations, times, color="BLACK", linestyle="dashed", label="RHC Integers")
    iter_cons = iterated_cons_add(dim=dim, iterations=num_iterations)
    times = [time * 1000 for time, _ in iter_cons]
    iterations = [iteration for _, iteration in iter_cons]
    plt.plot(iterations, times, color="BLACK", label="List Integers")
    plt.xlabel("Value $n$ in operation $1 + n$")
    plt.ylabel("Time (msec)")
    plt.title("Time to compute $1 + n$")
    plt.legend()
    plt.show()
