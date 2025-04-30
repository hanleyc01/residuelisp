"""Module containing functions for interpreting and manipulating vector
symbols.
"""

import sys
from dataclasses import dataclass
from typing import Any

import numpy as np

from syntax import lex, parse
from vsa import VSA, AnyVSA, VSAdtype

from .encoding import (AssociativeMemory, EncodingEnvironment,
                       IntegerEncodingScheme, encode, make_cons)


@dataclass
class InterpreterError(Exception):
    """Exception thrown by the interpeter."""

    msg: str


@dataclass
class EvalEnvironment[T: (VSA[np.complex128], VSA[np.float64])]:
    """The evaluation environment, which contains a local associative memory
    for evaluation contexts, as well as a global definition memory, which
    remains constant throughout the evaluation of the program.
    """

    define_mem: AssociativeMemory[T]
    locals_: AssociativeMemory[T] | None


def is_approx_eq[T: (
    VSA[np.complex128],
    VSA[np.float64],
)](x: T, y: T, env: EncodingEnvironment[T], floor: float = 0.2) -> bool:
    """Approximate equality between two vector symbols."""
    similarity = env.vsa.similarity(x.data, y.data)
    return similarity > floor


def check_atomic[T: (
    VSA[np.complex128],
    VSA[np.float64],
)](
    expr: T,
    enc_env: EncodingEnvironment[T],
    eval_env: EvalEnvironment[T],
    floor: float = 0.2,
) -> T:
    """Determine whether or not a value is an atomic value or a list."""
    value = enc_env.associative_memory.deref(expr)
    if value is None:
        return enc_env.codebook["#t"]

    close_to_phi = (
        enc_env.vsa.similarity(value.data, enc_env.codebook["__phi"].data)
        * enc_env.codebook["#f"].data
    )
    far_from_phi = (
        max(
            (2 * floor)
            - enc_env.vsa.similarity(value.data, enc_env.codebook["__phi"].data),
            0.0,
        )
        * enc_env.codebook["#t"].data
    )

    return enc_env.cleanup_memory.recall(
        enc_env.vsa.from_array(enc_env.vsa.bundle(close_to_phi, far_from_phi))
    )


def car[T: (
    VSA[np.complex128],
    VSA[np.float64],
)](expr: T, enc_env: EncodingEnvironment[T], eval_env: EvalEnvironment[T]) -> T:
    """Treat the expression as a semantic pointer and dereference it, returning
    the second element of the underlying tuple chunk.

    Args:
    -   expr (VSA): A vector-symbol semantic pointer.
    -   enc_env (EncodingEnvironment): The encoding environment.
    -   eval_env (EvalEnv): The encoding environment.

    Returns:
        The second element in the underlying tuple chunk.

    Raises:
        `InterpreterError`.
    """
    value = enc_env.associative_memory.deref(expr)
    if value is None:
        raise InterpreterError("Segmentation fault in car hehe")

    return enc_env.cleanup_memory.recall(
        enc_env.vsa.from_array(
            enc_env.vsa.unbind(value.data, enc_env.codebook["__lhs"].data)
        )
    )


def cdr[T: (
    VSA[np.complex128],
    VSA[np.float64],
)](expr: T, enc_env: EncodingEnvironment[T], eval_env: EvalEnvironment[T]) -> T:
    """Treat the expression as a semantic pointer and dereference it,
    returning the first element of an underlying tuple chunk.

    Args:
    -   expr (VSA): A vector-symbol semantic pointer.
    -   enc_env (EncodingEnvironment): The encoding environment.
    -   eval_env (EvalEnv): The encoding environment.

    Returns:
        A vector-symbol corresponding to the first element.

    Raises:
        `InterpreterError`.
    """
    value = enc_env.associative_memory.deref(expr)
    if value is None:
        raise InterpreterError("Segmentation fault in cdr hehe")

    return enc_env.cleanup_memory.recall(
        enc_env.vsa.from_array(
            enc_env.vsa.unbind(value.data, enc_env.codebook["__rhs"].data)
        )
    )


def evaluate_lambda[T: (
    VSA[np.complex128],
    VSA[np.float64],
)](
    function_body: T, enc_env: EncodingEnvironment[T], eval_env: EvalEnvironment[T]
) -> T:
    raise Exception("TODO")


def evaluate_define[T: (
    VSA[np.complex128],
    VSA[np.float64],
)](define_body: T, enc_env: EncodingEnvironment[T], eval_env: EvalEnvironment[T]) -> T:
    raise Exception("TODO")


def evaluate_application[T: (
    VSA[np.complex128],
    VSA[np.float64],
)](
    rator: T, rand: T, enc_env: EncodingEnvironment[T], eval_env: EvalEnvironment[T]
) -> T:
    raise Exception("TODO")


def evaluate[T: (
    VSA[np.complex128],
    VSA[np.float64],
)](expr: T, enc_env: EncodingEnvironment[T], eval_env: EvalEnvironment[T]) -> T:
    """Evalute an encoded vector-symbol.

    Args:
    -   expr (VSA): A vector-symbol representing an expression in the lisp.
    -   enc_env (EncodingEnv): The encoding environment used in encoding.
    -   eval_env (EvalEnvironment): The evaluation environment.

    Returns:
        A vector-symbol representing the result of the evaluation.

    Raises:
        `InterpreterError`.
    """
    if is_approx_eq(
        check_atomic(expr, enc_env, eval_env), enc_env.codebook["#t"], enc_env
    ):
        print("expression is an atom", file=sys.stderr)

        if eval_env.locals_ is None:
            res = eval_env.define_mem.deref(expr)
            if res is None:
                print("value not found in `define_mem`", file=sys.stderr)
                return expr
            return res
        else:
            res = eval_env.locals_.deref(expr)
            if res is None:
                print("value not found in `locals_`", file=sys.stderr)
                return expr
            return res

    head, tail = car(expr, enc_env, eval_env), cdr(expr, enc_env, eval_env)

    if is_approx_eq(head, enc_env.codebook["lambda"], enc_env):
        return evaluate_lambda(tail, enc_env, eval_env)
    elif is_approx_eq(head, enc_env.codebook["define"], enc_env):
        return evaluate_define(tail, enc_env, eval_env)
    else:
        return evaluate_application(head, tail, enc_env, eval_env)

    raise Exception("TODO")


def closest[T: (
    VSA[np.complex128],
    VSA[np.float64],
)](value: T, enc_env: EncodingEnvironment[T]) -> str:
    """Return the closest key for which `value` matches too in the encoding
    environment's codebook.

    Args:
    -   value (VSA): The vector-symbol to search the coding environment for.
    -   enc_env (EncodingEnvironment): The encoding environment.

    Returns:
        A string corresponding to the closest key, the string "NONE" otherwise.
    """
    max_sim = 0.0
    max_word = "NONE"
    for word in enc_env.codebook.keys():
        sim = enc_env.vsa.similarity(enc_env.codebook[word].data, value.data)
        if sim > max_sim:
            max_sim = sim
            max_word = word
    return max_word


def decode[T: (
    VSA[np.complex128],
    VSA[np.float64],
)](expr: T, enc_env: EncodingEnvironment[T], eval_env: EvalEnvironment[T]) -> (
    str | list[Any] | tuple[Any, ...]
):
    """Decode a vector symbol to a Python object.

    Args:
    -   expr (VSA): A vector-symbol, which is the product of `encode`.
    -   enc_env (EncodingEnvironment): The encoding environment used to create
            `expr`.
    -   eval_env (EvalEnvironment): The evaluation environment.

    Returns:
        A Python object corresponding to the encoded value.

    Raises:
        `InterpreterError`.
    """
    if is_approx_eq(
        check_atomic(expr, enc_env, eval_env), enc_env.codebook["#t"], enc_env
    ):
        return closest(expr, enc_env)
    else:
        return ""


def interpret[T: (
    VSA[np.complex128],
    VSA[np.float64],
)](
    src: str, vsa: type[T], dim: int, integer_encoding_scheme: IntegerEncodingScheme
) -> None:
    """Interpret a source-level string of the language.

    Args:
    -   src (str): The source level representation of the language.
    -   vsa (type[VSA]): The VSA you want to use in interpretation.
    -   dim (int): The dimensionality of the interpretation.
    -   integer_encoding_scheme (IntegerEncodingScheme): The integer encoding
            scheme used in encoding and interpretation.

    Returns:
        Another vector-symbol that is the result of the evaluation.

    Raises:
        `InterpreterError`.
    """
    tokens = lex(src)
    intr_rep = parse(tokens)

    encoding_env = EncodingEnvironment(
        vsa=vsa, dim=dim, integer_encoding_scheme=integer_encoding_scheme
    )
    encoded_rep = encode(intr_rep, encoding_env)

    eval_env = EvalEnvironment(
        AssociativeMemory(encoding_env.vsa, encoding_env.dim), None
    )

    result = evaluate(encoded_rep, encoding_env, eval_env)
    decoded_result = decode(result, encoding_env, eval_env)
    print(decoded_result)
