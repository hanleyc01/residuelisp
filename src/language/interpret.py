"""Module containing functions for interpreting and manipulating vector
symbols.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from vsa import VSA, AnyVSA, VSAdtype

from .encoding import EncodingEnvironment


@dataclass
class InterpreterError(Exception):
    msg: str


def is_approx_equal[T: (
    VSA[np.complex128],
    VSA[np.float64],
)](x: T, y: T, env: EncodingEnvironment[T], floor: float = 0.2) -> bool:
    """Approximate equality between two vector symbols."""
    similarity = env.vsa.similarity(x.data, y.data)
    return similarity > floor


def car[T: (
    VSA[np.complex128],
    VSA[np.float64],
)](expr: T, env: EncodingEnvironment[T]) -> T:
    deref = env.associative_memory.deref(expr)
    raise InterpreterError("unable to find symbol in associative memory")


def cdr[T: (
    VSA[np.complex128],
    VSA[np.float64],
)](expr: T, env: EncodingEnvironment[T]) -> T:
    raise Exception("TODO")


def decode[T: (
    VSA[np.complex128],
    VSA[np.float64],
)](expr: T, env: EncodingEnvironment[T]) -> str | list[Any] | tuple[Any, ...]:
    return ""
