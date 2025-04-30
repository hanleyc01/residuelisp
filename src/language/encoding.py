"""Module containing the encoding functions converting intermediate
representaitons into vector-symbolic representations.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import cast

import numpy as np
from numpy.typing import NDArray

from syntax import (KEYWORDS, OPERATORS, VALUES, Intr, IntrAtom, IntrList,
                    Token, TokenKind)
from vsa import FHRR, HRR, RHC, VSA, AnyVSA, VSAdtype


class IntegerEncodingScheme(Enum):
    """The integer encoding scheme to be used by the encoding function.

    - `ListIntegers`: Represent integers as a list.
    - `RHCIntegers`: Represent integers as with Residue Hyperdimensional
      Computing. This only works if the VSA provided to the encoding function
      supports conversions to and from `RHC`.
    """

    ListIntegers = auto()
    RHCIntegers = auto()


class CleanupMemory[T: (VSA[np.complex128], VSA[np.float64])]:
    """A simple clean-up memory."""

    vsa: type[T]
    memory_matrix: NDArray[np.float64] | NDArray[np.complex128]
    dim: int
    max_trace: int
    incr: int
    size: int

    def __init__(self, vsa: type[T], dim: int, max_trace: int = 100) -> None:
        self.vsa = vsa
        self.dim = dim
        self.max_trace = max_trace
        self.incr = max_trace
        self.size = 0

        # TODO: this is a very hacky fix, please fix this
        if vsa == HRR:
            self.memory_matrix = np.zeros((max_trace, dim))
        elif vsa == FHRR or vsa == RHC:
            self.memory_matrix = np.zeros((max_trace, dim), dtype=np.complex128)

    def memorize(self, x: T, name: str | None = None) -> T:
        if self.size >= self.max_trace:
            self.memory_matrix = np.concatenate(
                [self.memory_matrix, np.zeros((self.incr, self.dim))], axis=0
            )
            self.max_trace += self.incr

        self.memory_matrix[self.size, :] = x.data
        self.size += 1

        if name is not None:
            print(f"stored {name} at row {self.size-1}")

        return x

    def recall(self, x: T) -> T:
        # activations = [self.vsa.similarity(x.data, m) for m in self.memory_matrix]
        activations = []
        for m in self.memory_matrix:
            if x.data.dtype == m.dtype:
                sim = self.vsa.similarity(x.data, m)
            else:
                raise ValueError("Unexpected dtype for VSA")
            activations.append(sim)
        return self.vsa.from_array(self.memory_matrix[np.argmax(activations), :])  # type: ignore


class AssociativeMemory[T: (VSA[np.complex128], VSA[np.float64])]:
    """Associative memory used for semantic pointers."""

    vsa: type[T]
    dim: int
    assoc: dict[T, T]
    _theta: float

    def __init__(self, vsa: type[T], dim: int) -> None:
        self.vsa = vsa
        self.assoc = {}
        self._theta = 0.2
        self.dim = dim

    def alloc(self, trace: T) -> T:
        """Allocate trace into the associative memory, assigning it a new
        semantic pointer, and returning back that pointer.
        """
        ptr = self.vsa.new(self.dim)
        self.assoc[ptr] = trace
        return ptr

    def deref(self, ptr: T) -> T | None:
        """Dereference a semantic pointer, returning none if the dictionary
        is empty.
        """
        if len(self.assoc) == 0:
            return None

        trace = self.assoc.get(ptr)
        if trace is not None:
            return trace

        keys = list(self.assoc.keys())
        acts = [self.vsa.similarity(ptr.data, key.data) for key in keys]
        nearest_index = np.argmax(acts)

        if (acts[nearest_index]) > self._theta:
            return self.assoc[keys[nearest_index]]

        return None


class EncodingEnvironment[T: (VSA[np.complex128], VSA[np.float64])]:
    """Class for representing the encoding environment.

    Args:
        vsa (type[AnyVSA]): The type of the encoding environment, which
            will determine what kind of vector-symbols are returned from the
            encoding function.
        dim (int): The dimensionality of the vector-symbols.
    """

    vsa: type[T]
    dim: int
    codebook: dict[str, T]
    cleanup_memory: CleanupMemory[T]
    associative_memory: AssociativeMemory[T]
    integer_encoding_scheme: IntegerEncodingScheme

    def __init__(
        self,
        vsa: type[T],
        dim: int,
        integer_encoding_scheme: IntegerEncodingScheme,
    ) -> None:
        self.vsa = vsa
        self.dim = dim
        self.codebook = EncodingEnvironment.initial_codebook(self.vsa, self.dim)
        self.cleanup_memory = CleanupMemory(self.vsa, self.dim)
        self.associative_memory = AssociativeMemory(self.vsa, self.dim)
        self.integer_encoding_scheme = integer_encoding_scheme

    # initial codebook with constants
    @staticmethod
    def initial_codebook(vsa: type[T], dim: int) -> dict[str, T]:
        codebook = {}

        for keyword in KEYWORDS.keys():
            codebook[keyword] = vsa.new(dim)

        for operator in OPERATORS.keys():
            codebook[operator] = vsa.new(dim)

        for value in VALUES.keys():
            codebook[value] = vsa.new(dim)

        codebook["__phi"] = vsa.new(dim)
        codebook["__lhs"] = vsa.new(dim)
        codebook["__rhs"] = vsa.new(dim)
        codebook["__args"] = vsa.new(dim)
        codebook["__body"] = vsa.new(dim)
        codebook["__func"] = vsa.new(dim)

        return codebook


@dataclass
class EncodingError(Exception):
    """An exception for errors that occur in encoding."""

    msg: str


def encode_list_integer[T: (
    VSA[np.complex128],
    VSA[np.float64],
)](cont: str, env: EncodingEnvironment[T]) -> T:
    """Encode an integer as a list

    Args:
        cont (str): The integer.
        env (EncodingEnvironment): The encoding environment.

    Returns:
        An encoded integer using the list scheme.

    Raises:
        `EncodingError`.
    """

    try:
        conti = int(cont)
    except:
        raise EncodingError(f"`{cont}` is not a valid integer!")

    if conti < 0:
        raise EncodingError(f"{conti} is expected to be a positive number, sorry!")

    if conti == 0:
        return env.codebook["nil"]

    else:

        base = env.codebook["nil"]
        for i in range(conti):
            base = _cons(env.codebook["nil"], base, env)

        return base


def encode_rhc_integer[T: (
    VSA[np.complex128],
    VSA[np.float64],
)](cont: str, env: EncodingEnvironment[T]) -> T:
    raise EncodingError("TODO")


def encode_atom[T: (
    VSA[np.complex128],
    VSA[np.float64],
)](cont: str, env: EncodingEnvironment[T]) -> T:
    """Encode an atom. If the atom is already present, return that value,
    otherwise create a new value and return that.

    Args:
        cont (str): The atom string.
        env (EncodingEnvironment): The encoding environment.

    Returns:
        A vector symbol for that atom.

    Raises:
        `EncodingError`
    """
    if cont in env.codebook:
        return env.codebook[cont]
    else:
        new_symbol = env.vsa.new(env.dim)
        env.codebook[cont] = new_symbol
        return new_symbol


# Make a semantic pointer for tuples
def _cons[T: (
    VSA[np.complex128],
    VSA[np.float64],
)](head: T, tail: T, env: EncodingEnvironment[T]) -> T:
    head_ = env.cleanup_memory.memorize(head).data
    tail_ = env.cleanup_memory.memorize(tail).data

    tuplev = env.vsa.bundle(
        env.vsa.bind(head_, env.codebook["__lhs"].data),
        env.vsa.bundle(
            env.vsa.bind(tail_, env.codebook["__rhs"].data),
            env.codebook["__phi"].data,
        ),
    )

    ptr = env.associative_memory.alloc(env.vsa.from_array(tuplev))
    return ptr


def encode_list[T: (
    VSA[np.complex128],
    VSA[np.float64],
)](xs: list[Intr], env: EncodingEnvironment[T],) -> T:
    """Encode a list as a vector symbol.

    Args:
        xs (list[Intr]): A list of intermediate representations.
        env (EncodingEnvironment): The encoding environment.

    Returns:
        A semantic pointer that, when dereferenced using the associative
        memory, returns the raw tuple.

    Raises:
        `EncodingError`.
    """

    if len(xs) == 0:
        return env.codebook["nil"]

    head, *tail = xs
    headv = encode(head, env)

    if len(tail) == 0:
        return _cons(headv, env.codebook["nil"], env)
    else:
        tailv = encode_list(tail, env)
        return _cons(headv, tailv, env)


def encode[T: (
    VSA[np.complex128],
    VSA[np.float64],
)](intr: Intr, env: EncodingEnvironment[T],) -> T:
    """Encode an intermediate representation into the vector-symbolic
    architecture provided for as an argument to `Env`.

    Args:
        intr (Intr): The intermediate source representation.
        env (EncodingEnvironment): The encoding environment.

    Returns:
        A vector symbolic representation of the intermediate syntax.

    Raises:
        `EncodingError`.
    """

    match intr:

        case IntrAtom(x):

            if (
                x.kind == TokenKind.Int
                and env.integer_encoding_scheme == IntegerEncodingScheme.ListIntegers
            ):
                return encode_list_integer(x.cont, env)

            elif (
                x.kind == TokenKind.Int
                and env.integer_encoding_scheme == IntegerEncodingScheme.RHCIntegers
            ):
                if env.vsa not in [RHC, FHRR]:
                    raise EncodingError("RHC integer encoding mismatch")

                return encode_rhc_integer(x.cont, env)

            else:
                return encode_atom(x.cont, env)

        case IntrList(xs):

            return encode_list(xs, env)
