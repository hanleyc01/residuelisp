"""Module containing the encoding functions converting intermediate
representaitons into vector-symbolic representations.
"""

import sys
from collections import UserDict
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
    """A simple clean-up memory.

    Args:
    -   vsa (type[VSA]): The VSA class you want to pass in to use as a
            similarity comparison.
    -   dim (int): The dimensionality of the vectors.
    -   max_trace (int): The maximum amount of traces you want to
            pre-allocate. This is used as the increment.
    """

    vsa: type[T]
    """The vector symbolic architecture class that will be used for
    comparison.
    """
    memory_matrix: NDArray[np.float64] | NDArray[np.complex128]
    """The raw memory matrix."""
    dim: int
    """The dimensionality of the memory."""
    max_trace: int
    """The maximum amount of traces initially pre-allocated."""
    incr: int
    """The increment by which the memory will be increased by, assigned to
    the value of `max_trace`.
    """
    size: int
    """The current number of items in memory."""

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
        """Memorize a value into the cleanup memory.

        Args:
        -   x (VSA): A vector-symbol.
        -   name (str | None): Defaults to `None`, if not `None`, then
            will print debug information about storing the value in memory.

        Returns:
            The vector symbol `x` passed.
        """
        if self.size >= self.max_trace:
            self.memory_matrix = np.concatenate(
                [self.memory_matrix, np.zeros((self.incr, self.dim))], axis=0
            )
            self.max_trace += self.incr

        self.memory_matrix[self.size, :] = x.data
        self.size += 1

        # if name is not None:
        #     print(f"stored {name} at row {self.size-1}", file=sys.stderr)

        return x

    def recall(self, x: T) -> T:
        """Recall a value in the cleanup memory.

        Args:
        -   x (VSA): A vector-symbol to recall.

        Returns:
            The recalled trace.
        """
        activations = [self.vsa.similarity(x.data, m) for m in self.memory_matrix]
        return self.vsa.from_array(self.memory_matrix[np.argmax(activations), :])  # type: ignore


class AssociativeMemory[T: (VSA[np.complex128], VSA[np.float64])]:
    """Associative memory used for semantic pointers.

    The associative memmory pulls double duty, both as a store for semantic
    pointers to tuple chunks as well as function chunks. For more information
    about both of these values, see `.encoding.EncodingEnvironment`,
    `.encoding.make_cons`, and `.interpreter.make_function_pointer`.

    Interaction with the associative memory comes in two forms: the first is
    through *allocation*. This creates a fresh vector-symbol from the
    provided VSA and associates it with the value provided, using
    `AssociativeMemory.alloc`.

    The other way of interacting with the memory is by using `AssociativeMemory.associate`,
    which takes a key and value and directly maps them in memory.

    Retrieval is done through a single method `AssociativeMemory.deref`. It is
    named as such in order to hammer home the semantic pointer analogy, but
    it can be used in cases where directly associate two values in memory.

    Args:
    -   vsa (type[VSA]): A VSA class which has methods to be used in comparison.
    -   dim (int): The dimensionality of the vectors used.
    """

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

    def associate(self, key: T, trace: T) -> None:
        """Directly associate a key with a trace."""
        self.assoc[key] = trace

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


class Codebook[T: (VSA[np.complex128], VSA[np.float64])](UserDict[str, T]):
    """A Codebook is just a built-in dictionary, which has the additional
    `reverse` method.

    Args:
    -   data (dict[str, VSA]): A dictionary mapping strings to vector-symbols.
    """

    def reverse(self) -> dict[T, str]:
        return {v: k for (k, v) in self.data.items()}


class EncodingEnvironment[T: (VSA[np.complex128], VSA[np.float64])]:
    """Class for representing the encoding environment.

    The encoding environment contains all of the things necessary for
    encoding as well as important elements for interpretation. In particular,
    we pass to the environment a class that implements VSA (over an
    acceptable dtype), and these operations will be used in composing together
    the language.

    As slots of the class are a clean-up memory, associative memory, and
    integer encoding scheme used in encoding and interpretation. The
    clean-up memory's use is self-explanatory. However, associative memory
    pulls double duty in encoding and interpretation. It is where we 'allocate'
    and create semantic pointers to (1) tuple chunks (see `.encoding.make_cons`)
    and (2) function chunks (see `.interpreter.make_function_pointer`).

    Semantic pointers are just fresh vector-symbols, and they can be used
    in interpretation as such.

    Args:
    -   vsa (type[VSA]): The type of the encoding environment, which
            will determine what kind of vector-symbols are returned from the
            encoding function.
    -   dim (int): The dimensionality of the vector-symbols.
    -   integer_encoding_scheme (IntegerEncodingScheme): Defaults to
        `IntegerEncodingScheme.ListIntegers`.
    """

    vsa: type[T]
    dim: int
    codebook: Codebook[T]
    cleanup_memory: CleanupMemory[T]
    associative_memory: AssociativeMemory[T]
    integer_encoding_scheme: IntegerEncodingScheme

    def __init__(
        self,
        vsa: type[T],
        dim: int,
        integer_encoding_scheme: IntegerEncodingScheme = IntegerEncodingScheme.ListIntegers,
    ) -> None:
        self.vsa = vsa
        self.dim = dim
        self.codebook = Codebook(
            EncodingEnvironment.initial_codebook(self.vsa, self.dim)
        )
        self.cleanup_memory = CleanupMemory(self.vsa, self.dim)
        self.associative_memory = AssociativeMemory(self.vsa, self.dim)
        self.integer_encoding_scheme = integer_encoding_scheme

        for key, value in self.codebook.items():
            self.cleanup_memory.memorize(value, name=key)

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

        # tuple chunks
        codebook["__phi"] = vsa.new(dim)
        codebook["__lhs"] = vsa.new(dim)
        codebook["__rhs"] = vsa.new(dim)

        # function chunks
        codebook["__args"] = vsa.new(dim)
        codebook["__body"] = vsa.new(dim)
        codebook["__func"] = vsa.new(dim)

        # integers
        codebook["__int"] = vsa.new(dim)

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
    -   cont (str): The integer.
    -   env (EncodingEnvironment): The encoding environment.

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
            base = make_cons(env.codebook["nil"], base, env)

        return base


def encode_rhc_integer[T: (
    VSA[np.complex128],
    VSA[np.float64],
)](cont: str, env: EncodingEnvironment[T]) -> T:
    """Encode an integer using RHC.

    Args:
    -   cont (str): A string that is an integer.
    -   env (EncodingEnvironment): The encoding environment.
    """
    try:
        conti = int(cont)
    except:
        raise EncodingError(f"`{cont}` is not a valid integer!")

    if conti < 0:
        raise EncodingError(f"{conti} is expected to be a positive number, sorry!")

    else:
        return env.vsa.bundle(RHC.encode(env.dim, conti), env.codebook["__int"])  # type: ignore


def encode_atom[T: (
    VSA[np.complex128],
    VSA[np.float64],
)](cont: str, env: EncodingEnvironment[T]) -> T:
    """Encode an atom. If the atom is already present, return that value,
    otherwise create a new value and return that.

    Args:
    -   cont (str): The atom string.
    -   env (EncodingEnvironment): The encoding environment.

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


def make_cons[T: (
    VSA[np.complex128],
    VSA[np.float64],
)](head: T, tail: T, env: EncodingEnvironment[T]) -> T:
    """Make a semantic pointer to a tuple chunk.

    The format of the tuple chunk is similar to that of the function
    pointer chunk (see `.interpreter.make_function_pointer`). Given
    our VSA, it is formed by bundling together role-filler pairs of the values
    with `__lhs` (for the first element) and `__rhs` (for the second element)
    with a tag that denotes this as a tuple `__phi`.
    ```
    (x * __lhs) + (y * __rhs) + __phi
    ```
    We 'allocate' a pointer in our associative memory, which just means
    we create a new vector-symbol that is associated with this value,
    and return that.

    Operating over this chunk requires that we *dereference* the value. This
    functionality is given by `.encoding.AssociativeMemory.deref`, which
    merely searches through the associative memory to find the closest value
    to the semantic pointer.

    Args:
    -   head (VSA): The first element of the tuple.
    -   tail (VSA): The second element of the tuple.
    -   env (EncodingEnvironment): The encoding environment.

    Returns:
        A semantic pointer to the tuple chunk in associative memory.

    Raises:
        `EncodingError`.
    """
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
    -   xs (list[Intr]): A list of intermediate representations.
    -   env (EncodingEnvironment): The encoding environment.

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
        return make_cons(headv, env.codebook["nil"], env)
    else:
        tailv = encode_list(tail, env)
        return make_cons(headv, tailv, env)


def encode[T: (
    VSA[np.complex128],
    VSA[np.float64],
)](intr: Intr, env: EncodingEnvironment[T],) -> T:
    """Encode an intermediate representation into the vector-symbolic
    architecture provided for as an argument to `Env`.

    Args:
    -   intr (Intr): The intermediate source representation.
    -   env (EncodingEnvironment): The encoding environment.

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
