"""Module for encoding intermediate representations into vectors."""

from collections import UserDict
from enum import Enum, auto

import equinox as eqx
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu

from syntax import (
    KEYWORDS,
    OPERATORS,
    VALUES,
    Intr,
    IntrAtom,
    IntrList,
    Token,
    TokenKind,
)

from .memory import CleanupMemory, AssociativeMemory
from .vsa import fhrr_random


class IntegerEncodingScheme(Enum):
    """The integer encoding scheme to be used by the encoding function.

    - `ListIntegers`: Represent integers as a list.
    - `RHCIntegers`: Represent integers as with Residue Hyperdimensional
      Computing. This only works if the VSA provided to the encoding function
      supports conversions to and from `RHC`.
    """

    ListIntegers = auto()
    RHCIntegers = auto()


class Codebook(UserDict[str, jax.Array]):
    """A Codebook is just a thin wrapper around a dictionary.

    Attributes:
    -   codebook (dict[str, jax.Array]): The codebook dictionary.
    """

    def reverse(self) -> dict[jax.Array, str]:
        """Reverse the codebook, returning a dictionary mapping each vector
        to its corresponding symbol.

        Returns:
            A dictionary mapping each vector to its corresponding symbol.
        """
        return {v: k for k, v in self.items()}

    @classmethod
    def initial_codebook(cls, key, dim: int) -> "Codebook":
        d = dict()

        keywords = KEYWORDS.keys()
        operators = OPERATORS.keys()
        values = VALUES.keys()
        internal_words = [
            "__rhs",
            "__lhs",
            "__phi",
            "__args",
            "__body",
            "__func",
            "__int",
        ]
        reserved_words = keywords + operators + values + internal_words
        keys = jr.split(key, len(reserved_words))

        for key, word in zip(keys, reserved_words):
            d[word] = fhrr_random(key, 1, dim)

        return cls(d)


class EncodingEnvironment(eqx.Module):
    """The encoding environment for encoding intermediate representations.

    Attributes:
    -   scheme (IntegerEncodingScheme): The integer encoding scheme to use.
    -   memory (CleanupMemory): The cleanup memory to use.
    -   codebook (Codebook): The codebook to use for encoding.
    """

    scheme: IntegerEncodingScheme
    integer_encoding: IntegerEncodingScheme
    cleanup_memory: CleanupMemory
    associative_memory: AssociativeMemory
    dim: int

    @classmethod
    def init(
        cls, key: jax.Array, dim: int, integer_encoding: IntegerEncodingScheme
    ) -> "EncodingEnvironment":
        codebook = Codebook.initial_codebook(key, dim)
        cleanup_memory = CleanupMemory.init(init_traces=100, dim=dim)
        associative_memory = AssociativeMemory.init(init_traces=100, dim=dim)

        for key, value in codebook.items():
            jax.debug.print(f"Initializing codebook with {key}")
            _, cleanup_memory = cleanup_memory.memorize(value)

        return cls(
            scheme=integer_encoding,
            cleanup_memory=cleanup_memory,
            associative_memory=associative_memory,
            codebook=codebook,
            dim=dim,
        )
