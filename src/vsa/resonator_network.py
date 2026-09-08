"""Implementation of the [Resonator Network](https://direct.mit.edu/neco/article-abstract/32/12/2311/95651/Resonator-Networks-1-An-Efficient-Solution-for?redirectedFrom=fulltext).

The goal is that the Resonator network should be VSA-agnostic.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import reduce
from typing import Callable, Literal, Self, cast

import numpy as np
import numpy.typing as npt

from vsa.common import ArrayC128
from vsa.vsa import VSA

__all__ = [
    "ConvergenceError",
    "Decoding",
    "ResonatorNetwork",
    "identity",
    "phasor_projection",
]

VSAS = ["fhrr", "hrr", "rhc", "trhc"]
VSA_TYPES = Literal["fhrr", "hrr", "rhc", "trhc"]


@dataclass
class ConvergenceError(Exception):
    """Raised when decoding has failed to converge."""

    msg: str
    estimates: list[npt.NDArray[np.generic]]
    iters: int

    def __str__(self) -> str:
        return f"{self.msg} (after {self.iters} iterations)"


@dataclass
class Decoding[T: np.generic]:
    """Result of a successful factorization."""

    indices: tuple[int, ...]
    """Argmax indices into each codebook"""
    estimates: list[npt.NDArray[T]]
    iters: int
    similarity: float
    """The similarity between the estimates and the target."""


def phasor_projection(x: ArrayC128) -> ArrayC128:
    """Preserve the phase of each component, discard its magnitude."""
    mag = np.abs(x)
    return np.divide(x, mag, out=np.ones_like(x), where=mag > 0)


def identity[T: np.generic](x: npt.NDArray[T]) -> npt.NDArray[T]:
    """No nonlinearity."""
    return x


# TODO: add support for single length codebooks
@dataclass
class ResonatorNetwork[T: np.generic]:
    r"""Factorization of a highly-bound VSA representation $s \approx \text{bind}(x_1, x_2, ..., x_F)$ by iterating:
    $$
    x_i(t+1) = g \left( X_i X_i^H \,\text{unbind}(s, \otimes_{j \neq i} x_j(t)) \right),
    $$
    where $X_i$ is the codebook and $g$ is a non-linearity.
    """

    vsa: type[VSA[T]]
    codebooks: Sequence[npt.NDArray[T]]
    non_linearity: Callable[[npt.NDArray[T]], npt.NDArray[T]] = field(default=identity)
    max_iters: int = field(default=100)
    sim_threshold: float = field(default=0.95)
    synchronous: bool = field(default=True)
    rng: np.random.Generator = field(default=np.random.default_rng())

    @property
    def dim(self) -> int:
        """Dimensionality of the input patterns."""
        return self.codebooks[0].shape[1]

    @property
    def num_factors(self) -> int:
        """Number of factors that the network is estimating."""
        return len(self.codebooks)

    def __post_init__(self) -> None:
        assert all(cb.shape[1] == self.dim for cb in self.codebooks), (
            "Each entry of each codebook should have the same dimensionality"
        )

    def project(self, codebook: npt.NDArray[T], x: npt.NDArray[T]) -> npt.NDArray[T]:
        """Similarity measure between the codebook and the input patterns.

        Args:
        -   codebook: The codebook to project into.
        -   x: The input pattern to project.

        Returns:
            The projected pattern.
        """
        return np.conjugate(codebook) @ x / self.dim

    def cleanup(
        self, codebook: npt.NDArray[T], x: npt.NDArray[T]
    ) -> tuple[npt.NDArray[T], npt.NDArray[T]]:
        """Project the input back into the codebook space.

        Args:
        -   codebook: The codebook to project into.
        -   x: The input pattern to project.

        Returns:
            The projected pattern.
        """
        coeffs = self.project(codebook, x)
        return self.non_linearity(
            cast(npt.NDArray[T], np.matmul(coeffs, codebook))
        ), coeffs

    def residual(
        self, s: npt.NDArray[T], estimates: Sequence[npt.NDArray[T]], i: int
    ) -> npt.NDArray[T]:
        """Unbind the every factor but the i-th from the target pattern.

        Args:
        -   s: The target pattern.
        -   estimates: The estimated factors.
        -   i: The index of the factor to unbind.

        Returns:
            The residual pattern.
        """
        others = reduce(self.vsa.bind, (e for j, e in enumerate(estimates) if j != i))
        return self.vsa.unbind(s, others)

    def step(
        self, s: npt.NDArray[T], estimates: Sequence[npt.NDArray[T]]
    ) -> tuple[list[npt.NDArray[T]], list[npt.NDArray[T]]]:
        """One step over all of the factors."""

        working = list(estimates)
        new, all_coeffs = [], []
        for i, codebook in enumerate(self.codebooks):
            source = estimates if self.synchronous else working
            est, coeffs = self.cleanup(codebook, self.residual(s, source, i))
            new.append(est)
            all_coeffs.append(coeffs)
            working[i] = est
        return new, all_coeffs

    def initial_estimates(self) -> list[npt.NDArray[T]]:
        """The initial estimates for the entire network, which is the superposition of each entry in the codebook."""
        return [self.non_linearity(cb.sum(axis=0)) for cb in self.codebooks]

    def decode(self, s: npt.NDArray[T]) -> Decoding[T]:
        estimates: Sequence[npt.NDArray[T]] = self.initial_estimates()
        seen: set[tuple[int, ...]] = set()

        for it in range(1, self.max_iters + 1):
            estimates, coeffs = self.step(s, estimates)
            indices = tuple(int(np.argmax(np.abs(c))) for c in coeffs)

            if indices in seen:
                guess = reduce(
                    self.vsa.bind,
                    (cb[k] for cb, k in zip(self.codebooks, indices, strict=True)),
                )
                sim = self.vsa.similarity(guess, s)
                if sim >= self.sim_threshold:
                    return Decoding(indices, estimates, it, sim)
                raise ConvergenceError(
                    "settled on spurious fixed point / limit cycle", estimates, it
                )

            seen.add(indices)

        raise ConvergenceError("exceeded max iterations", estimates, self.max_iters)

    def with_(self, **kwargs) -> Self:
        return type(self)(**{**self.__dict__, **kwargs})
