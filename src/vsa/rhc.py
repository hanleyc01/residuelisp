"""Implementation of Residue Hyperdimensional Computing (RHC)."""

from __future__ import annotations

import cmath
import math
from typing import cast

import numpy as np
import numpy.typing as npt

from .common import ArrayC128
from .fhrr import FHRR
from .vsa import VSA

DEFAULT_MODULI = [3, 5, 7, 13]


class RHC(VSA[np.complex128]):
    """Residue Hyperdimensional Computing (RHC) vector-symbolic algebra,
    as defined by [Kymn et al. (2025)](https://direct.mit.edu/neco/article/37/1/1/125267/Computing-With-Residue-Numbers-in-High-Dimensional).
    A novel vector-symbolic algebra that supports additional algebraic
    operations: namely, a pseudo-addition and pseudo-multiplication
    (pseudo as they do not obey all of the conventional properties of
    addition and multiplication).

    RHC are a bit different from other VSAs as they require more structural
    information. Given that they are an implementation of *residue arithmetic*,
    we have to carry around references to the moduli used to create vectors.

    Args:
    -   data (npt.NDArray[np.complex128]): A complex array.
    -   moduli (list[int]): The list of moduli for the residue arithmetic,
        defaults to `.rhc.DEFAULT_MODULI`. All of the moduli are co-prime
        with one another.
    -   roots (list[ArrayC128]): A list of complex arrays, where the array at
        index `i` corresponds to the complex roots of `i` in `moduli`.
    -   phis (list[ArrayC128]): A list of complex ararys which are the complex
        angle roots of unity of the corresponding moduli.
    """

    def __init__(
        self,
        data: ArrayC128,
        moduli: list[int] = DEFAULT_MODULI,
        roots: list[ArrayC128] | None = None,
        phis: list[ArrayC128] | None = None,
    ) -> None:
        self.data = data
        self.moduli = moduli

        dim = data.size
        if roots is not None:
            self.roots = roots
        else:
            self.roots = RHC.get_roots(dim, self.moduli)

        if phis is not None:
            self.phis = phis
        else:
            self.phis = RHC.get_phis(dim, self.moduli, self.roots)

    @staticmethod
    def roots_of_unity(dim: int, mod: int) -> ArrayC128:
        roots = [2 * math.pi]
        incr = (2 * math.pi) / mod
        curr = incr
        while curr < 2 * math.pi:
            roots.append(curr)
            curr += incr

        sample_roots = np.vectorize(lambda _: np.random.choice(roots))
        angles = sample_roots(np.zeros(dim, dtype=complex))
        return cast(ArrayC128, angles)

    @staticmethod
    def get_roots(dim: int, moduli: list[int]) -> list[ArrayC128]:
        roots = []
        for mod in moduli:
            roots.append(RHC.roots_of_unity(dim, mod))
        return roots

    @staticmethod
    def get_phis(
        dim: int, moduli: list[int], roots: list[ArrayC128]
    ) -> list[ArrayC128]:
        phis = []
        for i, mod in enumerate(moduli):
            phis.append(np.exp(cmath.sqrt(-1)) * roots[i])
        return phis

    @staticmethod
    def bind(x: ArrayC128, y: ArrayC128) -> ArrayC128:
        return np.multiply(x, y)

    @staticmethod
    def bundle(x: ArrayC128, y: ArrayC128) -> ArrayC128:
        return x + y

    @staticmethod
    def inv(x: ArrayC128) -> ArrayC128:
        return np.conjugate(x)

    @staticmethod
    def unbind(x: ArrayC128, y: ArrayC128) -> ArrayC128:
        return RHC.bind(x, RHC.inv(y))

    @staticmethod
    def similarity(x: ArrayC128, y: ArrayC128) -> float:
        # return float((np.dot(np.conjugate(x.T), y)) / x.size)
        return abs(FHRR.similarity(x, y))

    @staticmethod
    def from_array(x: ArrayC128) -> RHC:
        raise NotImplementedError()

    @staticmethod
    def new(dim: int) -> RHC:
        raise NotImplementedError()

    @staticmethod
    def encode(dim: int, num: int, moduli: list[int] = DEFAULT_MODULI) -> RHC:
        """Encode a natural number into RHC.

        Args:
        -   dim (int): The dimension of the RHC vector that you're requesting.
        -   num (int): The natural number you wish to encode.
        -   moduli (list[int]): Defaults to `.rhc.DEFAULT_MODULI`, list of
            coprime integers to serve as the moduli.

        Returns:
            An RHC encoded natural number.
        """

        data = np.ones(dim, dtype=complex)
        prod = RHC(data, moduli=moduli)

        phis = prod.phis
        for phi in phis:
            prod.data = RHC.bind(prod.data, phi)

        return prod**num

    def __str__(self) -> str:
        return f"RHC({self.data}, {self.moduli=})"

    def __hash__(self) -> int:
        return hash(self.data.tobytes())

    def __add__(self, rhs: RHC | float | int | complex) -> RHC:
        if isinstance(rhs, RHC):
            return RHC(
                RHC.bundle(self.data, rhs.data),
                moduli=self.moduli,
                phis=self.phis,
                roots=self.roots,
            )
        elif isinstance(rhs, float) or isinstance(rhs, int) or isinstance(rhs, complex):
            return RHC(
                self.data + rhs, roots=self.roots, moduli=self.moduli, phis=self.phis
            )
        else:
            raise TypeError(f"Inappropriate argument type: {type(rhs)}")

    def __radd__(self, rhs: RHC | float | int | complex) -> RHC:
        if isinstance(rhs, RHC):
            return RHC(
                RHC.bundle(self.data, rhs.data),
                moduli=self.moduli,
                phis=self.phis,
                roots=self.roots,
            )
        elif isinstance(rhs, float) or isinstance(rhs, int) or isinstance(rhs, complex):
            return RHC(
                rhs + self.data, roots=self.roots, moduli=self.moduli, phis=self.phis
            )
        else:
            raise TypeError(f"Inappropriate argument type: {type(rhs)}")

    def __mul__(self, rhs: RHC | float | int | complex) -> RHC:
        if isinstance(rhs, RHC):
            return RHC(
                RHC.bind(self.data, rhs.data),
                moduli=self.moduli,
                phis=self.phis,
                roots=self.roots,
            )
        elif isinstance(rhs, float) or isinstance(rhs, int) or isinstance(rhs, complex):
            return RHC(
                self.data * rhs, roots=self.roots, moduli=self.moduli, phis=self.phis
            )
        else:
            raise TypeError(f"Inappropriate argument type: {type(rhs)}")

    def __rmul__(self, rhs: RHC | float | int | complex) -> RHC:
        if isinstance(rhs, RHC):
            return RHC(
                RHC.bind(self.data, rhs.data),
                moduli=self.moduli,
                phis=self.phis,
                roots=self.roots,
            )
        elif isinstance(rhs, float) or isinstance(rhs, int) or isinstance(rhs, complex):
            return RHC(
                rhs * self.data, roots=self.roots, moduli=self.moduli, phis=self.phis
            )
        else:
            raise TypeError(f"Inappropriate argument type: {type(rhs)}")

    def __truediv__(self, rhs: RHC | float | int | complex) -> RHC:
        if isinstance(rhs, RHC):
            return RHC(
                RHC.unbind(self.data, rhs.data),
                moduli=self.moduli,
                phis=self.phis,
                roots=self.roots,
            )
        elif isinstance(rhs, float) or isinstance(rhs, int) or isinstance(rhs, complex):
            return RHC(
                self.data / rhs, roots=self.roots, moduli=self.moduli, phis=self.phis
            )
        else:
            raise TypeError(f"Inappropriate argument type: {type(rhs)}")

    def __invert__(self) -> RHC:
        return RHC(
            RHC.inv(self.data), moduli=self.moduli, roots=self.roots, phis=self.phis
        )

    def __neg__(self) -> RHC:
        return RHC(-self.data, moduli=self.moduli, roots=self.roots, phis=self.phis)

    def __matmul__(self, other: RHC | ArrayC128) -> float | ArrayC128:
        if isinstance(other, RHC):
            return self.data @ other.data
        elif isinstance(other, np.ndarray) and other.dtype == np.complex128:
            if len(other.shape) == 2:
                return (self.data @ other).astype(np.complex128)
            else:
                return self.data @ other
        else:
            raise TypeError(f"Innapropriate argument type {type(other)}")

    def sim(self, rhs: RHC | ArrayC128) -> float:
        if isinstance(rhs, RHC):
            return RHC.similarity(self.data, rhs.data)
        elif isinstance(rhs, np.ndarray) and rhs.dtype == np.complex128:
            return RHC.similarity(self.data, rhs)
        else:
            raise TypeError(f"Inappropriate argument type: {type(rhs)}")

    def __pow__(self, rhs: int) -> RHC:
        return RHC(self.data**rhs, moduli=self.moduli, roots=self.roots, phis=self.phis)
