"""Implementation of Residue Hyperdimensional Computing (RHC)."""

from __future__ import annotations

import cmath
import math
import sys
from typing import Callable, Mapping, cast

import numpy as np
import numpy.random
import numpy.typing as npt
from numpy.fft import fft, ifft

from .common import ArrayC128
from .vsa import VSA

DEFAULT_MODULI = [3, 5, 7, 11]


def z(dim: int, distr: Callable[[], float]) -> npt.NDArray[np.complex128]:
    z = np.zeros(shape=(dim,), dtype=np.complex128)
    for i in range(dim):
        z[i] = np.exp(cmath.sqrt(-1) * distr())
    return z


def fpe(dim: int, x: int, distr: Callable[[], float]) -> npt.NDArray[np.complex128]:
    """Fractional Power Encoding, from Def. 2 in Kymn et al."""
    return z(dim, distr) ** x


def fpe_mod_m(dim: int, x: int, mod: int) -> npt.NDArray[np.complex128]:
    phases = np.unique(2 * math.pi * np.arange(mod, dtype=np.float64) / mod)
    distr = np.vectorize(lambda: np.random.choice(phases))
    return fpe(dim, x, distr)


def fpe_kernel(x: npt.NDArray[np.complex128], y: npt.NDArray[np.complex128]) -> float:
    return float((x.dot(np.conj(y)).real) / x.shape[0])


class RHC(VSA[np.complex128]):
    codebook: dict[int, npt.NDArray[np.complex128]] = {}
    data: npt.NDArray[np.complex128]
    moduli: list[int] = DEFAULT_MODULI
    z_ms: list[npt.NDArray[np.complex128]] = []

    def __init__(
        self,
        data: npt.NDArray[np.complex128],
        moduli: list[int] = DEFAULT_MODULI,
        z_m: list[npt.NDArray[np.complex128]] | None = None,
    ) -> None:
        self.data = data
        self.moduli = moduli
        if z_m is None:
            self.z_ms = [fpe_mod_m(data.size, 1, mod) for mod in self.moduli]
        else:
            self.z_ms = z_m

    @staticmethod
    def encode(dim: int, x: int, moduli: list[int] = DEFAULT_MODULI) -> RHC:
        if moduli != RHC.moduli:
            RHC.moduli = moduli
            RHC.z_ms = [fpe_mod_m(dim, 1, mod) for mod in RHC.moduli]

        if not all(z_m.size == dim for z_m in RHC.z_ms):
            RHC.z_ms = [fpe_mod_m(dim, 1, mod) for mod in RHC.moduli]

        data: npt.NDArray[np.complex128] = np.ones(shape=(dim,), dtype=np.complex128)
        for z_m in RHC.z_ms:
            data = data * z_m
        RHC.codebook[x] = data**x
        return RHC(data**x)

    @staticmethod
    def bind(
        x: npt.NDArray[np.complex128], y: npt.NDArray[np.complex128]
    ) -> npt.NDArray[np.complex128]:
        return x * y

    @staticmethod
    def unbind(
        x: npt.NDArray[np.complex128], y: npt.NDArray[np.complex128]
    ) -> npt.NDArray[np.complex128]:
        return RHC.bind(x, np.conj(y))

    @staticmethod
    def bundle(
        x: npt.NDArray[np.complex128], y: npt.NDArray[np.complex128]
    ) -> npt.NDArray[np.complex128]:
        return x + y

    @staticmethod
    def inv(x: npt.NDArray[np.complex128]) -> npt.NDArray[np.complex128]:
        return np.conjugate(x)

    @staticmethod
    def from_array(data: npt.NDArray[np.complex128]) -> RHC:
        raise Exception("TODO")

    @staticmethod
    def new(dim: int) -> RHC:
        raise NotImplementedError()

    @staticmethod
    def similarity(
        x: npt.NDArray[np.complex128], y: npt.NDArray[np.complex128]
    ) -> float:
        sim = fpe_kernel(x, y)
        return sim

    def __str__(self) -> str:
        return f"RHC({self.data=}, {self.moduli})"

    def __hash__(self) -> int:
        return hash(self.data.tobytes())

    def __add__(
        self, rhs: RHC | VSA[np.complex128] | VSA[np.float64] | float | int | complex
    ) -> RHC:
        if isinstance(rhs, RHC):
            return RHC(
                RHC.bundle(self.data, rhs.data), moduli=self.moduli, z_m=self.z_ms
            )
        if isinstance(rhs, VSA):
            return RHC(self.data + rhs.data, moduli=self.moduli, z_m=self.z_ms)
        elif isinstance(rhs, float) or isinstance(rhs, int) or isinstance(rhs, complex):
            return RHC(self.data + rhs, moduli=self.moduli)
        else:
            raise TypeError(f"Inappropriate argument type: {type(rhs)}")

    def __sub__(
        self, rhs: RHC | VSA[np.complex128] | VSA[np.float64] | float | int | complex
    ) -> RHC:
        if isinstance(rhs, RHC):
            return RHC(
                self.data - rhs.data,
                moduli=self.moduli,
            )
        if isinstance(rhs, VSA):
            return RHC(
                self.data - rhs.data,
                moduli=self.moduli,
            )
        elif isinstance(rhs, float) or isinstance(rhs, int) or isinstance(rhs, complex):
            return RHC(
                self.data - rhs,
                moduli=self.moduli,
            )
        else:
            raise TypeError(f"Inappropriate argument type: {type(rhs)}")

    def __radd__(
        self, rhs: RHC | VSA[np.complex128] | VSA[np.float64] | float | int | complex
    ) -> RHC:
        if isinstance(rhs, RHC):
            return RHC(
                RHC.bundle(self.data, rhs.data),
                moduli=self.moduli,
            )
        elif isinstance(rhs, VSA):
            return RHC(
                self.data + rhs.data,
                moduli=self.moduli,
            )
        elif isinstance(rhs, float) or isinstance(rhs, int) or isinstance(rhs, complex):
            return RHC(
                rhs + self.data,
                moduli=self.moduli,
            )
        else:
            raise TypeError(f"Inappropriate argument type: {type(rhs)}")

    def __mul__(self, rhs: RHC | float | int | complex) -> RHC:
        if isinstance(rhs, RHC):
            return RHC(
                RHC.bind(self.data, rhs.data),
                moduli=self.moduli,
            )
        elif isinstance(rhs, float) or isinstance(rhs, int) or isinstance(rhs, complex):
            return RHC(
                self.data * rhs,
                moduli=self.moduli,
            )
        else:
            raise TypeError(f"Inappropriate argument type: {type(rhs)}")

    def __rmul__(self, rhs: RHC | float | int | complex) -> RHC:
        if isinstance(rhs, RHC):
            return RHC(
                RHC.bind(self.data, rhs.data),
                moduli=self.moduli,
            )
        elif isinstance(rhs, float) or isinstance(rhs, int) or isinstance(rhs, complex):
            return RHC(
                rhs * self.data,
                moduli=self.moduli,
            )
        else:
            raise TypeError(f"Inappropriate argument type: {type(rhs)}")

    def __truediv__(self, rhs: RHC | float | int | complex) -> RHC:
        if isinstance(rhs, RHC):
            return RHC(
                RHC.unbind(self.data, rhs.data),
                moduli=self.moduli,
            )
        elif isinstance(rhs, float) or isinstance(rhs, int) or isinstance(rhs, complex):
            return RHC(self.data / rhs, moduli=self.moduli)
        else:
            raise TypeError(f"Inappropriate argument type: {type(rhs)}")

    def __invert__(self) -> RHC:
        return RHC(RHC.inv(self.data), moduli=self.moduli)

    def __neg__(self) -> RHC:
        return RHC(-self.data, moduli=self.moduli)

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
        return RHC(self.data**rhs, moduli=self.moduli)


def crt(residues: list[int], moduli: list[int]) -> int:
    """Chinese remainder theorem decoding of residues given moduli."""
    import math

    M = math.prod(moduli)
    x = 0
    for a_i, m_i in zip(residues, moduli):
        M_i = M // m_i
        t_i = pow(M_i, -1, m_i)
        x += a_i * M_i * t_i
    return x % M


# generate codebooks for moduli
def _get_codebooks(
    moduli: list[int], z_ms: list[npt.NDArray[np.complex128]]
) -> Mapping[int, npt.NDArray[np.complex128]]:
    codebooks = {}
    for i, mod in enumerate(moduli):
        codes = np.zeros(shape=(z_ms[0].size, mod), dtype=np.complex128)
        for j in range(mod):
            codes[:, j] = z_ms[i] ** j
        codebooks[mod] = codes
    return codebooks


def _act(v: npt.NDArray[np.complex128]) -> npt.NDArray[np.complex128]:
    mag = np.abs(v)
    return np.divide(v, mag, where=(mag != 0))


def resonator_decoding(residue_number: RHC, max_iters: int = 200) -> tuple[
    list[dict[int, npt.NDArray[np.complex128]]],
    Mapping[int, npt.NDArray[np.complex128]],
]:
    """Resonator decoding for RHC."""
    data = residue_number.data
    z_ms = residue_number.z_ms
    moduli = residue_number.moduli
    codebooks = _get_codebooks(moduli, z_ms)

    factors = {}
    for mod, book in codebooks.items():
        factors[mod] = cast(npt.NDArray[np.complex128], np.sum(book, axis=1))

    iters = [factors]
    for i in range(max_iters):
        prev_it = iters[-1]
        curr_it = {}
        for mod in moduli:
            other_fs = np.array(
                [other_factor for omod, other_factor in prev_it.items() if omod != mod]
            ).T
            other_fs = np.prod(np.conj(other_fs), axis=1)
            raw_guess = data * other_fs
            cbook = codebooks[mod]
            curr_it[mod] = _act(cbook @ np.conj(cbook).T @ (raw_guess))
        iters.append(curr_it)

    return iters, codebooks
