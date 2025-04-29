"""Implementation of Holographic Reduced Representations in the frequency
domain (FHRR).
"""

from __future__ import annotations

import cmath
import math

import numpy as np
import numpy.typing as npt

from .common import ArrayC128
from .hrr import HRR
from .vsa import VSA


class FHRR(VSA[np.complex128]):
    """Holographic reduced representations in the frequency domain.

    To initialize a new array, call `FHRR.uniform(dim: int)`, where `dim`
    is a scalar value denoting the dimensionality of the array.
    """

    data: ArrayC128

    def __init__(self, data: ArrayC128) -> None:
        self.data = data

    @staticmethod
    def from_array(data: ArrayC128) -> "FHRR":
        return FHRR(data)

    @staticmethod
    def uniform(dim: int) -> FHRR:
        thetas = np.random.uniform(high=math.pi * 2, size=dim)
        return FHRR(np.exp(thetas * cmath.sqrt(-1)))

    @staticmethod
    def from_hrr(x: HRR) -> FHRR:
        complex_array = np.ndarray.astype(np.fft.fft(x.data), np.complex128)
        return FHRR(complex_array)

    @staticmethod
    def new(dim: int) -> FHRR:
        return FHRR.uniform(dim)

    @staticmethod
    def bind(x: ArrayC128, y: ArrayC128) -> ArrayC128:
        return np.multiply(x, y)

    @staticmethod
    def inv(x: ArrayC128) -> ArrayC128:
        return np.conjugate(x)

    @staticmethod
    def unbind(x: ArrayC128, y: ArrayC128) -> ArrayC128:
        return FHRR.bind(x, FHRR.inv(y))

    @staticmethod
    def bundle(x: ArrayC128, y: ArrayC128) -> ArrayC128:
        return x + y

    @staticmethod
    def similarity(x: ArrayC128, y: ArrayC128) -> float:
        return float(np.dot(np.conjugate(x.T), y).real / x.size)

    def __add__(self, rhs: FHRR | float | int) -> FHRR:
        if isinstance(rhs, FHRR):
            return FHRR(FHRR.bundle(self.data, rhs.data))
        elif isinstance(rhs, int):
            return FHRR(self.data + rhs)
        elif isinstance(rhs, float):
            return FHRR(self.data + rhs)
        else:
            raise TypeError(f"Innapropriate argument type: {type(rhs)}")

    def __radd__(self, rhs: int | float | FHRR) -> FHRR:
        if isinstance(rhs, FHRR):
            return FHRR(self.data + rhs.data)
        elif isinstance(rhs, int):
            return FHRR(self.data + rhs)
        elif isinstance(rhs, float):
            return FHRR(self.data + rhs)
        else:
            raise TypeError(f"Innapropriate argument type: {type(rhs)}")

    def __sub__(self, rhs: int | float | FHRR) -> FHRR:
        if isinstance(rhs, FHRR):
            return FHRR(self.data - rhs.data)
        elif isinstance(rhs, int):
            return FHRR(self.data - rhs)
        elif isinstance(rhs, float):
            return FHRR(self.data - rhs)
        else:
            raise TypeError(f"Innapropriate argument type: {type(rhs)}")

    def __mul__(self, rhs: int | float | FHRR) -> FHRR:
        if isinstance(rhs, FHRR):
            return FHRR(FHRR.bind(self.data, rhs.data))
        elif isinstance(rhs, int):
            return FHRR(self.data * rhs)
        elif isinstance(rhs, float):
            return FHRR(self.data * rhs)
        else:
            raise TypeError(f"Innapropriate argument type: {type(rhs)}")

    def __rmul__(self, rhs: int | float | FHRR) -> FHRR:
        if isinstance(rhs, FHRR):
            return FHRR(FHRR.bind(self.data, rhs.data))
        elif isinstance(rhs, int):
            return FHRR(self.data * rhs)
        elif isinstance(rhs, float):
            return FHRR(self.data * rhs)
        else:
            raise TypeError(f"Innapropriate argument type: {type(rhs)}")

    def __truediv__(self, rhs: FHRR | int | float) -> FHRR:
        if isinstance(rhs, FHRR):
            return FHRR(FHRR.unbind(self.data, rhs.data))
        elif isinstance(rhs, int):
            return FHRR(self.data / rhs)
        elif isinstance(rhs, float):
            return FHRR((self.data / rhs).astype(np.float64))
        else:
            raise TypeError(f"Innapropriate argument type: {type(rhs)}")

    def __invert__(self) -> FHRR:
        return FHRR(FHRR.inv(self.data))

    def __neg__(self) -> FHRR:
        return FHRR(-self.data)

    def magnitude(self) -> float:
        return math.sqrt(self.data @ self.data) / self.data.size

    def __matmul__(self, other: FHRR | ArrayC128) -> float | ArrayC128:
        if isinstance(other, FHRR):
            return self.data @ other.data
        elif isinstance(other, np.ndarray) and other.dtype == np.float64:
            if len(other.shape) == 2:
                return (self.data @ other).astype(np.float64)
            else:
                return self.data @ other
        else:
            raise TypeError(f"Innapropriate argument type {type(other)}")

    def sim(self, other: FHRR | ArrayC128) -> float:
        if isinstance(other, FHRR):
            return FHRR.similarity(self.data, other.data)
        elif isinstance(other, np.ndarray) and other.dtype == np.float64:
            return FHRR.similarity(self.data, other)
        else:
            raise TypeError(f"Innapropriate argument type {type(other)}")

    def __str__(self) -> str:
        return f"FHRR({self.data})"

    def __hash__(self) -> int:
        return hash(self.data.tobytes())
