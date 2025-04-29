"""Implementation of Tony Plate's Holographic Reduced Representations (HRR).

For the reference implementation, see
(this link)[https://github.com/ecphory/hrr/blob/main/hrr/hrr.py].
"""

from __future__ import annotations

import math
from typing import cast

import numpy as np
import numpy.typing as npt
from numpy.fft import fft, ifft

from . import vsa
from .common import ArrayF64


class HRR(vsa.VSA[np.float64]):
    """Holographic reduced representation vectors.

    The vectors of HRR are sampled from a normal distribution. Implements
    binding through circular convolution.
    """

    data: ArrayF64

    def __init__(self, data: ArrayF64) -> None:
        self.data = data

    @staticmethod
    def normal(size: int, sd: float | None = None) -> HRR:
        if sd is None:
            sd = 1.0 / math.sqrt(size)
        data = np.random.normal(scale=sd, size=size)
        data /= np.linalg.norm(data)
        return HRR(data)

    @staticmethod
    def from_array(x: ArrayF64) -> "HRR":
        return HRR(x)

    @staticmethod
    def new(dim: int) -> HRR:
        return HRR.normal(dim)

    @staticmethod
    def bind(x: ArrayF64, y: ArrayF64) -> ArrayF64:
        return cast(ArrayF64, ifft(fft(x) * fft(y)).real)

    @staticmethod
    def bundle(x: ArrayF64, y: ArrayF64) -> ArrayF64:
        return x + y

    @staticmethod
    def inv(x: ArrayF64) -> ArrayF64:
        return x[np.r_[0, x.size - 1 : 0 : -1]]

    @staticmethod
    def unbind(x: ArrayF64, y: ArrayF64) -> ArrayF64:
        return HRR.bind(x, HRR.inv(y))

    @staticmethod
    def similarity(x: ArrayF64, y: ArrayF64) -> float:
        return float((x @ y) / np.linalg.norm(x.data))

    def __add__(self, rhs: HRR | float | int) -> HRR:
        if isinstance(rhs, HRR):
            return HRR(HRR.bundle(self.data, rhs.data))
        elif isinstance(rhs, int):
            return HRR(self.data + rhs)
        elif isinstance(rhs, float):
            return HRR(self.data + rhs)
        else:
            raise TypeError(f"Innapropriate argument type: {type(rhs)}")

    def __radd__(self, rhs: int | float | HRR) -> HRR:
        if isinstance(rhs, HRR):
            return HRR(self.data + rhs.data)
        elif isinstance(rhs, int):
            return HRR(self.data + rhs)
        elif isinstance(rhs, float):
            return HRR(self.data + rhs)
        else:
            raise TypeError(f"Innapropriate argument type: {type(rhs)}")

    def __sub__(self, rhs: int | float | HRR) -> HRR:
        if isinstance(rhs, HRR):
            return HRR(self.data - rhs.data)
        elif isinstance(rhs, int):
            return HRR(self.data - rhs)
        elif isinstance(rhs, float):
            return HRR(self.data - rhs)
        else:
            raise TypeError(f"Innapropriate argument type: {type(rhs)}")

    def __mul__(self, rhs: int | float | HRR) -> HRR:
        if isinstance(rhs, HRR):
            return HRR(HRR.bind(self.data, rhs.data))
        elif isinstance(rhs, int):
            return HRR(self.data * rhs)
        elif isinstance(rhs, float):
            return HRR(self.data * rhs)
        else:
            raise TypeError(f"Innapropriate argument type: {type(rhs)}")

    def __rmul__(self, rhs: int | float | HRR) -> HRR:
        if isinstance(rhs, HRR):
            return HRR(HRR.bind(self.data, rhs.data))
        elif isinstance(rhs, int):
            return HRR(self.data * rhs)
        elif isinstance(rhs, float):
            return HRR(self.data * rhs)
        else:
            raise TypeError(f"Innapropriate argument type: {type(rhs)}")

    def __truediv__(self, rhs: HRR | int | float) -> HRR:
        if isinstance(rhs, HRR):
            return HRR(HRR.unbind(self.data, rhs.data))
        elif isinstance(rhs, int):
            return HRR(self.data / rhs)
        elif isinstance(rhs, float):
            return HRR((self.data / rhs).astype(np.float64))
        else:
            raise TypeError(f"Innapropriate argument type: {type(rhs)}")

    def __invert__(self) -> HRR:
        return HRR(HRR.inv(self.data))

    def __neg__(self) -> HRR:
        return HRR(-self.data)

    def magnitude(self) -> float:
        return math.sqrt(self.data @ self.data) / self.data.size

    def __matmul__(self, other: HRR | ArrayF64) -> float | ArrayF64:
        if isinstance(other, HRR):
            return self.data @ other.data
        elif isinstance(other, np.ndarray) and other.dtype == np.float64:
            if len(other.shape) == 2:
                return (self.data @ other).astype(np.float64)
            else:
                return self.data @ other
        else:
            raise TypeError(f"Innapropriate argument type {type(other)}")

    def sim(self, other: HRR | ArrayF64) -> float:
        if isinstance(other, HRR):
            return HRR.similarity(self.data, other.data)
        elif isinstance(other, np.ndarray) and other.dtype == np.float64:
            return HRR.similarity(self.data, other)
        else:
            raise TypeError(f"Innapropriate argument type {type(other)}")

    def __str__(self) -> str:
        return f"HRR({self.data})"

    def __hash__(self) -> int:
        return hash(self.data.tobytes())
