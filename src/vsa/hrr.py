"""Implementation of Tony Plate's Holographic Reduced Representations (HRR).

For the reference implementation, see
[this link](https://github.com/ecphory/hrr/blob/main/hrr/hrr.py).
"""

from __future__ import annotations

import math
from typing import Self, cast, override

import numpy as np
from numpy.fft import fft, ifft

from . import vsa
from .common import ArrayF64


class HRR(vsa.VSA[np.float64]):
    """Holographic reduced representation vectors.

    The vectors of HRR are sampled from a normal distribution. Implements
    binding through circular convolution.
    """

    data: ArrayF64
    dtype = np.float64

    def __init__(self, data: ArrayF64) -> None:
        self.data = data

    @classmethod
    def normal(cls, size: int, sd: float | None = None) -> Self:
        """Create a new HRR by sampling from the normal distribution.

        Args:
        -   size (int): The dimensionality of the new HRR vector-symbol.
        -   sd (float | None): Defaults to `None`, the standard deviation
            of the normal distribution.

        Returns:
            A new HRR vector-symbol.
        """
        if sd is None:
            sd = 1.0 / math.sqrt(size)
        data = np.random.normal(scale=sd, size=size)
        data /= np.linalg.norm(data)
        return cls(data)

    @override
    @classmethod
    def from_array(cls, array: ArrayF64) -> Self:
        """Create a new HRR from an array.

        Args:
        -   x (npt.NDArray[np.float64]): A raw float array.

        Returns:
            A new HRR vector-symbol, drawn from `x`.
        """
        return cls(array)

    @override
    @classmethod
    def new(cls, dim: int) -> Self:
        """Create a new vector-symbol.

        Args:
        -   dim (int): The dimensionality of the new vector-symbol.

        Returns:
            A new HRR vector-symbol.
        """
        return cls.normal(dim)

    @override
    @staticmethod
    def bind(x: ArrayF64, y: ArrayF64) -> ArrayF64:
        """The product operation in the HRR VSA.

        Args:
        -   x (npt.NDArray[np.float64]): The left-hand side of the operation.
        -   y (npt.NDArray[np.float64]): The right-hand side of the operation.

        Returns:
            The circular convolution of `x` and `y`. Here, it is implemented
            through the fast Fourier transform.
        """
        return cast(ArrayF64, ifft(fft(x) * fft(y)).real)

    @override
    @staticmethod
    def bundle(x: ArrayF64, y: ArrayF64) -> ArrayF64:
        """The HRR VSA sum operation.

        Args:
        -   x (npt.NDArray[np.float64]): The left-hand side of the operation.
        -   y (npt.NDArray[np.float64]): The right-hand side of the operation.

        Returns:
            The element-wise sum of the left-hand side and the right-hand side.
        """
        return x + y

    @staticmethod
    def inv(x: ArrayF64) -> ArrayF64:
        """The approximate inverse for HRR.

        Args:
        -   x (npt.NDArray[np.float64]): The left-hand side of the operation.

        Returns:
            The approximate inverse of `x`.
        """

        return x[np.r_[0, x.size - 1 : 0 : -1]]

    @override
    @staticmethod
    def unbind(x: ArrayF64, y: ArrayF64) -> ArrayF64:
        """The unbinding operatin in the HRR VSA.

        Args:
        -   x (npt.NDArray[np.float64]): The left-hand side of the operation.
        -   y (npt.NDArray[np.float64]): The right-hand side of the operation.

        Returns:
            The binding of the left-hand side with the approximate inverse
            of the right hand side.
        """
        return HRR.bind(x, HRR.inv(y))

    @override
    @staticmethod
    def similarity(x: ArrayF64, y: ArrayF64) -> float:
        """Approximated kernel for HRR. Measures the 'distance' between
        the left-hand and right-hand side.

        Args:
        -   x (npt.NDArray[np.float64]): The left-hand side of the operation.
        -   y (npt.NDArray[np.float64]): The right-hand side of the operation.

        Returns:
            The 'distance' between the left-hand side and the right-hand
            side, a value between -1 and 1.
        """
        mag = float(np.linalg.norm(x) * np.linalg.norm(y))
        if mag == 0.0:
            return 0.0
        else:
            return float(np.dot(x, y) / mag)

    def __add__(self, rhs: HRR | float) -> Self:
        """See `HRR.bundle`."""
        cls = type(self)
        if isinstance(rhs, HRR):
            return cls(cls.bundle(self.data, rhs.data))
        else:
            return cls(self.data + rhs)

    def __radd__(self, rhs: HRR | float) -> Self:
        """See `HRR.bundle`."""
        cls = type(self)
        if isinstance(rhs, HRR):
            return cls(cls.bundle(self.data, rhs.data))
        else:
            return cls(self.data + rhs)

    def __sub__(self, rhs: HRR | float) -> Self:
        """Element-wise subtraction."""
        cls = type(self)
        if isinstance(rhs, HRR):
            return cls(self.data - rhs.data)
        else:
            return cls(self.data - rhs)

    def __mul__(self, rhs: HRR | float) -> Self:
        """Scalar multiplication or `HRR.bind`."""
        cls = type(self)
        if isinstance(rhs, HRR):
            return cls(cls.bind(self.data, rhs.data))
        else:
            return cls(self.data * rhs)

    def __rmul__(self, rhs: HRR | float) -> Self:
        """Scalar multiplication or `HRR.bind`."""
        cls = type(self)
        if isinstance(rhs, HRR):
            return cls(cls.bind(self.data, rhs.data))
        else:
            return cls(self.data * rhs)

    def __truediv__(self, rhs: HRR | float) -> Self:
        """Scalar division or `HRR.unbind`."""
        cls = type(self)
        if isinstance(rhs, HRR):
            return cls(cls.unbind(self.data, rhs.data))
        elif isinstance(rhs, int):
            return cls(self.data / rhs)
        else:
            return cls((self.data / rhs).astype(np.float64))

    def __invert__(self) -> Self:
        """See `HRR.inv`."""
        cls = type(self)
        return cls(cls.inv(self.data))

    def __neg__(self) -> Self:
        """Element-wise negation."""
        return type(self)(-self.data)

    def magnitude(self) -> float:
        """The magnitude of the raw vector."""
        return math.sqrt(self.data @ self.data) / self.data.size

    def __matmul__(self, other: HRR | ArrayF64) -> float | ArrayF64:
        """Matrix multiplication."""
        if isinstance(other, HRR):
            return self.data @ other.data
        else:
            if len(other.shape) == 2:
                return (self.data @ other).astype(np.float64)
            else:
                return self.data @ other

    def sim(self, other: HRR | ArrayF64) -> float:
        """See `HRR.similarity`."""
        if isinstance(other, HRR):
            return HRR.similarity(self.data, other.data)
        else:
            return HRR.similarity(self.data, other)

    @override
    def __str__(self) -> str:
        return f"{type(self).__name__}({self.data})"

    @override
    def __hash__(self) -> int:
        return hash(self.data.tobytes())
