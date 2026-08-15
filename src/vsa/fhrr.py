"""Implementation of Holographic Reduced Representations in the frequency
domain (FHRR).
"""

from __future__ import annotations

import cmath
import math
from typing import Self

import numpy as np

from .common import ArrayC128
from .hrr import HRR
from .vsa import VSA


class FHRR(VSA[np.complex128]):
    """Holographic reduced representations in the frequency domain.

    To initialize a new array, call `FHRR.uniform(dim: int)`, where `dim`
    is a scalar value denoting the dimensionality of the array.
    """

    data: ArrayC128
    dtype = np.complex128

    def __init__(self, data: ArrayC128) -> None:
        self.data = data

    @classmethod
    def from_array(cls, array: ArrayC128) -> Self:
        return cls(array)

    @classmethod
    def uniform(cls, dim: int) -> Self:
        """Initialize a new FHRR sampled from the uniform distribution.

        Args:
        -   dim (int): The dimensionality of the FHRR vector symbol.

        Returns:
            A new FHRR vector symbol.
        """
        thetas = np.random.uniform(high=math.pi * 2, size=dim)
        return cls(np.exp(thetas * cmath.sqrt(-1)))

    @classmethod
    def from_hrr(cls, x: HRR) -> Self:
        """Convert an HRR into an FHRR.

        Args:
        -   x (HRR): An HRR vector symbol.

        Returns:
            An FHRR vector symbol.
        """
        complex_array = np.ndarray.astype(np.fft.fft(x.data), np.complex128)
        return cls(complex_array)

    @classmethod
    def new(cls, dim: int) -> Self:
        """Static method creating a new FHRR using `FHRR.uniform`.

        Args:
        -   dim (int): The dimensionality of the new vector symbol.

        Returns:
            A new FHRR vector symbol.
        """
        return cls.uniform(dim)

    @staticmethod
    def bind(x: ArrayC128, y: ArrayC128) -> ArrayC128:
        """The product of the FHRR VSA.

        Args:
        -   x (npt.NDArray[np.complex128]): A complex vector left-hand side.
        -   y (npt.NDArray[np.complex128]): A complex vector right-hand side.

        Returns:
            The element-wise product of both `x` and `y`.
        """
        return np.multiply(x, y)

    @staticmethod
    def inv(x: ArrayC128) -> ArrayC128:
        """Returns the approximate inverse of a vector-symbol, in FHRR this is
        the complex conjugate.

        Args:
        -   x (npt.NDArray[np.complex128]): A complex vector-symbol.

        Returns:
            The approximate inverse of the vector-symbol.
        """
        return np.conjugate(x)

    @staticmethod
    def unbind(x: ArrayC128, y: ArrayC128) -> ArrayC128:
        """The inverse of binding in the FHRR VSA.

        Args:
        -   x (npt.NDArray[np.complex128]): A complex vector left-hand side.
        -   y (npt.NDArray[np.complex128]): A complex vector right-hand side.

        Returns:
            The element-wise product of both `x` and the inverse of `y`.
        """

        return FHRR.bind(x, FHRR.inv(y))

    @staticmethod
    def bundle(x: ArrayC128, y: ArrayC128) -> ArrayC128:
        """The sum operation in the FHRR VSA.

        Args:
        -   x (npt.NDArray[np.complex128]): A complex vector left-hand side.
        -   y (npt.NDArray[np.complex128]): A complex vector right-hand side.

        Returns:
            The element-wise sum of both `x` and `y`.
        """
        return x + y

    @staticmethod
    def similarity(x: ArrayC128, y: ArrayC128) -> float:
        """Similarity comparison in the FHRR VSA.

        Args:
        -   x (npt.NDArray[np.complex128]): A complex vector left-hand side.
        -   y (npt.NDArray[np.complex128]): A complex vector right-hand side.

        Returns:
            A scalar between -1 and 1 which measures the 'distance' between
            the two vector-symbols.
        """
        return float(np.dot(np.conjugate(x.T), y).real / x.size)

    def __add__(self, rhs: FHRR | float) -> Self:
        """See `FHRR.bundle`."""
        cls = type(self)
        if isinstance(rhs, FHRR):
            return cls(cls.bundle(self.data, rhs.data))
        else:
            return cls(self.data + rhs)

    def __radd__(self, rhs: FHRR | float) -> Self:
        """See `FHRR.bundle`."""
        cls = type(self)
        if isinstance(rhs, FHRR):
            return cls(cls.bundle(self.data, rhs.data))
        else:
            return cls(self.data + rhs)

    def __sub__(self, rhs: FHRR | float) -> Self:
        """Element-wise subtraction."""
        cls = type(self)
        if isinstance(rhs, FHRR):
            return cls(self.data - rhs.data)
        else:
            return cls(self.data - rhs)

    def __mul__(self, rhs: FHRR | float) -> Self:
        """See `FHRR.bind`."""
        cls = type(self)
        if isinstance(rhs, FHRR):
            return cls(cls.bind(self.data, rhs.data))
        else:
            return cls(self.data * rhs)

    def __rmul__(self, rhs: FHRR | float) -> Self:
        """See `FHRR.bind`."""
        cls = type(self)
        if isinstance(rhs, FHRR):
            return cls(cls.bind(self.data, rhs.data))
        else:
            return cls(self.data * rhs)

    def __truediv__(self, rhs: FHRR | float) -> Self:
        """See `FHRR.unbind`."""
        cls = type(self)
        if isinstance(rhs, FHRR):
            return cls(cls.unbind(self.data, rhs.data))
        else:
            return cls(self.data / rhs)

    def __invert__(self) -> Self:
        """See `FHRR.inv`."""
        cls = type(self)
        return cls(cls.inv(self.data))

    def __neg__(self) -> Self:
        """Element-wise negation of each of the elements of the vector-symbol."""
        return type(self)(-self.data)

    def magnitude(self) -> float:
        """The magnitude of the vector-symbol."""
        return math.sqrt(self.data @ self.data) / self.data.size

    def __matmul__(self, other: FHRR | ArrayC128) -> float | ArrayC128:
        """See `np.ndarray.__matmul__`."""
        if isinstance(other, FHRR):
            return self.data @ other.data
        elif isinstance(other, np.ndarray) and other.dtype == np.float64:
            if len(other.shape) == 2:
                return (self.data @ other).astype(np.complex128)
            else:
                return self.data @ other
        else:
            raise TypeError(f"Innapropriate argument type {type(other)}")

    def sim(self, other: FHRR | ArrayC128) -> float:
        """See `FHRR.similarity`."""
        if isinstance(other, FHRR):
            return FHRR.similarity(self.data, other.data)
        elif isinstance(other, np.ndarray) and other.dtype == np.float64:
            return FHRR.similarity(self.data, other)
        else:
            raise TypeError(f"Innapropriate argument type {type(other)}")

    def __str__(self) -> str:
        return f"{type(self).__name__}({self.data})"

    def __hash__(self) -> int:
        return hash(self.data.tobytes())
