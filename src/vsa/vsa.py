"""Module `vsa`.

Module defining the abstract base class of `VSA`s.
"""

from abc import ABCMeta, abstractmethod
from typing import ClassVar, Self

import numpy as np
import numpy.typing as npt

__all__ = ["VSA"]


class VSA[T: np.generic](metaclass=ABCMeta):
    """Abstract base class of all VSA implementations.

    All VSA's must implement the following methods:
    ```
    # Vector symbolic binding
    def bind(x: npt.NDArray[T], y: npt.NDArray[T]) -> npt.NDArray[T]
    # Vector symbolic bundling or superposition
    def bundle(x: npt.NDArray[T], y: npt.NDArray[T]) -> npt.NDArray[T]
    # Vector symbolic unbinding
    def unbind(x: npt.NDArray[T], y: npt.NDArray[T]) -> npt.NDArray[T]
    # Vector similarity kernel
    def similarity(x: npt.NDArray[T], y: npt.NDArray[T]) -> float
    # Generation of a new vector
    def new(cls, dim: int) -> Self
    # Generation of a new VSA vector from an array
    def from_array(cls, array: npt.NDArray[T]) -> Self
    # Conversion of the VSA vector to a hash value
    def __hash__(self) -> int
    ```
    """

    data: npt.NDArray[T]
    dtype: ClassVar[type[np.generic]]

    @staticmethod
    @abstractmethod
    def bind(x: npt.NDArray[T], y: npt.NDArray[T]) -> npt.NDArray[T]:
        """Vector symbolic binding."""
        ...

    @staticmethod
    @abstractmethod
    def bundle(x: npt.NDArray[T], y: npt.NDArray[T]) -> npt.NDArray[T]:
        """Vector symbolic bundling."""
        ...

    @staticmethod
    @abstractmethod
    def unbind(x: npt.NDArray[T], y: npt.NDArray[T]) -> npt.NDArray[T]:
        """Vector symbolic unbinding."""
        ...

    @staticmethod
    @abstractmethod
    def similarity(x: npt.NDArray[T], y: npt.NDArray[T]) -> float:
        """Vector symbolic similarity."""
        ...

    @classmethod
    @abstractmethod
    def new(cls, dim: int) -> Self:
        """Initialize a new vector."""
        ...

    @classmethod
    @abstractmethod
    def from_array(cls, array: npt.NDArray[T]) -> Self:
        """Create a VSA from an array."""
        ...

    @abstractmethod
    def __hash__(self) -> int: ...
