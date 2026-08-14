"""Module `vsa`.

Module defining the abstract base class of `VSA`s.
"""

from abc import ABCMeta, abstractmethod
from typing import Self

import numpy as np
import numpy.typing as npt


class VSA[T: np.generic](metaclass=ABCMeta):
    """Abstract base class of all VSA implementations.

    We define this class in order
    """

    data: npt.NDArray[T]

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
