"""Module `vsa`.

Module defining the abstract base class of `VSA`s.
"""

from abc import ABCMeta, abstractmethod, abstractproperty
from typing import Any

import numpy.typing as npt


class VSA[T: Any](metaclass=ABCMeta):
    """Abstract base class of all VSA implementations.

    We define this class in order
    """

    data: npt.NDArray[T]

    @classmethod
    @abstractmethod
    def bind(cls, x: npt.NDArray[T], y: npt.NDArray[T]) -> npt.NDArray[T]:
        """Vector symbolic binding."""
        ...

    @classmethod
    @abstractmethod
    def bundle(cls, x: npt.NDArray[T], y: npt.NDArray[T]) -> npt.NDArray[T]:
        """Vector symbolic bundling."""
        ...

    @classmethod
    @abstractmethod
    def unbind(cls, x: npt.NDArray[T], y: npt.NDArray[T]) -> npt.NDArray[T]:
        """Vector symbolic unbinding."""
        ...

    @classmethod
    @abstractmethod
    def similarity(cls, x: npt.NDArray[T], y: npt.NDArray[T]) -> float:
        """Vector symbolic similarity."""
        ...

    @classmethod
    @abstractmethod
    def new(cls, dim: int) -> "VSA[T]":
        """Initialize a new vector."""
        ...

    @classmethod
    @abstractmethod
    def from_array(cls, array: npt.NDArray[T]) -> "VSA[T]":
        """Create a VSA from an array."""
        ...

    @abstractmethod
    def __hash__(self) -> int: ...
