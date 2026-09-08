"""Abstract base class for VSA list encoding."""

from abc import ABC, abstractmethod
from typing import Self

import numpy as np

from vsa import VSA


class VSAList[T: np.generic](ABC, VSA[T]):
    @abstractmethod
    @classmethod
    def constructor(cls) -> Self: ...

    @abstractmethod
    def car(cls) -> Self: ...

    @abstractmethod
    @classmethod
    def cdr(cls) -> Self: ...
