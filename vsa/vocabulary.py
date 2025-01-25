import numpy as np
from typing import Dict
from abc import ABCMeta, abstractmethod


class Vocabulary(metaclass=ABCMeta):
    """
    `Vocabulary`

    Abstract base class for vector symbolic algebras.
    """

    @property
    @abstractmethod
    def dim(self) -> int:
        pass

    @property
    @abstractmethod
    def symbols(self) -> Dict[str, np.ndarray]:
        pass

    @abstractmethod
    def bind(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def superpose(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def sim(self, x: np.ndarray, y: np.ndarray) -> float:
        pass

    @abstractmethod
    def vector_gen(self) -> np.ndarray:
        pass

    @abstractmethod
    def __getitem__(self, key: str) -> np.ndarray:
        pass