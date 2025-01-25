from .vocabulary import Vocabulary
from .utils import hash_array

import numpy as np
from typing import Dict, List, Tuple, Any

import math
import cmath

class FHRR(Vocabulary):
    """
    `FHRR`

    Vocabulary and associated methods for Fourier Holographic Reduced
    Representations.
    """

    _dim: int
    _symbols: Dict[str, np.ndarray]

    def __init__(
        self, dim: int, symbols: Dict[str, np.ndarray] | List[str] = {}
    ) -> None:
        self._dim = dim

        if isinstance(symbols, dict):
            self._symbols = symbols
        elif isinstance(symbols, list):
            self._symbols = {}
            for symbol in symbols:
                self._symbols[symbol] = self.vector_gen()
        else:
            raise TypeError(
                "expected `Dict[str, np.ndarray] | List[str]`", symbols
            )

    @property
    def dim(self) -> int:
        """
        `dim`

        The dimensionality of the vocabulary.
        """
        return self._dim

    @property
    def symbols(self) -> Dict[str, np.ndarray]:
        """
        `symbols`

        Associative array between strings and vectors in the vocabulary.
        """
        return self._symbols

    @symbols.setter
    def symbols(self, value: Dict[str, np.ndarray] | List[str]) -> None:
        if isinstance(value, dict):
            self._symbols = value
        elif isinstance(value, list):
            self._symbols = {}
            for symbol in value:
                self._symbols[symbol] = self.vector_gen()
        else:
            raise TypeError(
                "expected `Dict[str, np.ndarray] | List[str]`", value
            )

    def bind(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        `bind`

        Element-wise multiplication between vectors `x` and `y`.
        """
        assert x.dtype == y.dtype, "dtypes must match"
        assert x.size == y.size, "sizes must match"
        return np.multiply(x, y)

    def superpose(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        `superpose`

        Element-wise addition between vectors `x` and `y`.
        """
        assert x.dtype == y.dtype, "dtypes must match"
        assert x.size == y.size, "sizes must match"
        res = x + y
        # return res / np.linalg.norm(res)
        return res

    def vector_gen(self) -> np.ndarray:
        """
        `vector_gen`

        Initialize a new vector.
        """
        thetas = np.random.uniform(high=math.pi * 2, size=self.dim)
        return np.exp(thetas * cmath.sqrt(-1))

    def __getitem__(self, key: str) -> np.ndarray:
        return self._symbols[key]

    def sim(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        `sim`

        Vector similarity between vectors `x` and `y`.
        """
        assert x.dtype == y.dtype, "dtypes must match"
        assert x.size == y.size, "sizes must match"
        return np.dot(np.conjugate(x.T), y).real / x.size

    def inv(self, x: np.ndarray) -> np.ndarray:
        """
        `inv`

        Vector inverse.
        """
        return np.conjugate(x)

    def unbind(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        `unbind`

        Element-wise multiplication between vector `x` and the
        complex conjugate of `y`.
        """
        return self.bind(x, self.inv(y))

    def reverse(self) -> Dict[Tuple[Any, ...], str]:
        return {hash_array(v): k for (k, v) in self._symbols.items()}

    def set_symbols(self, symbols: List[str]) -> None:
        for symbol in symbols:
            self._symbols[symbol] = self.vector_gen()

    def add_symbol(self, symbol: str) -> None:
        self._symbols[symbol] = self.vector_gen()
