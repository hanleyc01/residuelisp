import numpy as np
import typing as t
import math
import cmath

from .vocabulary import Vocabulary

class RHC(Vocabulary):
    """Residue Hyperdimensional Computing VSA."""

    _dim: int
    _symbols: t.Dict[str, np.ndarray]
    _moduli: t.Dict[int, np.ndarray]
    _invs: t.Dict[int, np.ndarray]

    def __init__(self, dim: int, symbols: t.Dict[str, np.ndarray] = {}) -> None:
        assert dim > 0, "No such thing as negative dimension."
        self._dim = dim
        self._symbols = symbols
        self._moduli = self.make_moduli([3, 5, 7])

    @property
    def dim(self) -> int:
        return self._dim
    
    @property
    def symbols(self) -> t.Dict[str, np.ndarray]:
        return self._symbols

    def bind(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.multiply(x, y)

    def superpose(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return x + y

    def sim(self, x: np.ndarray, y: np.ndarray) -> float:
        return np.dot(x.T, np.conjugate(y)).real / x.size

    def vector_gen(self) -> np.ndarray:
        raise TypeError("Unable to generate a base vector")

    def encode(self, x: int) -> np.ndarray:
        pass

    @property
    def moduli(self) -> t.Dict[int, np.ndarray]:
        return self._moduli

    @staticmethod
    def _roots_of_unity(m: int, dim: int) -> np.ndarray:
        """
        `roots_of_unity`

        Generate `m` roots of unity of size `dim`.
        """
        incr = 2 * math.pi / m
        points_of_unity = []
        curr_value = incr
        while curr_value < 2 * math.pi:
            points_of_unity.append(curr_value)
            curr_value += incr
        
        v = np.zeros(dim)
        choose = np.vectorize(lambda _: np.random.choice(points_of_unity))
        return choose(v)

    def make_moduli(self, mods: t.List[int]) -> t.Dict[int, np.ndarray]:
        moduli = {}
        for mod in mods:
            moduli[mod] = (RHC._roots_of_unity(mod, self.dim))
        return moduli

    def __getitem__(self, key: str) -> np.ndarray:
        return self._symbols[key]