import numpy as np
from vsa.vocabulary import Vocabulary
from typing import Dict, List, Tuple, Literal
import math
import cmath


class RHC(Vocabulary):
    """
    Residue Hyperdimensional Computing.

    Attributes:
        dim: An integer representing the dimensionality of the vectors produced.
        symbols: An optional dictionary mapping labels to symbols.
        moduli: A list of integers which are the moduli of the encoding scheme.
    """

    _dim: int
    _symbols: Dict[str, np.ndarray]
    _moduli: List[int]
    _roots: Dict[int, np.ndarray]

    _codebook: Dict[int, np.ndarray]

    def __init__(
        self, dim: int, moduli: List[int], symbols: Dict[str, np.ndarray] = {}
    ) -> None:
        self._dim = dim
        self._symbols = symbols
        self._moduli = moduli
        self.__post_init__()

    def __post_init__(self) -> None:
        self._roots, self._phis = self._get_phis()

        for key, value in self._phis.items():
            self._symbols[str(key)] = value

        self._invs = self._get_invs()

        self._codebook = {}
        for i in range(-100, 10, 1):
            self._codebook[i] = self.encode(i)

    @property
    def dim(self) -> int:
        """
        The dimensionality of the RHC.
        """
        return self._dim

    @property
    def symbols(self) -> Dict[str, np.ndarray]:
        """
        The set of symbols and their values.
        """
        return self._symbols

    @property
    def moduli(self) -> List[int]:
        """
        The moduli of the RHC.
        """
        return self._moduli

    def _get_phis(self) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
        """
        For each modulus `m`, sample from the `m`-th roots of unity to
        produce an `self.dim`-dimensional vector.

        Returns:
            A tuple of a dictionary mapping moduli to their real roots of unity,
            and a dictionary mapping moduli to the complex angle roots of unity.
        """

        real_phis = {}
        phis = {}
        for modulus in self.moduli:
            real_phis[modulus] = self._roots_of_unity(modulus)
            phis[modulus] = np.exp(cmath.sqrt(-1) * real_phis[modulus])
        return real_phis, phis

    def _roots_of_unity(self, modulus: int) -> np.ndarray:
        """Sample from the `modulus` roots of unity to create a
            `self.dim`-dimensional vector.

        Args:
            modulus: An integer `m` to sample from the `m`th roots of unity.

        Returns:
            A real vector of sampled elements from the `modulus` roots of unity.
        """

        # generate the `modulus`-roots of unity of the unit circle
        roots = [2 * math.pi]
        incr = (2 * math.pi) / modulus
        curr = incr
        while curr < 2 * math.pi:
            roots.append(curr)
            curr += incr

        # create a vector of angles sampled from the `modulus`-roots of unity
        sample_roots = np.vectorize(lambda _: np.random.choice(roots))
        angles = sample_roots(np.zeros(self.dim))
        return angles

    def _get_invs(self) -> Dict[int, np.ndarray]:
        """
        Anti-base vectors for each moduli defined by the modular
        multiplicative inverses of the real angles.

        Returns:
            A dictionary with the moduli as the keys and the inverses
            of their associated vectors as values.
        """
        # Implementation is ripped from `inverse_phases` function by Kymn
        invs = {}
        for modulus, roots in self._roots.items():
            inv = np.zeros_like(roots)
            for i in range(roots.size):
                if np.round(np.angle(roots[i])).astype(int) == 0:
                    inv[i] = 0
                else:
                    spin = int(np.round(np.angle(roots[i]) * modulus))
                    inv[i] = pow(int(np.round(roots[i])), -1, modulus)
            invs[modulus] = inv
        return invs

    def bind(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Bind two `RHC` vectors to create a new, orthogonal vector.

        Returns:
            A new `self.dim`-dimensional vector.
        """
        return np.multiply(x, y)

    def superpose(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Put vectors `x` and `y` into superposition.

        Returns:
            A new `self.dim`-dimensional vector.
        """
        return x + y

    def inv(self, x: np.ndarray) -> np.ndarray:
        """
        Invert a `self.dim`-dimensional vector.

        Returns:
            A new `self.dim`-dimensional vector which is the conjugate of `x`.
        """
        return np.conjugate(x)

    def add(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Add two encoded integers, or: `x (n_1) * x (n_2) = x (n_1 + n_2)`

        Returns:
            A new complex `np.ndarray` which satisfies the above property.
        """
        return self.bind(x, y)

    def _resonator_mul(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Implement multiplicative binding by decoding `y` into an integer,
        and then performing element-wise exponentiation.

        Returns:
            A new complex `np.ndarray`.
        """
        y_n = self.decode(y)
        return np.pow(x, y)

    def _kymn_mul(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Implement Kymn's method for multiplicative binding, which requires
        calling a resonator network to recover the base vectors of `x`
        and `y`, and then using the anti-base vector for each to
        perform multiplicative binding.

        Returns:
            A new complex `np.ndarray`.
        """
        raise NotImplementedError("TODO")

    def mul(
        self,
        x: np.ndarray,
        y: np.ndarray,
        decoder_method: Literal["decode", "kymn"] = "decode",
    ) -> np.ndarray:
        if decoder_method == "decode":
            return self._resonator_mul(x, y)
        elif decoder_method == "kymn":
            return self._kymn_mul(x, y)
        else:
            raise ValueError("Unexpected decoder variant", decoder_method)

    def sub(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Subtract two encoded integers.
        """
        return self.add(x, self.inv(y))

    def div(
        self,
        x: np.ndarray,
        y: np.ndarray,
        decoder_method: Literal["decode", "kymn"] = "decode",
    ) -> np.ndarray:
        raise NotImplementedError("TODO")

    def sim(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Args:
            x: A `self.dim`-dimensional vector.
            y: A `self.dim`-dimensional vector.

        Returns:
            A float in `[-1, 1]` measuring the similarity between the two
                vectors. `0` if orthogonal, `-1, 1` if related.
        """
        return np.dot(x.T, np.conjugate(y)).real / self.dim

    def encode(self, n: int) -> np.ndarray:
        """
        Encode integer `n` into an RHC vector which is the Hadamard product
        of each the moduli vectors exponentiated to `n`.

        Returns:
            A complex `self.dim`-dimensional vector.
        """
        prod = np.ones(self.dim)
        for mod in self._phis.values():
            prod = self.bind(prod, mod)
        return prod**n

    def _residue_decode(self, es: List[int]) -> int:
        """
        Decode a list of integers using the Chinese remainder theorem to its
        corresponding integer.

        For more information about the algorithm used, refer to Garner (1959),
        and https://personal.utdallas.edu/~ivor/ce6305/m5p.pdf.

        Returns:
            An integer corresponding to the unique list of values provided by
                `es`.
        """
        m = np.prod(self.moduli).astype(int)

        x = 0
        for i in range(len(es)):
            m_i = self.moduli[i]
            m_i_inv = pow(m, -1, mod=m_i)
            x += ((m / m_i) * ((m_i_inv * es[i]) % m_i)) % m

        return x

    def gt(self, x: np.ndarray, y: np.ndarray) -> bool:
        raise NotImplementedError("TODO")

    def lt(self, x: np.ndarray, y: np.ndarray) -> bool:
        raise NotImplementedError("TODO")

    def decode(self, x: np.ndarray) -> int:
        """
        Decode some `RHC` hypervector `x` into an integer using a codebook.

        Returns:
            An integer.

        Raises:
            ValueError if there is no convergence.
        """
        ints, encs = tuple(zip(*self._codebook.items()))
        encs = [self.sim(x, enc) for enc in encs]
        return ints[np.argmax(encs)]

    def vector_gen(self) -> np.ndarray:
        raise TypeError(
            "TypeError: cannot just generate new RHC vectors in this implementation"
        )

    def __getitem__(self, key: str) -> np.ndarray:
        return self.symbols[key]
