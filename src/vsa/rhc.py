"""Implementation of Residue Hyperdimensional Computing (RHC)."""

from typing import ClassVar

import numpy as np
from more_itertools import is_prime

from .common import ArrayC128, ArrayI64
from .fhrr import FHRR

__all__ = ["RHC"]


class RHC(FHRR):
    """Residue Hyperdimensional Computing (RHC) vector symbolic architecture."""

    data: ArrayC128
    moduli: ClassVar[list[int]] = [3, 5, 7, 11]
    basis: ClassVar[list[ArrayC128]] = []
    basis_exponents: ClassVar[list[ArrayI64]] = []
    anti_basis: ClassVar[list[ArrayC128]] = []

    @classmethod
    def set_moduli(cls, moduli: list[int]) -> None:
        """Set the moduli for the RHC basis.

        Note that, if one wants to perform multiplication, ideally
        the moduli should be chosen to be coprime.

        Args:
        -   moduli (list[int]): The moduli for the RHC basis.
        """
        cls.moduli = moduli

    @staticmethod
    def generate_base_vector(
        rng: np.random.Generator, modulus: int, dim: int
    ) -> tuple[ArrayC128, ArrayI64]:
        """Generates an RHC base vector in the frequency domain.

        Args:
        -   rng (np.random.Generator): The random number generator.
        -   modulus (int): The modulus of the base vector.
        -   dim (int): The dimension of the base vector.

        Returns:
            An RHC base vector in the frequency domain.
        """

        k_choices = np.zeros(dim, dtype=int)
        k_choices[0] = 0

        half_len = dim // 2
        k_choices[1:half_len] = rng.choice(modulus, half_len - 1)

        if dim % 2 == 0:
            k_choices[half_len] = (
                0 if rng.random() > 0.5 else (modulus // 2 if modulus % 2 == 00 else 0)
            )
            k_choices[half_len + 1 :] = -k_choices[half_len - 1 : 0 : -1]
        else:
            k_choices[half_len + 1 :] = -k_choices[half_len:0:-1]

        phases = 2 * np.pi * k_choices / modulus
        z_freq = np.exp(1j * phases)
        return z_freq, k_choices

    @classmethod
    def generate_basis(cls, dim: int) -> None:
        rng = np.random.default_rng()
        generated = [cls.generate_base_vector(rng, m, dim) for m in cls.moduli]
        cls.basis = [z for z, _ in generated]
        cls.basis_exponents = [k for _, k in generated]
        cls.anti_basis = []

    @staticmethod
    def generate_anti_base_vector(exponents: ArrayI64, modulus: int) -> ArrayC128:
        if not is_prime(modulus):
            raise ValueError(f"modulus {modulus} is not prime")

        inverse = np.array(
            [0] + [pow(t, -1, modulus) for t in range(1, modulus)], dtype=np.int64
        )

        return np.exp(2j * np.pi * inverse[exponents % modulus] / modulus).astype(
            np.complex128
        )

    @classmethod
    def generate_anti_basis(cls) -> None:
        if cls.anti_basis:
            return
        if not cls.basis_exponents:
            raise ValueError("basis must be generated before anti-basis")

        if not cls.basis:
            raise ValueError("basis must be generated before anti-basis")

        cls.anti_basis = [
            cls.generate_anti_base_vector(exponents, m)
            for exponents, m in zip(cls.basis_exponents, cls.moduli, strict=True)
        ]

    @classmethod
    def number(
        cls,
        num: int,
        dim: int,
    ) -> "RHC":
        """Create an RHC vector from a number.

        Args:
        -   num (int): The number to convert.
        -   dim (int): The dimension of the vector.

        Returns:
            An RHC vector representing the number.

        Raises:
        -   ValueError: If the basis dimension does not match the dimension of the vector.
        """

        if not cls.basis:
            cls.generate_basis(dim)

        basis = cls.basis

        if basis[0].shape[0] != dim:
            raise ValueError("basis dimension must match dim")

        rhc_num = basis[0] ** num
        for i in range(1, len(basis)):
            rhc_num = cls.bind(rhc_num, basis[i] ** num)

        return cls(rhc_num)

    @classmethod
    def residue_add(cls, x: ArrayC128, y: ArrayC128) -> ArrayC128:
        """Perform RHC arithmetical addition.

        Args:
        -   x (ArrayC128): The first vector.
        -   y (ArrayC128): The second vector.

        Returns:
            The result of the addition.
        """
        return cls.bind(x, y)

    @classmethod
    def residue_sub(cls, x: ArrayC128, y: ArrayC128) -> ArrayC128:
        """Perform RHC arithmetical subtraction.

        Args:
        -   x (ArrayC128): The first vector.
        -   y (ArrayC128): The second vector.

        Returns:
            The result of the subtraction.
        """
        return cls.unbind(x, y)

    @classmethod
    def factor(cls, x: ArrayC128) -> ArrayC128:
        r"""Recover the per-modulus components $m_k u_i$ for each modulus $m_k$ and feature $u_i$
        in the input vector `x`.

        Args:
        -   x (ArrayC128): The input vector.

        Returns:
            The per-modulus components of the input vector, $\mathbb{C}^{n \times d}$,
            where $n$ is the number of moduli and $d$ the number of features.
        """

        assert cls.moduli and cls.anti_basis and cls.basis_exponents
        assert all(is_prime(m) for m in cls.moduli)

        n = len(cls.moduli)
        d = x.shape[0] // n

        out = np.zeros((n, d), dtype=cls.dtype)

        return out

    @classmethod
    def residue_mul(cls, x: ArrayC128, y: ArrayC128) -> ArrayC128:
        r"""Perform RHC multiplicative binding.

        RHC multiplicative binding is the operation, denoted by $\otimes$, such
        that:
        $$
            RHC(x \times y) = RHC(x) \otimes RHC(y).
        $$
        """
        raise NotImplementedError("TODO")
        # assert cls.moduli and cls.anti_basis and cls.basis_exponents
        # assert all(is_prime(m) for m in cls.moduli)

        # xs = cls.factor(x)
        # ys = cls.factor(y)

        # out = np.ones_like(xs, dtype=cls.dtype)
        # for k, m in enumerate(cls.moduli):
        #     prod = cls.multiply_residue_phases(xs[k], ys[k], m)
        #     prod = cls.multiply_residue_phases(prod, cls.anti_basis[k], m)
        #     out = np.multiply(out, prod)

        # return cast(ArrayC128, out)

    @classmethod
    def residue_div(cls, x: ArrayC128, y: ArrayC128) -> ArrayC128:
        r"""Perform RHC division binding.

        RHC multiplicative binding is the operation, denoted by $\otimes^{-1]$, such
        that:
        $$
            RHC(x / y) \approx RHC(x) \otimes^{-1} RHC(y).
        $$

        Since division is not well-defined for the residue-number encoding,
        we will have to make do with a partial mapping.
        """
        raise NotImplementedError("TODO")
