"""Time-domain Residue Hyperdimensional Computing."""

from typing import ClassVar, Self

import numpy as np

from .common import ArrayF64
from .hrr import HRR


class TRHC(HRR):
    """Time-domain Residue Hyperdimensional Computing.

    So-called, because it deals with RHC in the time-domain, as opposed to the frequency
    domain.
    """

    data: ArrayF64
    moduli: ClassVar[list[int]] = [3, 5, 7, 11]
    basis: ClassVar[list[ArrayF64]] = []

    @staticmethod
    def generate_base_vector(
        rng: np.random.Generator, modulus: int, dim: int
    ) -> ArrayF64:
        """Generates an RHC base vector in the time domain.

        Args:
        -   rng (np.random.Generator): The random number generator.
        -   modulus (int): The modulus of the base vector.
        -   dim (int): The dimension of the base vector.

        Returns:
            An TRHC base vector in the time domain.
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
        z_time = np.fft.ifft(z_freq)
        return z_time.real

    @classmethod
    def generate_basis(cls, dim: int) -> None:
        rng = np.random.default_rng()
        for mod in cls.moduli:
            cls.basis.append(cls.generate_base_vector(rng, mod, dim))

    @classmethod
    def number(
        cls,
        num: int,
        dim: int,
        alternative_basis: list[ArrayF64] | None = None,
    ) -> Self:
        """Create an TRHC vector from a number.

        Args:
        -   num (int): The number to convert.
        -   dim (int): The dimension of the vector.
        -   alternative_basis (list[ArrayF64] | None): An optional alternative basis to use.

        Returns:
            An TRHC vector representing the number.

        Raises:
        -   ValueError: If `alternative_basis` is provided and is empty.
        -   ValueError: If the basis dimension does not match the dimension of the vector.
        """

        if not cls.basis:
            cls.generate_basis(dim)

        basis = cls.basis

        if alternative_basis is not None and len(alternative_basis) == 0:
            raise ValueError("alternative_basis must not be empty")
        elif alternative_basis is not None and isinstance(
            alternative_basis[0], np.ndarray
        ):
            basis = alternative_basis

        if basis[0].shape[0] != dim:
            raise ValueError("basis dimension must match dim")

        rhc_num = basis[0] ** num
        for i in range(1, len(basis)):
            rhc_num = cls.bind(rhc_num, basis[i] ** num)

        return cls(rhc_num)

    @classmethod
    def residue_add(cls, x: ArrayF64, y: ArrayF64) -> ArrayF64:
        """Perform TRHC arithmetical addition.

        Args:
        -   x (ArrayF64): The first vector.
        -   y (ArrayF64): The second vector.

        Returns:
            The result of the addition.
        """
        return cls.bind(x, y)

    @classmethod
    def residue_sub(cls, x: ArrayF64, y: ArrayF64) -> ArrayF64:
        """Perform TRHC arithmetical subtraction.

        Args:
        -   x (ArrayF64): The first vector.
        -   y (ArrayF64): The second vector.

        Returns:
            The result of the subtraction.
        """
        return cls.unbind(x, y)

    @classmethod
    def residue_mul(cls, x: ArrayF64, y: ArrayF64) -> ArrayF64:
        """Perform TRHC arithmetical multiplication.

        Args:
        -   x (ArrayF64): The first vector.
        -   y (ArrayF64): The second vector.

        Returns:
            The result of the multiplication.
        """
        raise NotImplementedError("TODO")

    @classmethod
    def residue_div(cls, x: ArrayF64, y: ArrayF64) -> ArrayF64:
        """Perform TRHC arithmetical division.

        Args:
        -   x (ArrayF64): The first vector.
        -   y (ArrayF64): The second vector.

        Returns:
            The result of the division.
        """
        raise NotImplementedError("TODO")
