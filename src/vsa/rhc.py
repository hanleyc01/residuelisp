"""Implementation of Residue Hyperdimensional Computing (RHC)."""

from __future__ import annotations

import cmath
import math

import numpy as np
import numpy.typing as npt

from .common import ArrayC128
from .vsa import VSA


class RHC(VSA[np.complex128]):
    """Residue Hyperdimensional Computing (RHC) vector symbolic architecture."""

    moduli: list[int]
    roots: dict[int, ArrayC128]

    def __init__(
        self, moduli: list[int], roots: dict[int, ArrayC128], data: ArrayC128
    ) -> None:
        self.moduli = moduli
        self.roots = roots
        self.data = data
