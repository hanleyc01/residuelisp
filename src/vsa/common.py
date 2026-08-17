import numpy as np
import numpy.typing as npt

from .vsa import VSA

__all__ = [
    "AnyVSA",
    "ArrayC128",
    "ArrayF64",
    "ArrayI64",
    "VSAdtype",
]


type ArrayF64 = npt.NDArray[np.float64]
"""Type alias for double arrays."""

type ArrayC128 = npt.NDArray[np.complex128]
"""Type alias for double complex arrays."""

type ArrayI64 = npt.NDArray[np.int64]
"""Type alias for 64-bit signed integer arrays."""

# NOTE: update this if any new VSAs are implemented which do not use the same
# data types.
type VSAdtype = np.float64 | np.complex128
"""Type alias for VSA types implemented in this module."""

type AnyVSA = VSA[np.float64] | VSA[np.complex128]
