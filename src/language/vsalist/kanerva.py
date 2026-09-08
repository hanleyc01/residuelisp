"""Kanerva list representation."""

import numpy as np

from .vsalist import VSAList


class KanervaList[T: np.generic](VSAList[T]): ...
