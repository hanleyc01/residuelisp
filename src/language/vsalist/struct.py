"""Structured lists."""

import numpy as np

from .vsalist import VSAList


class StructList[T: np.generic](VSAList[T]): ...
