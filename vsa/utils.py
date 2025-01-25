import numpy as np
from typing import Tuple, Any


def hash_array(x: np.ndarray) -> Tuple[Any, ...]:
    return (*x,)