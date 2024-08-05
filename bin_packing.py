from typing import Any, Tuple
from torch import Tensor as Tensor

from slm_classes import Part


def allocate_bin_packing_2d(
    batch: Any,
    part: Part, 
    orientation: int, 
    ) -> Tuple[Any, bool]:
    ...