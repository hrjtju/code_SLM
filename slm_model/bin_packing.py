from typing import Any, Tuple, Union
from torch import Tensor as Tensor

from rectpack import *

# Version Using Rectpack Package
def allocate_bin_packing_2d(
    batch: Union[PackerBBF, PackerBFF, PackerBNF, PackerGlobal, PackerOnlineBBF, PackerOnlineBFF, PackerOnlineBNF],
    part_info: dict
    ) -> Tuple[Any, bool]:
    
    before_pack_ls = batch.rect_list()
    
    # try adding part
    batch.add_rect(part_info["L"], part_info["W"], rid={"type": part_info["type"], "height": part_info["H"]})
    
    after_pack_ls = batch.rect_list()
    
    # Returns True if the the rect is added to the bin, otherwise return False.
    if len(before_pack_ls) == len(after_pack_ls):
        return batch, False
    else:
        return batch, True