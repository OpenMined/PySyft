from typing import List

from syft.generic.pointers.multi_pointer import MultiPointerTensor
from syft.generic.pointers.object_pointer import ObjectPointer


def combine_pointers(*pointers: List[ObjectPointer]) -> MultiPointerTensor:
    """Accepts a list of pointers and returns them as a
    MultiPointerTensor. See MultiPointerTensor docs for
    details.

    Arg:
        *pointers: a list of pointers to tensors (including
            their wrappers like normal)

    """

    return MultiPointerTensor(children=pointers).wrap()
