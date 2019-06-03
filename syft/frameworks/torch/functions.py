from .tensors.interpreters import MultiPointerTensor
from .workers import VirtualWorker


def combine_pointers(*pointers: VirtualWorker) -> MultiPointerTensor:
    """Accepts a list of pointers and returns them as a
    MultiPointerTensor. See MultiPointerTensor docs for
    details.

    Arg:
        *pointers: a list of pointers to tensors (including
            their wrappers like normal)

    """

    return MultiPointerTensor(children=pointers).wrap()
