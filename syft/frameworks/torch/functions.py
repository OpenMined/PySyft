from .tensors.interpreters import MultiPointerTensor


def unite_pointers(*pointers):
    return MultiPointerTensor(children=pointers).wrap()
