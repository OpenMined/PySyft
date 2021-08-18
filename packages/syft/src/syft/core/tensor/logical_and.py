# relative
from .autodp.single_entity_phi import SingleEntityPhiTensor
from .passthrough import is_acceptable_simple_type


def logical_and(a, b) -> SingleEntityPhiTensor:
    if isinstance(a, SingleEntityPhiTensor):
        if is_acceptable_simple_type(b) or a.child.shape == b.child.shape:
            if isinstance(b, SingleEntityPhiTensor):
                if a.entity != b.entity:
                    return NotImplemented
                data = a.child and b.child
            else:
                data = a.child and b
            min_vals = a.min_vals * 0.0
            max_vals = a.max_vals * 0.0 + 1.0
            entity = a.entity
            return SingleEntityPhiTensor(
                child=data,
                entity=entity,
                min_vals=min_vals,
                max_vals=max_vals,
                scalar_manager=a.scalar_manager,
            )
        else:
            raise Exception(
                f"Tensor shapes do not match for __eq__: {len(a.child)} != {len(b.child)}"
            )
    elif isinstance(b, SingleEntityPhiTensor):
        return logical_and(a=b, b=a)
    else:
        if is_acceptable_simple_type(a) or (a.shape == b.shape):
            return b.__class__(b.child * a)
        elif is_acceptable_simple_type(b) or (b.shape == a.shape):
            return a.__class__(a.child * b)
        raise Exception(
            f"Tensor shapes do not match for __eq__: {len(a.child)} != {len(b.child)}"
        )
