# third party
import numpy as np

# relative
# from .single_entity_phi import SingleEntityPhiTensor as SEPT
from .initial_gamma import InitialGammaTensor


# x is a SingleEntityPhiTensor, but we can't import it here since it creates a circular import error.
def convert_to_gamma_tensor(x) -> InitialGammaTensor:  # type: ignore
    """Helper function to convert (x) SEPT to InitialGammaTensor, for private-private operations"""
    return InitialGammaTensor(
        values=x.child,
        min_vals=x.min_vals,
        max_vals=x.max_vals,
        entities=np.array(x.entity, dtype=object).repeat(len(x.child.flatten())),
        scalar_manager=x.scalar_manager,
    )
