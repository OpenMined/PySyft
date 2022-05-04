# third party
import flax
from jax import numpy as jnp


# TODO: Sq_l2 value may (?) be a float if min/max val provided are floats and not array of floats.
def epsilon_spent(
    sigma: float, sq_l2_norm_value: jnp.array, lipschitz_bound: float, alpha: int
) -> jnp.array:
    """
    This calculates the privacy budget (epsilon) spent at a DATA SUBJECT level.
    This is based on the Individual Privacy Accounting via a Renyi Filter paper (https://arxiv.org/abs/2008.11193)

    - ALPHA: the ORDER of the Renyi Divergence used in Renyi Differential Privacy
    - SIGMA: normalized noise level- std divided by global L2 sensitivity

    - LIPSCHITZ_BOUND: Lipschitz constant of a query with respect to the output of a query on a data point
    For linear queries- this is equal to 1
    For non-linear queries- this can be found using the GammaTensor.lipschitz_bound property

    - SQ_L2_NORM_VALUE: This is the L2 norm.
    IF THIS IS CALCULATED USING THE REAL VALUES OF EACH DATA SUBJECT (i.e. GammaTensor.value) -> this is PRIVATE, and
    any privacy budget calculated with this value CANNOT be shown to the User/Data Scientist.
    To calculate a Privacy budget that you CAN show to the data scientist, please pass in the upper bound of the
    squared L2 norm. This is calculated by using the metadata (max val and min val ) instead of the real values.
    """

    if sigma <= 0:
        raise Exception("Sigma should be above 0")
    if alpha <= 0:
        raise Exception(
            "Alpha (order of Renyi Divergence in RDP) should be a positive integer"
        )
    return alpha * (lipschitz_bound**2) * sq_l2_norm_value / (2 * (sigma**2))


@flax.struct.dataclass
class GaussianMechanism:
    sigma: float
    public_sq_l2_norm: jnp.array
    private_sq_l2_norm: jnp.array
    lipschitz_bound: float
    entity_indices: jnp.array
    entity_mappings: jnp.array
    RDP_off: bool = False
    approxDP_off: bool = False
    delta0: float = 0.0
    name: str = "Gaussian"

    def __post_init__(self) -> None:
        # TODO: Check to see if public or private value should be passed in
        if self.private_sq_l2_norm:
            _ = epsilon_spent(
                sigma=self.sigma,
                sq_l2_norm_value=self.private_sq_l2_norm,
                lipschitz_bound=self.lipschitz_bound,
                alpha=0,
            )
        pass
