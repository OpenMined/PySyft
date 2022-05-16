import numpy as np
from .data_subject_ledger import load_cache
from .data_subject_ledger import RDPParams
from .data_subject_ledger import compute_rdp_constant


def calibrate_sigma(rdp_params: RDPParams, query_limit: int = 5) -> None:
    """
    Adjust the value of sigma chosen to have a 90% chance of being less than query_limit
    """
    rdp_constants = compute_rdp_constant(rdp_params, private=False)
    CONSTANT2EPSILSON_CACHE_FILENAME = "constant2epsilon_300k.npy"
    _cache_constant2epsilon = load_cache(filename=CONSTANT2EPSILSON_CACHE_FILENAME)

    max_rdp = np.max(rdp_constants)
    budget_spend = _cache_constant2epsilon.take((max_rdp - 1).astype(np.int64))

    if budget_spend >= query_limit:
        # calculate the value of sigma to get to

        # This is the first index in the cache that has an epsilon >= query limit
        threshold_index = np.searchsorted(_cache_constant2epsilon, query_limit)

        # There is a 90% chance the final budget spend will be below the query_limit
        selected_rdp_value = np.random.choice(np.arange(threshold_index-9, threshold_index+1))

        """
        DERIVATION (for posterity)
        
        Variables:
        - rdp_t: the target/desired rdp value
        - rdp_old: the old/previous rdp value
        - sigma_t: the value of sigma needed to get rdp_t (we're calculating this)
        - sigma_old: the previous value of sigma that was used to get rdp_old
        
        rdp_t/rdp_old = (1/sigma_t^2)/(1/sigma_old^2)
        => rdp_t/rdp_old = sigma_old^2 / sigma_t^2
        => sigma_t = sigma_old * sqrt(rdp_old / rdp_t)
        """
        new_sigma = selected_rdp_value * np.sqrt(max_rdp/selected_rdp_value)
        rdp_params.sigmas = np.ones_like(rdp_params.sigmas) * new_sigma

    return None
