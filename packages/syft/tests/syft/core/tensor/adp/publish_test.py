# # stdlib
# from typing import Any
#
# # third party
# import numpy as np
# from numpy.typing import ArrayLike
# import pytest
#
# # syft absolute
# import syft as sy
# from syft.core.adp.data_subject_ledger import DataSubjectLedger
# from syft.core.adp.data_subject_list import DataSubjectArray
# from syft.core.adp.ledger_store import DictLedgerStore
# from syft.core.tensor.autodp.gamma_tensor import GammaTensor
# from syft.core.tensor.autodp.phi_tensor import PhiTensor as PT
# from syft.core.tensor.lazy_repeat_array import lazyrepeatarray as lra


def test_gamma_inputs() -> None:
    """Test that exceptions are thrown for incorrect inputs when calling GT.publish()"""
    pass


def test_phi_inputs() -> None:
    """Test that exceptions are thrown for incorrect inputs when calling PT.publish()"""
    pass


def test_imaginary_bounds_for_mechanism() -> None:
    """Test that the system fails with grace when l2_norm is imaginary for some reason"""
    pass


def test_sigma_too_large() -> None:
    """Test that sigma being too large results in low PB spend"""
    pass


def test_sigma_too_small() -> None:
    """Test that sigma being too small results in high PB spend"""
    pass


def test_ledger_creating() -> None:
    """Test that a new Data Subject Ledger was initialized properly"""
    pass


def test_ledger_fetching() -> None:
    """Test that an old Data Subject Ledger was fetched properly"""
    pass


def test_publish_high_values() -> None:
    """Test that publish works when the Tensor has gigantic values"""
    pass


def test_publish_single_value() -> None:
    """Test that publish works when the Tensor only has 1 value in it"""
    pass


def test_rdp_constants_gigantic() -> None:
    """Test that rdp_constants work if they're gigantic."""
    pass


def test_rdp_constants_negative() -> None:
    """Test that the system doesn't break if RDP_constants are negative"""
    pass


def test_cache_index_conversion() -> None:
    """Test that the function to convert RDP_constants to Cache Indices doesn't generate invalid outputs."""
    pass


def test_cache_expansion() -> None:
    """Test the cache operates reasonably when the index provided is greater than its size."""
    pass


def test_cache_gigantic_expansion() -> None:
    """If the cache needs to become a gigantic size, compute its values in a 1-off way."""
    # Note: Also test that PB for Data Subjects is accurate recomputed.
    pass


def test_cache_index_negative() -> None:
    """Test the cache throws an Exception of some kind when the indices are negative."""
    pass


def test_cache_correct_mapping() -> None:
    """Test that the cache maps RDP_constants to the proper indices."""
    pass


def test_cache_rounding_small_numbers() -> None:
    """Test the error in epsilon spend when rdp_constant is a float, and has to be
    turned into an integer for the index."""
    for cache_index in range(50_000)):
        assert cache[cache_index + 0.8] - cache[cache_index] <= 0.2

    cache_indices = np.arange(50_000) + 0.8
    int_cache_indices = cache_indices.astype(np.int64)
    eps_spends = cache.take(cache_indices)
    eps_spends_
    pass


def test_cache_rounding_large_numbers() -> None:
    """Test the error in epsilon spend when rdp_constant is a float, and has to be
    turned into an integer for the index."""
    pass


def test_correct_epsilon_spends() -> None:
    """Test that for a given value, the epsilon is as expected."""
    pass


def test_multiple_publish_single_value() -> None:
    """Test that when publish filters values, and we're only publishing a single value,
    calling .all() doesn't result in system failure"""
    pass
