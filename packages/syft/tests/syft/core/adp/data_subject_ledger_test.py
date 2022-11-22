# third party
import numpy as np

# syft absolute
import syft as sy
from syft.core.adp.data_subject_ledger import DataSubjectLedger
from syft.core.adp.data_subject_ledger import convert_constants_to_indices


def test_data_subject_ledger_serde() -> None:
    rdp_constants = np.array([1, 2, 3], np.float64)

    dsl = DataSubjectLedger()
    dsl._rdp_constants = rdp_constants

    ser = sy.serialize(dsl, to_bytes=True)
    de = sy.deserialize(ser, from_bytes=True)

    assert de == dsl
    assert all(de._rdp_constants == rdp_constants)


def test_cache_indexing_correctness() -> None:
    dsl = DataSubjectLedger()
    for i in [0.0001, 1, 50, 51, 100, 200_000]:
        # This is the theoretical (correct) epsilon value
        theoretical_epsilon = dsl._get_optimal_alpha_for_constant(i)[1]

        cache_value = dsl._cache_constant2epsilon[
            convert_constants_to_indices(np.array([i]))
        ][0]
        eps = dsl._get_epsilon_spend(np.array([i]))[0]
        assert round(theoretical_epsilon, 8) == round(cache_value, 8) == round(eps, 8)
