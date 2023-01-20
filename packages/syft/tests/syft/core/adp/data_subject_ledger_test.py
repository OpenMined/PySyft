# third party
import numpy as np

# syft absolute
import syft as sy
from syft.core.adp.data_subject_ledger import DataSubjectLedger
from syft.core.adp.data_subject_ledger import convert_constants_to_indices
from syft.core.adp.ledger_store import DictLedgerStore


def test_data_subject_ledger_serde() -> None:
    rdp_constants = {"natsu": np.array(0.214)}

    dsl = DataSubjectLedger()
    dsl._rdp_constants = rdp_constants

    ser = sy.serialize(dsl, to_bytes=True)
    de = sy.deserialize(ser, from_bytes=True)

    assert de == dsl
    assert de._rdp_constants == rdp_constants


def test_cache_bypass() -> None:
    """
    Check that the cache is bypassed for gigantic RDP constants
    """
    gigantic_rdp_constant = np.array([1e8, 2e8])
    dsl = DataSubjectLedger()
    eps = dsl._get_epsilon_spend(gigantic_rdp_constant)
    direct_eps = np.array(
        [dsl._get_optimal_alpha_for_constant(i)[1] for i in gigantic_rdp_constant]
    )
    assert (
        eps > dsl._cache_constant2epsilon[-1]
    ).all(), "It seems the cache has been modified?"
    assert (direct_eps == eps).all(), "It seems epsilon was incorrectly calculated."


def test_cache_indexing_correctness() -> None:
    dsl = DataSubjectLedger()
    for i in [0.0001, 1, 50, 51, 100, 200_000]:
        # This is the theoretical (correct) epsilon value
        theoretical_epsilon = dsl._get_optimal_alpha_for_constant(i)[1]

        cache_value = dsl._cache_constant2epsilon[
            convert_constants_to_indices(np.array([i]))
        ][0]
        eps = dsl._get_epsilon_spend(np.array([i]))[0]
        assert round(theoretical_epsilon, 7) == round(cache_value, 7) == round(eps, 7)


def test_cache() -> None:
    """Ensure the most up to date RDP-to-epsilon cache is being used."""
    ledger_store = DictLedgerStore()
    user_key = b"1322"
    ledger = DataSubjectLedger.get_or_create(store=ledger_store, user_key=user_key)

    assert (
        ledger._cache_constant2epsilon[0] == 0.05372712063485988
    ), "The first value in the cache is incorrect"
    assert (
        ledger._cache_constant2epsilon[1] == 0.07773597369831031
    ), "Has the DP cache been changed?"

    rdp_700k = convert_constants_to_indices(np.array([700_000]))
    assert (
        ledger._cache_constant2epsilon.take(rdp_700k)[0] == 706213.1816144075
    ), "Has the DP cache been changed?"
    rdp_50 = convert_constants_to_indices(np.array([50]))
    assert (
        ledger._cache_constant2epsilon.take(rdp_50)[0] == 100.68990516105825
    ), "Has the DP cache been changed?"
    assert (
        len(ledger._cache_constant2epsilon) >= 1_200_000
    ), "Has the cache been changed?"


def test_cache_invalidation() -> None:
    """The cache was built assuming a particular value of delta (1e-6), and shouldn't contain any zero values."""
    ledger_store = DictLedgerStore()
    user_key = b"1483"
    ledger = DataSubjectLedger.get_or_create(store=ledger_store, user_key=user_key)

    assert ledger.delta == 1e-6, "The cache has been changed or is invalid."
    assert (
        ledger._cache_constant2epsilon.all()
    ), "There is a zero epsilon value in the cache- major security flaw."
    assert (
        ledger._cache_constant2epsilon > 0
    ).all(), "Negative epsilon value in the cache- major security flaw."


def test_ledger() -> None:
    """Test that the ledgers are created properly and updated properly"""
    ledger_store = DictLedgerStore()
    user_key = b"9738"
    ledger = DataSubjectLedger.get_or_create(store=ledger_store, user_key=user_key)

    assert ledger._rdp_constants == {}, "Ledger was not empty upon initialization"
    query1_constants = {"Coco": 47, "Eduardo": 99, "Bloo": 104, "Madame Foster": 7879}
    ledger.update_rdp_constants(query1_constants)
    for data_subject in query1_constants:
        assert ledger._rdp_constants[data_subject] == query1_constants[data_subject]

    query2_constants = {
        "Coco": 9001,
        "Eduardo": 9365,
        "Bloo": 1896,
        "Madame Foster": 12,
    }
    ledger.update_rdp_constants(query2_constants)
    for data_subject in query2_constants:
        assert (
            ledger._rdp_constants[data_subject]
            == query2_constants[data_subject] + query1_constants[data_subject]
        )
