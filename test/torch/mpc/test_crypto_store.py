import pytest

from syft.exceptions import EmptyCryptoPrimitiveStoreError


def test_primitives_usage(workers):
    me, alice, bob, crypto_provider = (
        workers["me"],
        workers["alice"],
        workers["bob"],
        workers["james"],
    )
    me.crypto_store.provide_primitives(["fss_eq"], [alice, bob], n_instances=6)
    _ = alice.crypto_store.get_keys("fss_eq", 2, remove=False)

    assert len(alice.crypto_store.fss_eq[0]) == 6

    keys = alice.crypto_store.get_keys("fss_eq", 4, remove=True)

    assert len(keys[0]) == 4
    assert len(alice.crypto_store.fss_eq[0]) == 2

    with pytest.raises(EmptyCryptoPrimitiveStoreError):
        _ = alice.crypto_store.get_keys("fss_eq", 4, remove=True)
