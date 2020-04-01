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
    _ = alice.crypto_store.get_keys("eq", 2, remove=False)

    keys = alice.crypto_store.get_keys("eq", 4, remove=True)

    assert len(keys[0]) == 4

    with pytest.raises(EmptyCryptoPrimitiveStoreError):
        _ = alice.crypto_store.get_keys("eq", 4, remove=True)
