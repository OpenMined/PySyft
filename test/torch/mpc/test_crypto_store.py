import pytest

from syft.exceptions import EmptyCryptoPrimitiveStoreError
from syft.frameworks.torch.mpc.przs import RING_SIZE, PRZS, gen_alpha_3of3, gen_alpha_2of3


def test_primitives_usage(workers):
    me, alice, bob, crypto_provider = (
        workers["me"],
        workers["alice"],
        workers["bob"],
        workers["james"],
    )
    me.crypto_store.provide_primitives("fss_eq", [alice, bob], n_instances=6)
    _ = alice.crypto_store.get_keys("fss_eq", 2, remove=False)

    assert len(alice.crypto_store.fss_eq[0]) == 6

    keys = alice.crypto_store.get_keys("fss_eq", 4, remove=True)

    assert len(keys[0]) == 4
    assert len(alice.crypto_store.fss_eq[0]) == 2

    with pytest.raises(EmptyCryptoPrimitiveStoreError):
        _ = alice.crypto_store.get_keys("fss_eq", 4, remove=True)


def test_przs_alpha_3of3(workers):
    alice, bob, james = (
        workers["alice"],
        workers["bob"],
        workers["james"],
    )

    workers_vals = [alice, bob, james]
    PRZS.setup(workers_vals)

    values = [gen_alpha_3of3(worker).get() for worker in workers_vals]

    sum_values = sum(values)

    assert sum_values.item() % RING_SIZE == 0


def test_przs_alpha_2of3(workers):
    alice, bob, james = (
        workers["alice"],
        workers["bob"],
        workers["james"],
    )

    workers_vals = [alice, bob, james]
    PRZS.setup(workers_vals)

    values = [gen_alpha_2of3(worker).get() for worker in workers_vals]

    """
        Worker i holds (alpha_i, and alpha_i-1)
        Here we do:
        ((alpha_i, alpha_i-1), (alpha_i+1, alpha_i))
    """
    paired_values = zip(values, [*values[1:], values[0]])
    for alphas_worker_cur, alphas_worker_next in paired_values:
        alpha_cur, _ = alphas_worker_cur
        _, alpha_prev = alphas_worker_next

        assert alpha_cur == alpha_prev
