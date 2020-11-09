import pytest
import torch as th
import numpy as np

from syft.frameworks.torch.mpc.fss import DPF, DIF, n

# Note: the FSS class is tested in the Sycret package
# (Python wrapper level and Rust level)


@pytest.mark.parametrize("op", ["eq", "le"])
def test_using_crypto_store(workers, op):
    alice, bob, me = workers["alice"], workers["bob"], workers["me"]
    class_ = {"eq": DPF, "le": DIF}[op]
    th_op = {"eq": th.eq, "le": th.le}[op]
    gather_op = {"eq": "__add__", "le": "__add__"}[op]
    primitive = {"eq": "fss_eq", "le": "fss_comp"}[op]

    me.crypto_store.provide_primitives(primitive, [alice, bob], n_instances=6)
    keys_a = alice.crypto_store.get_keys(primitive, 3, remove=True)
    keys_b = bob.crypto_store.get_keys(primitive, 3, remove=True)

    print(f"Got keys {keys_a} {keys_b}")

    alpha_a = np.frombuffer(np.ascontiguousarray(keys_a[1:, 0:4]), dtype=np.uint32).astype(
        np.uint64
    )
    alpha_b = np.frombuffer(np.ascontiguousarray(keys_b[1:, 0:4]), dtype=np.uint32).astype(
        np.uint64
    )

    print(f"And alpha {alpha_a + alpha_b}")

    x = th.IntTensor([0, 2, -2])
    np_x = x.numpy()
    x_masked = (np_x + alpha_a + alpha_b).astype(np.uint64)

    print("Time to eval!")
    y0 = class_.eval(0, x_masked, keys_a)
    print(f"Evaluating the rest")
    y1 = class_.eval(1, x_masked, keys_b)

    np_result = getattr(y0, gather_op)(y1)
    print(f"np result {np_result}")
    th_result = th.native_tensor(np_result)
    print(f"th result {th_result}")
    print(f"Should be {th_op(x, 0)}")

    assert (th_result == th_op(x, 0)).all()
