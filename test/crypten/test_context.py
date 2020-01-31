import pytest
from syft.frameworks.crypten.context import run_multiworkers


def test_context(workers):
    # self, alice and bob
    n_workers = 3
    alice = workers["alice"]
    bob = workers["bob"]

    @run_multiworkers([alice, bob], master_addr="127.0.0.1")
    def test_three_parties():
        pass  # pragma: no cover

    return_values = test_three_parties()
    # A toy function is ran at each party, and they should all decrypt
    # a tensor with value [90., 100.]
    expected_value = [90.0, 100.0]
    for rank in range(n_workers):
        assert (
            return_values[rank] == expected_value
        ), "Crypten party with rank {} don't match expected value {} != {}".format(
            rank, return_values[rank], expected_value
        )
