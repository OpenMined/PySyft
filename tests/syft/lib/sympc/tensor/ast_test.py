# stdlib
import operator

# third party
import pytest
from sympc.tensor import AdditiveSharingTensor
import torch

# syft absolute
import syft as sy
from syft.lib.sympc.session import SySession

TEST_VALUES = [
    (4, -5),
    (torch.tensor([42, -32, 12]), 20),
    (25, torch.tensor([32, 12, -5])),
    (torch.tensor([15, 2353, 23, -50]), torch.tensor([123, 43, 23, -5])),
    (4.512312, torch.tensor([123.123, 5123.321, 123.32])),
]


def test_ast_exception_share() -> None:
    alice = sy.VirtualMachine(name="alice")
    bob = sy.VirtualMachine(name="bob")

    alice_client = alice.get_client()
    bob_client = bob.get_client()

    session = SySession(parties=[alice_client, bob_client])

    with pytest.raises(ValueError):
        AdditiveSharingTensor(secret=42, session=session)


@pytest.mark.parametrize("private", [True, False])
@pytest.mark.parametrize("operation", ["add", "sub", "mul"])
def test_ast_operation(private: bool, operation: str) -> None:
    alice = sy.VirtualMachine(name="alice")
    bob = sy.VirtualMachine(name="bob")

    alice_client = alice.get_client()
    bob_client = bob.get_client()

    session = SySession(parties=[alice_client, bob_client])
    session.setup_mpc()

    for x_secret, y_secret in TEST_VALUES:
        x = AdditiveSharingTensor(secret=x_secret, session=session)

        if private:
            y = AdditiveSharingTensor(secret=y_secret, session=session)
        else:
            y = y_secret

        op = getattr(operator, operation)
        res = op(x, y)
        res_expected = op(x_secret, y_secret)

        if not isinstance(res_expected, torch.Tensor):
            res_expected = torch.tensor([res_expected])

        res_expected = res_expected.float()

        assert torch.allclose(
            res.reconstruct(), res_expected
        ), f"Fail for {x_secret} and {y_secret}"
