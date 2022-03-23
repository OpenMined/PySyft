# third party
from nacl.encoding import HexEncoder
from nacl.signing import SigningKey
import numpy as np
import pytest

# syft absolute
import syft as sy
from syft.core.adp.adversarial_accountant import AdversarialAccountant
from syft.core.adp.entity import Entity
from syft.core.adp.ledger_store import DictLedgerStore
from syft.core.adp.scalar.phi_scalar import PhiScalar
from syft.core.node.common.node_manager.dict_store import DictStore
from syft.core.tensor.tensor import Tensor


def test_autodp_phiscalar_can_publish() -> None:
    domain = sy.Domain("Alice", store_type=DictStore, ledger_store=DictLedgerStore)

    def encode_key(key: SigningKey) -> str:
        return key.encode(encoder=HexEncoder).decode("utf-8")

    key = SigningKey.generate()
    domain.users.signup(
        name="Bob",
        email="bob@gmail.com",
        password="letmein",
        budget=100,
        role=1,
        private_key=encode_key(key),
        verify_key=encode_key(key.verify_key),
    )

    x = PhiScalar(0, 0.01, 1)
    y = PhiScalar(0, 0.02, 1)
    z = PhiScalar(0, 0.02, 1)

    o = x * x + y * y + z
    z = o * o * o

    z.publish(acc=domain.acc, sigma=0.2, user_key=key.verify_key)

    domain.acc.print_ledger()
    assert len(domain.acc.entities) == 3


def test_autodp_phiscalar_cannot_publish() -> None:
    domain = sy.Domain("Alice", store_type=DictStore, ledger_store=DictLedgerStore)

    def encode_key(key: SigningKey) -> str:
        return key.encode(encoder=HexEncoder).decode("utf-8")

    key = SigningKey.generate()
    domain.users.signup(
        name="Bob",
        email="bob@gmail.com",
        password="letmein",
        budget=0.0001,
        role=1,
        private_key=encode_key(key),
        verify_key=encode_key(key.verify_key),
    )

    x = PhiScalar(0, 0.01, 1)
    y = PhiScalar(0, 0.02, 1)
    z = PhiScalar(0, 0.02, 1)

    o = x * x + y * y + z
    z = o * o * o

    # domain.acc.max_budget = 0.0001
    z.publish(acc=domain.acc, sigma=0.2, user_key=key.verify_key)

    domain.acc.print_ledger()
    assert len(domain.acc.entities) == 0


# def test_autodp_phiscalar_substitute_publish(domain: sy.Domain) -> None:
#     def encode_key(key: SigningKey) -> str:
#         return key.encode(encoder=HexEncoder).decode("utf-8")
#
#     # create user with matching client key
#     client = domain.get_root_client()
#
#     key = client.signing_key
#     domain.users.signup(
#         name="Bob",
#         email="bob@gmail.com",
#         password="letmein",
#         budget=10,
#         role=1,
#         private_key=encode_key(key),
#         verify_key=encode_key(key.verify_key),
#     )
#
#     # create data
#
#     n = 10
#
#     # Load some sample data
#     data_batch = np.array([13] * n)
#
#     entities = list()
#     for i in range(n):
#         entities.append(Entity(name=str(i)))
#
#     # Upload a private dataset to the Domain object, as the root owner
#     data = sy.Tensor(data_batch).private(0, 20, entities=entities).tag("data")
#
#     # send data
#     data_ptr = data.send(client)
#
#     # do calculation
#     s_ptr = data_ptr.sum(0) / 10
#
#     out = s_ptr.publish(sigma=1)
#     result = out.get()
#
#     result_float = result.child.item() - 13
#     assert result_float < 5  # less than 5 inaccurate
#
#     domain.acc.print_ledger()
#     assert len(domain.acc.entities) == 10


@pytest.mark.xfail
def test_autodp_train_linear_model() -> None:
    # entities = [
    #     Entity(name="Tudor"),
    #     Entity(name="Madhava"),
    #     Entity(name="Kritika"),
    #     Entity(name="George"),
    # ]

    entity = Entity(name="Trask")

    x = (
        Tensor(np.array([[1, 1], [1, 0], [0, 1], [0, 0]]))
        .private(min_val=0, max_val=1, entity=entity)
        .autograd(requires_grad=True)
    )
    y = (
        Tensor(np.array([[1], [1], [0], [0]]))
        .private(min_val=0, max_val=1, entity=entity)
        .autograd(requires_grad=True)
    )

    weights = Tensor(np.random.uniform(size=(2, 1))).autograd(requires_grad=True)
    # print("type _weights ", type(_weights))
    # weights = _weights + 0

    acc = AdversarialAccountant(max_budget=7)

    for _ in range(1):
        batch_loss = 0

        pred = x.dot(weights)
        loss = np.mean(np.square(y - pred))
        loss.backward()

        weight_grad = weights.grad * 0.5
        weight_grad = weight_grad.publish(acc=acc, sigma=0.1)

        weights = weights - weight_grad
        batch_loss += loss.value

        acc.print_ledger()

    assert len(acc.entities) == 4
    assert batch_loss > 0
    assert True is False


@pytest.mark.xfail
def test_adding_scalars() -> None:
    weights = Tensor(np.random.uniform(size=(2, 1))).autograd(requires_grad=True)
    weights = weights + 0  # 0 has no requires_grad so this fails in the add op
    weights.backward()
