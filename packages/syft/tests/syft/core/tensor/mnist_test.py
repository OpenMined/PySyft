# third party
import numpy as np
import pytest
import torch as th

# syft absolute
import syft as sy
from syft import deserialize
from syft import serialize
from syft.core.adp.entity import Entity
from syft.core.tensor.tensor import Tensor


# MADHAVA: this needs fixing
@pytest.mark.xfail
def test_vs_torch() -> None:
    data_batch = np.random.rand(4, 28 * 28)
    label_batch = np.random.rand(4, 10)
    data = Tensor(data_batch).autograd(requires_grad=True)
    target = Tensor(label_batch).autograd(requires_grad=True)

    weights = np.random.rand(28 * 28, 10)

    pred = data.dot(weights)
    diff = target - pred
    pre_loss = np.square(diff)

    pre_loss.backward()
    pred.grad

    th_data = th.tensor(data.child.child, requires_grad=True)
    th_target = th.tensor(target.child.child, requires_grad=True)
    th_weights = th.tensor(weights, requires_grad=True)

    th_pred = th.tensor(th_data.mm(th_weights), requires_grad=True)
    th_diff = th_target - th_pred
    th_pre_loss = th_diff * th_diff

    th_pre_loss.sum().backward()
    th_pred.grad

    assert diff == th_diff.clone().detach().numpy()
    assert pre_loss == th_pre_loss.clone().detach().numpy()
    assert pred == th_pred.clone().detach().numpy()
    assert pred.grad == th_pred.grad.clone().detach().numpy()


# MADHAVA: this needs fixing
@pytest.mark.xfail
def test_train_mnist() -> None:
    data_batch = np.random.rand(4, 28 * 28)
    label_batch = np.random.rand(4, 10)

    bob = Entity(name="Bob")

    data = Tensor(data_batch).private(0.01, 1, entity=bob).autograd(requires_grad=True)
    target = (
        Tensor(label_batch).private(0.01, 1, entity=bob).autograd(requires_grad=True)
    )

    weights = Tensor(np.random.rand(28 * 28, 10)).autograd(requires_grad=True)

    for _ in range(10):
        pred = data.dot(weights)
        diff = target - pred
        pre_loss = np.square(diff)
        loss = np.mean(pre_loss)
        _ = -diff
        loss.backward()

        wdiff = weights.grad * 0.01
        weights = -wdiff + weights

    assert loss._data_child < 10.0


# MADHAVA: this needs fixing
@pytest.mark.xfail
def test_serde_tensors() -> None:
    data = np.random.rand(4, 10)
    bob = Entity(name="Bob")

    # Step 1: upload a private dataset as the root owner
    data = (
        Tensor(data)
        .private(0.01, 1, entity=bob)
        .autograd(requires_grad=True)
        .tag("data")
    )

    ser = serialize(data)

    de = deserialize(ser)

    comp_left = data
    comp_right = de

    assert type(comp_left) == type(comp_right)
    while hasattr(comp_left, "child"):
        comp_left = comp_left.child
        comp_right = comp_right.child
    assert not hasattr(comp_left, "child")
    assert not hasattr(comp_right, "child")
    assert type(comp_left) == type(comp_right)

    assert (de._data_child == data._data_child).all()


# MADHAVA: this needs fixing
@pytest.mark.xfail
def test_send_tensors(root_client: sy.VirtualMachineClient) -> None:
    data = np.random.rand(4, 10)
    bob = Entity(name="Bob")

    # Step 1: upload a private dataset as the root owner
    data = Tensor(data).private(0.01, 1, entity=bob).autograd(requires_grad=True)
    data_ptr = data.send(root_client, tags=["data"])
    assert len(root_client.store) == 1

    res = data_ptr.get()

    comp_left = data
    comp_right = res

    assert type(comp_left) == type(comp_right)
    while hasattr(comp_left, "child"):
        comp_left = comp_left.child
        comp_right = comp_right.child
    assert not hasattr(comp_left, "child")
    assert not hasattr(comp_right, "child")
    assert type(comp_left) == type(comp_right)

    assert (res._data_child == data._data_child).all()


# MADHAVA: this needs fixing
@pytest.mark.xfail
def test_basic_publish_entities_event() -> None:
    domain = sy.Domain("My Amazing Domain", max_budget=10)
    root_client = domain.get_root_client()

    data_batch = np.random.rand(4, 10)

    trask = Entity(name="Trask")
    kritika = Entity(name="Kritika")
    madhava = Entity(name="Madhava")
    tudor = Entity(name="Tudor")

    entities = [trask, kritika, madhava, tudor]

    # Step 1: upload a private dataset as the root owner
    data = (
        Tensor(data_batch)
        .private(0.01, 1, entities=entities)
        .autograd(requires_grad=True)
        .tag("data")
    )

    data.send(root_client)

    # # Step 2: user connects to domain with a new verify_key
    client = domain.get_client()

    data_ptr = client.store["data"]

    y_ptr = data_ptr.sum(0)

    # can't get the private data
    with pytest.raises(Exception):
        y_ptr.get()

    y_pub_ptr = y_ptr.publish(client=client, sigma=0.1)
    y_pub = y_pub_ptr.get()
    assert y_pub._data_child.ndim == 1
    assert len(y_pub._data_child) == 10


# MADHAVA: this needs fixing
@pytest.mark.xfail
def test_basic_publish_entity_event() -> None:
    domain = sy.Domain("My Amazing Domain", max_budget=10)
    root_client = domain.get_root_client()

    data_batch = np.random.rand(4, 10)

    thanos = Entity(name="Thanos")

    # Step 1: upload a private dataset as the root owner
    data = (
        Tensor(data_batch)
        .private(0.01, 1, entity=thanos)
        .autograd(requires_grad=True)
        .tag("data")
    )

    data.send(root_client)

    # # Step 2: user connects to domain with a new verify_key
    client = domain.get_client()

    data_ptr = client.store["data"]

    y_ptr = data_ptr.gamma  # no sum op so just convert
    y_ptr = y_ptr.sum(0)

    # can't get the private data
    with pytest.raises(Exception):
        y_ptr.get()

    y_pub_ptr = y_ptr.publish(client=client, sigma=0.1)
    y_pub = y_pub_ptr.get()
    assert y_pub._data_child.ndim == 1
    assert len(y_pub._data_child) == 10


# MADHAVA: this needs fixing
@pytest.mark.xfail
def test_train_publish_entities_event() -> None:
    domain = sy.Domain("My Amazing Domain", max_budget=10)
    root_client = domain.get_root_client()

    data_batch = np.random.rand(1, 3)
    label_batch = np.random.rand(1, 3)

    thanos = Entity(name="Thanos")

    # trask = Entity(name="Trask")
    # kritika = Entity(name="Kritika")
    # madhava = Entity(name="Madhava")
    # tudor = Entity(name="Tudor")

    # entities = [trask, kritika, madhava, tudor]

    # Step 1: upload a private dataset as the root owner
    data = (
        Tensor(data_batch)
        .private(0.01, 1, entity=thanos)
        .autograd(requires_grad=True)
        .tag("data")
    )

    target = (
        Tensor(label_batch).private(0.01, 1, entity=thanos).autograd(requires_grad=True)
    ).tag("target")

    data.send(root_client)
    target.send(root_client)

    # # Step 2: user connects to domain with a new verify_key
    client = domain.get_client()

    data_ptr = client.store["data"]
    target_ptr = client.store["target"]

    weights = Tensor(np.random.rand(3, 3)).autograd(requires_grad=True)
    weights_ptr = weights.send(client)

    for _ in range(1):
        pred = data_ptr.dot(weights_ptr)
        diff = target_ptr - pred

        # pre_loss = client.numpy.square(diff)  # cant use remote ufuncs yet
        pre_loss = diff * diff

        loss = client.numpy.mean(pre_loss)
        loss = loss.resolve_pointer_type()
        loss.backward()

        wdiff = weights_ptr.grad * 0.01
        weights_ptr = -wdiff + weights_ptr

    gamma_ptr = weights_ptr.gamma
    weights_pub_ptr = gamma_ptr.publish(client=client, sigma=0.1)
    updated_weights = weights_pub_ptr.get()

    assert updated_weights._data_child.ndim == weights._data_child.ndim
    assert not (updated_weights._data_child == weights._data_child).all()


# TODO: @Madhava Make work
# def test_simulated_publish_event() -> None:
#     domain = sy.Domain("My Amazing Domain", max_budget=10)
#     root_client = domain.get_root_client()
#
#     data_batch = np.random.rand(4, 28 * 28)
#     label_batch = np.random.rand(4, 10)
#
#     bob = Entity(name="Bob")
#
#     # Step 1: upload a private dataset as the root owner
#     data = Tensor(data_batch).private(0.01, 1, entity=bob).autograd(requires_grad=True).tag("data")
#     target = (
#         Tensor(label_batch).private(0.01, 1, entity=bob).autograd(requires_grad=True)
#     ).tag("target")
#
#     root_client.send(data)
#     root_client.send(target)
#
#     # Step 2: user connects to domain
#
#     # (this has a new verify key)
#     client = domain.get_client()
#
#     data = client.store['data']
#     target = client.store['target']
#
#     weights = Tensor(np.random.rand(28 * 28, 10)).autograd(requires_grad=True)
#     weights_ptr = weights.send(client)
#
#     for i in range(10):
#         pred = data.dot(weights_ptr)
#         diff = target - pred
#         pre_loss = np.square(diff)
#         loss = np.mean(pre_loss)
#         extraneous_thing = -diff
#         loss.backward()
#
#         wdiff = weights_ptr.grad * 0.01
#         weights_ptr = -wdiff + weights_ptr
#
#     # init_with_budget_remaining shoudl default to true
#     simulated_accountant = client.create_simulated_accountant(init_with_budget_remaining=True)
#
#     weights_ptr_downloadable = weights_ptr.publish(acc=simulated_accountant, sigma=0.1)
#
#     # return pointer to GammaScalar
#     ptr_to_budget = simulated_accountant.calculate_remaining_budget()
#
#     # publish GammaScalar like you would normally
#     downloadable_budget = ptr_to_budget.publish(acc=client.accountant, sigma=0.1)
#     print(downloadable_budget.get())
