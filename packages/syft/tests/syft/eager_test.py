# third party
import numpy as np

# syft absolute
from syft.service.action.action_object import ActionObject
from syft.service.action.plan import planify
from syft.types.twin_object import TwinObject


def test_eager_permissions(worker, guest_client):
    root_domain_client = worker.root_client
    input_obj = TwinObject(
        private_obj=np.array([[3, 3, 3], [3, 3, 3]]),
        mock_obj=np.array([[1, 1, 1], [1, 1, 1]]),
    )

    input_ptr = root_domain_client.api.services.action.set(input_obj)

    pointer = guest_client.api.services.action.get_pointer(input_ptr.id)

    input_ptr = root_domain_client.api.services.action.set(input_obj)

    pointer = guest_client.api.services.action.get_pointer(input_ptr.id)

    flat_ptr = pointer.flatten()

    res_guest = guest_client.api.services.action.get(flat_ptr.id)
    assert not isinstance(res_guest, ActionObject)
    res_root = flat_ptr.get_from(root_domain_client)
    assert all(res_root == [3, 3, 3, 3, 3, 3])


def test_plan(worker, guest_client):
    root_domain_client = worker.root_client
    guest_client = worker.guest_client

    @planify
    def my_plan(x=np.array([[2, 2, 2], [2, 2, 2]])):  # noqa: B008
        y = x.flatten()
        z = y.prod()
        return z

    plan_ptr = my_plan.send(guest_client)
    input_obj = TwinObject(
        private_obj=np.array([[3, 3, 3], [3, 3, 3]]),
        mock_obj=np.array([[1, 1, 1], [1, 1, 1]]),
    )

    input_obj = root_domain_client.api.services.action.set(input_obj)
    pointer = guest_client.api.services.action.get_pointer(input_obj.id)
    res_ptr = plan_ptr(x=pointer)

    # guest cannot access result
    assert not isinstance(
        guest_client.api.services.action.get(res_ptr.id), ActionObject
    )

    # root can access result
    assert (
        res_ptr.get_from(root_domain_client)
        == np.array([[3, 3, 3], [3, 3, 3]]).flatten().prod()
    )

    # guest can request result
    res_ptr.request(guest_client)

    # root approves result
    root_domain_client.api.services.request[0].approve_with_client(root_domain_client)

    assert res_ptr.get_from(guest_client) == 729


def test_plan_with_function_call(worker, guest_client):
    root_domain_client = worker.root_client
    guest_client = worker.guest_client

    @planify
    def my_plan(x=np.array([[2, 2, 2], [2, 2, 2]])):  # noqa: B008
        y = x.flatten()
        w = guest_client.api.lib.numpy.sum(y)
        return w

    plan_ptr = my_plan.send(guest_client)
    input_obj = TwinObject(
        private_obj=np.array([[3, 3, 3], [3, 3, 3]]),
        mock_obj=np.array([[1, 1, 1], [1, 1, 1]]),
    )

    input_obj = root_domain_client.api.services.action.set(input_obj)
    pointer = guest_client.api.services.action.get_pointer(input_obj.id)
    res_ptr = plan_ptr(x=pointer)

    assert res_ptr.get_from(root_domain_client) == 18


def test_plan_with_object_instantiation(worker, guest_client):
    @planify
    def my_plan(x=np.array([1, 2, 3, 4, 5, 6])):  # noqa: B008
        return x + 1

    root_domain_client = worker.root_client

    plan_ptr = my_plan.send(guest_client)

    input_obj = TwinObject(
        private_obj=np.array([1, 2, 3, 4, 5, 6]), mock_obj=np.array([1, 1, 1, 1, 1, 1])
    )

    _id = root_domain_client.api.services.action.set(input_obj).id
    pointer = guest_client.api.services.action.get_pointer(_id)

    res_ptr = plan_ptr(x=pointer)

    assert all(res_ptr.get_from(root_domain_client) == np.array([2, 3, 4, 5, 6, 7]))


def test_setattribute(worker, guest_client):
    root_domain_client = worker.root_client

    private_data, mock_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), np.array(
        [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
    )

    obj = TwinObject(private_obj=private_data, mock_obj=mock_data)

    assert private_data.dtype != np.int32

    obj_pointer = root_domain_client.api.services.action.set(obj)
    obj_pointer = guest_client.api.services.action.get_pointer(obj_pointer.id)

    original_id = obj_pointer.id

    # note that this messes up the data and the shape of the array
    obj_pointer.dtype = np.int32

    # local object is updated
    assert obj_pointer.id.id in worker.action_store.data
    assert obj_pointer.id != original_id

    res = obj_pointer.get_from(root_domain_client)

    # check if updated
    assert res.dtype == np.int32

    private_data.dtype = np.int32
    mock_data.dtype = np.int32

    assert (res == private_data).all()
    assert (obj_pointer.syft_action_data == mock_data).all()
    assert not (obj_pointer.syft_action_data == private_data).all()


def test_getattribute(worker, guest_client):
    root_domain_client = worker.root_client
    obj = TwinObject(
        private_obj=np.array([[1, 2, 3], [4, 5, 6]]),
        mock_obj=np.array([[1, 1, 1], [1, 1, 1]]),
    )

    obj_pointer = root_domain_client.api.services.action.set(obj)
    obj_pointer = guest_client.api.services.action.get_pointer(obj_pointer.id)
    size_pointer = obj_pointer.size

    # check result
    assert size_pointer.id.id in worker.action_store.data
    assert size_pointer.get_from(root_domain_client) == 6


def test_eager_method(worker, guest_client):
    root_domain_client = worker.root_client

    obj = TwinObject(
        private_obj=np.array([[1, 2, 3], [4, 5, 6]]),
        mock_obj=np.array([[1, 1, 1], [1, 1, 1]]),
    )

    obj_pointer = root_domain_client.api.services.action.set(obj)
    obj_pointer = guest_client.api.services.action.get_pointer(obj_pointer.id)

    flat_pointer = obj_pointer.flatten()

    assert flat_pointer.id.id in worker.action_store.data
    # check result
    assert all(
        flat_pointer.get_from(root_domain_client) == np.array([1, 2, 3, 4, 5, 6])
    )


def test_eager_dunder_method(worker, guest_client):
    root_domain_client = worker.root_client

    obj = TwinObject(
        private_obj=np.array([[1, 2, 3], [4, 5, 6]]),
        mock_obj=np.array([[1, 1, 1], [1, 1, 1]]),
    )

    obj_pointer = root_domain_client.api.services.action.set(obj)
    obj_pointer = guest_client.api.services.action.get_pointer(obj_pointer.id)

    first_row_pointer = obj_pointer[0]

    assert first_row_pointer.id.id in worker.action_store.data
    # check result
    assert all(first_row_pointer.get_from(root_domain_client) == np.array([1, 2, 3]))
