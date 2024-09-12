# third party
import numpy as np
import pytest

# syft absolute
from syft.service.action.plan import planify
from syft.types.errors import SyftException
from syft.types.twin_object import TwinObject

# relative
from ..utils.custom_markers import currently_fail_on_python_3_12


@pytest.mark.skip(reason="Disabled until we bring back eager execution")
def test_eager_permissions(worker, guest_client):
    root_datasite_client = worker.root_client

    assert root_datasite_client.settings.enable_eager_execution(enable=True)

    guest_client = worker.guest_client

    input_obj = TwinObject(
        private_obj=np.array([[3, 3, 3], [3, 3, 3]]),
        mock_obj=np.array([[1, 1, 1], [1, 1, 1]]),
    )

    input_ptr = input_obj.send(root_datasite_client)

    pointer = guest_client.api.services.action.get_pointer(input_ptr.id)

    input_ptr = input_obj.send(root_datasite_client)

    pointer = guest_client.api.services.action.get_pointer(input_ptr.id)

    flat_ptr = pointer.flatten()

    with pytest.raises(SyftException) as exc:
        guest_client.api.services.action.get(flat_ptr.id)

    # TODO: Improve this error msg
    assert exc.type == SyftException
    assert "denied" in str(exc.value)

    res_root = root_datasite_client.api.services.action.get(flat_ptr.id)
    assert all(res_root == [3, 3, 3, 3, 3, 3])


@pytest.mark.skip(reason="Disabled until we bring back eager execution")
def test_plan(worker):
    root_datasite_client = worker.root_client

    assert root_datasite_client.settings.enable_eager_execution(enable=True)

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

    input_ptr = input_obj.send(root_datasite_client)

    pointer = guest_client.api.services.action.get_pointer(input_ptr.id)
    res_ptr = plan_ptr(x=pointer)

    # guest cannot access result
    with pytest.raises(SyftException):
        guest_client.api.services.action.get(res_ptr.id)

    # root can access result
    assert (
        root_datasite_client.api.services.action.get(res_ptr.id)
        == np.array([[3, 3, 3], [3, 3, 3]]).flatten().prod()
    )

    # guest can request result
    res_ptr.request(guest_client)

    # root approves result
    root_datasite_client.api.services.request[-1].approve_with_client(
        root_datasite_client
    )

    assert res_ptr.get_from(guest_client) == 729


@pytest.mark.skip(reason="Disabled until we bring back eager execution")
@currently_fail_on_python_3_12(raises=AttributeError)
def test_plan_with_function_call(worker, guest_client):
    root_datasite_client = worker.root_client

    assert root_datasite_client.settings.enable_eager_execution(enable=True)

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

    input_obj = input_obj.send(root_datasite_client)
    pointer = guest_client.api.services.action.get_pointer(input_obj.id)
    res_ptr = plan_ptr(x=pointer)

    assert root_datasite_client.api.services.action.get(res_ptr.id) == 18


@pytest.mark.skip(reason="Disabled until we bring back eager execution")
def test_plan_with_object_instantiation(worker, guest_client):
    root_datasite_client = worker.root_client

    assert root_datasite_client.settings.enable_eager_execution(enable=True)

    guest_client = worker.guest_client

    @planify
    def my_plan(x=np.array([1, 2, 3, 4, 5, 6])):  # noqa: B008
        return x + 1

    plan_ptr = my_plan.send(guest_client)

    input_obj = TwinObject(
        private_obj=np.array([1, 2, 3, 4, 5, 6]), mock_obj=np.array([1, 1, 1, 1, 1, 1])
    )

    _id = input_obj.send(root_datasite_client).id
    pointer = guest_client.api.services.action.get_pointer(_id)

    res_ptr = plan_ptr(x=pointer)

    assert all(
        root_datasite_client.api.services.action.get(res_ptr.id).syft_action_data
        == np.array([2, 3, 4, 5, 6, 7])
    )


@pytest.mark.skip(reason="Disabled until we bring back eager execution")
def test_setattribute(worker, guest_client):
    root_datasite_client = worker.root_client

    assert root_datasite_client.settings.enable_eager_execution(enable=True)

    guest_client = worker.guest_client

    private_data, mock_data = (
        np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]),
    )

    obj = TwinObject(private_obj=private_data, mock_obj=mock_data)

    assert private_data.dtype != np.int32

    obj_pointer = obj.send(root_datasite_client)
    obj_pointer = guest_client.api.services.action.get_pointer(obj_pointer.id)

    original_id = obj_pointer.id

    # note that this messes up the data and the shape of the array
    obj_pointer.dtype = np.int32

    # local object is updated
    assert obj_pointer.id.id in worker.action_store._data
    assert obj_pointer.id != original_id

    res = root_datasite_client.api.services.action.get(obj_pointer.id)

    # check if updated
    assert res.dtype == np.int32

    private_data.dtype = np.int32
    mock_data.dtype = np.int32

    assert (res == private_data).all()
    assert (obj_pointer.syft_action_data == mock_data).all()
    assert not (obj_pointer.syft_action_data == private_data).all()


@pytest.mark.skip(reason="Disabled until we bring back eager execution")
def test_getattribute(worker, guest_client):
    root_datasite_client = worker.root_client
    assert root_datasite_client.settings.enable_eager_execution(enable=True)
    guest_client = worker.guest_client

    obj = TwinObject(
        private_obj=np.array([[1, 2, 3], [4, 5, 6]]),
        mock_obj=np.array([[1, 1, 1], [1, 1, 1]]),
    )

    obj_pointer = obj.send(root_datasite_client)
    obj_pointer = guest_client.api.services.action.get_pointer(obj_pointer.id)
    size_pointer = obj_pointer.size

    # check result
    assert size_pointer.id.id in worker.action_store._data
    assert root_datasite_client.api.services.action.get(size_pointer.id) == 6


@pytest.mark.skip(reason="Disabled until we bring back eager execution")
def test_eager_method(worker, guest_client):
    root_datasite_client = worker.root_client
    assert root_datasite_client.settings.enable_eager_execution(enable=True)
    guest_client = worker.guest_client

    obj = TwinObject(
        private_obj=np.array([[1, 2, 3], [4, 5, 6]]),
        mock_obj=np.array([[1, 1, 1], [1, 1, 1]]),
    )

    obj_pointer = obj.send(root_datasite_client)
    obj_pointer = guest_client.api.services.action.get_pointer(obj_pointer.id)

    flat_pointer = obj_pointer.flatten()

    assert flat_pointer.id.id in worker.action_store._data
    # check result
    assert all(
        root_datasite_client.api.services.action.get(flat_pointer.id)
        == np.array([1, 2, 3, 4, 5, 6])
    )


@pytest.mark.skip(reason="Disabled until we bring back eager execution")
def test_eager_dunder_method(worker, guest_client):
    root_datasite_client = worker.root_client
    assert root_datasite_client.settings.enable_eager_execution(enable=True)
    guest_client = worker.guest_client

    obj = TwinObject(
        private_obj=np.array([[1, 2, 3], [4, 5, 6]]),
        mock_obj=np.array([[1, 1, 1], [1, 1, 1]]),
    )

    obj_pointer = obj.send(root_datasite_client)
    obj_pointer = guest_client.api.services.action.get_pointer(obj_pointer.id)

    first_row_pointer = obj_pointer[0]

    assert first_row_pointer.id.id in worker.action_store._data
    # check result
    assert all(
        root_datasite_client.api.services.action.get(first_row_pointer.id)
        == np.array([1, 2, 3])
    )
