# third party
import numpy as np
import pandas as pd

# syft absolute
from syft.service.action.action_object import ActionObject
from syft.service.action.action_object import convert_to_pointers
from syft.service.action.action_service import CollectionSearchContext
from syft.service.action.action_service import depointerize_collection_elements
from syft.service.action.plan import planify
from syft.service.context import AuthedServiceContext
from syft.service.user.user_roles import ServiceRole
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
    res_root = root_domain_client.api.services.action.get(flat_ptr.id)
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
        root_domain_client.api.services.action.get(res_ptr.id)
        == np.array([[3, 3, 3], [3, 3, 3]]).flatten().prod()
    )

    # guest can request result
    res_ptr.request(guest_client)

    # root approves result
    root_domain_client.api.services.request[0].approve_with_client(root_domain_client)

    assert res_ptr.get() == 729


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

    assert root_domain_client.api.services.action.get(res_ptr.id) == 18


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

    assert all(
        root_domain_client.api.services.action.get(res_ptr.id)
        == np.array([2, 3, 4, 5, 6, 7])
    )


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

    res = root_domain_client.api.services.action.get(obj_pointer.id)

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
    assert root_domain_client.api.services.action.get(size_pointer.id) == 6


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
        root_domain_client.api.services.action.get(flat_pointer.id)
        == np.array([1, 2, 3, 4, 5, 6])
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


def test_depointerize_collection_elements(worker):
    client = worker.root_client
    guest_client = worker.guest_client
    input_obj = TwinObject(
        private_obj=pd.DataFrame({"A": [1, 2, 3]}),
        mock_obj=pd.DataFrame({"A": [1, 2, 3]}),
    )
    input_obj2 = TwinObject(
        private_obj=pd.DataFrame({"A": [4, 5, 6]}),
        mock_obj=pd.DataFrame({"A": [4, 5, 6]}),
    )
    input_obj3 = ActionObject.from_obj(
        pd.DataFrame({"A": [4, 5, 6]}),
    )

    ptr1 = client.api.services.action.set(input_obj)
    ptr2 = client.api.services.action.set(input_obj2)
    ptr3 = client.api.services.action.set(input_obj3)
    c1, c2, c3 = [
        CollectionSearchContext(
            context=AuthedServiceContext(
                credentials=guest_client.credentials.verify_key,
                role=ServiceRole.GUEST,
                node=worker,
            ),
            get_mock=False,
            action_service=worker.get_service("actionservice"),
        )
        for i in range(3)
    ]
    assert depointerize_collection_elements([[[ptr1]], ptr2, "a"], c1)[
        1
    ].collection_contains_non_twins
    assert depointerize_collection_elements([[[ptr1]], ptr2, ptr3], c2)[
        1
    ].collection_contains_non_twins
    assert not depointerize_collection_elements([[[ptr1]], ptr2], c3)[
        1
    ].collection_contains_non_twins

    # tests whether nested things are not pointerized
    args, _ = convert_to_pointers(
        client.api, client.api.node_uid, args=([ptr1, ptr2],), kwargs=dict()
    )
    arg = args[0]
    first_element = arg.syft_action_data[0]
    # the first element should be a dataframe, and not a pointer!
    assert type(first_element) == pd.DataFrame


def test_call_function_with_collection_of_pointers(worker):
    client = worker.root_client
    input_obj = TwinObject(
        private_obj=pd.DataFrame({"A": [1, 2, 3]}),
        mock_obj=pd.DataFrame({"A": [1, 2, 3]}),
    )
    input_obj2 = TwinObject(
        private_obj=pd.DataFrame({"A": [4, 5, 6]}),
        mock_obj=pd.DataFrame({"A": [4, 5, 6]}),
    )

    ptr1 = client.api.services.action.set(input_obj)
    ptr2 = client.api.services.action.set(input_obj2)

    res_ptr = client.api.lib.pandas.concat([ptr1, ptr2])
    res = res_ptr.get_from(client)
    assert type(res) == pd.DataFrame and len(res) == 6


def test_setitem(worker):
    client = worker.root_client
    input_obj = TwinObject(
        private_obj=np.array([[3, 3, 3], [3, 3, 3]]),
        mock_obj=np.array([[1, 1, 1], [1, 1, 1]]),
    )

    input_ptr = client.api.services.action.set(input_obj)
    input_ptr[0, 0] = 5

    assert input_ptr.get_from(client)[0, 0] == 5


def test_setitem_with_pointer_as_index(worker):
    client = worker.root_client
    input_obj2 = TwinObject(
        private_obj=ActionObject.from_obj(0), mock_obj=ActionObject.from_obj(0)
    )
    input_obj = TwinObject(
        private_obj=np.array([[3, 3, 3], [3, 3, 3]]),
        mock_obj=np.array([[1, 1, 1], [1, 1, 1]]),
    )

    input_ptr = client.api.services.action.set(input_obj)
    input_ptr2 = client.api.services.action.set(input_obj2)
    input_ptr[0, input_ptr2] = 5
    input_ptr
    res = input_ptr.get_from(client)
    assert res[0, 0] == 5


def test_setitem_with_pointer_as_value(worker):
    client = worker.root_client
    input_obj = TwinObject(
        private_obj=np.array([[3, 3, 3], [3, 3, 3]]),
        mock_obj=np.array([[1, 1, 1], [1, 1, 1]]),
    )

    input_obj2 = TwinObject(
        private_obj=ActionObject.from_obj(5), mock_obj=ActionObject.from_obj(2)
    )

    input_ptr = client.api.services.action.set(input_obj)

    val = client.api.services.action.set(input_obj2)

    input_ptr[0, 0] = val
    assert input_ptr.get_from(client)[0, 0] == 5

    assert all(
        root_domain_client.api.services.action.get(first_row_pointer.id)
        == np.array([1, 2, 3])
    )
