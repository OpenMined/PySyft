# third party
import numpy as np

# syft absolute
from syft import ActionObject
from syft.client.api import SyftAPICall
from syft.service.action.action_object import Action
from syft.service.response import SyftError
from syft.types.uid import LineageID


def test_actionobject_method(worker):
    root_domain_client = worker.root_client
    action_store = worker.get_service("actionservice").store
    obj = ActionObject.from_obj("abc")
    pointer = root_domain_client.api.services.action.set(obj)
    assert len(action_store.data) == 1
    res = pointer.capitalize()
    assert len(action_store.data) == 2
    assert res[0] == "A"


def test_lib_function_action(worker):
    root_domain_client = worker.root_client
    numpy_client = root_domain_client.api.lib.numpy
    res = numpy_client.zeros_like([1, 2, 3])

    assert isinstance(res, ActionObject)
    assert all(res == np.array([0, 0, 0]))
    assert len(worker.get_service("actionservice").store.data) > 0


def test_call_lib_function_action2(worker):
    root_domain_client = worker.root_client
    assert root_domain_client.api.lib.numpy.add(1, 2) == 3


def test_lib_class_init_action(worker):
    root_domain_client = worker.root_client
    numpy_client = root_domain_client.api.lib.numpy
    res = numpy_client.float32(4.0)

    assert isinstance(res, ActionObject)
    assert res == np.float32(4.0)
    assert len(worker.get_service("actionservice").store.data) > 0


def test_call_lib_wo_permission(worker):
    root_domain_client = worker.root_client
    fname = ActionObject.from_obj("my_fake_file")
    obj1_pointer = fname.send(root_domain_client)
    action = Action(
        path="numpy",
        op="fromfile",
        args=[LineageID(obj1_pointer.id)],
        kwargs={},
        result_id=LineageID(),
    )
    kwargs = {"action": action}
    api_call = SyftAPICall(
        node_uid=worker.id, path="action.execute", args=[], kwargs=kwargs
    )
    res = root_domain_client.api.make_call(api_call)
    assert isinstance(res, SyftError)


def test_call_lib_custom_signature(worker):
    root_domain_client = worker.root_client
    # concatenate has a manually set signature
    assert all(
        root_domain_client.api.lib.numpy.concatenate(
            ([1, 2, 3], [4, 5, 6])
        ).syft_action_data
        == np.array([1, 2, 3, 4, 5, 6])
    )


# def test_pointer_addition():
#     worker, context = setup_worker()

#     x1 = np.array([1, 2, 3])

#     x2 = np.array([2, 3, 4])

#     x1_action_obj = NumpyArrayObject(syft_action_data=x1)

#     x2_action_obj = NumpyArrayObject(syft_action_data=x2)

#     action_service_set_method = worker._get_service_method_from_path(
#         "ActionService.set"
#     )

#     pointer1 = action_service_set_method(context, x1_action_obj)

#     assert pointer1.is_ok()

#     pointer1 = pointer1.ok()

#     pointer2 = action_service_set_method(context, x2_action_obj)

#     assert pointer2.is_ok()

#     pointer2 = pointer2.ok()

#     def mock_func(self, action, sync) -> Result:
#         action_service_execute_method = worker._get_service_method_from_path(
#             "ActionService.execute"
#         )
#         return action_service_execute_method(context, action)

#     with mock.patch(
#         "syft.core.node.new.action_object.ActionObjectPointer.execute_action", mock_func
#     ):
#         result = pointer1 + pointer2

#         assert result.is_ok()

#         result = result.ok()

#         actual_result = x1 + x2

#         action_service_get_method = worker._get_service_method_from_path(
#             "ActionService.get"
#         )

#         result_action_obj = action_service_get_method(context, result.id)

#         assert result_action_obj.is_ok()

#         result_action_obj = result_action_obj.ok()

#         print("actual result addition result: ", actual_result)
#         print("Result of adding pointers: ", result_action_obj.syft_action_data)
#         assert (result_action_obj.syft_action_data == actual_result).all()
