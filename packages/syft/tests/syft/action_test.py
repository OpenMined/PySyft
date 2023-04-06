# stdlib

# third party

# syft absolute
from syft import ActionObject


def test_actionobject_method(worker):
    root_domain_client = worker.root_client
    action_store = worker.get_service("actionservice").store
    obj = ActionObject.from_obj("abc")
    pointer = root_domain_client.api.services.action.set(obj)
    assert len(action_store.data) == 1
    res = pointer.capitalize()
    assert len(action_store.data) == 2
    assert res[0] == "A"


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
