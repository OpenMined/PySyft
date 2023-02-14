# stdlib
from unittest import mock

# third party
import numpy as np
from result import Result

# syft absolute
from syft.core.node.new.action_service import NumpyArrayObject
from syft.core.node.new.api import APIRegistry
from syft.core.node.new.api import SyftAPI
from syft.core.node.new.context import AuthedServiceContext
from syft.core.node.new.credentials import SyftSigningKey
from syft.core.node.worker import Worker


def setup_worker():
    test_signing_key = SyftSigningKey.generate()
    credentials = test_signing_key.verify_key
    worker = Worker(name="Test Worker", signing_key=test_signing_key.signing_key)
    context = AuthedServiceContext(node=worker, credentials=credentials)

    api = SyftAPI.for_user(node=worker)

    APIRegistry.set_api_for(node_uid=worker.id, api=api)

    return worker, context


def test_pointer_addition():
    worker, context = setup_worker()

    x1 = np.array([1, 2, 3])

    x2 = np.array([2, 3, 4])

    x1_action_obj = NumpyArrayObject(syft_action_data=x1)

    x2_action_obj = NumpyArrayObject(syft_action_data=x2)

    action_service_set_method = worker._get_service_method_from_path(
        "ActionService.set"
    )

    pointer1 = action_service_set_method(context, x1_action_obj)

    assert pointer1.is_ok()

    pointer1 = pointer1.ok()

    pointer2 = action_service_set_method(context, x2_action_obj)

    assert pointer2.is_ok()

    pointer2 = pointer2.ok()

    def mock_func(self, action, sync) -> Result:
        action_service_execute_method = worker._get_service_method_from_path(
            "ActionService.execute"
        )
        return action_service_execute_method(context, action)

    with mock.patch(
        "syft.core.node.new.action_object.ActionObjectPointer.execute_action", mock_func
    ):
        result = pointer1 + pointer2

        assert result.is_ok()

        result = result.ok()

        actual_result = x1 + x2

        action_service_get_method = worker._get_service_method_from_path(
            "ActionService.get"
        )

        result_action_obj = action_service_get_method(context, result.id)

        assert result_action_obj.is_ok()

        result_action_obj = result_action_obj.ok()

        print("actual result addition result: ", actual_result)
        print("Result of adding pointers: ", result_action_obj.syft_action_data)
        assert (result_action_obj.syft_action_data == actual_result).all()
