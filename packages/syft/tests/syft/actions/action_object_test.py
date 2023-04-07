# stdlib

# stdlib
from typing import Any
from typing import Tuple
from typing import Type

# third party
import numpy as np
import pytest

# syft absolute
from syft.core.node.new.action_data_empty import ActionDataEmpty
from syft.core.node.new.action_object import Action
from syft.core.node.new.action_object import ActionObject
from syft.core.node.new.action_object import HOOK_ALWAYS
from syft.core.node.new.action_object import PreHookContext
from syft.core.node.new.action_object import make_action_side_effect
from syft.core.node.new.action_object import propagate_node_uid
from syft.core.node.new.action_object import send_action_side_effect
from syft.core.node.new.action_types import action_type_for_type


# Test Action class
@pytest.mark.parametrize(
    "path_op",
    [
        ("str", "__len__"),
        ("ActionDataEmpty", "__version__"),
    ],
)
def test_action_sanity(path_op: Tuple[str, str]):
    path, op = path_op

    remote_self = Action.make_result_id(None)
    result_id = Action.make_result_id(None)
    new_action = Action(
        path=path,
        op=op,
        remote_self=remote_self,
        args=[],
        kwargs={},
        result_id=result_id,
    )
    assert new_action is not None
    assert new_action.full_path == f"{path}.{op}"
    assert new_action.syft_history_hash != 0


# Test ActionObject class
@pytest.mark.parametrize("orig_obj", ["abc", 1, 2.3, False, ActionDataEmpty()])
def test_actionobject_from_obj_sanity(orig_obj: Any):
    # no id
    obj = ActionObject.from_obj(orig_obj)
    assert obj.is_ok()
    assert obj.ok().id is not None
    assert obj.ok().syft_history_hash is not None

    # with id
    obj_id = Action.make_id(None)
    obj = ActionObject.from_obj(orig_obj, id=obj_id)
    assert obj.is_ok()
    assert obj.ok().id == obj_id
    assert obj.ok().syft_history_hash == hash(obj_id)

    # with id and lineage id
    obj_id = Action.make_id(None)
    lin_obj_id = Action.make_result_id(obj_id)
    obj = ActionObject.from_obj(orig_obj, id=obj_id, syft_lineage_id=lin_obj_id)
    assert obj.is_ok()
    assert obj.ok().id == obj_id
    assert obj.ok().syft_history_hash == lin_obj_id.syft_history_hash


def test_actionobject_from_obj_fail():
    obj_id = Action.make_id(None)
    lineage_id = Action.make_result_id(None)

    obj = ActionObject.from_obj("abc", id=obj_id, syft_lineage_id=lineage_id)
    assert obj.is_err()


@pytest.mark.parametrize("dtype", [int, float, str, Any, bool])
def test_actionobject_make_empty_sanity(dtype: Type):
    syft_type = action_type_for_type(dtype)

    obj = ActionObject.empty(
        syft_internal_type=syft_type, id=None, syft_lineage_id=None
    )
    assert obj.is_ok()
    assert obj.ok().id is not None
    assert obj.ok().syft_history_hash is not None

    # with id
    obj_id = Action.make_id(None)
    obj = ActionObject.empty(syft_internal_type=syft_type, id=obj_id)
    assert obj.is_ok()
    assert obj.ok().id == obj_id
    assert obj.ok().syft_history_hash == hash(obj_id)

    # with id and lineage id
    obj_id = Action.make_id(None)
    lin_obj_id = Action.make_result_id(obj_id)
    obj = ActionObject.empty(
        syft_internal_type=syft_type, id=obj_id, syft_lineage_id=lin_obj_id
    )
    assert obj.is_ok()
    assert obj.ok().id == obj_id
    assert obj.ok().syft_history_hash == lin_obj_id.syft_history_hash


@pytest.mark.parametrize("orig_obj", ["abc", 1, 2.3, False, ActionDataEmpty()])
def test_actionobject_hooks_init(orig_obj: Any):
    obj = ActionObject.from_obj(orig_obj)
    assert obj.is_ok()
    obj = obj.ok()

    assert HOOK_ALWAYS in obj._syft_pre_hooks__
    assert HOOK_ALWAYS in obj._syft_post_hooks__

    assert make_action_side_effect in obj._syft_pre_hooks__[HOOK_ALWAYS]
    assert send_action_side_effect in obj._syft_pre_hooks__[HOOK_ALWAYS]
    assert propagate_node_uid in obj._syft_post_hooks__[HOOK_ALWAYS]


@pytest.mark.parametrize(
    "orig_obj_op",
    [
        ("abc", "__len__"),
        (np.asarray([1, 2, 3]), "shape"),
    ],
)
def test_actionobject_hooks_make_action_side_effect(orig_obj_op: Any):
    orig_obj, op = orig_obj_op
    action_type_for_type(type(orig_obj))

    obj = ActionObject.from_obj(orig_obj)
    obj = obj.ok()

    context = PreHookContext(obj=obj, op_name=op)
    result = make_action_side_effect(context)
    assert result.is_ok()

    context, args, kwargs = result.ok()
    assert context.action is not None
    assert isinstance(context.action, Action)
    assert context.action.full_path.endswith("." + op)


def test_actionobject_hooks_send_action_side_effect(worker):
    orig_obj = "abc"
    op = "capitalize"

    obj = ActionObject.from_obj(orig_obj)
    obj = obj.ok()

    root_domain_client = worker.root_client
    pointer = root_domain_client.api.services.action.set(obj)

    context = PreHookContext(obj=pointer, op_name=op)
    result = send_action_side_effect(context)
    assert result.is_ok()

    context, args, kwargs = result.ok()
    assert context.result_id is not None


def test_actionobject_hooks_propagate_node_uid_err(worker):
    orig_obj = "abc"
    op = "capitalize"

    obj = ActionObject.from_obj(orig_obj)
    obj = obj.ok()

    context = PreHookContext(obj=obj, op_name=op)
    result = propagate_node_uid(context, op=op, result="orig_obj")
    assert result.is_err()


def test_actionobject_hooks_propagate_node_uid_ok(worker):
    orig_obj = "abc"
    op = "capitalize"

    obj_id = Action.make_id(None)
    obj = ActionObject.from_obj(orig_obj)
    obj = obj.ok()

    obj.syft_point_to(obj_id)

    context = PreHookContext(obj=obj, op_name=op)
    result = propagate_node_uid(context, op=op, result="orig_obj")
    assert result.is_ok()


def test_actionobject_syft_point_to():
    orig_obj = "abc"

    obj_id = Action.make_id(None)
    obj = ActionObject.from_obj(orig_obj)
    obj = obj.ok()

    obj.syft_point_to(obj_id)

    assert obj.syft_node_uid == obj_id


@pytest.mark.parametrize(
    "testcase",
    [
        ("abc", "capitalize", "Abc"),
    ],
)
def test_actionobject_syft_execute(worker, testcase):
    orig_obj, op, expected = testcase

    obj = ActionObject.from_obj(orig_obj)
    obj = obj.ok()

    root_domain_client = worker.root_client
    pointer = root_domain_client.api.services.action.set(obj)

    context = PreHookContext(obj=pointer, op_name=op)
    result = make_action_side_effect(context)
    context, _, _ = result.ok()

    action_result = context.obj.syft_execute_action(context.action, sync=True)
    assert action_result == expected


def test_actionobject_syft_make_action():
    raise NotImplementedError()


def test_actionobject_syft_make_method_action():
    raise NotImplementedError()


def test_actionobject_syft_get_path():
    raise NotImplementedError()


# TODO: improve
@pytest.mark.parametrize(
    "testcase",
    [
        ("abc", "capitalize", "Abc"),
        ("a b c", "strip", "abc"),
        ("123", "isnumeric", True),
    ],
)
def test_actionobject_syft_getattr(worker, testcase):
    orig_obj, attribute, expected = testcase

    obj = ActionObject.from_obj(orig_obj)
    obj = obj.ok()

    assert obj.__getattribute__(attribute) == expected


# TODO
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
