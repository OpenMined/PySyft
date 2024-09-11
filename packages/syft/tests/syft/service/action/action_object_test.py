# stdlib
from collections.abc import Callable
from enum import Enum
import inspect
import math
import sys
from typing import Any

# third party
import numpy as np
import pandas as pd
import pytest

# syft absolute
from syft.service.action.action_data_empty import ActionDataEmpty
from syft.service.action.action_object import Action
from syft.service.action.action_object import ActionObject
from syft.service.action.action_object import ActionType
from syft.service.action.action_object import HOOK_ALWAYS
from syft.service.action.action_object import HOOK_ON_POINTERS
from syft.service.action.action_object import PreHookContext
from syft.service.action.action_object import make_action_side_effect
from syft.service.action.action_object import propagate_server_uid
from syft.service.action.action_object import send_action_side_effect
from syft.service.action.action_types import action_type_for_type
from syft.service.response import SyftSuccess
from syft.store.blob_storage import SyftObjectRetrieval
from syft.types.errors import SyftException
from syft.types.uid import LineageID
from syft.types.uid import UID


def helper_make_action_obj(orig_obj: Any):
    return ActionObject.from_obj(orig_obj)


def helper_make_action_pointers(worker, obj, *args, **kwargs):
    root_datasite_client = worker.root_client
    res = obj.send(root_datasite_client)
    obj_pointer = root_datasite_client.api.services.action.get_pointer(res.id)

    # The args and kwargs should automatically be pointerized by obj_pointer
    return obj_pointer, args, kwargs


# Test Action class
@pytest.mark.parametrize(
    "path_op",
    [
        # (object, operation)
        ("str", "__len__"),
        ("ActionDataEmpty", "__version__"),
        ("int", "__add__"),
        ("float", "__add__"),
        ("bool", "__and__"),
        ("tuple", "count"),
        ("list", "count"),
        ("dict", "keys"),
        ("set", "add"),
    ],
)
def test_action_sanity(path_op: tuple[str, str]):
    path, op = path_op

    remote_self = LineageID()
    new_action = Action(
        path=path,
        op=op,
        remote_self=remote_self,
        args=[],
        kwargs={},
    )
    assert new_action is not None
    assert new_action.full_path == f"{path}.{op}"
    assert new_action.syft_history_hash != 0


# Test ActionObject class
@pytest.mark.parametrize(
    "orig_obj",
    [
        "abc",
        1,
        2.3,
        False,
        (1, 2, 3),
        [1, 2, 3],
        {"a": 1, "b": 2},
        {1, 2, 3},
        ActionDataEmpty(),
    ],
)
def test_actionobject_from_obj_sanity(orig_obj: Any):
    # no id
    obj = ActionObject.from_obj(orig_obj)
    assert obj.id is not None
    assert obj.syft_history_hash is not None

    # with id
    obj_id = UID()
    obj = ActionObject.from_obj(orig_obj, id=obj_id)
    assert obj.id == obj_id
    assert obj.syft_history_hash == hash(obj_id)

    # with id and lineage id
    obj_id = UID()
    lin_obj_id = LineageID(obj_id)
    obj = ActionObject.from_obj(orig_obj, id=obj_id, syft_lineage_id=lin_obj_id)
    assert obj.id == obj_id
    assert obj.syft_history_hash == lin_obj_id.syft_history_hash


def test_actionobject_from_obj_fail_id_mismatch():
    obj_id = UID()
    lineage_id = LineageID()

    with pytest.raises(ValueError):
        ActionObject.from_obj("abc", id=obj_id, syft_lineage_id=lineage_id)


@pytest.mark.parametrize("dtype", [int, float, str, Any, bool, dict, set, tuple, list])
def test_actionobject_make_empty_sanity(dtype: type):
    syft_type = action_type_for_type(dtype)

    obj = ActionObject.empty(
        syft_internal_type=syft_type, id=None, syft_lineage_id=None
    )
    assert obj.id is not None
    assert obj.syft_history_hash is not None

    # with id
    obj_id = UID()
    obj = ActionObject.empty(syft_internal_type=syft_type, id=obj_id)
    assert obj.id == obj_id
    assert obj.syft_history_hash == hash(obj_id)

    # with id and lineage id
    obj_id = UID()
    lin_obj_id = LineageID(obj_id)
    obj = ActionObject.empty(
        syft_internal_type=syft_type, id=obj_id, syft_lineage_id=lin_obj_id
    )
    assert obj.id == obj_id
    assert obj.syft_history_hash == lin_obj_id.syft_history_hash


@pytest.mark.parametrize(
    "orig_obj",
    [
        "abc",
        1,
        2.3,
        False,
        (1, 2, 3),
        [1, 2, 3],
        {"a": 1, "b": 2},
        {1, 2, 3},
        ActionDataEmpty(),
    ],
)
def test_actionobject_hooks_init(orig_obj: Any):
    obj = ActionObject.from_obj(orig_obj)

    assert HOOK_ALWAYS in obj.syft_pre_hooks__
    assert HOOK_ALWAYS in obj.syft_post_hooks__
    assert HOOK_ON_POINTERS in obj.syft_pre_hooks__
    assert HOOK_ON_POINTERS in obj.syft_post_hooks__

    assert make_action_side_effect in obj.syft_pre_hooks__[HOOK_ALWAYS]


def test_actionobject_add_pre_hooks():
    # Eager execution is disabled by default
    obj = ActionObject.from_obj(1)

    assert make_action_side_effect in obj.syft_pre_hooks__[HOOK_ALWAYS]
    assert send_action_side_effect not in obj.syft_pre_hooks__[HOOK_ON_POINTERS]
    assert propagate_server_uid not in obj.syft_post_hooks__[HOOK_ALWAYS]

    # eager exec tests:
    obj._syft_add_pre_hooks__(eager_execution=True)
    obj._syft_add_post_hooks__(eager_execution=True)

    assert make_action_side_effect in obj.syft_pre_hooks__[HOOK_ALWAYS]
    assert send_action_side_effect in obj.syft_pre_hooks__[HOOK_ON_POINTERS]
    assert propagate_server_uid in obj.syft_post_hooks__[HOOK_ALWAYS]


@pytest.mark.parametrize(
    "orig_obj_op",
    [
        # (object, operation)
        ("abc", "__len__"),
        (ActionDataEmpty(), "__version__"),
        (1, "__add__"),
        (1.2, "__add__"),
        (True, "__and__"),
        ((1, 2, 3), "count"),
        ([1, 2, 3], "count"),
        ({"a": 1, "b": 2}, "keys"),
        ({1, 2, 3}, "add"),
    ],
)
def test_actionobject_hooks_make_action_side_effect(orig_obj_op: Any):
    orig_obj, op = orig_obj_op
    action_type_for_type(type(orig_obj))

    obj = ActionObject.from_obj(orig_obj)

    context = PreHookContext(obj=obj, op_name=op)
    context, args, kwargs = make_action_side_effect(context).unwrap()
    assert context.action is not None
    assert isinstance(context.action, Action)
    assert context.action.full_path.endswith("." + op)


def test_actionobject_hooks_send_action_side_effect_err_no_id(worker):
    orig_obj = "abc"
    op = "capitalize"

    obj = ActionObject.from_obj(orig_obj)

    context = PreHookContext(obj=obj, op_name=op)
    result = send_action_side_effect(context)
    assert result.is_err()


def test_actionobject_hooks_send_action_side_effect_err_invalid_args(worker):
    orig_obj, op, args, kwargs = (1, 2, 3), "count", [], {}  # count expect one argument

    obj = helper_make_action_obj(orig_obj)
    obj_pointer, args_pointers, kwargs_pointers = helper_make_action_pointers(
        worker, obj, *args, **kwargs
    )

    context = PreHookContext(obj=obj_pointer, op_name=op)
    result = send_action_side_effect(context, *args_pointers, **kwargs_pointers)
    assert result.is_err()


@pytest.mark.parametrize(
    "orig_obj_op",
    [
        # (object, operation, *args, **kwargs)
        (1, "__len__", [1], {}),
        (1.2, "__len__", [1], {}),
        (True, "__len__", [True], {}),
        ([1, 2, 3], "__len__", [4], {}),
        ({"a": 1, "b": 2}, "__len__", [7], {}),
        ({1, 2, 3}, "__len__", [5], {}),
    ],
)
def test_actionobject_hooks_send_action_side_effect_ignore_op(
    root_datasite_client, orig_obj_op
):
    orig_obj, op, args, kwargs = orig_obj_op

    obj = helper_make_action_obj(orig_obj)
    obj = obj.send(root_datasite_client)

    context = PreHookContext(obj=obj, op_name=op)
    result = send_action_side_effect(context, *args, **kwargs)
    assert result.is_err()


@pytest.mark.parametrize(
    "orig_obj_op",
    [
        # (object, operation, *args, **kwargs)
        ("abc", "capitalize", [], {}),
        ("abc", "find", ["b"], {}),
        # (ActionDataEmpty(), "__version__", [], {}), TODO :ActionService cannot handle ActionDataEmpty
        (1, "__add__", [1], {}),
        (1.2, "__add__", [1], {}),
        (True, "__and__", [True], {}),
        ((1, 2, 3), "count", [1], {}),
        ([1, 2, 3], "count", [1], {}),
        ([1, 2, 3], "append", [4], {}),
        # ({"a"  :1, "b" : 2}, "keys", [], {}), TODO: dict_keys cannot be serialized
        ({"a": 1, "b": 2}, "update", [{"c": 3}], {}),
        ({1, 2, 3}, "add", [5], {}),
        ({1, 2, 3}, "clear", [], {}),
    ],
)
def test_actionobject_hooks_send_action_side_effect_ok(worker, orig_obj_op):
    orig_obj, op, args, kwargs = orig_obj_op

    obj = helper_make_action_obj(orig_obj)

    obj_pointer, args_pointers, kwargs_pointers = helper_make_action_pointers(
        worker, obj, *args, **kwargs
    )

    context = PreHookContext(obj=obj_pointer, op_name=op, action_type=ActionType.METHOD)
    result = send_action_side_effect(context, *args_pointers, **kwargs_pointers)
    assert result.is_ok()

    context, args, kwargs = result.ok()
    assert context.result_id is not None


def test_actionobject_hooks_propagate_server_uid_err():
    orig_obj = "abc"
    op = "capitalize"

    obj = ActionObject.from_obj(orig_obj)

    context = PreHookContext(obj=obj, op_name=op)
    result = propagate_server_uid(context, op=op, result="orig_obj")
    assert result.is_err()


def test_actionobject_hooks_propagate_server_uid_ok():
    orig_obj = "abc"
    op = "capitalize"

    obj_id = UID()
    obj = ActionObject.from_obj(orig_obj)

    obj.syft_point_to(obj_id)

    context = PreHookContext(obj=obj, op_name=op)
    result = propagate_server_uid(context, op=op, result="orig_obj")
    assert result.is_ok()


def test_actionobject_syft_point_to():
    orig_obj = "abc"

    obj_id = UID()
    obj = ActionObject.from_obj(orig_obj)

    obj.syft_point_to(obj_id)

    assert obj.syft_server_uid == obj_id


@pytest.mark.parametrize(
    "testcase",
    [
        # (object, operation, *args, **kwargs, expected_result)
        ("abc", "capitalize", [], {}, "Abc"),
        ("abc", "find", ["b"], {}, 1),
        (1, "__add__", [1], {}, 2),
        (1.2, "__add__", [1], {}, 2.2),
        (True, "__and__", [False], {}, False),
        ((1, 1, 3), "count", [1], {}, 2),
        ([1, 2, 1], "count", [1], {}, 2),
        (complex(1, 2), "conjugate", [], {}, complex(1, -2)),
    ],
)
def test_actionobject_syft_execute_ok(worker, testcase):
    orig_obj, op, args, kwargs, expected = testcase

    obj = helper_make_action_obj(orig_obj)

    obj_pointer, args_pointers, kwargs_pointers = helper_make_action_pointers(
        worker, obj, *args, **kwargs
    )

    context = PreHookContext(obj=obj_pointer, op_name=op, action_type=ActionType.METHOD)
    context, _, _ = make_action_side_effect(
        context, *args_pointers, **kwargs_pointers
    ).unwrap()

    action_result = context.obj.syft_execute_action(context.action, sync=True)
    assert action_result == expected

    action_result = context.obj._syft_output_action_object(action_result)
    assert isinstance(action_result, ActionObject)


@pytest.mark.parametrize(
    "testcase",
    [
        # (object, operation, *args, **kwargs)
        ("abc", "capitalize", [], {}),
        ("abc", "find", ["b"], {}),
        (1, "__add__", [1], {}),
        (1.2, "__add__", [1], {}),
        (True, "__and__", [False], {}),
        ((1, 1, 3), "count", [1], {}),
        ([1, 2, 1], "count", [1], {}),
        ([1, 2, 3], "append", [4], {}),
        ({"a": 1, "b": 2}, "update", [{"c": 3}], {}),
        ({1, 2, 3}, "add", [5], {}),
        ({1, 2, 3}, "clear", [], {}),
    ],
)
def test_actionobject_syft_make_action(worker, testcase):
    orig_obj, op, args, kwargs = testcase

    obj = helper_make_action_obj(orig_obj)
    obj_pointer, args_pointers, kwargs_pointers = helper_make_action_pointers(
        worker, obj, *args, **kwargs
    )

    path = str(type(orig_obj))
    action = obj.syft_make_action(path, op, args=args_pointers, kwargs=kwargs_pointers)

    assert action.full_path.endswith("." + op)


@pytest.mark.parametrize(
    "testcase",
    [
        # (object, operation, *args, **kwargs)
        ("abc", "capitalize", [], {}),
        ("abc", "find", ["b"], {}),
        (1, "__add__", [1], {}),
        (1.2, "__add__", [1], {}),
        (True, "__and__", [False], {}),
        ((1, 1, 3), "count", [1], {}),
        ([1, 2, 1], "count", [1], {}),
        ([1, 2, 3], "append", [4], {}),
        ({"a": 1, "b": 2}, "update", [{"c": 3}], {}),
        ({1, 2, 3}, "add", [5], {}),
        ({1, 2, 3}, "clear", [], {}),
        (complex(1, 2), "conjugate", [], {}),
    ],
)
def test_actionobject_syft_make_action_with_self(worker, testcase):
    orig_obj, op, args, kwargs = testcase

    obj = helper_make_action_obj(orig_obj)
    obj_pointer, args_pointers, kwargs_pointers = helper_make_action_pointers(
        worker, obj, *args, **kwargs
    )

    action = obj.syft_make_action_with_self(
        op, args=args_pointers, kwargs=kwargs_pointers
    )

    assert action.full_path.endswith("." + op)


@pytest.mark.parametrize(
    "testcase",
    [
        # (object, operation, *args, **kwargs)
        ("abc", "capitalize", [], {}),
        ("abc", "find", ["b"], {}),
        (1, "__add__", [1], {}),
        (1.2, "__add__", [1], {}),
        (True, "__and__", [False], {}),
        ((1, 1, 3), "count", [1], {}),
        ([1, 2, 1], "count", [1], {}),
        ([1, 2, 3], "append", [4], {}),
        ({"a": 1, "b": 2}, "update", [{"c": 3}], {}),
        ({1, 2, 3}, "add", [5], {}),
        ({1, 2, 3}, "clear", [], {}),
        (complex(1, 2), "conjugate", [], {}),
    ],
)
def test_actionobject_syft_make_remote_method_action(worker, testcase):
    orig_obj, op, args, kwargs = testcase

    obj = helper_make_action_obj(orig_obj)
    obj_pointer, args_pointers, kwargs_pointers = helper_make_action_pointers(
        worker, obj, *args, **kwargs
    )

    remote_cbk = obj.syft_remote_method(op)
    action = remote_cbk(*args_pointers, **kwargs_pointers)

    assert action.full_path.endswith("." + op)


@pytest.mark.parametrize(
    "testcase",
    [
        # object
        "abc",
        1,
        1.2,
        True,
        (1, 1, 3),
        [1, 2, 1],
        {"a": 1, "b": 2},
        {1, 2, 3},
        complex(1, 2),
    ],
)
def test_actionobject_syft_get_path(testcase):
    orig_obj = testcase
    obj = helper_make_action_obj(orig_obj)

    assert obj.syft_get_path() == type(orig_obj).__name__


@pytest.mark.parametrize(
    "testcase",
    [
        # object
        "abc",
        1,
        1.2,
        True,
        (1, 1, 3),
        [1, 2, 1],
        {"a": 1, "b": 2},
        {1, 2, 3},
        complex(1, 2),
    ],
)
def test_actionobject_syft_send_get(worker, testcase):
    root_datasite_client = worker.root_client
    root_datasite_client._fetch_api(root_datasite_client.credentials)
    action_store = worker.services.action.stash

    orig_obj = testcase
    obj = helper_make_action_obj(orig_obj)

    assert len(action_store._data) == 0

    ptr = obj.send(root_datasite_client)
    assert len(action_store._data) == 1
    retrieved = ptr.get()

    assert obj.syft_action_data == retrieved


@pytest.mark.parametrize(
    "testcase",
    [
        # object
        "abc",
        1,
        1.2,
        True,
        (1, 1, 3),
        [1, 2, 1],
        {"a": 1, "b": 2},
        {1, 2, 3},
        complex(1, 2),
    ],
)
def test_actionobject_syft_passthrough_attrs(testcase):
    obj = helper_make_action_obj(testcase)

    assert str(obj) == str(testcase)


@pytest.mark.parametrize(
    "testcase",
    [
        # object
        "abc",
        (1, 1, 3),
        [1, 2, 1],
        {"a": 1, "b": 2},
        {1, 2, 3},
    ],
)
def test_actionobject_syft_dont_wrap_output_attrs(testcase):
    obj = helper_make_action_obj(testcase)

    assert not hasattr(len(obj), "id")
    assert not hasattr(len(obj), "syft_history_hash")


def test_actionobject_syft_get_attr_context():
    orig_obj = "test"
    obj = helper_make_action_obj(orig_obj)

    assert obj._syft_get_attr_context("capitalize") is orig_obj
    assert obj._syft_get_attr_context("__add__") is orig_obj
    assert obj._syft_get_attr_context("syft_action_data") is obj.syft_action_data


@pytest.mark.parametrize(
    "testcase",
    [
        # (object, operation, *args, **kwargs, expected_result)
        (1, "__add__", [1], {}, 2),
        (1.2, "__add__", [1], {}, 2.2),
        (True, "__and__", [False], {}, False),
        ([1, 2, 3], "append", [4], {}, [1, 2, 3, 4]),
        ({"a": 1, "b": 2}, "update", [{"c": 3}], {}, {"a": 1, "b": 2, "c": 3}),
        ({1, 2, 3}, "add", [5], {}, {1, 2, 3, 5}),
        ({1, 2, 3}, "clear", [], {}, {}),
        (complex(1, 2), "conjugate", [], {}, complex(1, -2)),
    ],
)
@pytest.mark.skip(reason="Disabled until we bring back eager execution")
def test_actionobject_syft_execute_hooks(worker, testcase):
    client = worker.root_client
    assert client.settings.enable_eager_execution(enable=True)

    orig_obj, op, args, kwargs, expected = testcase

    obj = helper_make_action_obj(orig_obj)

    obj_pointer, args_pointers, kwargs_pointers = helper_make_action_pointers(
        worker, obj, *args, **kwargs
    )
    obj_pointer.syft_point_to(client.id)

    context = PreHookContext(obj=obj_pointer, op_name=op, action_type=ActionType.METHOD)

    context, result_args, result_kwargs = obj_pointer._syft_run_pre_hooks__(
        context, name=op, args=args_pointers, kwargs=kwargs_pointers
    )
    assert context.result_id is not None

    context.obj.syft_server_uid = UID()
    result = obj_pointer._syft_run_post_hooks__(context, name=op, result=obj_pointer)
    assert result.syft_server_uid == context.obj.syft_server_uid


@pytest.mark.parametrize(
    "testcase",
    [
        # object
        "abc",
        1,
        1.2,
        True,
        (1, 1, 3),
        [1, 2, 1],
        {"a": 1, "b": 2},
        {1, 2, 3},
        complex(1, 2),
    ],
)
def test_actionobject_syft_wrap_attribute_for_bool_on_nonbools(testcase):
    obj = helper_make_action_obj(testcase)

    assert isinstance(bool(obj), bool)


@pytest.mark.parametrize(
    "orig_obj",
    [
        # object
        "abc",
        1,
        1.2,
        True,
        (1, 1, 3),
        [1, 2, 1],
        {"a": 1, "b": 2},
        {1, 2, 3},
        complex(1, 2),
    ],
)
def test_actionobject_syft_wrap_attribute_for_properties(orig_obj):
    obj = helper_make_action_obj(orig_obj)

    # test properties from the original object
    for method in dir(orig_obj):
        klass_method = getattr(type(orig_obj), method, None)
        if klass_method is None:
            continue

        if isinstance(klass_method, property) or inspect.isdatadescriptor(klass_method):
            prop = getattr(obj, method)
            assert prop is not None
            assert isinstance(prop, ActionObject)
            assert hasattr(prop, "id")
            assert hasattr(prop, "syft_server_uid")
            assert hasattr(prop, "syft_history_hash")


@pytest.mark.parametrize(
    "orig_obj",
    [
        # object
        "abc",
        1,
        1.2,
        True,
        (1, 1, 3),
        [1, 2, 1],
        {"a": 1, "b": 2},
        {1, 2, 3},
        complex(1, 2),
    ],
)
def test_actionobject_syft_wrap_attribute_for_methods(orig_obj):
    obj = helper_make_action_obj(orig_obj)

    # test properties from the original object
    for name in dir(orig_obj):
        method = getattr(obj, name)
        klass_method = getattr(type(orig_obj), name, None)
        if klass_method is None:
            continue

        if isinstance(klass_method, property) or inspect.isdatadescriptor(klass_method):
            # ignore properties
            continue

        assert method is not None
        assert isinstance(method, Callable)


class AttrScenario(Enum):
    AS_OBJ = 0
    AS_PTR = 1
    AS_UID = 2
    AS_LUID = 3


def helper_prepare_obj_for_scenario(scenario: AttrScenario, worker, obj: ActionObject):
    if scenario == AttrScenario.AS_OBJ:
        return obj
    elif scenario == AttrScenario.AS_PTR:
        obj, _, _ = helper_make_action_pointers(worker, obj, *[], **{})
        return obj
    else:
        raise ValueError(scenario)


@pytest.mark.parametrize("scenario", [AttrScenario.AS_OBJ, AttrScenario.AS_PTR])
def test_actionobject_syft_getattr_str(worker, scenario):
    orig_obj = "a bC"

    obj = ActionObject.from_obj(orig_obj)
    obj = helper_prepare_obj_for_scenario(scenario, worker, obj)

    assert obj == orig_obj
    assert obj != "sdfsfs"

    assert obj.capitalize() == orig_obj.capitalize()
    assert obj.casefold() == orig_obj.casefold()
    assert obj.endswith("C") == orig_obj.endswith("C")  # noqa
    assert obj.isascii() == orig_obj.isascii()  # noqa
    assert obj.isdigit() == orig_obj.isdigit()  # noqa
    assert obj.upper() == orig_obj.upper()
    assert "C" in obj
    assert "z" not in obj
    assert obj[0] == orig_obj[0]
    assert f"test {obj}" == f"test {orig_obj}"
    assert obj > "a"
    assert obj < "zzzz"
    for idx, c in enumerate(obj):
        assert c == orig_obj[idx]
        assert obj[idx] == orig_obj[idx]
    for idx, c in enumerate(orig_obj):
        assert c == obj[idx]

    assert sorted(obj) == sorted(orig_obj)
    assert list(obj) == list(orig_obj)
    assert list(reversed(obj)) == list(reversed(orig_obj))


def test_actionobject_syft_getattr_str_history():
    obj1 = ActionObject.from_obj("abc")
    obj2 = ActionObject.from_obj("xyz")

    res1 = obj1 + obj2
    res2 = obj1 + obj2
    assert res1.syft_history_hash == res2.syft_history_hash


@pytest.mark.parametrize("scenario", [AttrScenario.AS_OBJ, AttrScenario.AS_PTR])
def test_actionobject_syft_getattr_list(worker, scenario):
    orig_obj = [3, 2, 1, 4]

    obj = ActionObject.from_obj(orig_obj)
    obj = helper_prepare_obj_for_scenario(scenario, worker, obj)

    assert 1 in obj
    assert obj[0] == 3

    for idx, item in enumerate(obj):
        assert item == orig_obj[idx]
    for idx, item in enumerate(orig_obj):
        assert item == obj[idx]

    assert obj == orig_obj
    assert len(obj) == 4
    assert obj.count(1) == 1
    assert obj.append(5) == [1, 2, 3, 4, 5]
    assert obj.sort() == [1, 2, 3, 4, 5]
    assert obj.clear() == []


def test_actionobject_syft_getattr_list_history():
    obj1 = ActionObject.from_obj([1, 2, 3, 4])
    obj2 = ActionObject.from_obj([5, 6, 7])

    res1 = obj1.extend(obj2)
    res2 = obj1.extend(obj2)
    assert res1.syft_history_hash == res2.syft_history_hash


@pytest.mark.parametrize("scenario", [AttrScenario.AS_OBJ, AttrScenario.AS_PTR])
def test_actionobject_syft_getattr_dict(worker, scenario):
    orig_obj = {"a": 1, "b": 2}

    obj = ActionObject.from_obj(orig_obj)
    obj = helper_prepare_obj_for_scenario(scenario, worker, obj)

    assert obj == orig_obj
    assert obj["a"] == 1
    assert obj.update({"c": 3}) == {"a": 1, "b": 2, "c": 3}
    assert "a" in obj
    assert obj["a"] == 1
    assert obj.clear() == {}


def test_actionobject_syft_getattr_dict_history():
    obj1 = ActionObject.from_obj({"a": 1, "b": 2})
    obj2 = ActionObject.from_obj({"c": 1, "b": 2})

    res1 = obj1.update(obj2)
    res2 = obj1.update(obj2)
    assert res1.syft_history_hash == res2.syft_history_hash


@pytest.mark.parametrize("scenario", [AttrScenario.AS_OBJ, AttrScenario.AS_PTR])
def test_actionobject_syft_getattr_tuple(worker, scenario):
    orig_obj = (1, 2, 3, 4, 4)

    obj = ActionObject.from_obj(orig_obj)
    obj = helper_prepare_obj_for_scenario(scenario, worker, obj)

    assert obj == orig_obj
    assert obj.count(4) == 2
    assert obj.index(2) == 1
    assert len(obj) == 5
    assert 1 in obj
    assert obj[0] == 1

    for idx, item in enumerate(obj):
        assert item == orig_obj[idx]
    for idx, item in enumerate(orig_obj):
        assert item == obj[idx]


@pytest.mark.parametrize("scenario", [AttrScenario.AS_OBJ, AttrScenario.AS_PTR])
def test_actionobject_syft_getattr_set(worker, scenario):
    orig_obj = {1, 2, 3, 4}

    obj = ActionObject.from_obj(orig_obj)
    obj = helper_prepare_obj_for_scenario(scenario, worker, obj)

    assert obj == orig_obj
    assert obj.add(4) == {1, 2, 3, 4}
    assert obj.intersection({1, 2, 121}) == {1, 2}
    assert len(obj) == 4


def test_actionobject_syft_getattr_set_history():
    obj1 = ActionObject.from_obj({1, 2, 3, 4})
    obj2 = ActionObject.from_obj({1, 2})

    res1 = obj1.intersection(obj2)
    res2 = obj1.intersection(obj2)
    assert res1.syft_history_hash == res2.syft_history_hash


@pytest.mark.parametrize("orig_obj", [True, False])
@pytest.mark.parametrize("scenario", [AttrScenario.AS_OBJ, AttrScenario.AS_PTR])
def test_actionobject_syft_getattr_bool(orig_obj, worker, scenario):
    obj = ActionObject.from_obj(orig_obj)
    obj = helper_prepare_obj_for_scenario(scenario, worker, obj)

    assert obj.__and__(False) == (orig_obj and False)  # noqa
    assert obj.__or__(False) == (orig_obj or False)  # noqa
    assert (not obj) == (not orig_obj)  # noqa
    assert (obj and True) == (orig_obj and True)  # noqa
    assert (True and obj) == (orig_obj and True)  # noqa
    assert (obj and False) == (orig_obj and False)  # noqa
    assert (False and obj) == (orig_obj and False)  # noqa
    assert (obj or False) == (orig_obj or False)  # noqa
    assert (False or obj) == (orig_obj or False)  # noqa
    assert (obj or True) == (orig_obj or True)  # noqa
    assert (True or obj) == (orig_obj or True)  # noqa
    assert (obj + obj) == orig_obj + orig_obj


def test_actionobject_syft_getattr_bool_history():
    orig_obj = True

    obj1 = ActionObject.from_obj(orig_obj)
    obj2 = ActionObject.from_obj(orig_obj)
    res1 = obj1 or obj2
    res2 = obj1 or obj2

    assert res1.syft_history_hash == res2.syft_history_hash


@pytest.mark.parametrize("orig_obj", [-5, 0, 5])
@pytest.mark.parametrize("scenario", [AttrScenario.AS_OBJ, AttrScenario.AS_PTR])
def test_actionobject_syft_getattr_int(orig_obj: int, worker, scenario):
    obj = ActionObject.from_obj(orig_obj)
    obj = helper_prepare_obj_for_scenario(scenario, worker, obj)

    assert obj == orig_obj
    assert obj != orig_obj + 1
    assert str(obj) == str(orig_obj)
    assert obj.__add__(1) == orig_obj + 1
    assert obj.__sub__(1) == orig_obj - 1
    assert obj.__mul__(2) == 2 * orig_obj
    assert obj < orig_obj + 1
    assert obj <= orig_obj
    assert obj > orig_obj - 1
    assert obj >= orig_obj - 1
    assert obj**2 == orig_obj**2
    assert obj**3 == orig_obj**3
    assert 2**obj == 2**orig_obj
    assert obj % 2 == orig_obj % 2
    assert obj / 2 == orig_obj / 2
    assert obj // 2 == orig_obj // 2
    if obj != 0:
        assert 10 % obj == 10 % orig_obj
        assert 11 / obj == 11 / orig_obj
        assert 11 // obj == 11 // orig_obj
    assert bool(obj) == bool(orig_obj)
    assert float(obj) == float(orig_obj)
    assert round(obj) == round(orig_obj)
    assert obj + 2 == 2 + orig_obj
    assert 2 + obj == 2 + orig_obj
    assert obj - 2 == orig_obj - 2
    assert 7 - obj == 7 - orig_obj
    assert 2 * obj == 2 * orig_obj
    assert obj * 2 == 2 * orig_obj
    assert -obj == -orig_obj
    assert +obj == +orig_obj
    assert abs(obj) == abs(orig_obj)
    assert math.ceil(obj) == math.ceil(orig_obj)
    assert math.floor(obj) == math.floor(orig_obj)

    # bitwise
    assert (obj | 3) == (orig_obj | 3)
    assert (3 | obj) == (orig_obj | 3)
    assert (obj & 3) == (orig_obj & 3)
    assert (3 & obj) == (orig_obj & 3)
    assert (obj ^ 3) == (orig_obj ^ 3)
    assert (3 ^ obj) == (orig_obj ^ 3)
    assert ~obj == ~orig_obj
    assert (obj >> 1) == (orig_obj >> 1)
    assert (obj << 1) == (orig_obj << 1)
    if obj > 0:
        assert (3 << obj) == (3 << orig_obj)
        assert (3 >> obj) == (3 >> orig_obj)


def test_actionobject_syft_getattr_int_history():
    orig_obj = 5
    obj1 = ActionObject.from_obj(orig_obj)
    obj2 = ActionObject.from_obj(orig_obj)
    res1 = obj1 + obj2
    res2 = obj1 + obj2
    assert res1.syft_history_hash == res2.syft_history_hash


@pytest.mark.parametrize("orig_obj", [-5.5, 0.0, 5.5])
@pytest.mark.parametrize("scenario", [AttrScenario.AS_OBJ, AttrScenario.AS_PTR])
def test_actionobject_syft_getattr_float(orig_obj: float, worker, scenario):
    obj = ActionObject.from_obj(orig_obj)
    obj = helper_prepare_obj_for_scenario(scenario, worker, obj)

    assert obj == orig_obj
    assert obj != orig_obj + 1
    assert str(obj) == str(orig_obj)
    assert obj.__add__(1.1) == orig_obj + 1.1
    assert obj.__sub__(1.5) == orig_obj - 1.5
    assert obj.__mul__(2) == 2 * orig_obj
    assert obj < orig_obj + 1
    assert obj <= orig_obj
    assert obj > orig_obj - 1
    assert obj >= orig_obj - 1
    assert bool(obj) == bool(orig_obj)
    assert float(obj) == float(orig_obj)
    assert int(obj) == int(orig_obj)
    assert round(obj) == round(orig_obj)
    assert obj**2 == orig_obj**2
    assert obj**3 == orig_obj**3
    assert 2**obj == 2**orig_obj
    assert obj + 2 == 2 + orig_obj
    assert 2 + obj == 2 + orig_obj
    assert obj - 2 == orig_obj - 2
    assert 7 - obj == 7 - orig_obj
    assert 2 * obj == 2 * orig_obj
    assert obj * 2 == 2 * orig_obj
    assert obj / 2 == orig_obj / 2
    assert obj // 2 == orig_obj // 2
    if obj != 0:
        assert 11 / obj == 11 / orig_obj
        assert 11 // obj == 11 // orig_obj
    assert -obj == -orig_obj
    assert +obj == +orig_obj
    assert abs(obj) == abs(orig_obj)
    assert math.ceil(obj) == math.ceil(orig_obj)
    assert math.floor(obj) == math.floor(orig_obj)
    assert math.trunc(obj) == math.trunc(orig_obj)


def test_actionobject_syft_getattr_float_history():
    obj1 = ActionObject.from_obj(5.5)
    obj2 = ActionObject.from_obj(5.2)

    res1 = obj1 + obj2
    res2 = obj1 + obj2

    assert res1.syft_history_hash == res2.syft_history_hash


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="This is a hackish way to test attribute set/get, and it might fail on Windows or OSX",
)
def test_actionobject_syft_getattr_np(worker):
    orig_obj = np.array([1, 2, 3])

    obj = ActionObject.from_obj(orig_obj)

    assert obj.dtype == orig_obj.dtype

    for dtype in ["int64", "float64"]:
        obj.dtype = dtype
        assert obj.dtype == dtype


def test_actionobject_syft_getattr_pandas(worker):
    orig_obj = pd.DataFrame([[1, 2, 3]], columns=["1", "2", "3"])

    obj = ActionObject.from_obj(orig_obj)

    assert (obj.columns == orig_obj.columns).all()

    obj.columns = ["a", "b", "c"]
    assert (obj.columns == ["a", "b", "c"]).all()


def test_actionobject_delete(worker):
    """
    Test deleting action objects and their corresponding blob storage entries
    """
    root_client = worker.root_client

    # small object with no blob store entry
    data_small = np.random.randint(0, 100, size=3)
    action_obj = ActionObject.from_obj(data_small)
    action_obj.send(root_client)
    assert action_obj.syft_blob_storage_entry_id is None
    del_res = root_client.api.services.action.delete(uid=action_obj.id)
    assert isinstance(del_res, SyftSuccess)

    # big object with blob store entry
    num_elements = 25 * 1024 * 1024
    data_big = np.random.randint(0, 100, size=num_elements)  # 4 bytes per int32
    action_obj_2 = ActionObject.from_obj(data_big)
    action_obj_2.send(root_client)
    assert isinstance(action_obj_2.syft_blob_storage_entry_id, UID)
    read_res = root_client.api.services.blob_storage.read(
        action_obj_2.syft_blob_storage_entry_id
    )
    assert isinstance(read_res, SyftObjectRetrieval)
    del_res = root_client.api.services.action.delete(uid=action_obj_2.id)
    assert isinstance(del_res, SyftSuccess)
    with pytest.raises(SyftException):
        read_res = root_client.api.services.blob_storage.read(
            action_obj_2.syft_blob_storage_entry_id
        )
