# stdlib
from typing import Any
from typing import Tuple
from typing import Type

# third party
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


def helper_make_action_obj(orig_obj: Any):
    obj_id = Action.make_id(None)
    lin_obj_id = Action.make_result_id(obj_id)
    obj = ActionObject.from_obj(orig_obj, id=obj_id, syft_lineage_id=lin_obj_id)

    return obj.ok()


def helper_make_action_args(*args, **kwargs):
    act_args = []
    act_kwargs = {}

    for v in args:
        act_args.append(helper_make_action_obj(v))

    for v in kwargs:
        act_kwargs[v] = helper_make_action_obj(kwargs[v])

    return act_args, act_kwargs


def helper_make_action_pointers(worker, obj, *args, **kwargs):
    args_pointers, kwargs_pointers = [], {}

    root_domain_client = worker.root_client
    obj_pointer = root_domain_client.api.services.action.set(obj)

    for arg in args:
        root_domain_client.api.services.action.set(arg)
        args_pointers.append(arg.id)

    for key in kwargs:
        root_domain_client.api.services.action.set(kwargs[key])
        kwargs_pointers[key] = kwargs[key].id

    return obj_pointer, args_pointers, kwargs_pointers


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
        set({1, 2, 3, 3}),
        ActionDataEmpty(),
    ],
)
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


def test_actionobject_from_obj_fail_id_mismatch():
    obj_id = Action.make_id(None)
    lineage_id = Action.make_result_id(None)

    obj = ActionObject.from_obj("abc", id=obj_id, syft_lineage_id=lineage_id)
    assert obj.is_err()


@pytest.mark.parametrize("dtype", [int, float, str, Any, bool, dict, set, tuple, list])
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
        set({1, 2, 3, 3}),
        ActionDataEmpty(),
    ],
)
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
        # (object, operation)
        ("abc", "__len__"),
        (ActionDataEmpty(), "__version__"),
        (int(1), "__add__"),
        (float(1.2), "__add__"),
        (True, "__and__"),
        ((1, 2, 3), "count"),
        ([1, 2, 3], "count"),
        ({"a": 1, "b": 2}, "keys"),
        (set({1, 2, 3, 3}), "add"),
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


def test_actionobject_hooks_send_action_side_effect_err_no_id(worker):
    orig_obj = "abc"
    op = "capitalize"

    obj = ActionObject.from_obj(orig_obj)
    obj = obj.ok()

    context = PreHookContext(obj=obj, op_name=op)
    result = send_action_side_effect(context)
    assert result.is_err()


def test_actionobject_hooks_send_action_side_effect_err_invalid_args(worker):
    orig_obj, op, args, kwargs = (1, 2, 3), "count", [], {}  # count expect one argument

    obj = helper_make_action_obj(orig_obj)
    args, kwargs = helper_make_action_args(*args, **kwargs)

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
        ("abc", "__len__", [], {}),
        (int(1), "__len__", [1], {}),
        (float(1.2), "__len__", [1], {}),
        (True, "__len__", [True], {}),
        ((1, 2, 3), "__len__", [], {}),
        ([1, 2, 3], "__len__", [4], {}),
        ({"a": 1, "b": 2}, "__len__", [], {}),
        (set({1, 2, 3, 3}), "__len__", [5], {}),
    ],
)
def test_actionobject_hooks_send_action_side_effect_ignore_op(orig_obj_op):
    orig_obj, op, args, kwargs = orig_obj_op

    obj = helper_make_action_obj(orig_obj)
    args, kwargs = helper_make_action_args(*args, **kwargs)

    context = PreHookContext(obj=obj, op_name=op)
    result = send_action_side_effect(context, *args, **kwargs)
    assert result.is_ok()

    context, args, kwargs = result.ok()
    assert context.result_id is None  # operation was ignored


@pytest.mark.parametrize(
    "orig_obj_op",
    [
        # (object, operation, *args, **kwargs)
        ("abc", "capitalize", [], {}),
        ("abc", "find", ["b"], {}),
        # (ActionDataEmpty(), "__version__", [], {}), TODO :ActionService cannot handle ActionDataEmpty
        (int(1), "__add__", [1], {}),
        (float(1.2), "__add__", [1], {}),
        (True, "__and__", [True], {}),
        ((1, 2, 3), "count", [1], {}),
        ([1, 2, 3], "count", [1], {}),
        ([1, 2, 3], "append", [4], {}),
        # ({"a"  :1, "b" : 2}, "keys", [], {}), TODO: dict_keys cannot be serialized
        ({"a": 1, "b": 2}, "update", [{"c": 3}], {}),
        (set({1, 2, 3, 3}), "add", [5], {}),
        (set({1, 2, 3, 3}), "clear", [], {}),
    ],
)
def test_actionobject_hooks_send_action_side_effect_ok(worker, orig_obj_op):
    orig_obj, op, args, kwargs = orig_obj_op

    obj = helper_make_action_obj(orig_obj)
    args, kwargs = helper_make_action_args(*args, **kwargs)

    obj_pointer, args_pointers, kwargs_pointers = helper_make_action_pointers(
        worker, obj, *args, **kwargs
    )

    context = PreHookContext(obj=obj_pointer, op_name=op)
    result = send_action_side_effect(context, *args_pointers, **kwargs_pointers)
    assert result.is_ok()

    context, args, kwargs = result.ok()
    assert context.result_id is not None


def test_actionobject_hooks_propagate_node_uid_err():
    orig_obj = "abc"
    op = "capitalize"

    obj = ActionObject.from_obj(orig_obj)
    obj = obj.ok()

    context = PreHookContext(obj=obj, op_name=op)
    result = propagate_node_uid(context, op=op, result="orig_obj")
    assert result.is_err()


def test_actionobject_hooks_propagate_node_uid_ok():
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
        # (object, operation, *args, **kwargs, expected_result)
        ("abc", "capitalize", [], {}, "Abc"),
        ("abc", "find", ["b"], {}, 1),
        (int(1), "__add__", [1], {}, 2),
        (float(1.2), "__add__", [1], {}, 2.2),
        (True, "__and__", [False], {}, False),
        ((1, 1, 3), "count", [1], {}, 2),
        ([1, 2, 1], "count", [1], {}, 2),
        ([1, 2, 3], "append", [4], {}, [1, 2, 3, 4]),
        ({"a": 1, "b": 2}, "update", [{"c": 3}], {}, {"a": 1, "b": 2, "c": 3}),
        (set({1, 2, 3, 3}), "add", [5], {}, set({1, 2, 3, 5})),
        (set({1, 2, 3, 3}), "clear", [], {}, set({})),
    ],
)
def test_actionobject_syft_execute_ok(worker, testcase):
    orig_obj, op, args, kwargs, expected = testcase

    obj = helper_make_action_obj(orig_obj)
    args, kwargs = helper_make_action_args(*args, **kwargs)

    obj_pointer, args_pointers, kwargs_pointers = helper_make_action_pointers(
        worker, obj, *args, **kwargs
    )

    context = PreHookContext(obj=obj_pointer, op_name=op)
    result = make_action_side_effect(context, *args_pointers, **kwargs_pointers)
    context, _, _ = result.ok()

    action_result = context.obj.syft_execute_action(context.action, sync=True)
    assert action_result == expected


@pytest.mark.parametrize(
    "testcase",
    [
        # (object, operation, *args, **kwargs)
        ("abc", "capitalize", [], {}),
        ("abc", "find", ["b"], {}),
        (int(1), "__add__", [1], {}),
        (float(1.2), "__add__", [1], {}),
        (True, "__and__", [False], {}),
        ((1, 1, 3), "count", [1], {}),
        ([1, 2, 1], "count", [1], {}),
        ([1, 2, 3], "append", [4], {}),
        ({"a": 1, "b": 2}, "update", [{"c": 3}], {}),
        (set({1, 2, 3, 3}), "add", [5], {}),
        (set({1, 2, 3, 3}), "clear", [], {}),
    ],
)
def test_actionobject_syft_make_action(worker, testcase):
    orig_obj, op, args, kwargs = testcase

    obj = helper_make_action_obj(orig_obj)
    args, kwargs = helper_make_action_args(*args, **kwargs)
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
        (int(1), "__add__", [1], {}),
        (float(1.2), "__add__", [1], {}),
        (True, "__and__", [False], {}),
        ((1, 1, 3), "count", [1], {}),
        ([1, 2, 1], "count", [1], {}),
        ([1, 2, 3], "append", [4], {}),
        ({"a": 1, "b": 2}, "update", [{"c": 3}], {}),
        (set({1, 2, 3, 3}), "add", [5], {}),
        (set({1, 2, 3, 3}), "clear", [], {}),
    ],
)
def test_actionobject_syft_make_method_action(worker, testcase):
    orig_obj, op, args, kwargs = testcase

    obj = helper_make_action_obj(orig_obj)
    args, kwargs = helper_make_action_args(*args, **kwargs)
    obj_pointer, args_pointers, kwargs_pointers = helper_make_action_pointers(
        worker, obj, *args, **kwargs
    )

    action = obj.syft_make_method_action(op, args=args_pointers, kwargs=kwargs_pointers)

    assert action.full_path.endswith("." + op)


@pytest.mark.parametrize(
    "testcase",
    [
        # (object, operation, *args, **kwargs)
        ("abc", "capitalize", [], {}),
        ("abc", "find", ["b"], {}),
        (int(1), "__add__", [1], {}),
        (float(1.2), "__add__", [1], {}),
        (True, "__and__", [False], {}),
        ((1, 1, 3), "count", [1], {}),
        ([1, 2, 1], "count", [1], {}),
        ([1, 2, 3], "append", [4], {}),
        ({"a": 1, "b": 2}, "update", [{"c": 3}], {}),
        (set({1, 2, 3, 3}), "add", [5], {}),
        (set({1, 2, 3, 3}), "clear", [], {}),
    ],
)
def test_actionobject_syft_make_remote_method_action(worker, testcase):
    orig_obj, op, args, kwargs = testcase

    obj = helper_make_action_obj(orig_obj)
    args, kwargs = helper_make_action_args(*args, **kwargs)
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
        int(1),
        float(1.2),
        True,
        (1, 1, 3),
        [1, 2, 1],
        {"a": 1, "b": 2},
        set({1, 2, 3, 3}),
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
        int(1),
        float(1.2),
        True,
        (1, 1, 3),
        [1, 2, 1],
        {"a": 1, "b": 2},
        set({1, 2, 3, 3}),
    ],
)
def test_actionobject_syft_send_get(worker, testcase):
    root_domain_client = worker.root_client
    action_store = worker.get_service("actionservice").store

    orig_obj = testcase
    obj = helper_make_action_obj(orig_obj)

    assert len(action_store.data) == 0

    obj.send(root_domain_client)
    assert len(action_store.data) == 1

    retrieved = obj.get_from(root_domain_client)

    assert obj.syft_action_data == retrieved


@pytest.mark.parametrize(
    "testcase",
    [
        # (object, operation, *args, **kwargs, expected_result)
        (int(1), "__add__", [1], {}, 2),
        (float(1.2), "__add__", [1], {}, 2.2),
        (True, "__and__", [False], {}, False),
        ([1, 2, 3], "append", [4], {}, [1, 2, 3, 4]),
        ({"a": 1, "b": 2}, "update", [{"c": 3}], {}, {"a": 1, "b": 2, "c": 3}),
        (set({1, 2, 3, 3}), "add", [5], {}, set({1, 2, 3, 5})),
        (set({1, 2, 3, 3}), "clear", [], {}, set({})),
    ],
)
def test_actionobject_syft_execute_hooks_inplace(worker, testcase):
    client = worker.root_client
    orig_obj, op, args, kwargs, expected = testcase

    obj = helper_make_action_obj(orig_obj)
    args, kwargs = helper_make_action_args(*args, **kwargs)

    obj_pointer, args_pointers, kwargs_pointers = helper_make_action_pointers(
        worker, obj, *args, **kwargs
    )
    obj_pointer.syft_point_to(client.id)

    context = PreHookContext(obj=obj_pointer, op_name=op)

    context, result_args, result_kwargs = obj_pointer._syft_run_pre_hooks__(
        context, name=op, args=args_pointers, kwargs=kwargs_pointers
    )
    assert context.result_id is not None

    context.obj.syft_node_uid = Action.make_id(None)
    result = obj_pointer._syft_run_post_hooks__(context, name=op, result=obj_pointer)
    assert result.syft_node_uid == context.obj.syft_node_uid


def test_actionobject_syft_getattr_str():
    orig_obj = "a bC"

    obj = ActionObject.from_obj(orig_obj)
    obj = obj.ok()

    assert obj == orig_obj
    assert obj.capitalize() == "A bc"
    assert obj.casefold() == "a bc"
    assert obj.endswith("C") == True  # noqa
    assert obj.isascii() == True  # noqa
    assert obj.isdigit() == False  # noqa
    assert obj.upper() == "A BC"


def test_actionobject_syft_getattr_list():
    orig_obj = [3, 2, 1, 4]

    obj = ActionObject.from_obj(orig_obj)
    obj = obj.ok()

    assert obj == orig_obj
    assert len(obj) == 4
    assert obj.count(1) == 1
    assert obj.append(5) == [1, 2, 3, 4, 5]
    assert obj.sort() == [1, 2, 3, 4, 5]
    assert obj.clear() == []


def test_actionobject_syft_getattr_dict():
    orig_obj = {"a": 1, "b": 2}

    obj = ActionObject.from_obj(orig_obj)
    obj = obj.ok()

    assert obj == orig_obj
    assert obj.get("a") == 1
    assert obj.update({"c": 3}) == {"a": 1, "b": 2, "c": 3}
    assert obj.clear() == {}


def test_actionobject_syft_getattr_tuple():
    orig_obj = (1, 2, 3, 4, 4)

    obj = ActionObject.from_obj(orig_obj)
    obj = obj.ok()

    assert obj == orig_obj
    assert obj.count(4) == 2
    assert obj.index(2) == 1
    assert len(obj) == 5


def test_actionobject_syft_getattr_set():
    orig_obj = set({1, 2, 3, 4})

    obj = ActionObject.from_obj(orig_obj)
    obj = obj.ok()

    assert obj == orig_obj
    assert obj.add(4) == set({1, 2, 3, 4})
    assert obj.intersection(set({1, 2, 121})) == set({1, 2})
    assert len(obj) == 4


def test_actionobject_syft_getattr_bool():
    orig_obj = True

    obj = ActionObject.from_obj(orig_obj)
    obj = obj.ok()

    assert obj.__and__(False) == False  # noqa
    assert obj.__or__(False) == True  # noqa


def test_actionobject_syft_getattr_int():
    orig_obj = 5

    obj = ActionObject.from_obj(orig_obj)
    obj = obj.ok()

    assert obj == orig_obj
    assert str(obj) == "5"
    assert obj.__add__(1) == 6
    assert obj.__sub__(1) == 4
    assert obj.__mul__(2) == 10
    assert obj < 6
    assert obj <= 5
    assert obj > 4
    assert obj >= 4
    assert obj % 2 == 1
    assert bool(obj) == 1
    assert float(obj) == 5.0
