# stdlib
from dataclasses import dataclass
from types import FunctionType
from typing import Callable

# third party
import pytest

# syft absolute
from syft.core.node.new.transforms import NotNone
from syft.core.node.new.transforms import TransformContext
from syft.core.node.new.transforms import drop
from syft.core.node.new.transforms import geteitherattr
from syft.core.node.new.transforms import keep
from syft.core.node.new.transforms import make_set_default


@pytest.mark.parametrize(
    "syft_obj, context",
    [
        ("admin_user", "authed_context"),
        ("guest_user", "node_context"),
    ],
)
def test_transformcontext(syft_obj, context, request):
    syft_obj = request.getfixturevalue(syft_obj)
    context = request.getfixturevalue(context)

    transform_context = TransformContext.from_context(
        obj=syft_obj,
        context=context,
    )

    assert isinstance(transform_context, TransformContext)

    if hasattr(context, "credentials"):
        assert transform_context.credentials == context.credentials

    if hasattr(context, "node"):
        assert transform_context.node == context.node

    node_context = transform_context.to_node_context()

    assert node_context == context


@pytest.mark.parametrize(
    "output, key",
    [
        ({"my_key": "value"}, "not_my_key"),
        ({"my_key": "value"}, "my_key"),
    ],
)
@pytest.mark.parametrize("default", [NotNone, "DefaultValue"])
def test_geteitherattr(output, key, default):
    @dataclass
    class MockObject:
        my_key = "MockValue"

    _self = MockObject()

    try:
        value = geteitherattr(_self=_self, output=output, key=key, default=default)
        if key == "not_my_key":
            if default == NotNone:
                assert value == _self.my_key
            else:
                assert value == default
        else:
            value = geteitherattr(_self=_self, output=output, key=key, default=default)

            assert value == output["my_key"]
    except Exception as error:
        assert isinstance(error, AttributeError)


@pytest.mark.parametrize(
    "key, value",
    [
        ("dict_key", "dict_value"),
        ("obj_key", "obj_value"),
        ("no_key", "no_value"),
    ],
)
def test_make_set_default(faker, key, value, node_context):
    result = make_set_default(key, value)
    assert isinstance(result, FunctionType)
    assert isinstance(result, Callable)

    @dataclass
    class MockObject:
        obj_key: str

        def __iter__(self):
            yield from self.__dict__.items()

    mock_obj = MockObject(obj_key=faker.name())

    transform_context = TransformContext.from_context(
        obj=mock_obj, context=node_context
    )

    resultant_context = result(transform_context)

    assert isinstance(resultant_context, TransformContext)

    if key == "no_key":
        assert transform_context == resultant_context
    elif key == "dict_key":
        assert key in resultant_context.output
        assert resultant_context.output[key] == "dict_value"
    elif key == "obj_key":
        assert key in resultant_context.output
        assert resultant_context.output[key] == mock_obj.obj_key


def test_drop(faker, node_context):
    @dataclass
    class MockObject:
        name: str
        age: int
        company: str

        def __iter__(self):
            yield from self.__dict__.items()

    mock_obj = MockObject(
        name=faker.name(),
        age=faker.random_int(),
        company=faker.company(),
    )

    list_keys = ["company", "address"]

    result = drop(list_keys)
    assert isinstance(result, FunctionType)
    assert isinstance(result, Callable)

    transform_context = TransformContext.from_context(
        obj=mock_obj, context=node_context
    )

    expected_output = dict(mock_obj).copy()

    for key in list_keys:
        expected_output.pop(key, None)

    resultant_context = result(transform_context)

    assert isinstance(resultant_context, TransformContext)

    for key in list_keys:
        assert key not in resultant_context.output
    assert resultant_context.obj == mock_obj
    assert resultant_context.output == expected_output


def test_keep(faker, node_context):
    @dataclass
    class MockObject:
        name: str
        age: int
        company: str

        def __iter__(self):
            yield from self.__dict__.items()

    mock_obj = MockObject(
        name=faker.name(),
        age=faker.random_int(),
        company=faker.company(),
    )

    list_keys = ["company", "invalid_key"]

    result = keep(list_keys)
    assert isinstance(result, FunctionType)
    assert isinstance(result, Callable)

    transform_context = TransformContext.from_context(
        obj=mock_obj, context=node_context
    )

    mock_obj_dict = dict(mock_obj)
    expected_output = {}

    for key in list_keys:
        if key in mock_obj_dict:
            expected_output[key] = mock_obj_dict[key]
        else:
            expected_output[key] = None

    resultant_context = result(transform_context)
    assert isinstance(resultant_context, TransformContext)

    assert resultant_context.obj == mock_obj
    assert resultant_context.output == expected_output
