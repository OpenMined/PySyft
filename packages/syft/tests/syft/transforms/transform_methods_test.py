# stdlib
from collections.abc import Callable
from dataclasses import dataclass
from types import FunctionType

# third party
from pydantic import EmailStr
from pydantic_core import PydanticCustomError
import pytest

# syft absolute
from syft.types.transforms import NotNone
from syft.types.transforms import TransformContext
from syft.types.transforms import add_credentials_for_key
from syft.types.transforms import add_server_uid_for_key
from syft.types.transforms import drop
from syft.types.transforms import generate_id
from syft.types.transforms import geteitherattr
from syft.types.transforms import keep
from syft.types.transforms import make_set_default
from syft.types.transforms import rename
from syft.types.transforms import validate_email
from syft.types.transforms import validate_url
from syft.types.uid import UID


@pytest.mark.parametrize(
    "syft_obj, context",
    [
        ("admin_user", "authed_context"),
        ("guest_user", "server_context"),
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

    if hasattr(context, "server"):
        assert transform_context.server == context.server

    server_context = transform_context.to_server_context()

    assert server_context == context


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
def test_make_set_default(faker, key, value, server_context):
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
        obj=mock_obj, context=server_context
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


def test_drop(faker, server_context):
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
        obj=mock_obj, context=server_context
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


def test_keep(faker, server_context):
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
        obj=mock_obj, context=server_context
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


def test_rename(faker, server_context):
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

    key_to_rename = "name"
    new_name_for_key = "full_name"

    result = rename(old_key=key_to_rename, new_key=new_name_for_key)
    assert isinstance(result, FunctionType)
    assert isinstance(result, Callable)

    transform_context = TransformContext.from_context(
        obj=mock_obj, context=server_context
    )

    mock_obj_dict = dict(mock_obj)
    expected_output = mock_obj_dict
    expected_output[new_name_for_key] = expected_output[key_to_rename]
    del expected_output[key_to_rename]

    resultant_context = result(transform_context)
    assert isinstance(resultant_context, TransformContext)

    assert resultant_context.obj == mock_obj
    assert resultant_context.output == expected_output


def test_generate_id(faker, server_context):
    @dataclass
    class MockObject:
        name: str
        age: int
        company: str

        def __iter__(self):
            yield from self.__dict__.items()

    @dataclass
    class MockObjectWithId:
        id: UID | None
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

    transform_context = TransformContext.from_context(
        obj=mock_obj, context=server_context
    )

    result = generate_id(context=transform_context)
    assert isinstance(result, TransformContext)
    assert "id" in result.output
    assert isinstance(result.output["id"], UID)

    mock_obj = MockObjectWithId(
        id=None,
        name=faker.name(),
        age=faker.random_int(),
        company=faker.company(),
    )

    transform_context = TransformContext.from_context(
        obj=mock_obj, context=server_context
    )

    result = generate_id(context=transform_context)
    assert isinstance(result, TransformContext)
    assert "id" in result.output
    assert isinstance(result.output["id"], UID)

    uid = UID()
    mock_obj = MockObjectWithId(
        id=uid,
        name=faker.name(),
        age=faker.random_int(),
        company=faker.company(),
    )

    transform_context = TransformContext.from_context(
        obj=mock_obj, context=server_context
    )

    result = generate_id(context=transform_context)
    assert isinstance(result, TransformContext)
    assert "id" in result.output
    assert result.output["id"] == uid


def test_add_credentials_for_key(faker, authed_context):
    @dataclass
    class MockObject:
        name: str

        def __iter__(self):
            yield from self.__dict__.items()

    mock_obj = MockObject(
        name=faker.name(),
    )

    transform_context = TransformContext.from_context(
        obj=mock_obj, context=authed_context
    )

    key = "random_key"

    result_func = add_credentials_for_key(key=key)
    assert isinstance(result_func, FunctionType)
    result = result_func(context=transform_context)
    assert isinstance(result, TransformContext)
    assert key in result.output
    assert result.output[key] == authed_context.credentials


def test_add_server_uid_for_key(faker, server_context):
    @dataclass
    class MockObject:
        name: str

        def __iter__(self):
            yield from self.__dict__.items()

    mock_obj = MockObject(
        name=faker.name(),
    )

    transform_context = TransformContext.from_context(
        obj=mock_obj, context=server_context
    )

    key = "random_uid_key"

    result_func = add_server_uid_for_key(key=key)
    assert isinstance(result_func, FunctionType)
    result = result_func(context=transform_context)
    assert isinstance(result, TransformContext)
    assert key in result.output
    assert result.output[key] == server_context.server.id


def test_validate_url(faker, server_context):
    @dataclass
    class MockObject:
        url: str | None

        def __iter__(self):
            yield from self.__dict__.items()

    mock_obj = MockObject(url=None)

    transform_context = TransformContext.from_context(
        obj=mock_obj, context=server_context
    )

    # no change in context if url is None
    result = validate_url(transform_context)
    assert isinstance(result, TransformContext)
    assert result == transform_context

    url = faker.url()[:-1]
    url_with_port = f"{url}:{faker.port_number()}"
    mock_obj = MockObject(url=url_with_port)

    transform_context = TransformContext.from_context(
        obj=mock_obj, context=server_context
    )

    result = validate_url(transform_context)
    assert isinstance(result, TransformContext)
    assert result.output["url"] == url


def test_validate_email(faker, server_context):
    @dataclass
    class MockObject:
        email: str

        def __iter__(self):
            yield from self.__dict__.items()

    mock_obj = MockObject(email=None)
    transform_context = TransformContext.from_context(
        obj=mock_obj, context=server_context
    )
    result = validate_email(transform_context)
    assert isinstance(result, TransformContext)
    assert result == transform_context

    mock_obj = MockObject(email=faker.email())
    transform_context = TransformContext.from_context(
        obj=mock_obj, context=server_context
    )
    result = validate_email(transform_context)
    assert isinstance(result, TransformContext)
    assert EmailStr._validate(result.output["email"])
    assert result.output["email"] == mock_obj.email

    mock_obj = MockObject(email=faker.name())
    transform_context = TransformContext.from_context(
        obj=mock_obj, context=server_context
    )

    with pytest.raises(PydanticCustomError):
        validate_email(transform_context)
