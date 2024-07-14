# stdlib
from collections.abc import Callable
import inspect

# third party
import pytest

# syft absolute
from syft.types import transforms
from syft.types.syft_object import SyftBaseObject
from syft.types.syft_object_registry import SyftObjectRegistry
from syft.types.transforms import TransformContext
from syft.types.transforms import validate_klass_and_version


class MockObjectFromSyftBaseObj(SyftBaseObject):
    __canonical_name__ = "MockObjectFromSyftBaseObj"
    __version__ = 1

    value: int | None = None


class MockObjectToSyftBaseObj(SyftBaseObject):
    __canonical_name__ = "MockObjectToSyftBaseObj"
    __version__ = 1

    value: int | None = None


@pytest.mark.parametrize(
    "klass_from, klass_to",
    [
        (MockObjectFromSyftBaseObj, MockObjectToSyftBaseObj),
        (MockObjectFromSyftBaseObj.__canonical_name__, MockObjectToSyftBaseObj),
        (MockObjectFromSyftBaseObj, MockObjectToSyftBaseObj.__canonical_name__),
        (MockObjectFromSyftBaseObj, 2),
        (1, MockObjectToSyftBaseObj.__canonical_name__),
    ],
)
@pytest.mark.parametrize("version_from", [None, 1])
@pytest.mark.parametrize("version_to", [None, 1])
def test_validate_klass_and_version(
    klass_from,
    klass_to,
    version_from,
    version_to,
):
    if klass_from == 1 or klass_to == 2:
        with pytest.raises(NotImplementedError):
            validate_klass_and_version(
                klass_from,
                klass_to,
                version_from,
                version_to,
            )
    else:
        expected_result = (
            MockObjectFromSyftBaseObj.__canonical_name__,
            (
                version_from
                if isinstance(klass_from, str)
                else MockObjectFromSyftBaseObj.__version__
            ),
            MockObjectToSyftBaseObj.__canonical_name__,
            (
                version_to
                if isinstance(klass_to, str)
                else MockObjectToSyftBaseObj.__version__
            ),
        )
        result = validate_klass_and_version(
            klass_from, klass_to, version_from, version_to
        )

        assert result == expected_result


def test_generate_transform_wrapper(faker, monkeypatch, server_context):
    mock_value = faker.random_int()

    def mock_transform_method(context: TransformContext) -> TransformContext:
        context.output["value"] = mock_value
        return context

    resultant_wrapper = transforms.generate_transform_wrapper(
        klass_from=MockObjectFromSyftBaseObj,
        klass_to=MockObjectToSyftBaseObj,
        transforms=[mock_transform_method],
    )

    assert resultant_wrapper.__name__ == "wrapper"
    signature = inspect.signature(resultant_wrapper)
    assert signature.parameters["self"].annotation == MockObjectFromSyftBaseObj
    assert signature.return_annotation == MockObjectToSyftBaseObj

    output = resultant_wrapper(
        MockObjectFromSyftBaseObj(),
        server_context,
    )
    assert isinstance(output, MockObjectToSyftBaseObj)
    assert output.value == mock_value


def test_transform_method(monkeypatch):
    mock_klass_from_str, mock_version_from = (
        MockObjectFromSyftBaseObj.__canonical_name__,
        MockObjectFromSyftBaseObj.__version__,
    )

    mock_klass_to_str, mock_version_to = (
        MockObjectToSyftBaseObj.__canonical_name__,
        MockObjectToSyftBaseObj.__version__,
    )

    mock_syft_transform_registry = {}
    mapping_key = f"{mock_klass_from_str}_{mock_version_from}_x_{mock_klass_to_str}_{mock_version_to}"

    def mock_add_transform(
        klass_from: str,
        version_from: int,
        klass_to: str,
        version_to: int,
        method: Callable,
    ):
        mock_syft_transform_registry[mapping_key] = method

    def mock_validate_klass_and_version(
        klass_from,
        version_from,
        klass_to,
        version_to,
    ):
        return (
            mock_klass_from_str,
            mock_version_from,
            mock_klass_to_str,
            mock_version_to,
        )

    def mock_method():
        return True

    def mock_wrapper():
        return mock_method

    monkeypatch.setattr(
        transforms,
        "validate_klass_and_version",
        mock_validate_klass_and_version,
    )

    monkeypatch.setattr(
        SyftObjectRegistry,
        "add_transform",
        mock_add_transform,
    )

    result = transforms.transform_method(
        klass_from=MockObjectFromSyftBaseObj,
        klass_to=MockObjectToSyftBaseObj,
    )

    assert result.__name__ == "decorator"
    assert result(mock_method) == mock_method
    assert mapping_key in mock_syft_transform_registry
    assert mock_syft_transform_registry[mapping_key] == mock_method
    assert mock_syft_transform_registry[mapping_key]() == mock_method()

    def mock_generate_transform_wrapper(
        klass_from: type, klass_to: type, transforms: list[Callable]
    ):
        return mock_wrapper

    monkeypatch.setattr(
        transforms,
        "generate_transform_wrapper",
        mock_generate_transform_wrapper,
    )

    result = transforms.transform(
        klass_from=MockObjectFromSyftBaseObj,
        klass_to=MockObjectToSyftBaseObj,
    )

    assert result.__name__ == "decorator"
    resultant_func = result(mock_method)
    assert resultant_func == mock_method
    assert resultant_func() == mock_method()
    assert mapping_key in mock_syft_transform_registry
    assert mock_syft_transform_registry[mapping_key] == mock_wrapper
