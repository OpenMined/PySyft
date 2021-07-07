# third party
import pytest

# syft absolute
from syft.core.dataset.dataset import Dataset
from syft.core.dataset.schema import Schema
from syft.core.dataset.schema_entry import Entry


@pytest.fixture(scope="module")
def typed_schema():
    schema = Schema()
    schema.Name = Entry(entry_type=str, observability="public")
    schema.Age = Entry(entry_type=int, observability="private")
    schema.compile(pa_scheme=True)
    return schema


@pytest.fixture(scope="module")
def untyped_schema():
    schema = Schema()
    schema.Name = Entry(observability="public")
    schema.Age = Entry(observability="private")
    schema.compile(pa_scheme=False)
    return schema


def test_pylist_typed(typed_schema):
    data = [["Tudor", "Madhava", "Andrew", "George"], [100, 101, 102, 103]]

    dataset = Dataset(typed_schema)

    dataset.add_pylist(data)
    dataset.add_pylist(data)


def test_pylist_untyped(untyped_schema):
    data = [["Tudor", "Madhava", "Andrew", "George"], [100, 101, 102, 103]]

    dataset = Dataset(untyped_schema)
    dataset.add_pylist(data)
    dataset.add_pylist(data)


@pytest.mark.xfail
def test_pylist_typed_wrong_dtype(typed_schema):
    data = [["Tudor", "Madhava", "Andrew", "George", 100], [100, 101, 102, 103, 104]]

    dataset = Dataset(typed_schema)
    dataset.add_pylist(data)


@pytest.mark.xfail
def test_pylist_typed_wrong_dtype(untyped_schema):
    data = [["Tudor", "Madhava", "Andrew", "George"], [100, 101, 102, 103]]

    dataset = Dataset(untyped_schema)
    dataset.add_pylist(data)
