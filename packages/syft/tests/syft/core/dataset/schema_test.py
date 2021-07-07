# syft absolute
from syft.core.dataset.schema import Schema
from syft.core.dataset.schema_entry import Entry


def test_schema_creation():
    schema = Schema()
    schema.Name = Entry(entry_type=str, observability="public")
    schema.Age = Entry(entry_type=int, observability="private")
    schema.compile()
