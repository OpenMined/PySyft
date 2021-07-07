# syft absolute
from syft.core.dataset.schema_entry import Entry


def test_creation():
    entry = Entry(entry_type=str, observability="public", values_set={1, 2, 3})
