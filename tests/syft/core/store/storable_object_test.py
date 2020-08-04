from syft.core.store import StorableObject
from syft.core.common import UID
from syft.core.common.serde import _deserialize

def test_create_obj():
    key = UID()
    data = UID()
    description = "This is a dummy test"
    tags = ["dummy", "test"]
    StorableObject(key=key, data=data, description=description, tags=tags)
