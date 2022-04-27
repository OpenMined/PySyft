import tensorflow_federated as tff
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.proto.v0 import computation_pb2 as pb
from ......core.common.uid import UID


class PySyftDataBackend(tff.framework.DataBackend):
    
    def __init__(self, store):
        self._store = store

    async def materialize(self, data: pb.Data, type_spec: computation_types.Type):
        print(data.uri)
        uid = UID.from_string(data.uri)
        return self._store.get(uid)
    
class TestDataBackend(tff.framework.DataBackend):

  def __init__(self, uri, value, type_spec):
    self._uri = uri
    self._value = value
    self._type_spec = computation_types.to_type(type_spec)

  async def materialize(self, data, type_spec):
    assert isinstance(data, pb.Data)
    assert isinstance(type_spec, computation_types.Type)
    assert data.uri == self._uri
    assert str(type_spec) == str(self._type_spec)
    return self._value
