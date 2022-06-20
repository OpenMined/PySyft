# stdlib
import collections

# third party
import tensorflow as tf
import tensorflow_federated as tff
from tensorflow_federated.proto.v0 import computation_pb2 as pb
from tensorflow_federated.python.core.impl.types import computation_types

# relative
from ......core.common.uid import UID


class MedNISTBackend(tff.framework.DataBackend):
    
    def __init__(self, node):
        self._node = node

    async def materialize(self, data: pb.Data, type_spec: computation_types.Type):
        dataset_objs = self._node.datasets.get(data.uri)[1]
        images = self._node.store.get(dataset_objs[0].obj).data.child.child.child
        labels = self._node.store.get(dataset_objs[1].obj).data.child.child.child

        dataset = tf.data.Dataset.from_tensor_slices((images, labels))

        def preprocess(images, labels):
            return collections.OrderedDict(
                x = tf.reshape(images, [-1, 64*64]),
                y = tf.reshape(labels, [-1, 1]) // (2**16),
            )
            
        return dataset.map(preprocess)

class PySyftDataBackend(tff.framework.DataBackend):
    
    def __init__(self, store):
        self._store = store

    async def materialize(self, data: pb.Data, type_spec: computation_types.Type):
        # print(data.uri)
        uid = UID.from_string(data.uri)
        # print(self._store.get(uid).data.numpy())
        return self._store.get(uid).data.child.child.child
    
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
