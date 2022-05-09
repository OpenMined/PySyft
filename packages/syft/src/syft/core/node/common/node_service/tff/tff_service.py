# stdlib
from typing import List
from typing import Optional
from typing import Type
from unittest import result
# third party
from nacl.signing import VerifyKey
import tensorflow_federated as tff
# import torch as th
# relative
from ......util import traceback_and_raise
from ....abstract.node import AbstractNode
from ......core.common.uid import UID

from ..auth import service_auth
from ..node_service import ImmediateNodeServiceWithReply
from ..node_service import EventualNodeServiceWithoutReply
from .tff_messages import TFFMessage
from .tff_messages import TFFReplyMessage
from .data_backend import TestDataBackend, PySyftDataBackend
from tensorflow_federated.proto.v0 import computation_pb2 as pb
import tensorflow as tf
from absl.testing import absltest
import asyncio
from tensorflow_federated.python.core.impl.executors import eager_tf_executor
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.api import computations
# from tensorflow_federated.python.core.impl.executors import executor_test_utils
from tensorflow_federated.python.core.impl.executors import executor_stacks
from tensorflow_federated.python.core.impl.context_stack import set_default_context
# from tensorflow_federated.python.core.impl.context_stack import context_stack_test_utils
from tensorflow_federated.python.core.backends.native import compiler
from tensorflow_federated.python.core.impl.execution_contexts import async_execution_context

# from pybind11_abseil import status
import torch as th

def custom_data_descriptor(uris, data_type):
    num_clients = len(uris)
    data_type_proto = tff.framework.serialize_type(data_type)
    arguments = [pb.Computation(type=data_type_proto, data=pb.Data(uri=uri)) for uri in uris]
    return tff.framework.DataDescriptor(None, 
                                        arguments, 
                                        # tf.int32
                                        # data_type,
                                        # data_type_proto,
                                        tff.FederatedType(data_type, tff.CLIENTS), 
                                        # 1
                                        num_clients
                                        )

def create_keras_model():
  return tf.keras.models.Sequential([
      tf.keras.layers.InputLayer(input_shape=(784,)),
      tf.keras.layers.Dense(10, kernel_initializer='zeros'),
      tf.keras.layers.Softmax(),
  ])
  
# def model_fn():
#   keras_model = create_keras_model()
#   return tff.learning.from_keras_model(
#       keras_model,
#       input_spec=preprocessed_example_dataset.element_spec,
#       loss=tf.keras.losses.SparseCategoricalCrossentropy(),
#       metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

def get_ctx(data_backend):
  def ex_fn(
      device: tf.config.LogicalDevice) -> tff.framework.DataExecutor:
    return tff.framework.DataExecutor(
        tff.framework.EagerTFExecutor(device),
        data_backend=data_backend)
  factory = tff.framework.local_executor_factory(leaf_executor_fn=ex_fn)
  return async_execution_context.AsyncExecutionContext(executor_fn=factory)

async def test_data_descriptor(node):
    uris = [key.to_string() for key in node.store.keys()]
    data_type = tff.TensorType(dtype=tf.int32, shape=[3])
    # data_type = computation_types.TensorType(tff.types.SequenceType(tf.int32))
    data_desc = custom_data_descriptor(uris, data_type)

    # @tff.federated_computation(tff.types.FederatedType(tf.int32, tff.CLIENTS))
    # def foo(x):
    #     return tff.federated_sum(x)
    
    @tff.federated_computation(tff.types.FederatedType(tff.TensorType(dtype=tf.int32, shape=[3]), tff.CLIENTS))
    def foo(x):
        @tff.tf_computation(tff.TensorType(dtype=tf.int32, shape=[3]))
        def local_sum(nums):
            return tf.math.reduce_sum(nums)
        return tff.federated_sum(tff.federated_map(local_sum, x))

    backend = PySyftDataBackend(node.store)
    context = get_ctx(backend)
    tff.framework.set_default_context(context)

    result = await foo(data_desc)
    print(result)


async def train_model(store):
    backend = PySyftDataBackend(store)
    ex = tff.framework.DataExecutor(
        tff.framework.EagerTFExecutor(),
        backend
    )
    
    @computations.tf_computation()
    def foo(x):
        return x * 20.0

    # with executor_test_utils.install_executor(executor_stacks.local_executor_factory()):
    #     result = foo(ds)

async def test_materialize(store, uri, type_signature, expected_value):
    backend = PySyftDataBackend(store)
    value = await backend.materialize(
            pb.Data(uri=uri), tff.to_type(type_signature)
        )
    print(value)
    assert value == expected_value

async def test_raises_no_uri(store):
    backend = PySyftDataBackend(store)
    await backend.materialize(pb.Data(), tff.to_type(()))

async def test_raises_unknown_uri(store):
    backend = PySyftDataBackend(store)
    await backend.materialize(pb.Data(uri='unknown_uri'), tff.to_type(()))

class TFFService(ImmediateNodeServiceWithReply):
    @staticmethod
    @service_auth(guests_welcome=True)
    def process(
        node: AbstractNode, msg: TFFMessage, verify_key: Optional[VerifyKey] = None
    ) -> TFFReplyMessage:
        if verify_key is None:
            traceback_and_raise("Can't process TFFService with no verification key.")
        tff.backends.native.execution_contexts.set_local_async_python_execution_context(reference_resolving_clients=True)
        asyncio.ensure_future(test_data_descriptor(node))
        # tensor = node.store.get(node.store.keys()[0])
        # print(dir(tensor))
        # print(tensor.data.numpy())
        # print(dir(tensor.data))
        result = msg.payload.run(node=node, verify_key=verify_key)
        return TFFReplyMessage(payload=result, address=msg.reply_to)

    @staticmethod
    def message_handler_types() -> List[Type[TFFMessage]]:
        return [TFFMessage]
