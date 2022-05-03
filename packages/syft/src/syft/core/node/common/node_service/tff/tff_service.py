# stdlib
from typing import List
from typing import Optional
from typing import Type
from unittest import result
# third party
from nacl.signing import VerifyKey
import tensorflow_federated as tff
from torch import eq

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
                                        data_type_proto,
                                        # tff.FederatedType(data_type, tff.CLIENTS), 
                                        # 1
                                        num_clients
                                        )

async def test_data_descriptor(node):
    uris = [key.to_string() for key in node.store.keys()]
    data_type = computation_types.TensorType(tf.int32)
    data_desc = custom_data_descriptor(uris, data_type)

    @computations.tf_computation(tf.int32)
    def foo(x):
        return x + 1

    backend = PySyftDataBackend(node.store)
    ex = tff.framework.DataExecutor(
        tff.framework.EagerTFExecutor(),
        backend
    )
    ex_fn = lambda device: ex
    factory = executor_stacks.local_executor_factory(
        leaf_executor_fn=ex_fn
        )
    # context = context_stack_test_utils.TestContext(factory)
    context = async_execution_context.AsyncExecutionContext(
        executor_fn=factory,
        # compiler_fn=compiler.transform_to_native_form,
    )
    set_default_context.set_default_context(context)

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
        result = msg.payload.run(node=node, verify_key=verify_key)
        return TFFReplyMessage(payload=result, address=msg.reply_to)

    @staticmethod
    def message_handler_types() -> List[Type[TFFMessage]]:
        return [TFFMessage]
