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

# from pybind11_abseil import status
import torch as th

def custom_data_descriptor(uris, data_type):
    num_clients = len(uris)
    data_type_proto = tff.framework.serialize_type(data_type)
    arguments = [pb.Data(uri=uri, type_data=data_type_proto) for uri in uris]
    return tff.framework.DataDescriptor(None, arguments, tff.FederatedType(data_type, tff.CLIENTS), num_clients)

def test_data_descriptor(node):
    print('ce plm')
    uris = [key.to_string() for key in node.store.keys()]
    data_desc = custom_data_descriptor(uris, ())
    print('ce plm')

    @computations.tf_computation()
    def foo(x):
        return x * 20.0

    # with executor_test_utils.install_executor(executor_stacks.local_executor_factory()):
    #     result = foo(data_desc)
    #     print(result)


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


async def support(store):
    backend = PySyftDataBackend(store)
    uris = [key.to_string() for key in store.keys()]
    num_clients = len(uris)
    data_type = ()
    data_type_proto = tff.framework.serialize_type(data_type)
    arguments = []
    for uri in uris:
        arguments.append(pb.Data(uri=uri, type=data_type_proto))
    data_descriptor = tff.framework.DataDescriptor(None, arguments, tff.FederatedType(data_type, tff.CLIENTS), num_clients)
    # data = await backend.materialize(pb.Data(uri=key.to_string()), ())
    # string = await tff.federated_computation(lambda: 'Hello World')()
    # print("in async context", string)
    # return string

async def test_custom_backend(store):
    pass
    # await test_materialize(store, )
    # await test_raises_no_uri(store)
    # await test_raises_unknown_uri(store)

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
        print('plm')
        print(node.store.keys())
        # import nest_asyncio
        # import uvloop
        # nest_asyncio.apply()
        # asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
        # print(asyncio.get_event_loop_policy())
        # print(dir(asyncio.get_event_loop_policy()))
        tff.backends.native.execution_contexts.set_local_async_python_execution_context(reference_resolving_clients=True)
        # loop = asyncio.get_event_loop()
        
        # loop.run_until_complete(asyncio.wait([loop.create_task(support())]))
        # loop.close()
        # asyncio.ensure_future(support(node.store))
        test_data_descriptor(node)
        # asyncio.ensure_future(test_custom_backend())
        # test_custom_backend_materize()
        # s = 0
        # for x in range(100000000):
        #     s += x
        # print(s)
        # loop = asyncio.get_running_loop()
        # loop.run_until_complete(lambda: await tff.federated_computation(lambda: 'Hello World')())
        # print(res)

        result = msg.payload.run(node=node, verify_key=verify_key)
        return TFFReplyMessage(payload=result, address=msg.reply_to)

    @staticmethod
    def message_handler_types() -> List[Type[TFFMessage]]:
        return [TFFMessage]
