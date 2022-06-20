# stdlib
import asyncio
import collections
from ctypes import Structure
import functools
import logging
from optparse import Option
import os
from typing import Any
from typing import List
from typing import Optional
from typing import Tuple
from typing import Type
from unittest import result
from xml.dom.minidom import Element

# third party
from absl.testing import absltest
from matplotlib import backend_bases
from nacl.signing import VerifyKey
import numpy as np
from sympy import init_printing
from sympy import total_degree
import tensorflow as tf
import tensorflow_federated as tff
from tensorflow_federated.proto.v0 import computation_pb2 as pb
from tensorflow_federated.python.core.api import computations

# from tensorflow_federated.python.core.impl.context_stack import context_stack_test_utils
from tensorflow_federated.python.core.backends.native import compiler
from tensorflow_federated.python.core.impl.context_stack import set_default_context
from tensorflow_federated.python.core.impl.execution_contexts import (
    async_execution_context,
)

# from tensorflow_federated.python.core.impl.executors import executor_test_utils
from tensorflow_federated.python.core.impl.executors import eager_tf_executor
from tensorflow_federated.python.core.impl.executors import executor_stacks
from tensorflow_federated.python.core.impl.types import computation_types

# from pybind11_abseil import status
import torch as th

# syft absolute
from syft.core.tensor.tensor import Tensor

# relative
from ......core.common.uid import UID

# import torch as th
from ......util import traceback_and_raise
from ....abstract.node import AbstractNode
from ..auth import service_auth
from ..node_service import EventualNodeServiceWithoutReply
from ..node_service import ImmediateNodeServiceWithReply
from .data_backend import MedNISTBackend
from .data_backend import PySyftDataBackend
from .data_backend import TestDataBackend
from .tff_messages import TFFMessage
from .tff_messages import TFFReplyMessage


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
      tf.keras.layers.InputLayer(input_shape=(64*64,)),
      tf.keras.layers.Dense(6, kernel_initializer='zeros'),
      tf.keras.layers.Softmax(),
  ])
  
def model_fn(input_spec):
  keras_model = create_keras_model()
  return tff.learning.from_keras_model(
      keras_model,
      input_spec=input_spec,
      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

async def test_syft_tensor(node):
    backend = PySyftDataBackend(node.store)
    uris = [key.to_string() for key in node.store.keys() if type(node.store.get(key).data) == Tensor]
    for i, uri in enumerate(uris):
        data = await backend.materialize(pb.Data(uri=uri), tff.to_type(()))
        print("shape:", i, data.shape)

def get_ctx(data_backend):
  def ex_fn(
      device: tf.config.LogicalDevice) -> tff.framework.DataExecutor:
    return tff.framework.DataExecutor(
        tff.framework.EagerTFExecutor(device),
        data_backend=data_backend)
  factory = tff.framework.local_executor_factory(leaf_executor_fn=ex_fn)
  return async_execution_context.AsyncExecutionContext(executor_fn=factory)

async def test_data_descriptor(node):
    uris = [key.to_string() for key in node.store.keys() if type(node.store.get(key).data) == Tensor]
    uris = [uris[2]]
    data_type = tff.TensorType(dtype=tf.int32, shape=[58954,64,64])
    # data_type = computation_types.TensorType(tff.types.SequenceType(tf.int32))
    data_desc = custom_data_descriptor(uris, data_type)

    # @tff.federated_computation(tff.types.FederatedType(tf.int32, tff.CLIENTS))
    # def foo(x):
    #     return tff.federated_sum(x)
    
    @tff.federated_computation(tff.types.FederatedType(tff.TensorType(dtype=tf.int32, shape=[58954, 64, 64]), tff.CLIENTS))
    def foo(x):
        @tff.tf_computation(tff.TensorType(dtype=tf.int32, shape=[58954, 64, 64]))
        def local_sum(nums):
            return tf.math.reduce_sum(nums)
        return tff.federated_sum(tff.federated_map(local_sum, x))

    backend = PySyftDataBackend(node.store)
    context = get_ctx(backend)
    tff.framework.set_default_context(context)

    result = await foo(data_desc)
    print("WUUUT", result)

async def test_train_model(node):
    uris = ['03824e77-3d62-426a-bdea-e836ba210c2b']
    # element_type = tff.types.StructWithPythonType(
    #     tff.TensorType(dtype=tf.int32, shape=[58954,64,64]),  
    #     container_type=collections.OrderedDict)
    input_spec = collections.OrderedDict(
            x=tf.TensorSpec(shape=(1,64*64), dtype=tf.int32, name=None),
            y=tf.TensorSpec(shape=(1,1), dtype=tf.int32, name=None),
    )
    
    # element_type = tff.TensorType(dtype=tf.int32, shape=[58954,64,64])
    element_type = tff.types.StructWithPythonType(
        input_spec,
        container_type=collections.OrderedDict)
    data_type = tff.types.SequenceType(element_type)
    # data_type = element_type

    data_descriptor = custom_data_descriptor(uris, data_type)
    backend = MedNISTBackend(node)
    
    context = get_ctx(backend)
    tff.framework.set_default_context(context)

    iterative_process = tff.learning.build_federated_averaging_process(
    lambda : model_fn(input_spec),
    client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.02),
    server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0))

    state = await iterative_process.initialize()
    
    logging.info('Training Ready')

    state, metrics = await iterative_process.next(state, data_descriptor)
    logging.info('round 1, metrics={}'.format(metrics))
    
    NUM_ROUNDS = 5
    for round_num in range(2, NUM_ROUNDS):
        state, metrics = await iterative_process.next(state, data_descriptor)
        logging.info('round {:2d}, metrics={}'.format(round_num, metrics))

    logging.info('Training Succesful')
    # assert False

async def test_materialize(node, uri, type_signature, expected_value):
    logger = logging.getLogger('tensorflow_federated')
    logger.setLevel(level=logging.NOTSET)
    logger = logging.getLogger()
    logger.setLevel(level=logging.NOTSET)
    backend = MedNISTBackend(node)
    value = await backend.materialize(
            pb.Data(uri=uri), tff.to_type(type_signature)
        )
    # print(value['x'].shape)
    print(dir(value))
    # shard_dataset = value.shard(num_shards=1, index=0)
    # print(dir(shard_dataset))
    # iter = value.make_one_shot_iterator()
    # for x in value:
    #     print(x['y'])

    assert False
    # assert value == expected_value

async def test_raises_no_uri(store):
    backend = PySyftDataBackend(store)
    await backend.materialize(pb.Data(), tff.to_type(()))

async def test_raises_unknown_uri(store):
    backend = PySyftDataBackend(store)
    await backend.materialize(pb.Data(uri='unknown_uri'), tff.to_type(()))


async def tff_train_federated(
    initialize: tff.Computation,
    train: tff.Computation,
    train_data_source: tff.program.FederatedDataSource,
    evaluation: tff.Computation,
    evaluation_data_source: tff.program.FederatedDataSource,
    total_rounds: int,
    number_of_clients: int,
    train_output_managers: Optional[List[tff.program.ReleaseManager]] = None,
    evaluation_output_managers: Optional[List[tff.program.ReleaseManager]] = None,
    model_output_manager: Optional[tff.program.ReleaseManager] = None,
    program_state_manager: Optional[tff.program.ProgramStateManager] = None
):
    tff.program.check_in_federated_context()
    logging.info('Running program logic')

    if program_state_manager is not None:
        structure = initialize()
        program_state, version = await program_state_manager.load_latest(structure)
    else:
        program_state = None

    if program_state is not None:
        logging.info('Loaded program state at version %d', version)

        state, start_round = program_state

    else:
        logging.info("Initializing state")
        state = initialize()
        start_round = 1
        
    async with tff.async_utils.ordered_tasks() as tasks:
        
        train_data_iterator = train_data_source.iterator()

        for round_number in range(start_round, total_rounds + 1):
            tasks.add_callable(functools.partial(logging.info, 'Running round %d of training', round_number))

            train_data = train_data_iterator.select(number_of_clients)
            state, metrics = train(state, train_data)
            
            if train_output_managers is not None:
                tasks.add_all(*[m.release(metrics, round_number) for m in train_output_managers])
            
            if program_state_manager is not None:
                program_state = (state, start_round)
                tasks.add(program_state_manager.save(program_state, round_number))
                
        tasks.add_callable(
            functools.partial(logging.info, 'Running one round of evaluation')
        )
        
        evaluation_data_iterator = evaluation_data_source.iterator()
        evaluation_data = evaluation_data_iterator.select(number_of_clients)
        evaluation_metrics = evaluation(state, evaluation_data)
        
        
        if evaluation_output_managers is not None:
            tasks.add_all(*[
                m.release(evaluation_metrics, round_number)
                for m in train_output_managers
            ])
        
        if model_output_manager is not None:
            tasks.add(model_output_manager.release(state))

METRICS_TOTAL_SUM = 'total_sum'

@tff.tf_computation()
def initialize() -> int:
  """Returns the initial state."""
  return 0

@tff.tf_computation(tff.SequenceType(tf.int32))
def _sum_dataset(dataset: tf.data.Dataset) -> int:
  """Returns the sum of all the integers in `dataset`."""
  return dataset.reduce(tf.cast(0, tf.int32), tf.add)

@tff.tf_computation(tf.int32, tf.int32)
def _sum_integers(x: int, y: int) -> int:
  """Returns the sum two integers."""
  return x + y

@tff.federated_computation(
    tff.type_at_server(tf.int32),
    tff.type_at_clients(tff.SequenceType(tf.int32)))
def train(
    server_state: int, client_data: tf.data.Dataset
) -> Tuple[int, collections.OrderedDict[str, Any]]:
  """Computes the sum of all the integers on the clients.
  Computes the sum of all the integers on the clients, updates the server state,
  and returns the updated server state and the following metrics:
  * `sum_client_data.METRICS_TOTAL_SUM`: The sum of all the client_data on the
    clients.
  Args:
    server_state: The server state.
    client_data: The data on the clients.
  Returns:
    A tuple of the updated server state and the train metrics.
  """
  client_sums = tff.federated_map(_sum_dataset, client_data)
  total_sum = tff.federated_sum(client_sums)
  updated_state = tff.federated_map(_sum_integers, (server_state, total_sum))
  metrics = collections.OrderedDict([
      (METRICS_TOTAL_SUM, total_sum),
  ])
  return updated_state, metrics

@tff.federated_computation(
    tff.type_at_server(tf.int32),
    tff.type_at_clients(tff.SequenceType(tf.int32)))
def evaluation(
    server_state: int,
    client_data: tf.data.Dataset) -> collections.OrderedDict[str, Any]:
  """Computes the sum of all the integers on the clients.
  Computes the sum of all the integers on the clients and returns the following
  metrics:
  * `sum_client_data.METRICS_TOTAL_SUM`: The sum of all the client_data on the
    clients.
  Args:
    server_state: The server state.
    client_data: The data on the clients.
  Returns:
    The evaluation metrics.
  """
  del server_state  # Unused.
  client_sums = tff.federated_map(_sum_dataset, client_data)
  total_sum = tff.federated_sum(client_sums)
  metrics = collections.OrderedDict([
      (METRICS_TOTAL_SUM, total_sum),
  ])

  return metrics


def tff_program():
    
    total_rounds = 10
    number_of_clients = 3
    OUTPUT_DIR = 'some_dir'
    
    # tff.backends.native.execution_contexts.set_local_async_python_execution_context(reference_resolving_clients=True)
    context = tff.backends.native.create_local_async_python_execution_context(
    )
    context = tff.program.NativeFederatedContext(context)
    tff.framework.set_default_context(context)
    
    to_int32 = lambda x: tf.cast(x, tf.int32)
    datasets = [tf.data.Dataset.range(10).map(to_int32)] * 3
    train_data_source = tff.program.DatasetDataSource(datasets)
    evaluation_data_source = tff.program.DatasetDataSource(datasets)

    #TODO
    # initialize = initialize
    # train = train
    # evaluation = evaluation
    
    train_output_managers = [tff.program.LoggingReleaseManager()]
    evaluation_output_managers = [tff.program.LoggingReleaseManager()]
    model_output_manager = tff.program.LoggingReleaseManager()

    summary_dir = os.path.join(OUTPUT_DIR, 'summary')
    tensorboard_manager = tff.program.TensorBoardReleaseManager(summary_dir)
    train_output_managers.append(tensorboard_manager)

    csv_path = os.path.join(OUTPUT_DIR, 'evaluation_metrics.csv')
    csv_manager = tff.program.CSVFileReleaseManager(csv_path)
    evaluation_output_managers.append(csv_manager)

    program_state_manager = tff.program.FileProgramStateManager(
        OUTPUT_DIR
    )

    asyncio.ensure_future(tff_train_federated(
        initialize=initialize,
        train=train,
        train_data_source=train_data_source,
        evaluation=evaluation,
        evaluation_data_source=evaluation_data_source,
        total_rounds=total_rounds,
        number_of_clients=number_of_clients,
        train_output_managers=train_output_managers,
        evaluation_output_managers=evaluation_output_managers,
        model_output_manager=model_output_manager,
        program_state_manager=program_state_manager
    ))


class TFFService(ImmediateNodeServiceWithReply):
    @staticmethod
    @service_auth(guests_welcome=True)
    def process(
        node: AbstractNode, msg: TFFMessage, verify_key: Optional[VerifyKey] = None
    ) -> TFFReplyMessage:
        if verify_key is None:
            traceback_and_raise("Can't process TFFService with no verification key.")
        
        
        
        # dataset_id = '03824e77-3d62-426a-bdea-e836ba210c2b'

        
        # print(node.datasets.get(dataset_id))
        # uid_images = node.datasets.get(dataset_id)[1][0].obj
        # uid_labels = node.datasets.get(dataset_id)[1][0].obj
        # # print(node.datasets.get('de6332f0-2c20-4904-9816-96b74b26d4ad')[1][0].obj)
        # # print(uid.replace("-",""))
        # # for key in node.store.keys():
        # #     print(key.to_string())
        # #     # t = node.store.get(key)
        # #     # print(t)
        # # print(dir(node.store))
        # tensor = node.store.get(uid_images)
        # print(tensor.data.child.child.child.shape)
        # tff.backends.native.execution_contexts.set_local_async_python_execution_context(reference_resolving_clients=True)
        # # asyncio.ensure_future(test_data_descriptor(node))
        # logging.basicConfig(level=logging.INFO)

            
        # # logger = logging.getLogger('tensorflow_federated')
        # # logger.setLevel(level=logging.NOTSET)
        # # logger = logging.getLogger()
        # # logger.setLevel(level=logging.NOTSET)
        # # asyncio.ensure_future(test_train_model(node))
        # input_spec = collections.OrderedDict(
        #     [
        #         ('x', tf.TensorSpec(shape=(1,64,64), dtype=np.int32, name=None)),
        #         ('y', tf.TensorSpec(shape=(1,1), dtype=np.int32, name=None)),
        #     ]
        # )
        
        # # element_type = tff.TensorType(dtype=tf.int32, shape=[58954,64,64])
        # # element_type = tff.types.StructWithPythonType(
        # #     input_spec,
        # #     container_type=collections.OrderedDict)
        # # data_type = tff.types.SequenceType(element_type)
        # exp_value = collections.OrderedDict(
        #     x = node.store.get(uid_images).data.child.child.child,
        #     y = node.store.get(uid_labels).data.child.child.child,
        # )
        
        # # asyncio.ensure_future(test_materialize(node, '03824e77-3d62-426a-bdea-e836ba210c2b', (), exp_value))
        
        # # asyncio.ensure_future(test_syft_tensor(node))
        
        # # tensor = node.store.get(node.store.keys()[0])
        # # print(dir(tensor))
        # # print(tensor.data.numpy())
        # # print(dir(tensor.data))
        # # print(dir(node.store))
        # # for key in node.store.keys():
        # #     tensor = node.store.get(key).data
        # #     if type(tensor) == Tensor:
        # #         # print(dir(node.store.get(tensor).data))
        # #         # print(node.store.get(tensor).data)
        #         print("CE Plm:", type(tensor.child.child.child))
        
        # print(msg.payload.id_dataset)
        # print(msg.payload.params)
        # print(msg.payload.model_bytes)
        # print(msg.payload.more_stuff)
        tff_program()
        result = msg.payload.run(node=node, verify_key=verify_key)
        return TFFReplyMessage(payload=result, address=msg.reply_to)

    @staticmethod
    def message_handler_types() -> List[Type[TFFMessage]]:
        return [TFFMessage]
