# stdlib
import asyncio
import collections
import functools
import logging
import os
from typing import Any
from typing import List
from typing import Optional
from typing import Tuple
from typing import Type

# third party
from nacl.signing import VerifyKey
import tensorflow as tf
import tensorflow_federated as tff
from tensorflow_federated.proto.v0 import computation_pb2 as pb
from tensorflow_federated.python.core.impl.execution_contexts import (
    async_execution_context,
)

# relative
# import torch as th
from ......util import traceback_and_raise
from .....tensor.tensor import Tensor
from ....abstract.node import AbstractNode
from ......core.common.uid import UID
import numpy as np
import zipfile
import io

from ..auth import service_auth
from ..node_service import ImmediateNodeServiceWithReply

# from .data_backend import MedNISTBackend
# from .data_backend import PySyftDataBackend
from .tff_messages import TFFMessage
from .tff_messages import TFFReplyMessage
from .data_backend import TestDataBackend, PySyftDataBackend, MedNISTBackend
from tensorflow_federated.proto.v0 import computation_pb2 as pb
import tensorflow as tf
from absl.testing import absltest
import asyncio
from tensorflow_federated.python.core.impl.executors import eager_tf_executor
from tensorflow_federated.python.core.impl.types import computation_types
# from tensorflow_federated.python.core.impl.executors import executor_test_utils
from tensorflow_federated.python.core.impl.executors import executor_stacks
from tensorflow_federated.python.core.impl.context_stack import set_default_context
# from tensorflow_federated.python.core.impl.context_stack import context_stack_test_utils
from tensorflow_federated.python.core.backends.native import compiler
from tensorflow_federated.python.core.impl.execution_contexts import async_execution_context
from tensorflow_federated.python.learning.models import functional
from syft.core.tensor.tensor import Tensor
import logging
import collections
import os
import functools

# from pybind11_abseil import status
import torch as th

METRICS_TOTAL_SUM = 'total_sum'
@tff.tf_computation()
def _initialize() -> int:
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
def _train(
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
def _evaluation(
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

@tff.tf_computation
async def tff_train_federated(
    initialize: tff.Computation,
    train: tff.Computation,
    train_data_source: tff.program.FederatedDataSource,
    evaluation: tff.Computation,
    evaluation_data_source: tff.program.FederatedDataSource,
    total_rounds: int,
    number_of_clients: int,
    train_output_managers: List[tff.program.ReleaseManager],
    evaluation_output_managers: List[tff.program.ReleaseManager],
    model_output_manager: tff.program.ReleaseManager,
    program_state_manager: tff.program.ProgramStateManager,
):
    tff.program.check_in_federated_context()
    logging.info("Running program logic")

    if program_state_manager is not None:
        structure = initialize()
        program_state, version = program_state_manager.load_latest(structure)
    else:
        program_state = None

    if program_state is not None:
        logging.info("Loaded program state at version %d", version)

        state, start_round = program_state

    else:
        logging.info("Initializing state")
        state = initialize()
        start_round = 1

    async with tff.async_utils.ordered_tasks() as tasks:

        train_data_iterator = train_data_source.iterator()

        for round_number in range(start_round, total_rounds + 1):
            tasks.add_callable(
                functools.partial(
                    logging.info, "Running round %d of training", round_number
                )
            )

            train_data = train_data_iterator.select(number_of_clients)
            state, metrics = train(state, train_data)
            
            # if train_output_managers is not None:
            #     tasks.add_all(*[m.release(metrics, round_number) for m in train_output_managers])
            
            # if program_state_manager is not None:
            #     program_state = (state, start_round)
            #     tasks.add(program_state_manager.save(program_state, round_number))
            value = await metrics['train']['sparse_categorical_accuracy'].get_value()
            print(value)
            
                
        tasks.add_callable(
            functools.partial(logging.info, "Running one round of evaluation")
        )

        evaluation_data_iterator = evaluation_data_source.iterator()
        evaluation_data = evaluation_data_iterator.select(number_of_clients)
        evaluation_metrics = evaluation(state, evaluation_data)
        
        
        # if evaluation_output_managers is not None:
        #     tasks.add_all(*[
        #         m.release(evaluation_metrics, round_number)
        #         for m in train_output_managers
        #     ])
        
        # if model_output_manager is not None:
        #     tasks.add(model_output_manager.release(state))

def tff_program(
    node, 
    params,
    model
):    
    dataset_id = str(params['dataset_id']) 
    total_rounds = int(params['rounds']) 
    number_of_clients = int(params['no_clients'])
    OUTPUT_DIR = str(params['OUTPUT_DIR'])
    noise_multiplier = float(params['noise_multiplier'])
    clients_per_round = int(params['clients_per_round'])
    
    # tff.backends.native.execution_contexts.set_local_async_python_execution_context(reference_resolving_clients=True)
    context = tff.backends.native.create_local_async_python_execution_context()
    context = tff.program.NativeFederatedContext(context)
    tff.framework.set_default_context(context)

    dataset_objs = node.datasets.get(dataset_id)[1]
    images = node.store.get(dataset_objs[0].obj).data.child.child.decode()
    labels = node.store.get(dataset_objs[1].obj).data.child.child.decode()


    def preprocess(images, labels):
            return [
                tf.reshape(images, [-1, 64*64]),
                tf.reshape(labels, [-1, 1]),
            ]

    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    
    datasets = [dataset.map(preprocess)] * number_of_clients 


    train_data_source = tff.program.DatasetDataSource(datasets)
    evaluation_data_source = tff.program.DatasetDataSource(datasets)

    # TODO parametrize this
    input_spec = collections.OrderedDict(
            x=tf.TensorSpec(shape=(1,64*64), dtype=tf.int32, name=None),
            y=tf.TensorSpec(shape=(1,1), dtype=tf.int32, name=None),
    )

    aggregation_factory = tff.learning.model_update_aggregator.dp_aggregator(
      noise_multiplier, clients_per_round)
    
    iterative_process = tff.learning.build_federated_averaging_process(
    lambda: model_fn(input_spec=input_spec),
    client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.02),
    server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0),
    model_update_aggregation_factory=aggregation_factory)

    initialize = iterative_process.initialize
    train = iterative_process.next
    evaluation = tff.learning.build_federated_evaluation(lambda: model_fn(input_spec))

    train_output_managers = [tff.program.LoggingReleaseManager()]
    evaluation_output_managers = [tff.program.LoggingReleaseManager()]
    model_output_manager = tff.program.LoggingReleaseManager()

    # add some date in the name of the folders

    summary_dir = os.path.join(OUTPUT_DIR, 'summary')
    tensorboard_manager = tff.program.TensorBoardReleaseManager(summary_dir)
    train_output_managers.append(tensorboard_manager)

    csv_path = os.path.join(OUTPUT_DIR, "evaluation_metrics.csv")
    csv_manager = tff.program.CSVFileReleaseManager(csv_path)
    evaluation_output_managers.append(csv_manager)

    program_state_dir = os.path.join(OUTPUT_DIR, 'program_state')
    program_state_manager = tff.program.FileProgramStateManager(
        program_state_dir
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
        
        logging.basicConfig(level=logging.INFO)
        # logger = logging.getLogger('absl')
        # logger.setLevel(level=logging.INFO)
        
        # parse params
        params = msg.payload.params
        print(params)
        
        # read model
        zf = zipfile.ZipFile(io.BytesIO(msg.payload.model_bytes), 'r')
        zf.extractall('tmp_dir')
        functional_model_reloaded = tff.learning.models.load_functional_model('tmp_dir')
        model = functional.model_from_functional(functional_model_reloaded)
        
        tff_program(node, params, model)
        
        
        
        result = msg.payload.run(node=node, verify_key=verify_key)
        return TFFReplyMessage(payload=result, address=msg.reply_to)

    @staticmethod
    def message_handler_types() -> List[Type[TFFMessage]]:
        return [TFFMessage]




