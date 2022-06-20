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
from ..auth import service_auth
from ..node_service import ImmediateNodeServiceWithReply

# from .data_backend import MedNISTBackend
# from .data_backend import PySyftDataBackend
from .tff_messages import TFFMessage
from .tff_messages import TFFReplyMessage


def create_keras_model():
    return tf.keras.models.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(64 * 64,)),
            tf.keras.layers.Dense(6, kernel_initializer="zeros"),
            tf.keras.layers.Softmax(),
        ]
    )


def model_fn(input_spec):
    keras_model = create_keras_model()
    return tff.learning.from_keras_model(
        keras_model,
        input_spec=input_spec,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )


def get_ctx(data_backend):
    def ex_fn(device: tf.config.LogicalDevice) -> tff.framework.DataExecutor:
        return tff.framework.DataExecutor(
            tff.framework.EagerTFExecutor(device), data_backend=data_backend
        )

    factory = tff.framework.local_executor_factory(leaf_executor_fn=ex_fn)
    return async_execution_context.AsyncExecutionContext(executor_fn=factory)


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
    program_state_manager: Optional[tff.program.ProgramStateManager] = None,
):
    tff.program.check_in_federated_context()
    logging.info("Running program logic")

    if program_state_manager is not None:
        structure = initialize()
        program_state, version = await program_state_manager.load_latest(structure)
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

            if train_output_managers is not None:
                tasks.add_all(
                    *[m.release(metrics, round_number) for m in train_output_managers]
                )

            if program_state_manager is not None:
                program_state = (state, start_round)
                tasks.add(program_state_manager.save(program_state, round_number))

        tasks.add_callable(
            functools.partial(logging.info, "Running one round of evaluation")
        )

        evaluation_data_iterator = evaluation_data_source.iterator()
        evaluation_data = evaluation_data_iterator.select(number_of_clients)
        evaluation_metrics = evaluation(state, evaluation_data)

        if evaluation_output_managers is not None:
            tasks.add_all(
                *[
                    m.release(evaluation_metrics, round_number)
                    for m in train_output_managers
                ]
            )

        if model_output_manager is not None:
            tasks.add(model_output_manager.release(state))


METRICS_TOTAL_SUM = "total_sum"


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
    tff.type_at_server(tf.int32), tff.type_at_clients(tff.SequenceType(tf.int32))
)
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
    metrics = collections.OrderedDict(
        [
            (METRICS_TOTAL_SUM, total_sum),
        ]
    )
    return updated_state, metrics


@tff.federated_computation(
    tff.type_at_server(tf.int32), tff.type_at_clients(tff.SequenceType(tf.int32))
)
def evaluation(
    server_state: int, client_data: tf.data.Dataset
) -> collections.OrderedDict[str, Any]:
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
    metrics = collections.OrderedDict(
        [
            (METRICS_TOTAL_SUM, total_sum),
        ]
    )

    return metrics


def tff_program():

    total_rounds = 10
    number_of_clients = 3
    OUTPUT_DIR = "some_dir"

    # tff.backends.native.execution_contexts.set_local_async_python_execution_context(reference_resolving_clients=True)
    context = tff.backends.native.create_local_async_python_execution_context()
    context = tff.program.NativeFederatedContext(context)
    tff.framework.set_default_context(context)

    # to_int32 = lambda x: tf.cast(x, tf.int32)
    datasets = [tf.data.Dataset.range(10).map(lambda x: tf.cast(x, tf.int32))] * 3
    train_data_source = tff.program.DatasetDataSource(datasets)
    evaluation_data_source = tff.program.DatasetDataSource(datasets)

    # TODO
    # initialize = initialize
    # train = train
    # evaluation = evaluation

    train_output_managers = [tff.program.LoggingReleaseManager()]
    evaluation_output_managers = [tff.program.LoggingReleaseManager()]
    model_output_manager = tff.program.LoggingReleaseManager()

    summary_dir = os.path.join(OUTPUT_DIR, "summary")
    tensorboard_manager = tff.program.TensorBoardReleaseManager(summary_dir)
    train_output_managers.append(tensorboard_manager)

    csv_path = os.path.join(OUTPUT_DIR, "evaluation_metrics.csv")
    csv_manager = tff.program.CSVFileReleaseManager(csv_path)
    evaluation_output_managers.append(csv_manager)

    program_state_manager = tff.program.FileProgramStateManager(OUTPUT_DIR)

    asyncio.ensure_future(
        tff_train_federated(
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
            program_state_manager=program_state_manager,
        )
    )


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
