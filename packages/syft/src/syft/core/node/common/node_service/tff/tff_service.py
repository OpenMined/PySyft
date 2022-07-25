# stdlib
import asyncio
import collections
import functools
import io
import logging
import os
from typing import Any
from typing import List
from typing import Optional
from typing import Tuple
from typing import Type
from unittest import result
import zipfile
import json

# third party
from nacl.signing import VerifyKey
import numpy as np
try:
    import tensorflow as tf
    import tensorflow_federated as tff
    from tensorflow_federated.python.program import value_reference

except:
    pass
# from tensorflow_federated.proto.v0 import computation_pb2 as pb

# from tensorflow_federated.python.core.backends.native import compiler
# from tensorflow_federated.python.core.impl.context_stack import set_default_context
# from tensorflow_federated.python.core.impl.execution_contexts import (
#     async_execution_context,
# )

# from tensorflow_federated.python.core.impl.executors import executor_test_utils
# from tensorflow_federated.python.core.impl.executors import eager_tf_executor
# from tensorflow_federated.python.core.impl.types import computation_types
# from tensorflow_federated.python.learning.models import functional


# from pybind11_abseil import status
import torch as th

# relative
from ......core.common.uid import UID

# import torch as th
from ......logger import debug
from ......util import traceback_and_raise
from .....tensor.tensor import Tensor
from ....abstract.node import AbstractNode
from ..auth import service_auth
from ..node_service import ImmediateNodeServiceWithReply
from .tff_messages import TFFMessage
from .tff_messages import TFFReplyMessage

async def tff_train_federated(
    initialize,#: tff.Computation,
    train,#: tff.Computation,
    train_data_source,#: tff.program.FederatedDataSource,
    total_rounds: int,
    number_of_clients: int,
    train_output_managers,#: List[tff.program.ReleaseManager],
) -> None:
    results = []
    state = initialize()
    start_round = 1
    
    # This makes tasks run in order 
    async with tff.async_utils.ordered_tasks() as tasks:

        train_data_iterator = train_data_source.iterator()

        # Main training loop
        for round_number in range(start_round, total_rounds + 1):
            tasks.add_callable(
                functools.partial(
                    debug, f"Running round {round_number} of training" 
                )
            )

            train_data = train_data_iterator.select(number_of_clients)
            output = train(state, train_data)
            state = output.state
            metrics = output.metrics

            if train_output_managers is not None:
                tasks.add_all(
                    *[m.release(metrics, round_number) for m in train_output_managers]
                )
                value = await value_reference.materialize_value(metrics)
                results.append(value)

    state = await value_reference.materialize_value(state)
    return results, state


def tff_program(node, params, func_model) -> None:
    dataset_id = str(params["dataset_id"])
    total_rounds = int(params["rounds"])
    number_of_clients = int(params["no_clients"])
    noise_multiplier = float(params["noise_multiplier"])
    clients_per_round = int(params["clients_per_round"])

    context = tff.backends.native.create_local_async_python_execution_context()
    context = tff.program.NativeFederatedContext(context)
    tff.framework.set_default_context(context)

    # This needs customization
    dataset_objs = node.datasets.get(dataset_id)[1]
    images = node.store.get(dataset_objs[0].obj).data.child.child
    labels = node.store.get(dataset_objs[1].obj).data.child.child

    # TODO: replace the model so this is not needed
    def preprocess(images, labels):
        return [
            tf.reshape(images, [-1, 64 * 64]),
            tf.reshape(labels, [-1, 1]),
        ]

    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    datasets = [dataset.map(preprocess)] * number_of_clients
    train_data_source = tff.program.DatasetDataSource(datasets)
    
    
    # Create Iterative Process based on custom functional model
    # TODO: add DP
    def model_fn():
        return tff.learning.models.model_from_functional(func_model)
    
    aggregation_factory = tff.learning.model_update_aggregator.dp_aggregator(
      noise_multiplier, clients_per_round)

    
    iterative_process = tff.learning.algorithms.build_unweighted_fed_avg(
        model_fn,
        client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.02),
        server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0),
        model_aggregator=aggregation_factory
    )

    initialize = iterative_process.initialize
    train = iterative_process.next
    train_output_managers = [tff.program.LoggingReleaseManager()]
    
    results, state = asyncio.run(
        tff_train_federated(
            initialize=initialize,
            train=train,
            train_data_source=train_data_source,
            total_rounds=total_rounds,
            number_of_clients=number_of_clients,
            train_output_managers=train_output_managers,
        )
    )
    
    return results, state
    

class TFFService(ImmediateNodeServiceWithReply):
    @staticmethod
    @service_auth(guests_welcome=True)
    def process(
        node: AbstractNode, msg: TFFMessage, verify_key: Optional[VerifyKey] = None
    ) -> TFFReplyMessage:
        if verify_key is None:
            traceback_and_raise("Can't process TFFService with no verification key.")

        # Run specific msg function, currently doesn't do anything 
        result = msg.payload.run(node=node, verify_key=verify_key)
        
        # parse params
        params = msg.payload.params

        # read model
        zf = zipfile.ZipFile(io.BytesIO(msg.payload.model_bytes), "r")
        zf.extractall("tmp_dir")
        model = tff.learning.models.load_functional_model(
            "tmp_dir"
        )
    
        # run the training based on the example from:
        # https://github.com/tensorflow/federated/tree/main/tensorflow_federated/examples/program
        metrics, state = tff_program(node, params, model)

        # serialize model weights
        memfile = io.BytesIO()
        np.save(memfile, state.global_model_weights.trainable)
        serialized_trainable = memfile.getvalue()
        
        memfile= io.BytesIO()
        np.save(memfile, state.global_model_weights.non_trainable)
        serialized_non_trainable = memfile.getvalue()

        # Full response
        payload = {
            'metrics': metrics, 
            'trainable': serialized_trainable, 
            'non_trainable': serialized_non_trainable
            }

        return TFFReplyMessage(payload=str(payload), address=msg.reply_to)


    @staticmethod
    def message_handler_types() -> List[Type[TFFMessage]]:
        return [TFFMessage]
