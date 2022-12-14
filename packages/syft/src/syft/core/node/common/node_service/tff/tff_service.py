# stdlib
import asyncio
import collections
import functools
import io
from typing import Any
from typing import List
from typing import Optional
from typing import Type
from typing import Union
import zipfile

# third party
from nacl.signing import VerifyKey
import numpy as np

try:
    # third party
    import tensorflow as tf
    import tensorflow_federated as tff
    from tensorflow_federated.python.program import value_reference

except Exception:  # nosec
    # no tff
    pass

# relative
from ......logger import debug
from ......util import traceback_and_raise
from ....abstract.node import AbstractNode
from ....common import UID
from ..auth import service_auth
from ..node_service import ImmediateNodeServiceWithReply
from .tff_messages import TFFMessage
from .tff_messages import TFFReplyMessage


async def tff_train_federated(
    initialize: "tff.Computation",
    train: "tff.Computation",
    train_data_source: "tff.program.FederatedDataSource",
    total_rounds: int,
    number_of_clients: int,
    train_output_managers: "List[tff.program.ReleaseManager]",
) -> Union[List, Any]:
    results = []
    state = initialize()
    start_round = 1

    # This makes tasks run in order
    async with tff.async_utils.ordered_tasks() as tasks:

        train_data_iterator = train_data_source.iterator()

        # Main training loop
        for round_number in range(start_round, total_rounds + 1):
            tasks.add_callable(
                functools.partial(debug, f"Running round {round_number} of training")
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


def tff_program(
    node: AbstractNode, params: dict, func_model: "tf.keras.Model"
) -> Union[List, Any]:
    total_rounds = int(params["rounds"])
    number_of_clients = int(params["no_clients"])
    noise_multiplier = float(params["noise_multiplier"])
    clients_per_round = int(params["clients_per_round"])

    train_data_id = UID.from_string(str(params["train_data_id"]))
    label_data_id = UID.from_string(str(params["label_data_id"]))

    context = tff.backends.native.create_local_async_python_execution_context()
    context = tff.program.NativeFederatedContext(context)
    tff.framework.set_default_context(context)

    # Using the data ids we are fetching the data from the domain store
    # Currently we do not support training using our DP tensors or our SMPC tensors
    # so we will strip those layers and get the raw data
    train_data = node.store.get(train_data_id).data.child.child
    labels = node.store.get(label_data_id).data.child.child

    train_shape = list(train_data.shape)
    train_shape[0] = -1

    # If might happen that our targets will be a one dimensional array
    # like the numerical labels in case of image classification
    label_data_shape = list(labels.shape)
    if len(label_data_shape) == 1:
        label_data_shape = [-1, 1]
    else:
        label_data_shape[0] = -1

    def preprocess(train_data: np.array, labels: np.array) -> list:
        return [
            tf.reshape(train_data, train_shape),
            tf.reshape(labels, label_data_shape),
        ]

    # For our PoC we will only duplicate the data for each client
    # In the next release we will move this at the level of the network
    # so we could think of each client as a separate domain
    dataset = tf.data.Dataset.from_tensor_slices((train_data, labels))
    datasets = [dataset.map(preprocess)] * number_of_clients
    train_data_source = tff.program.DatasetDataSource(datasets)

    def model_fn() -> tff.learning.Model:
        return tff.learning.models.model_from_functional(func_model)

    # Currently we support the DP functions from TFF
    aggregation_factory = tff.learning.model_update_aggregator.dp_aggregator(
        noise_multiplier, clients_per_round
    )

    # Some improvements would be parametrizing this two hardcoded values
    iterative_process = tff.learning.algorithms.build_unweighted_fed_avg(
        model_fn,
        client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.02),
        server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0),
        model_aggregator=aggregation_factory,
    )

    initialize = iterative_process.initialize
    train = iterative_process.next
    train_output_managers = [tff.program.LoggingReleaseManager()]

    # run the federated training and waiting for the metrics and the state
    # to extract from it the model weights
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


def aux_recursive_od2d(dit: collections.OrderedDict) -> dict:
    new_dict = {}
    for key in dit:
        if type(dit[key]) == collections.OrderedDict:
            new_elem = aux_recursive_od2d(dit[key])
            new_dict[key] = new_elem
        else:
            new_dict[key] = dit[key]
    return new_dict


def ordereddict2dict(list_dict: list) -> list:
    new_list = []
    for od in list_dict:
        new_dict = aux_recursive_od2d(od)
        new_list.append(new_dict)
    return new_list


class TFFService(ImmediateNodeServiceWithReply):
    @staticmethod
    @service_auth(guests_welcome=True)
    def process(
        node: AbstractNode, msg: TFFMessage, verify_key: Optional[VerifyKey] = None
    ) -> TFFReplyMessage:
        if verify_key is None:
            traceback_and_raise("Can't process TFFService with no verification key.")

        # parse params
        params = msg.payload.params

        # read model
        zf = zipfile.ZipFile(io.BytesIO(msg.payload.model_bytes), "r")
        zf.extractall("tmp_dir")
        model = tff.learning.models.load_functional_model("tmp_dir")

        # run the training based on the example from:
        # https://github.com/tensorflow/federated/tree/main/tensorflow_federated/examples/program
        metrics, state = tff_program(node, params, model)

        # serialize model weights
        memfile = io.BytesIO()
        np.save(memfile, state.global_model_weights.trainable)
        serialized_trainable = memfile.getvalue()

        memfile = io.BytesIO()
        np.save(memfile, state.global_model_weights.non_trainable)
        serialized_non_trainable = memfile.getvalue()

        # Full response
        payload = {
            "metrics": ordereddict2dict(metrics),
            "trainable": serialized_trainable,
            "non_trainable": serialized_non_trainable,
        }

        # Run specific msg function, currently doesn't do anything. only for linting
        res = msg.payload.run(payload=str(payload), node=node, verify_key=verify_key)

        return TFFReplyMessage(payload=res, address=msg.reply_to)

    @staticmethod
    def message_handler_types() -> List[Type[TFFMessage]]:
        return [TFFMessage]
