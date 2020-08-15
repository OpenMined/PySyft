# PyGrid imports
# Syft dependencies
import abc

import syft as sy
from syft.execution.placeholder import PlaceHolder
from syft.execution.state import State
from syft.serde import protobuf
from syft_proto.execution.v1.state_pb2 import State as StatePB

from .exceptions import ModelNotFoundError
from .warehouse import Warehouse
from .model import Model, ModelCheckPoint


class ModelManager(metaclass=abc.ABCMeta):
    def __init__(self):
        self._models = Warehouse(Model)
        self._model_checkpoints = Warehouse(ModelCheckPoint)

    def create(self, model, process):
        # Register new model
        _model_obj = self._models.register(flprocess=process)

        # Save model initial weights into ModelCheckpoint
        self._model_checkpoints.register(
            value=model, model=_model_obj, number=1, alias="latest"
        )

        return _model_obj

    def save(self, model_id: int, data: bin):
        """Create a new model checkpoint.

        Args:
            model_id: Model ID.
            data: Model data.
        Returns:
            model_checkpoint: ModelCheckpoint instance.
        """

        checkpoints_count = self._model_checkpoints.count(model_id=model_id)

        # Reset "latest" alias
        self._model_checkpoints.modify(
            {"model_id": model_id, "alias": "latest"}, {"alias": ""}
        )

        # Create new checkpoint
        new_checkpoint = self._model_checkpoints.register(
            model_id=model_id, value=data, number=checkpoints_count + 1, alias="latest"
        )
        return new_checkpoint

    def load(self, **kwargs):
        """Load model's Checkpoint."""
        _check_point = self._model_checkpoints.last(**kwargs)

        if not _check_point:
            raise ModelNotFoundError

        return _check_point

    def get(self, **kwargs):
        """Retrieve the model instance object.

        Args:
            process_id : Federated Learning Process ID attached to this model.
        Returns:
            model : SQL Model Object.
        Raises:
            ModelNotFoundError (PyGridError) : If model not found.
        """
        _model = self._models.last(**kwargs)

        if not _model:
            raise ModelNotFoundError

        return _model

    @staticmethod
    def serialize_model_params(params):
        """Serializes list of tensors into State/protobuf."""
        model_params_state = State(
            state_placeholders=[PlaceHolder().instantiate(param) for param in params]
        )

        # make fake local worker for serialization
        worker = sy.VirtualWorker(hook=None)

        pb = protobuf.serde._bufferize(worker, model_params_state)
        serialized_state = pb.SerializeToString()

        return serialized_state

    @staticmethod
    def unserialize_model_params(bin: bin):
        """Unserializes model or checkpoint or diff stored in db to list of
        tensors."""
        state = StatePB()
        state.ParseFromString(bin)
        worker = sy.VirtualWorker(hook=None)
        state = protobuf.serde._unbufferize(worker, state)
        model_params = state.tensors()
        return model_params
