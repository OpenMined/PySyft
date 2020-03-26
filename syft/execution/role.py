import re
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

# TODO torch shouldn't be used here
import torch

import syft as sy
from syft.execution.computation import ComputationAction
from syft.execution.placeholder import PlaceHolder
from syft.execution.placeholder_id import PlaceholderId
from syft.execution.state import State
from syft.generic.frameworks.types import FrameworkTensor
from syft.generic.object import AbstractObject
from syft.generic.object_storage import ObjectStorage
from syft.workers.abstract import AbstractWorker


class Role(AbstractObject, ObjectStorage):
    """
    Roles will mainly be used to build protocols but are still a work in progress.
    """

    def __init__(
        self,
        state: State = None,
        actions: List[ComputationAction] = None,
        placeholders: Dict[Union[str, int], PlaceHolder] = None,
        input_placeholder_ids: Tuple[int, str] = None,
        output_placeholder_ids: Tuple[int, str] = None,
        state_tensors=None,
        # General kwargs
        id: Union[str, int] = None,
        owner: "sy.workers.BaseWorker" = None,
        tags: List[str] = None,
        description: str = None,
    ):
        owner = owner or sy.local_worker
        AbstractObject.__init__(self, id, owner, tags, description, child=None)

        self.owner = owner
        self.actions = actions or []
        self.state = state or State(owner=owner)
        # All placeholders
        self.placeholders = placeholders or {}
        # Input placeholders, stored by id
        self.input_placeholder_ids = input_placeholder_ids or ()  # TODO init with args
        # Output placeholders
        self.output_placeholder_ids = output_placeholder_ids or ()  # TODO init with args

        # # state_tensors are provided when plans are created using func2plan
        # if state_tensors is not None:
        #     # we want to make sure in that case that the state is empty
        #     assert state is None
        #     for tensor in state_tensors:
        #         placeholder = sy.PlaceHolder(
        #             tags={"#state", f"#{self.var_count + 1}"}, id=tensor.id, owner=self.owner
        #         )
        #         self.var_count += 1
        #         placeholder.instantiate(tensor)
        #         self.state.state_placeholders.append(placeholder)
        #         self.placeholders[tensor.id] = placeholder

    def register_computation_inputs(self, args):
        """ Takes input arguments for this role and generate placeholders.
        """
        # TODO Should we be able to rebuild?
        self.input_placeholder_ids = tuple(self.build_placeholders(arg).value for arg in args)

    def register_computation_outputs(self, results):
        """ Takes output tensors for this role and generate placeholders.
        """
        results = (results,) if not isinstance(results, tuple) else results
        self.output_placeholder_ids = tuple(
            self.build_placeholders(result).value for result in results
        )

    def register_computation_action(self, log):  # TODO find better argument naming
        """ Build placeholders and store action.
        """
        command, response = log
        command_placeholder_ids = self.build_placeholders(command)
        return_placeholder_ids = self.build_placeholders(response)

        if not isinstance(return_placeholder_ids, (list, tuple)):
            return_placeholder_ids = (return_placeholder_ids,)
        action = ComputationAction(*command_placeholder_ids, return_ids=return_placeholder_ids)
        self.actions.append(action)

    def execute_computation(self, args):
        """ Make the role execute all its actions using args as computation inputs.
        """
        self.instantiate_computation_inputs(args)
        for action in self.actions:
            self.execute_computation_action(action)

        output_placeholders = tuple(
            self.placeholders[output_id] for output_id in self.output_placeholder_ids
        )
        result = tuple(p.child for p in output_placeholders)

        if len(result) == 1:
            return result[0]
        return result

    def instantiate_computation_inputs(self, args):
        """ Takes input arguments for this role and generate placeholders.
        """
        input_placeholders = tuple(
            self.placeholders[input_id] for input_id in self.input_placeholder_ids
        )
        self.instantiate(input_placeholders, args)

    def execute_computation_action(self, action):  # TODO find better argument naming
        """ Build placeholders and store action.
        """
        cmd, _self, args, kwargs, return_placeholder = (
            action.name,
            action.target,  # target is equivalent to the "self" in a method
            action.args,
            action.kwargs,
            action.return_ids,
        )
        _self = self.fecth_placeholders_from_ids(_self)
        args = self.fecth_placeholders_from_ids(args)
        kwargs = self.fecth_placeholders_from_ids(kwargs)
        return_placeholder = self.fecth_placeholders_from_ids(return_placeholder)

        if _self is None:
            response = eval(cmd)(*args, **kwargs)  # nosec
        else:
            response = getattr(_self, cmd)(*args, **kwargs)
        if not isinstance(response, (list, tuple)):
            response = (response,)
        self.instantiate(return_placeholder, response)

    def build_placeholders(self, obj):
        """
        Replace in an object all FrameworkTensors with Placeholder ids
        """
        if isinstance(obj, (tuple, list)):
            r = [self.build_placeholders(o) for o in obj]
            return type(obj)(r)
        elif isinstance(obj, dict):
            return {key: self.build_placeholders(value) for key, value in obj.items()}
        elif isinstance(obj, FrameworkTensor):
            if obj.id in self.placeholders:
                return self.placeholders[obj.id].id
            placeholder = PlaceHolder(id=obj.id, owner=self.owner)
            self.placeholders[placeholder.id.value] = placeholder  # TODO clean the .id.value
            return placeholder.id
        elif isinstance(obj, (int, float, str, bool, torch.dtype, torch.Size)):
            return obj
        else:
            return None

    def fecth_placeholders_from_ids(self, obj):
        """
        Replace in an object all ids with Placeholders
        """
        if isinstance(obj, (tuple, list)):
            r = [self.fecth_placeholders_from_ids(o) for o in obj]
            return type(obj)(r)
        elif isinstance(obj, dict):
            return {key: self.fecth_placeholders_from_ids(value) for key, value in obj.items()}
        elif isinstance(obj, PlaceholderId):
            return self.placeholders[obj.value]
        else:
            return obj

    @staticmethod
    def instantiate(placeholder, response):
        """
        Utility function to instantiate recursively an object containing placeholders with a similar object but containing tensors
        """
        # TODO should this be in placeholder.py instead?
        if placeholder is not None:
            if isinstance(placeholder, PlaceHolder):
                placeholder.instantiate(response)
            elif isinstance(placeholder, (list, tuple)):
                for ph, rep in zip(placeholder, response):
                    Role.instantiate(ph, rep)
            else:
                raise ValueError(
                    f"Response of type {type(response)} is not supported in plan actions"
                )

    @staticmethod
    def simplify(worker: AbstractWorker, role: "Role") -> tuple:
        """
        This function takes the attributes of a Role and saves them in a tuple
        Args:
            worker (AbstractWorker): the worker doing the serialization
            role (Role): a Role object
        Returns:
            tuple: a tuple holding the attributes of the Role object
        """
        return (
            sy.serde.msgpack.serde._simplify(worker, role.id),
            sy.serde.msgpack.serde._simplify(worker, role.actions),
            sy.serde.msgpack.serde._simplify(worker, role.state),
            sy.serde.msgpack.serde._simplify(worker, role.placeholders),
            role.input_placeholder_ids,
            role.output_placeholder_ids,
        )

    @staticmethod
    def detail(worker: AbstractWorker, role_tuple: "Role") -> tuple:
        """
        This function reconstructs a Role object given its attributes in the form of a tuple.
        Args:
            worker: the worker doing the deserialization
            role_tuple: a tuple holding the attributes of the Role
        Returns:
            role: a Role object
        """
        (
            id_,
            actions,
            state,
            placeholders,
            input_placeholder_ids,
            output_placeholder_ids,
        ) = role_tuple

        id_ = sy.serde.msgpack.serde._detail(worker, id_)
        actions = sy.serde.msgpack.serde._detail(worker, actions)
        state = sy.serde.msgpack.serde._detail(worker, state)
        placeholders = sy.serde.msgpack.serde._detail(worker, placeholders)

        return Role(
            id=id_,
            owner=worker,
            actions=actions,
            state=state,
            placeholders=placeholders,
            input_placeholder_ids=input_placeholder_ids,
            output_placeholder_ids=output_placeholder_ids,
        )
