from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

import copy

from syft.generic.frameworks import framework_packages

import syft as sy
from syft.execution.action import Action
from syft.execution.placeholder import PlaceHolder
from syft.execution.placeholder_id import PlaceholderId
from syft.execution.state import State
from syft.generic.frameworks.types import FrameworkTensor
from syft.generic.object import AbstractObject
from syft.generic.object_storage import ObjectStorage
from syft.workers.abstract import AbstractWorker

from syft_proto.execution.v1.role_pb2 import Role as RolePB


class Role:
    """
    Roles will mainly be used to build protocols but are still a work in progress.
    """

    def __init__(
        self,
        state: State = None,
        actions: List[Action] = None,
        placeholders: Dict[Union[str, int], PlaceHolder] = None,
        input_placeholder_ids: Tuple[int, str] = None,
        output_placeholder_ids: Tuple[int, str] = None,
        # General kwargs
        id: Union[str, int] = None,
    ):
        self.id = id or sy.ID_PROVIDER.pop()

        self.actions = actions or []

        # All placeholders
        self.placeholders = placeholders or {}
        # Input placeholders, stored by id
        self.input_placeholder_ids = input_placeholder_ids or ()
        # Output placeholders
        self.output_placeholder_ids = output_placeholder_ids or ()

        self.state = state or State()

    def input_placeholders(self):
        return [self.placeholders[id_] for id_ in self.input_placeholder_ids]

    def output_placeholders(self):
        return [self.placeholders[id_] for id_ in self.output_placeholder_ids]

    def register_input(self, arg_):
        """ Takes input argument for this role and generate placeholder.
        """
        self.input_placeholder_ids += (self._store_placeholders(arg_).value,)

    def register_inputs(self, args_):
        """ Takes input arguments for this role and generate placeholders.
        """
        # TODO Should we be able to rebuild?
        self.input_placeholder_ids += tuple(
            self._store_placeholders(arg).value for arg in args_ if isinstance(arg, PlaceHolder)
        )

    def register_output(self, result):
        """ Takes output tensor for this role and generate placeholder.
        """
        self.output_placeholder_ids += (self._store_placeholders(result).value,)

    def register_outputs(self, results):
        """ Takes output tensors for this role and generate placeholders.
        """
        results = (results,) if not isinstance(results, tuple) else results
        results += tuple(self._store_placeholders(result) for result in results)
        self.output_placeholder_ids = tuple(
            result.value for result in results if isinstance(result, PlaceholderId)
        )

    def register_action(self, traced_action, action_type):
        """ Build placeholders and store action.
        """
        command, response = traced_action
        command_placeholder_ids = self._store_placeholders(command)
        return_placeholder_ids = None

        if response is not None:
            return_placeholder_ids = self._store_placeholders(response)
            if not isinstance(return_placeholder_ids, (list, tuple)):
                return_placeholder_ids = (return_placeholder_ids,)

        action = action_type(*command_placeholder_ids, return_ids=return_placeholder_ids)
        self.actions.append(action)

    def register_state_tensor(self, tensor, owner):
        placeholder = sy.PlaceHolder(id=tensor.id, role=self, owner=owner)
        placeholder.instantiate(tensor)
        self.state.state_placeholders.append(placeholder)
        # TODO isn't it weird that state placeholders are both in state and plan?
        self.placeholders[tensor.id] = placeholder

    def execute(self, args_):
        """ Make the role execute all its actions using args_ as inputs.
        """
        self._instantiate_inputs(args_)
        for action in self.actions:
            self._execute_action(action)

        output_placeholders = tuple(
            self.placeholders[output_id] for output_id in self.output_placeholder_ids
        )

        return tuple(p.child for p in output_placeholders)

    def _instantiate_inputs(self, args_):
        """ Takes input arguments for this role and generate placeholders.
        """
        input_placeholders = tuple(
            self.placeholders[input_id] for input_id in self.input_placeholder_ids
        )
        PlaceHolder.instantiate_placeholders(input_placeholders, args_)

    def _execute_action(self, action):
        """ Build placeholders and store action.
        """
        cmd, _self, args_, kwargs_, return_placeholder = (
            action.name,
            action.target,  # target is equivalent to the "self" in a method
            action.args,
            action.kwargs,
            action.return_ids,
        )
        _self = self._fetch_placeholders_from_ids(_self)
        args_ = self._fetch_placeholders_from_ids(args_)
        kwargs_ = self._fetch_placeholders_from_ids(kwargs_)
        return_placeholder = self._fetch_placeholders_from_ids(return_placeholder)

        if _self is None:
            method = self._fetch_package_method(cmd)
            response = method(*args_, **kwargs_)
        else:
            response = getattr(_self, cmd)(*args_, **kwargs_)

        if not isinstance(response, (tuple, list)):
            response = (response,)

        PlaceHolder.instantiate_placeholders(return_placeholder, response)

    def _fetch_package_method(self, cmd):
        cmd_path = cmd.split(".")

        package_name = cmd_path[0]
        subpackage_names = cmd_path[1:-1]
        method_name = cmd_path[-1]

        package = framework_packages[package_name]
        for subpackage_name in subpackage_names:
            package = getattr(package, subpackage_name)
        method = getattr(package, method_name)
        return method

    def _store_placeholders(self, obj):
        """
        Replace in an object all FrameworkTensors with Placeholder ids
        """
        if isinstance(obj, (tuple, list)):
            r = [self._store_placeholders(o) for o in obj]
            return type(obj)(r)
        elif isinstance(obj, dict):
            return {key: self._store_placeholders(value) for key, value in obj.items()}
        elif isinstance(obj, PlaceHolder):
            if obj.id.value not in self.placeholders:
                self.placeholders[obj.id.value] = obj
            return obj.id
        else:
            return obj

    def _fetch_placeholders_from_ids(self, obj):
        """
        Replace in an object all ids with Placeholders
        """
        if isinstance(obj, (tuple, list)):
            r = [self._fetch_placeholders_from_ids(o) for o in obj]
            return type(obj)(r)
        elif isinstance(obj, dict):
            return {key: self._fetch_placeholders_from_ids(value) for key, value in obj.items()}
        elif isinstance(obj, PlaceholderId):
            return self.placeholders[obj.value]
        else:
            return obj

    def copy(self):
        # TODO not the cleanest method ever
        placeholders = {}
        old_ids_2_new_ids = {}
        for ph in self.placeholders.values():
            copy = ph.copy()
            old_ids_2_new_ids[ph.id.value] = copy.id.value
            placeholders[copy.id.value] = copy

        new_input_placeholder_ids = tuple(
            old_ids_2_new_ids[self.placeholders[input_id].id.value]
            for input_id in self.input_placeholder_ids
        )
        new_output_placeholder_ids = tuple(
            old_ids_2_new_ids[self.placeholders[output_id].id.value]
            for output_id in self.output_placeholder_ids
        )

        state_placeholders = []
        for ph in self.state.state_placeholders:
            new_ph = PlaceHolder(id=old_ids_2_new_ids[ph.id.value], owner=ph.owner).instantiate(
                ph.child
            )
            state_placeholders.append(new_ph)

        state = State(state_placeholders)

        def _replace_placeholder_ids(obj):
            if isinstance(obj, (tuple, list)):
                r = [_replace_placeholder_ids(o) for o in obj]
                return type(obj)(r)
            elif isinstance(obj, dict):
                return {key: _replace_placeholder_ids(value) for key, value in obj.items()}
            elif isinstance(obj, PlaceholderId):
                return PlaceholderId(old_ids_2_new_ids[obj.value])
            else:
                return obj

        new_actions = []
        for action in self.actions:
            action_type = type(action)
            target = _replace_placeholder_ids(action.target)
            args_ = _replace_placeholder_ids(action.args)
            kwargs_ = _replace_placeholder_ids(action.kwargs)
            return_ids = _replace_placeholder_ids(action.return_ids)
            new_actions.append(action_type(action.name, target, args_, kwargs_, return_ids))

        return Role(
            state=state,
            actions=new_actions,
            placeholders=placeholders,
            input_placeholder_ids=new_input_placeholder_ids,
            output_placeholder_ids=new_output_placeholder_ids,
            id=sy.ID_PROVIDER.pop(),
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

        # TODO should state.state_placeholders be a dict as self.placeholders?
        # Then, if placeholder not found in self.placeholders, fetch it from
        # state.state_placeholders. This would prevent us from having the following lines.
        # Or need to rethink states
        for ph in state.state_placeholders:
            placeholders[ph.id.value] = ph

        role = Role(
            id=id_,
            actions=actions,
            input_placeholder_ids=input_placeholder_ids,
            output_placeholder_ids=output_placeholder_ids,
        )
        for ph in placeholders.values():
            ph.role = role
        for ph in state.state_placeholders:
            ph.role = role

        role.placeholders = placeholders
        role.state = state

        return role

    @staticmethod
    def bufferize(worker: AbstractWorker, role: "Role") -> tuple:
        """
        This function takes the attributes of a Role and saves them in a Protobuf message
        Args:
            worker (AbstractWorker): the worker doing the serialization
            role (Role): a Role object
        Returns:
            RolePB: a Protobuf message holding the unique attributes of the Role object
        """
        protobuf_role = RolePB()

        sy.serde.protobuf.proto.set_protobuf_id(protobuf_role.id, role.id)

        protobuf_actions = [
            sy.serde.protobuf.serde._bufferize(worker, action) for action in role.actions
        ]
        protobuf_role.actions.extend(protobuf_actions)

        protobuf_role.state.CopyFrom(sy.serde.protobuf.serde._bufferize(worker, role.state))

        protobuf_placeholders = [
            sy.serde.protobuf.serde._bufferize(worker, placeholder)
            for placeholder in role.placeholders.values()
        ]
        protobuf_role.placeholders.extend(protobuf_placeholders)

        for id_ in role.input_placeholder_ids:
            sy.serde.protobuf.proto.set_protobuf_id(protobuf_role.input_placeholder_ids.add(), id_)
        for id_ in role.output_placeholder_ids:
            sy.serde.protobuf.proto.set_protobuf_id(protobuf_role.output_placeholder_ids.add(), id_)

        return protobuf_role

    @staticmethod
    def unbufferize(worker: AbstractWorker, protobuf_role: RolePB) -> tuple:
        """
        This function reconstructs a Role object given its attributes in the form of a Protobuf message.
        Args:
            worker: the worker doing the deserialization
            protobuf_role: a Protobuf message holding the attributes of the Role
        Returns:
            role: a Role object
        """
        id_ = sy.serde.protobuf.proto.get_protobuf_id(protobuf_role.id)

        actions = [
            sy.serde.protobuf.serde._unbufferize(worker, action) for action in protobuf_role.actions
        ]

        state = sy.serde.protobuf.serde._unbufferize(worker, protobuf_role.state)

        placeholders = [
            sy.serde.protobuf.serde._unbufferize(worker, placeholder)
            for placeholder in protobuf_role.placeholders
        ]
        placeholders = {placeholder.id.value: placeholder for placeholder in placeholders}
        # TODO should state.state_placeholders be a dict as self.placeholders?
        # Then, if placeholder not found in self.placeholders, fetch it from
        # state.state_placeholders. This would prevent us from having the following lines.
        # Or need to rethink states
        for ph in state.state_placeholders:
            placeholders[ph.id.value] = ph

        input_placeholder_ids = tuple(
            sy.serde.protobuf.proto.get_protobuf_id(ph_id)
            for ph_id in protobuf_role.input_placeholder_ids
        )
        output_placeholder_ids = tuple(
            sy.serde.protobuf.proto.get_protobuf_id(ph_id)
            for ph_id in protobuf_role.output_placeholder_ids
        )

        role = Role(
            id=id_,
            actions=actions,
            input_placeholder_ids=input_placeholder_ids,
            output_placeholder_ids=output_placeholder_ids,
        )
        for ph in placeholders.values():
            ph.role = role
        for ph in state.state_placeholders:
            ph.role = role

        role.placeholders = placeholders
        role.state = state

        return role
