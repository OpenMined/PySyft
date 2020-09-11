from typing import Dict
from typing import List
from typing import Tuple
from typing import Union
from typing import Callable

import syft
from syft.generic.frameworks import framework_packages

import syft as sy
from syft.execution.action import Action
from syft.execution.placeholder import PlaceHolder
from syft.execution.placeholder_id import PlaceholderId
from syft.execution.state import State
from syft.execution.tracing import FrameworkWrapper
from syft.generic.frameworks.types import FrameworkTensor
from syft.generic.abstract.syft_serializable import SyftSerializable
from syft.workers.abstract import AbstractWorker

from syft_proto.execution.v1.role_pb2 import Role as RolePB


class Role(SyftSerializable):
    """
    Roles will mainly be used to build protocols but are still a work in progress.
    """

    def __init__(
        self,
        id: Union[str, int] = None,
        worker: AbstractWorker = None,
        state: State = None,
        actions: List[Action] = None,
        placeholders: Dict[Union[str, int], PlaceHolder] = None,
        input_placeholder_ids: Tuple[int, str] = None,
        output_placeholder_ids: Tuple[int, str] = None,
    ):
        self.id = id or sy.ID_PROVIDER.pop()
        self.worker = worker or sy.local_worker

        self.actions = actions or []

        # All placeholders
        self.placeholders = placeholders or {}
        # Input placeholders, stored by id
        self.input_placeholder_ids = input_placeholder_ids or ()
        # Output placeholders
        self.output_placeholder_ids = output_placeholder_ids or ()

        self.state = state or State()
        self.tracing = False

        for name, package in framework_packages.items():
            tracing_wrapper = FrameworkWrapper(package=package, role=self)
            setattr(self, name, tracing_wrapper)

    def input_placeholders(self):
        return [self.placeholders[id_] for id_ in self.input_placeholder_ids]

    def output_placeholders(self):
        return [self.placeholders[id_] for id_ in self.output_placeholder_ids]

    @staticmethod
    def nested_object_traversal(obj: any, leaf_function: Callable, leaf_type: type):
        """
        Class method to iterate through a tree-like structure, where the branching is determined
        by the elements of list, tuples and dicts, returning the same tree-like structure with a
        function applied to its leafs.

        Args:
            obj: The tree-like structure, can be only the root as well.
            leaf_function: The function to be applied on the leaf nodes of the tree-like structure.
            leaf_type: On what type on function to apply the function, if the types won't match,
            the leaf is returned, to apply on all leafs pass any.

        Returns:
            Same structure as the obj argument, but with the function applied to the leaf elements.
        """
        if isinstance(obj, (list, tuple)):
            result = [Role.nested_object_traversal(elem, leaf_function, leaf_type) for elem in obj]
            return type(obj)(result)
        elif isinstance(obj, dict):
            return {
                k: Role.nested_object_traversal(v, leaf_function, leaf_type)
                for k, v in sorted(obj.items())
            }
        elif isinstance(obj, leaf_type):
            return leaf_function(obj)
        else:
            return obj

    def register_input(self, arg_):
        """Takes input argument for this role and generate placeholder."""
        self.input_placeholder_ids += (self._store_placeholders(arg_).value,)

    def register_inputs(self, args_):
        """Takes input arguments for this role and generate placeholders."""
        # TODO Should we be able to rebuild?
        def traversal_function(obj):
            if obj.id.value not in self.placeholders:
                self.placeholders[obj.id.value] = obj
            self.input_placeholder_ids.append(obj.id.value)

        self.input_placeholder_ids = []
        Role.nested_object_traversal(args_, traversal_function, PlaceHolder)
        self.input_placeholder_ids = tuple(self.input_placeholder_ids)

    def register_output(self, result):
        """Takes output tensor for this role and generate placeholder."""
        self.output_placeholder_ids += (self._store_placeholders(result).value,)

    def register_outputs(self, results):
        """Takes output tensors for this role and generate placeholders."""

        def traversal_function(obj):
            if obj.id.value not in self.placeholders:
                self.placeholders[obj.id.value] = obj
            self.output_placeholder_ids.append(obj.id.value)

        results = (results,) if not isinstance(results, tuple) else results
        self.output_placeholder_ids = []
        Role.nested_object_traversal(results, traversal_function, PlaceHolder)
        self.output_placeholder_ids = tuple(self.output_placeholder_ids)

    def register_action(self, traced_action, action_type):
        """Build placeholders and store action."""
        command, response = traced_action
        command_placeholder_ids = self._store_placeholders(command)
        return_placeholder_ids = None

        if response is not None:
            return_placeholder_ids = self._store_placeholders(response)
            if not isinstance(return_placeholder_ids, (list, tuple)):
                return_placeholder_ids = (return_placeholder_ids,)

        action = action_type(*command_placeholder_ids, return_ids=return_placeholder_ids)
        self.actions.append(action)

    def register_state_tensor(self, tensor):
        placeholder = sy.PlaceHolder(id=tensor.id, role=self)
        placeholder.instantiate(tensor)
        self.state.state_placeholders.append(placeholder)
        # TODO isn't it weird that state placeholders are both in state and plan?
        self.placeholders[tensor.id] = placeholder

    def reset(self):
        """Remove the trace actions on this Role to make it possible to build
        a Plan or a Protocol several times.
        """
        self.actions = []
        self.input_placeholder_ids = ()
        self.output_placeholder_ids = ()
        # We don't want to remove placeholders coming from the state
        state_ph_ids = [ph.id.value for ph in self.state.state_placeholders]
        self.placeholders = {
            ph_id: ph for ph_id, ph in self.placeholders.items() if ph_id in state_ph_ids
        }

    def execute(self):
        """Make the role execute all its actions."""
        for action in self.actions:
            self._execute_action(action)

        output_placeholders = tuple(
            self.placeholders[output_id] for output_id in self.output_placeholder_ids
        )

        return tuple(p.child for p in output_placeholders)

    def load(self, tensor):
        """Load tensors used in a protocol from worker's local store"""
        # TODO mock for now, load will use worker's store in a future work
        if self.tracing:
            return PlaceHolder.create_from(tensor, role=self, tracing=True)
        else:
            return tensor

    def load_state(self):
        """Load tensors used in a protocol from worker's local store"""
        return self.state.read()

    def instantiate_inputs(self, args_):
        """Takes input arguments for this role and generate placeholders."""

        def traversal_function(obj):
            placeholder = input_placeholders.pop(0)
            placeholder.instantiate(obj)

        input_placeholders = [
            self.placeholders[input_id] for input_id in self.input_placeholder_ids
        ]

        Role.nested_object_traversal(args_, traversal_function, FrameworkTensor)

    def _execute_action(self, action):
        """Build placeholders and store action."""
        cmd, _self, args_, kwargs_, return_values = (
            action.name,
            action.target,  # target is equivalent to the "self" in a method
            action.args,
            action.kwargs,
            action.return_ids,
        )
        _self = self._fetch_placeholders_from_ids(_self)
        args_ = self._fetch_placeholders_from_ids(args_)
        kwargs_ = self._fetch_placeholders_from_ids(kwargs_)
        return_values = self._fetch_placeholders_from_ids(return_values)

        # We can only instantiate placeholders, filter them
        return_placeholders = []
        Role.nested_object_traversal(
            return_values, lambda ph: return_placeholders.append(ph), PlaceHolder
        )

        if _self is None:
            method = self._fetch_package_method(cmd)
            response = method(*args_, **kwargs_)
        else:
            response = getattr(_self, cmd)(*args_, **kwargs_)

        if not isinstance(response, (tuple, list)):
            response = (response,)

        PlaceHolder.instantiate_placeholders(return_placeholders, response)

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

        def traversal_function(obj):
            if obj.id.value not in self.placeholders:
                self.placeholders[obj.id.value] = obj
            return obj.id

        return Role.nested_object_traversal(obj, traversal_function, PlaceHolder)

    def _fetch_placeholders_from_ids(self, obj):
        """
        Replace in an object all ids with Placeholders
        """
        return Role.nested_object_traversal(
            obj, lambda x: self.placeholders[x.value], PlaceholderId
        )

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
            new_ph = PlaceHolder(id=old_ids_2_new_ids[ph.id.value]).instantiate(ph.child)
            state_placeholders.append(new_ph)

        state = State(state_placeholders)

        _replace_placeholder_ids = lambda obj: Role.nested_object_traversal(
            obj, lambda x: PlaceholderId(old_ids_2_new_ids[x.value]), PlaceholderId
        )

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

    def _get_command_framework(self, action: Action):
        """Helper method that returns framework module associated with command."""
        if action.target is None:
            framework_name, command = action.name.split(".", 2)
            return getattr(syft, framework_name, syft.framework)

        elif isinstance(action.target, PlaceholderId):
            ph = self.placeholders.get(action.target.value, None)
            if ph is not None:
                framework_name = ph.child.__module__.split(".")[0]
                return getattr(syft, framework_name, syft.framework)

        return None

    def _is_inplace_action(self, action: Action):
        """
        Helper method that returns True if action contains inplace operation.
        """
        framework = self._get_command_framework(action)
        if framework is not None:
            return framework.is_inplace_method(action.name.split(".")[-1])
        else:
            return False

    def _is_state_change_action(self, action: Action):
        """
        Helper method that returns True if action affects module state.
        """
        framework = self._get_command_framework(action)
        if framework is not None:
            return framework.is_global_state_change_method(action.name.split(".")[-1])
        else:
            return False

    def _prune_actions(self):
        """
        Removes unnecessary actions and placeholders.
        """

        def action_affects_placeholder_ids(action: Action, ids: set, inplace_only=False) -> bool:
            """Returns true if action updates provided placeholder ids"""

            # Operation has side-effect on connected placeholder(s)
            target_ids = get_action_placeholder_ids(action, "target")
            affects_connected_ph = len(
                target_ids.intersection(ids)
            ) > 0 and self._is_inplace_action(action)

            if inplace_only or affects_connected_ph:
                return affects_connected_ph

            # Operation resulted in connected placeholder(s)
            return_ids = get_action_placeholder_ids(action, "return")
            returns_connected_ph = len(return_ids.intersection(ids)) > 0
            return returns_connected_ph

        def get_action_placeholder_ids(action, scope="all"):
            """Returns PlaceholderId's used by Action"""
            ids = set()
            attrs = {
                "all": ["target", "args", "kwargs", "return_ids"],
                "return": ["return_ids"],
                "target": ["target"],
            }
            for attr in attrs.get(scope):
                Role.nested_object_traversal(
                    getattr(action, attr), lambda ph_id: ids.add(ph_id.value), PlaceholderId
                )

            return ids

        def find_connected_placeholder_ids(start_action_idx, ph_id):
            """Returns all placeholders affecting given PlaceholderId (including itself)"""
            placeholders = {ph_id} if not isinstance(ph_id, set) else ph_id
            actions_idx = set()

            # We need to examine actions in the order opposite to execution order
            # To track placeholders that affected `id` starting from the given action
            # and above in the execution flow
            for action_idx_rev, action in enumerate(reversed(self.actions[: start_action_idx + 1])):
                action_idx = start_action_idx - action_idx_rev
                if action_affects_placeholder_ids(action, placeholders):
                    placeholders |= get_action_placeholder_ids(action)
                    actions_idx.add(action_idx)

            return placeholders, actions_idx

        connected_placeholder_ids = set()
        connected_actions_idx = set()

        # Find placeholders that affect output placeholders, starting from the last action
        if len(self.output_placeholder_ids) > 0:
            placeholder_ids, actions_idx = find_connected_placeholder_ids(
                len(self.actions) - 1, set(self.output_placeholder_ids)
            )
            connected_placeholder_ids |= placeholder_ids
            connected_actions_idx |= actions_idx

        # Find inputs that have side-effects and all placeholders connected with them
        input_ids = set(self.input_placeholder_ids)
        for action_idx, action in enumerate(self.actions):
            if action_affects_placeholder_ids(action, input_ids, inplace_only=True):
                target_ids = get_action_placeholder_ids(action, "target")
                placeholder_ids, actions_idx = find_connected_placeholder_ids(
                    action_idx, target_ids
                )
                connected_placeholder_ids |= placeholder_ids
                connected_actions_idx |= actions_idx

        # Remove actions that do not affect input/output placeholders
        # Exception is actions that affect module state, like `torch.manual_seed(n)`
        self.actions = [
            a
            for i, a in enumerate(self.actions)
            if i in connected_actions_idx or self._is_state_change_action(a)
        ]

        # Remove unused placeholders, except inputs
        connected_placeholder_ids |= input_ids
        self.placeholders = {
            ph_id: ph
            for ph_id, ph in self.placeholders.items()
            if ph_id in connected_placeholder_ids
        }

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
        This function reconstructs a Role object given its attributes in the form of a
        Protobuf message.

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

    @staticmethod
    def get_protobuf_schema() -> RolePB:
        return RolePB
