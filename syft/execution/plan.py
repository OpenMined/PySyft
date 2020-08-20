from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

import copy
import inspect
import io
import torch
import warnings
import traceback

import syft as sy
from syft.execution.placeholder import PlaceHolder
from syft.execution.role import Role
from syft.execution.tracing import FrameworkWrapper
from syft.execution.type_wrapper import NestedTypeWrapper
from syft.execution.translation.abstract import AbstractPlanTranslator
from syft.execution.translation.default import PlanTranslatorDefault
from syft.execution.translation.torchscript import PlanTranslatorTorchscript
from syft.execution.translation.threepio import PlanTranslatorTfjs
from syft.execution.translation import TranslationTarget
from syft.generic.frameworks import framework_packages
from syft.generic.frameworks.types import FrameworkTensor
from syft.generic.frameworks.types import FrameworkLayerModule
from syft.generic.abstract.sendable import AbstractSendable
from syft.generic.pointers.pointer_plan import PointerPlan
from syft.workers.abstract import AbstractWorker
from syft.frameworks.torch.tensors.interpreters.autograd import AutogradTensor

from syft_proto.execution.v1.plan_pb2 import Plan as PlanPB


class func2plan(object):
    """Decorator which converts a function to a plan.

    Converts a function containing sequential pytorch code into
    a plan object which can be sent to any arbitrary worker.

    This class should be used only as a decorator.
    """

    def __init__(self, args_shape=None, state=None, trace_autograd=False, args_dtypes=()):
        self.args_shape = args_shape
        self.args_dtypes = args_dtypes
        self.state_tensors = state or ()
        # include_state is used to distinguish if the initial plan is a function or a class:
        # if it's a function, then the state should be provided in the args, so include_state
        # will be true. And to know if it was indeed a function, we just need to see if a
        # "manual" state was provided.
        self.include_state = state is not None
        self.trace_autograd = trace_autograd

    def __call__(self, plan_function):
        plan = Plan(
            name=plan_function.__name__,
            include_state=self.include_state,
            forward_func=plan_function,
            state_tensors=self.state_tensors,
            id=sy.ID_PROVIDER.pop(),
            owner=sy.local_worker,
        )

        # Build the plan automatically
        if self.args_shape:
            args_ = PlaceHolder.create_placeholders(self.args_shape, self.args_dtypes)
            try:
                plan.build(*args_, trace_autograd=self.trace_autograd)
            except TypeError as e:
                raise ValueError(
                    "Automatic build using @func2plan failed!\nCheck that:\n"
                    " - you have provided the correct number of shapes in args_shape\n"
                    " - you have no simple numbers like int or float as args. If you do "
                    "so, please consider using a tensor instead."
                )
        return plan


class Plan(AbstractSendable):
    """
    A Plan stores a sequence of actions, just like a function.

    A Plan is intended to store a sequence of actions, just like a function,
    but it allows to send this sequence of actions to remote workers and to keep a
    reference to it. This way, to compute remotely this sequence of actions on some remote
    input referenced through pointers, instead of sending multiple messages you need now to send a
    single message with the references of the plan and the pointers.

    Specifically, a Plan contains only ComputationAction and does not concern itself with
    operations covered by CommunicationAction. Use Protocol to cover both types of actions.

    All arguments are optional.

    Args:
        name: the name of the name
        state: store the plan tensors like model parameters
        include_state: if true, implies that the plan is a function, else a class. If true, the
            state is re-integrated in the args to be accessed within the function
        is_built: state if the plan has already been built.
        placeholders: dict of placeholders used in the plan
        actions: list of commands (called actions)
        forward_func: the function to be transformed into a plan
        state_tensors: a tuple of state elements. It can be used to populate a state
        id: plan id
        owner: plan owner
        tags: plan tags
        description: plan description
        base_framework: The underlying framework (pytorch, tensorflow) which the
                        plan is to be executed in
        frameworks: A list of frameworks which the plan will also support
    """

    _build_translators = []
    _wrapped_frameworks = {}

    def __init__(
        self,
        name: str = None,
        include_state: bool = False,
        is_built: bool = False,
        forward_func=None,
        state_tensors=[],
        role: Role = None,
        # General kwargs
        id: Union[str, int] = None,
        owner: "sy.workers.BaseWorker" = None,
        tags: List[str] = None,
        input_types: list = None,
        description: str = None,
        roles: Dict[str, Role] = None,
        base_framework: str = TranslationTarget.PYTORCH.value,
    ):
        super().__init__(id, owner, tags, description, child=None)

        # Plan instance info
        self.name = name or self.__class__.__name__

        self.role = role or Role()

        if role is None:
            for st in state_tensors:
                self.role.register_state_tensor(st)

        self.include_state = include_state
        self.is_building = False
        self.state_attributes = {}
        self.is_built = is_built
        self.torchscript = None
        self.input_types = input_types
        self.validate_input_types = True
        self.tracing = False
        self._base_framework = base_framework
        self.roles = roles or {base_framework: self.role}

        # The plan has not been sent so it has no reference to remote locations
        self.pointers = {}

        if not hasattr(self, "forward"):
            self.forward = forward_func or None

        self.__name__ = self.__repr__()  # For PyTorch jit tracing compatibility

        # List of available translations
        self.translations = []

    @property
    def state(self):
        return self.role.state

    @property
    def actions(self):
        return self.role.actions

    @property
    def base_framework(self):
        return self._base_framework

    @base_framework.setter
    def base_framework(self, val):
        if val in self.roles:
            self._base_framework = val
            self.role = self.roles[self._base_framework]
            return
        raise ValueError(
            "Value given does not match any available Roles."
            " Please check to see if the proper translations have been added to Plan."
        )

    def parameters(self):
        """
        This is defined to match the torch api of nn.Module where .parameters()

        Returns:
            The model tensors / parameters
        """
        if self.state is not None:
            return self.state.tensors()
        else:
            return []

    def build(self, *args, trace_autograd=False):
        """Builds the plan.

        First, run the function to be converted in a plan in a context which
        activates the tracing and record the actions in trace.logs

        Second, store the result ids temporarily to helper ordering the output
        placeholders at return time

        Third, loop through the trace logs and replace the tensors found in the
        actions logged by PlaceHolders. Record those actions in
        plan.actions

        Args:
            args: Input arguments to run the plan
        """
        # Reset previous build
        self.role.reset()

        def build_nested_arg(arg, leaf_function):
            if isinstance(arg, list):
                return [build_nested_arg(obj, leaf_function) for obj in arg]
            elif isinstance(arg, tuple):
                return tuple(build_nested_arg(obj, leaf_function) for obj in arg)
            elif isinstance(arg, dict):
                return {k: build_nested_arg(v, leaf_function) for k, v in arg.items()}
            else:
                return leaf_function(arg)

        # Enable tracing
        self.toggle_tracing(True)
        self.is_building = True

        # Check the types
        self.input_types = NestedTypeWrapper(args)

        # Run once to build the plan
        if trace_autograd:
            # Wrap arguments that require gradients with AutogradTensor,
            # to be able to trace autograd operations
            args = build_nested_arg(
                args,
                lambda x: AutogradTensor().on(x, wrap=False)
                if isinstance(x, FrameworkTensor)
                else PlaceHolder.create_from(x, role=self.role, tracing=True),
            )
            # Add Placeholder after AutogradTensor in the chain
            # so that all operations that happen inside AutogradTensor are recorded by Placeholder
            args_placeholders = build_nested_arg(
                args,
                lambda x: PlaceHolder.insert(x, AutogradTensor, role=self.role, tracing=True)
                if not isinstance(x, PlaceHolder)
                else x,
            )
        else:
            # Add Placeholder on top of each arg
            args = args_placeholders = build_nested_arg(
                args, lambda x: PlaceHolder.create_from(x, role=self.role, tracing=True)
            )

        # Add state to args if needed
        if self.include_state:
            args += (self.state,)

        # Check the plan arguments to see what framework wrappers we might need to send to the plan
        framework_kwargs = {}

        forward_args = inspect.getfullargspec(self.forward).args
        for f_name, wrap_framework_func in Plan._wrapped_frameworks.items():
            if f_name in forward_args:
                framework_kwargs[f_name] = wrap_framework_func(self.role)

        results = self.forward(*args, **framework_kwargs)

        # Register inputs in role
        self.role.register_inputs(args_placeholders)

        # Register outputs in role

        results_placeholders = PlaceHolder.recursive_extract(results)
        self.role.register_outputs(results_placeholders)

        # Disable tracing
        self.toggle_tracing(False)
        self.is_building = False
        self.role._prune_actions()
        self.is_built = True

        # Build registered translations
        for translator in Plan._build_translators:
            try:
                self.add_translation(translator)
                self.translations.append(translator)
            except:
                warnings.warn(
                    f"Failed to translate Plan with {translator.__name__}: {traceback.format_exc()}"
                )

        return results

    def toggle_tracing(self, value=None):
        self.tracing = value if value is not None else not self.tracing
        self.state.tracing = self.tracing
        for ph in self.role.placeholders.values():
            ph.tracing = self.tracing

    def copy(self):
        """Creates a copy of a plan."""
        plan_copy = Plan(
            name=self.name,
            role=self.role.copy(),
            include_state=self.include_state,
            is_built=self.is_built,
            id=sy.ID_PROVIDER.pop(),
            owner=self.owner,
            tags=self.tags,
            input_types=self.input_types,
            description=self.description,
            base_framework=self._base_framework,
            roles={fw_name: role.copy() for fw_name, role in self.roles.items()},
        )

        plan_copy.torchscript = self.torchscript

        return plan_copy

    def __setattr__(self, name, value):
        """Add new tensors or parameter attributes to the state and register them
        in the owner's registry
        """
        if isinstance(value, torch.jit.ScriptModule):
            object.__setattr__(self, name, value)
        elif isinstance(value, FrameworkTensor):
            self.role.register_state_tensor(value)
            self.state_attributes[name] = value
        elif isinstance(value, FrameworkLayerModule):
            for param in value.parameters():
                self.role.register_state_tensor(param)
            self.state_attributes[name] = value
        else:
            object.__setattr__(self, name, value)

    def __getattr__(self, name):
        if name not in self.state_attributes:
            raise AttributeError("State attribute not found.")

        value = self.state_attributes[name]
        if not self.is_building:
            return value

        if isinstance(value, FrameworkTensor):
            return self.role.placeholders[value.id]
        elif isinstance(value, FrameworkLayerModule):
            # We need to deepcopy here otherwise the real layer is modified when the
            # Plan is being built
            copied_layer = copy.deepcopy(value)
            for copied_param, param in zip(copied_layer.named_parameters(), value.parameters()):
                (copied_name, _) = copied_param
                copied_layer._parameters[copied_name] = self.role.placeholders[param.id]

            return copied_layer

    def __call__(self, *args):
        """
        Calls a plan execution with some arguments.

        When possible, run the original function to improve efficiency. When
        it's not, for example if you fetched the plan from a remote worker,
        then run it from the tape of actions:
        - Instantiate input placeholders
        - for each recorded action, run the action on the placeholders
          and use the result(s) to instantiate to appropriate placeholder.
        - Return the instantiation of all the output placeholders.
        """
        if self.forward is not None:
            if self.include_state:
                args = (*args, self.state)
            return self.forward(*args)
        else:
            if self.validate_input_types:
                self.input_types.input_check(self, args)
            self.role.instantiate_inputs(args)
            result = self.role.execute()
            if len(result) == 1:
                return result[0]
            return result

    def run(self, args_: Tuple, result_ids: List[Union[str, int]]):
        """Controls local or remote plan execution.
        If the plan doesn't have the plan built, first build it using the original function.

        Args:
            args_: Arguments used to run plan.
            result_ids: List of ids where the results will be stored.
        """
        # TODO: can we reuse result_ids?
        return self.__call__(*args_)

    def send(self, *locations: AbstractWorker) -> PointerPlan:
        """Send plan to locations.

        If the plan was not built locally it will raise an exception.
        If `force` = true plan is going to be sent either way.

        Args:
            locations: List of workers.
            force: A boolean indicating if this action should be forced.
        """
        if not self.is_built:
            raise RuntimeError("A plan needs to be built before being sent to a worker.")

        if len(locations) == 1:
            location = locations[0]

            # Check if plan was already sent at the location
            if location in self.pointers:
                return self.pointers[location]

            # Send the Plan
            pointer = self.owner.send(self, workers=location)

            self.pointers[location] = pointer
        else:
            ids_at_location = []
            for location in locations:
                if location in self.pointers:
                    # Use the pointer that was already sent
                    pointer = self.pointers[location]
                else:
                    # Send the Plan
                    pointer = self.owner.send(self, workers=location)

                    self.pointers[location] = pointer

                ids_at_location.append(pointer.id_at_location)

            pointer = sy.PointerPlan(location=locations, id_at_location=ids_at_location)

        return pointer

    def get_args_shape(self):
        """Returns input tensors shapes"""
        if not self.is_built:
            raise RuntimeError("A plan needs to be built before input shapes can be known.")

        return [ph.expected_shape for ph in self.role.input_placeholders()]

    def create_dummy_args(self):
        """Returns dummy arguments matching built Plan arguments' types"""
        if not self.is_built:
            raise RuntimeError("A plan needs to be built before input shapes can be known.")

        def traverse_nested_types(arg, leaf_function):
            if isinstance(arg, list):
                return [traverse_nested_types(obj, leaf_function) for obj in arg]
            elif isinstance(arg, tuple):
                return tuple(traverse_nested_types(obj, leaf_function) for obj in arg)
            elif isinstance(arg, dict):
                return {k: traverse_nested_types(v, leaf_function) for k, v in arg.items()}
            else:
                return leaf_function(arg)

        input_placeholders = (ph for ph in self.role.input_placeholders())

        def create_dummy(input_type, input_placeholder):
            if issubclass(input_type, FrameworkTensor):
                tensors = PlaceHolder.create_placeholders(
                    [input_placeholder.expected_shape], [input_placeholder.expected_dtype]
                )
                var = tensors[0]
                if input_type != type(var):
                    var = input_type(var)
                return var
            else:
                return input_type()

        return traverse_nested_types(
            self.input_types.nested_input_types,
            lambda input_type: create_dummy(input_type, input_placeholders.__next__()),
        )

    @staticmethod
    def register_build_translator(translator: "AbstractPlanTranslator"):
        Plan._build_translators.append(translator)

    @staticmethod
    def register_framework(f_name, f_package):
        """
        When we use methods defined in a framework (like: torch.randn) we have a framework
        wrapper that helps as register and keep track of what methods are called
        With the below lines, we "register" what frameworks we have support to handle
        Args:
            f_name (String): framework name (eg. torch, crypten)
            f_package (imported module): imported library
        """

        def call_wrapped_framework(role):
            return FrameworkWrapper(f_package, role)

        Plan._wrapped_frameworks[f_name] = call_wrapped_framework

    def add_translation(self, plan_translator: "AbstractPlanTranslator"):
        role = plan_translator(self).translate()
        if isinstance(role, Role):
            self.roles[plan_translator.framework] = role
        return self

    def remove_translation(self, plan_translator: "AbstractPlanTranslator" = PlanTranslatorDefault):
        plan_translator(self).remove()
        return self

    def get_(self):
        self.state.get_()
        return self

    get = get_

    def get_pointers(self):
        return self.pointers

    def fix_precision_(self, *args, **kwargs):
        self.state.fix_precision_(*args, **kwargs)
        return self

    fix_precision = fix_prec_ = fix_prec = fix_precision_

    def float_precision_(self):
        self.state.float_precision_()
        return self

    float_precision = float_prec_ = float_prec = float_precision_

    def share_(self, *args, **kwargs):
        self.state.share_(*args, **kwargs)
        return self

    share = share_

    def create_pointer(
        self, owner, garbage_collect_data, location=None, id_at_location=None, tags=None, **kwargs
    ):
        """
        Create a pointer to the plan

        Args:
            owner: the owner of the pointer
            garbage_collect_data: if true, when the pointer is deleted, the remote target
                        is garbaged collected
            location: the location of the pointer
            id_at_location: the remote id at location
            tags: the tags inherited from the Plan

        Returns:
            PointerPlan: pointer to the plan
        """
        return PointerPlan(
            owner=owner,
            location=location or self.owner,
            id_at_location=id_at_location or self.id,
            garbage_collect_data=garbage_collect_data,
            tags=tags,
        )

    def __str__(self):
        """Returns the string representation of Plan."""
        out = "<"
        out += str(type(self)).split("'")[1].split(".")[-1]
        out += " " + str(self.name)
        out += " id:" + str(self.id)
        out += " owner:" + str(self.owner.id)

        if self.tags is not None and len(self.tags):
            out += " Tags:"
            for tag in self.tags:
                out += " " + str(tag)

        if self.is_built:
            out += " built"

        out += ">"
        out += "\n"
        _self = self

        # out += f"def {self.name}("
        # out += ", ".join(f"arg_{extract_tag(p)}" for p in self.find_placeholders("input"))
        # out += "):\n"
        # for action in self.actions:
        #     line = "    "
        #     if action.return_ids is not None:
        #         if isinstance(action.return_ids, PlaceHolder):
        #             tag = extract_tag(action.return_ids)
        #             line += f"_{tag} = "
        #         elif isinstance(action.return_ids, tuple):
        #             line += (
        #                 ", ".join(
        #                     f"_{extract_tag(o)}" if isinstance(o, PlaceHolder) else str(o)
        #                     for o in action.return_ids
        #                 )
        #                 + " = "
        #             )
        #         else:
        #             line += str(action.return_ids) + " = "
        #     if action.target is not None:
        #         line += f"_{extract_tag(self.placeholders[action.target.value])}."
        #     line += action.name + "("
        #     line += ", ".join(
        #         f"_{extract_tag(arg)}" if isinstance(arg, PlaceHolder) else str(arg)
        #         for arg in action.args
        #     )
        #     if action.kwargs:
        #         line += ", " + ", ".join(f"{k}={w}" for k, w in action.kwargs.items())
        #     line += ")\n"
        #     out += line

        # out += "    return "
        # out += ", ".join(f"_{extract_tag(p)}" for p in self.find_placeholders("output"))

        return out

    def __repr__(self):
        return self.__str__()

    @staticmethod
    def replace_non_instanciated_placeholders(plan: "Plan") -> "Plan":
        # Replace non-instanciated placeholders from plan.placeholders by instanciated placeholders
        # from state.state_placeholders
        # NOTE Maybe state shouldn't contain instanciated placeholders but values directly?
        state_placeholders = {ph.id.value: ph for ph in plan.state.state_placeholders}
        plan.placeholders = {**plan.placeholders, **state_placeholders}

        return plan

    @staticmethod
    def simplify(worker: AbstractWorker, plan: "Plan") -> tuple:
        """
        This function takes the attributes of a Plan and saves them in a tuple
        Args:
            worker (AbstractWorker): the worker doing the serialization
            plan (Plan): a Plan object
        Returns:
            tuple: a tuple holding the unique attributes of the Plan object

        """
        if not plan.is_built:
            raise RuntimeError("A Plan needs to be built before being serialized.")

        return (
            sy.serde.msgpack.serde._simplify(worker, plan.id),
            sy.serde.msgpack.serde._simplify(worker, plan.role),
            sy.serde.msgpack.serde._simplify(worker, plan.include_state),
            sy.serde.msgpack.serde._simplify(worker, plan.name),
            sy.serde.msgpack.serde._simplify(worker, plan.tags),
            sy.serde.msgpack.serde._simplify(worker, plan.description),
            sy.serde.msgpack.serde._simplify(worker, plan.torchscript),
            sy.serde.msgpack.serde._simplify(worker, plan.input_types),
            sy.serde.msgpack.serde._simplify(worker, plan._base_framework),
            sy.serde.msgpack.serde._simplify(worker, plan.roles),
        )

    @staticmethod
    def detail(worker: AbstractWorker, plan_tuple: tuple) -> "Plan":
        """This function reconstructs a Plan object given its attributes in the form of a tuple.
        Args:
            worker: the worker doing the deserialization
            plan_tuple: a tuple holding the attributes of the Plan
        Returns:
            plan: a Plan object
        """
        (
            id_,
            role,
            include_state,
            name,
            tags,
            description,
            torchscript,
            input_types,
            base_framework,
            roles,
        ) = plan_tuple

        id_ = sy.serde.msgpack.serde._detail(worker, id_)
        role = sy.serde.msgpack.serde._detail(worker, role)
        name = sy.serde.msgpack.serde._detail(worker, name)
        tags = sy.serde.msgpack.serde._detail(worker, tags)
        description = sy.serde.msgpack.serde._detail(worker, description)
        torchscript = sy.serde.msgpack.serde._detail(worker, torchscript)
        input_types = sy.serde.msgpack.serde._detail(worker, input_types)
        base_framework = sy.serde.msgpack.serde._detail(worker, base_framework)
        roles = sy.serde.msgpack.serde._detail(worker, roles)

        plan = sy.Plan(
            role=role,
            include_state=include_state,
            is_built=True,
            id=id_,
            owner=worker,
            name=name,
            tags=tags,
            description=description,
            input_types=input_types,
            base_framework=base_framework,
            roles=roles,
        )

        plan.torchscript = torchscript

        return plan

    @staticmethod
    def bufferize(worker: AbstractWorker, plan: "Plan") -> PlanPB:
        """
        This function takes the attributes of a Plan and saves them in a Protobuf message
        Args:
            worker (AbstractWorker): the worker doing the serialization
            plan (Plan): a Plan object
        Returns:
            PlanPB: a Protobuf message holding the unique attributes of the Plan object
        """
        if not plan.is_built:
            raise RuntimeError("A Plan needs to be built before being serialized.")

        protobuf_plan = PlanPB()

        sy.serde.protobuf.proto.set_protobuf_id(protobuf_plan.id, plan.id)

        protobuf_plan.role.CopyFrom(sy.serde.protobuf.serde._bufferize(worker, plan.role))

        protobuf_plan.include_state = plan.include_state
        protobuf_plan.name = plan.name
        protobuf_plan.tags.extend(plan.tags)

        if protobuf_plan.description:
            protobuf_plan.description = plan.description

        if plan.torchscript:
            protobuf_plan.torchscript = plan.torchscript.save_to_buffer()

        if plan.input_types:
            input_types = sy.serde.protobuf.serde._bufferize(worker, plan.input_types)
            protobuf_plan.input_types.CopyFrom(input_types)

        protobuf_plan.base_framework = plan._base_framework

        if plan.roles:
            for framework_name, role in plan.roles.items():
                protobuf_plan.roles.get_or_create(framework_name).CopyFrom(
                    sy.serde.protobuf.serde._bufferize(worker, role)
                )

        return protobuf_plan

    @staticmethod
    def unbufferize(worker: AbstractWorker, protobuf_plan: PlanPB) -> "Plan":
        """This function reconstructs a Plan object given its attributes in the form of a Protobuf message
        Args:
            worker: the worker doing the deserialization
            protobuf_plan: a Protobuf message holding the attributes of the Plan
        Returns:
            plan: a Plan object
        """
        id_ = sy.serde.protobuf.proto.get_protobuf_id(protobuf_plan.id)

        role = sy.serde.protobuf.serde._unbufferize(worker, protobuf_plan.role)

        name = protobuf_plan.name
        tags = set(protobuf_plan.tags) if protobuf_plan.tags else None
        description = protobuf_plan.description if protobuf_plan.description else None
        input_types = sy.serde.protobuf.serde._unbufferize(worker, protobuf_plan.input_types)
        base_framework = protobuf_plan.base_framework

        roles = {}
        for framework_name in protobuf_plan.roles:
            roles[framework_name] = sy.serde.protobuf.serde._unbufferize(
                worker, protobuf_plan.roles[framework_name]
            )

        plan = Plan(
            role=role,
            include_state=protobuf_plan.include_state,
            is_built=True,
            id=id_,
            owner=worker,
            name=name,
            tags=tags,
            description=description,
            input_types=input_types,
            base_framework=base_framework,
            roles=roles,
        )

        if protobuf_plan.torchscript:
            torchscript = io.BytesIO(protobuf_plan.torchscript)
            plan.torchscript = torch.jit.load(torchscript)

        return plan

    @property
    def code(self) -> str:
        """Returns string representation of Plan actions"""
        input_names = {id: f"arg_{i + 1}" for i, id in enumerate(self.role.input_placeholder_ids)}
        output_names = {id: f"out_{i + 1}" for i, id in enumerate(self.role.output_placeholder_ids)}
        state_names = {
            ph.id.value: f"state_{i + 1}" for i, ph in enumerate(self.role.state.state_placeholders)
        }
        var_names = {**input_names, **output_names, **state_names}

        out = f"def {self.name}("
        out += ", ".join([var_names[id] for id in self.role.input_placeholder_ids])
        out += "):\n"
        for action in self.role.actions:
            out += f"    {action.code(var_names)}\n"

        out += "    return "
        out += ", ".join([var_names[id] for id in self.role.output_placeholder_ids])

        return out

    @staticmethod
    def get_protobuf_schema() -> PlanPB:
        return PlanPB


# Auto-register Plan build-time translations
Plan.register_build_translator(PlanTranslatorTorchscript)
Plan.register_build_translator(PlanTranslatorTfjs)

# Auto-register Plan build-time frameworks
for f_name, f_package in framework_packages.items():
    Plan.register_framework(f_name, f_package)
