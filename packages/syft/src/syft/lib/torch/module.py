# stdlib
import ast
from collections import OrderedDict
import copy
import importlib
import inspect
from itertools import islice
import os
from pathlib import Path
import sys
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterator
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

# third party
import torch

# syft absolute
import syft as sy
from syft.core.node.common.action.save_object_action import SaveObjectAction
from syft.core.plan.plan_builder import ROOT_CLIENT
from syft.core.plan.plan_builder import make_plan
from syft.lib.python import _SyNone

# syft relative
from .. import lib_ast
from ...core.pointer.pointer import Pointer
from ...generate_wrapper import GenerateWrapper
from ...lib.util import full_name_with_qualname
from ...logger import critical
from ...logger import info
from ...logger import traceback_and_raise
from ...proto.lib.torch.module_pb2 import Module as Module_PB
from ..python.collections import OrderedDict as SyOrderedDict
from ..python.util import downcast
from ..python.util import upcast

# from ...core.node.common.service.auth import AuthorizationException


def repr_to_kwargs(repr_str: str) -> Tuple[List[Any], Dict[Any, Any]]:
    # for example: repr_str = Conv2d(...).extra_repr()
    # produces:    > str("1, 32, kernel_size=(3, 3), stride=(1, 1)")
    # then we just have to split it into args and kwargs
    # using ast.literal_eval we can use python to give us the real primitive types
    # from the strings in a safe way
    # str("1 ") becomes int(1)
    # str("(1, 2) ") becomes tuple(1, 2)
    args: List[Any] = []
    kwargs: Dict[Any, Any] = {}
    parts = repr_str.split(",")

    # tuples are split by commas as well, so we will keep a tab on open parentheses
    # then concat with "," until we find a close parentheses
    # TODO: make work nested with a count and add tests
    para_open = False
    buffer = ""
    for part in parts:
        try:
            if "(" in part:
                para_open = True
                buffer = ""
            if para_open is True:
                buffer += part + ","
                if ")" in part:
                    # remove trailing ,
                    part = buffer[:-1]
                    buffer = ""
                    para_open = False
                else:
                    continue

            string = part.strip()
            if "=" not in string:
                # its an arg
                arg = ast.literal_eval(string)
                args.append(arg)
            else:
                # its a kwarg
                kv = string.split("=")
                key = str(kv[0])
                string = kv[1].strip()
                value = ast.literal_eval(string)
                kwargs[key.strip()] = value
        except Exception as e:
            info(f"ast.literal_eval failed to parse part: {string}. {e}")

    return (args, kwargs)


class Module:
    """
    This is our equivalent of torch.nn.Module and aims to have the same external
    interface. We will need to support both torch Modules and Module Pointers.
    """

    def __init__(self, torch_ref: Any) -> None:
        self.setup(torch_ref=torch_ref)

    def setup(self, torch_ref: Any) -> None:
        # the remote torch means the model is remote
        self.remote_model: Optional["Module"] = None
        self.local_model: Optional["Module"] = None
        self.duet = None
        if "syft" in full_name_with_qualname(klass=type(torch_ref)):
            info("> Creating remote model")
            self.is_local = False
        else:
            # otherwise we have a local model
            info("> Creating local model")
            self.is_local = True

        self.torch_ref = torch_ref
        self.training = False
        self._modules: OrderedDict[str, Module] = OrderedDict()
        real_module = torch_ref.nn.Module()
        self.__dict__["real_module"] = real_module  # bypass getattr/setattr
        # if issubclass(type(real_module), Pointer):
        #     try:
        #         # TODO: this needs fixing but should be on by default for now
        #         # https://github.com/OpenMined/PySyft/issues/5242
        #         real_module.searchable = True
        #     except AuthorizationException as e:
        #         print(f"Cant make real_module searchable. {e}")

    def __setattr__(self, name: str, value: Union[Any, "Module"]) -> None:
        # this is how we catch the modules being set during subclass init
        # bug where torch.nn.modules isn't the full name on some imports
        # TODO: fix this properly
        # third party
        import torch

        if "torch.nn" in full_name_with_qualname(klass=type(value)) or isinstance(
            value, torch.nn.Module
        ):
            modules = self.__dict__.get("_modules")
            if modules is not None:
                modules[name] = value

            # attach all the sub modules to a real module so that we can have a
            # remote module pointer that acts like a real model
            real_module: Optional[OrderedDict] = self.__dict__.get("real_module")
            if real_module is not None:
                real_module.add_module(name, value)  # type: ignore
        else:
            object.__setattr__(self, name, value)

    def __getattr__(self, name: str) -> Union[Any, "Module"]:
        modules: Optional[OrderedDict] = self.__dict__.get("_modules")
        if modules is not None:
            if name in modules:
                return modules[name]

        return object.__getattribute__(self, name)

    def train(self, mode: bool = True) -> "Module":
        self.training = mode
        for _, module in self.modules.items():
            module.train(mode)
        return self

    def eval(self) -> "Module":
        return self.train(False)

    def __call__(
        self, *args: Union[List[Any], Tuple[Any, ...]], **kwargs: Dict[Any, Any]
    ) -> Any:
        return self.forward(*args, **kwargs)

    @property
    def modules(self) -> OrderedDict:
        modules = self.__dict__.get("_modules")
        if modules is not None:
            return modules
        return OrderedDict()

    # local list of remote ListPointers of TensorPointers
    def parameters(self, recurse: bool = True) -> Optional[List[Any]]:
        params_list: Optional[List[Any]] = None

        if self.is_local is True:
            # we are local so use normal torch params
            params_list = []
        for _, module in self.modules.items():
            params = module.parameters(recurse)
            if params_list is None:
                # only on remote create a remote list so we can concat the param list
                # pointers without having to actually get them
                self.duet = params.client
                params_list = self.duet.syft.lib.python.List()  # type: ignore
            # either way lets concat them until we have a big list of parameters
            params_list += params
        return params_list

    def cuda(self, device: Any) -> "Module":
        for _, module in self.modules.items():
            module.cuda(device)
        return self

    def cpu(self) -> "Module":
        for _, module in self.modules.items():
            module.cpu()
        return self

    def load_state_dict(self, input: Union[str, os.PathLike, Dict[str, Any]]) -> None:
        if not self.is_local:
            info("> This model is remote so try calling .get()")
            return None

        state_dict = {}
        if isinstance(input, (str, os.PathLike)):
            with open(Path(input), "rb") as f:
                state_dict = torch.load(f)
        else:
            state_dict = dict(input)

        if not issubclass(type(state_dict), dict):
            traceback_and_raise(
                f"  Invalid input: {type(input)}. "
                + "Try inputting a state_dict or .pth file."
            )

        info("> Loading model weights")
        layers: Dict[str, Any] = {}
        for save_key, values in state_dict.items():
            parts = save_key.split(".")
            if len(parts) < 2:
                info(f"  state dict key is too short: {save_key}")
                continue
            layer = parts[0]
            attr = parts[1]
            if layer not in layers:
                layers[layer] = {}
            layers[layer][attr] = values

        for layer, sd in layers.items():
            local_layer = getattr(self, layer, None)
            if local_layer is not None and hasattr(local_layer, "load_state_dict"):
                d = local_layer.load_state_dict(sd)
                info(f"  {layer} state dict loaded with: {d}")
            else:
                info(f"  Model doesnt have layer {layer}")

        info("> Finished loading weights")
        return None

    def state_dict(self) -> Optional[Dict[str, Any]]:
        if not self.is_local:
            info("> This model is remote so try calling .get()")
            return None

        info("> Saving model weights")
        model_state_dict = OrderedDict()
        for name, module in self.modules.items():
            if hasattr(module, "state_dict"):
                for k, v in module.state_dict().items():
                    save_key = f"{name}.{k}"
                    model_state_dict[save_key] = v

        info("> Finished saving weights")
        return model_state_dict

    def save(self, path: Union[str, bytes, os.PathLike]) -> None:
        if not self.is_local:
            info("> This model is remote so try calling .get()")
            return

        state_dict = self.state_dict()
        torch.save(state_dict, path)

    def load(self, path: Union[str, os.PathLike]) -> None:
        if not self.is_local:
            info("> This model is remote so try calling .get()")
            return

        self.load_state_dict(input=path)

    def send(self, client: Any, send_parameters: bool = True) -> Any:
        if not self.is_local:
            info("> This model is remote so try calling .get()")
            return

        info("> Sending local model")

        remote_model = copy.copy(self)
        remote_model.setup(torch_ref=client.torch)
        remote_model.duet = client

        for name, module in self.modules.items():
            fqn = full_name_with_qualname(klass=type(module))
            klass = client.lib_ast.query(fqn, obj_type=type(module))
            module_repr = module.extra_repr()
            args, kwargs = repr_to_kwargs(repr_str=module_repr)
            remote_module_ptr = klass(*args, **kwargs)
            remote_model.__setattr__(name, remote_module_ptr)

            # if the remote module has state_dict lets get it
            if (
                send_parameters
                and hasattr(module, "state_dict")
                and hasattr(remote_module_ptr, "load_state_dict")
            ):
                local_state_ord_dict = module.state_dict()
                # cast to dict because OrderedDict is not supported

                # get a blocking copy of the state_dict
                info(f"  Sending local layer: {name}")
                # cant import Dict / PrimitiveFactory due to circular imports
                remote_state_dict_ptr = client.syft.lib.python.Dict(
                    dict(local_state_ord_dict)
                )
                # iterate through the key, values
                # weights and biases should be in there
                remote_module_ptr.load_state_dict(remote_state_dict_ptr)

        info("\n> Finished sending local model <\n\n")
        self.remote_model = remote_model
        return self.remote_model

    def get(
        self,
        request_block: bool = False,
        timeout_secs: int = 20,
        reason: str = "",
        delete_obj: bool = False,
    ) -> Optional["Module"]:

        if self.is_local:
            info("> This model is local. Maybe you meant to call .send()?")
            return None

        info("> Downloading remote model")

        local_model = copy.copy(self)
        local_model.setup(torch_ref=torch)
        local_model.duet = self.duet

        for layer_name, module in self.modules.items():
            module_parts = module.path_and_name.split(".")
            klass_name = module_parts.pop()
            klass = getattr(sys.modules[".".join(module_parts)], klass_name)
            repr_ptr = module.extra_repr()

            module_repr = repr_ptr.get(
                request_block=request_block,
                reason=reason,
                timeout_secs=timeout_secs,
            )

            if module_repr is None:
                info(f"  Request for {reason} extra_repr failed, skipping layer")
                continue

            args, kwargs = repr_to_kwargs(repr_str=module_repr.upcast())
            local_module = klass(*args, **kwargs)

            # the local real module has been set on the sy module
            local_model.__setattr__(layer_name, local_module)

            try:
                # if the remote module has state_dict lets get it
                if hasattr(module, "state_dict") and hasattr(
                    local_module, "load_state_dict"
                ):
                    info("loading remote state dict")
                    sd_ptr = module.state_dict()
                    # get a blocking copy of the state_dict
                    info(f"  Downloading remote layer: {layer_name}")
                    state_dict = sd_ptr.get(
                        request_block=request_block,
                        reason=reason,
                        timeout_secs=timeout_secs,
                        delete_obj=delete_obj,
                    )
                    # We have to recreate the OrderedDict for load_state_dict to work
                    ordered_state_dict = OrderedDict()
                    for elem, item in state_dict.items():
                        ordered_state_dict[str(elem)] = item
                    # iterate through the key, values
                    # weights and biases should be in there
                    if state_dict is not None:
                        # TODO: support torch.nn.modules.module._IncompatibleKeys
                        local_module.load_state_dict(ordered_state_dict)
                    else:
                        info(
                            f"  Failed to get {layer_name} state_dict, skipping layer."
                        )

            except Exception as e:
                critical(f"  Failed to download remote state for {layer_name}.")
                traceback_and_raise(e)

        info("\n> Finished downloading remote model <\n\n")
        self.local_model = local_model
        return self.local_model

    # zero them so we know they are copied
    def zero_layers(self) -> None:
        for m in self.modules.values():
            if hasattr(m, "weight"):
                m.weight.requires_grad_(False).zero_()
            if hasattr(m, "bias"):
                m.bias.requires_grad_(False).zero_()

    # easy way to check the weights have changed
    def debug_sum_layers(self) -> None:
        info("> Summing layers for debugging: ")
        for n, m in self.modules.items():
            if hasattr(m, "state_dict"):
                if self.is_local:
                    state_dict = m.state_dict()
                else:
                    state_dict = m.state_dict().get()

                for k, v in state_dict.items():
                    if hasattr(v, "sum"):
                        s = v.sum().item()
                        info(f"  Layer {n} sum({k}): {s}")


def module_in_ast(module_type: type) -> bool:
    fqn = full_name_with_qualname(klass=module_type)
    try:
        return bool(lib_ast.query(fqn, obj_type=module_type))
    except ValueError:
        return False


def object2proto(obj: torch.nn.Module, is_child: bool = False) -> Module_PB:

    proto = Module_PB()
    if module_in_ast(type(obj)):
        proto.python_module = type(obj).__module__
    else:
        proto.python_module = "_USER_DEFINED"
        proto.forward.CopyFrom(sy.serialize(obj._forward_plan))

    proto.module_type = type(obj).__name__
    proto.module_repr = obj.extra_repr()

    # Requires klass.store_init_args() when building AST.
    if hasattr(obj, "_init_args") and hasattr(obj, "_init_kwargs"):
        module_args = sy.serialize(sy.lib.python.List(obj._init_args))
        proto.module_args.CopyFrom(module_args)
        module_kwargs = sy.serialize(sy.lib.python.Dict(obj._init_kwargs))
        proto.module_kwargs.CopyFrom(module_kwargs)

    if hasattr(obj, "_uid2attr"):
        proto._uid2attr.CopyFrom(sy.serialize(SyOrderedDict(obj._uid2attr)))

    proto.parameters.CopyFrom(sy.serialize(SyOrderedDict(obj._parameters)))

    for n, m in obj.named_children():
        child_proto = object2proto(m, is_child=True)
        child_proto.module_name = n
        proto.children.append(child_proto)

    return proto


def proto2object(proto: Module_PB) -> torch.nn.Module:
    is_userdefined = proto.python_module == "_USER_DEFINED"

    if is_userdefined:
        obj_type = type(
            proto.module_type,
            (torch.nn.Module,),
            {},
        )
    else:
        obj_type = getattr(
            importlib.import_module(proto.python_module), proto.module_type
        )

    if proto.HasField("module_args") and proto.HasField("module_kwargs"):
        args = upcast(sy.deserialize(proto.module_args))
        kwargs = upcast(sy.deserialize(proto.module_kwargs))
    elif proto.HasField("module_repr"):
        # TODO make Modules defined by torch use module_args and module_kwargs,
        # and remove module_repr from proto
        args, kwargs = repr_to_kwargs(repr_str=proto.module_repr)
    else:
        raise ValueError("Could not infer args and kwargs from Module proto.")

    obj = obj_type(*args, **kwargs)

    for name, param in sy.deserialize(proto.parameters).items():
        # if we don't do this check, some torch.nn layers fail ( e.g. Conv2D with bias=False)
        if not isinstance(param, _SyNone):
            setattr(obj, str(name), param)

    if proto.HasField("forward"):
        forward_plan = sy.deserialize(proto.forward)
        obj._forward_plan = forward_plan
        compile_and_forward = create_compile_and_forward_fn(obj)
        obj.__call__ = compile_and_forward
        obj.forward = compile_and_forward
        # obj.__call__ = forward_plan
        # obj.forward = forward_plan

    for child_proto in proto.children:
        setattr(obj, str(child_proto.module_name), sy.deserialize(child_proto))

    if proto.HasField("_uid2attr"):
        obj._uid2attr = sy.deserialize(proto._uid2attr)

    if is_userdefined:
        recompile(obj)

    return obj


def create_compile_and_forward_fn(obj: "SyModule") -> Callable:
    """Wraps a forward plan in a function that first recompiles the plan, and then
    executes the plan

    Args:
        obj (SyModule): the SyModule
    """

    def _compile_and_forward(*args, **kwargs):  # type: ignore
        recompile(obj)
        return obj._forward_plan(*args, **kwargs)

    return _compile_and_forward


def recompile(sy_module: "SyModule") -> None:
    """Recompiles the forward plan, if the object state has changed since the
    forward plan was created, we update the plan here

    Args:
        sy_module (SyModule): the module to compile
    """
    if hasattr(sy_module, "_forward_plan"):
        for action in sy_module._forward_plan.actions:  # type: ignore
            if (
                isinstance(action, SaveObjectAction)
                and action.obj.id in sy_module._uid2attr
            ):
                action.obj.data = getattr(
                    sy_module, str(sy_module._uid2attr[action.obj.id])
                )


GenerateWrapper(
    wrapped_type=torch.nn.Module,
    import_path="torch.nn.Module",
    protobuf_scheme=Module_PB,
    type_object2proto=object2proto,
    type_proto2object=proto2object,
)


class ForwardToPlanConverter(type):
    """This metaclass ensures that:
    1) the object is initialized when calling Object()
    2) obj._make_forward_plan() is called after initialization
    """

    def __call__(cls: Any, *args, **kwargs) -> Any:  # type: ignore
        # TODO: check if contains input_size
        obj = type.__call__(cls, *args, **kwargs)
        obj._make_forward_plan()
        return obj


class SyModule(torch.nn.Module, metaclass=ForwardToPlanConverter):
    """A `SyModule` is the pointable equivalent of a torch.nn.Module. In order to make
    SyModule remotely executable, its `.forward` method is converted into a `Plan` object
    when initializing a `SyModule` object. This object has two "modes", in which it behaves
    differently. During the "forward plan building stage" it transforms parameters and submodules
    into pointer when the user retrieves them. After plan building the model behaves more
    like a regular torch.nn.Module, but instead of running a forward method, the user executes
    a `Plan`. As the user does not need to understand the building stage, and the .forward API
    is fairly similar to a regular torch.nn.Module, there is no need to understand all internals
    to use this module.

    """

    def __init__(  # type: ignore
        self,
        *args,
        input_size: Optional[Tuple[int]] = None,
        inputs: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs,
    ):
        """
        Args:
            input_size (Optional[Tuple[int]], optional): input_size of the Module,
                needs to be defined or inferrable. Defaults to None.
            inputs (Optional[Dict[str, torch.Tensor]], optional): dictionary of dummy input tensors.
                Use this argument instead of `input_size` if there are multiple forward inputs,
                or if the forward input is not a FloatTensor. Defaults to None.

        Raises:
            ValueError: [description]
        """
        super().__init__(*args, **kwargs)
        self.building_forward = False
        self._parameter_pointers: Dict[str, Pointer] = dict()

        if (input_size is None) == (inputs is None):
            raise ValueError(
                "Either `input_size` or `inputs` should be specified, but not both."
            )

        self.input_size = input_size
        self.inputs = inputs

    def _make_forward_plan(self) -> None:
        """Convert forward function into a `Plan` object

        Raises:
            ValueError: `.forward` method must be defined
        """
        if getattr(self.forward, __name__, None) == "_forward_unimplemented":  # type: ignore
            raise ValueError("Missing .forward() method for Module")

        inputs = self._get_forward_inputs()

        self.building_forward = True
        plan = make_plan(self.forward, inputs=inputs)  # type: ignore
        self.forward = self._local_forward
        self._forward_plan = plan
        self.__call__ = plan
        self._create_uid2attr()
        self.building_forward = False
        self._remove_plan_action_data()

    def _remove_plan_action_data(self) -> None:
        """
        Sets `action.obj.data` for each symodule action in `self._forward_plan` to `None`.

        This greatly reduces the proto memory footprint;
        The whole state of `self` is saved in the action, which will be recompiled anyway.
        """

        # Remove module action data
        for action in self._forward_plan.actions:
            if isinstance(action, SaveObjectAction) and action.obj.id in self._uid2attr:
                action.obj.data = downcast(None)

    def _local_forward(self, *args, **kwargs):  # type: ignore
        recompile(self)
        return self._forward_plan(*args, **kwargs)

    def _create_uid2attr(self) -> None:
        self._uid2attr = {
            param.id_at_location: attr_name
            for attr_name, param in self._parameter_pointers.items()
        }

    def __getattr__(self, name: str) -> Any:
        """A custom getattr method. When retrieving a torch.nn.Module or a torch.nn.Parameter
        *during forward plan building*, SyModule instead returns a Pointer to this attribute.
        The first time an attribute is retrieved, we send it to the plan builder VM, and store
        it in self._parameters_pointers, which will be used for plan Recompilation during
        *deserialization*. If an attribute is requested again, we return the pointer from
        `_parameters_pointers`

        Args:
            name (str): name of the attr

        Returns:
            Any: Attribute value or Pointer to it
        """
        # this is __getattr__ instead of __getattribute__ because of the structure of torch.nn.Module
        if name in self._parameter_pointers and self.building_forward:
            return self._parameter_pointers[name]

        res = super().__getattr__(name)
        if (
            isinstance(res, (torch.nn.Module, torch.nn.Parameter))
            and self.building_forward
        ):
            res_ptr = res.send(ROOT_CLIENT)
            self._parameter_pointers[name] = res_ptr
            return res_ptr
        else:
            return res

    def _get_inp_keys(self) -> List[str]:
        """Get key for the `.forward` argument
        Returns:
            str: input key
        """

        forward_signature = inspect.signature(self.forward)
        args = list(forward_signature.parameters.items())
        if len(args) == 0:
            raise ValueError("SyModules requires more than one argument, and no kwargs")
        inp_keys = []
        for k, v in args:
            if v.default is not inspect.Parameter.empty:
                raise ValueError("SyModules accept only args, not kwargs")
            inp_keys.append(k)
        return inp_keys

    def _get_forward_inputs(self) -> Dict[str, Pointer]:
        """Get the dummy inputs for generating the .forward `Plan`

        Returns:
            Dict[str: Any]: inputs for .forward
        """
        inp_keys = self._get_inp_keys()

        if hasattr(self, "inputs") and isinstance(self.inputs, dict):
            if set(inp_keys) != set(self.inputs.keys()):
                raise ValueError(
                    "Given `inputs` dict and expected `forward` inputs do not match."
                )
            inputs = {k: v.send(ROOT_CLIENT) for k, v in self.inputs.items()}

        elif hasattr(self, "input_size") and isinstance(self.input_size, tuple):
            if len(inp_keys) != 1:
                raise ValueError(
                    "`.forward` method has more than one input, define dummy inputs with `inputs` kwarg."
                )
            inputs = {inp_keys[0]: torch.rand(self.input_size).send(ROOT_CLIENT)}

        else:
            raise ValueError(
                "SyModule needs either `input_size`: Tuple(Int) or `inputs`: Dict[str, Tensor] as kwarg"
                "to trace the forward plan."
                "Also, make sure to call **super().__init__(**kwargs)** in ALL your SyModules"
                ""
            )

        return inputs


class SySequential(SyModule):
    """The Syft equivalent of torch.nn.Sequential"""

    def __init__(self, *args, input_size: Optional[Tuple[int]] = None, **kwargs):  # type: ignore
        """initializes SySequential and stores the submodules

        input_size (Tuple[Int], optional): input_size of the Module, needs to be defined or inferrable.
            Defaults to None.
        """
        if input_size is None:
            input_size = self._infer_input_size(*args)

        super().__init__(input_size=input_size, **kwargs)
        for idx, module in enumerate(args):
            setattr(self, str(idx), module)
        self.n_modules = len(args)

    def _infer_input_size(self, *args: Any) -> Tuple[int]:
        """Infer input size from first child

        Returns:
            Tuple[int]: input size of first child SyModule.
        """
        if hasattr(args[0], "input_size"):
            return args[0].input_size
        else:
            raise ValueError(
                "Could not infer `input_size` from children modules."
                "Either 1) define `input_size` as a kwarg of SySequential OR"
                " 2) define `input_size`: Tuple[int] as kwarg on the first child module."
            )

    def __iter__(self):  # type: ignore
        if self.building_forward:
            return iter([getattr(self, str(i)) for i in range(self.n_modules)])
        else:
            return iter(self._modules.values())

    def _get_item_by_idx(self, iterator: Iterator, idx: int) -> SyModule:
        """Get the idx-th item of the iterator"""
        size = self.n_modules
        if not -size <= idx < size:
            raise IndexError(f"index {idx} is out of range")
        return next(islice(iterator, idx, None))

    def __getitem__(self, idx: int) -> SyModule:
        if isinstance(idx, slice):  # type: ignore
            raise ValueError("SySequential does not support slices")
        else:
            return self._get_item_by_idx(self._modules.values(), idx)

    def __setitem__(self, idx: int, module: Module) -> None:
        key = self._get_item_by_idx(self._modules.keys(), idx)
        return setattr(self, key, module)

    def __delitem__(self, idx: Union[slice, int]) -> None:
        if isinstance(idx, slice):
            raise ValueError("SySequential does not support slices")
        else:
            key = self._get_item_by_idx(self._modules.keys(), idx)
            delattr(self, key)

    def forward(self, x: Any) -> Any:  # type: ignore
        """Sequentially call submodule.forward

        Args:
            x (Any, optional): input. Defaults to None.

        Returns:
            Any: Module output
        """
        out = x
        for i, module in enumerate(self):
            # handle indexing in the block, or in the sequential?
            if module.__class__.__name__ == "ModulePointer":
                # user defined module
                out = module.forward(x=out)[0]
            else:
                # AST module
                out = module(out)
        return out

    def _get_inp_key(self) -> str:
        """Get key for the `.forward` argument, allways x for this module

        Returns:
            str: "x"
        """
        return "x"
