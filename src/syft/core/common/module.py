# stdlib
from collections import OrderedDict
from typing import Any
from typing import List
from typing import Optional
from typing import Union

# third party
from loguru import logger
import torch


# circular imports when using the syft.lib.full_name_with_qualname version
def full_name_with_qualname(klass: type) -> str:
    return f"{klass.__module__}.{klass.__qualname__}"


class Module:
    """
    This is our equivalent of torch.nn.Module and aims to have the same external
    interface. We will need to support both torch Modules and Module Pointers.
    """

    def __init__(self) -> None:
        self.training = False
        self._modules: OrderedDict[str, Module] = OrderedDict()

    # this is how we catch the modules being set during subclass init
    def __setattr__(self, name: str, value: Union[Any, "Module"]) -> None:
        # bug where torch.nn.modules isnt the full name on some imports
        # TODO: fix this properly
        if "torch.nn" in full_name_with_qualname(klass=type(value)):
            modules = self.__dict__.get("_modules")
            if modules is not None:
                modules[name] = value
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

    def forward(self, x: Any) -> Any:
        raise NotImplementedError

    def __call__(self, input: Any) -> Any:
        return self.forward(x=input)

    @property
    def modules(self) -> OrderedDict:
        modules = self.__dict__.get("_modules")
        if modules is not None:
            return modules
        return OrderedDict()

    # local list of remote ListPointers of TensorPointers
    def parameters(self, recurse: bool = True) -> List[Any]:
        duet = None
        params_list = None
        for _, module in self.modules.items():
            param_pointers = module.parameters()
            if duet is None:
                duet = param_pointers.client
                # params_list must be a remote List
                params_list = duet.syft.lib.python.List()
            params_list += param_pointers
        return params_list

    def cuda(self, device: Any) -> "Module":
        for _, module in self.modules.items():
            module.cuda(device)
        return self

    def cpu(self) -> "Module":
        for _, module in self.modules.items():
            module.cpu()
        return self

    # zero them so we know they are copied
    def zero_layers(self) -> None:
        for m in self.modules.values():
            if hasattr(m, "weight"):
                m.weight.requires_grad_(False).zero_()
            if hasattr(m, "bias"):
                m.bias.requires_grad_(False).zero_()

    # easy way to check the weights have changed
    def debug_sum_layers(self) -> None:
        for n, m in self.modules.items():
            if hasattr(m, "state_dict"):
                for k, v in m.state_dict().items():
                    if hasattr(v, "sum"):
                        s = v.sum().item()
                        log = f"> Layer {n} sum({k}): {s}"
                        print(log)
                        logger.debug(log)

    # copy the weights, there are blocking requests so make sure to respond
    def copy_remote_state(
        self,
        remote_model: Any,
        request_name: str,
        reason: str,
        timeout_secs: int,
        delete_obj: bool = False,
        skip_layers: List[str] = [],
    ) -> None:
        # loop over models module pointers
        for n, m in remote_model.modules.items():
            try:
                if n in skip_layers:
                    log = f"> Skipping: {n}"
                    print(log)
                    logger.debug(log)
                    continue
                if hasattr(self, n):
                    # get the equivalent layer for the local model
                    local_m = getattr(self, n)

                    # if the remote layer has a state_dict
                    if hasattr(m, "state_dict"):
                        sd_ptr = m.state_dict()
                        # get a blocking copy of the state_dict
                        log = f"> Downloading remote: {n}"
                        print(log)
                        logger.debug(log)
                        state_dict = sd_ptr.get(
                            request_block=True,
                            request_name=request_name,
                            reason=reason,
                            timeout_secs=timeout_secs,
                            delete_obj=delete_obj,
                        )
                        # iterate through the key, values
                        # weights and biases should be in there
                        for key, value in state_dict.items():
                            key_str = key.upcast()
                            # if the local models module has the same property
                            # e.g. .weight or .bias
                            if hasattr(local_m, key_str):
                                # if the downloaded value is not a Parameter
                                # its a tensor so we need to convert it
                                if not isinstance(value, torch.nn.Parameter):
                                    value = torch.nn.Parameter(value)

                                # set it
                                setattr(local_m, key_str, value)
                                log = f">> Setting {key_str} copy on local {n}"
                                print(log)
                                logger.debug(log)
                log = "> Finished downloading model"
                print(log)
                logger.debug(log)
            except Exception as e:
                logger.error(f"Failed to download remote state for {n}. {e}")
