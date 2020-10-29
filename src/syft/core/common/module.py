# stdlib
from collections import OrderedDict
from typing import Any
from typing import List
from typing import Optional
from typing import Union


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
        if "torch.nn.modules" in full_name_with_qualname(klass=type(value)):
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
    def parameters(self, params_list: Any = [], recurse: bool = True) -> List[Any]:
        # params_list = torch.python.List()
        for _, module in self.modules.items():
            param_pointers = module.parameters()
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
