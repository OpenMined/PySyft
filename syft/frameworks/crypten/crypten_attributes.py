from types import ModuleType

from syft.generic.frameworks.attributes import FrameworkAttributes
import crypten


class CryptenAttributes(FrameworkAttributes):
    """Adds crypten module related custom attributes.

    CryptenAttributes is a special class where all custom attributes related
    to the torch module can be added. Any global parameter, configuration,
    or reference relating to PyTorch should be stored here instead of
    attaching it directly to some other part of the global namespace.

    The main reason we need this is because the hooking process occasionally
    needs to save global objects, notably including what methods to hook and
    what methods to NOT hook.

    This will hold all necessary attributes PySyft needs.

    Args:
        torch: A ModuleType indicating the torch module
        hook: A TorchHook to stash
    """

    # Subclasses must provide the following class attributes
    ALIAS = "crypten"
    Tensor = crypten.mpc.MPCTensor  # This is not used but needs to be provided

    def __init__(self, crypten: ModuleType, hook: ModuleType) -> None:
        """Initialization of the CrypTenAttributes class."""

        # Stash the hook here for global access elsewhere
        self.hook = hook

        self.inplace_methods = {
            "encrypt",
            "decrypt",
            "eval",
            "train",
            "zero_grad",
            "backward",
            "update_parameters",
        }
        self.global_state_change_methods = {}

    def is_inplace_method(self, method_name):
        """Determine if a method is inplace or not.

        Currently, this is used for crypten.nn.module and we consider that
        some methods from there are inplace (see self.inplace_methods)

        We need to do this because plans actions are getting prunned and
        we might trace with a plan stuff like module.encrypt().

        If the is_inplace_method and the is_global_state_change_method both
        return False than that action is pruned inside the Plans
        and we do not want that

        Args:
            method_name: The name for the method.
        Returns:
            Boolean denoting if the method is inplace or not.
        """
        return method_name in self.inplace_methods

    def is_global_state_change_method(self, method_name):
        """
        We consider that all methods from crypten.nn.module are not changing
        the global state (an example from torch is when we do torch.seed)
        """
        return False
