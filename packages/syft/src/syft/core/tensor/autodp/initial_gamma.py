# stdlib
from typing import Any
from typing import List as TypeList
from typing import Optional
from typing import Union

# third party
import numpy as np

# relative
from ...adp.vm_private_scalar_manager import VirtualMachinePrivateScalarManager
from ...common.serde.recursive import RecursiveSerde
from ...common.uid import UID
from ..passthrough import PassthroughTensor
from .intermediate_gamma import IntermediateGammaTensor


def numpy2list(np_obj: np.ndarray) -> TypeList:
    return [list(np_obj.flatten()), np_obj.shape]


def list2numpy(l_shape: Any) -> np.ndarray:
    list_length = l_shape[0]
    shape = l_shape[1]
    return np.array(list_length).reshape(shape)


class InitialGammaTensor(IntermediateGammaTensor, RecursiveSerde):

    __attr_allowlist__ = [
        "uid",
        "values",
        "min_vals",
        "max_vals",
        "entities",
        "scalar_manager",
        "term_tensor",
        "coeff_tensor",
        "bias_tensor",
        "child",
    ]

    __serde_overrides__ = {"entities": [numpy2list, list2numpy]}

    def __init__(
        self,
        values: Union[IntermediateGammaTensor, PassthroughTensor, np.ndarray],
        min_vals: np.ndarray,
        max_vals: np.ndarray,
        entities: Any,  # List[Entity] gives flatten errors
        scalar_manager: Optional[VirtualMachinePrivateScalarManager] = None,
    ) -> None:
        self.uid = UID()
        self.values = values  # child
        self.min_vals = min_vals
        self.max_vals = max_vals
        self.entities = entities
        if scalar_manager is None:
            self.scalar_manager = VirtualMachinePrivateScalarManager()
        else:
            self.scalar_manager = scalar_manager

        flat_values = self.values.flatten()
        flat_min_vals = self.min_vals.flatten()
        flat_max_vals = self.max_vals.flatten()
        flat_entities = self.entities.flatten()

        some_symbols = list()
        for i in range(flat_values.shape[0]):
            prime = self.scalar_manager.get_symbol(
                min_val=flat_min_vals[i],
                value=flat_values[i],
                max_val=flat_max_vals[i],
                entity=flat_entities[i],
            )
            some_symbols.append(prime)

        term_tensor = np.array(some_symbols).reshape(list(self.values.shape) + [1])
        coeff_tensor = (term_tensor * 0) + 1
        bias_tensor = self.values * 0

        super().__init__(
            term_tensor=term_tensor,
            coeff_tensor=coeff_tensor,
            bias_tensor=bias_tensor,
            scalar_manager=self.scalar_manager,
        )
