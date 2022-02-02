# stdlib
from typing import Any
from typing import List as TypeList
from typing import Optional
from typing import Union

# third party
import numpy as np

# relative
from ...adp.entity import DataSubjectGroup
from ...adp.vm_private_scalar_manager import VirtualMachinePrivateScalarManager
from ...common.serde.serializable import serializable
from ...common.uid import UID
from ..passthrough import PassthroughTensor  # type: ignore
from ..smpc.share_tensor import ShareTensor
from .adp_tensor import ADPTensor
from .intermediate_gamma import IntermediateGammaTensor


def numpy2list(np_obj: np.ndarray) -> TypeList:
    return [list(np_obj.flatten()), np_obj.shape]


def list2numpy(l_shape: Any) -> np.ndarray:
    list_length = l_shape[0]
    shape = l_shape[1]
    return np.array(list_length).reshape(shape)


@serializable(recursive_serde=True)
class InitialGammaTensor(IntermediateGammaTensor, ADPTensor):

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

    sharetensor_values: Optional[ShareTensor]

    def __init__(
        self,
        values: Union[IntermediateGammaTensor, PassthroughTensor, np.ndarray],
        min_vals: np.ndarray,
        max_vals: np.ndarray,
        entities: Any,  # List[Entity] gives flatten errors
        scalar_manager: Optional[VirtualMachinePrivateScalarManager] = None,
    ) -> None:
        self.uid = UID()

        if isinstance(values, ShareTensor):
            self.sharetensor_values = values
            self.values = values.child
        else:
            self.sharetensor_values = None
            self.values = values

        self.min_vals = min_vals
        self._min_vals_cache = min_vals

        self.max_vals = max_vals
        self._max_vals_cache = max_vals

        self.entities = entities
        if scalar_manager is None:
            self.scalar_manager = VirtualMachinePrivateScalarManager()
        else:
            self.scalar_manager = scalar_manager

        flat_values = self.values.flatten()
        flat_min_vals = self.min_vals.flatten()
        flat_max_vals = self.max_vals.flatten()

        # If it's a list of lists, then it should still work
        if isinstance(self.entities, np.ndarray):
            flat_entities = self.entities.flatten()
        else:
            flat_entities = self.entities

        some_symbols = list()
        for i in range(flat_values.shape[0]):
            prime = self.scalar_manager.get_symbol(
                min_val=flat_min_vals[i],
                value=flat_values[i],
                max_val=flat_max_vals[i],
                entity=flat_entities[i]
                if not isinstance(flat_entities, DataSubjectGroup)
                else flat_entities,
            )
            some_symbols.append(prime)

        term_tensor = (
            np.array(some_symbols)
            .reshape(list(self.values.shape) + [1])
            .astype(np.int32)
        )
        coeff_tensor = (term_tensor * 0) + 1
        bias_tensor = self.values * 0

        if isinstance(entities, np.ndarray):
            unique_entities = set(list(entities.flatten()))
        else:
            unique_entities = set(entities)

        super().__init__(
            term_tensor=term_tensor,
            coeff_tensor=coeff_tensor,
            bias_tensor=bias_tensor,
            scalar_manager=self.scalar_manager,
            unique_entities=unique_entities,
        )
