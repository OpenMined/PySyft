# stdlib
from typing import List
from typing import Optional
import uuid

# third party
import numpy as np

# relative
from ...adp.entity import Entity
from ...adp.vm_private_scalar_manager import VirtualMachinePrivateScalarManager
from ..types import SupportedChainType

# syft relative
from .intermediate_gamma import IntermediateGammaTensor


class InitialGammaTensor(IntermediateGammaTensor):
    def __init__(
        self,
        values: SupportedChainType,
        min_vals: np.ndarray,
        max_vals: np.ndarray,
        entities: List[Entity],
        scalar_manager: Optional[VirtualMachinePrivateScalarManager] = None,
    ) -> None:
        self.uid = uuid.uuid4()
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
