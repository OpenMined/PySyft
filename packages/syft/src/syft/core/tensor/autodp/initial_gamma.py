# stdlib
import uuid

# third party
import numpy as np

# relative
# syft relative
from ...adp.vm_private_scalar_manager import VirtualMachinePrivateScalarManager
from .intermediate_gamma import IntermediateGammaTensor
from ...common.serde.recursive import RecursiveSerde
from ...common.uid import UID

def numpy2list(np_obj):
    return [list(np_obj.flatten()),np_obj.shape]


def list2numpy(l_shape):
    l = l_shape[0]
    shape = l_shape[1]
    return np.array(l).reshape(shape)


class InitialGammaTensor(IntermediateGammaTensor, RecursiveSerde):

    __attr_allowlist__ = ['uid', 'values', 'min_vals', 'max_vals', 'entities', 'scalar_manager',
                          'term_tensor', 'coeff_tensor', 'bias_tensor', 'child']

    __serde_overrides__ = {'entities': [numpy2list, list2numpy]}

    def __init__(
        self,
        values,
        min_vals,
        max_vals,
        entities,
        scalar_manager=VirtualMachinePrivateScalarManager(),
    ):
        self.uid = UID()
        self.values = values  # child
        self.min_vals = min_vals
        self.max_vals = max_vals
        self.entities = entities
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
