# stdlib
from packages.syft.src.syft.core.tensor.autodp.single_entity_phi import (
    SingleEntityPhiTensor,
)
from typing import Union

# third party
import numpy as np

# relative
from .passthrough import PassthroughTensor

AcceptableSimpleType = Union[int, bool, float, np.ndarray]
SupportedChainType = Union[PassthroughTensor, AcceptableSimpleType]
FlexibleSingleEntityPhiTensorType = Union[SingleEntityPhiTensor, NotImplemented]
