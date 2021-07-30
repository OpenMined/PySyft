# stdlib
from typing import Union
from typing import Type

# third party
import numpy as np

# relative
from .passthrough import PassthroughTensor

AcceptableSimpleType = Union[int, bool, float, np.ndarray]
SupportedChainType = Union[Type[PassthroughTensor], AcceptableSimpleType]
