from typing import Union
from .passthrough import PassthroughTensor
import numpy as np

AcceptableSimpleType = Union[int, bool, float, np.ndarray]
SupportedChainType = Union[PassthroughTensor, AcceptableSimpleType]
