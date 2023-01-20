# stdlib
from typing import Any

# relative
from .fixed_precision_tensor import FixedPrecisionTensor
from .manager import TensorChainManager
from .passthrough import SupportedChainType  # type: ignore


class FixedPrecisionTensorAncestor(TensorChainManager):
    def fix_precision(self, base: int = 10, precision: int = 3) -> Any:
        self.child: SupportedChainType = FixedPrecisionTensor(
            base=base, precision=precision, value=self.child  # type: ignore
        )
        return self

    # def decode(self) -> Any:
    #     if not isinstance(self.child, FixedPrecisionTensor):
    #         raise ValueError(f"self.child should be FPT but is {type(self.child)}")

    #     res = self.child.decode()
    #     return res
