# future
from __future__ import annotations

# stdlib
from typing import Optional
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # relative
    from .... import Tensor

SMPC_CONTEXT: dict = {}
FPT_CONTEXT: dict = {}
tensor_values: Optional[Tensor] = None
