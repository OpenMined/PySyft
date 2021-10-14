# CLEANUP NOTES (for ISHAN):
# - add documentation for each method
# - add comments inline explaining each piece
# - add a unit test for each method (at least)

# future
from __future__ import annotations

# stdlib
from typing import Any
from typing import List as TypeList
from typing import Optional

# third party
from nacl.signing import VerifyKey


# the most generic class
class Scalar:
    """
    A Scalar is the most generic class, which keeps track of the current value, and a data-independent
    min-val and max-val.
    """

    def publish(
        self, acc: Any, user_key: VerifyKey, sigma: float = 1.5
    ) -> TypeList[Any]:
        """Adversarial accountant adds Gaussian noise and publishes the scalar's value"""
        # relative
        from ...publish import publish

        return publish([self], acc=acc, sigma=sigma, user_key=user_key)

    @property
    def max_val(self) -> Optional[float]:
        raise NotImplementedError

    @property
    def min_val(self) -> Optional[float]:
        raise NotImplementedError

    @property
    def value(self) -> Optional[float]:
        raise NotImplementedError

    def __str__(self) -> str:
        return (
            "<"
            + str(type(self).__name__)
            + ": ("
            + str(self.min_val)
            + " < "
            + str(self.value)
            + " < "
            + str(self.max_val)
            + ")>"
        )

    def __repr__(self) -> str:
        return str(self)
