# stdlib
from functools import partial
import sys

# third party
import pytest

PYTHON_AT_LEAST_3_12 = sys.version_info >= (3, 12)
FAIL_ON_PYTHON_3_12_REASON = "Does not work yet on Python>=3.12 and numpy>=1.26"

currently_fail_on_python_3_12 = partial(
    pytest.mark.xfail,
    PYTHON_AT_LEAST_3_12,
    reason=FAIL_ON_PYTHON_3_12_REASON,
)
