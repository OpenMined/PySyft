# third party
import numpy as np
import pytest

# syft absolute
from syft.serde.lib_permissions import *
from syft.serde.lib_service_registry import *
from syft.service.response import SyftAttributeError
from syft.service.response import SyftError

PYTHON_ARRAY = [0, 1, 2, 3]


@pytest.mark.parametrize("func", "func_arguments"[("array", PYTHON_ARRAY)])
def test_numpy_functions(func, func_arguments, numpy_syft_instance):
    np_sy = numpy_syft_instance
    try:
        result = eval(f"np_sy.{func}({func_arguments})")
    except Exception as e:
        assert (
            e == SyftAttributeError
        ), f"Can not evalute function {func} with arguments {func_arguments}"
    else:
        original_result = eval(f"np.{func}({func_arguments})")
        assert result == original_result
        # what else should be tested
