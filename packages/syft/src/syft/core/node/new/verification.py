from typing import Callable

import numpy as np
import pandas as pd

from .action_object import ActionObject
from .response import SyftSuccess
from .response import SyftError


def code_verification(func: Callable):
    """ Compares history hashes of an Empty Action Object to that of the real action object

    Inputs:
    - func:: a Callable whose sole argument should be the Private Dataset(s) being used. Constraints:
        - Input arguments are the private datasets being used
        - Output arguments are the results requested.

    Outputs:
    - boolean:: if history hashes match
    """
    def wrapper(*args, **kwargs):
        trace_assets = []
        for asset in args:
            if not isinstance(asset, ActionObject):
                raise Exception(f"ActionObject expected, instead received: {type(asset)}")
            # Manual type casting for now, to automate later
            if isinstance(asset.syft_action_data, np.ndarray):
                empty_obj: ActionObject = ActionObject(id=asset.id, syft_result_obj=np.ndarray([]))
            elif isinstance(asset.syft_action_data, pd.DataFrame):
                empty_obj: ActionObject = ActionObject(id=asset.id, syft_result_obj=pd.DataFrame())
            else:
                raise NotImplementedError(f"Trace mode not yet automated for type: {type(asset.syft_action_data)}")

            trace_assets.append(empty_obj)

        print("Evaluating function normally to obtain history hash")
        results = func(*args, **kwargs).syft_history_hash
        print(8 * "(-(-_(-_-)_-)-)")

        print("Tracing function to obtain history hash")
        traced_results = func(*trace_assets, **kwargs).syft_history_hash
        print(8 * "(-(-_(-_-)_-)-)")

        # assert len(results) == len(traced_results)
        hashes_match = results == traced_results
        if hashes_match:
            msg = f"Code Verification passed with matching hashes of {results}! Congratulations, and thank you for supporting PySyft!"
            return SyftSuccess(message=msg)
        else:
            msg = f"Hashes do not match! Target hashes were: {results} but Traced hashes were: {traced_results}. Please try checking the logs."
            return SyftError(message=msg)
    return wrapper
