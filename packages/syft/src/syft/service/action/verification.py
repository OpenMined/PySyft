# stdlib
from collections.abc import Callable
from typing import Any

# third party
import numpy as np
import pandas as pd

# relative
from ...types.errors import SyftException
from ..response import SyftResponseMessage
from ..response import SyftSuccess
from .action_object import ActionObject


def verify_result(
    func: Callable,
    private_inputs: ActionObject | list[ActionObject],
    private_outputs: ActionObject | list[ActionObject],
) -> SyftResponseMessage:
    """Verify a single result of Code Verification"""
    trace_assets = []
    if not isinstance(private_inputs, list):
        private_inputs = [private_inputs]

    for asset in private_inputs:
        if not isinstance(asset, ActionObject):
            raise Exception(f"ActionObject expected, instead received: {type(asset)}")
        # Manual type casting for now, to automate later
        if isinstance(asset.syft_action_data, np.ndarray):
            trace_assets.append(
                ActionObject(id=asset.id, syft_result_obj=np.ndarray([]))
            )
        elif isinstance(asset.syft_action_data, pd.DataFrame):
            trace_assets.append(
                ActionObject(id=asset.id, syft_result_obj=pd.DataFrame())
            )
        else:
            raise NotImplementedError(
                f"Trace mode not yet automated for type: {type(asset.syft_action_data)}"
            )

    print("Code Verification in progress.")
    traced_results = func(*trace_assets)

    if isinstance(private_outputs, list):
        target_hashes_list = [output.syft_history_hash for output in private_outputs]
        traced_hashes_list = [result.syft_history_hash for result in traced_results]
        return compare_hashes(target_hashes_list, traced_hashes_list, traced_results)
    else:
        target_hashes = private_outputs.syft_history_hash
        traced_hashes = traced_results.syft_history_hash
        return compare_hashes(target_hashes, traced_hashes, traced_results)


def compare_hashes(
    target_hashes: list[int] | int,
    traced_hashes: list[int] | int,
    traced_results: Any,
) -> SyftSuccess:
    if target_hashes == traced_hashes:
        msg = "Code Verification passed with matching hashes! Congratulations, and thank you for supporting PySyft!"
        return SyftSuccess(message=msg)
    else:
        msg = (
            f"Hashes do not match! Target hashes were: {target_hashes} but Traced hashes were: {traced_results}. "
            f"Please try checking the logs."
        )
        raise SyftException(public_message=msg)


def code_verification(func: Callable) -> Callable:
    """Compares history hashes of an Empty Action Object to that of the real action object

    Inputs:
    - func:: a Callable whose sole argument should be the Private Dataset(s) being used. Constraints:
        - Input arguments are the private datasets being used
        - Output arguments are the results requested.

    Outputs:
    - boolean:: if history hashes match
    """

    def wrapper(*args: Any, **kwargs: Any) -> SyftSuccess:
        trace_assets = []
        for asset in args:
            if not isinstance(asset, ActionObject):
                raise SyftException(
                    public_message=f"ActionObject expected, instead received: {type(asset)}"
                )
            # Manual type casting for now, to automate later
            if isinstance(asset.syft_action_data, np.ndarray):
                trace_assets.append(
                    ActionObject(id=asset.id, syft_result_obj=np.ndarray([]))
                )
            elif isinstance(asset.syft_action_data, pd.DataFrame):
                trace_assets.append(
                    ActionObject(id=asset.id, syft_result_obj=pd.DataFrame())
                )
            else:
                raise NotImplementedError(
                    f"Trace mode not yet automated for type: {type(asset.syft_action_data)}"
                )

        print("Evaluating function normally to obtain history hash")
        results = func(*args, **kwargs).syft_history_hash
        print(8 * "(-(-_(-_-)_-)-)")

        print("Tracing function to obtain history hash")
        traced_results = func(*trace_assets, **kwargs).syft_history_hash
        print(8 * "(-(-_(-_-)_-)-)")

        # assert len(results) == len(traced_results)
        hashes_match = results == traced_results
        if hashes_match:
            msg = (
                f"Code Verification passed with matching hashes of {results}! Congratulations, and thank you for "
                f"supporting PySyft!"
            )
            return SyftSuccess(message=msg)
        else:
            msg = (
                f"Hashes do not match! Target hashes were: {results} but Traced hashes were: {traced_results}. "
                f"Please try checking the logs."
            )
            raise SyftException(public_message=msg)

    return wrapper
