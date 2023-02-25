# stdlib

# third party
import numpy as np
import pandas as pd
import pytest

# syft absolute
import syft as sy
from syft.core.node.new.action_service import NumpyArrayObject

PORT = 9082
CANADA_DOMAIN_PORT = PORT
ITALY_DOMAIN_PORT = PORT + 1
LOCAL_ENCLAVE_PORT = 8010


def download_dataset(url: str) -> np.ndarray:
    data = pd.read_csv(
        "https://raw.githubusercontent.com/OpenMined/datasets/main/PUMS_DATA/canada_data.csv",
        header=None,
        dtype=str,
    )
    np_data = data._values.astype(np.str_)
    return NumpyArrayObject(syft_action_data=np_data)


code = """
from opendp.transformations import make_cast, make_impute_constant
from opendp.mod import enable_features
from opendp.transformations import make_clamp, make_bounded_sum
from opendp.mod import binary_search_chain
from opendp.measurements import make_base_discrete_laplace
from opendp.mod import binary_search_param
import numpy as np

enable_features('contrib')

canada_data_income = canada_data.syft_action_data[:,4]
italy_data_income = italy_data.syft_action_data[:,4]
combined_income = list(np.concatenate((canada_data_income,italy_data_income)))

# the greatest number of records that any one individual can influence in the dataset
max_influence = 1

# establish public information
col_names = ["age", "sex", "educ", "race", "income", "married"]
# we can also reasonably intuit that age and income will be numeric,
#     as well as bounds for them, without looking at the data
age_bounds = (0, 100)
income_bounds = (0, 150_000)

# make a transformation that casts from a vector of strings to a vector of ints
cast_str_int = (
    # Cast Vec<str> to Vec<Option<int>>
    make_cast(TIA=str, TOA=int) >>
    # Replace any elements that failed to parse with 0, emitting a Vec<int>
    make_impute_constant(0)
)


preprocessed_income = cast_str_int(combined_income)

#bounded income transformation

bounded_income_sum = (

    # Clamp income values
    make_clamp(bounds=income_bounds) >>
    # These bounds must be identical to the clamp bounds, otherwise chaining will fail
    make_bounded_sum(bounds=income_bounds)
)


#DP transformation
discovered_scale = binary_search_param(
    lambda s: bounded_income_sum >> make_base_discrete_laplace(scale=s),
    d_in=max_influence,
    d_out=1.)


dp_sum = bounded_income_sum >> make_base_discrete_laplace(scale=discovered_scale)

dp_result = dp_sum(preprocessed_income)
"""


@pytest.mark.oblv
def test_enclave_manual_code_submission() -> None:
    # Step1: Login Phase
    canada_root = sy.login(
        email="info@openmined.org", password="changethis", port=CANADA_DOMAIN_PORT
    )
    italy_root = sy.login(
        email="info@openmined.org", password="changethis", port=ITALY_DOMAIN_PORT
    )

    # Step 2: Uploading to Domain Nodes
    canada_numpy_data = download_dataset(
        "https://raw.githubusercontent.com/OpenMined/datasets/main/PUMS_DATA/canada_data.csv"
    )
    canada_numpy_ptr = canada_root.api.services.action.set(canada_numpy_data)

    italy_numpy_data = download_dataset(
        "https://raw.githubusercontent.com/OpenMined/datasets/main/PUMS_DATA/italy_data.csv"
    )
    italy_numpy_ptr = italy_root.api.services.action.set(italy_numpy_data)

    # TODO 🟣 Modify to use Data scientist account credentials
    depl = sy.oblv.deployment_client.DeploymentClient(
        deployment_id="d-2dfedbb1-7904-493b-8793-1a9554badae7",
        oblv_client=None,
        domain_clients=[canada_root, italy_root],
        user_key_name="first",
    )  # connection_port key can be added to set the port on which oblv_proxy will run

    depl.initiate_connection(LOCAL_ENCLAVE_PORT)

    # Step 3: Manual code data preparation Phase
    inputs = {}
    canada_inputs = {"canada_data": canada_numpy_ptr.id}
    italy_inputs = {"italy_data": italy_numpy_ptr.id}
    inputs[canada_root] = canada_inputs
    inputs[italy_root] = italy_inputs

    outputs = ["dp_result"]

    # Step 4 :Code Submission Phase
    task_id = depl.request_code_execution(inputs=inputs, code=code, outputs=outputs)
    # Step 5: Code review phase
    print(
        canada_root.api.services.task.review_task(
            task_id=task_id, reason="Our IT team approved the code", approve=True
        )
    )

    print(
        italy_root.api.services.task.review_task(
            task_id=task_id, reason="Our IT team approved the code", approve=True
        )
    )

    # Step6: Result Retrieval Phase
    res_id = depl.api.services.task.get_task(task_id).outputs["output_id"]
    res = depl.api.services.task.get(res_id).base_dict

    assert len(res) == 1
    assert "dp_result" in res
    assert res["dp_result"] >= 0
    assert res["dp_result"] <= 40_000_000
