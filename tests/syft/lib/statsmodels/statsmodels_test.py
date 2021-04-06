# third party
import pytest

# syft absolute
import syft as sy
from syft.util import get_root_data_path


@pytest.mark.vendor(lib="statsmodels")
def test_glm(root_client: sy.VirtualMachineClient) -> None:

    # stdlib
    import os
    import re
    import shutil
    import urllib.request

    # third party
    import pandas as pd
    import statsmodels

    FAMILY = [
        statsmodels.genmod.families.Binomial,
        statsmodels.genmod.families.Gamma,
        statsmodels.genmod.families.Gaussian,
        statsmodels.genmod.families.InverseGaussian,
        statsmodels.genmod.families.NegativeBinomial,
        statsmodels.genmod.families.Poisson,
        statsmodels.genmod.families.Tweedie,
    ]

    UNNECESSARY_STR = r"Time(.*)(?=Pearson)|Date(.*)(?=Deviance)"

    sy.load("pandas")
    sy.load("statsmodels")

    # create a virtual machine
    vm = sy.VirtualMachine()
    client = vm.get_root_client()

    # download data
    csv_file = "mort_match_nhis_all_years.csv"
    zip_file = f"{csv_file}.zip"
    url = f"https://datahub.io/madhava/mort_match_nhis_all_years/r/{zip_file}"
    data_path = f"{get_root_data_path()}/CDC"
    zip_path = f"{data_path}/{zip_file}"
    csv_path = f"{data_path}/{csv_file.upper()}"
    if not os.path.exists(zip_path):
        os.makedirs(data_path, exist_ok=True)
        urllib.request.urlretrieve(url, zip_path)
    if not os.path.exists(csv_path):
        shutil.unpack_archive(zip_path, data_path)
    assert os.path.exists(csv_path)

    # load data
    df = pd.read_csv(csv_path)
    df = df.head(100)
    df_ptr = df.send(client)

    # Drop any missing values in the dataset (those under 18)
    df = df.dropna(subset=["MORTSTAT"])
    df_ptr = df_ptr.dropna(subset=["MORTSTAT"])
    # Keep only the eligible portion
    df = df[df["ELIGSTAT"] == 1]
    df_ptr = df_ptr[df_ptr["ELIGSTAT"] == 1]

    # Ignore people > 80
    df = df[df["AGE_P"] <= 80]
    df_ptr = df_ptr[df_ptr["AGE_P"] <= 80]

    # A person is alive if MORTSTAT==0
    df["is_alive"] = df["MORTSTAT"] == 0
    df_ptr["is_alive"] = df_ptr["MORTSTAT"] == 0

    # Assign a helpful column for sex (0==male, 1==female)
    df["sex"] = "male"
    df_ptr["sex"] = "male"
    # df.loc[df["SEX"] == 2, "sex"] = "female"

    # explanatory variable
    x = df["AGE_P"]
    x_ptr = df_ptr["AGE_P"]

    # add constant
    _x = statsmodels.api.add_constant(x)
    _x_ptr = client.statsmodels.api.add_constant(x_ptr)

    # dependent variable
    _y = df["is_alive"]
    _y_ptr = df_ptr["is_alive"]

    # test all possible combinations of families and links
    for family in FAMILY:
        for link in family.links:

            model = statsmodels.genmod.generalized_linear_model.GLM(
                _y, _x, family=family(link=link())
            )
            result = model.fit()
            summary = result.summary().as_csv()

            remote_model = client.statsmodels.genmod.generalized_linear_model.GLM(
                _y_ptr, _x_ptr, family=family(link=link())
            )
            remote_result = remote_model.fit()
            # `get` corresponds to `summary().as_csv()`
            remote_summary = remote_result.get()

            # remove unnnecessary strings such as proccesing time and date
            summary = re.sub(UNNECESSARY_STR, "", summary)
            remote_summary = re.sub(UNNECESSARY_STR, "", remote_summary)

            assert summary == remote_summary
