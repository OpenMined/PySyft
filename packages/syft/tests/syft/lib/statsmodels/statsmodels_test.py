# stdlib
import re

# third party
import pytest

# syft absolute
import syft as sy

pd = pytest.importorskip("pandas")
statsmodels = pytest.importorskip("statsmodels")

sy.load("pandas")
sy.load("sklearn")


@pytest.mark.vendor(lib="statsmodels")
def test_glm(root_client: sy.VirtualMachineClient) -> None:
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

    # load data
    url = "https://raw.githubusercontent.com/chemo-wakate/tutorial-6th/master/beginner/data/winequality-red.txt"
    df = pd.read_csv(url, sep="\t")
    df["quality"] = df["quality"].apply(lambda x: 1 if x >= 6 else 0)
    df = df.sample(100, random_state=42)
    df_ptr = df.send(root_client)

    # explanatory variable
    x = df["fixed acidity"]
    x_ptr = df_ptr["fixed acidity"]

    # add constant
    _x = statsmodels.api.add_constant(x)
    _x_ptr = root_client.statsmodels.api.add_constant(x_ptr)

    # dependent variable
    _y = df["quality"]
    _y_ptr = df_ptr["quality"]

    # test all possible combinations of families and links
    for family in FAMILY:
        for link in family.links:

            print(family, link)

            model = statsmodels.genmod.generalized_linear_model.GLM(
                _y, _x, family=family(link=link())
            )
            result = model.fit()
            summary = result.summary().as_csv()

            remote_model = root_client.statsmodels.genmod.generalized_linear_model.GLM(
                _y_ptr, _x_ptr, family=family(link=link())
            )
            remote_result = remote_model.fit()
            # `get` corresponds to `summary().as_csv()`
            remote_summary = remote_result.get()

            # remove unnnecessary strings such as proccesing time and date
            summary = re.sub(UNNECESSARY_STR, "", summary)
            remote_summary = re.sub(UNNECESSARY_STR, "", remote_summary)

            assert summary == remote_summary
