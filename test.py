# third party
import pandas as pd

# syft absolute
import syft as sy
from syft.util.util import autocache

domain = sy.login(
    url="http://localhost:8081", email="info@openmined.org", password="changethis"
)

##### Create dataset
country = sy.DataSubject(name="Country", aliases=["country_code"])
canada = sy.DataSubject(name="Canada", aliases=["country_code:ca"])
germany = sy.DataSubject(name="Germany", aliases=["country_code:de"])
spain = sy.DataSubject(name="Spain", aliases=["country_code:es"])
france = sy.DataSubject(name="France", aliases=["country_code:fr"])
japan = sy.DataSubject(name="Japan", aliases=["country_code:jp"])
uk = sy.DataSubject(name="United Kingdom", aliases=["country_code:uk"])
usa = sy.DataSubject(name="United States of America", aliases=["country_code:us"])
australia = sy.DataSubject(name="Australia", aliases=["country_code:au"])
india = sy.DataSubject(name="India", aliases=["country_code:in"])
country.add_member(canada)
country.add_member(germany)
country.add_member(spain)
country.add_member(france)
country.add_member(japan)
country.add_member(uk)
country.add_member(usa)
country.add_member(australia)
country.add_member(india)

country.members
registry = domain.data_subject_registry
response = registry.add_data_subject(country)
data_subjects = domain.api.services.data_subject.get_all()
canada_dataset_url = "https://github.com/OpenMined/datasets/blob/main/trade_flow/ca%20-%20feb%202021.csv?raw=True"
df = pd.read_csv(autocache(canada_dataset_url))

ca_data = df[0:10]
mock_ca_data = df[10:20]
dataset = sy.Dataset(name="Canada Trade Value")
dataset.set_description("""Canada Trade Data""")
dataset.add_citation("Person, place or thing")
dataset.add_url("https://github.com/OpenMined/datasets/tree/main/trade_flow")
dataset.add_contributor(
    role=sy.roles.UPLOADER,
    name="Andrew Trask",
    email="andrew@openmined.org",
    note="Andrew runs this domain and prepared the dataset metadata.",
)

dataset.add_contributor(
    role=sy.roles.EDITOR,
    name="Madhava Jay",
    email="madhava@openmined.org",
    note="Madhava tweaked the description to add the URL because Andrew forgot.",
)
ctf = sy.Asset(name="canada_trade_flow")
ctf.set_description("""all the datas""")
ctf.add_contributor(
    role=sy.roles.UPLOADER,
    name="Andrew Trask",
    email="andrew@openmined.org",
    note="Andrew runs this domain and prepared the asset.",
)
ctf.set_obj(ca_data)
ctf.set_shape((10, 22))
ctf.add_data_subject(canada)
dataset.add_asset(ctf)
ctf.set_mock(mock_ca_data, mock_is_real=False)
domain.upload_dataset(dataset)


### Create request

results = domain.api.services.dataset.get_all()
dataset = results[-1]
mock = dataset.assets[0].mock


@sy.syft_function(
    input_policy=sy.ExactMatch(trade_data=mock),
    output_policy=sy.SingleExecutionExactOutput(),
)
def sum_trade_value_mil(trade_data):
    # third party
    from opendp.mod import enable_features
    import pandas as pd

    enable_features("contrib")
    # third party
    from opendp.measurements import make_base_laplace

    aggregate = 0.0
    base_lap = make_base_laplace(scale=5.0)
    noise = base_lap(aggregate)

    df = trade_data
    total = df["Trade Value (US$)"].sum()
    return (float(total / 1_000_000), float(noise))


domain.api.services.code.request_code_execution(sum_trade_value_mil)
