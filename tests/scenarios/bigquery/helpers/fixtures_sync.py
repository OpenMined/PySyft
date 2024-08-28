# stdlib
from typing import Any

# third party
from faker import Faker
from helpers.users import TestUser
import pandas as pd

# syft absolute
import syft as sy
from syft import autocache
from syft.service.user.user_roles import ServiceRole


def make_user(
    name: str | None = None,
    email: str | None = None,
    password: str | None = None,
    role: ServiceRole = ServiceRole.DATA_SCIENTIST,
):
    fake = Faker()
    if name is None:
        name = fake.name()
    if email is None:
        email = fake.email()
    if password is None:
        password = fake.password()

    return TestUser(name=name, email=email, password=password, role=role)


def make_admin(email="info@openmined.org", password="changethis"):
    fake = Faker()
    return make_user(
        email=email, password=password, name=fake.name(), role=ServiceRole.ADMIN
    )


def trade_flow_df():
    canada_dataset_url = "https://github.com/OpenMined/datasets/blob/main/trade_flow/ca%20-%20feb%202021.csv?raw=True"
    df = pd.read_csv(autocache(canada_dataset_url))
    return df


def trade_flow_df_mock(df):
    return df[10:20]


def user_exists(root_client, email: str) -> bool:
    users = root_client.api.services.user
    for user in users:
        if user.email == email:
            return True
    return False


def create_user(root_client, test_user):
    if not user_exists(root_client, test_user.email):
        fake = Faker()
        root_client.register(
            name=test_user.name,
            email=test_user.email,
            password=test_user.password,
            password_verify=test_user.password,
            institution=fake.company(),
            website=fake.url(),
        )
    else:
        print("User already exists", test_user)


def dataset_exists(root_client, dataset_name: str) -> bool:
    datasets = root_client.api.services.dataset
    for dataset in datasets:
        if dataset.name == dataset_name:
            return True
    return False


def upload_dataset(user_client, dataset):
    if not dataset_exists(user_client, dataset):
        user_client.upload_dataset(dataset)
    else:
        print("Dataset already exists")


def create_dataset(name: str):
    df = trade_flow_df()
    ca_data = df[0:10]
    mock_ca_data = trade_flow_df_mock(df)
    dataset = sy.Dataset(name=name)
    dataset.set_description("Canada Trade Data Markdown Description")
    dataset.set_summary("Canada Trade Data Short Summary")
    dataset.add_citation("Person, place or thing")
    dataset.add_url("https://github.com/OpenMined/datasets/tree/main/trade_flow")
    dataset.add_contributor(
        name="Andrew Trask",
        email="andrew@openmined.org",
        note="Andrew runs this datasite and prepared the dataset metadata.",
    )
    dataset.add_contributor(
        name="Madhava Jay",
        email="madhava@openmined.org",
        note="Madhava tweaked the description to add the URL because Andrew forgot.",
    )
    ctf = sy.Asset(name="canada_trade_flow")
    ctf.set_description(
        "Canada trade flow represents export & import of different commodities to other countries"
    )
    ctf.add_contributor(
        name="Andrew Trask",
        email="andrew@openmined.org",
        note="Andrew runs this datasite and prepared the asset.",
    )
    ctf.set_obj(ca_data)
    ctf.set_shape(ca_data.shape)
    ctf.set_mock(mock_ca_data, mock_is_real=False)
    dataset.add_asset(ctf)
    return dataset


def make_server(request: Any | None = None, server_name: str | None = None) -> Any:
    print("making server")
    if server_name is None:
        faker = Faker()
        server_name = faker.name()
    server = sy.orchestra.launch(
        name=server_name,
        port="auto",
        dev_mode=True,
        reset=True,
        n_consumers=1,
        create_producer=True,
    )

    def cleanup():
        print("landing server")
        server.land()

    if not request:
        print("WARNING: No pytest request supplied, no finalizer added")
    else:
        request.addfinalizer(cleanup)
    return server
