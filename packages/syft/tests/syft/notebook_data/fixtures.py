# third party
import numpy as np

# from syft import autocache
import pytest

# syft absolute
import syft as sy
from syft.service.data_subject.data_subject import DataSubjectCreate as DataSubject


@pytest.fixture
def node():
    node = sy.orchestra.launch(name="test-domain-1", port="auto", dev_mode=True)
    return node


@pytest.fixture
def client(node):
    client = node.login(email="info@openmined.org", password="changethis")

    return client


@pytest.fixture
def data_subjects():
    country = DataSubject(name="Country", aliases=["country_code"])
    canada = DataSubject(name="Canada", aliases=["country_code:ca"])
    germany = DataSubject(name="Germany", aliases=["country_code:de"])
    spain = DataSubject(name="Spain", aliases=["country_code:es"])
    france = DataSubject(name="France", aliases=["country_code:fr"])
    japan = DataSubject(name="Japan", aliases=["country_code:jp"])
    uk = DataSubject(name="United Kingdom", aliases=["country_code:uk"])
    usa = DataSubject(name="United States of America", aliases=["country_code:us"])
    australia = DataSubject(name="Australia", aliases=["country_code:au"])
    india = DataSubject(name="India", aliases=["country_code:in"])

    country.add_member(canada)
    country.add_member(germany)
    country.add_member(spain)
    country.add_member(france)
    country.add_member(japan)
    country.add_member(uk)
    country.add_member(usa)
    country.add_member(australia)
    country.add_member(india)

    return country


@pytest.fixture
def sync_client_server(client, data_subjects):
    return_value = client.data_subject_registry.add_data_subject(data_subjects)


@pytest.fixture
def dataset(client):
    dataset = sy.Dataset(name="Canada Trade Value")
    client.upload_dataset(dataset)
    return dataset


@pytest.fixture
def sync_sample_dataset_and_asset(client):
    sample_dataset = sy.Dataset(name="My Sample Dataset")
    sample_data = np.array([6.0, 34, 78, 91.3, 21.5])
    mock_sample_data = np.array([7.0, 54, 88, 11, 28.3])

    sample_asset = sy.Asset(name="Sample Data")
    sample_asset.set_obj(sample_data)
    sample_asset.set_mock(mock_sample_data, mock_is_real=False)
    sample_asset.set_shape(sample_data.shape)
    sample_dataset.add_asset(sample_asset)

    client.upload_dataset(sample_data)


@pytest.fixture
def Andrew_contributor_data(dataset):
    name = "Andrew Trask"
    email = "andrew@openmined.org"
    note = "Andrew runs this domain and prepared the asset."

    dataset.add_contributor(name=name, email=email, note=note)


@pytest.fixture
def Madhava_contributor_data(dataset):
    name = "Madhava Jay"
    email = "madhava@openmined.org"
    note = "Madhava tweaked the description to add the URL because Andrew forgot."

    dataset.add_contributor(name=name, email=email, note=note)


@pytest.fixture
def jane_client_credentials(client):
    response = client.register(
        name="Jane Doe",
        email="jane@caltech.edu",
        password="abc123",
        password_verify="abc123",
        institution="Caltech",
        website="https://www.caltech.edu/",
    )

    return response


@pytest.fixture
def sheldon_cooper_credentials():
    name = "Sheldon Cooper"
    email = "sheldon@caltech.edu"
    password = "abc123"
    institution = "Caltech"
    website = "https://www.caltech.edu/"

    return (name, email, password, institution, website)


@pytest.fixture
def batman_credentials(node):
    response = node.register(
        email="batman@gotham.com",
        password="1rIzHAx6uQaP",
        password_verify="1rIzHAx6uQaP",
        name="Batman",
    )

    return response


@pytest.fixture
def robin_credentials(node):
    response = node.register(
        email="robin@gotham.com",
        password="5v1ei4OM2N4m",
        password_verify="5v1ei4OM2N4m",
        name="Robin",
    )

    return response


@pytest.fixture
def joker_credentials(client):
    response = client.register(
        email="joker@gotham.com",
        password="joker123",
        password_verify="joker123",
        name="Joker",
    )

    return response


@pytest.fixture
def bane_credentials(node):
    response = node.register(
        email="bane@gotham.com",
        password="SKY5cC2zQPRP",
        password_verify="SKY5cC2zQPRP",
        name="Bane",
    )

    return response


@pytest.fixture
def riddler_credentials():
    response = node.register(
        email="riddler@gotham.com",
        password="7eVGUuNDyH8P",
        password_verify="7eVGUuNDyH8P",
        name="Riddler",
    )

    return response
