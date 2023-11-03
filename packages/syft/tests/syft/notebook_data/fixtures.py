import pytest
import syft as sy
sy.requires(SYFT_VERSION)
from syft import autocache
import pandas as pd

@pytest.fixture
def data_subjects():
    
    canada = sy.DataSubject(name="Canada", aliases=["country_code:ca"])
    germany = sy.DataSubject(name="Germany", aliases=["country_code:de"])
    spain = sy.DataSubject(name="Spain", aliases=["country_code:es"])
    france = sy.DataSubject(name="France", aliases=["country_code:fr"])
    japan = sy.DataSubject(name="Japan", aliases=["country_code:jp"])
    uk = sy.DataSubject(name="United Kingdom", aliases=["country_code:uk"])
    usa = sy.DataSubject(name="United States of America", aliases=["country_code:us"])
    australia = sy.DataSubject(name="Australia", aliases=["country_code:au"])
    india = sy.DataSubject(name="India", aliases=["country_code:in"])
    
    country_list = [canada, germany, spain, france, japan, uk, usa, australia, india]

    return country_list

@pytest.fixture
def default_root_credentials():

    email = "info@openmined.org"
    password = "changethis"

    return (email, password)

@pytest.fixture
def Andrew_contributor_data():

    name = "Andrew Trask"
    email = "andrew@openmined.org"
    note = "Andrew runs this domain and prepared the asset."

    return(name, email, note)

@pytest.fixture
def Madhava_contributor_data():

    name = "Madhava Jay"
    email = "madhava@openmined.org"
    note = "Madhava tweaked the description to add the URL because Andrew forgot."

    return(name, email, note)

@pytest.fixture
def jane_client_credentials():

    email = "jane@caltech.edu"
    password = "abc123"

    return (email, password)

@pytest.fixture
def sheldon_cooper_credentials():

    name="Sheldon Cooper"
    email="sheldon@caltech.edu"
    password="abc123"
    institution="Caltech"
    website="https://www.caltech.edu/"

    return (name, email, password, institution, website)

@pytest.fixture
def batman_credentials():

    email="batman@gotham.com"
    password="1rIzHAx6uQaP"

    return (email, password)

@pytest.fixture
def robin_credentials():

    email="robin@gotham.com"
    password="5v1ei4OM2N4m"

    return (email, password)

@pytest.fixture
def joker_credentials():

    email="joker@gotham.com"
    password="joker123"

    return (email, password)

@pytest.fixture
def bane_credentials():

    email="bane@gotham.com"
    password="SKY5cC2zQPRP"

    return (email, password)

@pytest.fixture
def riddler_credentials():

    email="riddler@gotham.com"
    password="7eVGUuNDyH8P"
    
    return (email, password)