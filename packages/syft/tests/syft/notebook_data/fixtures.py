import pytest
from syft.service.data_subject.data_subject import DataSubjectCreate as DataSubject
#from syft import autocache
import pandas as pd

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