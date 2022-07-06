# third party
import pytest
from testbook import testbook


@pytest.fixture(scope="module")
def tb():
    with testbook("../courses/L5_RemoteDataScience.ipynb", execute=True) as tb:
        yield tb


def test_login(tb):
    ds_domain = tb.ref("ds_domain")
    assert ds_domain is not None
    assert ds_domain.version is not None


def test_privacy_budget(tb):
    ds_domain = tb.ref("ds_domain")
    assert ds_domain.privacy_budget.resolve() >= 50.0


def test_covid_dataset(tb):
    dataset_name = tb.ref("ds_domain.datasets[-1].name")
    assert dataset_name == "COVID19 Cases in 175 countries"
    assert tb.ref("covid_ds") is not None


def test_result_publish(tb):
    assert tb.ref("result") is not None
    assert tb.ref("published_result") is not None
    published_data = tb.ref("published_data")
    assert len(published_data) == 7


def test_load_data_as_dataframe(tb):
    data_df = tb.ref("data_df")
    assert data_df is not None
    assert tb.ref("isinstance(data_df, pd.DataFrame)") is True
    assert data_df.shape[0] == 7
