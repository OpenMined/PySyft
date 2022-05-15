# third party
import pytest
from testbook import testbook


@pytest.fixture(scope="module")
def tb():
    with testbook("../courses/L3_DataPreparation.ipynb", execute=True) as tb:
        yield tb


def test_data_acquisition(tb):
    assert tb.cell_output_text(8) == "(2280, 451)"


def test_quality_check(tb):
    assert tb.cell_output_text(21) == "53"
    assert tb.cell_output_text(24) == "129"


def test_data_and_pygrid(tb):
    assert (
        "raw_data is of type: <class 'pandas.core.frame.DataFrame'>\nraw_data is of type: <class 'numpy.ndarray'>"
        in tb.cell_output_text(31)
    )
    assert (
        "test_data is of type: <class 'torch.Tensor'>\ntest_data is of type: <class 'numpy.ndarray'>"
        in tb.cell_output_text(32)
    )
    assert (
        "random_data is of dtype: float64\nrandom_data is now of type: int32"
        in tb.cell_output_text(36)
    )


def test_loading_data_to_pygrid(tb):
    sy = tb.ref("sy")
    assert sy.__version__ is not None
    domain_node = tb.ref("domain_node")
    # Check if the domain node was initialized
    assert domain_node is not None
    # Data was uploaded successfully
    assert tb.cell_output_text(43) is not None
    # Check if dataset is initialized
    assert domain_node.datasets is not None
    # Viewing the dataset
    assert tb.cell_output_text(46) is not None
