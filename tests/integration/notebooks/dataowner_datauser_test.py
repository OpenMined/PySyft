import os
from nbclient.client import execute

# third party
import pytest
from testbook import testbook

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
NOTEBOOK_PATH = ROOT_PATH + '/notebooks'


@pytest.fixture(scope='module')
def notebook_run():
    with testbook(NOTEBOOK_PATH + '/nb_for_integration_tests.ipynb', execute=True) as nb:
        yield nb


@testbook(NOTEBOOK_PATH + '/nb_for_integration_tests.ipynb', execute=True)
def test_dataowner_can_connect_to_node(notebook_run):
    domain_name = notebook_run.ref("do_domain_node.name")
    assert domain_name == 'test_node'

