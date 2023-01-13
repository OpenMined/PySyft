import pytest
import requests


def dummy_test():
    res = requests.get("http://127.0.0.1:8010/")
    assert res.status_code==200
    