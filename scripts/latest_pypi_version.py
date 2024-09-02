# third party
import requests

PROJECT_NAME = "syft"
PYPI_JSON_URL = f"https://pypi.org/pypi/{PROJECT_NAME}/json"


def get_latest_pypi_version():
    response = requests.get(PYPI_JSON_URL)
    data = response.json()
    return data["info"]["version"]


print(get_latest_pypi_version())
