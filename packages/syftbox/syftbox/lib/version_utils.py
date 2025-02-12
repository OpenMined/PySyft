import json
from os.path import dirname
from typing import Dict, List, Union

from packaging import version


def get_version_dict() -> Dict[str, List[str]]:
    with open(dirname(dirname(__file__)) + "/server2client_version.json") as json_file:
        version_matrix = json.load(json_file)
    return version_matrix


def get_range_for_version(version_string: str) -> Union[List[str], str]:
    version_matrix = get_version_dict()
    if version_string not in version_matrix:
        if version.parse(version_string) > max([version.parse(version_range) for version_range in version_matrix]):
            return "New version, we can't know if it is compatible, please upgrade if errors occur."
        else:
            return "Old version not in the compatibility matrix, proceed at your own discretion."
    else:
        return version_matrix[version_string]
