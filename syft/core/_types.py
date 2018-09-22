from typing import Dict, List, Union, Any

PYTHON_ENCODE_RETURN_TYPE = Union[
    Dict[str, str],
    Dict[str, List[str]],
    Dict[str, Dict[str, List[str]]],
    List[Union[Dict[str, str], Dict[str, List[str]], Dict[str, Dict[str, List[str]]], List[str]]],
    # Dict[Union[str, Any],
    #      List[Union[Dict[str, str], Dict[str, List[str]],
    #                 Dict[str, Dict[str, List[str]]],
    #                 List[Union[Dict[str, str], Dict[str, ...], ..., ...]], ...]]],
    List[str]
]

CUSTOM_OBJECT_HOOK_RETURN_TYPE = Union[Dict[str, Union[int, Any]], slice]
