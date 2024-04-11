def dict_upper_keys(env_dict: dict[str, str]) -> dict[str, str]:
    """Convert all keys in a dictionary to uppercase"""

    return {key.upper(): val for key, val in env_dict.items()}


def dict_lower_keys(env_dict: dict[str, str]) -> dict[str, str]:
    """Convert all keys in a dictionary to lowercase"""

    return {key.lower(): val for key, val in env_dict.items()}
