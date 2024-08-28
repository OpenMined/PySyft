# stdlib


def json_diff(json1: dict, json2: dict) -> dict:
    """
    Calculate the difference between two JSON objects and return the differences as a JSON object.

    Args:
        json1 (dict): The first JSON object.
        json2 (dict): The second JSON object.

    Returns:
        dict: A JSON object representing the differences.
    """

    def compare_dicts(d1: dict, d2: dict) -> dict:
        diffs = {}

        # Keys in d1 but not in d2 (deleted)
        for key in d1.keys() - d2.keys():
            diffs[key] = {"status": "deleted", "value": d1[key]}

        # Keys in d2 but not in d1 (added)
        for key in d2.keys() - d1.keys():
            diffs[key] = {"status": "added", "value": d2[key]}

        # Keys in both, but with different values (updated)
        for key in d1.keys() & d2.keys():
            if isinstance(d1[key], dict) and isinstance(d2[key], dict):
                # Recursively compare nested dictionaries
                nested_diff_dict = compare_dicts(d1[key], d2[key])
                if nested_diff_dict:
                    diffs[key] = nested_diff_dict
            elif isinstance(d1[key], list) and isinstance(d2[key], list):
                # Compare lists
                nested_diff_list = compare_lists(d1[key], d2[key])
                if nested_diff_list:
                    diffs[key] = nested_diff_list  # type: ignore
            elif d1[key] != d2[key]:
                diffs[key] = {
                    "status": "updated",
                    "old_value": d1[key],
                    "new_value": d2[key],
                }

        return diffs

    def compare_lists(l1: list, l2: list) -> list | None:
        diffs = []
        max_len = max(len(l1), len(l2))

        for i in range(max_len):
            if i >= len(l1):
                diffs.append({"status": "added", "value": l2[i]})
            elif i >= len(l2):
                diffs.append({"status": "deleted", "value": l1[i]})
            elif isinstance(l1[i], dict) and isinstance(l2[i], dict):
                # Recursively compare dictionaries in lists
                nested_diff_dict = compare_dicts(l1[i], l2[i])
                if nested_diff_dict:
                    diffs.append(nested_diff_dict)
            elif isinstance(l1[i], list) and isinstance(l2[i], list):
                # Recursively compare nested lists
                nested_diff_list = compare_lists(l1[i], l2[i])
                if nested_diff_list:
                    diffs.append(nested_diff_list)
            elif l1[i] != l2[i]:
                diffs.append(
                    {"status": "updated", "old_value": l1[i], "new_value": l2[i]}
                )

        return diffs if diffs else None

    # Generate the JSON diff
    json_diff_result = compare_dicts(json1, json2)

    return json_diff_result
