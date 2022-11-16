# stdlib
import json
import sys


def replace_variable(search_string: str, replace_string: str) -> bool:
    status = False
    try:
        # filepath
        file_path = "notebooks/medical-federated-learning-program/data-owners/data-owners-presentation.ipynb"

        # read file
        with open(file_path) as fp:
            data = fp.read()

        # notebook dataset
        notebook = json.loads(data)

        print("\nREAD NOTEBOOK")

        # Update dataset url
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                source_code = "".join(cell["source"])
                if search_string in source_code:
                    for i in range(len(cell["source"])):
                        if search_string in cell["source"][i]:
                            cell["source"][i] = cell["source"][i].replace(
                                search_string, replace_string
                            )
                            print("FOUND")
                            status = True

        new_data = json.dumps(notebook)
        with open(file_path, "w") as fp:
            fp.write(new_data)
    except Exception as e:
        print(e)
        return False

    return status


if __name__ == "__main__":
    data_url_arg = sys.argv[1]
    data_search_string = 'MY_DATASET_URL = ""'
    data_url = f'MY_DATASET_URL = "{data_url_arg}"'
    print("dataset", data_url)
    status = replace_variable(data_search_string, data_url)
    print("replacing data url", status)
    if len(sys.argv) > 2:
        host_ip = sys.argv[2]
        host_ip_search = "auto_detect_domain_host_ip()"
        host_ip_replace = f'"{host_ip}"'
        print("ip", host_ip_replace)
        status = replace_variable(host_ip_search, host_ip_replace)
        print("replacing host ip", status)
    if len(sys.argv) > 3:
        institution = sys.argv[3]
        institution_search_string = 'DOMAIN_NAME = "My Institution Name"'
        institution_name = f'DOMAIN_NAME = "{institution}"'
        print("institution", institution_name)
        status = replace_variable(institution_search_string, institution_name)
        print("replacing institution name", status)

    if len(sys.argv) > 4:
        network_name = sys.argv[4]
        network_search = 'NETWORK_NAME = ""'
        network_variable = f'NETWORK_NAME = "{network_name}"'
        print("network", network_variable)
        status = replace_variable(network_search, network_variable)
        print("replacing network name", status)
