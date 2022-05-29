# stdlib
import json
import sys


def add_dataset_url_to_notebook(dataset_url):

    status = True

    try:
        data_url = f'MY_DATASET_URL = "{dataset_url}"'

        # search string
        search_string = 'MY_DATASET_URL = ""'

        # filepath
        file_path = "notebooks/medical-federated-learning-program/data-owners/data-owners-presentation.ipynb"

        # read file
        with open(file_path) as fp:
            data = fp.read()

        # notebook dataset
        notebook = json.loads(data)

        print("READ NOTEBOOK")

        # Update dataset url
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                source_code = "".join(cell["source"])
                if search_string in source_code:
                    for i in range(len(cell["source"])):
                        if search_string in cell["source"][i]:
                            cell["source"][i] = cell["source"][i].replace(
                                search_string, data_url
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
    # get data url from args
    data_url = sys.argv[-1]
    # update dataset url to notebook
    status = add_dataset_url_to_notebook(data_url)
    print(status)
