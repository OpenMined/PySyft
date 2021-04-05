# stdlib
from collections import OrderedDict
from typing import Dict

# third party
import pytest

# syft absolute
import syft as sy


@pytest.mark.vendor(lib="pandas")
def test_pandas(root_client: sy.VirtualMachineClient) -> None:
    sy.load("pandas")
    # third party
    import pandas as pd

    data = {
        "col_1": {0: 3, 1: 2, 2: 1, 3: 0},
        "col_2": {0: "a", 1: "b", 2: "c", 3: "d"},
    }
    df = pd.DataFrame.from_dict(data)

    df_ptr = df.send(root_client)

    df2 = df_ptr.get()
    assert df2.to_dict() == data


@pytest.mark.vendor(lib="pandas")
def test_pd_categoriesdtype(root_client: sy.VirtualMachineClient) -> None:
    sy.load("pandas")
    # third party
    import pandas as pd

    categories = ["b", "a"]
    ordered = False
    t = pd.CategoricalDtype(categories=categories, ordered=ordered)

    t_ptr = t.send(root_client)

    t2 = t_ptr.get()
    print(t2)
    assert t2.categories.to_list() == categories
    assert t2.ordered == ordered


@pytest.mark.vendor(lib="pandas")
def test_pd_categories(root_client: sy.VirtualMachineClient) -> None:
    sy.load("pandas")
    # third party
    import pandas as pd

    categories = ["b", "a"]
    ordered = False
    t = pd.Categorical(
        ["b", "a", "c", "a", "b"], categories=categories, ordered=ordered
    )

    t_ptr = t.send(root_client)

    t2 = t_ptr.get()
    print(t2)
    assert t2.categories.to_list() == categories
    assert t2.ordered == ordered
    assert t2.codes.tolist() == t.codes.tolist()


@pytest.mark.vendor(lib="pandas")
def test_slice_dataframe(root_client: sy.VirtualMachineClient) -> None:
    sy.load("pandas")
    # third party
    import pandas as pd

    data: Dict[str, Dict] = {
        "col_1": {0: 3, 1: 2, 2: 1, 3: 0},
        "col_2": {0: "a", 1: "b", 2: "c", 3: "d"},
    }
    df = pd.DataFrame.from_dict(data)

    df_ptr = df.send(root_client)

    df_reverse_ptr = df_ptr[::-1]  # use slice to reverse the column data
    df_reverse = df_reverse_ptr.get()
    data_reverse = df_reverse.to_dict()

    assert OrderedDict(data_reverse["col_1"]) != OrderedDict(data["col_1"])
    assert OrderedDict(data_reverse["col_2"]) != OrderedDict(data["col_2"])

    assert OrderedDict(data_reverse["col_1"]) == OrderedDict(
        reversed(list(data["col_1"].items()))
    )
    assert OrderedDict(data_reverse["col_2"]) == OrderedDict(
        reversed(list(data["col_2"].items()))
    )
