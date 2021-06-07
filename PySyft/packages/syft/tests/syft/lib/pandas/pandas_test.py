# stdlib
from collections import OrderedDict
from typing import Any
from typing import Dict
from typing import List

# third party
import pytest

# syft absolute
import syft as sy

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")
sy.load("pandas", "numpy")


@pytest.mark.vendor(lib="pandas")
def test_pandas(root_client: sy.VirtualMachineClient) -> None:
    data = {
        "col_1": {0: 3, 1: 2, 2: 1, 3: 0},
        "col_2": {0: "a", 1: "b", 2: "c", 3: "d"},
    }
    df = pd.DataFrame.from_dict(data)

    df_ptr = df.send(root_client)

    df2 = df_ptr.get()
    assert df2.to_dict() == data


@pytest.mark.parametrize("ordered", [True, False])
@pytest.mark.parametrize("categories", [["b", "a"], [1, 2, 3]])
@pytest.mark.vendor(lib="pandas")
def test_pd_categoriesdtype(
    root_client: sy.VirtualMachineClient,
    categories: List[Any],
    ordered: bool,
) -> None:
    t = pd.CategoricalDtype(categories=categories, ordered=ordered)

    t_ptr = t.send(root_client)

    t2 = t_ptr.get()
    print(t2)
    assert t2.categories.to_list() == categories
    assert t2.ordered == ordered


@pytest.mark.parametrize("ordered", [True, False])
@pytest.mark.parametrize("data", [["a", "a", "b", "f"], [1, 2, 3]])
@pytest.mark.vendor(lib="pandas")
def test_pd_categories(
    root_client: sy.VirtualMachineClient, data: List[Any], ordered: bool
) -> None:
    t = pd.Categorical(data, ordered=ordered)

    t_ptr = t.send(root_client)

    t2 = t_ptr.get()
    print(t2)
    assert (t2.categories.to_list() == t.categories).all()
    assert t2.ordered == ordered
    assert t2.codes.tolist() == t.codes.tolist()


@pytest.mark.vendor(lib="pandas")
def test_slice_dataframe(root_client: sy.VirtualMachineClient) -> None:
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


@pytest.mark.vendor(lib="pandas")
def test_pandas_json_normalize(root_client: sy.VirtualMachineClient) -> None:
    data = {"A": [1, 2]}
    df = pd.json_normalize(data)

    # create dict pointer
    sy_data = sy.lib.python.Dict(data)
    data_ptr = sy_data.send(root_client)

    remote_pandas = root_client.pandas
    df_ptr = remote_pandas.json_normalize(data_ptr)
    res_df = df_ptr.get()

    # Serde converts the list to an np.array. To allow comparison and prevent this test
    # being coupled with numpy as a dependency we just convert back to a list.
    res_df.iloc[0][0] = list(res_df.iloc[0][0])

    assert df.equals(res_df)
