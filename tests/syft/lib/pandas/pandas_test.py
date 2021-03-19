# third party
import pytest

# syft absolute
import syft as sy


@pytest.mark.vendor(lib="pandas")
def test_pandas() -> None:
    sy.load_lib("pandas")
    # third party
    import pandas as pd

    bob = sy.VirtualMachine(name="Bob")
    client = bob.get_root_client()

    data = {
        "col_1": {0: 3, 1: 2, 2: 1, 3: 0},
        "col_2": {0: "a", 1: "b", 2: "c", 3: "d"},
    }
    df = pd.DataFrame.from_dict(data)

    df_ptr = df.send(client)

    df2 = df_ptr.get()
    assert df2.to_dict() == data
