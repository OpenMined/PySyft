# third party
import numpy as np

# syft absolute
import syft as sy
from syft.core.adp.data_subject_ledger import DataSubjectLedger


def test_data_subject_ledger_serde() -> None:
    rdp_constants = np.array([1, 2, 3], np.float64)

    dsl = DataSubjectLedger()
    dsl._rdp_constants = rdp_constants

    ser = sy.serialize(dsl, to_bytes=True)
    de = sy.deserialize(ser, from_bytes=True)

    assert de == dsl
    assert all(de._rdp_constants == rdp_constants)
