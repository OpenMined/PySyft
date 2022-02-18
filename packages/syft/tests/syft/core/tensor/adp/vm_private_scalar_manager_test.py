# syft absolute
import syft as sy
from syft.core.adp.entity import Entity
from syft.core.adp.vm_private_scalar_manager import VirtualMachinePrivateScalarManager


def test_private_manager_serde() -> None:
    manager_one = VirtualMachinePrivateScalarManager()

    # check GammaScalar equality
    manager_one_prime = manager_one.get_symbol(
        min_val=float(0.0),
        value=float(10.0),
        max_val=float(100.0),
        entity=Entity("Phishan"),
    )

    gamma_scalar = manager_one.prime2symbol[manager_one_prime]
    gamma_ser = sy.serialize(gamma_scalar, to_bytes=True)
    gamma_de = sy.deserialize(gamma_ser, from_bytes=True)
    assert gamma_scalar == gamma_de

    manager_ser = sy.serialize(manager_one, to_bytes=True)
    manager_de = sy.deserialize(manager_ser, from_bytes=True)

    assert manager_one.prime_factory == manager_de.prime_factory
    assert manager_one.prime2symbol == manager_de.prime2symbol

    # check VirtualMachinePrivateScalarManager equality after serde
    assert manager_one == manager_de

    # check equality with separate VirtualMachinePrivateScalarManager
    manager_two = VirtualMachinePrivateScalarManager()
    manager_two_prime = manager_two.get_symbol(
        min_val=float(0.0),
        value=float(10.0),
        max_val=float(100.0),
        entity=Entity("Phishan"),
    )

    gamma_scalar2 = manager_two.prime2symbol[manager_two_prime]
    assert gamma_scalar == gamma_scalar2

    # the state is the same so the managers are the same
    assert manager_one == manager_two
