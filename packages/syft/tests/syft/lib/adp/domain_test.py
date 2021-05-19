# third party
import numpy as np

# syft absolute
import syft as sy
from syft.lib.adp.adversarial_accountant import AdversarialAccountant
from syft.lib.adp.entity import Entity
from syft.lib.adp.publish import publish
from syft.lib.adp.scalar import PhiScalar
from syft.lib.adp.tensor import Tensor

# def test_autodp_phiscalar_publish_domain(client: sy.VirtualMachineClient) -> None:
#     x_ptr = PhiScalar(0, 0.01, 1).send(client, tags=["x"])
#     y_ptr = PhiScalar(0, 0.02, 1).send(client, tags=["y"])
#     z_ptr = PhiScalar(0, 0.02, 1).send(client, tags=["z"])

#     assert x_ptr.__class__.__name__ == "PhiScalarPointer"

#     print(client.store)

#     o_ptr = x_ptr * x_ptr + y_ptr * y_ptr + z_ptr
#     p_ptr = o_ptr * o_ptr * o_ptr

#     o = o_ptr.get()
#     p = p_ptr.get()

#     print("o", o)
#     print("p", p)

#     # acc = AdversarialAccountant(max_budget=10)
#     # z.publish(acc=acc, sigma=0.2)

#     # publish([z, z], acc=acc, sigma=0.2)

#     # acc.print_ledger()
#     # assert len(acc.entities) == 3
#     # assert True is False
