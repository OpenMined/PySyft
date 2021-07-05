# stdlib
import time
from uuid import UUID

# third party
import numpy as np

# syft absolute
import syft as sy
from syft.core.common.uid import UID
from syft.core.tensor.fixed_precision_tensor import FixedPrecisionTensor
from syft.core.tensor.share_tensor import ShareTensor


def thread_func():
    time.sleep(4)
    print("Sending other share with id {id_other}")

    generator = np.random.default_rng(seed=42)
    id_other = UID(UUID(bytes=generator.bytes(16)))
    share2_ptr = share2.send(client, id_at_location=id_other)


vm = sy.VirtualMachine(name="alice")
client = vm.get_client()


value1 = np.array([1, 2, 3, 4, -5])

share1 = ShareTensor(rank=0, value=value1)
share1_ptr = share1.send(client)

value2 = np.array([100])
share2 = ShareTensor(rank=0, value=value2)


share3_ptr = share1_ptr.smpc_test()

thread = Thread(thread_func)


print(share3_ptr.get())
