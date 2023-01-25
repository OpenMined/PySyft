# third party
import numpy as np

# syft absolute
import syft as sy
from syft.core.node.new.action_service import ActionService
from syft.core.node.new.action_service import NumpyArrayObject

raw_data = np.array([1, 2, 3])

ds_client = sy.login(email="info@openmined.org", password="changethis", port=8081)

np_pointer = raw_data.send(ds_client).value
raw_data_2 = np.array([2, 3, 4])
np_pointer_2 = raw_data_2.send(ds_client).value
res = np_pointer - np_pointer_2
print(np_pointer.id)
print(np_pointer_2.id)
print(res)
print(ds_client.api.services.action.get(res.id).syft_action_data)
