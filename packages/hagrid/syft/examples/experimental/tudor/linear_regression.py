# stdlib
from typing import Any

# syft absolute
import syft as sy

alice = sy.VirtualMachine(name="alice")
alice_client = alice.get_root_client()
remote_python = alice_client.syft.lib.python


def get_permission(obj: Any) -> None:
    remote_obj = alice.store[obj.id_at_location]
    remote_obj.read_permissions[alice_client.verify_key] = obj.id_at_location


x_data = list(range(13))
y_data = [1, 3, 2, 5, 7, 8, 8, 9, 10, 12, 12, 13, 15]
zip(x_data, y_data)
x_data_ptr = sy.lib.python.List(x_data).send(alice_client)
y_data_ptr = sy.lib.python.List(y_data).send(alice_client)
print("gata")
# n_ptr = x_data_ptr.__len__()
#
# x_mean_ptr = sum(x_data_ptr) / n_ptr
# y_mean_ptr = sum(y_data_ptr) / n_ptr
#
# crossdev_list_ptr = remote_python.List()
# dev_list_ptr = remote_python.List()
#
# for x_ptr, y_ptr in zip(x_data_ptr, y_data_ptr):
#     crossdev_list_ptr.append(x_ptr * y_ptr)
#     dev_list_ptr.append(x_ptr * x_ptr)
#
#
# cross_dev_ptr = sum(crossdev_list_ptr) - n_ptr * x_mean_ptr * y_mean_ptr
# dev_ptr = sum(dev_list_ptr) - n_ptr * x_mean_ptr * x_mean_ptr
#
# b_1_ptr = cross_dev_ptr / dev_ptr
# b_0_ptr = y_mean_ptr - b_1_ptr * x_mean_ptr
#
# get_permission(b_1_ptr)
# get_permission(b_0_ptr)
#
# b_1 = b_1_ptr.get()
# b_0 = b_0_ptr.get()
#
# print(b_1)
# print(b_0)
