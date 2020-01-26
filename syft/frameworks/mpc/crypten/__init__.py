import crypten


def add_val(key: str, val: str):
    assert crypten.communicator.DistributedCommunicator.is_initialized()

    crypten.communicator.DistributedCommunicator.get().store.set(key, val)

def get_val(key: str):
    assert crypten.communicator.DistributedCommunicator.is_initialized()

    return crypten.communicator.DistributedCommunicator.get().store.get(key)


__all__ = ["add_val", "get_val"]
