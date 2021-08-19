# third party 
import torch as th
import time

# relative
from ...logger import error

class Compressor:
    """
    [Experimental: high-performace duet channel] accepts tensor objects and compresses using an optimum algorithm
    """

    def compress():
        pass

    def decompress():
        pass

def sync_compression(compression_params, client=None):
    if not hasattr(compression_params, "_client"):
        if not client:
            print('Compression params can only set by DS. Establish dual communication to enable DO setting.')
            error('Compression params can only set by DS. Establish dual communication to enable DO setting.')
            return False
    else:
        client = compression_params.client
    try:
        from syft.core.node.common.node_service.compression_manager.compression_messages import CompressionParamsMessage
        msg = CompressionParamsMessage(
            address=client.address,
            reply_to=client.address,
            compression_params=compression_params
        )
        resp = client.send_immediate_msg_with_reply(msg)
        if resp.resp_msg == 'Compression params updated successfully!':
            return True
    except:
        pass
    print('Compression params not updated: unable to sync compression_params.')
    error('Compression params not updated: unable to sync compression_params.')
    return False

def get_connection_speed(compression_params, client):
    try:
        from syft.core.node.common.node_service.compression_manager.compression_messages import CompressionParamsMessage
        sent = time.time()
        msg = CompressionParamsMessage(
            address=client.address,
            reply_to=client.address,
            status='connection_testing',
            compression_params=compression_params,
            time=sent,
        )
        size = len(msg._object2proto().SerializeToString())
        resp = client.send_immediate_msg_with_reply(msg)
        if resp.status == 'connection_testing_resp':
            recvd = time.time()
            return (2 * size) / (recvd - sent)
    except:
        pass
    print('Compression params not updated: unable to get connection speed.')
    error('Compression params not updated: unable to get connection speed.')
    return 0.0

def pack_grace(values, indices, size):
    res1 = th.cat((values, th.Tensor([0]*(len(size)-1) + [len(size)])))
    res2 = th.cat((indices, th.Tensor(list(size))))
    return th.cat((res1.reshape(1, -1), res2.reshape(1, -1)), dim=0)

def unpack_grace(packed):
    size_len = int(packed[0, -1])
    size = th.Size(packed[1, -size_len:].int().tolist())
    indices = packed[1, :-size_len].long()
    values = packed[0, :-size_len]

    return (values, indices), size