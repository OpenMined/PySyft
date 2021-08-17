# third party 
import torch as th
class Compressor:
    """
    [Experimental: high-performace duet channel] accepts tensor objects and compresses using an optimum algorithm
    """

    def compress():
        pass

    def decompress():
        pass

def pack_grace(values, indices, size):
    res1 = th.cat((values, th.Tensor([0]*(len(size)-1) + [len(size)])))
    res2 = th.cat((indices, th.Tensor(list(size))))
    return th.cat((res1.reshape(1, -1), res2.reshape(1, -1)), dim=0)

def unpack_grace(packed):
    size_len = int(packed[0, -1])
    size = th.Size(packed[1, -size_len:].int().tolist())
    values = packed[0, :-size_len]
    indices = packed[1, :-size_len]

    return (values, indices), size