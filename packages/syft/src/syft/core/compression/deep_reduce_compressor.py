import torch

from .specialized_compressor import SpecializedCompressor as Compressor
from .compression_params import compression_params
from .compressor import pack_grace
from .compressor import unpack_grace

"""
DeepReduce compression algorithms.
adapted from implementation by Author:Kelly Kostopoulou https://github.com/hangxu0304/DeepReduce
Kostopoulou K, Xu H, Dutta A, et al. DeepReduce: A Sparse-tensor Communication Framework for Distributed Deep Learning[J]. arXiv preprint arXiv:2102.03112, 2021.
"""

########################################################################
# DeepReduce Framework


class SparseCompressor(object):
    """Interface for compressing and decompressing a given sparse tensor."""

    @staticmethod
    def compress(sparse_tensor, ctx):
        """Compresses a sparse tensor and returns it with the context needed to decompress it."""
        raise NotImplemented("compress was not implemented.")

    @staticmethod
    def decompress(sparse_tensor, ctx):
        """Decompress a sparse tensor with the given context."""
        raise NotImplemented("decompress was not implemented.")


def deepreduce_from_params(params):
    from grace_dl.dist.helper import grace_from_params
    # only sparsification compressors are valid if params['deepreduce'] is not None
    if params['compressor'] in ['SKCompressCPU', 'sketch', 'SKCompressGPU']:
        hash_30m = torch.load("./deepreduce/hash_table_18m_int.pt").cuda()
        params['hash_table'] = hash_30m
    grc = grace_from_params(params)

    deepreduce = params.get('deepreduce', None)  # value, index, both
    deepreduce_wrapper = {'value': ValueCompressor,
                          'index': IndexCompressor,
                          'both': DeepReduce}
    if deepreduce:
        # hash table has to be passed in by dictionary to avoid excessive memory footprints on GPU
        if params.get('index', 'bloom') in ['bloom']:
            hash_30m = torch.load("/data/scratch/hang/deepreduce/hash_table_18m_int.pt").cuda()
            params['hash_table'] = hash_30m
        compressor = deepreduce_wrapper[deepreduce](grc.compressor, params)
        grc.compressor = compressor

    return grc


class ValueCompressor(Compressor):

    def __init__(self, sparsifier=sparsifier, params=None):
        super().__init__(tensors_size_are_same=sparsifier.tensors_size_are_same)
        self.sparsifier = sparsifier
        self.params = params
        self.val_compressor = compressor[self.params.get('value', 'polyfit')]
        if self.params.get('value', 'polyfit') not in ['polyfit']:
            self.tensors_size_are_same = False

    def compress(self, tensor, name):
        tensors, ctx = self.sparsifier.compress(tensor, name)
        vals, idxs = tensors
        shape = ctx

        # for performance reason, we don't apply compression for small sparse tensors
        # in our experiments when top-r ratio is set to 0.01, we use 1000 to filter tensors.
        if shape.numel() > 1000:
            sparse_tensor = vals, idxs, tensor.size()
            import time
            # torch.cuda.synchronize()
            start = time.time()
            vals, idxs, shape = self.val_compressor.compress(sparse_tensor, self.params)
            if self.params.get('micro-benchmark', False):
                # torch.cuda.synchronize()
                # print(f'val_compression time:{time.time() - start}')
            ctx = shape
        return (vals, idxs), ctx

    def decompress(self, tensors, ctx):
        shape = ctx
        vals, idxs = tensors

        if shape.numel() > 1000:
            import time
            # torch.cuda.synchronize()
            start = time.time()
            fitted_sparse_tensor = vals, idxs, shape
            vals, idxs, shape = self.val_compressor.decompress(fitted_sparse_tensor, self.params)
            if self.params.get('micro-benchmark', False):
                # torch.cuda.synchronize()
                # print(f'val_decompression time:{time.time() - start}')
                dense_tensor_bits = shape.numel() * 32
                # print(f'idx_relative_volume: {(tensor_bits([tensors[1]]) / dense_tensor_bits):.4f}')
                # print(f'val_relative_volume: {(tensor_bits([tensors[0]]) / dense_tensor_bits):.4f}')
        tensor_decompressed = self.sparsifier.decompress((vals, idxs), shape)
        return tensor_decompressed


class IndexCompressor(Compressor):

    def __init__(self, sparsifier=sparsifier, params=None):
        super().__init__(tensors_size_are_same=sparsifier.tensors_size_are_same)
        self.sparsifier = sparsifier
        self.params = params
        self.idx_compressor = compressor[self.params.get('index', 'bloom')]
        if self.params.get('index', 'bloom') not in ['bloom', ]:
            self.tensors_size_are_same = False

    def compress(self, tensor, name):
        tensors, ctx = self.sparsifier.compress(tensor, name)
        vals, idxs = tensors
        shape = ctx

        if shape.numel() > 1000:
            sparse_tensor = vals, idxs, tensor.size()
            self.params['dense_tensor'] = tensor

            import time
            # torch.cuda.synchronize()
            start = time.time()

            vals, idxs, shape = self.idx_compressor.compress(sparse_tensor, self.params)

            if self.params.get('micro-benchmark', False):
                # torch.cuda.synchronize()
                # print(f'idx_compression time:{time.time() - start}')

            ctx = shape
        return (vals, idxs), ctx

    def decompress(self, tensors, ctx):
        shape = ctx
        vals, idxs = tensors

        if shape.numel() > 1000:
            bf_sparse_tensor = vals, idxs, shape

            import time
            # torch.cuda.synchronize()
            start = time.time()

            vals, idxs, shape = self.idx_compressor.decompress(bf_sparse_tensor, self.params)

            if self.params.get('micro-benchmark', False):
                # torch.cuda.synchronize()
                # print(f'idx_decompression time:{time.time() - start}')
                dense_tensor_bits = shape.numel() * 32
                # print(f'idx_relative_volume: {(tensor_bits([tensors[1]]) / dense_tensor_bits):.4f}')
                # print(f'val_relative_volume: {(tensor_bits([tensors[0]]) / dense_tensor_bits):.4f}')

        tensor_decompressed = self.sparsifier.decompress((vals, idxs), shape)
        return tensor_decompressed


class DeepReduce(Compressor):
    grad_compressor = True
    def __init__(self, sparsifier=sparsifier, params=compression_params.getattr("deep_reduce", None)):
        super().__init__(tensors_size_are_same=sparsifier.tensors_size_are_same)
        self.sparsifier = sparsifier
        self.params = params
        self.val_compressor = compressor[self.params.get('value', 'polyfit')]
        self.idx_compressor = compressor[self.params.get('index', 'bloom')]

    @staticmethod
    def pack_(mapping, max_val=None):
        '''
        :param mapping: values to be packed
        :param bits: number of bits needed to represent each value
        :return: 64-bit singed integer tensor
        currently it only supports packing 3 values into one signed 64bit integer,
        '''
        bits = 21
        mapping = mapping.long()
        N = mapping.numel()
        chunk_size = 63 // bits
        padding = torch.zeros([chunk_size - N % chunk_size], device=mapping.device).long()
        padded = torch.cat([mapping, padding], dim=0).view(chunk_size, -1)
        encode = padded[0] * pow(2, 2 * bits) + padded[1] * pow(2, bits) + padded[2]
        return torch.cat([encode, torch.tensor([N]).cuda()], dim=0)

    @staticmethod
    def unpack_(encode):
        bits = 21
        N = encode[-1]
        encode = encode[:-1]
        a1 = encode // pow(2, 2 * bits)
        a3 = encode % pow(2, bits)
        a2 = (encode // pow(2, bits)) % pow(2, bits)
        unpack = torch.cat([a1, a2, a3], dim=0)[:N]
        return unpack.int()

    @staticmethod
    def pack(mapping, max_val=None):
        from math import log, ceil
        import cupy
        max_val = mapping.max().item() if max_val is None else max_val
        N = mapping.numel()
        bits = log(max_val, 2)

        t = mapping.int()
        a = [(t % 256).byte(), (t // 256 % 256).byte(), (t // pow(2, 16) % 256).byte(), (t // pow(2, 24)).byte(), ]
        body_len = int(ceil(bits) // 8)  # 0,1,2,3
        body = a[:body_len]
        head = a[body_len]

        head_len = ceil(bits) % 8
        exp = []
        if head_len > 0:
            head = cupy.asarray(head)
            for i in range(head_len):
                e = torch.as_tensor(cupy.packbits(head % 2), device="cuda")
                exp.append(e)
                head = cupy.right_shift(head, 1)

        N = torch.cuda.ByteTensor([N % 256, N // 256 % 256, N // pow(2, 16) % 256, N // pow(2, 24)])
        bits = torch.cuda.ByteTensor([ceil(bits)])
        encode = torch.cat([N, bits] + body + exp, dim=0)

        return encode

    @staticmethod
    def unpack(encode):
        from math import log, ceil
        import cupy
        N, bits, encode = encode.split([4, 1, encode.numel() - 5])
        N = torch.sum(N.int() * torch.cuda.IntTensor([1, 256, 256 ** 2, 256 ** 3])).item()
        bits = bits.item()
        body_len = int(bits // 8)  # 0,1,2,3
        head_len = ceil(bits) % 8
        decode = torch.zeros([N], device='cuda').int()
        if body_len > 0:
            body, exp = encode.split([body_len * N, encode.numel() - body_len * N])
            body = body.view(body_len, -1).int()
            for i in range(body_len):
                decode += body[i] * pow(2, i * 8)
        else:
            exp = encode

        if head_len > 0:
            head = torch.zeros([N], device='cuda').int()
            exp = exp.view(head_len, -1)
            for i in range(head_len):
                t = torch.as_tensor(cupy.unpackbits(cupy.asarray(exp[i])), device="cuda")[:N]
                head += t.int() * pow(2, i)
            decode += head * pow(2, body_len * 8)

        return decode.long()

    def compress(self, tensor, name='default'):
        tensors, ctx = self.sparsifier.compress(tensor, name)
        vals, idxs = tensors
        shape = ctx

        import time
        # torch.cuda.synchronize()
        start = time.time()

        if shape.numel() > 1000:
            sparse_tensor = vals, idxs, tensor.size()
            vals, idxs, _ = self.idx_compressor.compress(sparse_tensor, self.params)
            new_idxs = torch.arange(vals.numel(), device=vals.device)
            vals, mapping, shape = self.val_compressor.compress((vals, new_idxs, shape), self.params)
            # mapping = mapping.int()
            # mapping = self.pack(mapping, max_val=mapping.numel() - 1)
            ctx = shape
            tensors = (vals, idxs, mapping)

        return pack_grace(vals, idxs, ctx)

    def decompress(self, packed):
        tensors, ctx = unpack_grace(packed)
        
        shape = ctx

        import time
        # torch.cuda.synchronize()
        start = time.time()

        if shape.numel() > 1000:
            vals, idxs, mapping = tensors
            # mapping = self.unpack(mapping)
            fitted_sparse_tensor = vals, mapping, shape
            vals, _, _ = self.val_compressor.decompress(fitted_sparse_tensor, self.params)

            bf_sparse_tensor = mapping, idxs, shape
            _, idxs, _ = self.idx_compressor.decompress(bf_sparse_tensor, self.params)

            idxs = idxs[mapping.long()]
        else:
            vals, idxs = tensors

        if self.params.get('micro-benchmark', False):
            # torch.cuda.synchronize()
            # print(f'_decompression time:{time.time() - start}')
            dense_tensor_bits = shape.numel() * 32
            # print(f'idx_relative_volume: {(tensor_bits(tensors[1:]) / dense_tensor_bits):.4f}')
            # print(f'val_relative_volume: {(tensor_bits([tensors[0]]) / dense_tensor_bits):.4f}')

        tensor_decompressed = self.sparsifier.decompress((vals, idxs), shape)
        return tensor_decompressed


########################################################################
# PolyFit on GPU, not order-preserving

def GetInputMatrix_Polynomial(N, degree, device):
    '''
    degree: polynomial degree
    N: the number of elements in fitting values
    '''
    x = torch.arange(1, N + 1, device=device).view(-1, 1).float()
    t0 = torch.ones(degree + 1, device=device).view(1, -1)
    basis = torch.matmul(x, t0)

    t1 = torch.ones(N, device=device).view(-1, 1)
    t2 = torch.arange(0, degree + 1, device=device).view(1, -1).float()
    exp = torch.matmul(t1, t2)

    X_mat = torch.pow(basis, exp)
    del x, t0, t1, t2, basis, exp
    return X_mat


def LeastSquares(X, y):  # returns (X'X)^-1 X'y
    X = X.double()
    y = y.double()
    Xtrans = torch.transpose(X, 0, 1)
    tmp = torch.matmul(Xtrans, X)
    # torch.inverse on small matrix is much faster on cpu than GPU, see here:
    # the size of tmp is [degree x degree]
    # https://github.com/pytorch/pytorch/issues/2219
    inverse = torch.inverse(tmp.cpu()).cuda()

    theta_estimates = torch.matmul(torch.matmul(inverse, Xtrans), y)
    del Xtrans, tmp, inverse, X, y
    return theta_estimates


def RestoreValues(N, coefficients):
    degree = coefficients.numel() - 1
    X = GetInputMatrix_Polynomial(N, degree, device=coefficients.device)
    X = X.double()
    y = torch.matmul(X, coefficients.view(-1, 1))
    del X
    return y.view(-1)


# def get_segments(N, num_pos=0):
#     '''
#     for same N, it returns same segments
#     '''
#     segments = []
#     for r in [1/5, 1/10, 1 / 30, 1 / 100, 1/300, 1/1000, 1/3000, 1 / 10000, 1 / 30000, 1/100000]:
#         if int(N*r) > 30:
#             segments.append(int(N*r))
#     segments = segments[::-1] + [N-2*sum(segments)] + segments
#     return segments


def get_segments(N, num_pos=0):
    '''
    for different nodes,num_pos in the grad are different, thus segments are different
    need to set tensors_size_are_same=False for allgather communication
    this is useful for TopK due to the discontinuity between positive and negative values
    '''
    segments, pos, neg = [], [], []
    num_neg = N - num_pos
    for r in [1 / 5, 1 / 10, 1 / 30, 1 / 100, 1 / 300, 1 / 1000, 1 / 3000, 1 / 10000, 1 / 30000, 1 / 100000]:
        if int(num_pos * r) > 30:
            pos.append(int(num_pos * r))
        if int(num_neg * r) > 30:
            neg.append(int(num_neg * r))
    segments = pos[::-1] + [num_pos - sum(pos)] + [num_neg - sum(neg)] + neg

    return segments


class PolyFit(SparseCompressor):
    order_preserving = False
    @staticmethod
    def compress(sparse_tensor, params):
        sort = params.get('sort', False)
        degree = params.get('poly_degree', 5)
        vals, idxs, shape = sparse_tensor
        N = idxs.numel()
        y_all = vals
        num_pos = torch.sum(y_all > 0.0)

        if not sort:
            y_all, mapping = y_all.sort(descending=True)
            idxs = idxs[mapping]

        coefficients = []
        segments = get_segments(N, num_pos.item())

        for y in y_all.split(segments):
            n = y.numel()
            X = GetInputMatrix_Polynomial(n, degree, device=y_all.device)
            a = LeastSquares(X, y)
            coefficients.append(a)
            del X

        coefficients.append(num_pos.double().view(-1))
        coefficients_tensor = torch.cat(coefficients, dim=0)
        fitted_sparse_tensor = coefficients_tensor, idxs, shape
        del vals, y_all, idxs
        return fitted_sparse_tensor

    @staticmethod
    def decompress(fitted_sparse_tensor, params):
        coefficients_tensor, idxs, shape = fitted_sparse_tensor
        N = idxs.numel()
        coefficients_tensor, num_pos = coefficients_tensor.split([coefficients_tensor.numel() - 1, 1])
        segments = get_segments(N, num_pos.int().item())
        chunk_size = int(coefficients_tensor.numel()/len(segments))
        coefficients = coefficients_tensor.split(chunk_size)
        y_fit = []
        for a, n in zip(coefficients, segments):
            y_fit.append(RestoreValues(n, a))
        vals = torch.cat(y_fit, dim=0).float()
        sparse_tensor = vals, idxs, shape
        del y_fit, idxs
        return sparse_tensor


########################################################################
# Bloom on GPU, not order-preserving

class Bloomfilter(set):

    def __init__(self, size, num_hash, params, bit_array=None):
        super(Bloomfilter, self).__init__()
        if bit_array is not None:
            self.bit_array = bit_array
        else:
            self.bit_array = torch.zeros(size, device='cuda', dtype=torch.bool)
        self.size = size
        self.num_hash = min(num_hash, params['hash_table'].size()[1])
        self.params = params

    def __len__(self):
        return self.size

    def pack_bitarray(self):
        import cupy
        byte_tensor = self.bit_array.byte()
        pack = torch.as_tensor(cupy.packbits(cupy.asarray(byte_tensor)), device="cuda")
        self.bit_array = pack

    def unpack_bitarray(self):
        import cupy
        unpack = torch.as_tensor(cupy.unpackbits(cupy.asarray(self.bit_array)), device="cuda")
        self.bit_array = unpack[:self.size].bool()

    def add(self, items):
        '''
        items: list of integers or torch int tensor
        '''
        hash_table = self.params['hash_table'][:, :self.num_hash] % self.size
        index = hash_table[items.long()].flatten()
        self.bit_array[index.long()] = 1
        del index, hash_table

    def query(self, query_range):
        '''
        query_range will be range[0,query_range)
        returns all query results in this range
        '''
        hash_table = self.params['hash_table'][:, :self.num_hash] % self.size
        index = hash_table[:query_range]
        bits = self.bit_array[index.long()]
        mask = torch.sum(bits, dim=1) == bits.size()[1]
        positives = torch.arange(query_range, device='cuda')[mask]
        del index, bits, mask, hash_table
        return positives

    def policy(self, positives, k, policy):
        if policy == 'leftmost':
            res = positives[:k]
        elif policy == 'random':
            # if positives.numel() is always same, then with same seed, you will get same permutation
            # in practice, when we get all non-zero elements from gradient, positives.numel() won't be the
            # same across different iterations. However, if use TopK with fixed ratio here, we need to set
            # different seeds at different iterations.
            torch.manual_seed(42)
            keys = torch.randperm(positives.numel(), device=positives.device)[:k]
            res = positives[keys]
        elif policy == 'p0':
            res = positives
        return res


def get_BFconfig(capacity, fpr):
    import math
    # num_hash only depends on fpr
    num_hash = math.log(1 / fpr, 2)
    num_bits = num_hash * capacity / 0.693147180
    return math.ceil(num_hash), math.ceil(num_bits)


class Bloom(SparseCompressor):
    order_preserving = False
    @staticmethod
    def compress(sparse_tensor, params):
        vals, idxs, shape = sparse_tensor
        grad_size = shape.numel()
        num_indices = vals.numel()

        fpr = 0.1 * num_indices / grad_size
        fpr = params.get('fpr', fpr)
        policy = params.get('policy', 'leftmost')
        num_hash, bf_size = get_BFconfig(num_indices, fpr)
        bloom = Bloomfilter(bf_size, num_hash, params)
        bloom.add(idxs)

        # apply fpaware
        dense_tensor = params.get('dense_tensor', None)
        if dense_tensor is not None:
            query_res = bloom.query(grad_size)
            new_idxs = bloom.policy(query_res, idxs.numel(), policy)
            vals = dense_tensor.flatten()[new_idxs]

        if policy == 'p0':
            num_indices = torch.as_tensor([num_indices], dtype=vals.dtype, device=vals.device)
            vals = torch.cat([num_indices, vals], dim=0)

        bloom.pack_bitarray()
        bf_sparse_tensor = vals, bloom.bit_array, shape

        del vals, bloom
        return bf_sparse_tensor

    @staticmethod
    def decompress(bf_sparse_tensor, params):
        vals, bit_array, shape = bf_sparse_tensor
        policy = params.get('policy', 'leftmost')
        if policy == 'p0':
            num_indices, vals = vals.split([1, vals.numel()-1])
            num_indices = int(num_indices.item())
        else:
            num_indices = vals.numel()
        grad_size = shape.numel()
        fpr = 0.1 * num_indices / grad_size
        fpr = params.get('fpr', fpr)

        num_hash, bf_size = get_BFconfig(num_indices, fpr)
        bloom = Bloomfilter(bf_size, num_hash, params, bit_array=bit_array)
        bloom.unpack_bitarray()
        query_res = bloom.query(grad_size)
        idxs = bloom.policy(query_res, num_indices, policy)
        sparse_tensor = vals, idxs, shape
        del bloom, query_res
        return sparse_tensor


########################################################################
# PolyFit on CPU, not order-preserving
import numpy as np
import numpy.polynomial.polynomial as poly
import warnings
warnings.simplefilter('ignore', np.RankWarning)


def find_breaks(curve, num_of_breaks=10):
    # find breaks in a recursive ascending order
    y = curve
    breaks = []
    break_index = 0

    for i in range(num_of_breaks):
        if len(y) < 20 * num_of_breaks:
            break
        line = np.linspace(y[0], y[-1], len(y))
        distance = np.abs(line - y)
        break_index += np.argmax(distance)
        if (len(curve)-break_index) < 20 * num_of_breaks:
            break
        breaks.append(break_index)
        y = curve[break_index:]
    return breaks


def fit_curve(curve, breaks, poly_degree=5):
    breaks = [0] + breaks + [len(curve)]
    size = breaks[-1]
    x = list(range(size))

    yy = [curve[breaks[i - 1]:breaks[i]] for i in range(1, len(breaks))]
    xx = [x[breaks[i - 1]:breaks[i]] for i in range(1, len(breaks))]
    coefficients = []
    # plt.plot(x, curve)
    # # print("==debug==", breaks, )
    for i in range(len(xx)):
        x = xx[i]
        y = yy[i]
        # # print("==Debug==")
        # # print(x, "\n", y)
        z, _ = poly.polyfit(x, y, poly_degree, full=True)
        # set full=True to turn off the rank warning
        coefficients.append(z)
    return coefficients, breaks


def restore_curve(coefficients, breaks):
    size = breaks[-1]
    x = list(range(size))
    xx = [x[breaks[i - 1]:breaks[i]] for i in range(1, len(breaks))]
    curve_fit = []
    for i in range(len(xx)):
        x = xx[i]
        z = coefficients[i]
        # y_fit = [np.poly1d(z)(i) for i in x]
        y_fit = list(poly.polyval(x, z))
        # plt.plot(x, y_fit)
        curve_fit += y_fit
    #     # print(coefficients)
    return curve_fit


class PolyFitCPU(SparseCompressor):
    order_preserving = False
    @staticmethod
    def compress(sparse_tensor, params):
        vals, idxs, shape = sparse_tensor
        vals_sorted, mask = torch.sort(vals, descending=True)  # sorted in descending order
        indices_sorted = idxs[mask]
        vals_sorted = vals_sorted.cpu()

        # fit values
        num_of_breaks = 5
        poly_degree = 5

        vals_sorted = np.asarray(vals_sorted)
        # vals_sorted = np.array([float(x) for x in vals_sorted])
        num_pos = np.sum(vals_sorted > 0)
        if num_pos == 0:
            # all negtive values
            y = vals_sorted
            breaks = find_breaks(y, num_of_breaks)
            coefficients, breaks = fit_curve(y, breaks, poly_degree)

        elif num_pos == len(vals_sorted):
            # all positive values
            y = vals_sorted[::-1]  # reverse positive vals order to be ascending
            breaks = find_breaks(y, num_of_breaks)
            breaks = [len(y) - x for x in breaks[::-1]]

            y = vals_sorted  # fit positive vals in original order
            coefficients, breaks = fit_curve(y, breaks, poly_degree)

        else:
            vals_pos = vals_sorted[vals_sorted > 0]
            vals_neg = vals_sorted[vals_sorted < 0]

            y = vals_pos[::-1]  # reverse positive vals order to be ascending
            breaks = find_breaks(y, num_of_breaks)
            breaks_pos = [len(y) - x for x in breaks[::-1]]

            y = vals_neg
            breaks_neg = find_breaks(y, num_of_breaks)
            breaks_neg = [num_pos + x for x in breaks_neg]

            breaks = breaks_pos + [num_pos] + breaks_neg
            y = vals_sorted
            coefficients, breaks = fit_curve(y, breaks, poly_degree)

        coeff_tensor = torch.tensor(np.asarray(coefficients), dtype=torch.float64, device=idxs.device).flatten()
        breaks_tensor = torch.tensor(np.asarray(breaks), dtype=torch.int32, device=idxs.device)

        # todo: encode coeff_tensor and breaks_tensor into one tensor
        compressed_vals = coeff_tensor, breaks_tensor

        return compressed_vals, indices_sorted, shape

    @staticmethod
    def decompress(sparse_tensor, params):
        compressed_vals, idxs, shape = sparse_tensor
        coeff_tensor, breaks_tensor = compressed_vals

        coefficients = np.asarray(coeff_tensor.cpu())
        breaks = np.asarray(breaks_tensor.cpu())
        coefficients = np.reshape(coefficients, [len(breaks) - 1, -1])
        vals = restore_curve(coefficients, breaks)
        vals = torch.tensor(np.asarray(vals), dtype=torch.float32, device=idxs.device)

        return vals, idxs, shape


########################################################################
# Bloom on CPU, not order-preserving
from pybloomfilter import BloomFilter


class BloomCPU(SparseCompressor):
    order_preserving = False
    @staticmethod
    def compress(sparse_tensor, params):
        vals, idxs, shape = sparse_tensor
        grad_size = shape.numel()
        err_rate = len(idxs) * 0.1 / grad_size
        bf = BloomFilter(len(idxs), err_rate, '/tmp/test.bf')

        for i, index in enumerate(idxs):
            bf.add(int(index))

        # apply fpaware
        dense_tensor = params.get('dense_tensor', None)
        if dense_tensor is not None:
            idx_pos = []
            for idx in range(grad_size):
                if idx in bf:
                    idx_pos.append(idx)
            new_idxs = idx_pos[:len(idxs)]
            vals = dense_tensor.flatten()[new_idxs]

        bf64 = bf.to_base64()
        bf_tensor = torch.tensor(np.asarray([int(x) for x in bf64]), dtype=torch.uint8, device=vals.device)

        return vals, bf_tensor, shape

    @staticmethod
    def decompress(sparse_tensor, params):
        vals, bf_tensor, shape = sparse_tensor
        grad_size = shape.numel()
        b64_repr = bytes(list(np.asarray(bf_tensor.cpu())))
        bf = BloomFilter.from_base64('/tmp/test.bf', b64_repr)
        idx_pos = []

        for idx in range(grad_size):
            if idx in bf:
                idx_pos.append(idx)
        new_indices = idx_pos[:len(vals)]
        idxs = torch.as_tensor(new_indices, device=vals.device)
        return vals, idxs, shape


########################################################################
# Gzip on CPU, order-preserving

class Gzip(SparseCompressor):
    order_preserving = True
    @staticmethod
    def compress(sparse_tensor, params):
        import struct
        import zlib
        vals, idxs, shape = sparse_tensor
        data = vals.cpu()
        packed = struct.pack(f'{data.numel()}f', *data)
        zlib_packed = zlib.compress(packed)
        vals = torch.as_tensor(list(zlib_packed), dtype=torch.uint8, device=idxs.device)
        return vals, idxs, shape

    @staticmethod
    def decompress(gzip_sparse_tensor, params):
        import struct
        import zlib
        gzip_vals, idxs, shape = gzip_sparse_tensor
        gzip_vals = gzip_vals.cpu()
        packed = zlib.decompress(bytes(gzip_vals))
        vals = struct.unpack(f'{int(len(packed)//4)}f', packed)
        vals = torch.as_tensor(vals, dtype=torch.float, device=idxs.device)
        return vals, idxs, shape


########################################################################
# Huffman on CPU, order-preserving

class Huffman(SparseCompressor):
    order_preserving = True
    @staticmethod
    def compress(sparse_tensor, params):
        vals, idxs, shape = sparse_tensor

        import struct
        from dahuffman import HuffmanCodec
        data = torch.arange(shape.numel()).int()
        packed = struct.pack(f'{data.numel()}i', *data)
        codec = HuffmanCodec.from_data(packed)

        data = idxs.cpu().int()
        packed = struct.pack(f'{data.numel()}i', *data)
        huff_encoded = codec.encode(packed)

        idxs = torch.as_tensor(list(huff_encoded), dtype=torch.uint8, device=vals.device)
        return vals, idxs, shape

    @staticmethod
    def decompress(sparse_tensor, params):
        vals, idxs, shape = sparse_tensor
        import struct
        from dahuffman import HuffmanCodec
        data = torch.arange(shape.numel()).int()
        packed = struct.pack(f'{data.numel()}i', *data)
        codec = HuffmanCodec.from_data(packed)

        encoded = idxs.cpu()
        packed = codec.decode(bytes(encoded))
        idxs = struct.unpack(f'{int(len(packed) // 4)}i', packed)
        idxs = torch.as_tensor(idxs, dtype=torch.long, device=vals.device)
        return vals, idxs, shape


########################################################################
# RLE on CPU, not order-preserving

class RunLength(SparseCompressor):
    order_preserving = False
    @staticmethod
    def compress(sparse_tensor, params):
        vals, idxs, shape = sparse_tensor
        bitmap = torch.zeros([shape.numel()])
        bitmap[idxs.cpu().long()] = 1

        idxs, mapping = idxs.sort(descending=False)
        vals = vals[mapping]

        count = 0
        encode = []
        last = 0
        for val in bitmap:
            if val == last:
                count += 1
            else:
                encode.append(count)
                count = 1
                last = val
        encode.append(count)

        idxs = DeepReduce.pack(torch.as_tensor(encode, device=vals.device))
        return vals, idxs, shape

    @staticmethod
    def decompress(rle_sparse_tensor, params):
        vals, idxs, shape = rle_sparse_tensor
        encode = DeepReduce.unpack(idxs).cpu()

        decode = []
        for i, freq in enumerate(encode):
            val = i % 2
            decode += [val for _ in range(freq)]

        bitmap = torch.as_tensor(decode,  device=vals.device)
        idxs = torch.where(bitmap.bool())[0]
        return vals, idxs, shape


########################################################################
# QSGD on GPU, order-preserving

class QSGD(SparseCompressor):
    order_preserving = True
    @staticmethod
    def compress(sparse_tensor, params):
        vals_all, idxs, shape = sparse_tensor
        quantum_num = params.get('quantum_num', 127)
        bucket_size = params.get('bucket_size', 512)

        vals_qsgd = []
        for vals in vals_all.split(bucket_size):
            norm = vals.norm()
            abs_gradient = vals.abs()

            level_float = quantum_num / norm * abs_gradient
            previous_level = level_float.floor()
            prob = torch.empty_like(vals).uniform_()
            is_next_level = (prob < (level_float - previous_level)).type(torch.float32)
            new_level = (previous_level + is_next_level)

            sign = vals.sign()
            tensor_compressed = (new_level * sign)
            tensor_compressed = tensor_compressed.type(torch.int8 if quantum_num < 128 else torch.int16)

            #encode norm
            import struct
            packed = struct.pack('f', norm.cpu().item())
            norm = torch.as_tensor([b-128 for b in packed], dtype=torch.int8, device=idxs.device)

            vals = torch.cat([tensor_compressed, norm], dim=0)
            vals_qsgd.append(vals)
        vals = torch.cat(vals_qsgd, dim=0)

        return vals, idxs, shape

    @staticmethod
    def decompress(sparse_tensor, params):
        vals_qsgd, idxs, shape = sparse_tensor
        quantum_num = params.get('quantum_num', 127)
        bucket_size = params.get('bucket_size', 512)

        vals_all = []
        for vals in vals_qsgd.split(bucket_size+4):
            vals, norm = vals.split([vals.numel()-4, 4])

            # decode norm
            import struct
            norm = norm.int() + 128
            norm = bytes(norm.cpu().type(torch.ByteTensor))
            norm = struct.unpack('f', norm)[0]

            decode_output = vals.type(torch.float32)
            vals = norm / quantum_num * decode_output
            vals_all.append(vals)

        vals  = torch.cat(vals_all, dim=0)
        return vals, idxs, shape

def sparsify(tensor, compress_ratio):
    tensor = tensor.flatten()
    k = max(1, int(tensor.numel() * compress_ratio))
    _, indices = torch.topk(tensor.abs(), k, sorted=False,)
    values = torch.gather(tensor, 0, indices)
    return values, indices


def desparsify(tensors, numel):
    values, indices = tensors
    tensor_decompressed = torch.zeros(numel, dtype=values.dtype, layout=values.layout, device=values.device)
    tensor_decompressed.scatter_(0, indices, values)
    return tensor_decompressed


class TopKCompressor(Compressor):

    def __init__(self, compress_ratio):
        super().__init__()
        self.compress_ratio = compress_ratio

    def compress(self, tensor, name):
        tensors = sparsify(tensor, self.compress_ratio)
        ctx = tensor.numel(), tensor.size()
        return tensors, ctx

    def decompress(self, tensors, ctx):
        """Decompress by filling empty slots with zeros and reshape back using the original shape"""
        numel, shape = ctx
        tensor_decompressed = desparsify(tensors, numel)
        return tensor_decompressed.view(shape)


################################################################

sparsifier = TopKCompressor(compression_params.deep_reduce['compress_ratio'])

compressor = {
    "bloom": Bloom,
    "polyfit": PolyFit,
    "bloom_cpu": BloomCPU,
    "polyfit_cpu": PolyFitCPU,
    "gzip": Gzip,
    "huffman": Huffman,
    "rle": RunLength,
    "qsgd": QSGD,
}