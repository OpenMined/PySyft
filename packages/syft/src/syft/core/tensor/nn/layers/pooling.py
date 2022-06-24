# third party
import numpy as np
from ..utils import dp_zeros

# relative
from ...autodp.phi_tensor import PhiTensor
from .base import Layer


class AvgPool(Layer):
    """Average pooling operation for spatial data.
    Parameters
    ----------
    pool_size : tuple of 2 integers,
        factors by which to downscale (vertical, horizontal).
        (2, 2) will halve the image in each dimension.
    Returns
    -------
    4D numpy.array
        with shape `(nb_samples, channels, pooled_rows, pooled_cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(samples, pooled_rows, pooled_cols, channels)` if dim_ordering='tf'.
    """

    def __init__(self, pool_size):
        self.pool_size = pool_size

        self.out_shape = 0
        self.out_shape = None
        self.input_shape = None

    def connect_to(self, prev_layer):
        assert 5 > len(prev_layer.out_shape) >= 3

        old_h, old_w = prev_layer.out_shape[-2:]
        pool_h, pool_w = self.pool_size
        new_h, new_w = old_h // pool_h, old_w // pool_w

        assert old_h % pool_h == old_w % pool_w == 0

        self.out_shape = prev_layer.out_shape[:-2] + (new_h, new_w)

    def forward(self, input: PhiTensor, *args, **kwargs):

        # shape
        self.input_shape = input.shape
        pool_h, pool_w = self.pool_size
        new_h, new_w = self.out_shape[-2:]

        # forward
        # outputs = np.zeros(self.input_shape[:-2] + self.out_shape[-2:])
        #outputs = PhiTensor(child=outputs, data_subjects=np.zeros_like(outputs), min_vals=0, max_vals=1)
        outputs = dp_zeros(self.input_shape[:-2] + self.out_shape[-2:], data_subjects=input.data_subjects)

        ndim = len(input.shape)
        if ndim == 4:
            nb_batch, nb_axis, _, _ = input.shape

            for a in np.arange(nb_batch):
                for b in np.arange(nb_axis):
                    for h in np.arange(new_h):
                        for w in np.arange(new_w):
                            outputs[a, b, h, w] = input[a, b, h:h + pool_h, w:w + pool_w].mean()

        elif ndim == 3:
            nb_batch, _, _ = input.shape

            for a in np.arange(nb_batch):
                for h in np.arange(new_h):
                    for w in np.arange(new_w):
                        outputs[a, h, w] = input[a, h:h + pool_h, w:w + pool_w].mean()

        else:
            raise ValueError()

        return outputs

    def backward(self, pre_grad: PhiTensor, *args, **kwargs):
        new_h, new_w = self.out_shape[-2:]
        pool_h, pool_w = self.pool_size
        length = np.prod(self.pool_size)

        layer_grads = dp_zeros(self.input_shape, pre_grad.data_subjects)
        # layer_grads = PhiTensor(child=layer_grads, data_subjects=self.input_data_subjects, min_vals=0, max_vals=1)

        ndim = len(pre_grad.shape)

        if ndim == 4:
            nb_batch, nb_axis, _, _ = pre_grad.shape

            for a in np.arange(nb_batch):
                for b in np.arange(nb_axis):
                    for h in np.arange(new_h):
                        for w in np.arange(new_w):
                            h_shift, w_shift = h * pool_h, w * pool_w
                            layer_grads[a, b, h_shift: h_shift + pool_h, w_shift: w_shift + pool_w] = \
                                pre_grad[a, b, h, w] * (1.0/length)

        elif ndim == 3:
            nb_batch, _, _ = pre_grad.shape

            for a in np.arange(nb_batch):
                for h in np.arange(new_h):
                    for w in np.arange(new_w):
                        h_shift, w_shift = h * pool_h, w * pool_w
                        layer_grads[a, h_shift: h_shift + pool_h, w_shift: w_shift + pool_w] = \
                            pre_grad[a, h, w] * (1.0/length)

        else:
            raise ValueError()

        return layer_grads


class MaxPool(Layer):
    """Max pooling operation for spatial data.
    Parameters
    ----------
    pool_size : tuple of 2 integers,
        factors by which to downscale (vertical, horizontal).
        (2, 2) will halve the image in each dimension.
    Returns
    -------
    4D numpy.array
        with shape `(nb_samples, channels, pooled_rows, pooled_cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(samples, pooled_rows, pooled_cols, channels)` if dim_ordering='tf'.
    """
    def __init__(self, pool_size):
        self.pool_size = pool_size

        self.input_shape = None
        self.out_shape = None
        self.last_input = None

    def connect_to(self, prev_layer):
        # prev_layer.out_shape: (nb_batch, ..., height, width)
        assert len(prev_layer.out_shape) >= 3

        old_h, old_w = prev_layer.out_shape[-2:]
        pool_h, pool_w = self.pool_size
        new_h, new_w = old_h // pool_h, old_w // pool_w

        assert old_h % pool_h == old_w % pool_w == 0

        self.out_shape = prev_layer.out_shape[:-2] + (new_h, new_w)

    def forward(self, input, *args, **kwargs):
        # shape
        self.input_shape = input.shape
        pool_h, pool_w = self.pool_size
        new_h, new_w = self.out_shape[-2:]

        # forward
        self.last_input = input
        outputs = np.zeros(self.input_shape[:-2] + self.out_shape[-2:])
        outputs = PhiTensor(child=outputs, data_subjects=np.zeros_like(outputs), min_vals=0, max_vals=1)

        ndim = len(input.shape)

        if ndim == 4:
            nb_batch, nb_axis, _, _ = input.shape

            for a in np.arange(nb_batch):
                for b in np.arange(nb_axis):
                    for h in np.arange(new_h):
                        for w in np.arange(new_w):
                            outputs[a, b, h, w] = input[a, b, h:h + pool_h, w:w + pool_w].max()

        elif ndim == 3:
            nb_batch, _, _ = input.shape

            for a in np.arange(nb_batch):
                for h in np.arange(new_h):
                    for w in np.arange(new_w):
                        outputs[a, h, w] = input[a, h:h + pool_h, w:w + pool_w].max()

        else:
            raise ValueError()


        return outputs

    def backward(self, pre_grad, *args, **kwargs):
        new_h, new_w = self.out_shape[-2:]
        pool_h, pool_w = self.pool_size

        layer_grads = np.zeros(self.input_shape)
        layer_grads = PhiTensor(child=layer_grads, data_subjects=np.zeros_like(layer_grads), min_vals=0, max_vals=1)

        ndim = len(pre_grad.shape)

        if ndim == 4:
            nb_batch, nb_axis, _, _ = pre_grad.shape

            for a in np.arange(nb_batch):
                for b in np.arange(nb_axis):
                    for h in np.arange(new_h):
                        for w in np.arange(new_w):
                            patch = self.last_input[a, b, h:h + pool_h, w:w + pool_w]
                            max_idx = patch.unravel_argmax()
                            #                             max_idx = np.unravel_index(patch.argmax(), patch.shape)

                            h_shift, w_shift = h * pool_h + max_idx[0], w * pool_w + max_idx[1]
                            layer_grads[a, b, h_shift, w_shift] = pre_grad[a, b, h, w]
        #                             layer_grads[a, b, h_shift, w_shift] = pre_grad[a, b, a, w]

        elif ndim == 3:
            nb_batch, _, _ = pre_grad.shape

            for a in np.arange(nb_batch):
                for h in np.arange(new_h):
                    for w in np.arange(new_w):
                        patch = self.last_input[a, h:h + pool_h, w:w + pool_w]
                        max_idx = patch.unravel_argmax()
                        #                         max_idx = np.unravel_index(patch.argmax(), patch.shape)
                        h_shift, w_shift = h * pool_h + max_idx[0], w * pool_w + max_idx[1]
                        layer_grads[a, h_shift, w_shift] = pre_grad[a, h, w]
        #                         layer_grads[a, h_shift, w_shift] = pre_grad[a, a, w]

        else:
            raise ValueError()

        return layer_grads
