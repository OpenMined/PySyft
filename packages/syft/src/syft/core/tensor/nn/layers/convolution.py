# third party
import numpy as np

# relative
from ...autodp.phi_tensor import PhiTensor
from ..activations import leaky_ReLU
from ..initializations import XavierInitialization
from .base import Layer


class Convolution(Layer):
    """
    If this is the first layer in a model, provide the keyword argument `input_shape`
    (tuple of integers, does NOT include the sample axis, N.),
    e.g. `input_shape=(3, 128, 128)` for 128x128 RGB pictures.
    """

    def __init__(self, nb_filter, filter_size, input_shape=None, stride=1):
        self.nb_filter = nb_filter
        self.filter_size = filter_size
        self.input_shape = input_shape
        self.stride = stride

        self.W, self.dW = None, None
        self.b, self.db = None, None
        self.out_shape = None
        self.last_output = None
        self.last_input = None

        self.init = XavierInitialization()
        self.activation = leaky_ReLU()

    def connect_to(self, prev_layer=None):
        if prev_layer is None:
            assert self.input_shape is not None
            input_shape = self.input_shape
        else:
            input_shape = prev_layer.out_shape

        # input_shape: (batch size, num input feature maps, image height, image width)
        assert len(input_shape) == 4

        nb_batch, pre_nb_filter, pre_height, pre_width = input_shape
        if isinstance(self.filter_size, tuple):
            filter_height, filter_width = self.filter_size
        elif isinstance(self.filter_size, int):
            filter_height = filter_width = self.filter_size
        else:
            raise NotImplementedError

        height = (pre_height - filter_height) // self.stride + 1
        width = (pre_width - filter_width) // self.stride + 1

        # output shape
        self.out_shape = (nb_batch, self.nb_filter, height, width)
        print(self.out_shape)

        # filters
        self.W = self.init((self.nb_filter, pre_nb_filter, filter_height, filter_width))
        self.b = np.zeros((self.nb_filter,))

    def forward(self, input: PhiTensor, *args, **kwargs):

        self.last_input = input

        # TODO: This could fail if the DP Tensor has < 4 dimensions

        # shape
        nb_batch, input_depth, old_img_h, old_img_w = input.shape
        if isinstance(self.filter_size, tuple):
            filter_height, filter_width = self.filter_size
        elif isinstance(self.filter_size, int):
            filter_height = filter_width = self.filter_size
        else:
            raise NotImplementedError

        new_img_h, new_img_w = self.out_shape[2:]

        # init
        outputs = np.zeros((nb_batch, self.nb_filter, new_img_h, new_img_w))



        # convolution operation
        for x in np.arange(nb_batch):
            for y in np.arange(self.nb_filter):
                for h in np.arange(new_img_h):
                    for w in np.arange(new_img_w):
                        h_shift, w_shift = h * self.stride, w * self.stride
                        # patch: (input_depth, filter_h, filter_w)
                        patch = input[x, :, h_shift: h_shift + filter_height, w_shift: w_shift + filter_width]
                        outputs[x, y, h, w] = np.sum(patch.child * self.W[y]) + self.b[y]

        # nonlinear activation
        # self.last_output: (nb_batch, output_depth, image height, image width)

        # TODO: Min/max vals are direct function of private data- fix this when we have time
        outputs = PhiTensor(
            child=outputs,data_subjects=np.zeros_like(outputs),
            min_vals=outputs.min(), max_vals=outputs.max()
        )
        self.last_output = self.activation.forward(outputs)

        return self.last_output

    def backward(self, pre_grad, *args, **kwargs):

        # shape
        assert pre_grad.shape == self.last_output.shape
        nb_batch, input_depth, old_img_h, old_img_w = self.last_input.shape
        new_img_h, new_img_w = self.out_shape[2:]

        if isinstance(self.filter_size, tuple):
            filter_height, filter_width = self.filter_size
        elif isinstance(self.filter_size, int):
            filter_height = filter_width = self.filter_size
        else:
            raise NotImplementedError

#         filter_h, filter_w = self.filter_size
        old_img_h, old_img_w = self.last_input.shape[-2:]

        # gradients
        self.dW = np.zeros((self.W.shape))
        self.db = np.zeros((self.b.shape))
        delta = pre_grad * self.activation.derivative()

        # dW
        for r in np.arange(self.nb_filter):
            for t in np.arange(input_depth):
                for h in np.arange(filter_height):
                    for w in np.arange(filter_width):
                        input_window = self.last_input[:, t,
                                       h:old_img_h - filter_height + h + 1:self.stride,
                                       w:old_img_w - filter_width + w + 1:self.stride]
                        delta_window = delta[:, r]
                        self.dW[r, t, h, w] = np.sum(input_window * delta_window) / nb_batch

        # db
        for r in np.arange(self.nb_filter):
            self.db[r] = np.sum(delta[:, r]) / nb_batch

        # dX
        if not self.first_layer:
            layer_grads = np.zeros(self.last_input.shape)
            for b in np.arange(nb_batch):
                for r in np.arange(self.nb_filter):
                    for t in np.arange(input_depth):
                        for h in np.arange(new_img_h):
                            for w in np.arange(new_img_w):
                                h_shift, w_shift = h * self.stride, w * self.stride
                                layer_grads[b, t, h_shift:h_shift + filter_height, w_shift:w_shift + filter_width] += \
                                    self.W[r, t] * delta[b, r, h, w]
            return layer_grads

    @property
    def params(self):
        return self.W, self.b

    @property
    def grads(self):
        return self.dW, self.db
