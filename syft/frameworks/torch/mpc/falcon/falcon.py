import torch

import syft as sy
from syft.frameworks.torch.mpc.falcon.falcon_helper import FalconHelper

l = 2 ** 5
L = 2 ** l
p = 37


def conv2d(filters: sy.ReplicatedSharingTensor, image, padding=0):
    is_wrapper = filters.is_wrapper or image.is_wrapper

    if filters.is_wrapper:
        filters = filters.child

    if image.is_wrapper:
        image = image.child

    image_batches, image_channels, image_width, image_height = image.shape
    channels_out, filter_channels, filter_width, filter_height = filters.shape
    image = FalconHelper.unfold(image, filter_height, padding)
    filters = filters.view(channels_out, -1)
    result = filters @ image
    output_size = (image_height - filter_height + 2 * padding) + 1
    result = result.view(-1, channels_out, output_size, output_size)

    if is_wrapper:
        result = result.wrap()
    return result


def private_compare(x_bit_sh, r_bit, beta_b, m):
    # 2)
    u = FalconHelper.determine_sign(x_bit_sh - r_bit, beta_b)
    # 3)
    w = FalconHelper.xor(x_bit_sh, r_bit)
    # 4)
    wc = w.flip(-1).cumsum(-1).flip(-1) - w
    c = u + wc + 1
    # 6)
    # TODO: Figure out how to do prod() privately?
    c_p = c.reconstruct().prod()
    d = (m * c_p).reconstruct()
    # 7)
    beta_prime = torch.tensor([int(d != 0)]).share(
        *x_bit_sh.players, protocol="falcon", field=2, **FalconHelper.no_wrap
    )
    # 8)
    return FalconHelper.xor(beta_b, beta_prime)
