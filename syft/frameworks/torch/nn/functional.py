import torch as th


# The nn classes calls functional methods.
# Hence overloading functional methods allows
# to add custom functionality which also works
# with nn.layers.


def dropout(input, p=0.5, training=True, inplace=False):
    """
    Args:
        p: probability of an element to be zeroed. Default: 0.5
        training: If training, cause dropout layers are not used during evaluation of model
        inplace: If set to True, will do this operation in-place. Default: False
    """

    if training:
        binomial = th.distributions.binomial.Binomial(probs=1 - p)

        # we must convert the normal tensor to fixed precision before multiplication
        noise = (
            (binomial.sample(input.shape).type(th.FloatTensor) * (1.0 / (1.0 - p)))
            .fix_prec(**input.get_class_attributes())
            .child
        )

        if inplace:
            input = input * noise
            return input

        return input * noise

    return input
