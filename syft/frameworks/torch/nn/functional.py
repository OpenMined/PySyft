import torch


def linear(*args):
    """
    Un-hook the function to have its detailed behaviour
    """
    return torch.nn.functional.native_linear(*args)


def dropout(input, p=0.5, training=True, inplace=False):
    """
    Args:
        p: probability of an element to be zeroed. Default: 0.5
        training: If training, cause dropout layers are not used during evaluation of model
        inplace: If set to True, will do this operation in-place. Default: False
    """

    if training:
        binomial = torch.distributions.binomial.Binomial(probs=1 - p)

        # we must convert the normal tensor to fixed precision before multiplication
        # Note that: Weights of a model are alwasy Float values
        # Hence input will always be of type FixedPrecisionTensor > ...
        noise = (binomial.sample(input.shape).type(torch.FloatTensor) * (1.0 / (1.0 - p))).fix_prec(
            **input.get_class_attributes(), no_wrap=True
        )

        if inplace:
            input = input * noise
            return input

        return input * noise

    return input


def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    """
        Proposal for unrolling convolution: (WIP)

        # Change to tuple if not one
        stride = torch.nn.modules.utils._pair(stride)
        padding = torch.nn.modules.utils._pair(padding)
        dilation = torch.nn.modules.utils._pair(dilation)
        
        if padding != (0, 0):
            padding_mode = "constant"
            input = torch.nn.functional.pad(
                input, (padding[1], padding[1], padding[0], padding[0]), padding_mode
            )
        
        # Unrolling the input
        unrolled_input = torch.tensor([])
        for ba in range(input.shape[0]):
            batch = torch.tensor([])
            for ch in range(input.shape[1]):
                channel = []
                for i in range(0,input.shape[2],stride[0]):
                    if i+weight.shape[2]>input.shape[2]:
                        break
                    for j in range(0,input.shape[3],stride[1]):
                        temp_array = []
                        if j+weight.shape[3]>input.shape[3]:
                            break
                        for n in range(weight.shape[2]):
                            for m in range(weight.shape[3]): # assuming that weight is n x m
                                temp_array.append(input[ba, ch, i+n,j+m])
                        channel.append(temp_array)
                channel = torch.tensor(channel)
                if ch == 0:
                    batch = channel
                else:
                    batch = torch.cat([batch, channel], dim=1)
            output_tensor = batch
            if ba == 0:
                unrolled_input = output_tensor.reshape([1,*list(output_tensor.shape)])
            else:
                output_tensor = output_tensor.unsqueeze(dim=0)
                unrolled_input = torch.cat((unrolled_input,output_tensor),dim=0)
        # Dividing the batches into groups
        _unrolled_input = unrolled_input.transpose(1,2).unfold(1,unrolled_input.shape[2]//groups,unrolled_input.shape[2]//groups)
                
        # Unrolling the weights 
        batch_w = torch.tensor([])
        for b in range(weight.shape[0]):
            ch_w = torch.tensor([])
            for ch in range(weight.shape[1]):
                ch_w = torch.cat((ch_w,weight[b,ch].flatten()))
            ch_w = ch_w.reshape([ch_w.shape[0],1])
            if b == 0:
                batch_w = ch_w
            else:
                batch_w = torch.cat([ batch_w,ch_w], dim=1)
        
        if groups == 1:
            res = torch.matmul(_unrolled_input,batch_w).transpose(1,2)
        else:
            n = 0
            c = int(batch_w.shape[1]/g)
            for m in range(0, batch_w.shape[1], c):
                mul = torch.matmul(_unrolled_input[0,n:n+_unrolled_input.shape[-2]-1], batch_w.narrow(1,m,c))
                if m == 0:
                    res =  mul
                else:
                    res = torch.cat([res,mul], 0)
                n+=_unrolled_input.shape[-2]-1
        # Calculating the output shape
        input_width, input_height = (input.shape[2], input.shape[3])
        weight_width, weight_height = (weight.shape[2], weight.shape[3])
        output_width = (input_width-weight_width)//stride[0]+1
        output_height = (input_height-weight_height)//stride[1]+1
        shape = [input.shape[0], weight.shape[0], output_width, output_height]
        
        return res.reshape(shape)
    """





    """
    Overloads torch.nn.functional.conv2d to be able to use MPC on convolutional networks.
    The idea is to build new tensors from input and weight to compute a
    matrix multiplication equivalent to the convolution.
    Args:
        input: input image
        weight: convolution kernels
        bias: optional additive bias
        stride: stride of the convolution kernels
        padding:  implicit paddings on both sides of the input.
        dilation: spacing between kernel elements
        groups: split input into groups, in_channels should be divisible by the number of groups
    Returns:
        the result of the convolution (FixedPrecision Tensor)
    """

    assert len(input.shape) == 4
    assert len(weight.shape) == 4

    # Change to tuple if not one
    stride = torch.nn.modules.utils._pair(stride)
    padding = torch.nn.modules.utils._pair(padding)
    dilation = torch.nn.modules.utils._pair(dilation)

    # Extract a few useful values
    batch_size, nb_channels_in, nb_rows_in, nb_cols_in = input.shape
    nb_channels_out, nb_channels_kernel, nb_rows_kernel, nb_cols_kernel = weight.shape

    if bias is not None:
        assert len(bias) == nb_channels_out

    # Check if inputs are coherent
    assert nb_channels_in == nb_channels_kernel * groups
    assert nb_channels_in % groups == 0
    assert nb_channels_out % groups == 0

    # Compute output shape
    nb_rows_out = int(
        ((nb_rows_in + 2 * padding[0] - dilation[0] * (nb_rows_kernel - 1) - 1) / stride[0]) + 1
    )
    nb_cols_out = int(
        ((nb_cols_in + 2 * padding[1] - dilation[1] * (nb_cols_kernel - 1) - 1) / stride[1]) + 1
    )

    # Apply padding to the input
    if padding != (0, 0):
        padding_mode = "constant"
        input = torch.nn.functional.pad(
            input, (padding[1], padding[1], padding[0], padding[0]), padding_mode
        )
        # Update shape after padding
        nb_rows_in += 2 * padding[0]
        nb_cols_in += 2 * padding[1]

    # We want to get relative positions of values in the input tensor that are used by one filter convolution.
    # It basically is the position of the values used for the top left convolution.
    pattern_ind = []
    for ch in range(nb_channels_in):
        for r in range(nb_rows_kernel):
            for c in range(nb_cols_kernel):
                pixel = r * nb_cols_in * dilation[0] + c * dilation[1]
                pattern_ind.append(pixel + ch * nb_rows_in * nb_cols_in)

    # The image tensor is reshaped for the matrix multiplication:
    # on each row of the new tensor will be the input values used for each filter convolution
    # We will get a matrix [[in values to compute out value 0],
    #                       [in values to compute out value 1],
    #                       ...
    #                       [in values to compute out value nb_rows_out*nb_cols_out]]
    im_flat = input.view(batch_size, -1)
    im_reshaped = []
    for cur_row_out in range(nb_rows_out):
        for cur_col_out in range(nb_cols_out):
            # For each new output value, we just need to shift the receptive field
            offset = cur_row_out * stride[0] * nb_cols_in + cur_col_out * stride[1]
            tmp = [ind + offset for ind in pattern_ind]
            im_reshaped.append(im_flat[:, tmp])
    im_reshaped = torch.stack(im_reshaped).permute(1, 0, 2)

    # The convolution kernels are also reshaped for the matrix multiplication
    # We will get a matrix [[weights for out channel 0],
    #                       [weights for out channel 1],
    #                       ...
    #                       [weights for out channel nb_channels_out]].TRANSPOSE()
    weight_reshaped = weight.view(nb_channels_out // groups, -1).t()

    # Now that everything is set up, we can compute the result
    if groups > 1:
        res = []
        chunks_im = torch.chunk(im_reshaped, groups, dim=2)
        chunks_weights = torch.chunk(weight_reshaped, groups, dim=0)
        for g in range(groups):
            tmp = chunks_im[g].matmul(chunks_weights[g])
            res.append(tmp)
        res = torch.cat(res, dim=2)
    else:
        res = im_reshaped.matmul(weight_reshaped)

    # Add a bias if needed
    if bias is not None:
        if bias.is_wrapper and res.is_wrapper:
            res += bias
        elif bias.is_wrapper:
            res += bias.child
        else:
            res += bias

    # ... And reshape it back to an image
    res = (
        res.permute(0, 2, 1)
        .view(batch_size, nb_channels_out, nb_rows_out, nb_cols_out)
        .contiguous()
    )
    return res


def _pool(tensor, kernel_size: int = 2, stride: int = 2, mode="max"):
    output_shape = (
        (tensor.shape[0] - kernel_size) // stride + 1,
        (tensor.shape[1] - kernel_size) // stride + 1,
    )
    kernel_size = (kernel_size, kernel_size)
    b = torch.ones(tensor.shape)  # when torch.Tensor.stride() is supported: replace with A.stride()
    a_strides = b.stride()
    a_w = torch.as_strided(
        tensor,
        size=output_shape + kernel_size,
        stride=(stride * a_strides[0], stride * a_strides[1]) + a_strides,
    )
    a_w = a_w.reshape(-1, *kernel_size)
    result = []
    if mode is "max":
        for channel in range(a_w.shape[0]):
            result.append(a_w[channel].max())
    elif mode is "mean":
        for channel in range(a_w.shape[0]):
            result.append(torch.mean(a_w[channel]))
    else:
        raise ValueError("unknown pooling mode")

    result = torch.stack(result).reshape(output_shape)
    return result


def pool2d(tensor, kernel_size: int = 2, stride: int = 2, mode="max"):
    assert len(tensor.shape) < 5
    if len(tensor.shape) == 2:
        return _pool(tensor, kernel_size, stride, mode)
    if len(tensor.shape) == 3:
        return torch.squeeze(pool2d(torch.unsqueeze(tensor, dim=0), kernel_size, stride, mode))
    batches = tensor.shape[0]
    channels = tensor.shape[1]
    out_shape = (
        batches,
        channels,
        (tensor.shape[2] - kernel_size) // stride + 1,
        (tensor.shape[3] - kernel_size) // stride + 1,
    )
    result = []
    for batch in range(batches):
        for channel in range(channels):
            result.append(_pool(tensor[batch][channel], kernel_size, stride, mode))
    result = torch.stack(result).reshape(out_shape)
    return result


def maxpool2d(tensor, kernel_size: int = 2, stride: int = 2):
    return pool2d(tensor, kernel_size, stride)


def avgpool2d(tensor, kernel_size: int = 2, stride: int = 2):
    return pool2d(tensor, kernel_size, stride, mode="mean")
