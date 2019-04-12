# This is just a convenience lookup table which maps from the string
# representation of a tensor type to a method which can cast to that
# type. This is useful for casting to ensure that two tensors have the
# same type.

# Example: if you have a tensor x and y and you want to cast y to be
# the same type as x, you can run...

# y = tensortype2caster[x.type()](y)

# and y will be properly casted

# TODO: make sure all types are represented here.

tensortype2caster = {}

tensortype2caster["torch.CharTensor"] = lambda x: x.char()
tensortype2caster["torch.HalfTensor"] = lambda x: x.half()
tensortype2caster["torch.FloatTensor"] = lambda x: x.float()
tensortype2caster["torch.DoubleTensor"] = lambda x: x.double()
tensortype2caster["torch.IntTensor"] = lambda x: x.int()
tensortype2caster["torch.LongTensor"] = lambda x: x.long()
tensortype2caster["torch.ByteTensor"] = lambda x: x.byte()
