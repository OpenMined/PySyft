from collections import OrderedDict
import pytest
from syft.serde import serde
import io
import numpy
import torch
import syft

# Make dict of type codes
CODE = OrderedDict()
for cls, simplifier in serde.simplifiers.items():
    CODE[cls] = simplifier[0]


########################################################################
# Functions that return list of serde samples in the following format:
# [
#   tuple(
#    original_value,
#    simplified_value,
#    custom_simplified_values_comparison_function, # optional
#    custom_detailed_values_comparison_function, # optional
#   ),
#   ...
# ]
########################################################################

########################################################################
# Native types.
########################################################################

# Dict.
def make_dict(**kwargs):
    return [
        (
            {1: "hello", 2: "world"},
            (CODE[dict], (
                (1, (CODE[str], (b"hello",))),
                (2, (CODE[str], (b"world",))),
            ))
        ),
        (
            {"hello": "world"},
            (CODE[dict], (
                ((CODE[str], (b"hello",)), (CODE[str], (b"world",))),
            ))
        ), (
            {},
            (CODE[dict], ())
        ),
    ]


# List.
def make_list(**kwargs):
    return [
        (
            ["hello", "world"],
            (CODE[list], (
                (CODE[str], (b"hello",)),
                (CODE[str], (b"world",)),
            ))
        ),
        (
            ["hello"],
            (CODE[list], (
                (CODE[str], (b"hello",)),
            ))
        ),
        (
            [],
            (CODE[list], ())
        ),
    ]


# Tuple.
def make_tuple(**kwargs):
    return [
        (
            ("hello", "world"),
            (CODE[tuple], (
                (CODE[str], (b"hello",)),
                (CODE[str], (b"world",)),
            ))
        ),
        (
            ("hello",),
            (CODE[tuple], (
                (CODE[str], (b"hello",)),
            ))
        ),
        (
            tuple(),
            (CODE[tuple], ())
        ),
    ]


# Set.
def make_set(**kwargs):
    return [
        (
            {"hello", "world"},
            (CODE[set], (
                (CODE[str], (b"world",)),
                (CODE[str], (b"hello",)),
            )),
            # Compare tuples as sets because set order is undefined
            lambda simplified, expected: simplified[0] == expected[0] and set(simplified[1]) == set(simplified[1])
        ),
        (
            {"hello"},
            (CODE[set], (
                (CODE[str], (b"hello",)),
            ))
        ),
        (
            set([]),
            (CODE[set], ())
        ),
    ]


# Slice.
def make_slice(**kwargs):
    return [
        (
            slice(10, 20, 30),
            (CODE[slice], (10, 20, 30))
        ),
        (
            slice(10, 20),
            (CODE[slice], (10, 20, None))
        ),
        (
            slice(10),
            (CODE[slice], (
                None, 10, None
            ))
        ),
    ]


# Range.
def make_range(**kwargs):
    return [
        (
            range(1, 3, 4),
            (CODE[range], (1, 3, 4))
        ),
        (
            range(1, 3),
            (CODE[range], (1, 3, 1))
        ),
    ]


# String.
def make_str(**kwargs):
    return [
        ("a string", (CODE[str], (b"a string",))),
        ("", (CODE[str], (b"",))),
    ]


# Int.
def make_int(**kwargs):
    return [
        (5, 5),
    ]


# Float.
def make_float(**kwargs):
    return [
        (5.1, 5.1),
    ]


# Ellipsis.
def make_ellipsis(**kwargs):
    return [(..., (CODE[type(Ellipsis)], b""))]


########################################################################
# Numpy.
########################################################################

# numpy.ndarray
def make_numpy_ndarray(**kwargs):
    np_array = numpy.random.random((2, 2))
    return [
        (
            np_array,
            (CODE[type(np_array)], (np_array.tobytes(), np_array.shape, np_array.dtype.name)),
            None,
            lambda detailed, original: numpy.array_equal(detailed, original)
        ),
    ]


########################################################################
# PyTorch.
########################################################################

# Utility functions.


def compare_modules(detailed, original):
    input = torch.randn(1, 1, 3, 3)
    return isinstance(detailed, torch.jit.ScriptModule) and \
           detailed.code == original.code and \
           detailed(input).equal(original(input))


def save_to_buffer(tensor) -> bin:
    binary_stream = io.BytesIO()
    torch.save(tensor, binary_stream)
    return binary_stream.getvalue()


# torch.device
def make_torch_device(**kwargs):
    torch_device = torch.device("cpu")
    return [
        (
            torch_device,
            (CODE[type(torch_device)], "cpu")
        ),
    ]


# torch.jit.ScriptModule
def make_torch_scriptmodule(**kwargs):
    class ScriptModule(torch.jit.ScriptModule):
        def __init__(self):
            super(ScriptModule, self).__init__()

        @torch.jit.script_method
        def forward(self, x):
            return x + 2

    sm = ScriptModule()
    return [
        (
            sm,
            (CODE[torch.jit.ScriptModule], sm.save_to_buffer()),
            None,
            compare_modules,
        ),
    ]


# torch.jit.TopLevelTracedModule
def make_torch_topleveltracedmodule(**kwargs):
    model = torch.nn.Sequential(torch.nn.Conv2d(1, 1, 3))
    tm = torch.jit.trace(model, torch.randn(1, 1, 3, 3))
    return [
        (
            tm,
            (CODE[torch.jit.TopLevelTracedModule], tm.save_to_buffer()),
            None,
            compare_modules,
        ),
    ]


# torch.nn.parameter.Parameter
def make_torch_parameter(**kwargs):
    param = torch.nn.Parameter(torch.randn(3, 3))
    return [
        (
            param,
            (CODE[torch.nn.Parameter], (param.id, serde._simplify(param.data), param.requires_grad, None)),
            None,
            lambda detailed, original: detailed.data.equal(original.data) and \
                                       detailed.id == original.id and \
                                       detailed.requires_grad == original.requires_grad
        ),
    ]


# torch.Tensor
def make_torch_tensor(**kwargs):
    tensor = torch.randn(3, 3)
    tensor.tags = ["tag1", "tag2"]
    tensor.description = "desc"
    return [
        (
            tensor,
            (CODE[torch.Tensor], (tensor.id, save_to_buffer(tensor), None, None, ["tag1", "tag2"], "desc")),
            None,
            lambda detailed, original: detailed.data.equal(original.data) and \
                                       detailed.id == original.id and \
                                       detailed.requires_grad == original.requires_grad
        ),
    ]


# torch.Size
def make_torch_size(**kwargs):
    return [
        (
            torch.randn(3, 3).size(),
            (CODE[torch.Size], (3, 3)),
        ),
    ]


########################################################################
# PySyft.
########################################################################

def comp(detailed, original):
    print(detailed)
    print(original)
    return False


# AdditiveSharingTensor
def make_additivesharingtensor(**kwargs):
    workers = kwargs["workers"]
    alice, bob, james = workers["alice"], workers["bob"], workers["james"]
    tensor = torch.tensor([[3.1, 4.3]]).fix_prec().share(alice, bob, crypto_provider=james)
    ast = tensor.child.child
    return [
        (
            ast,
            (
                CODE[syft.frameworks.torch.tensors.interpreters.additive_shared.AdditiveSharingTensor],
                (ast.id, ast.field, ast.crypto_provider.id, serde._simplify(ast.child))
            ),
            None,
            lambda detailed, original: detailed.id == original.id and \
                                       detailed.field == original.field and \
                                       detailed.child.keys() == original.child.keys()
        )
    ]


# FixedPrecisionTensor
def make_fixedprecisiontensor(**kwargs):
    workers = kwargs["workers"]
    alice, bob, james = workers["alice"], workers["bob"], workers["james"]
    t = torch.tensor([[3.1, 4.3]])
    t.tags = ["tag1", "tag2"]
    t.description = "desc"
    fpt = t.fix_prec(base=12, precision_fractional=5).share(alice, bob, crypto_provider=james)
    return [
        (
            fpt,
            (
                CODE[syft.frameworks.torch.tensors.interpreters.precision.FixedPrecisionTensor],
                (
                    fpt.id,
                    fpt.field,
                    fpt.base,
                    fpt.precision_fractional,
                    fpt.kappa,
                    (CODE[list], ((CODE[str], (b"tag1",)), (CODE[str], (b"tag2",)))),
                    (CODE[str], (b"desc",)),
                    None
                )
            ),
            None,
            lambda detailed, original: detailed.child.id == original.child.id and \
                                       detailed.child.field == original.child.field and \
                                       detailed.child.base == original.child.base and \
                                       detailed.child.precision_fractional == original.child.precision_fractional and \
                                       detailed.child.kappa == original.child.kappa
        )
    ]


# Dictionary containing test samples functions
samples = OrderedDict()

# Native
samples[float] = make_float
samples[int] = make_int
samples[dict] = make_dict
samples[tuple] = make_tuple
samples[list] = make_list
samples[set] = make_set
samples[slice] = make_slice
samples[str] = make_str
samples[range] = make_range
samples[type(Ellipsis)] = make_ellipsis

# Numpy
samples[numpy.ndarray] = make_numpy_ndarray

# PyTorch
samples[torch.device] = make_torch_device
samples[torch.jit.ScriptModule] = make_torch_scriptmodule
samples[torch.jit.TopLevelTracedModule] = make_torch_topleveltracedmodule
samples[torch.nn.Parameter] = make_torch_parameter
samples[torch.jit.TopLevelTracedModule] = make_torch_topleveltracedmodule
samples[torch.Tensor] = make_torch_tensor
samples[torch.Size] = make_torch_size

# PySyft
samples[syft.frameworks.torch.tensors.interpreters.additive_shared.AdditiveSharingTensor] = make_additivesharingtensor
samples[syft.frameworks.torch.tensors.interpreters.precision.FixedPrecisionTensor] = make_additivesharingtensor


def test_serde_coverage():
    """Checks all types in serde are tested"""
    for cls, _ in serde.simplifiers.items():
        has_sample = cls in samples
        assert has_sample is True, "Serde for %s is not tested" % cls


@pytest.mark.parametrize("cls", samples)
def test_serde_roundtrip(cls, workers):
    """Checks that values passed through serialization-deserialization stay same"""
    _samples = samples[cls](workers=workers)
    for obj, *params in _samples:
        simplified_obj = serde._simplify(obj)
        detailed_obj = serde._detail(syft.hook.local_worker, simplified_obj)
        if len(params) >= 3 and params[2] is not None:
            # Custom detailed objects comparison function.
            comp_func = params[2]
            assert comp_func(detailed_obj, obj) is True
        else:
            assert type(detailed_obj) == type(obj)
            assert detailed_obj == obj


@pytest.mark.parametrize("cls", samples)
def test_serde_simplify(cls, workers):
    """Checks that simplified structures match expected"""
    _samples = samples[cls](workers=workers)
    for sample in _samples:
        obj, expected_simplified_obj = sample[0], sample[1]
        simplified_obj = serde._simplify(obj)

        if len(sample) >= 3 and sample[2] is not None:
            # Custom simplified objects comparison function.
            comp_func = sample[2]
            assert comp_func(simplified_obj, expected_simplified_obj) is True
        else:
            assert simplified_obj == expected_simplified_obj
