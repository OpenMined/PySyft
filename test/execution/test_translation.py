import pytest
import torch as th
import torch.nn as nn
import torch.nn.functional as F

import syft as sy
from syft.execution.placeholder import PlaceHolder
from syft.execution.plan import Plan
from syft.execution.translation import TranslationTarget
from syft.execution.translation.torchscript import PlanTranslatorTorchscript
from syft.execution.translation.threepio import PlanTranslatorTfjs
from syft.serde.serde import deserialize
from syft.serde.serde import serialize


@pytest.fixture(scope="function", autouse=True)
def mnist_example():
    """
    Prepares simple model-centric federated learning training plan example.
    """
    # Disable translators
    Plan._build_translators = []

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(784, 392)
            self.fc2 = nn.Linear(392, 10)

        def forward(self, x):
            x = self.fc1(x)
            x = F.relu(x)
            x = self.fc2(x)
            return x

    model = Net()

    def set_model_params(module, params_list, start_param_idx=0):
        """Set params list into model recursively"""
        param_idx = start_param_idx

        for name, param in module._parameters.items():
            module._parameters[name] = params_list[param_idx]
            param_idx += 1

        for name, child in module._modules.items():
            if child is not None:
                param_idx = set_model_params(child, params_list, param_idx)

        return param_idx

    def softmax_cross_entropy_with_logits(logits, targets, batch_size):
        """Calculates softmax entropy
        Args:
            * logits: (NxC) outputs of dense layer
            * targets: (NxC) one-hot encoded labels
            * batch_size: value of N, temporarily required because Plan cannot trace .shape
        """
        # numstable logsoftmax
        norm_logits = logits - logits.max()
        log_probs = norm_logits - norm_logits.exp().sum(dim=1, keepdim=True).log()
        # reduction = mean
        return -(targets * log_probs).sum() / batch_size

    def naive_sgd(param, **kwargs):
        return param - kwargs["lr"] * param.grad

    @sy.func2plan()
    def train(data, targets, lr, batch_size, model_parameters):
        # load model params
        set_model_params(model, model_parameters)

        # forward
        logits = model(data)

        # loss
        loss = softmax_cross_entropy_with_logits(logits, targets, batch_size)

        # backward
        loss.backward()

        # step
        updated_params = [naive_sgd(param, lr=lr) for param in model_parameters]

        # accuracy
        pred = th.argmax(logits, dim=1)
        targets_idx = th.argmax(targets, dim=1)
        acc = pred.eq(targets_idx).sum().float() / batch_size

        return (loss, acc, *updated_params)

    # Dummy inputs
    data = th.randn(3, 28 * 28)
    target = F.one_hot(th.tensor([1, 2, 3]), 10)
    lr = th.tensor([0.01])
    batch_size = th.tensor([3.0])
    model_state = list(model.parameters())

    # Build Plan
    train.build(data, target, lr, batch_size, model_state, trace_autograd=True)

    return train, data, target, lr, batch_size, model_state


def test_plan_can_be_jit_traced(hook, workers):
    args_shape = [(1,)]

    @sy.func2plan(args_shape=args_shape, state=(th.tensor([1.0]),))
    def foo(x, state):
        (bias,) = state.read()
        x = x * 2
        return x + bias

    assert isinstance(foo.__str__(), str)
    assert len(foo.actions) > 0
    assert foo.is_built

    t = th.tensor([1.0, 2])
    x = foo(t)

    assert (x == th.tensor([3.0, 5])).all()

    args = PlaceHolder.create_placeholders(args_shape)
    torchscript_plan = th.jit.trace(foo, args)

    y = torchscript_plan(t)

    assert (y == th.tensor([3.0, 5])).all()


def test_func_plan_can_be_translated_to_tfjs(hook, workers):
    Plan._build_translators = []

    @sy.func2plan(args_shape=[(3, 3)])
    def plan(x):
        x = x * 2
        x = x.abs()
        return x

    orig_plan = plan.copy()

    plan_js = plan.copy()
    plan_js.add_translation(PlanTranslatorTfjs)
    plan_js.base_framework = TranslationTarget.TENSORFLOW_JS.value
    assert plan_js.role.actions[0].name == "tf.mul"
    assert len(plan_js.role.actions[0].args) == 2
    assert len(plan_js.role.input_placeholders()) == len(orig_plan.role.input_placeholders())
    assert len(plan_js.role.output_placeholders()) == len(orig_plan.role.output_placeholders())

    # Test plan caching
    plan_js2 = plan_js.copy()
    plan_js2.add_translation(PlanTranslatorTfjs)
    plan_js2.base_framework = TranslationTarget.TENSORFLOW_JS.value
    assert plan_js2.role.actions[0].name == "tf.mul"
    assert len(plan_js2.role.actions[0].args) == 2

    # check that translation can be done after serde
    serde_plan = deserialize(serialize(orig_plan))
    serde_plan.add_translation(PlanTranslatorTfjs)
    serde_plan.base_framework = TranslationTarget.TENSORFLOW_JS.value
    assert serde_plan.role.actions[0].name == "tf.mul"
    assert len(serde_plan.role.actions[0].args) == 2

    # check that translation is not lost after serde
    serde_plan_full = deserialize(serialize(plan_js))
    assert serde_plan_full.role.actions[0].name == "tf.mul"
    assert len(serde_plan_full.role.actions[0].args) == 2


def test_cls_plan_can_be_translated_to_tfjs(hook, workers):
    Plan._build_translators = []

    class Net(sy.Plan):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(2, 3)
            self.fc2 = nn.Linear(3, 1)

        def forward(self, x):
            x = self.fc1(x)
            x = F.relu(x)
            x = self.fc2(x)
            return x

    plan = Net()
    plan.build(th.zeros(10, 2))
    js_plan = plan.copy()
    js_plan.add_translation(PlanTranslatorTfjs)
    js_plan.base_framework = TranslationTarget.TENSORFLOW_JS.value

    assert js_plan.role.actions[0].name == "tf.transpose"
    assert js_plan.role.actions[1].name == "tf.matMul"
    assert js_plan.role.actions[2].name == "tf.relu"
    assert js_plan.role.actions[3].name == "tf.transpose"
    assert js_plan.role.actions[4].name == "tf.matMul"


def test_func_plan_can_be_translated_to_torchscript(hook, workers):
    # Disable build time auto translation
    Plan._build_translators = []

    @sy.func2plan(args_shape=[(3, 3)])
    def plan(x):
        x = x * 2
        x = x.abs()
        return x

    orig_plan = plan.copy()

    inp = th.tensor([1, -1, 2])
    res1 = plan(inp)
    plan.add_translation(PlanTranslatorTorchscript)
    res2 = plan.torchscript(inp)
    assert (res1 == res2).all()

    # check that translation can be done after serde
    serde_plan = deserialize(serialize(orig_plan))
    serde_plan.add_translation(PlanTranslatorTorchscript)
    res3 = serde_plan.torchscript(inp)
    assert (res1 == res3).all()

    # check that translation is not lost after serde
    serde_plan_full = deserialize(serialize(plan))
    res4 = serde_plan_full.torchscript(inp)
    assert (res1 == res4).all()


def test_cls_plan_can_be_translated_to_torchscript(hook, workers):
    # Disable build time auto translation
    Plan._build_translators = []

    class Net(sy.Plan):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(2, 3)
            self.fc2 = nn.Linear(3, 1)

        def forward(self, x):
            x = self.fc1(x)
            x = F.relu(x)
            x = self.fc2(x)
            return x

    plan = Net()
    plan.build(th.zeros(10, 2))
    orig_plan = plan.copy()

    inp = th.randn(10, 2)

    res1 = plan(inp)
    plan.add_translation(PlanTranslatorTorchscript)
    res2 = plan.torchscript(inp, plan.parameters())
    assert (res1 == res2).all()

    # check that translation can be done after serde
    serde_plan = deserialize(serialize(orig_plan))
    serde_plan.add_translation(PlanTranslatorTorchscript)
    res3 = serde_plan.torchscript(inp, serde_plan.parameters())
    assert (res1 == res3).all()

    # check that translation is not lost after serde
    serde_plan_full = deserialize(serialize(plan))
    res4 = serde_plan_full.torchscript(inp, serde_plan_full.parameters())
    assert (res1 == res4).all()


def test_plan_translation_remove(hook, workers):
    # Disable build time auto translation
    Plan._build_translators = []

    @sy.func2plan(args_shape=[(3, 3)])
    def plan(x):
        x = x * 2
        x = x.abs()
        return x

    plan.add_translation(PlanTranslatorTorchscript)

    full_plan = plan.copy()
    assert full_plan.torchscript is not None

    assert plan.torchscript is not None
    assert len(plan.role.actions) > 0

    plan.remove_translation()
    assert plan.torchscript is not None
    assert len(plan.role.actions) == 0

    plan.remove_translation(PlanTranslatorTorchscript)
    assert plan.torchscript is None
    assert len(plan.role.actions) == 0

    full_plan.remove_translation(PlanTranslatorTorchscript)
    assert full_plan.torchscript is None
    assert len(full_plan.role.actions) > 0


def test_plan_translated_on_build(hook, workers):
    # Enable torchscript translator
    Plan.register_build_translator(PlanTranslatorTorchscript)

    @sy.func2plan(args_shape=[(3, 3)])
    def plan(x):
        x = x * 2
        x = x.abs()
        return x

    inp = th.tensor([1, -1, 2])
    res1 = plan(inp)
    res2 = plan.torchscript(inp)
    assert (res1 == res2).all()


def test_backward_autograd_can_be_translated(hook, workers):
    @sy.func2plan(args_shape=[(5, 5)], trace_autograd=True)
    def autograd_test(X):
        y = X * 5
        y = -y.log() / 2
        y = y.sum()
        y.backward()
        return X.grad

    X = th.ones(5, 5, requires_grad=True)

    # Result of torch autograd
    torch_grads = autograd_test(X)

    # Translate to torchscript
    autograd_test.add_translation(PlanTranslatorTorchscript)

    # Result of torchscript'ed traced backprop
    ts_plan_grads = autograd_test.torchscript(X)

    # (debug out)
    print("Torchscript Plan:\n", autograd_test.torchscript.code)

    # Test all results are equal
    assert torch_grads.eq(ts_plan_grads).all()


def test_fl_mnist_example_training_can_be_translated_torchscript(hook, workers, mnist_example):
    train, data, target, lr, batch_size, model_state = mnist_example

    # Execute with original forward function (native torch autograd)
    res_torch = train(data, target, lr, batch_size, model_state)

    # Execute traced operations (traced autograd)
    train.forward = None
    res_syft_traced = train(data, target, lr, batch_size, model_state)

    # Translate syft Plan to torchscript and execute it
    train.add_translation(PlanTranslatorTorchscript)
    res_torchscript = train.torchscript(data, target, lr, batch_size, model_state)

    # (debug out)
    print(train.torchscript.code)

    # All variants should be equal
    for i, out in enumerate(res_torch):
        assert th.allclose(out, res_syft_traced[i])
        assert th.allclose(out, res_torchscript[i])


def test_fl_mnist_example_training_can_be_translated_tfjs(hook, workers, mnist_example):
    train, data, target, lr, batch_size, model_state = mnist_example

    # Translate syft Plan to tfjs
    train.add_translation(PlanTranslatorTfjs)
    train.base_framework = TranslationTarget.TENSORFLOW_JS.value

    # (debug out)
    print(train.code)
