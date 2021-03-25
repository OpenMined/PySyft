# third party
import torch as th

# syft absolute
import syft as sy
from syft import make_plan
from syft.core.plan.plan_builder import ROOT_CLIENT
from syft.core.plan.translation.torchscript.plan_translate import translate


def test_mnist_demo_plan_translation() -> None:
    class MLP(sy.Module):
        """
        Simple model with method for loss and hand-written backprop.
        """

        def __init__(self, torch_ref=th):  # type: ignore
            super(MLP, self).__init__(torch_ref=torch_ref)
            self.fc1 = torch_ref.nn.Linear(10, 3)
            self.relu = torch_ref.nn.ReLU()
            self.fc2 = torch_ref.nn.Linear(3, 3)

        def forward(self, x):  # type: ignore
            self.z1 = self.fc1(x)
            self.a1 = self.relu(self.z1)
            return self.fc2(self.a1)

        def backward(self, X, error):  # type: ignore
            z1_grad = (error @ self.fc2.state_dict()["weight"]) * (self.a1 > 0).float()
            fc1_weight_grad = z1_grad.t() @ X
            fc1_bias_grad = z1_grad.sum(0)
            fc2_weight_grad = error.t() @ self.a1
            fc2_bias_grad = error.sum(0)
            return fc1_weight_grad, fc1_bias_grad, fc2_weight_grad, fc2_bias_grad

        def softmax_cross_entropy_with_logits(self, logits, target, batch_size):  # type: ignore
            probs = self.torch_ref.softmax(logits, dim=1)
            loss = -(target * self.torch_ref.log(probs)).mean()
            loss_grad = (probs - target) / batch_size
            return loss, loss_grad

        def accuracy(self, logits, targets, batch_size):  # type: ignore
            pred = self.torch_ref.argmax(logits, dim=1)
            targets_idx = self.torch_ref.argmax(targets, dim=1)
            acc = pred.eq(targets_idx).sum().float() / batch_size
            return acc

    model = MLP(torch_ref=th)  # type: ignore

    # Dummy inputs
    bs = 3
    classes_num = 3
    data = th.randn(bs, 10)
    target = th.nn.functional.one_hot(th.randint(0, classes_num, [bs]), classes_num)
    lr = th.tensor([0.1])
    batch_size = th.tensor([bs])
    model_state_zeros = sy.lib.python.collections.OrderedDict(
        {k: v * 0 for k, v in model.state_dict().items()}
    )
    model_state = sy.lib.python.collections.OrderedDict(model.state_dict())

    @make_plan
    def training_plan(  # type: ignore
        data=data,
        targets=target,
        lr=lr,
        batch_size=batch_size,
        model_state=model_state_zeros,
    ):
        # send the model to plan builder (but not its default params)
        model_ptr = model.send(ROOT_CLIENT, send_parameters=False)

        # set model params from input
        model_ptr.__dict__["real_module"].load_state_dict(model_state)

        # forward
        logits = model_ptr(data)

        # loss
        loss, loss_grad = model_ptr.softmax_cross_entropy_with_logits(
            logits, targets, batch_size
        )

        # backward
        grads = model_ptr.backward(data, loss_grad)

        # SGD step
        updated_params = tuple(
            param - lr * grad for param, grad in zip(model_ptr.parameters(), grads)
        )

        # accuracy
        acc = model_ptr.accuracy(logits, targets, batch_size)

        # return things
        return (loss, acc, *updated_params)

    # Translate to torchscript
    ts_plan = translate(training_plan)

    # debug out
    print(ts_plan.torchscript.code)

    # Execute translated plan
    ts_res = ts_plan(data, target, lr, batch_size, model_state)

    # Check that the plan also works as usual
    vm = sy.VirtualMachine()
    client = vm.get_client()

    plan_ptr = training_plan.send(client)
    res_ptr = plan_ptr(
        data=data, targets=target, lr=lr, batch_size=batch_size, model_state=model_state
    )
    res = res_ptr.get()

    # Compare outputs
    for ts_out, out in zip(ts_res, res):
        assert th.equal(ts_out, out)
