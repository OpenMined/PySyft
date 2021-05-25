# stdlib
from collections import Counter
from collections import defaultdict
import uuid

# syft relative
from . import ops
from ..passthrough import PassthroughTensor


class AutogradTensor(PassthroughTensor):

    def __init__(self, child, requires_grad=False):
        super().__init__(child)

        # whether to run backpropagation or not
        self.requires_grad = requires_grad

        # tensor gradient
        self._grad = defaultdict(lambda: None)

        # operation used to create this tensor (if any)
        self._grad_fn = None

        # list of ops which use this tensor
        self.ops = list()

        self.backprop_id = None

        self.n_backwards = Counter()

    @property
    def grad(self):
        if self.backprop_id not in self._grad:
            return None
        return self._grad[self.backprop_id]

    @property
    def grad_fn(self):
        if not self.requires_grad:
            raise Exception('This tensor is not backpropagated')
        return self._grad_fn

    #     def __ge__(self, other):
    #         return AutogradTensor(self.child >= other.child, requires_grad=False)

    #     def __le__(self, other):
    #         return AutogradTensor(self.child <= other.child, requires_grad=False)

    def __add__(self, other):

        op = ops.add.AddOp()
        return op(self, other)

    def __sub__(self, other):
        op = ops.sub.SubOp()
        return op(self, other)

    def __mul__(self, other):
        op = ops.mul.MulOp()
        return op(self, other)

    def reshape(self, *shape):
        op = ops.reshape.ReshapeOp()
        return op(self, *shape)

    def copy(self):
        op = ops.copy.CopyOp()
        return op(self)

    def sum(self, *args, **kwargs):
        op = ops.sum.SumOp()
        return op(self, *args, **kwargs)

    def repeat(self, *args, **kwargs):
        op = ops.repeat.RepeatOp()
        return op(self, *args, *kwargs)

    def transpose(self, *dims):
        op = ops.transpose.TransposeOp()
        return op(self, *dims)

    def add_grad(self, grad):

        print("Adding grad:" + str(type(grad)) + " w/ backprop_id:" + str(self.backprop_id))

        if self._grad[self.backprop_id] is None:
            self._grad[self.backprop_id] = grad
        else:
            self._grad[self.backprop_id] = grad + self._grad[self.backprop_id]

    def backward(self, grad=None, backprop_id=None):

        if backprop_id is None:
            backprop_id = uuid.uuid4()

        self.n_backwards[backprop_id] += 1

        self.backprop_id = backprop_id

        print("called backward on:" + str(self.grad_fn) + " " + str(self.n_backwards) + " " + str(self.ops))

        if not self.grad_fn:
            return False

        if grad is None and self._grad[self.backprop_id] is None:
            # in case if this is last loss tensor
            grad = np.ones(self.shape)
            grad = self.__class__(grad, requires_grad=False)

        elif self.grad is not None:
            grad = self._grad[self.backprop_id]

        if not self.requires_grad:
            raise Exception('This tensor is not backpropagated')

        print(self.n_backwards[backprop_id], len(self.ops))

        # if all gradients are accounted for - backprop
        if self.n_backwards[backprop_id] >= len(self.ops):

            self.grad_fn.backward(grad, backprop_id=backprop_id)

        # if some gradietns appear to be missing - parse forward in
        # the graph to double check
        else:

            # investigate whether any of the missing ops are actually
            # going to get used.
            found_id = False

            n_direct_ops = 0
            for op in self.ops:
                if op.backprop_id is not None and op.backprop_id == backprop_id:
                    n_direct_ops += 1

            # if the number of operations we know will be backpropagating gradietns to us
            # exceeds the number of times we've been backpropgated into - then we know
            # we need to wait.
            if n_direct_ops > self.n_backwards[backprop_id]:
                print(str(n_direct_ops) + " > " + str(self.n_backwards[backprop_id]))
                print(len(self.ops))
                found_id = True

            else:

                for op in self.ops:
                    if op.backprop_id is None:
                        if op.out.find_backprop_id(self.backprop_id):
                            found_id = True
                            break

            print("Found id:" + str(found_id))
            if found_id:
                "do nothing - we're going to get another gradient"
            else:
                # backprop anyway - we've got all the grads we're gonna get
                self.grad_fn.backward(grad, backprop_id=backprop_id)

        return True

    def find_backprop_id(self, backprop_id):
        found_id = False

        for op in self.ops:
            if op.backprop_id is not None and op.backprop_id == backprop_id:
                return True

            if op.out.find_backprop_id(self.backprop_id):
                found_id = True
                break

        return found_id
