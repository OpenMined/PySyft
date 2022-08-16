# stdlib
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Type
from typing import Union

# relative
from ...adp.data_subject_ledger import DataSubjectLedger
from ...common.serde.serializable import serializable
from ..autodp.gamma_tensor import GammaTensor
from ..autodp.phi_tensor import PhiTensor
from ..tensor import Tensor
from .layers.base import Layer
from .loss import BinaryCrossEntropy
from .loss import Loss
from .optimizers import Adamax
from .optimizers import Optimizer


@serializable(recursive_serde=True)
class Model:
    __attr_allowlist__ = [
        "layers",
        "loss",
        "optimizer",
        "aggregated_loss",
    ]

    def __init__(self, layers: Optional[List[Type[Layer]]] = None):
        self.layers: List = [] if layers is None else layers
        self.aggregated_loss: float = 0.0

    def publish(
        self,
        deduct_epsilon_for_user: Callable,
        get_budget_for_user: Callable,
        ledger: DataSubjectLedger,
        sigma: float,
    ) -> Dict:
        print("Publish Model Weights")

        parameters = {}
        for i, layer in enumerate(self.layers):
            print("Layer", str(layer))

            print("Before  Publish")
            for param in layer.params:  # type: ignore
                print(param.shape, end=" ")
            print()
            if hasattr(layer, "params"):
                parameters[str(layer) + str(i)] = [
                    param.publish(
                        deduct_epsilon_for_user=deduct_epsilon_for_user,
                        get_budget_for_user=get_budget_for_user,
                        ledger=ledger,
                        sigma=sigma,
                    )
                    if isinstance(param, (GammaTensor))
                    else param
                    for param in layer.params  # type: ignore
                ]
                print("After  Publish")
                for param in parameters[str(layer) + str(i)]:
                    print(param.shape, end=" ")
                print()

        parameters["loss"] = self.aggregated_loss  # type: ignore

        return parameters

    def replace_weights(self, published_weights: dict) -> None:
        """
        For when you want to use published weights downloaded from a domain
        """
        for i, layer in enumerate(self.layers):
            params = published_weights[str(layer) + str(i)]
            if len(params) > 0:
                layer.params = params  # type: ignore

    def add(self, layer: Layer) -> None:
        if isinstance(layer, Layer):
            self.layers.append(layer)
        else:
            raise TypeError("PySyft doesn't recognize this kind of layer.")

    def initialize_weights(
        self,
        loss: Loss = BinaryCrossEntropy(),
        optimizer: Optimizer = Adamax(),
    ) -> None:
        self.layers[0].first_layer = True

        prev_layer = None
        for _, layer in enumerate(self.layers):
            layer.connect_to(prev_layer=prev_layer)
            prev_layer = layer

        self.loss = loss
        self.optimizer = optimizer

    def fit(
        self,
        X: Union[GammaTensor, PhiTensor, Tensor],
        Y: Union[GammaTensor, PhiTensor, Tensor],
        max_iter: int = 100,
        batch_size: int = 4,
        shuffle: bool = True,
        validation_split: float = 0.0,
        validation_data: Optional[Tuple] = None,
    ) -> None:
        print("Started Training")

        if isinstance(X, Tensor):
            X = X.child
        if isinstance(Y, Tensor):
            Y = Y.child

        # prepare data
        train_X = X  # .astype(get_dtype()) if np.issubdtype(np.float64, X.dtype) else X
        train_Y = Y  # .astype(get_dtype()) if np.issubdtype(np.float64, Y.dtype) else Y

        if 1.0 > validation_split > 0.0:
            split = int(train_Y.shape[0] * validation_split)
            valid_X, valid_Y = train_X[-split:], train_Y[-split:]
            train_X, train_Y = train_X[:-split], train_Y[:-split]
        elif validation_data is not None:
            valid_X, valid_Y = validation_data
        else:
            valid_X, valid_Y = None, None

        iter_idx = 0
        while iter_idx < max_iter:
            iter_idx += 1

            # train
            train_losses = []
            # train_predicts, train_targets =  [], []
            for b in range(train_Y.shape[0] // batch_size):
                batch_begin = b * batch_size
                batch_end = batch_begin + batch_size
                x_batch = train_X[batch_begin:batch_end]
                y_batch = train_Y[batch_begin:batch_end]

                # forward propagation
                y_pred = self.predict(x_batch)

                # backward propagation
                next_grad = self.loss.backward(y_pred, y_batch)
                for layer in self.layers[::-1]:
                    print("Backward layer", str(layer))
                    next_grad = layer.backward(next_grad)

                # update parameters
                self.optimizer.update(self.layers)

                # got loss and predict
                train_losses.append(self.loss.forward(y_pred, y_batch))

            if valid_X is not None and valid_Y is not None:
                valid_losses, valid_predicts, valid_targets = [], [], []  # type: ignore

                for b in range(valid_X.shape[0] // batch_size):
                    batch_begin = b * batch_size
                    batch_end = batch_begin + batch_size
                    x_val_batch = valid_X[batch_begin:batch_end]
                    y_val_batch = valid_Y[batch_begin:batch_end]

                    # forward propagation
                    y_val_pred = self.predict(x_val_batch)

                    # got loss and predict
                    valid_losses.append(self.loss.forward(y_val_pred, y_val_batch))
                    valid_predicts.extend(y_val_pred.child)
                    valid_targets.extend(y_val_batch.child)

    def step(
        self,
        x_batch: Union[GammaTensor, PhiTensor, Tensor],
        y_batch: Union[GammaTensor, PhiTensor, Tensor],
    ) -> None:

        x_batch = x_batch.child if isinstance(x_batch, Tensor) else x_batch
        y_batch = y_batch.child if isinstance(y_batch, Tensor) else y_batch

        print(type(x_batch), type(y_batch))
        print(x_batch.shape, y_batch.shape)

        # forward propagation
        y_pred = self.predict(x_batch)

        print("Predictions:", y_pred.child.argmax())
        print("Actual Value:", y_batch.child.shape)

        # backward propagation
        next_grad = self.loss.backward(y_pred, y_batch)
        for layer in self.layers[::-1]:
            print("Backward layer", str(layer))
            next_grad = layer.backward(next_grad)

        # update parameters
        print("Updating optimizer")
        self.optimizer.update(self.layers)

        # got loss and predict
        print("Predicting loss")
        loss = self.loss.forward(y_pred, y_batch)

        curr_loss = float(loss.child)
        self.aggregated_loss += curr_loss

        print("Loss:", self.aggregated_loss)

    def predict(
        self, X: Union[GammaTensor, PhiTensor, Tensor]
    ) -> Union[GammaTensor, PhiTensor, Tensor]:
        """Calculate an output Y for the given input X."""
        x_next = X
        for layer in self.layers:
            print("Forward layer", str(layer))
            x_next = layer.forward(x_next)
        y_pred = x_next
        return y_pred
