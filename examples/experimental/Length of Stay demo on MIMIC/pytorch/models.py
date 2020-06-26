"""Neural network architecture for Lengh of Stay demo on MIMIC.

Class implementing the network called FFN in the paper:
Purushotham, Sanjay & Meng, Chuizheng & Che, Zhengping & Liu, Yan. (2017).
Benchmark of Deep Learning Models on Large Healthcare MIMIC Datasets.
Journal of Biomedical Informatics. 83. 10.1016/j.jbi.2018.04.007.
Available on :
https://www.sciencedirect.com/science/article/pii/S1532046418300716
"""
import torch.nn as nn

ACTIVATION_FUNCTIONS = ["relu", "sigmoid", "linear"]


class FeedforwardNeuralNetwork(nn.Module):
    """Neural Network Architecture called FFN in the paper:
    Purushotham, Sanjay & Meng, Chuizheng & Che, Zhengping & Liu, Yan. (2017).
    Benchmark of Deep Learning Models on Large Healthcare MIMIC Datasets.
    Journal of Biomedical Informatics. 83. 10.1016/j.jbi.2018.04.007.
    Available on :
    https://www.sciencedirect.com/science/article/pii/S1532046418300716

    """

    def __init__(
        self,
        n_input: int,
        hidden_dim: int = None,
        ffn_depth: int = 2,
        final_activation: str = "linear",
    ):
        """
        Args:
            n_input (int): dimension of an example, i.e. the number of feature
            hidden_dim (int, optional): hidden layer size. Defaults to None.
            ffn_depth (int, optional): number of hidden layers. Defaults to 1.
            final_activation (str, optional): Name designating the activation
                function of the last layer. The possible choices are listed in
                ACTIVATION_FUNCTIONS. Defaults to "linear".
        """
        super(FeedforwardNeuralNetwork, self).__init__()
        if not hidden_dim:
            hidden_dim = 8 * n_input
        self.input_layer = nn.Sequential(nn.Linear(n_input, hidden_dim), nn.Sigmoid())
        self.hidden_layers = nn.ModuleList()
        for _ in range(ffn_depth - 1):
            self.hidden_layers.append(
                nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
            )
        assert final_activation.lower() in ACTIVATION_FUNCTIONS, (
            f"final activatation function {final_activation} isn't correct."
            f"It shoud be one of them: {ACTIVATION_FUNCTIONS}"
        )
        if final_activation.lower() == "relu":
            self.output_layer = nn.Sequential(nn.Linear(hidden_dim, 1), nn.ReLU(),)
        elif final_activation.lower() == "sigmoid":
            self.output_layer = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid(),)
        elif final_activation.lower() == "linear":
            self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.input_layer(x)
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.output_layer(x)
        return x
