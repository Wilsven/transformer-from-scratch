import torch
import torch.nn as nn


class PositionwiseFeedForwardBlock(nn.Module):
    """Fully-connected feed forward neural network used in the Encoder and Decoder.

    Attributes:
        linear_1 (nn.Linear): nn.Linear layer.
        relu (nn.ReLU): nn.ReLU activation function.
        dropout (nn.Dropout): nn.Dropout layer.
        linear_2 (nn.Linear): nn.Linear layer.
    """

    def __init__(self, d_model: int, d_ff: int, drop_prob: float):
        """
        Args:
            d_model (int): The number of model dimensions.
            d_ff (int): The number of dimensions for hidden layer
            drop_prob (float): Probability for nn.Dropout layer.
        """
        super().__init__()

        self.linear_1 = nn.Linear(d_model, d_ff, bias=True)  # W1 and b1
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)
        self.linear_2 = nn.Linear(d_ff, d_model, bias=True)  # W2 and b2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Shape: (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_ff) -> (batch_size, seq_len, d_model)
        return self.linear_2(self.dropout(self.relu(self.linear_1(x))))
