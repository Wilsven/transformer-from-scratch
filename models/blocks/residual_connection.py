import torch
import torch.nn as nn

from models.layers.layer_norm import LayerNormalization


class ResidualConnection(nn.Module):
    """Performs residual connections in the Transformer architecture.

    Attributes:
        dropout (nn.Dropout): nn.Dropout layer.
        norm (LayerNormalization): LayerNormalization block.
    """

    def __init__(self, d_model: int, drop_prob: float):
        """
        Args:
            d_model (int): The number of model dimensions.
            drop_prob (float): Probability for nn.Dropout layer.
        """
        super().__init__()

        self.dropout = nn.Dropout(p=drop_prob)
        self.norm = LayerNormalization(d_model)

    def forward(self, x: torch.Tensor, sublayer) -> torch.Tensor:
        """Performs residual connections.

        Args:
            x (torch.Tensor): Input tensor.
            sublayer: The sublayer the input tensor will be passed to and the initial input
                        will be added to the subsequent output.

        Returns:
            torch.Tensor: The output tensor after skip connection.
        """
        # There is a slight difference that we first apply the normalization first then the sublayer
        # In the original transformer paper, they applied the sublayer first before the normalization
        # return x + self.dropout(self.norm(sublayer(x)))
        return x + self.dropout(sublayer(self.norm(x)))
