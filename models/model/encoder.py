import torch
import torch.nn as nn

from models.blocks.encoder_layers import EncoderBlock
from models.layers.layer_norm import LayerNormalization


class Encoder(nn.Module):
    """Encoder in the Transformer architecture.

    Attributes:
        layers (nn.ModuleList): Consists of EncoderBlock applied N times.
        norm (LayerNormalization): LayerNormalization block.
    """

    def __init__(self, d_model: int, layers: nn.ModuleList[EncoderBlock]):
        """
        Args:
            d_model (int): The number of model dimensions.
            layers (nn.ModuleList[EncoderBlock]): Consists of EncoderBlock applied N times.
        """
        super().__init__()

        self.layers = layers
        self.norm = LayerNormalization(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Applies the EncoderBlock N times.

        Args:
            x (torch.Tensor): Input tensor to the encoder.
            mask (torch.Tensor): Mask to mask padding words.

        Returns:
            torch.Tensor: Normalized output tensor from the encoder after N passes.
        """
        for layer in self.layers:
            x = layer(x, mask)

        return self.norm(x)
