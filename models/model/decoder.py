import torch
import torch.nn as nn

from models.blocks.decoder_layer import DecoderBlock
from models.layers.layer_norm import LayerNormalization


class Decoder(nn.Module):
    """Decoder in the Transformer architecture.

    Attributes:
        layers (nn.ModuleList): Consists of DecoderBlock applied N times.
        norm (LayerNormalization): LayerNormalization block.
    """

    def __init__(self, d_model: int, layers: nn.ModuleList[DecoderBlock]):
        """
        Args:
            d_model (int): The number of model dimensions.
            layers (nn.ModuleList[DecoderBlock]): Consists of DecoderBlock applied N times.
        """
        super().__init__()

        self.layers = layers
        self.norm = LayerNormalization(d_model)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Applies the DecoderBlock N times.

        _extended_summary_

        Args:
            x (torch.Tensor): Input tensor to the decoder.
            encoder_output (torch.Tensor): Output tensor from the encoder.
            src_mask (torch.Tensor): Mask applied in the encoder.
            tgt_mask (torch.Tensor): Mask applied in the decoder.

        Returns:
            torch.Tensor: Normalized output tensor from the decoder after N passes.
        """

        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)

        return self.norm(x)
