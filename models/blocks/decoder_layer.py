import torch
import torch.nn as nn

from models.blocks.residual_connection import ResidualConnection
from models.layers.multi_head_attention import MultiHeadAttentionBlock
from models.layers.position_wise_feed_forward import \
    PositionwiseFeedForwardBlock


class DecoderBlock(nn.Module):
    """A Decoder Block.

    Attributes:
        self_attention_block (MultiHeadAttentionBlock): Self-attention block over multiple attention heads.
        cross_attention_block (MultiHeadAttentionBlock): Cross-attention taking in queries from the decoder,
                                                        keys and values from the encoder.
        feed_forward_block (PositionwiseFeedForwardBlock): Position-wise feed forward block in the decoder.
        residual_connections (nn.ModuleList[ResidualConnection]): Decoder has three residual connections.
    """

    def __init__(
        self,
        self_attention_block: MultiHeadAttentionBlock,
        cross_attention_block: MultiHeadAttentionBlock,
        feed_forward_block: PositionwiseFeedForwardBlock,
        drop_prob: float,
    ):
        """
        Args:
            self_attention_block (MultiHeadAttentionBlock): Self-attention block over multiple attention heads.
            cross_attention_block (MultiHeadAttentionBlock): Cross-attention taking in queries from the decoder,
                                                            keys and values from the encoder.
            feed_forward_block (PositionwiseFeedForwardBlock): Position-wise feed forward block in decoder.
            drop_prob (float): Probability for nn.Dropout layer.
        """
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList(
            [ResidualConnection(drop_prob) for _ in range(3)]
        )

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ):
        """One forward pass through the decoder. This can be applied N times.

        Args:
            x (torch.Tensor): Input tensor to the decoder.
            encoder_output (torch.Tensor): Output from the encoder.
            src_mask (torch.Tensor): Mask applied in the encoder.
            tgt_mask (torch.Tensor): Mask applied in the decoder.

        Returns:
            torch.Tensor: Output tensor from the decoder.
        """
        x = self.residual_connections[0](
            x, lambda x: self.self_attention_block(x, x, x, mask=tgt_mask)
        )
        x = self.residual_connections[1](
            x,
            lambda x: self.cross_attention_block(
                x, encoder_output, encoder_output, mask=src_mask
            ),
        )
        x = self.residual_connections[2](x, self.feed_forward_block(x))

        return x
