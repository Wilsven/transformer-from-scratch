import torch
import torch.nn as nn

from models.blocks.residual_connection import ResidualConnection
from models.layers.multi_head_attention import MultiHeadAttentionBlock
from models.layers.position_wise_feed_forward import \
    PositionwiseFeedForwardBlock


class EncoderBlock(nn.Module):
    """An Encoder Block.

    Attributes:
        self_attention_block (MultiHeadAttentionBlock): Self-attention over multiple attention heads.
        feed_forward_block (PositionwiseFeedForwardBlock): Position-wise feed forward block in the encoder.
        residual_connections (nn.ModuleList[ResidualConnection]): Encoder has two residual connections.
    """

    def __init__(
        self,
        self_attention_block: MultiHeadAttentionBlock,
        feed_forward_block: PositionwiseFeedForwardBlock,
        drop_prob: float,
    ):
        """
        Args:
            self_attention_block (MultiHeadAttentionBlock): Self-attention over multiple attention heads.
            feed_forward_block (PositionwiseFeedForwardBlock): Position-wise feed forward block in the encoder.
            drop_prob (float): Probability for nn.Dropout layer.
        """
        super().__init__()

        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList(
            [ResidualConnection(drop_prob) for _ in range(2)]
        )

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """One forward pass through the encoder. This can be applied N times.

        Args:
            x (torch.Tensor): Input tensor to the encoder.
            src_mask (torch.Tensor): Mask to mask padding words.

        Returns:
            torch.Tensor: Output tensor from the encoder.
        """
        x = self.residual_connections[0](
            x, lambda x: self.self_attention_block(x, x, x, mask=src_mask)
        )
        x = self.residual_connections[1](x, self.feed_forward_block(x))

        return x
