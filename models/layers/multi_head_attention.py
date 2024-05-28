import math
from typing import Optional

import torch
import torch.nn as nn


class MultiHeadAttentionBlock(nn.Module):
    """Applies Multi-Head Attention.

    Attributes:
        d_model (int): The number of model dimensions.
        h (int): Number of attention heads.
        d_k (int): The number of dimensions for each head.
        w_q (nn.Linear): Weights mapping to query vectors.
        w_k (nn.Linear): Weights mapping to key vectors.
        w_v (nn.Linear): Weights mapping to value vectors.
        w_o (nn.Linear): Weights mapping to output vectors.
        dropout (nn.Dropout): nn.Dropout layer.
    """

    def __init__(self, d_model: int, h: int, drop_prob: float):
        """
        Args:
            d_model (int): The number of model dimensions.
            h (int):  Number of attention heads.
            drop_prob (float): Probability for nn.Dropout layer.
        """
        super().__init__()

        self.d_model = d_model
        self.h = h
        assert (
            d_model % h == 0
        ), "Dimension of model must be divisible by number of heads"
        self.d_k = d_model // h

        # All linear layers have a bias term
        self.w_q = nn.Linear(d_model, d_model)  # Wq
        self.w_k = nn.Linear(d_model, d_model)  # Wk
        self.w_v = nn.Linear(d_model, d_model)  # Wv

        self.w_o = nn.Linear(d_model, d_model)  # Wo

        self.dropout = nn.Dropout(p=drop_prob)

    @staticmethod
    def attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        dropout: Optional[nn.Dropout] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor]:
        """Static method to calculate the scaled dot self-attention computation.

        Args:
            query (torch.Tensor): The query vector.
            key (torch.Tensor): The key vector.
            value (torch.Tensor): The value vector.
            dropout (Optional[nn.Dropout], optional): The nn.Dropout layer. Defaults to None.
            mask (Optional[torch.Tensor], optional): The lookahead/causal mask. Defaults to None.

        Returns:
            tuple[torch.Tensor]: Tuple constaining the scaled attention scores
                                and unscaled attention scores which will be used for visualization.
        """
        # Shape: (batch_size, h, seq_len, d_k)
        d_k = query.shape[-1]

        # Shape: (batch_size, h, seq_len, d_k) @ (batch_size, h, d_k, seq_len) -> (batch_size, h, seq_len, seq_len)
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)

        # Shape: (batch_size, h, seq_len, seq_len)
        attention_scores = attention_scores.softmax(dim=-1)

        if dropout is not None:
            attention_scores = dropout(attention_scores)

        # Shape: (batch_size, h, seq_len, seq_len) @ (batch_size, h, seq_len, d_k) -> (batch_size, h, seq_len, d_k)
        return (attention_scores @ value), attention_scores

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Computes the Multi-Head Attention.

        Args:
            q (torch.Tensor): The query vector.
            k (torch.Tensor):  The key vector.
            v (torch.Tensor):  The value vector.
            mask (Optional[torch.Tensor], optional): The lookahead/causal mask. Defaults to None.

        Returns:
            torch.Tensor: Returns a more contextually aware tensor.
        """
        batch_size, seq_len, _ = query.shape
        # Shape: (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_model)
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        # Shape: (batch_size, seq_len, d_model) -> (batch_size, seq_len, h, d_k)
        query = query.view(batch_size, seq_len, self.h, self.d_k)
        key = key.view(batch_size, seq_len, self.h, self.d_k)
        value = value.view(batch_size, seq_len, self.h, self.d_k)

        # For easier computation later, we transpose the second and third dimensions
        # Shape: (batch_size, seq_len, h, d_k) -> (batch_size, h, seq_len, d_k)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # Shape: (batch_size, h, seq_len, d_k)
        x, self.attention_scores = MultiHeadAttentionBlock.attention(
            query, key, value, dropout=self.dropout, mask=mask
        )

        # Shape: (batch_size, h, seq_len, d_k) -> (batch_size, seq_len, h, d_k)
        x = x.transpose(1, 2)
        # Combine the heads
        # Shape: (batch_size, seq_len, h, d_k) -> (batch_size, seq_len, h * d_k) -> (batch_size, seq_len, d_model)
        x = x.contiguous().view(batch_size, -1, self.h * self.d_k)

        # Shape: (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_model)
        return self.w_o(x)
