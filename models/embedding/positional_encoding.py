import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Applies absolute positional encoding to the sequence embeddings.

    Attributes:
        d_model (int): The number of model dimensions.
        seq_len (int): The sequence length.
        drop_prob (float): Probability for nn.Dropout layer.
    """

    def __init__(self, d_model: int, seq_len: int, drop_prob: float):
        """
        Args:
            d_model (int): The number of model dimensions.
            seq_len (int): The sequence length.
            drop_prob (float): Probability for nn.Dropout layer.
        """
        super().__init__()

        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(p=drop_prob)

        # Create a matrix of shape: (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # Create a vector of shape: (seq_len, ) -> (seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        # We will compute a modified version of the denominator in log space which is simpler and has greater numerical stability
        # Equation: 100000^-((2*i)/d_model) = exp((2*i) * -log(10000^(1/d_model))) = exp((2*i) * -log(10000) / d_model)
        # To be exact, this is the reciprocal of the denominator which explains the negative
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000) / d_model)
        )
        # Apply sin to even positions and cos to odd positions and we multiply because as previously explained,
        # we computed the reciprocal of the denominator or `div_term`
        pe[:, ::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension
        # Shape: (batch_size, seq_len, d_model) where the batch_size = 1
        pe = pe.unsqueeze(0)

        # If there is a tensor you'd want to save but not as learnable parameters,
        # you can use the `register_buffer` method. This way the tensor will be saved
        # in the file along with the state of the model
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies absolute positional encoding to the sequence embeddings.

        Args:
            x (torch.Tensor): Sequence embeddings.

        Returns:
            torch.Tensor: Sequence embeddings with absolute positional encoding applied.
        """
        # Shape: (batch_size, seq_len, d_model)
        x = x + (self.pe[:, : x.shape[1], :]).requires_grad(False)
        return self.dropout(x)
