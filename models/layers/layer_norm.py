import torch
import torch.nn as nn


class LayerNormalization(nn.Module):
    """Applies layer normalization across the model dimensions.

    Attributes:
        eps (float, optional): Epsilon for numerical stability in normalization computation. Defaults to 1e-6.
        gamma (nn.Parameter): Learnable gamma parameter which can be used by model to expand or contract the distribution.
        beta (nn.Parameter): Learnable beta parameter which can be used by model to shift the distribution.
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        """
        Args:
            d_model (int): The number of model dimensions.
            eps (float, optional): Epsilon for numerical stability in normalization computation. Defaults to 1e-6.
        """
        super().__init__()

        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))  # gamma (multiplicative)
        self.beta = nn.Parameter(torch.zeros(d_model))  # beta (additive)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies layer normalization across the model dimensions.

        Args:
            x (torch.Tensor): Input sequence tensor.

        Returns:
            torch.Tensor: Normalized tensor.
        """
        # Find the mean and variance across the last dimension
        # Shape: (batch_size, seq_len, d_model) -> # Shape: (batch_size, seq_len, 1)
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True)

        return self.gamma * ((x - mean) / torch.sqrt(var + self.eps)) + self.beta
