import torch
import torch.nn as nn


class ProjectionLayer(nn.Module):
    """Final linear layer the output of the decoder is passed through.

    Attributes:
        proj (nn.Linear): nn.Linear layer to project dimension of output tensor to dimension equal to vocabulary size.
    """

    def __init__(self, d_model: int, vocab_size: int):
        """
        Args:
            d_model (int): The number of model dimensions.
            vocab_size (int): The vocabulary size.
        """
        super().__init__()

        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor):
        """Projects output dimension to vocabulary size.

        Args:
            x (torch.Tensor): Output tensor from decoder.

        Returns:
            torch.Tensor: Final tensor with probability distribution across entire vocabulary.
        """
        # Shape: (batch_size, seq_len, d_model) -> (batch_size, seq_len, vocab_size)
        return torch.log_softmax(self.proj(x), dim=-1)
