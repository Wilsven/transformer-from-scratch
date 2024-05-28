import math

import torch
import torch.nn as nn


class TokenEmbeddings(nn.Module):
    """Converts the input words into embedding tokens.

    Attributes:
        d_model (int): The number of model dimensions.
        vocab_size (int): The vocabulary size.
        embedding (nn.Embedding): The embedding layer mapping the words to tokens.

    Methods:
        forward(x: torch.Tensor):
            The forward pass to convert the input words into embedding tokens.
    """

    def __init__(self, d_model: int, vocab_size: int):
        """
        Args:
            d_model (int): The number of model dimensions.
            vocab_size (int): The vocabulary size.
        """
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Converts the input words into embedding tokens.

        Args:
            x (torch.Tensor): Input words.

        Returns:
            torch.Tensor: Embedding token.
        """
        return self.embedding(x) * math.sqrt(self.d_model)
