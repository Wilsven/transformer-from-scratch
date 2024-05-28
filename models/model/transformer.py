import torch
import torch.nn as nn

from models.embedding.positional_encoding import PositionalEncoding
from models.embedding.token_embeddings import TokenEmbeddings
from models.layers.projection_layer import ProjectionLayer
from models.model.decoder import Decoder
from models.model.encoder import Encoder


class Transformer(nn.Module):
    """Transformer architecture.

    Attributes:
        encoder (Encoder): The Encoder.
        decoder (Decoder): The Decoder.
        src_embed (TokenEmbeddings): TokenEmbedding layer for source sentence.
        tgt_embed (TokenEmbeddings): TokenEmbedding layer for target sentence.
        src_pos (PositionalEncoding): PositionalEncoding layer for source embeddings.
        tgt_pos (PositionalEncoding): PositionalEncoding layer for target embeddings.
        projection_layer (ProjectionLayer): nn.Linear layer to project dimension of output tensor to dimension
                                            equal to vocabulary size.
    """

    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        src_embed: TokenEmbeddings,
        tgt_embed: TokenEmbeddings,
        src_pos: PositionalEncoding,
        tgt_pos: PositionalEncoding,
        projection_layer: ProjectionLayer,
    ):
        """
        Args:
            encoder (Encoder): _description_
            decoder (Decoder): _description_
            src_embed (TokenEmbeddings): _description_
            tgt_embed (TokenEmbeddings): _description_
            src_pos (PositionalEncoding): _description_
            tgt_pos (PositionalEncoding): _description_
            projection_layer (ProjectionLayer): _description_
        """
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """Encodes the source sentence.

        Args:
            src (torch.Tensor): Input tensor to the encoder.
            src_mask (torch.Tensor): Mask to mask padding words.

        Returns:
            torch.Tensor: Output tensor from encoder.
        """
        # Get token embeddings for source sentence
        src = self.src_embed(src)
        # Apply positional encodings
        src = self.src_pos(src)

        return self.encoder(src, mask=src_mask)

    def decode(
        self,
        tgt: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Decodes the target sentence.

        Args:
            tgt (torch.Tensor): Input tensor to the decoder.
            encoder_output (torch.Tensor): Output tensor from the encoder.
            src_mask (torch.Tensor): Mask applied in the encoder.
            tgt_mask (torch.Tensor): Mask applied in the decoder.

        Returns:
            torch.Tensor: Output tensor from decoder.
        """
        # Get token embeddings for target sentence
        tgt = self.tgt_embed(tgt)
        # Apply positional encodings
        tgt = self.tgt_pos(tgt)

        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """Apply final linear layer and softmax to get probability distribution over vocabulary.

        Args:
            x (torch.Tensor): Output tensor from decoder.

        Returns:
            torch.Tensor: Tensor of probability distribution over vocabulary.
        """
        return self.projection_layer(x)
