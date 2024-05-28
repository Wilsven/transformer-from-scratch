import math
from typing import Optional

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


class PositionwiseFeedForwardBlock(nn.Module):
    """Fully-connected feed forward neural network used in the Encoder and Decoder.

    Attributes:
        linear_1 (nn.Linear): nn.Linear layer.
        relu (nn.ReLU): nn.ReLU activation function.
        dropout (nn.Dropout): nn.Dropout layer.
        linear_2 (nn.Linear): nn.Linear layer.
    """

    def __init__(self, d_model: int, d_ff: int, drop_prob: float):
        """
        Args:
            d_model (int): The number of model dimensions.
            d_ff (int): The number of dimensions for hidden layer
            drop_prob (float): Probability for nn.Dropout layer.
        """
        super().__init__()

        self.linear_1 = nn.Linear(d_model, d_ff, bias=True)  # W1 and b1
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)
        self.linear_2 = nn.Linear(d_ff, d_model, bias=True)  # W2 and b2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Shape: (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_ff) -> (batch_size, seq_len, d_model)
        return self.linear_2(self.dropout(self.relu(self.linear_1(x))))


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


class ResidualConnection(nn.Module):
    """Performs residual connections in the Transformer architecture.

    Attributes:
        dropout (nn.Dropout): nn.Dropout layer.
        norm (LayerNormalization): LayerNormalization block.
    """

    def __init__(self, d_model: int, drop_prob: float):
        """
        Args:
            d_model (int): The number of model dimensions.
            drop_prob (float): Probability for nn.Dropout layer.
        """
        super().__init__()

        self.dropout = nn.Dropout(p=drop_prob)
        self.norm = LayerNormalization(d_model)

    def forward(self, x: torch.Tensor, sublayer) -> torch.Tensor:
        """Performs residual connections.

        Args:
            x (torch.Tensor): Input tensor.
            sublayer: The sublayer the input tensor will be passed to and the initial input
                        will be added to the subsequent output.

        Returns:
            torch.Tensor: The output tensor after skip connection.
        """
        # There is a slight difference that we first apply the normalization first then the sublayer
        # In the original transformer paper, they applied the sublayer first before the normalization
        # return x + self.dropout(self.norm(sublayer(x)))
        return x + self.dropout(sublayer(self.norm(x)))


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


def build_transformer(
    src_vocab_size: int,
    tgt_vocab_size: int,
    src_seq_len: int,
    tgt_seq_len: int,
    d_model: int = 512,
    N: int = 6,
    h: int = 8,
    drop_prob: float = 0.1,
    d_ff: int = 2048,
) -> Transformer:
    # Create embedding layers
    src_embed = TokenEmbeddings(d_model, src_vocab_size)
    tgt_embed = TokenEmbeddings(d_model, tgt_vocab_size)

    # Create the positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, drop_prob)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, drop_prob)

    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, drop_prob)
        encoder_feed_forward_block = PositionwiseFeedForwardBlock(
            d_model, d_ff, drop_prob
        )
        encoder_block = EncoderBlock(
            encoder_self_attention_block, encoder_feed_forward_block, drop_prob
        )
        encoder_blocks.append(encoder_block)

    # Create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, drop_prob)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, drop_prob)
        decoder_feed_forward_block = PositionwiseFeedForwardBlock(
            d_model, d_ff, drop_prob
        )
        decoder_block = DecoderBlock(
            decoder_self_attention_block,
            decoder_cross_attention_block,
            decoder_feed_forward_block,
        )
        decoder_blocks.append(decoder_block)

    # Create the Encoder and Decoder
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))

    # Create projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # Create the Transformer
    transformer = Transformer(
        encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer
    )

    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer
