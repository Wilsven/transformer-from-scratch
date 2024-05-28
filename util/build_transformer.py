import torch.nn as nn

from models.blocks.decoder_layer import DecoderBlock
from models.blocks.encoder_layers import EncoderBlock
from models.embedding.positional_encoding import PositionalEncoding
from models.embedding.token_embeddings import TokenEmbeddings
from models.layers.multi_head_attention import MultiHeadAttentionBlock
from models.layers.position_wise_feed_forward import PositionwiseFeedForwardBlock
from models.layers.projection_layer import ProjectionLayer
from models.model.decoder import Decoder
from models.model.encoder import Encoder
from models.model.transformer import Transformer


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
