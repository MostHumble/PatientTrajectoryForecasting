# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
import math
from torch import Tensor
import torch.nn as nn
from torch.nn import Transformer

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)
    
# Seq2Seq Network
class Seq2SeqTransformer(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 dim_feedforward: int,
                 dropout: float = 0.1):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = Transformer(d_model = emb_size,
                                       nhead = nhead,
                                       num_encoder_layers = num_encoder_layers,
                                       num_decoder_layers = num_decoder_layers,
                                       dim_feedforward = dim_feedforward,
                                       dropout = dropout,
                                       batch_first = True, norm_first = True)
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)

    def forward(self,
                src: Tensor,
                trg: Tensor,
                src_mask: Tensor,
                tgt_mask: Tensor,
                src_padding_mask: Tensor,
                tgt_padding_mask: Tensor,
                memory_key_padding_mask: Tensor):
        src_emb = self.src_tok_emb(src)
        tgt_emb = self.tgt_tok_emb(trg)
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(
                            self.src_tok_emb(src), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer.decoder(
                          self.tgt_tok_emb(tgt), memory,
                          tgt_mask)
    
    # We need to add source padding mask to avoid attending to source padding tokens
    def batch_encode(self, src: Tensor, src_mask: Tensor, src_key_padding_mask: Tensor):
        return self.transformer.encoder(
                            self.src_tok_emb(src), src_mask, src_key_padding_mask)
    # No need for batch_decode as we're generating one token at a time