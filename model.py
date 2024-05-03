# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
import math
import torch
import torch.nn as nn
from torch.nn import Transformer

class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: torch.Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: torch.Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)
    

class NoPositionalEncoding(nn.Module):
    def __init__(self):
        super(NoPositionalEncoding, self).__init__()

    def forward(self, x):
        return x
    
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
                 dropout: float = 0.1,
                 positional_encoding : bool = False):
        
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

        self.positional_encoding = PositionalEncoding(emb_size, dropout = dropout, maxlen = max(src_vocab_size, tgt_vocab_size)+1) if positional_encoding else NoPositionalEncoding()


    def forward(self,
                src: torch.Tensor,
                trg: torch.Tensor,
                src_mask: torch.Tensor,
                tgt_mask: torch.Tensor,
                src_padding_mask: torch.Tensor,
                tgt_padding_mask: torch.Tensor,
                memory_key_padding_mask: torch.Tensor):
        
        outs = self.transformer(self.positional_encoding(self.src_tok_emb(src)), self.positional_encoding(self.tgt_tok_emb(trg)),
                                src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor):
        
        return self.transformer.encoder(
                            self.positional_encoding(self.src_tok_emb(src)),  mask=src_mask)

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: torch.Tensor):
        return self.transformer.decoder(
                          self.positional_encoding(self.tgt_tok_emb(tgt)), memory = memory,
                          tgt_mask = tgt_mask)
    
    # We need to add source padding mask to avoid attending to source padding tokens
    def batch_encode(self, src: torch.Tensor, src_mask: torch.Tensor, src_key_padding_mask: torch.Tensor):
        return self.transformer.encoder(
                            self.positional_encoding(self.src_tok_emb(src)),  mask=src_mask, src_key_padding_mask=src_key_padding_mask)
    # No need for batch_decode as we're generating one token at a time