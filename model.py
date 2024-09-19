# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
import math

import torch
import torch.nn as nn
from torch.nn import Transformer


class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout: float, maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer("pos_embedding", pos_embedding)

    def forward(self, token_embedding: torch.Tensor):
        return self.dropout(
            token_embedding + self.pos_embedding[: token_embedding.size(1)]
        )  # seq_len


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
    def __init__(
        self,
        num_encoder_layers: int,
        num_decoder_layers: int,
        emb_size: int,
        nhead: int,
        src_vocab_size: int,
        tgt_vocab_size: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        positional_encoding: bool = False,
    ):

        super(Seq2SeqTransformer, self).__init__()
        self.transformer = Transformer(
            d_model=emb_size,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)

        self.positional_encoding = (
            PositionalEncoding(
                emb_size,
                dropout=dropout,
                maxlen=max(src_vocab_size, tgt_vocab_size) + 1,
            )
            if positional_encoding
            else NoPositionalEncoding()
        )

    def forward(
        self,
        src: torch.Tensor,
        trg: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
        src_padding_mask: torch.Tensor,
        tgt_padding_mask: torch.Tensor,
        memory_key_padding_mask: torch.Tensor,
    ):

        outs = self.transformer(
            self.positional_encoding(self.src_tok_emb(src)),
            self.positional_encoding(self.tgt_tok_emb(trg)),
            src_mask,
            tgt_mask,
            None,
            src_padding_mask,
            tgt_padding_mask,
            memory_key_padding_mask,
        )
        return self.generator(outs)

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor):

        return self.transformer.encoder(
            self.positional_encoding(self.src_tok_emb(src)), mask=src_mask
        )

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: torch.Tensor):
        return self.transformer.decoder(
            self.positional_encoding(self.tgt_tok_emb(tgt)),
            memory=memory,
            tgt_mask=tgt_mask,
        )

    # We need to add source padding mask to avoid attending to source padding tokens
    def batch_encode(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor,
        src_key_padding_mask: torch.Tensor,
    ):
        return self.transformer.encoder(
            self.positional_encoding(self.src_tok_emb(src)),
            mask=src_mask,
            src_key_padding_mask=src_key_padding_mask,
        )

    # No need for batch_decode as we're generating one token at a time

    # Seq2Seq Network


class Seq2SeqTransformerWithNotes(nn.Module):
    def __init__(
        self,
        num_encoder_layers: int,
        num_decoder_layers: int,
        emb_size: int,
        nhead: int,
        src_vocab_size: int,
        tgt_vocab_size: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        use_positional_encoding_notes: bool = False,
        positional_encoding: bool = False,
        bert = None,
    ):

        super(Seq2SeqTransformerWithNotes, self).__init__()
        self.bert = bert
        if bert is not None:
            self.bert.eval()

            if self.bert.config.strategy == "all":
                self.projection = LinearProjection(
                    bert.config.seq_len, bert.config.seq_len // 2
                )  # seq_len
        self.transformer = Transformer(
            d_model=emb_size,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.use_positional_encoding_notes = use_positional_encoding_notes
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        

        self.positional_encoding = (
            PositionalEncoding(
                emb_size,
                dropout=dropout,
                maxlen=max(src_vocab_size, tgt_vocab_size) + 1,
            )
            if positional_encoding
            else NoPositionalEncoding()
        )

    def forward(
        self,
        src: torch.Tensor,
        trg: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
        src_padding_mask: torch.Tensor,
        tgt_padding_mask: torch.Tensor,
        memory_key_padding_mask: torch.Tensor,
        notes_input_ids: torch.Tensor = None,
        notes_attention_mask: torch.Tensor = None,
        notes_token_type_ids: torch.Tensor = None,
        hospital_ids_lens=None,
    ):

        source_embeds = self.positional_encoding(self.src_tok_emb(src))
        # source_embeds : batch, seq_len, hidden dim
        if self.bert:

            with torch.inference_mode():

                out_notes = self.bert(
                    input_ids=notes_input_ids,
                    attention_mask=notes_attention_mask,
                    token_type_ids=notes_token_type_ids,
                    hospital_ids_lens=hospital_ids_lens,
                )

            # Fusion happens here
            # in case of reduction in num_encoder_layers dim
            # out_dim : batch, hidden dim
            if out_notes.ndim == 2:
                source_embeds = torch.cat(
                    (out_notes.unsqueeze(1), source_embeds), dim=1
                )

            # out_dim : batch, num_encoder_layers, hidden dim
            elif out_notes.ndim == 3:
                source_embeds = torch.cat((out_notes, source_embeds), dim=1)
            # out_dim : batch, num_encoder_layers, seq_len, hidden dim
            elif out_notes.ndim == 4:
                out_notes = self.projection(
                    out_notes
                )  # num_layers, batch_size (concatenated hospital admissions), hidden_dim
                batch_embeddings = []
                start_idx = 0

                for i, length in enumerate(hospital_ids_lens):
                    # we cat across seq_len, and truncate to 512 (max_seq_len_input)
                    if self.use_positional_encoding_notes:
                        batch_embeddings.append(
                            torch.cat(
                                [
                                    self.positional_encoding(
                                        out_notes[:, start_idx : start_idx + length]
                                    ).reshape(-1, 768),
                                    source_embeds[i],
                                ],
                                dim=0,
                            )[
                                :512,
                            ]
                        )
                    else:
                        batch_embeddings.append(
                            torch.cat(
                                [
                                    out_notes[
                                        :, start_idx : start_idx + length
                                    ].reshape(-1, 768),
                                    source_embeds[i],
                                ],
                                dim=0,
                            )[
                                :512,
                            ]
                        )
                    start_idx += length
                # num_layer, batch_size (true batch size), hidden_dim.
                source_embeds = torch.stack(batch_embeddings)

        outs = self.transformer(
            source_embeds,
            self.positional_encoding(self.tgt_tok_emb(trg)),
            src_mask,
            tgt_mask,
            None,
            src_padding_mask,
            tgt_padding_mask,
            memory_key_padding_mask,
        )

        return self.generator(outs)

    def encode(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor,
        notes_input_ids: torch.Tensor = None,
        notes_attention_mask: torch.Tensor = None,
        notes_token_type_ids: torch.Tensor = None,
        hospital_ids_lens: torch.Tensor = None,
    ):

        source_embeds = self.positional_encoding(self.src_tok_emb(src))

        if self.bert:

            with torch.inference_mode():

                out_notes = self.bert(
                    input_ids=notes_input_ids,
                    attention_mask=notes_attention_mask,
                    token_type_ids=notes_token_type_ids,
                    hospital_ids_lens=hospital_ids_lens,
                )

            # Fusion happens here
            # in case of reduction in num_encoder_layers dim
            # out_notes : batch, hidden dim
            if out_notes.ndim == 2:
                source_embeds = torch.cat(
                    (out_notes.unsqueeze(1), source_embeds), dim=1
                )

            # out_notes : batch, num_encoder_layers, hidden dim
            elif out_notes.ndim == 3:
                source_embeds = torch.cat((out_notes, source_embeds), dim=1)

            # out_notes : batch, num_encoder_layers, seq_len, hidden dim
            else:
                out_notes = self.projection(out_notes)
                batch_embeddings = []
                start_idx = 0

                for i, length in enumerate(hospital_ids_lens):
                    # num_layer, batch_size (true batch size), hidden_dim.
                    if self.use_positional_encoding_notes:
                        batch_embeddings.append(
                            torch.cat(
                                [
                                    self.positional_encoding(
                                        out_notes[:, start_idx : start_idx + length]
                                    ).reshape(-1, 768),
                                    source_embeds[i],
                                ],
                                dim=0,
                            )[
                                :512,
                            ]
                        )
                    else:
                        batch_embeddings.append(
                            torch.cat(
                                [
                                    out_notes[
                                        :, start_idx : start_idx + length
                                    ].reshape(-1, 768),
                                    source_embeds[i],
                                ],
                                dim=0,
                            )[
                                :512,
                            ]
                        )
                    start_idx += length

                source_embeds = torch.stack(batch_embeddings)

        return self.transformer.encoder(source_embeds, mask=src_mask)

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: torch.Tensor):
        return self.transformer.decoder(
            self.positional_encoding(self.tgt_tok_emb(tgt)),
            memory=memory,
            tgt_mask=tgt_mask,
        )

    # We need to add source padding mask to avoid attending to source padding tokens
    def batch_encode(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor,
        src_key_padding_mask: torch.Tensor,
        notes_input_ids: torch.Tensor = None,
        notes_attention_mask: torch.Tensor = None,
        notes_token_type_ids: torch.Tensor = None,
        hospital_ids_lens: torch.Tensor = None,
    ):

        source_embeds = self.positional_encoding(self.src_tok_emb(src))

        if self.bert:

            with torch.inference_mode():

                out_notes = self.bert(
                    input_ids=notes_input_ids,
                    attention_mask=notes_attention_mask,
                    token_type_ids=notes_token_type_ids,
                    hospital_ids_lens=hospital_ids_lens,
                )

            # Fusion happens here
            # in case of reduction in num_encoder_layers dim
            # out_dim : batch, hidden dim
            if out_notes.ndim == 2:
                source_embeds = torch.cat(
                    (out_notes.unsqueeze(1), source_embeds), dim=1
                )

            # out_dim : batch, num_encoder_layers, hidden dim
            elif out_notes.ndim == 3:
                source_embeds = torch.cat((out_notes, source_embeds), dim=1)
            else:
                out_notes = self.projection(out_notes)
                batch_embeddings = []
                start_idx = 0

                for i, length in enumerate(hospital_ids_lens):
                    # We then average across visits
                    # num_layer, batch_size (true batch size), hidden_dim.
                    if self.use_positional_encoding_notes:
                        batch_embeddings.append(
                            torch.cat(
                                [
                                    self.positional_encoding(
                                        out_notes[:, start_idx : start_idx + length]
                                    ).reshape(-1, 768),
                                    source_embeds[i],
                                ],
                                dim=0,
                            )[
                                :512,
                            ]
                        )
                    else:
                        batch_embeddings.append(
                            torch.cat(
                                [
                                    out_notes[
                                        :, start_idx : start_idx + length
                                    ].reshape(-1, 768),
                                    source_embeds[i],
                                ],
                                dim=0,
                            )[
                                :512,
                            ]
                        )
                    start_idx += length

                source_embeds = torch.stack(batch_embeddings)

        return self.transformer.encoder(
            source_embeds, mask=src_mask, src_key_padding_mask=src_key_padding_mask
        )


class LinearProjection(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(LinearProjection, self).__init__()

        # Define the linear layers and activation function
        self.input_to_hidden = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.GELU()
        self.hidden_to_output = nn.Linear(hidden_dim, 1)

    def forward(self, input_tensor):
        """
        Forward pass for the LinearProjection module.

        Args:
            input_tensor (torch.Tensor): Input tensor of shape (num_layers, batch_size, seq_len, input_dim)

        Returns:
            torch.Tensor: Output tensor of shape (num_layers, batch_size, hidden_dim)
        """
        permuted_tensor = input_tensor.permute(
            0, 1, 3, 2
        )  # (num_layers, batch_size, input_dim, seq_len)

        hidden_tensor = self.input_to_hidden(
            permuted_tensor
        )  # (num_layers, batch_size, hidden_dim, seq_len)

        activated_tensor = self.activation(
            hidden_tensor
        )  #  (num_layers, batch_size, hidden_dim, seq_len)

        output_tensor = self.hidden_to_output(
            activated_tensor
        )  # (num_layers, batch_size, hidden_dim, 1)

        output_tensor = output_tensor.squeeze(
            -1
        )  # (num_layers, batch_size, hidden_dim)

        return output_tensor
