"""
translation_models.py
---------------------
Two models for German -> English machine translation on IWSLT2017.

  GRUSeq2Seq             - GRU encoder-decoder with Bahdanau attention
  ConvS2S                - Convolutional Seq2Seq with multi-step attention
                           (Gehring et al. 2017, simplified)

Both models share the same interface:
    logits = model(src, tgt)
    logits shape: (batch, tgt_len, tgt_vocab_size)

During inference, pass the decoder input without teacher forcing:
    logits = model(src, tgt, teacher_forcing_ratio=0.0)   # GRU
    logits = model(src, tgt)                               # ConvS2S
    then argmax over the last dimension for predicted token ids.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 1. GRU Seq2Seq with Bahdanau Attention
# =============================================================================

class BahdanauAttention(nn.Module):
    """
    Additive attention (Bahdanau et al., 2015).
    Computes a context vector as a weighted sum of encoder outputs, where
    weights are learned from the current decoder hidden state.
    """
    def __init__(self, enc_hidden_size: int, dec_hidden_size: int):
        super().__init__()
        # enc_hidden_size * 2 because encoder is bidirectional
        self.Wa = nn.Linear(dec_hidden_size,     dec_hidden_size, bias=False)
        self.Ua = nn.Linear(enc_hidden_size * 2, dec_hidden_size, bias=False)
        self.Va = nn.Linear(dec_hidden_size, 1,  bias=False)

    def forward(
        self,
        hidden:          torch.Tensor,   # (batch, dec_hidden)
        encoder_outputs: torch.Tensor,   # (batch, src_len, enc_hidden * 2)
    ):
        # Broadcast hidden over src_len, then score each encoder position
        score = self.Va(torch.tanh(
            self.Wa(hidden).unsqueeze(1) + self.Ua(encoder_outputs)
        ))                                           # (batch, src_len, 1)
        weights = F.softmax(score.squeeze(2), dim=1) # (batch, src_len)
        context = torch.bmm(weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, weights                      # (batch, enc_hidden*2), (batch, src_len)


class GRUEncoder(nn.Module):
    def __init__(
        self,
        vocab_size:  int,
        embed_dim:   int,
        hidden_size: int,
        num_layers:  int,
        dropout:     float,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(
            embed_dim, hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # Merge forward + backward final states into a single decoder initial state
        self.fc      = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor):
        # src: (batch, src_len)
        embedded = self.dropout(self.embedding(src))
        outputs, hidden = self.gru(embedded)
        # outputs: (batch, src_len, hidden_size * 2)
        # hidden:  (num_layers * 2, batch, hidden_size)
        # Combine top-layer forward and backward final hidden states
        hidden = torch.tanh(
            self.fc(torch.cat([hidden[-2], hidden[-1]], dim=1))
        )   # (batch, hidden_size)
        return outputs, hidden


class GRUDecoder(nn.Module):
    def __init__(
        self,
        vocab_size:      int,
        embed_dim:       int,
        enc_hidden_size: int,
        dec_hidden_size: int,
        dropout:         float,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.attention = BahdanauAttention(enc_hidden_size, dec_hidden_size)
        # Input = current embedding + attention context
        self.gru = nn.GRU(
            embed_dim + enc_hidden_size * 2,
            dec_hidden_size,
            num_layers=1,
            batch_first=True,
        )
        # Prediction from GRU output + context + input embedding
        self.fc_out  = nn.Linear(dec_hidden_size + enc_hidden_size * 2 + embed_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        tgt_token:       torch.Tensor,   # (batch,)
        hidden:          torch.Tensor,   # (batch, dec_hidden)
        encoder_outputs: torch.Tensor,   # (batch, src_len, enc_hidden * 2)
    ):
        embedded = self.dropout(self.embedding(tgt_token.unsqueeze(1)))  # (batch, 1, embed)
        context, _ = self.attention(hidden, encoder_outputs)             # (batch, enc_hid*2)

        rnn_input = torch.cat([embedded.squeeze(1), context], dim=1).unsqueeze(1)
        output, new_hidden = self.gru(rnn_input, hidden.unsqueeze(0))
        # output:     (batch, 1, dec_hidden)
        # new_hidden: (1, batch, dec_hidden)

        prediction = self.fc_out(torch.cat([
            output.squeeze(1),
            context,
            embedded.squeeze(1),
        ], dim=1))   # (batch, vocab_size)

        return prediction, new_hidden.squeeze(0)


class GRUSeq2Seq(nn.Module):
    """
    GRU-based sequence-to-sequence model with Bahdanau attention.

    Args:
        src_vocab_size : source vocabulary size
        tgt_vocab_size : target vocabulary size
        embed_dim      : embedding dimensionality for both source and target
        hidden_size    : GRU hidden size (encoder is bidirectional; decoder uses same size)
        enc_layers     : number of encoder GRU layers
        dropout        : dropout probability
    """
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        embed_dim:      int   = 256,
        hidden_size:    int   = 256,
        enc_layers:     int   = 1,
        dropout:        float = 0.3,
    ):
        super().__init__()
        self.encoder       = GRUEncoder(src_vocab_size, embed_dim, hidden_size, enc_layers, dropout)
        self.decoder       = GRUDecoder(tgt_vocab_size, embed_dim, hidden_size, hidden_size, dropout)
        self.tgt_vocab_size = tgt_vocab_size

    def forward(
        self,
        src:                   torch.Tensor,
        tgt:                   torch.Tensor,
        teacher_forcing_ratio: float = 0.5,
    ) -> torch.Tensor:
        # src: (batch, src_len)
        # tgt: (batch, tgt_len)
        batch_size, tgt_len = tgt.shape
        outputs = torch.zeros(batch_size, tgt_len, self.tgt_vocab_size, device=src.device)

        encoder_outputs, hidden = self.encoder(src)
        input_token = tgt[:, 0]   # <sos> token

        for t in range(1, tgt_len):
            output, hidden = self.decoder(input_token, hidden, encoder_outputs)
            outputs[:, t]  = output
            use_teacher    = torch.rand(1).item() < teacher_forcing_ratio
            input_token    = tgt[:, t] if use_teacher else output.argmax(1)

        return outputs   # (batch, tgt_len, tgt_vocab_size)


# =============================================================================
# 2. ConvS2S - Convolutional Sequence-to-Sequence (Gehring et al., 2017)
# =============================================================================
# Replaces recurrence entirely with stacked 1D convolutions.
# Key architectural features:
#   - Encoder: convolutional blocks with Gated Linear Unit (GLU) activations
#              and residual connections; learned positional embeddings
#   - Decoder: causal convolutional blocks (left-only padding) with
#              multi-step dot-product attention to encoder outputs at each layer
#   - No sequential dependency between time steps -> fully parallelisable
# =============================================================================

class GLUBlock(nn.Module):
    """
    Single convolutional block with Gated Linear Unit activation and residual.

    GLU: splits the conv output in half along the channel dimension, then
    computes  output = first_half * sigmoid(second_half).
    This gating mechanism lets the network learn which features to pass forward.

    Args:
        channels    : number of input AND output channels (residual requires same size)
        kernel_size : temporal context window
        dropout     : dropout applied to the input before the convolution
        causal      : if True, uses left-only (causal) padding for decoder use
    """
    def __init__(
        self,
        channels:    int,
        kernel_size: int,
        dropout:     float = 0.1,
        causal:      bool  = False,
    ):
        super().__init__()
        self.dropout     = nn.Dropout(dropout)
        self.causal      = causal
        self.causal_pad  = kernel_size - 1
        # Non-causal: symmetric padding keeps sequence length unchanged
        padding = 0 if causal else (kernel_size - 1) // 2
        # Output channels * 2 so GLU can halve back to `channels`
        self.conv = nn.Conv1d(channels, channels * 2, kernel_size, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, seq_len)
        residual = x
        x = self.dropout(x)
        if self.causal:
            x = F.pad(x, (self.causal_pad, 0))   # left-only padding
        x = self.conv(x)
        x, gate = x.chunk(2, dim=1)              # split into value and gate
        x = x * torch.sigmoid(gate)              # GLU activation
        return x + residual                      # residual connection


class ConvS2SEncoder(nn.Module):
    def __init__(
        self,
        vocab_size:  int,
        embed_dim:   int,
        hidden_size: int,
        num_layers:  int,
        kernel_size: int,
        dropout:     float,
        max_len:     int = 512,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding   = nn.Embedding(max_len, embed_dim)
        # Project from embed_dim to hidden_size for conv layers
        self.embed_proj      = nn.Linear(embed_dim, hidden_size)
        self.layers          = nn.ModuleList([
            GLUBlock(hidden_size, kernel_size, dropout, causal=False)
            for _ in range(num_layers)
        ])
        # Project back to embed_dim so attention queries align with encoder output
        self.out_proj = nn.Linear(hidden_size, embed_dim)

    def forward(self, src: torch.Tensor):
        # src: (batch, src_len)
        batch, src_len = src.shape
        positions = torch.arange(src_len, device=src.device).unsqueeze(0).expand(batch, -1)

        x = self.token_embedding(src) + self.pos_embedding(positions)  # (batch, src_len, embed_dim)
        x = self.embed_proj(x).transpose(1, 2)   # (batch, hidden_size, src_len)

        for layer in self.layers:
            x = layer(x)

        x = x.transpose(1, 2)          # (batch, src_len, hidden_size)
        encoder_out = self.out_proj(x)  # (batch, src_len, embed_dim)  <- attention keys/values
        return encoder_out, x           # (embed_dim output, hidden_size output)


class ConvS2SDecoder(nn.Module):
    def __init__(
        self,
        vocab_size:  int,
        embed_dim:   int,
        hidden_size: int,
        num_layers:  int,
        kernel_size: int,
        dropout:     float,
        max_len:     int = 512,
    ):
        super().__init__()
        self.embed_dim       = embed_dim
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding   = nn.Embedding(max_len, embed_dim)
        self.embed_proj      = nn.Linear(embed_dim, hidden_size)
        # Causal conv blocks (left-padding only so decoder cannot see future tokens)
        self.layers = nn.ModuleList([
            GLUBlock(hidden_size, kernel_size, dropout, causal=True)
            for _ in range(num_layers)
        ])
        # Per-layer attention projection: hidden_size -> embed_dim for query
        self.attn_query_proj = nn.ModuleList([
            nn.Linear(hidden_size, embed_dim) for _ in range(num_layers)
        ])
        # Per-layer projection to add attention context back into hidden space
        self.attn_ctx_proj = nn.ModuleList([
            nn.Linear(embed_dim, hidden_size) for _ in range(num_layers)
        ])
        self.out_proj = nn.Linear(hidden_size, embed_dim)
        self.fc_out   = nn.Linear(embed_dim, vocab_size)
        self.scale    = math.sqrt(embed_dim)

    def forward(
        self,
        tgt:         torch.Tensor,   # (batch, tgt_len)
        encoder_out: torch.Tensor,   # (batch, src_len, embed_dim)  <- from ConvS2SEncoder
        tgt_emb_cache: torch.Tensor = None,  # reused below
    ) -> torch.Tensor:
        batch, tgt_len = tgt.shape
        positions = torch.arange(tgt_len, device=tgt.device).unsqueeze(0).expand(batch, -1)

        tgt_emb = self.token_embedding(tgt) + self.pos_embedding(positions)  # (batch, tgt_len, embed_dim)
        x = self.embed_proj(tgt_emb).transpose(1, 2)   # (batch, hidden_size, tgt_len)

        for layer, q_proj, ctx_proj in zip(self.layers, self.attn_query_proj, self.attn_ctx_proj):
            x = layer(x)   # (batch, hidden_size, tgt_len)

            # Multi-step attention: query = projected decoder state + target embedding
            query = q_proj(x.transpose(1, 2)) + tgt_emb  # (batch, tgt_len, embed_dim)
            attn_scores  = torch.bmm(query, encoder_out.transpose(1, 2)) / self.scale
            attn_weights = F.softmax(attn_scores, dim=-1)               # (batch, tgt_len, src_len)
            context      = torch.bmm(attn_weights, encoder_out)         # (batch, tgt_len, embed_dim)

            # Add attention context back into the hidden representation
            x = x + ctx_proj(context).transpose(1, 2)

        x = self.out_proj(x.transpose(1, 2))  # (batch, tgt_len, embed_dim)
        return self.fc_out(x)                  # (batch, tgt_len, vocab_size)


class ConvS2S(nn.Module):
    """
    Fully convolutional sequence-to-sequence model with multi-step attention.

    Args:
        src_vocab_size    : source vocabulary size
        tgt_vocab_size    : target vocabulary size
        embed_dim         : token and positional embedding dimensionality
        hidden_size       : number of channels in all convolutional layers
        num_encoder_layers: number of encoder GLU blocks
        num_decoder_layers: number of decoder GLU blocks
        kernel_size       : convolutional kernel size (same for encoder and decoder)
        dropout           : dropout probability
        max_len           : maximum sequence length for positional embeddings
    """
    def __init__(
        self,
        src_vocab_size:     int,
        tgt_vocab_size:     int,
        embed_dim:          int   = 256,
        hidden_size:        int   = 512,
        num_encoder_layers: int   = 4,
        num_decoder_layers: int   = 4,
        kernel_size:        int   = 3,
        dropout:            float = 0.1,
        max_len:            int   = 512,
    ):
        super().__init__()
        self.encoder = ConvS2SEncoder(
            src_vocab_size, embed_dim, hidden_size,
            num_encoder_layers, kernel_size, dropout, max_len,
        )
        self.decoder = ConvS2SDecoder(
            tgt_vocab_size, embed_dim, hidden_size,
            num_decoder_layers, kernel_size, dropout, max_len,
        )

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        # src: (batch, src_len)
        # tgt: (batch, tgt_len)  — pass tgt[:, :-1] during training
        encoder_out, _ = self.encoder(src)
        return self.decoder(tgt, encoder_out)   # (batch, tgt_len, tgt_vocab_size)
