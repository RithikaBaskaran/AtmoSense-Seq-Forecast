"""
model.py 

Exports:
    PositionalEncoding : sinusoidal positional encoding module
    AQITransformer     : encoder-decoder transformer for multi-pollutant forecasting



    model = AQITransformer(
        n_features = len(feat_cols),
        n_targets  = len(feat_cols),
        seq_len    = 72,
        pred_len   = 48,
    ).cuda()

    # Forward pass (teacher forcing during training)
    # src : (batch, 72, n_features)
    # tgt : (batch, 48, n_targets)   — shifted target sequence
    output = model(src, tgt)         # → (batch, 48, n_targets)
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    Injects time-order information into the input sequence.
    Each position gets a unique sin/cos pattern so the transformer
    knows hour 1 from hour 72.
    """
    def __init__(self, d_model: int, max_len: int = 1000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class AQITransformer(nn.Module):
    """
    Encoder-Decoder Transformer for multi-pollutant air quality forecasting.

    Input  → (batch, seq_len=72,  n_features)  — past 72 hrs, all pollutants
    Output → (batch, pred_len=48, n_targets)   — next 48 hrs, all pollutants

    Reference: adapted from Time Series Library (TSLib) by Wu et al. 2022
    https://github.com/thuml/Time-Series-Library
    """

    def __init__(
        self,
        n_features: int,
        n_targets: int,
        seq_len: int        = 72,
        pred_len: int       = 48,
        d_model: int        = 128,
        nhead: int          = 8,
        num_enc_layers: int = 4,
        num_dec_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float      = 0.1,
    ):
        super().__init__()
        self.seq_len  = seq_len
        self.pred_len = pred_len
        self.d_model  = d_model

        self.enc_input_proj = nn.Linear(n_features, d_model)
        self.dec_input_proj = nn.Linear(n_targets,  d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=1000, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_enc_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_dec_layers)

        self.output_proj = nn.Linear(d_model, n_targets)

    def _causal_mask(self, size: int, device: torch.device) -> torch.Tensor:
        mask = torch.triu(torch.ones(size, size, device=device), diagonal=1).bool()
        return mask

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        src_emb  = self.pos_enc(self.enc_input_proj(src))
        memory   = self.encoder(src_emb)
        tgt_emb  = self.pos_enc(self.dec_input_proj(tgt))
        tgt_mask = self._causal_mask(tgt.size(1), tgt.device)
        dec_out  = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
        return self.output_proj(dec_out)
