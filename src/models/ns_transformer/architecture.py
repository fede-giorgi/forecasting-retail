"""
Non-Stationary Transformer (NeurIPS 2022) architecture, adapted from the
reference implementation. Removed: client/cluster-level abstractions specific
to the electricity dataset. Kept: DSAttention with learned tau/delta, encoder
stack, decoder stack, projector. This is a global multi-series model — one
network forecasts all SKUs.

Tensors:
  enc_in   = number of input channels (target + dynamic_real features)
  c_out    = 1 (forecast Quantity per SKU)
  seq_len  = lookback in weeks (default 52 = 1 year)
  pred_len = forecast horizon (default 12)

For training and inference glue see ns_transformer/train.py.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.pe[:, : x.size(1)]


class DataEmbedding(nn.Module):
    def __init__(self, c_in: int, d_model: int, n_time_feat: int = 2, dropout: float = 0.1):
        super().__init__()
        self.value = nn.Linear(c_in, d_model)
        self.pos = PositionalEncoding(d_model)
        self.temp = nn.Linear(n_time_feat, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, x_mark):
        return self.drop(self.value(x) + self.pos(x) + self.temp(x_mark))


class DSAttention(nn.Module):
    """De-stationarized scaled dot-product attention (the NS-Transformer trick)."""
    def __init__(self, mask_flag: bool = False, dropout: float = 0.1):
        super().__init__()
        self.mask_flag = mask_flag
        self.drop = nn.Dropout(dropout)

    def forward(self, q, k, v, attn_mask=None, tau=None, delta=None):
        B, L, H, E = q.shape
        scale = 1.0 / math.sqrt(E)
        tau = 1.0 if tau is None else tau.unsqueeze(1).unsqueeze(1)
        delta = 0.0 if delta is None else delta.unsqueeze(1).unsqueeze(1)

        scores = torch.einsum("blhe,bshe->bhls", q, k) * tau + delta
        if self.mask_flag and attn_mask is not None:
            scores.masked_fill_(attn_mask, -1e9)
        a = self.drop(torch.softmax(scale * scores, dim=-1))
        out = torch.einsum("bhls,bshd->blhd", a, v)
        return out.contiguous(), a


class AttentionLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        d_keys = d_model // n_heads
        self.inner = DSAttention(dropout=dropout)
        self.q_proj = nn.Linear(d_model, d_keys * n_heads)
        self.k_proj = nn.Linear(d_model, d_keys * n_heads)
        self.v_proj = nn.Linear(d_model, d_keys * n_heads)
        self.out_proj = nn.Linear(d_keys * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, q, k, v, mask=None, tau=None, delta=None):
        B, L, _ = q.shape
        S = k.shape[1]
        H = self.n_heads
        q = self.q_proj(q).view(B, L, H, -1)
        k = self.k_proj(k).view(B, S, H, -1)
        v = self.v_proj(v).view(B, S, H, -1)
        out, _ = self.inner(q, k, v, mask, tau, delta)
        return self.out_proj(out.view(B, L, -1))


class _Block(nn.Module):
    """Shared encoder/decoder layer body (self-attn + FFN)."""
    def __init__(self, d_model, n_heads, d_ff, dropout, with_cross=False):
        super().__init__()
        self.self_attn = AttentionLayer(d_model, n_heads, dropout)
        self.cross_attn = AttentionLayer(d_model, n_heads, dropout) if with_cross else None
        self.conv1 = nn.Conv1d(d_model, d_ff, 1)
        self.conv2 = nn.Conv1d(d_ff, d_model, 1)
        self.n1 = nn.LayerNorm(d_model)
        self.n2 = nn.LayerNorm(d_model)
        self.n3 = nn.LayerNorm(d_model) if with_cross else None
        self.drop = nn.Dropout(dropout)

    def forward(self, x, cross=None, x_mask=None, c_mask=None, tau=None, delta=None):
        x = self.n1(x + self.drop(self.self_attn(x, x, x, x_mask, tau, None)))
        if self.cross_attn is not None:
            x = self.n2(x + self.drop(self.cross_attn(x, cross, cross, c_mask, tau, delta)))
            norm = self.n3
        else:
            norm = self.n2
        y = self.drop(F.gelu(self.conv1(x.transpose(-1, 1))))
        y = self.drop(self.conv2(y).transpose(-1, 1))
        return norm(x + y)


class Projector(nn.Module):
    """Learns tau (scale) / delta (shift) from raw + statistics for de-stationarization."""
    def __init__(self, enc_in, seq_len, hidden, output_dim):
        super().__init__()
        self.series_conv = nn.Conv1d(seq_len, 1, 3, padding=1, padding_mode="circular", bias=False)
        layers = [nn.Linear(2 * enc_in, hidden[0]), nn.ReLU()]
        for i in range(len(hidden) - 1):
            layers += [nn.Linear(hidden[i], hidden[i + 1]), nn.ReLU()]
        layers += [nn.Linear(hidden[-1], output_dim, bias=False)]
        self.net = nn.Sequential(*layers)

    def forward(self, x, stats):
        B = x.shape[0]
        x = self.series_conv(x)
        return self.net(torch.cat([x, stats], dim=1).view(B, -1))


class NonStationaryTransformer(nn.Module):
    def __init__(self, enc_in, c_out=1, seq_len=52, label_len=12, pred_len=12,
                 d_model=128, n_heads=4, e_layers=2, d_layers=1, d_ff=256,
                 dropout=0.1, p_hidden=(64,), n_time_feat=2):
        super().__init__()
        self.pred_len = pred_len
        self.label_len = label_len
        self.enc_emb = DataEmbedding(enc_in, d_model, n_time_feat, dropout)
        self.dec_emb = DataEmbedding(enc_in, d_model, n_time_feat, dropout)
        self.encoder = nn.ModuleList([_Block(d_model, n_heads, d_ff, dropout) for _ in range(e_layers)])
        self.enc_norm = nn.LayerNorm(d_model)
        self.decoder = nn.ModuleList([_Block(d_model, n_heads, d_ff, dropout, with_cross=True) for _ in range(d_layers)])
        self.dec_norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, c_out)
        self.tau_learner = Projector(enc_in, seq_len, list(p_hidden), output_dim=1)
        self.delta_learner = Projector(enc_in, seq_len, list(p_hidden), output_dim=seq_len)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        x_raw = x_enc.clone().detach()
        mu = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - mu
        sigma = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc = x_enc / sigma

        x_dec_new = torch.cat([x_enc[:, -self.label_len :, :],
                               torch.zeros_like(x_dec[:, -self.pred_len :, :])], dim=1).to(x_enc.device)

        tau = self.tau_learner(x_raw, sigma).exp()
        delta = self.delta_learner(x_raw, mu)

        enc_out = self.enc_emb(x_enc, x_mark_enc)
        for layer in self.encoder:
            enc_out = layer(enc_out, tau=tau, delta=delta)
        enc_out = self.enc_norm(enc_out)

        dec_out = self.dec_emb(x_dec_new, x_mark_dec)
        for layer in self.decoder:
            dec_out = layer(dec_out, cross=enc_out, tau=tau, delta=delta)
        dec_out = self.proj(self.dec_norm(dec_out))
        dec_out = dec_out * sigma + mu
        return dec_out[:, -self.pred_len :, :]
