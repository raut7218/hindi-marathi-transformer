import torch
import torch.nn as nn
from .attention import GQAAttention
from .norms import RMSNorm


def make_pad_mask(attn_mask):
    if attn_mask is None:
        return None
    mask = attn_mask.unsqueeze(1).unsqueeze(2)
    return (1.0 - mask) * -1e9


def make_causal_mask(seq_len, device, dtype):
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    mask = mask * -1e9
    return mask.to(dtype)


class FeedForward(nn.Module):
    def __init__(self, d_model, ffn_mult, dropout):
        super().__init__()
        hidden = d_model * ffn_mult
        self.fc1 = nn.Linear(d_model, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x):
        return self.dropout(self.fc2(self.act(self.fc1(x))))


class EncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads, ffn_mult, dropout):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = GQAAttention(d_model, n_heads, n_kv_heads, dropout, use_rope=True)
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = FeedForward(d_model, ffn_mult, dropout)

    def forward(self, x, attn_mask=None):
        h = self.attn(self.attn_norm(x), attn_mask=attn_mask)
        x = x + h
        h = self.ffn(self.ffn_norm(x))
        x = x + h
        return x


class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads, ffn_mult, dropout):
        super().__init__()
        self.self_norm = RMSNorm(d_model)
        self.self_attn = GQAAttention(d_model, n_heads, n_kv_heads, dropout, use_rope=True)
        self.cross_norm = RMSNorm(d_model)
        self.cross_attn = GQAAttention(d_model, n_heads, n_kv_heads, dropout, use_rope=False)
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = FeedForward(d_model, ffn_mult, dropout)

    def forward(self, x, self_mask=None, enc_out=None, enc_mask=None):
        h = self.self_attn(self.self_norm(x), attn_mask=self_mask)
        x = x + h
        if enc_out is not None:
            h = self.cross_attn(self.cross_norm(x), attn_mask=enc_mask, kv=enc_out)
            x = x + h
        h = self.ffn(self.ffn_norm(x))
        x = x + h
        return x


class EncoderModel(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, n_kv_heads, ffn_mult, dropout, max_seq_len):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            EncoderBlock(d_model, n_heads, n_kv_heads, ffn_mult, dropout)
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.max_seq_len = max_seq_len
        self.tie_weights()

    def tie_weights(self):
        self.lm_head.weight = self.embed.weight

    def forward(self, input_ids, attn_mask=None):
        x = self.embed(input_ids)
        pad_mask = make_pad_mask(attn_mask)
        for layer in self.layers:
            x = layer(x, attn_mask=pad_mask)
        x = self.norm(x)
        return x

    def forward_mlm(self, input_ids, attn_mask=None):
        x = self.forward(input_ids, attn_mask=attn_mask)
        return self.lm_head(x)


class DecoderModel(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, n_kv_heads, ffn_mult, dropout, max_seq_len):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            DecoderBlock(d_model, n_heads, n_kv_heads, ffn_mult, dropout)
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.max_seq_len = max_seq_len
        self.tie_weights()

    def tie_weights(self):
        self.lm_head.weight = self.embed.weight

    def forward(self, input_ids, tgt_mask=None, enc_mask=None, enc_out=None):
        x = self.embed(input_ids)
        seq_len = x.size(1)
        causal = make_causal_mask(seq_len, x.device, x.dtype)
        causal = causal.unsqueeze(0).unsqueeze(0)
        pad_mask = make_pad_mask(tgt_mask)
        if pad_mask is not None:
            causal = causal + pad_mask
        for layer in self.layers:
            x = layer(x, self_mask=causal, enc_out=enc_out, enc_mask=enc_mask)
        x = self.norm(x)
        return self.lm_head(x)


class Seq2SeqModel(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, n_kv_heads, ffn_mult, dropout, max_seq_len):
        super().__init__()
        self.encoder = EncoderModel(vocab_size, d_model, n_layers, n_heads, n_kv_heads, ffn_mult, dropout, max_seq_len)
        self.decoder = DecoderModel(vocab_size, d_model, n_layers, n_heads, n_kv_heads, ffn_mult, dropout, max_seq_len)

    def forward(self, src_ids, src_mask, tgt_in, tgt_mask=None):
        enc_out = self.encoder(src_ids, attn_mask=src_mask)
        enc_mask = make_pad_mask(src_mask)
        return self.decoder(tgt_in, tgt_mask=tgt_mask, enc_mask=enc_mask, enc_out=enc_out)
