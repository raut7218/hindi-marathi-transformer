import torch
import torch.nn as nn
from .attention import GQAAttention
from .norms import RMSNorm


def make_pad_mask(attn_mask):
    if attn_mask is None:
        return None
    mask = attn_mask.unsqueeze(1).unsqueeze(2)
    return (1.0 - mask.float()) * -1e4


def make_causal_mask(seq_len, device, dtype):
    mask = torch.full((seq_len, seq_len), 0.0, device=device, dtype=dtype)
    mask = mask.masked_fill(torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1), -1e4)
    return mask


class FeedForward(nn.Module):
    def __init__(self, d_model, ffn_dim, dropout):
        super().__init__()
        hidden = ffn_dim
        self.fc1 = nn.Linear(d_model, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x):
        return self.dropout(self.fc2(self.act(self.fc1(x))))


class EncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads, ffn_dim, dropout):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = GQAAttention(d_model, n_heads, n_kv_heads, dropout, use_rope=True)
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = FeedForward(d_model, ffn_dim, dropout)

    def forward(self, x, attn_mask=None):
        h = self.attn(self.attn_norm(x), attn_mask=attn_mask)
        x = x + h
        h = self.ffn(self.ffn_norm(x))
        x = x + h
        return x


class GPTDecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads, ffn_dim, dropout):
        super().__init__()
        self.self_norm = RMSNorm(d_model)
        self.self_attn = GQAAttention(d_model, n_heads, n_kv_heads, dropout, use_rope=True)
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = FeedForward(d_model, ffn_dim, dropout)

    def forward(self, x, self_mask=None):
        h = self.self_attn(self.self_norm(x), attn_mask=self_mask)
        x = x + h
        h = self.ffn(self.ffn_norm(x))
        x = x + h
        return x


class MTDecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads, ffn_dim, dropout):
        super().__init__()
        self.self_norm = RMSNorm(d_model)
        self.self_attn = GQAAttention(d_model, n_heads, n_kv_heads, dropout, use_rope=True)
        self.cross_norm = RMSNorm(d_model)
        self.cross_attn = GQAAttention(d_model, n_heads, n_kv_heads, dropout, use_rope=False)
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = FeedForward(d_model, ffn_dim, dropout)

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
    def __init__(
        self,
        vocab_size,
        d_model,
        n_layers,
        n_heads,
        n_kv_heads,
        ffn_dim=None,
        dropout=0.1,
        max_seq_len=512,
        ffn_mult=None,
    ):
        super().__init__()
        if ffn_dim is None:
            ffn_dim = d_model * (ffn_mult if ffn_mult is not None else 4)
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            EncoderBlock(d_model, n_heads, n_kv_heads, ffn_dim, dropout)
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.max_seq_len = max_seq_len
        self.tie_weights()

    def tie_weights(self):
        self.lm_head.weight = self.embed.weight

    def forward(self, input_ids, attn_mask=None, return_logits=False):
        x = self.embed(input_ids)
        pad_mask = make_pad_mask(attn_mask)
        for layer in self.layers:
            x = layer(x, attn_mask=pad_mask)
        x = self.norm(x)
        if return_logits:
            return self.lm_head(x)
        return x

    def forward_mlm(self, input_ids, attn_mask=None):
        return self.forward(input_ids, attn_mask=attn_mask, return_logits=True)


class DecoderModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model,
        n_layers,
        n_heads,
        n_kv_heads,
        ffn_dim=None,
        dropout=0.1,
        max_seq_len=1024,
        ffn_mult=None,
    ):
        super().__init__()
        if ffn_dim is None:
            ffn_dim = d_model * (ffn_mult if ffn_mult is not None else 4)
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            GPTDecoderBlock(d_model, n_heads, n_kv_heads, ffn_dim, dropout)
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
            x = layer(x, self_mask=causal)
        x = self.norm(x)
        return self.lm_head(x)


class MTDecoderModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model,
        n_layers,
        n_heads,
        n_kv_heads,
        ffn_dim=None,
        dropout=0.1,
        max_seq_len=512,
        ffn_mult=None,
    ):
        super().__init__()
        if ffn_dim is None:
            ffn_dim = d_model * (ffn_mult if ffn_mult is not None else 4)
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            MTDecoderBlock(d_model, n_heads, n_kv_heads, ffn_dim, dropout)
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
    def __init__(
        self,
        vocab_size=None,
        d_model=768,
        n_layers=12,
        n_heads=12,
        n_kv_heads=4,
        ffn_dim=None,
        dropout=0.1,
        max_seq_len=512,
        ffn_mult=None,
        src_vocab_size=None,
        tgt_vocab_size=None,
        encoder_config=None,
        decoder_config=None,
    ):
        super().__init__()
        if encoder_config is None:
            src_vocab_size = src_vocab_size or vocab_size
            encoder_config = {
                "vocab_size": src_vocab_size,
                "d_model": d_model,
                "n_layers": n_layers,
                "n_heads": n_heads,
                "n_kv_heads": n_kv_heads,
                "ffn_dim": ffn_dim,
                "ffn_mult": ffn_mult,
                "dropout": dropout,
                "max_seq_len": max_seq_len,
            }
        if decoder_config is None:
            tgt_vocab_size = tgt_vocab_size or vocab_size
            decoder_config = {
                "vocab_size": tgt_vocab_size,
                "d_model": d_model,
                "n_layers": n_layers,
                "n_heads": n_heads,
                "n_kv_heads": n_kv_heads,
                "ffn_dim": ffn_dim,
                "ffn_mult": ffn_mult,
                "dropout": dropout,
                "max_seq_len": max_seq_len,
            }
        self.encoder = EncoderModel(**encoder_config)
        self.decoder = MTDecoderModel(**decoder_config)

    def forward(self, src_ids, src_mask, tgt_in, tgt_mask=None):
        enc_out = self.encoder(src_ids, attn_mask=src_mask)
        enc_mask = make_pad_mask(src_mask)
        return self.decoder(tgt_in, tgt_mask=tgt_mask, enc_mask=enc_mask, enc_out=enc_out)
