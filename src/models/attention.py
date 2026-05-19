import torch
import torch.nn as nn
import torch.nn.functional as F
from .rope import apply_rope, RotaryEmbedding


class GQAAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads, dropout, use_rope=True):
        super().__init__()
        assert n_heads % n_kv_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        assert self.head_dim * n_heads == d_model

        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.use_rope = use_rope
        self.rope = RotaryEmbedding(self.head_dim) if use_rope else None

    def _shape(self, x, n_heads):
        b, t, _ = x.size()
        return x.view(b, t, n_heads, self.head_dim).transpose(1, 2)

    def forward(self, x, attn_mask=None, kv=None):
        if kv is None:
            kv = x
        q = self.q_proj(x)
        k = self.k_proj(kv)
        v = self.v_proj(kv)

        q = self._shape(q, self.n_heads)
        k = self._shape(k, self.n_kv_heads)
        v = self._shape(v, self.n_kv_heads)

        if self.use_rope:
            cos, sin = self.rope.get_cos_sin(q.size(-2), q.device, q.dtype)
            cos = cos[None, None, :, :]
            sin = sin[None, None, :, :]
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

        if self.n_heads != self.n_kv_heads:
            repeat = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)

        attn_scores = torch.matmul(q, k.transpose(-2, -1))
        attn_scores = attn_scores / (self.head_dim ** 0.5)

        if attn_mask is not None:
            attn_scores = attn_scores + attn_mask

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        out = torch.matmul(attn_probs, v)
        out = out.transpose(1, 2).contiguous().view(x.size(0), x.size(1), self.d_model)
        return self.out_proj(out)
