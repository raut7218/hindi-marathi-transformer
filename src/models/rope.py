import torch
import torch.nn as nn


def apply_rope(x, cos, sin):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    rot_x1 = x1 * cos - x2 * sin
    rot_x2 = x1 * sin + x2 * cos
    out = torch.stack([rot_x1, rot_x2], dim=-1)
    return out.flatten(-2)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None

    def get_cos_sin(self, seq_len, device, dtype):
        if (
            self._cos_cached is not None
            and self._sin_cached is not None
            and self._seq_len_cached >= seq_len
            and self._cos_cached.device == device
            and self._cos_cached.dtype == dtype
        ):
            return self._cos_cached[:seq_len], self._sin_cached[:seq_len]

        inv_freq = self.inv_freq.to(device=device)
        t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        cos = freqs.cos().to(dtype)
        sin = freqs.sin().to(dtype)
        self._seq_len_cached = seq_len
        self._cos_cached = cos
        self._sin_cached = sin
        return cos, sin
