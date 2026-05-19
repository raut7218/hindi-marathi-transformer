import torch


def apply_rope(x, cos, sin):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    rot_x1 = x1 * cos - x2 * sin
    rot_x2 = x1 * sin + x2 * cos
    out = torch.stack([rot_x1, rot_x2], dim=-1)
    return out.flatten(-2)


class RotaryEmbedding:
    def __init__(self, dim, base=10000):
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.inv_freq = inv_freq

    def get_cos_sin(self, seq_len, device, dtype):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        cos = freqs.cos().to(dtype)
        sin = freqs.sin().to(dtype)
        return cos, sin
