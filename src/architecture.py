"""A checkpoint-compatible document-level residual block, inspired by hierarchical encoders.

This is a research variant, not a claim of a novel hierarchical attention algorithm.
The token backbone is unchanged; the block sits before all task heads.
"""
import hashlib
import json
import torch
from torch import nn
import torch.nn.functional as F


def architecture_id(cfg):
    payload = {k: cfg.get(k, {}) for k in ("architecture_overrides", "document_bridge")}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


class DocumentBridge(nn.Module):
    def __init__(self, hidden_size, chunk_size=256, bottleneck=128, heads=4, mode="global"):
        super().__init__()
        if chunk_size < 1 or bottleneck < 1 or bottleneck % heads or mode not in {"global", "local"}:
            raise ValueError("Invalid document bridge configuration")
        self.chunk_size, self.mode = chunk_size, mode
        self.norm = nn.LayerNorm(hidden_size)
        self.down = nn.Linear(hidden_size, bottleneck, bias=False)
        self.attention = nn.MultiheadAttention(bottleneck, heads, dropout=0., batch_first=True)
        self.up = nn.Linear(bottleneck, hidden_size, bias=False)
        self.gate = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, hidden, attention_mask):
        batch, length, width = hidden.shape
        padding = (-length) % self.chunk_size
        mask = F.pad(attention_mask.bool(), (0, padding))
        normalized = F.pad(self.norm(hidden), (0, 0, 0, padding))
        chunks = normalized.reshape(batch, -1, self.chunk_size, width)
        weights = mask.reshape(batch, -1, self.chunk_size, 1).to(hidden.dtype)
        counts = weights.sum(2)
        summary = self.down((chunks * weights).sum(2) / counts.clamp_min(1))
        valid = counts.squeeze(-1) > 0
        # An empty row must not produce an all-masked attention softmax.
        safe_valid = valid.clone()
        safe_valid[:, 0] |= ~valid.any(-1)
        if self.mode == "local":
            # Parameter-matched control: each summary sees only itself.
            shape = summary.shape
            mixed, _ = self.attention(summary.reshape(-1, 1, shape[-1]),
                                      summary.reshape(-1, 1, shape[-1]),
                                      summary.reshape(-1, 1, shape[-1]), need_weights=False)
            mixed = mixed.reshape(shape)
        else:
            mixed, _ = self.attention(summary, summary, summary, key_padding_mask=~safe_valid,
                                      need_weights=False)
        update = self.up(F.gelu(summary + mixed)) * valid.unsqueeze(-1)
        update = update.repeat_interleave(self.chunk_size, dim=1)[:, :length]
        return hidden + torch.tanh(self.gate) * update * attention_mask.unsqueeze(-1)
