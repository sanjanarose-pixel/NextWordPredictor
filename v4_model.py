"""From-scratch V4 decoder-only Transformer with key padding masks."""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F


MODEL_CONFIG = {
    "vocab_size": 2048, "dimension": 64, "context_length": 64,
    "number_of_heads": 4, "number_of_blocks": 2, "dropout": 0.1,
}


class CausalAttention(nn.Module):
    def __init__(self, dimension: int, heads: int, dropout: float):
        super().__init__()
        if dimension % heads:
            raise ValueError("dimension must be divisible by heads")
        self.heads, self.head_dim = heads, dimension // heads
        self.qkv = nn.Linear(dimension, dimension * 3)
        self.proj = nn.Linear(dimension, dimension)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        b, t, d = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(b, t, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.heads, self.head_dim).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal_block = torch.triu(torch.ones(t, t, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal_block[None, None, :, :], torch.finfo(scores.dtype).min)
        if attention_mask is not None:
            if attention_mask.shape != (b, t):
                raise ValueError("attention_mask must have shape [batch, sequence]")
            scores = scores.masked_fill(~attention_mask[:, None, None, :].bool(), torch.finfo(scores.dtype).min)
        weights = self.dropout(torch.softmax(scores, dim=-1))
        y = (weights @ v).transpose(1, 2).contiguous().view(b, t, d)
        return self.dropout(self.proj(y))


class Block(nn.Module):
    def __init__(self, dimension: int, heads: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(dimension)
        self.attn = CausalAttention(dimension, heads, dropout)
        self.ln2 = nn.LayerNorm(dimension)
        self.ff = nn.Sequential(nn.Linear(dimension, 4 * dimension), nn.GELU(),
                                 nn.Linear(4 * dimension, dimension), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), attention_mask)
        return x + self.ff(self.ln2(x))


class V4LanguageModel(nn.Module):
    def __init__(self, vocab_size: int = MODEL_CONFIG["vocab_size"],
                 dimension: int = MODEL_CONFIG["dimension"],
                 context_length: int = MODEL_CONFIG["context_length"],
                 number_of_heads: int = MODEL_CONFIG["number_of_heads"],
                 number_of_blocks: int = MODEL_CONFIG["number_of_blocks"], dropout: float = 0.1,
                 pad_id: int = 0):
        super().__init__()
        self.config = {"vocab_size": vocab_size, "dimension": dimension,
                       "context_length": context_length, "number_of_heads": number_of_heads,
                       "number_of_blocks": number_of_blocks, "dropout": dropout, "pad_id": pad_id}
        self.context_length, self.pad_id = context_length, pad_id
        self.token_embedding = nn.Embedding(vocab_size, dimension, padding_idx=pad_id)
        self.position_embedding = nn.Embedding(context_length, dimension)
        self.embedding_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(Block(dimension, number_of_heads, dropout) for _ in range(number_of_blocks))
        self.final_norm = nn.LayerNorm(dimension)
        self.output_bias = nn.Parameter(torch.zeros(vocab_size))
        self.apply(self._initialize)
        # Padding row is deterministic zero; input/output weights remain tied.
        with torch.no_grad():
            self.token_embedding.weight[pad_id].zero_()

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        b, t = input_ids.shape
        if t > self.context_length:
            raise ValueError("Input exceeds V4 context length")
        if attention_mask is None:
            attention_mask = input_ids.ne(self.pad_id)
        pos = torch.arange(t, device=input_ids.device)
        x = self.embedding_dropout(self.token_embedding(input_ids) + self.position_embedding(pos))
        for block in self.blocks:
            x = block(x, attention_mask)
        x = self.final_norm(x)
        logits = F.linear(x, self.token_embedding.weight, self.output_bias)
        return logits


def masked_next_token_loss(logits: torch.Tensor, targets: torch.Tensor, pad_id: int) -> torch.Tensor:
    """Ignore padded target locations, while retaining EOS as a supervised target."""
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
                           ignore_index=pad_id)


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
