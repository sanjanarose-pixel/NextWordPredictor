"""A compact decoder-only Transformer for the new from-scratch model."""

import math

import torch
import torch.nn as nn


class CausalAttention(nn.Module):
    def __init__(self, dimension, number_of_heads, dropout):
        super().__init__()
        if dimension % number_of_heads:
            raise ValueError("Model dimension must divide evenly across attention heads.")
        self.number_of_heads = number_of_heads
        self.head_dimension = dimension // number_of_heads
        self.qkv = nn.Linear(dimension, dimension * 3)
        self.output = nn.Linear(dimension, dimension)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, sequence_length, dimension = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(batch_size, sequence_length, self.number_of_heads, self.head_dimension).transpose(1, 2)
        k = k.view(batch_size, sequence_length, self.number_of_heads, self.head_dimension).transpose(1, 2)
        v = v.view(batch_size, sequence_length, self.number_of_heads, self.head_dimension).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dimension)
        future_positions = torch.triu(
            torch.ones(sequence_length, sequence_length, device=x.device, dtype=torch.bool),
            diagonal=1,
        )
        scores = scores.masked_fill(future_positions, float("-inf"))
        weights = self.dropout(torch.softmax(scores, dim=-1))
        result = weights @ v
        result = result.transpose(1, 2).contiguous().view(batch_size, sequence_length, dimension)
        return self.dropout(self.output(result))


class TransformerBlock(nn.Module):
    """Pre-normalized attention and feed-forward layers improve training stability."""

    def __init__(self, dimension, number_of_heads, dropout):
        super().__init__()
        self.attention_norm = nn.LayerNorm(dimension)
        self.attention = CausalAttention(dimension, number_of_heads, dropout)
        self.feed_forward_norm = nn.LayerNorm(dimension)
        self.feed_forward = nn.Sequential(
            nn.Linear(dimension, dimension * 4),
            nn.GELU(),
            nn.Linear(dimension * 4, dimension),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attention(self.attention_norm(x))
        return x + self.feed_forward(self.feed_forward_norm(x))


class SmallLanguageModel(nn.Module):
    """A GPT-style decoder with tied input and output token embeddings."""

    def __init__(self, vocabulary_size, dimension=192, context_length=128,
                 number_of_heads=4, number_of_blocks=4, dropout=0.1):
        super().__init__()
        self.context_length = context_length
        self.token_embedding = nn.Embedding(vocabulary_size, dimension)
        self.position_embedding = nn.Embedding(context_length, dimension)
        self.embedding_dropout = nn.Dropout(dropout)
        self.blocks = nn.Sequential(*[
            TransformerBlock(dimension, number_of_heads, dropout)
            for _ in range(number_of_blocks)
        ])
        self.final_norm = nn.LayerNorm(dimension)
        self.output_bias = nn.Parameter(torch.zeros(vocabulary_size))
        self.apply(self._initialize_weights)

    @staticmethod
    def _initialize_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids):
        _, sequence_length = input_ids.shape
        if sequence_length > self.context_length:
            raise ValueError("Input is longer than the model context window.")
        positions = torch.arange(sequence_length, device=input_ids.device)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.embedding_dropout(x)
        x = self.blocks(x)
        x = self.final_norm(x)
        return torch.nn.functional.linear(x, self.token_embedding.weight, self.output_bias)
