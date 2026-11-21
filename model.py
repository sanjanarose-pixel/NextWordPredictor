"""The decoder-only Transformer used by the original training scripts.

Keep these layer names and dimensions in sync with the training code: saved
PyTorch checkpoints use them to match weights to the model.
"""

import math

import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dimension, number_of_heads):
        super().__init__()
        if embedding_dimension % number_of_heads != 0:
            raise ValueError("Embedding dimension must be divisible by number of heads.")

        self.number_of_heads = number_of_heads
        self.head_dimension = embedding_dimension // number_of_heads
        self.query = nn.Linear(embedding_dimension, embedding_dimension)
        self.key = nn.Linear(embedding_dimension, embedding_dimension)
        self.value = nn.Linear(embedding_dimension, embedding_dimension)
        self.output_layer = nn.Linear(embedding_dimension, embedding_dimension)

    def forward(self, x):
        batch_size, sequence_length, embedding_dimension = x.shape
        q = self.query(x).view(
            batch_size, sequence_length, self.number_of_heads, self.head_dimension
        ).transpose(1, 2)
        k = self.key(x).view(
            batch_size, sequence_length, self.number_of_heads, self.head_dimension
        ).transpose(1, 2)
        v = self.value(x).view(
            batch_size, sequence_length, self.number_of_heads, self.head_dimension
        ).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dimension)
        future_mask = torch.triu(
            torch.ones(sequence_length, sequence_length, device=x.device), diagonal=1
        ).bool()
        scores = scores.masked_fill(future_mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        output = weights @ v
        output = output.transpose(1, 2).contiguous().view(
            batch_size, sequence_length, embedding_dimension
        )
        return self.output_layer(output)


class FeedForward(nn.Module):
    def __init__(self, embedding_dimension):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(embedding_dimension, embedding_dimension * 4),
            nn.ReLU(),
            nn.Linear(embedding_dimension * 4, embedding_dimension),
        )

    def forward(self, x):
        return self.network(x)


class TransformerBlock(nn.Module):
    def __init__(self, embedding_dimension, number_of_heads):
        super().__init__()
        self.attention = MultiHeadAttention(embedding_dimension, number_of_heads)
        self.feed_forward = FeedForward(embedding_dimension)
        self.norm1 = nn.LayerNorm(embedding_dimension)
        self.norm2 = nn.LayerNorm(embedding_dimension)

    def forward(self, x):
        x = self.norm1(x + self.attention(x))
        return self.norm2(x + self.feed_forward(x))


class LanguageModel(nn.Module):
    """Causal language model with the same parameter layout as saved weights."""

    def __init__(
        self,
        vocabulary_size,
        embedding_dimension,
        context_length,
        number_of_heads,
        number_of_blocks,
    ):
        super().__init__()
        self.context_length = context_length
        self.token_embedding = nn.Embedding(vocabulary_size, embedding_dimension)
        self.position_embedding = nn.Embedding(context_length, embedding_dimension)
        self.transformer_blocks = nn.Sequential(*[
            TransformerBlock(embedding_dimension, number_of_heads)
            for _ in range(number_of_blocks)
        ])
        self.final_norm = nn.LayerNorm(embedding_dimension)
        self.output_layer = nn.Linear(embedding_dimension, vocabulary_size)

    def forward(self, input_ids):
        _, sequence_length = input_ids.shape
        if sequence_length > self.context_length:
            raise ValueError("Input sequence is longer than the model context length.")
        positions = torch.arange(sequence_length, device=input_ids.device)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.transformer_blocks(x)
        return self.output_layer(self.final_norm(x))
