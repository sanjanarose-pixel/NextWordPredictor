# ============================================================
# NEXT WORD PREDICTOR
# A small decoder-only Transformer trained from scratch
# ============================================================

import json
import math
import re
import torch
import torch.nn as nn
from collections import Counter
from pathlib import Path

# ----------------------------
# 1. SETTINGS
# ----------------------------

TRAIN_FILE = "dataset.txt"
VALIDATION_FILE = "validation.txt"

VOCABULARY_LIMIT = 20000
MIN_FREQUENCY = 2

EMBEDDING_DIMENSION = 128
CONTEXT_LENGTH = 64
NUMBER_OF_HEADS = 4
NUMBER_OF_BLOCKS = 2

BATCH_SIZE = 4
LEARNING_RATE = 0.0003
EPOCHS = 5
STEPS_PER_EPOCH = 100
VALIDATION_BATCHES = 20

CHECKPOINT_FILE = "language_model_checkpoint.pth"
VOCAB_FILE = "vocabulary.json"

torch.manual_seed(42)
device = torch.device("cpu")
print("Using device:", device)


# ----------------------------
# 2. LOAD AND TOKENIZE DATA
# ----------------------------

def read_text(filename):
    path = Path(filename)
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {filename}. Keep it in the same folder as language_model.py."
        )
    return path.read_text(encoding="utf-8")


def tokenize(text):
    text = text.lower()
    text = text.replace("@-@", "-")
    text = text.replace("@,@", ",")
    text = text.replace("@.@", ".")
    return re.findall(r"\w+(?:['’]\w+)*|[^\w\s]", text, flags=re.UNICODE)


print("\nLoading datasets...")
training_text = read_text(TRAIN_FILE)
validation_text = read_text(VALIDATION_FILE)

training_tokens = tokenize(training_text)
validation_tokens = tokenize(validation_text)

print("Training tokens:", len(training_tokens))
print("Validation tokens:", len(validation_tokens))


# ----------------------------
# 3. BUILD VOCABULARY
#    Vocabulary is built ONLY from training data.
# ----------------------------

word_counts = Counter(training_tokens)

vocabulary = ["<unk>"]
for word, count in word_counts.most_common():
    if count < MIN_FREQUENCY:
        break
    if len(vocabulary) >= VOCABULARY_LIMIT:
        break
    vocabulary.append(word)

word_to_id = {word: index for index, word in enumerate(vocabulary)}
id_to_word = {index: word for word, index in word_to_id.items()}

def encode(tokens):
    unknown_id = word_to_id["<unk>"]
    return [word_to_id.get(token, unknown_id) for token in tokens]

def decode(token_ids):
    return [id_to_word.get(int(token_id), "<unk>") for token_id in token_ids]

training_ids = torch.tensor(encode(training_tokens), dtype=torch.long)
validation_ids = torch.tensor(encode(validation_tokens), dtype=torch.long)

vocabulary_size = len(vocabulary)

print("\nVOCABULARY")
print("----------")
print("Unique training words:", len(word_counts))
print("Vocabulary size:", vocabulary_size)


# ----------------------------
# 4. CREATE RANDOM TRAINING BATCHES
#    Each row is a context and its one-token-shifted target.
# ----------------------------

if len(training_ids) <= CONTEXT_LENGTH + 1:
    raise ValueError("Training text is too short for the selected context length.")

if len(validation_ids) <= CONTEXT_LENGTH + 1:
    raise ValueError("Validation text is too short for the selected context length.")

training_windows = training_ids.unfold(0, CONTEXT_LENGTH + 1, 1)
validation_windows = validation_ids.unfold(0, CONTEXT_LENGTH + 1, 1)

def get_training_batch():
    indices = torch.randint(0, len(training_windows), (BATCH_SIZE,))
    batch = training_windows[indices]
    inputs = batch[:, :-1].contiguous()
    targets = batch[:, 1:].contiguous()
    return inputs, targets

def get_validation_batch(batch_number):
    # Use different, consecutive validation windows for each batch.
    start = batch_number * BATCH_SIZE
    end = start + BATCH_SIZE
    if end > len(validation_windows):
        start = max(0, len(validation_windows) - BATCH_SIZE)
        end = start + BATCH_SIZE
    batch = validation_windows[start:end]
    inputs = batch[:, :-1].contiguous()
    targets = batch[:, 1:].contiguous()
    return inputs, targets


# ----------------------------
# 5. TRANSFORMER COMPONENTS
# ----------------------------

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

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        q = q.view(batch_size, sequence_length, self.number_of_heads, self.head_dimension)
        k = k.view(batch_size, sequence_length, self.number_of_heads, self.head_dimension)
        v = v.view(batch_size, sequence_length, self.number_of_heads, self.head_dimension)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = q @ k.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_dimension)

        # A token can attend to itself and earlier tokens, never future tokens.
        mask = torch.triu(
            torch.ones(sequence_length, sequence_length, device=x.device),
            diagonal=1
        ).bool()

        scores = scores.masked_fill(mask, float("-inf"))
        attention_weights = torch.softmax(scores, dim=-1)
        output = attention_weights @ v

        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, sequence_length, embedding_dimension)

        return self.output_layer(output)


class FeedForward(nn.Module):
    def __init__(self, embedding_dimension):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(embedding_dimension, embedding_dimension * 4),
            nn.ReLU(),
            nn.Linear(embedding_dimension * 4, embedding_dimension)
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
        x = self.norm2(x + self.feed_forward(x))
        return x


class LanguageModel(nn.Module):
    def __init__(
        self,
        vocabulary_size,
        embedding_dimension,
        context_length,
        number_of_heads,
        number_of_blocks
    ):
        super().__init__()
        self.context_length = context_length

        self.token_embedding = nn.Embedding(vocabulary_size, embedding_dimension)
        self.position_embedding = nn.Embedding(context_length, embedding_dimension)

        self.transformer_blocks = nn.Sequential(
            *[
                TransformerBlock(embedding_dimension, number_of_heads)
                for _ in range(number_of_blocks)
            ]
        )

        self.final_norm = nn.LayerNorm(embedding_dimension)
        self.output_layer = nn.Linear(embedding_dimension, vocabulary_size)

    def forward(self, input_ids):
        batch_size, sequence_length = input_ids.shape

        if sequence_length > self.context_length:
            raise ValueError("Input sequence is longer than the model context length.")

        token_vectors = self.token_embedding(input_ids)
        positions = torch.arange(sequence_length, device=input_ids.device)
        position_vectors = self.position_embedding(positions)

        x = token_vectors + position_vectors
        x = self.transformer_blocks(x)
        x = self.final_norm(x)

        return self.output_layer(x)


# ----------------------------
# 6. CREATE MODEL AND OPTIMIZER
# ----------------------------

model = LanguageModel(
    vocabulary_size=vocabulary_size,
    embedding_dimension=EMBEDDING_DIMENSION,
    context_length=CONTEXT_LENGTH,
    number_of_heads=NUMBER_OF_HEADS,
    number_of_blocks=NUMBER_OF_BLOCKS
).to(device)

parameter_count = sum(parameter.numel() for parameter in model.parameters())

print("\nMODEL")
print("-----")
print("Parameters:", f"{parameter_count:,}")
print("Context length:", CONTEXT_LENGTH)
print("Embedding dimension:", EMBEDDING_DIMENSION)
print("Transformer blocks:", NUMBER_OF_BLOCKS)
print("Attention heads:", NUMBER_OF_HEADS)

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)


# ----------------------------
# 7. TRAINING AND VALIDATION
# ----------------------------

best_validation_loss = float("inf")

print("\nSTARTING TRAINING")
print("-----------------")
print(f"Epochs: {EPOCHS} | Training steps per epoch: {STEPS_PER_EPOCH}")

for epoch in range(EPOCHS):
    model.train()
    total_training_loss = 0.0

    for step in range(STEPS_PER_EPOCH):
        inputs, targets = get_training_batch()
        inputs = inputs.to(device)
        targets = targets.to(device)

        predictions = model(inputs)
        loss = loss_function(
            predictions.reshape(-1, vocabulary_size),
            targets.reshape(-1)
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_training_loss += loss.item()

        if (step + 1) % 20 == 0:
            print(
                f"Epoch {epoch + 1}/{EPOCHS} | "
                f"Step {step + 1}/{STEPS_PER_EPOCH} | "
                f"Loss: {loss.item():.4f}"
            )

    average_training_loss = total_training_loss / STEPS_PER_EPOCH

    model.eval()
    total_validation_loss = 0.0

    with torch.no_grad():
        for batch_number in range(VALIDATION_BATCHES):
            inputs, targets = get_validation_batch(batch_number)
            inputs = inputs.to(device)
            targets = targets.to(device)

            predictions = model(inputs)
            loss = loss_function(
                predictions.reshape(-1, vocabulary_size),
                targets.reshape(-1)
            )
            total_validation_loss += loss.item()

    average_validation_loss = total_validation_loss / VALIDATION_BATCHES

    print(f"\nEpoch {epoch + 1} results")
    print(f"Average training loss:   {average_training_loss:.4f}")
    print(f"Average validation loss: {average_validation_loss:.4f}")

    # Save the best model seen so far.
    if average_validation_loss < best_validation_loss:
        best_validation_loss = average_validation_loss

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch + 1,
                "validation_loss": best_validation_loss,
                "config": {
                    "vocabulary_size": vocabulary_size,
                    "embedding_dimension": EMBEDDING_DIMENSION,
                    "context_length": CONTEXT_LENGTH,
                    "number_of_heads": NUMBER_OF_HEADS,
                    "number_of_blocks": NUMBER_OF_BLOCKS
                }
            },
            CHECKPOINT_FILE
        )

        with open(VOCAB_FILE, "w", encoding="utf-8") as file:
            json.dump(vocabulary, file, ensure_ascii=False)

        print("Best checkpoint saved.")

print("\nTRAINING FINISHED")
print("-----------------")
print(f"Best validation loss: {best_validation_loss:.4f}")


# ----------------------------
# 8. LOAD BEST CHECKPOINT
# ----------------------------

checkpoint = torch.load(CHECKPOINT_FILE, map_location=device, weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


# ----------------------------
# 9. TEXT GENERATION
# ----------------------------

def generate_text(prompt, number_of_tokens=30, context_window=32,
                  temperature=0.8, top_k=20):
    if temperature <= 0:
        raise ValueError("Temperature must be greater than zero.")

    if context_window < 1 or context_window > model.context_length:
        raise ValueError(
            f"Context window must be between 1 and {model.context_length}."
        )

    prompt_tokens = tokenize(prompt)
    generated_ids = encode(prompt_tokens)

    if not generated_ids:
        raise ValueError("Please enter a prompt containing at least one token.")

    with torch.no_grad():
        for _ in range(number_of_tokens):
            context = generated_ids[-context_window:]
            input_tensor = torch.tensor([context], dtype=torch.long, device=device)

            logits = model(input_tensor)[0, -1, :]
            logits = logits / temperature

            # Keep only the top-k most likely next tokens.
            if top_k is not None and 0 < top_k < vocabulary_size:
                top_values, _ = torch.topk(logits, top_k)
                cutoff = top_values[-1]
                logits[logits < cutoff] = float("-inf")

            probabilities = torch.softmax(logits, dim=-1)
            next_token_id = torch.multinomial(probabilities, num_samples=1).item()
            generated_ids.append(next_token_id)

    generated_tokens = decode(generated_ids)
    result = " ".join(generated_tokens)
    result = re.sub(r"\s+([.,!?;:%)\]])", r"\1", result)
    result = re.sub(r"([(])\s+", r"\1", result)
    return result


# ----------------------------
# 10. GENERATION TEST
# ----------------------------

test_prompt = "The history of science"

print("\nTEXT GENERATION TEST")
print("--------------------")
print("Prompt:", test_prompt)

generated_text = generate_text(
    test_prompt,
    number_of_tokens=30,
    context_window=32,
    temperature=0.8,
    top_k=20
)

print("Generated text:", generated_text)
