# ============================================================
# NEXT WORD PREDICTOR
# A small decoder-only Transformer trained from scratch
# ============================================================

import json
import math
import re
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn


# ----------------------------
# 1. SETTINGS
# ----------------------------

TRAIN_FILE = "dataset.txt"
VALIDATION_FILE = "validation.txt"
TEST_FILE = "test.txt"

VOCABULARY_LIMIT = 20000
MIN_FREQUENCY = 2

EMBEDDING_DIMENSION = 128
CONTEXT_LENGTH = 64
NUMBER_OF_HEADS = 4
NUMBER_OF_BLOCKS = 2

BATCH_SIZE = 4
LEARNING_RATE = 0.0003

# This stage adds five more epochs to the model you already trained.
ADDITIONAL_EPOCHS = 5
STEPS_PER_EPOCH = 500
VALIDATION_BATCHES = 20
TEST_BATCHES = 20

# Keep the original Stage 1 checkpoint safe.
STAGE1_CHECKPOINT = "language_model_checkpoint.pth"
LATEST_CHECKPOINT = "language_model_latest_stage2.pth"
BEST_CHECKPOINT = "language_model_best_stage2.pth"
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
            f"Could not find {filename}. Keep it in the same folder as this script."
        )
    return path.read_text(encoding="utf-8")


def tokenize(text):
    text = text.lower()
    text = text.replace("@-@", "-")
    text = text.replace("@,@", ",")
    text = text.replace("@.@", ".")
    return re.findall(
        r"\w+(?:['’]\w+)*|[^\w\s]",
        text,
        flags=re.UNICODE
    )


print("\nLoading datasets...")
training_text = read_text(TRAIN_FILE)
validation_text = read_text(VALIDATION_FILE)
test_text = read_text(TEST_FILE)

training_tokens = tokenize(training_text)
validation_tokens = tokenize(validation_text)
test_tokens = tokenize(test_text)

print("Training tokens:", len(training_tokens))
print("Validation tokens:", len(validation_tokens))
print("Test tokens:", len(test_tokens))


# ----------------------------
# 3. BUILD VOCABULARY
#    Only training data is used.
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
test_ids = torch.tensor(encode(test_tokens), dtype=torch.long)

vocabulary_size = len(vocabulary)

print("\nVOCABULARY")
print("----------")
print("Unique training words:", len(word_counts))
print("Vocabulary size:", vocabulary_size)


# ----------------------------
# 4. CREATE DATA WINDOWS
# ----------------------------

def make_windows(token_ids, split_name):
    if len(token_ids) <= CONTEXT_LENGTH:
        raise ValueError(f"{split_name} text is too short.")
    return token_ids.unfold(0, CONTEXT_LENGTH + 1, 1)


training_windows = make_windows(training_ids, "Training")
validation_windows = make_windows(validation_ids, "Validation")
test_windows = make_windows(test_ids, "Test")


def get_training_batch():
    indices = torch.randint(0, len(training_windows), (BATCH_SIZE,))
    batch = training_windows[indices]
    inputs = batch[:, :-1].contiguous()
    targets = batch[:, 1:].contiguous()
    return inputs, targets


def make_spread_indices(number_of_windows, number_of_examples):
    """Choose evenly spaced windows across the complete split."""
    if number_of_examples > number_of_windows:
        raise ValueError("Requested more samples than available windows.")
    return torch.linspace(
        0, number_of_windows - 1, steps=number_of_examples
    ).long()


validation_indices = make_spread_indices(
    len(validation_windows), VALIDATION_BATCHES * BATCH_SIZE
).view(VALIDATION_BATCHES, BATCH_SIZE)

test_indices = make_spread_indices(
    len(test_windows), TEST_BATCHES * BATCH_SIZE
).view(TEST_BATCHES, BATCH_SIZE)


def get_evaluation_batch(windows, indices, batch_number):
    batch = windows[indices[batch_number]]
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
            raise ValueError(
                "Embedding dimension must be divisible by number of heads."
            )

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

        q = q.view(
            batch_size, sequence_length,
            self.number_of_heads, self.head_dimension
        ).transpose(1, 2)

        k = k.view(
            batch_size, sequence_length,
            self.number_of_heads, self.head_dimension
        ).transpose(1, 2)

        v = v.view(
            batch_size, sequence_length,
            self.number_of_heads, self.head_dimension
        ).transpose(1, 2)

        scores = q @ k.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_dimension)

        # Causal mask: each position can see itself and earlier positions only.
        mask = torch.triu(
            torch.ones(sequence_length, sequence_length, device=x.device),
            diagonal=1
        ).bool()

        scores = scores.masked_fill(mask, float("-inf"))
        attention_weights = torch.softmax(scores, dim=-1)
        output = attention_weights @ v

        output = output.transpose(1, 2).contiguous()
        output = output.view(
            batch_size, sequence_length, embedding_dimension
        )

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
        self.attention = MultiHeadAttention(
            embedding_dimension, number_of_heads
        )
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

        self.token_embedding = nn.Embedding(
            vocabulary_size, embedding_dimension
        )
        self.position_embedding = nn.Embedding(
            context_length, embedding_dimension
        )

        self.transformer_blocks = nn.Sequential(*[
            TransformerBlock(embedding_dimension, number_of_heads)
            for _ in range(number_of_blocks)
        ])

        self.final_norm = nn.LayerNorm(embedding_dimension)
        self.output_layer = nn.Linear(
            embedding_dimension, vocabulary_size
        )

    def forward(self, input_ids):
        batch_size, sequence_length = input_ids.shape

        if sequence_length > self.context_length:
            raise ValueError(
                "Input sequence is longer than the model context length."
            )

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

parameter_count = sum(
    parameter.numel() for parameter in model.parameters()
)

print("\nMODEL")
print("-----")
print("Parameters:", f"{parameter_count:,}")
print("Context length:", CONTEXT_LENGTH)
print("Embedding dimension:", EMBEDDING_DIMENSION)
print("Transformer blocks:", NUMBER_OF_BLOCKS)
print("Attention heads:", NUMBER_OF_HEADS)

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

model_config = {
    "vocabulary_size": vocabulary_size,
    "embedding_dimension": EMBEDDING_DIMENSION,
    "context_length": CONTEXT_LENGTH,
    "number_of_heads": NUMBER_OF_HEADS,
    "number_of_blocks": NUMBER_OF_BLOCKS
}


def save_checkpoint(filename, completed_epochs, best_loss):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "completed_epochs": completed_epochs,
            "best_validation_loss": best_loss,
            "config": model_config
        },
        filename
    )


def load_checkpoint(filename):
    checkpoint = torch.load(
        filename, map_location=device, weights_only=False
    )

    if checkpoint["config"] != model_config:
        raise ValueError(
            "Checkpoint architecture or vocabulary does not match this script."
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


# ----------------------------
# 7. EVALUATION
# ----------------------------

def evaluate(windows, indices, number_of_batches):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch_number in range(number_of_batches):
            inputs, targets = get_evaluation_batch(
                windows, indices, batch_number
            )
            inputs = inputs.to(device)
            targets = targets.to(device)

            predictions = model(inputs)
            loss = loss_function(
                predictions.reshape(-1, vocabulary_size),
                targets.reshape(-1)
            )
            total_loss += loss.item()

    return total_loss / number_of_batches


# ----------------------------
# 8. RESUME FROM EXISTING MODEL
# ----------------------------

print("\nLOADING PREVIOUS MODEL")
print("----------------------")

if Path(LATEST_CHECKPOINT).exists():
    checkpoint = load_checkpoint(LATEST_CHECKPOINT)
    completed_epochs = checkpoint["completed_epochs"]
    best_validation_loss = checkpoint["best_validation_loss"]
    print("Resuming Stage 2 from its latest checkpoint.")
    print("Stage 2 epochs already completed:", completed_epochs)

elif Path(STAGE1_CHECKPOINT).exists():
    checkpoint = load_checkpoint(STAGE1_CHECKPOINT)
    completed_epochs = 0
    print("Loaded your Stage 1 trained model.")
    print("Stage 1 epoch:", checkpoint.get("epoch", "unknown"))

    # Measure the old model using the new, evenly-spaced validation sample.
    best_validation_loss = evaluate(
        validation_windows, validation_indices, VALIDATION_BATCHES
    )
    print(
        "Stage 2 validation baseline:",
        f"{best_validation_loss:.4f}"
    )

    # Save this starting model as the initial Stage 2 best model.
    save_checkpoint(
        BEST_CHECKPOINT, completed_epochs, best_validation_loss
    )
else:
    raise FileNotFoundError(
        "Could not find the Stage 1 checkpoint. "
        f"Expected {STAGE1_CHECKPOINT} in this folder."
    )

with open(VOCAB_FILE, "w", encoding="utf-8") as file:
    json.dump(vocabulary, file, ensure_ascii=False)


# ----------------------------
# 9. CONTINUE TRAINING
# ----------------------------

print("\nSTARTING STAGE 2 TRAINING")
print("-------------------------")
print(
    f"Additional epochs: {ADDITIONAL_EPOCHS} | "
    f"Steps per epoch: {STEPS_PER_EPOCH}"
)

if completed_epochs >= ADDITIONAL_EPOCHS:
    print("Stage 2 has already completed the requested number of epochs.")
else:
    for epoch in range(completed_epochs, ADDITIONAL_EPOCHS):
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
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0
            )
            optimizer.step()

            total_training_loss += loss.item()

            if (step + 1) % 50 == 0:
                print(
                    f"Stage epoch {epoch + 1}/{ADDITIONAL_EPOCHS} | "
                    f"Step {step + 1}/{STEPS_PER_EPOCH} | "
                    f"Loss: {loss.item():.4f}"
                )

        average_training_loss = total_training_loss / STEPS_PER_EPOCH

        average_validation_loss = evaluate(
            validation_windows, validation_indices, VALIDATION_BATCHES
        )

        print(f"\nStage epoch {epoch + 1} results")
        print(f"Average training loss:   {average_training_loss:.4f}")
        print(f"Average validation loss: {average_validation_loss:.4f}")

        completed_epochs = epoch + 1

        # Always save the latest state so an interrupted run can resume.
        save_checkpoint(
            LATEST_CHECKPOINT,
            completed_epochs,
            best_validation_loss
        )
        print("Latest checkpoint saved.")

        # Save a separate copy of the best model.
        if average_validation_loss < best_validation_loss:
            best_validation_loss = average_validation_loss
            save_checkpoint(
                BEST_CHECKPOINT,
                completed_epochs,
                best_validation_loss
            )
            print("New best model saved.")

        # Update latest checkpoint with the current best score.
        save_checkpoint(
            LATEST_CHECKPOINT,
            completed_epochs,
            best_validation_loss
        )

print("\nSTAGE 2 TRAINING FINISHED")
print("-------------------------")
print("Best Stage 2 validation loss:", f"{best_validation_loss:.4f}")


# ----------------------------
# 10. LOAD BEST STAGE 2 MODEL
# ----------------------------

best_checkpoint = torch.load(
    BEST_CHECKPOINT, map_location=device, weights_only=False
)
model.load_state_dict(best_checkpoint["model_state_dict"])
model.eval()


# ----------------------------
# 11. TEST-SET EVALUATION
# ----------------------------

print("\nTEST-SET EVALUATION")
print("-------------------")

test_loss = evaluate(test_windows, test_indices, TEST_BATCHES)
print("Test loss:", f"{test_loss:.4f}")
print(
    "Note: test data is used only for final evaluation, "
    "not for training or model selection."
)


# ----------------------------
# 12. TEXT GENERATION
# ----------------------------

def generate_text(
    prompt,
    number_of_tokens=30,
    context_window=32,
    temperature=0.8,
    top_k=20
):
    if temperature <= 0:
        raise ValueError("Temperature must be greater than zero.")

    if context_window < 1 or context_window > model.context_length:
        raise ValueError(
            f"Context window must be between 1 and {model.context_length}."
        )

    if number_of_tokens < 0:
        raise ValueError("Number of tokens cannot be negative.")

    prompt_tokens = tokenize(prompt)
    generated_ids = encode(prompt_tokens)

    if not generated_ids:
        raise ValueError("Please enter a prompt containing at least one token.")

    model.eval()

    with torch.no_grad():
        for _ in range(number_of_tokens):
            context = generated_ids[-context_window:]
            input_tensor = torch.tensor(
                [context], dtype=torch.long, device=device
            )

            logits = model(input_tensor)[0, -1, :]
            logits = logits / temperature

            if top_k is not None and 0 < top_k < vocabulary_size:
                top_values, _ = torch.topk(logits, top_k)
                cutoff = top_values[-1]
                logits[logits < cutoff] = float("-inf")

            probabilities = torch.softmax(logits, dim=-1)
            next_token_id = torch.multinomial(
                probabilities, num_samples=1
            ).item()

            generated_ids.append(next_token_id)

    generated_tokens = decode(generated_ids)
    result = " ".join(generated_tokens)
    result = re.sub(r"\s+([.,!?;:%)\]])", r"\1", result)
    result = re.sub(r"([(])\s+", r"\1", result)
    return result


# ----------------------------
# 13. GENERATION TESTS
# ----------------------------

print("\nTEXT GENERATION TESTS")
print("---------------------")

prompts = [
    "The history of science",
    "In the United States",
    "The main character"
]

for prompt in prompts:
    print("\nPrompt:", prompt)
    print(
        "Generated:",
        generate_text(
            prompt,
            number_of_tokens=30,
            context_window=32,
            temperature=0.8,
            top_k=20
        )
    )

print("\nAll done.")
