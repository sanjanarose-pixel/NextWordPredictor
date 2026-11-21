
# ==========================================
# NEXT WORD PREDICTOR
# Step 1: Dataset and vocabulary preparation
# ==========================================

# Training dataset
dataset = [
    "The robot moved across the room",
    "The robot walked into the room",
    "The machine moved across the floor"
]


# STEP 1: Tokenization

tokenized_sentences = []

for sentence in dataset:
    words = sentence.lower().split()
    tokenized_sentences.append(words)


# STEP 2: Create vocabulary

vocabulary = []

for sentence in tokenized_sentences:
    for word in sentence:

        if word not in vocabulary:
            vocabulary.append(word)


# STEP 3: Assign IDs to words

word_to_id = {}

for i in range(len(vocabulary)):
    word_to_id[vocabulary[i]] = i


id_to_word = {}

for word in word_to_id:
    word_id = word_to_id[word]
    id_to_word[word_id] = word


# STEP 4: Generate training examples
# Use two words to predict the next word

training_inputs = []
training_targets = []

for sentence in tokenized_sentences:

    for i in range(len(sentence) - 2):

        first_word = sentence[i]
        second_word = sentence[i + 1]
        next_word = sentence[i + 2]

        first_id = word_to_id[first_word]
        second_id = word_to_id[second_word]
        target_id = word_to_id[next_word]

        training_inputs.append([first_id, second_id])
        training_targets.append(target_id)


# STEP 5: Display results

print("Vocabulary:", vocabulary)
print("Word to ID:", word_to_id)
print("ID to Word:", id_to_word)

print("\nTraining inputs:", training_inputs)
print("Training targets:", training_targets)

print("\nTotal vocabulary size:", len(vocabulary))
print("Total training examples:", len(training_inputs))


# ==========================================
# STEP 2: EMBEDDING LAYER
# ==========================================

import torch

# Number of unique words
vocabulary_size = len(vocabulary)

# Number of values representing each word
embedding_dimension = 4

# Create embedding layer
embedding_layer = torch.nn.Embedding(
    vocabulary_size,
    embedding_dimension
)

# Display embedding matrix
print("\nEmbedding matrix:")
print(embedding_layer.weight)

# Display matrix shape
print("\nMatrix shape:")
print(embedding_layer.weight.shape)

# Get embedding of "robot"
robot_id = word_to_id["robot"]

robot_embedding = embedding_layer(
    torch.tensor(robot_id)
)

print("\nRobot ID:", robot_id)
print("Robot embedding:", robot_embedding)
# Get the IDs of the two input words
first_word_id = word_to_id["the"]
second_word_id = word_to_id["robot"]

# Convert both IDs into tensors
first_word_tensor = torch.tensor(first_word_id)
second_word_tensor = torch.tensor(second_word_id)

# Get their embeddings
first_embedding = embedding_layer(first_word_tensor)
second_embedding = embedding_layer(second_word_tensor)

# Join both embeddings
combined_embedding = torch.cat(
    (first_embedding, second_embedding)
)

print("\nFirst word embedding:")
print(first_embedding)

print("\nSecond word embedding:")
print(second_embedding)

print("\nCombined embedding:")
print(combined_embedding)

print("\nCombined embedding shape:")
print(combined_embedding.shape)

# Neural network model

class NextWordModel(torch.nn.Module):

    def __init__(self, vocabulary_size, embedding_dimension):
        super().__init__()

        # Convert word IDs into vectors
        self.embedding = torch.nn.Embedding(
            vocabulary_size,
            embedding_dimension
        )

        # Hidden layer: 8 inputs -> 16 neurons
        self.hidden = torch.nn.Linear(
            embedding_dimension * 2,
            16
        )

        # Output layer: 16 neurons -> 9 scores
        self.output = torch.nn.Linear(
            16,
            vocabulary_size
        )

    def forward(self, input_words):

        # Get embeddings for both words
        embeddings = self.embedding(input_words)

        # Join the two embeddings
        combined = embeddings.reshape(-1)

        # Hidden layer
        hidden_output = self.hidden(combined)

        # ReLU activation
        hidden_output = torch.relu(hidden_output)

        # Output scores
        scores = self.output(hidden_output)

        return scores


# Create the model
model = NextWordModel(
    vocabulary_size,
    embedding_dimension
)

# Test the model
test_input = torch.tensor([0, 1])

scores = model(test_input)

print("\nModel output scores:")
print(scores)

print("\nOutput shape:")
print(scores.shape)
# Convert training data into PyTorch tensors

inputs = torch.tensor(training_inputs, dtype=torch.long)
targets = torch.tensor(training_targets, dtype=torch.long)

# Loss function
loss_function = torch.nn.CrossEntropyLoss()

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training loop
epochs = 500

for epoch in range(epochs):

    # Get predictions for all training examples
    predictions = []

    for input_words in inputs:
        scores = model(input_words)
        predictions.append(scores)

    predictions = torch.stack(predictions)

    # Calculate loss
    loss = loss_function(predictions, targets)

    # Clear old gradients
    optimizer.zero_grad()

    # Backpropagation
    loss.backward()

    # Update model weights
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")
# Test prediction

test_input = torch.tensor([0, 1])

with torch.no_grad():
    scores = model(test_input)
    probabilities = torch.softmax(scores, dim=0)

    predicted_id = torch.argmax(probabilities).item()

print("\nInput: the robot")
print("Predicted next word:", id_to_word[predicted_id])
print("Probabilities:", probabilities)
# Generate multiple words

def generate_text(starting_text, number_of_words):

    words = starting_text.lower().split()

    # Keep the last two words as context
    for _ in range(number_of_words):

        # Convert the last two words into IDs
        input_ids = [
            word_to_id[words[-2]],
            word_to_id[words[-1]]
        ]

        input_tensor = torch.tensor(input_ids)

        # Predict the next word
        with torch.no_grad():
            scores = model(input_tensor)
            predicted_id = torch.argmax(scores).item()

        # Convert ID back into a word
        predicted_word = id_to_word[predicted_id]

        # Add the predicted word
        words.append(predicted_word)

    return " ".join(words)


# Test text generation

print("\nGenerated text:")
print(generate_text("the robot", 4))

import re
from collections import Counter


# STEP 1: Read the dataset

with open("dataset.txt", "r", encoding="utf-8") as file:
    dataset_text = file.read()


# STEP 2: Convert text into tokens

def tokenize(text):

    text = text.lower()

    # Clean special WikiText formatting
    text = text.replace("@-@", "-")
    text = text.replace("@,@", ",")
    text = text.replace("@.@", ".")

    # Separate words and punctuation
    tokens = re.findall(
        r"\w+(?:['’]\w+)*|[^\w\s]",
        text,
        flags=re.UNICODE
    )

    return tokens


tokens = tokenize(dataset_text)


# STEP 3: Count word frequencies

word_counts = Counter(tokens)


# STEP 4: Build vocabulary


min_frequency = 2
max_vocab_size = 20000

vocabulary = ["<unk>"]

for word, count in word_counts.most_common():

    if count < min_frequency:
        break

    if len(vocabulary) >= max_vocab_size:
        break

    vocabulary.append(word)



# STEP 5: Assign numerical IDs

word_to_id = {
    word: index
    for index, word in enumerate(vocabulary)
}

id_to_word = {
    index: word
    for word, index in word_to_id.items()
}


# STEP 6: Convert tokens into numerical IDs

def encode(words):

    return [
        word_to_id.get(word, word_to_id["<unk>"])
        for word in words
    ]


# STEP 7: Convert numerical IDs back into words

def decode(token_ids):

    return [
        id_to_word.get(token_id, "<unk>")
        for token_id in token_ids
    ]


# STEP 8: Display results

encoded_sample = encode(tokens[:20])

print("\nTOKENIZER RESULTS")
print("------------------------")

print("Total characters:", len(dataset_text))
print("Total tokens:", len(tokens))
print("Unique words:", len(word_counts))
print("Vocabulary size:", len(vocabulary))

print("\nOriginal tokens:")
print(tokens[:20])

print("\nNumerical representation:")
print(encoded_sample)

print("\nDecoded representation:")
print(decode(encoded_sample))

import torch
from torch.utils.data import Dataset


# STEP 9: Load all three datasets

with open("validation.txt", "r", encoding="utf-8") as file:
    validation_text = file.read()

with open("test.txt", "r", encoding="utf-8") as file:
    test_text = file.read()


# STEP 10: Tokenize and encode each dataset

training_ids = encode(tokenize(dataset_text))
validation_ids = encode(tokenize(validation_text))
test_ids = encode(tokenize(test_text))


# STEP 11: Define the context window

context_length = 64


# STEP 12: Create a dataset of input-target pairs

class NextTokenDataset(Dataset):

    def __init__(self, token_ids, context_length):
        self.token_ids = token_ids
        self.context_length = context_length

    def __len__(self):
        return len(self.token_ids) - self.context_length

    def __getitem__(self, index):

        input_sequence = self.token_ids[
            index:index + self.context_length
        ]

        target_sequence = self.token_ids[
            index + 1:index + self.context_length + 1
        ]

        return (
            torch.tensor(input_sequence, dtype=torch.long),
            torch.tensor(target_sequence, dtype=torch.long)
        )


# STEP 13: Create training, validation and test datasets

train_dataset = NextTokenDataset(
    training_ids, context_length
)

validation_dataset = NextTokenDataset(
    validation_ids, context_length
)

test_dataset = NextTokenDataset(
    test_ids, context_length
)


# STEP 14: Display results

print("\nSEQUENCE PREPARATION RESULTS")
print("-----------------------------")

print("Training tokens:", len(training_ids))
print("Validation tokens:", len(validation_ids))
print("Test tokens:", len(test_ids))

print("Training examples:", len(train_dataset))
print("Validation examples:", len(validation_dataset))
print("Test examples:", len(test_dataset))

sample_input, sample_target = train_dataset[0]

print("\nFirst input sequence:")
print(decode(sample_input.tolist()))

print("\nFirst target sequence:")
print(decode(sample_target.tolist()))

# ==========================================
# TRANSFORMER - EMBEDDING LAYER
# ==========================================

import torch

# Model configuration
vocabulary_size = len(vocabulary)
embedding_dimension = 128
model_context_length = 64

# Create token embeddings
token_embedding = torch.nn.Embedding(
    vocabulary_size,
    embedding_dimension
)

# Create positional embeddings
position_embedding = torch.nn.Embedding(
    model_context_length,
    embedding_dimension
)

# Get one training example
sample_input, sample_target = train_dataset[0]

# Use the first 64 tokens
sample_input = sample_input[:model_context_length]

# Convert token IDs into vectors
token_vectors = token_embedding(sample_input)

# Create position numbers: 0, 1, 2, ... 63
positions = torch.arange(model_context_length)

# Convert positions into vectors
position_vectors = position_embedding(positions)

# Combine both embeddings
input_embeddings = token_vectors + position_vectors

print("\nEMBEDDING RESULTS")
print("-----------------")
print("Vocabulary size:", vocabulary_size)
print("Embedding dimension:", embedding_dimension)
print("Context length:", model_context_length)

print("Token embedding shape:", token_vectors.shape)
print("Position embedding shape:", position_vectors.shape)
print("Final embedding shape:", input_embeddings.shape)


# ==========================================
# TRANSFORMER - MULTI HEAD ATTENTION
# ==========================================

import math

class MultiHeadAttention(torch.nn.Module):

    def __init__(self, embedding_dimension, number_of_heads):

        super().__init__()

        self.number_of_heads = number_of_heads
        self.head_dimension = embedding_dimension // number_of_heads

        # Query, Key and Value layers
        self.query = torch.nn.Linear(
            embedding_dimension, embedding_dimension
        )

        self.key = torch.nn.Linear(
            embedding_dimension, embedding_dimension
        )

        self.value = torch.nn.Linear(
            embedding_dimension, embedding_dimension
        )

        # Combine the outputs of all heads
        self.output_layer = torch.nn.Linear(
            embedding_dimension, embedding_dimension
        )

    def forward(self, x):

        batch_size, sequence_length, embedding_dimension = x.shape

        # Generate Query, Key and Value
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # Divide features among attention heads
        Q = Q.view(
            batch_size, sequence_length,
            self.number_of_heads, self.head_dimension
        )

        K = K.view(
            batch_size, sequence_length,
            self.number_of_heads, self.head_dimension
        )

        V = V.view(
            batch_size, sequence_length,
            self.number_of_heads, self.head_dimension
        )

        # Move heads before sequence dimension
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Calculate attention scores
        scores = Q @ K.transpose(-2, -1)

        scores = scores / math.sqrt(self.head_dimension)

        # Causal mask
        mask = torch.triu(
            torch.ones(sequence_length, sequence_length),
            diagonal=1
        ).bool()

        scores = scores.masked_fill(mask, float("-inf"))

        # Attention probabilities
        attention_weights = torch.softmax(scores, dim=-1)

        # Combine values
        output = attention_weights @ V

        # Join all attention heads
        output = output.transpose(1, 2).contiguous()

        output = output.view(
            batch_size, sequence_length, embedding_dimension
        )

        # Final projection
        output = self.output_layer(output)

        return output


# Create multi-head attention
attention_layer = MultiHeadAttention(
    embedding_dimension=128,
    number_of_heads=4
)

# Test it
sample_embeddings = input_embeddings.unsqueeze(0)

attention_output = attention_layer(sample_embeddings)

print("\nMULTI HEAD ATTENTION RESULTS")
print("----------------------------")
print("Number of heads:", 4)
print("Features per head:", 32)
print("Input shape:", sample_embeddings.shape)
print("Output shape:", attention_output.shape)


# ==========================================
# TRANSFORMER - FEED FORWARD NETWORK
# ==========================================

class FeedForward(torch.nn.Module):

    def __init__(self, embedding_dimension):

        super().__init__()

        self.network = torch.nn.Sequential(

            # Expand 128 features to 512
            torch.nn.Linear(
                embedding_dimension,
                embedding_dimension * 4
            ),

            # Activation function
            torch.nn.ReLU(),

            # Reduce 512 features back to 128
            torch.nn.Linear(
                embedding_dimension * 4,
                embedding_dimension
            )
        )

    def forward(self, x):

        return self.network(x)


# Create feed-forward network
feed_forward = FeedForward(embedding_dimension=128)

# Pass attention output through it
feed_forward_output = feed_forward(attention_output)

print("\nFEED FORWARD RESULTS")
print("--------------------")
print("Input shape:", attention_output.shape)
print("Output shape:", feed_forward_output.shape)

# ==========================================
# TRANSFORMER - COMPLETE BLOCK
# ==========================================

class TransformerBlock(torch.nn.Module):

    def __init__(self, embedding_dimension, number_of_heads):

        super().__init__()

        # Multi-head attention
        self.attention = MultiHeadAttention(
            embedding_dimension,
            number_of_heads
        )

        # Feed-forward network
        self.feed_forward = FeedForward(
            embedding_dimension
        )

        # Layer normalization
        self.norm1 = torch.nn.LayerNorm(
            embedding_dimension
        )

        self.norm2 = torch.nn.LayerNorm(
            embedding_dimension
        )

    def forward(self, x):

        # Attention and residual connection
        attention_output = self.attention(x)

        x = self.norm1(x + attention_output)

        # Feed-forward and residual connection
        feed_forward_output = self.feed_forward(x)

        x = self.norm2(x + feed_forward_output)

        return x


# Create one Transformer block
transformer_block = TransformerBlock(
    embedding_dimension=128,
    number_of_heads=4
)

# Test the block
transformer_output = transformer_block(sample_embeddings)

print("\nTRANSFORMER BLOCK RESULTS")
print("-------------------------")
print("Input shape:", sample_embeddings.shape)
print("Output shape:", transformer_output.shape)

# ==========================================
# COMPLETE TRANSFORMER LANGUAGE MODEL
# ==========================================

class LanguageModel(torch.nn.Module):

    def __init__(self, vocabulary_size, embedding_dimension,
                 context_length, number_of_heads, number_of_blocks):

        super().__init__()

        self.context_length = context_length

        self.token_embedding = torch.nn.Embedding(
            vocabulary_size, embedding_dimension
        )

        self.position_embedding = torch.nn.Embedding(
            context_length, embedding_dimension
        )

        self.transformer_blocks = torch.nn.Sequential(
            *[
                TransformerBlock(embedding_dimension, number_of_heads)
                for _ in range(number_of_blocks)
            ]
        )

        self.final_norm = torch.nn.LayerNorm(embedding_dimension)

        self.output_layer = torch.nn.Linear(
            embedding_dimension, vocabulary_size
        )

    def forward(self, input_ids):

        batch_size, sequence_length = input_ids.shape

        # Token embeddings
        token_vectors = self.token_embedding(input_ids)

        # Positional embeddings
        positions = torch.arange(
            sequence_length, device=input_ids.device
        )

        position_vectors = self.position_embedding(positions)

        # Combine embeddings
        x = token_vectors + position_vectors

        # Transformer blocks
        x = self.transformer_blocks(x)

        # Final normalization
        x = self.final_norm(x)

        # Predict scores for every vocabulary token
        logits = self.output_layer(x)

        return logits


# ==========================================
# TEST COMPLETE MODEL
# ==========================================

model = LanguageModel(
    vocabulary_size=20000,
    embedding_dimension=128,
    context_length=64,
    number_of_heads=4,
    number_of_blocks=2
)

sample_ids = torch.tensor(
    [train_dataset[0][0][:64].tolist()]
)

predictions = model(sample_ids)

print("\nCOMPLETE LANGUAGE MODEL")
print("-----------------------")
print("Input shape:", sample_ids.shape)
print("Prediction shape:", predictions.shape)
print("Total parameters:", sum(p.numel() for p in model.parameters()))


# ==========================================
# TRAINING THE LANGUAGE MODEL
# ==========================================

from torch.utils.data import DataLoader

# Training settings
batch_size = 4
learning_rate = 0.0003
training_batches = 20
validation_batches = 5

# Create data loaders
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=batch_size,
    shuffle=False
)

# Loss function
loss_function = torch.nn.CrossEntropyLoss()

# Optimizer
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate
)

# ==========================================
# TRAINING
# ==========================================

model.train()
total_loss = 0

for batch_number, (inputs, targets) in enumerate(train_loader):

    # Forward pass
    predictions = model(inputs)

    # Calculate loss
    loss = loss_function(
        predictions.reshape(-1, vocabulary_size),
        targets.reshape(-1)
    )

    # Clear previous gradients
    optimizer.zero_grad()

    # Backpropagation
    loss.backward()

    # Prevent excessively large gradients
    torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_norm=1.0
    )

    # Update weights
    optimizer.step()

    total_loss += loss.item()

    if (batch_number + 1) % 5 == 0:
        print(
            f"Training batch {batch_number + 1}/{training_batches}, "
            f"Loss: {loss.item():.4f}"
        )

    if batch_number + 1 >= training_batches:
        break

average_training_loss = total_loss / training_batches

print("\nTRAINING COMPLETED")
print("------------------")
print(f"Average training loss: {average_training_loss:.4f}")

# ==========================================
# VALIDATION
# ==========================================

model.eval()
total_validation_loss = 0

with torch.no_grad():

    for batch_number, (inputs, targets) in enumerate(validation_loader):

        predictions = model(inputs)

        loss = loss_function(
            predictions.reshape(-1, vocabulary_size),
            targets.reshape(-1)
        )

        total_validation_loss += loss.item()

        if batch_number + 1 >= validation_batches:
            break

average_validation_loss = (
    total_validation_loss / validation_batches
)

print("\nVALIDATION RESULTS")
print("------------------")
print(f"Average validation loss: {average_validation_loss:.4f}")

# ==========================================
# SAVE TRAINED MODEL
# ==========================================

torch.save(model.state_dict(), "language_model.pth")

print("\nModel weights saved successfully!")

import re

def generate_text(prompt, number_of_words=20, context_window=32, temperature=1.0):
    model.eval()

    if context_window < 1 or context_window > model.context_length:
        raise ValueError(f"Context window must be between 1 and {model.context_length}.")

    prompt_tokens = tokenize(prompt)
    input_ids = encode(prompt_tokens)

    if not input_ids:
        raise ValueError("Please enter a prompt.")

    generated_ids = input_ids.copy()

    with torch.no_grad():
        for _ in range(number_of_words):
            context = generated_ids[-context_window:]

            input_tensor = torch.tensor([context], dtype=torch.long)

            predictions = model(input_tensor)

            next_token_logits = predictions[0, -1, :]

            next_token_logits = next_token_logits / temperature

            probabilities = torch.softmax(next_token_logits, dim=-1)

            next_token_id = torch.multinomial(probabilities, num_samples=1).item()

            generated_ids.append(next_token_id)

    generated_tokens = decode(generated_ids)

    result = " ".join(generated_tokens)

    result = re.sub(r"\s+([.,!?;:%)\]])", r"\1", result)
    result = re.sub(r"([(])\s+", r"\1", result)

    return result


print("\nTEXT GENERATION")
print("----------------")

prompt = "The history of science"

result = generate_text(
    prompt,
    number_of_words=20,
    context_window=32
)

print("Prompt:", prompt)
print("Generated text:", result)
