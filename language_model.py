
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