
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