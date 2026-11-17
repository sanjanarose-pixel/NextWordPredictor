
dataset = [
    "The robot moved across the room",
    "The robot walked into the room",
    "The machine moved across the floor"
]

vocabulary = []

for sentence in dataset:
    words = sentence.lower().split()

    for word in words:
        if word not in vocabulary:
            vocabulary.append(word)

word_to_id = {}

for index, word in enumerate(vocabulary):
    word_to_id[word] = index

print(word_to_id)

print("Robot ID:", word_to_id["robot"])

input_sequences = []
target_words = []

for sentence in dataset:
    words = sentence.lower().split()

    for i in range(1, len(words)):
        context = words[:i]
        target = words[i]

        input_ids = []

        for word in context:
            input_ids.append(word_to_id[word])

        target_id = word_to_id[target]

        input_sequences.append(input_ids)
        target_words.append(target_id)

print("Input sequences:", input_sequences)
print("Target words:", target_words)
