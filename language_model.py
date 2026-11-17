
dataset = [
    "The robot moved across the room",
    "The robot moved across the floor",
    "The robot walked into the room"
]

input_text = input("Enter two words: ").lower()
context = input_text.split()

next_words = []

for sentence in dataset:
    words = sentence.lower().split()

    for i in range(len(words) - len(context)):
        if words[i:i + len(context)] == context:
            next_words.append(words[i + len(context)])

word_counts = {}

for word in next_words:
    if word in word_counts:
        word_counts[word] += 1
    else:
        word_counts[word] = 1

total = len(next_words)

if total > 0:
    for word in word_counts:
        probability = word_counts[word] / total
        print(word, ":", round(probability * 100, 2), "%")
else:
    print("No matching context found.")
