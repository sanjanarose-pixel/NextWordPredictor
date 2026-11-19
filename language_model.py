
# Two input values
inputs = [2, 3]

# Three neurons in the hidden layer
weights = [
    [1.0, 0.2],
    [0.3, 0.8],
    [0.6, 0.4]
]

biases = [1, -1, 0.5]


# ReLU activation function
def relu(value):
    if value > 0:
        return value
    else:
        return 0


# Calculate outputs of hidden neurons
hidden_outputs = []

for i in range(3):

    # Weighted calculation
    total = 0

    for j in range(2):
        total = total + inputs[j] * weights[i][j]

    total = total + biases[i]

    # Apply activation
    output = relu(total)

    hidden_outputs.append(output)


print("Hidden layer outputs:", hidden_outputs)
# Output neuron weights
output_weights = [0.4, 0.7, 0.2]

# Output neuron bias
output_bias = 0.5

# Calculate final output
final_output = 0

for i in range(3):
    final_output += hidden_outputs[i] * output_weights[i]

final_output += output_bias

print("Final network output:", final_output)