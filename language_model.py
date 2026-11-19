
# Training dataset
training_inputs = [
    [2, 3],
    [4, 3],
    [5, 2],
    [1, 4]
]

training_targets = [10, 14, 12, 9]

# Initial parameters
weight1 = 1.0
weight2 = 1.0
bias = 0.0

learning_rate = 0.01

# Train for 100 epochs
for epoch in range(100):

    total_gradient_w1 = 0
    total_gradient_w2 = 0
    total_gradient_b = 0

    total_loss = 0

    # Go through all training examples
    for i in range(4):

        x1 = training_inputs[i][0]
        x2 = training_inputs[i][1]

        target = training_targets[i]

        # Forward propagation
        prediction = weight1 * x1 + weight2 * x2 + bias

        # Calculate loss
        error = prediction - target
        loss = error ** 2

        total_loss += loss

        # Backpropagation
        gradient_w1 = 2 * error * x1
        gradient_w2 = 2 * error * x2
        gradient_b = 2 * error

        # Accumulate gradients
        total_gradient_w1 += gradient_w1
        total_gradient_w2 += gradient_w2
        total_gradient_b += gradient_b

    # Calculate average gradients
    average_gradient_w1 = total_gradient_w1 / 4
    average_gradient_w2 = total_gradient_w2 / 4
    average_gradient_b = total_gradient_b / 4

    # Update parameters
    weight1 -= learning_rate * average_gradient_w1
    weight2 -= learning_rate * average_gradient_w2
    bias -= learning_rate * average_gradient_b

    # Calculate average loss
    average_loss = total_loss / 4

    if epoch % 10 == 0:
        print("Epoch:", epoch + 1)
        print("Average loss:", average_loss)
        print()

print("Final weights:", weight1, weight2)
print("Final bias:", bias)
