
# Input and expected output
x = 2
target = 10

# Initial weight and bias
weight = 1
bias = 0

# Learning rate
learning_rate = 0.1

# Train for 10 steps
for step in range(10):

    # Make prediction
    prediction = weight * x + bias

    # Calculate error
    error = prediction - target

    # Update weight and bias
    weight = weight - learning_rate * error * x
    bias = bias - learning_rate * error

    # Display progress
    print("Step:", step + 1)
    print("Prediction:", round(prediction, 2))
    print("Error:", round(error, 2))
    print()

# Final prediction
final_prediction = weight * x + bias

print("Final prediction:", round(final_prediction, 2))
print("Actual target:", target)
