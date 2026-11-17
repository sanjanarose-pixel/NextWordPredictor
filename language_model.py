
# Input and expected output
x = 2
target = 10

# Initial weight and bias
weight = 1
bias = 0

# Learning rate
learning_rate = 0.5

# Make a prediction
prediction = weight * x + bias

# Calculate error
error = prediction - target

print("Prediction:", prediction)
print("Target:", target)
print("Error:", error)

# Update weight
weight = weight - learning_rate * error * x

# Update bias
bias = bias - learning_rate * error

# Make another prediction
new_prediction = weight * x + bias

print("Updated weight:", weight)
print("Updated bias:", bias)
print("New prediction:", new_prediction)
