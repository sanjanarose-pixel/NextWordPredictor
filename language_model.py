
# Input and target
x = 2
target = 10

# Initial parameters
weight = 1
bias = 0

# Learning rate
learning_rate = 0.1

# FORWARD PROPAGATION
prediction = weight * x + bias

# Calculate loss
loss = (prediction - target) ** 2

print("Prediction:", prediction)
print("Loss:", loss)

# BACKPROPAGATION
weight_gradient = 2 * (prediction - target) * x
bias_gradient = 2 * (prediction - target)

print("Weight gradient:", weight_gradient)
print("Bias gradient:", bias_gradient)

# GRADIENT DESCENT
weight = weight - learning_rate * weight_gradient
bias = bias - learning_rate * bias_gradient

# New prediction
new_prediction = weight * x + bias
new_loss = (new_prediction - target) ** 2

print("Updated weight:", weight)
print("Updated bias:", bias)
print("New prediction:", new_prediction)
print("New loss:", new_loss)
