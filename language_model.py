
# Input
x = 2

# Weight and bias
weight = 2
bias = -5

# Neuron calculation
z = weight * x + bias

# Apply ReLU
if z > 0:
    output = z
else:
    output = 0

print("Before activation:", z)
print("After activation:", output)
