
# import numpy as np

# #features[hpurs,studied, slept]
# x = np.array([[2,5],[1,6],[3,4],[6,2],[7,3],[8,2]])
# #output labels(0=fail, 1=pass)
# y = np.array([[0],[0],[0],[1],[1],[1]])

# np.random.seed(42)
# weights=np.random.randn(2,1)
# bias=0

# def sigmoid(z):
#     return 1 / (1+np.exp(-z))

# #forward pass
# z=np.dot(x, weights)+bias
# predictions=sigmoid(z)    

# def binary_cross_entropy(y_true, y_pred):
#     epsilon = 1e-9
#     return -np.mean(
#         y_true * np.log(y_pred + epsilon) +
#         (1 - y_true) * np.log(1 - y_pred + epsilon)
#     )

# loss = binary_cross_entropy(y, predictions)

# print("\nLoss:")
# print(loss)

# print("weights")
# print(weights)

# print("\nbias")
# print(bias)

# print("\nRaw outputs z")
# print(z)

# print("\npredictions after sigmoid")
# print(predictions)


import numpy as np

X = np.array([
    [2, 5],
    [1, 6],
    [3, 4],
    [6, 2],
    [7, 3],
    [8, 2]
])

y = np.array([
    [0],
    [0],
    [0],
    [1],
    [1],
    [1]
])

np.random.seed(42)
weights = np.random.randn(2, 1)
bias = 0
learning_rate = 0.01
epochs = 1000

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def binary_cross_entropy(y_true, y_pred):
    epsilon = 1e-9
    return -np.mean(
        y_true * np.log(y_pred + epsilon) +
        (1 - y_true) * np.log(1 - y_pred + epsilon)
    )

for epoch in range(epochs):
    # Forward pass
    z = np.dot(X, weights) + bias
    predictions = sigmoid(z)

    # Loss
    loss = binary_cross_entropy(y, predictions)

    # Backpropagation
    error = predictions - y

    dw = np.dot(X.T, error) / len(y)
    db = np.mean(error)

    # Gradient descent update
    weights = weights - learning_rate * dw
    bias = bias - learning_rate * db

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")

print("\nFinal weights:")
print(weights)

print("\nFinal bias:")
print(bias)

print("\nFinal predictions:")
print(predictions)