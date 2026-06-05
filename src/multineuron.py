import numpy as np

# =========================
# DATASET
# =========================

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

# =========================
# SETUP
# =========================

np.random.seed(42)

input_neurons = 2
hidden_neurons = 4
output_neurons = 1

learning_rate = 0.01
epochs = 1000

# =========================
# INITIALIZE PARAMETERS
# =========================

W1 = np.random.randn(input_neurons, hidden_neurons)
b1 = np.zeros((1, hidden_neurons))

W2 = np.random.randn(hidden_neurons, output_neurons)
b2 = np.zeros((1, output_neurons))

print("W1 shape:", W1.shape)
print("b1 shape:", b1.shape)
print("W2 shape:", W2.shape)
print("b2 shape:", b2.shape)

# =========================
# ACTIVATION FUNCTIONS
# =========================

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def relu(z):
    return np.maximum(0, z)


# =========================
# FORWARD PASS
# =========================

Z1 = np.dot(X, W1) + b1
A1 = relu(Z1)

Z2 = np.dot(A1, W2) + b2
A2 = sigmoid(Z2)

print("\nZ1 shape:", Z1.shape)
print("A1 shape:", A1.shape)
print("Z2 shape:", Z2.shape)
print("A2 shape:", A2.shape)

print("\nFinal predictions:")
print(A2)

# =========================
# LOSS FUNCTION
# =========================

def binary_cross_entropy(y_true, y_pred):
    epsilon = 1e-9

    return -np.mean(
        y_true * np.log(y_pred + epsilon) +
        (1 - y_true) * np.log(1 - y_pred + epsilon)
    )

loss = binary_cross_entropy(y, A2)

print("\nLoss:")
print(loss)