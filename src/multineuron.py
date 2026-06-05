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

#=========================
# TRAINING LOOP
# =========================

for epoch in range(epochs):

    # Forward pass: input → hidden layer
    Z1 = np.dot(X, W1) + b1
    A1 = relu(Z1)

    # Forward pass: hidden layer → output layer
    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)

    # Calculate loss
    loss = binary_cross_entropy(y, A2)

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.6f}")
    
    # =========================
    # BACKPROPAGATION
    # =========================

    m = len(y)

    dZ2 = A2 - y
    dW2 = np.dot(A1.T, dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m

    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * (Z1 > 0)

    dW1 = np.dot(X.T, dZ1) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m

    # =========================
    # UPDATE PARAMETERS
    # =========================

    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1

    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

# =========================
# FINAL OUTPUT
# =========================

print("\nFinal Predictions:")
print(A2)

print("\nActual Labels:")
print(y)

print("\nFinal Probabilities:")
print(A2)

final_classes = (A2 >= 0.5).astype(int)
accuracy = np.mean(final_classes == y) * 100

print("\nAccuracy:")
print(f"{accuracy:.2f}%")

print("\nFinal Predicted Classes:")
print(final_classes)

print("\nActual Labels:")
print(y)


def predict(student):
    Z1 = np.dot(student, W1) + b1
    A1 = relu(Z1)

    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)

    predicted_class = (A2 >= 0.5).astype(int)

    return A2, predicted_class


new_student = np.array([[5, 3]])

probability, predicted_class = predict(new_student)

print("\nNew Student:")
print(new_student)

print("\nProbability of Passing:")
print(probability)

print("\nPredicted Class:")
print(predicted_class)

if predicted_class[0][0] == 1:
    print("\nResult: PASS")
else:
    print("\nResult: FAIL")

print("\nW1 Shape:", W1.shape)
print("b1 Shape:", b1.shape)

print("\nW2 Shape:", W2.shape)
print("b2 Shape:", b2.shape)