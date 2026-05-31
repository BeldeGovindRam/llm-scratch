
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
# INITIALIZE PARAMETERS
# =========================

np.random.seed(42)

weights = np.random.randn(2, 1)
bias = 0

learning_rate = 0.01
epochs = 1000

# =========================
# ACTIVATION FUNCTION
# =========================

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# =========================
# LOSS FUNCTION
# =========================

def binary_cross_entropy(y_true, y_pred):
    epsilon = 1e-9

    return -np.mean(
        y_true * np.log(y_pred + epsilon) +
        (1 - y_true) * np.log(1 - y_pred + epsilon)
    )

# =========================
# TRAINING LOOP
# =========================

for epoch in range(epochs):

    # Forward Pass
    z = np.dot(X, weights) + bias
    predictions = sigmoid(z)

    # Calculate Loss
    loss = binary_cross_entropy(y, predictions)

    # Backpropagation
    error = predictions - y

    dw = np.dot(X.T, error) / len(y)
    db = np.mean(error)

    # Gradient Descent Update
    weights = weights - learning_rate * dw
    bias = bias - learning_rate * db

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.6f}")

# =========================
# RESULTS
# =========================

print("\nFinal Weights:")
print(weights)

print("\nFinal Bias:")
print(bias)

print("\nPredictions On Training Data:")
print(predictions)

# =========================
# PREDICTION FUNCTION
# =========================

def predict(student):

    z = np.dot(student, weights) + bias
    probability = sigmoid(z)

    if probability >= 0.5:
        prediction = 1
    else:
        prediction = 0

    return probability, prediction

# =========================
# TEST NEW STUDENT
# =========================

new_student = np.array([5, 3])

probability, prediction = predict(new_student)

print("\n=========================")
print("NEW STUDENT")
print("=========================")

print("Features:")
print(f"Hours Studied = {new_student[0]}")
print(f"Hours Slept   = {new_student[1]}")

print("\nProbability Of Passing:")
print(probability)

print("\nPredicted Class:")
print(prediction)

if prediction == 1:
    print("\nResult: PASS")
else:
    print("\nResult: FAIL")