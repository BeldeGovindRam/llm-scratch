import numpy as np

#features[hpurs,studied, slept]
x = np.array([[2,5],[1,6],[3,4],[6,2],[7,3],[8,2]])
#output labels(0=fail, 1=pass)
y = np.array([[0],[0],[0],[1],[1],[1]])

np.random.seed(42)
weights=np.random.randn(2,1)
bias=0

def sigmoid(z):
    return 1 / (1+np.exp(-z))

#forward pass
z=np.dot(x, weights)+bias
predictions=sigmoid(z)    

def binary_cross_entropy(y_true, y_pred):
    epsilon = 1e-9
    return -np.mean(
        y_true * np.log(y_pred + epsilon) +
        (1 - y_true) * np.log(1 - y_pred + epsilon)
    )

loss = binary_cross_entropy(y, predictions)

print("\nLoss:")
print(loss)

print("weights")
print(weights)

print("\nbias")
print(bias)

print("\nRaw outputs z")
print(z)

print("\npredictions after sigmoid")
print(predictions)
