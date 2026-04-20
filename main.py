import numpy as np
# XOR NN from scratch
# 2:relu:4:sigmoid:1

lr = 0.5


# sigmoid function
def sigmoid(val):
    value = (np.exp(-val) + 1) ** -1
    return value  # sigmoid(0) = 0.5


# inputs:
x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

y = np.array([[0], [1], [1], [0]])

w1 = np.random.randn(2, 4) * 0.1
b1 = np.random.randn(1, 4) * 0.1
w2 = np.random.randn(4, 1) * 0.1
b2 = np.random.randn(1, 1) * 0.1

# print(w1, b1, w2, b2)

for i in range(10000):
    # forward pass
    z1 = np.dot(x, w1) + b1
    a1 = np.maximum(0, z1)
    z2 = np.dot(a1, w2) + b2
    a2 = sigmoid(z2)  # the prediction from the net

    print(f"random preds: {a2}")

    # backprop
    # derivative of mse = 2(a2 - y) over n -->(4)

    dL_da2 = (a2 - y) / 2
    da2_dz2 = a2 * (1 - a2)
    dL_dz2 = dL_da2 * da2_dz2
    dL_dw2 = np.dot(a1.T, dL_dz2)
    dL_db2 = np.sum(dL_dz2, axis=0, keepdims=True)
    dL_da1 = np.dot(dL_dz2, w2.T)
    dL_dz1 = dL_da1 * (z1 > 0)
    dL_dw1 = np.dot(x.T, dL_dz1)
    dL_db1 = np.sum(dL_dz1, axis=0, keepdims=True)

    w1 = w1 - lr * dL_dw1
    w2 = w2 - lr * dL_dw2
    b1 = b1 - lr * dL_db1
    b2 = b2 - lr * dL_db2

    loss = np.mean((a2 - y) ** 2)  # mse
    print(f"loss: {loss}")

    if i % 1000 == 0:
        print(f"loss: {loss}")
