import numpy as np
# XOR NN from scratch
# 2:relu:4:sigmoid:1

lr = 0.85


# sigmoid function
def sigmoid(val):
    value = (np.exp(-val) + 1) ** -1
    return value  # sigmoid(0) = 0.5

# leaky ReLU func
def LRELU(val, alpha = 0.015):
    value = np.maximum(alpha*val, val)
    return value

# leaky ReLU deriv func
def DLRELU(val, alpha = 0.015):
    value = np.where(val>0, 1, alpha)
    return value

# inputs:
x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

y = np.array([[0], [1], [1], [0]])

w1 = np.random.randn(2, 4) * 0.1
b1 = np.random.randn(1, 4) * 0.1
w2 = np.random.randn(4, 1) * 0.1
b2 = np.random.randn(1, 1) * 0.1

# print(w1, b1, w2, b2)

for i in range(250000):
    # forward pass
    z1 = np.dot(x, w1) + b1
#    a1 = np.maximum(0, z1) # ReLU
    a1 = LRELU(z1) # LRELU
    z2 = np.dot(a1, w2) + b2
    a2 = sigmoid(z2)  # the prediction from the net

    print(f"preds: {a2}")

    # backprop
    # derivative of mse = 2(a2 - y) over n -->(4)

    dL_da2 = (a2 - y) / 2
    da2_dz2 = a2 * (1 - a2)
    dL_dz2 = dL_da2 * da2_dz2
    dL_dw2 = np.dot(a1.T, dL_dz2)
    dL_db2 = np.sum(dL_dz2, axis=0, keepdims=True)
    dL_da1 = np.dot(dL_dz2, w2.T)
 #   dL_dz1 = dL_da1 * (z1 > 0) # relu
    dL_dz1 = dL_da1 * DLRELU(z1) # DLRELU
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

x1 = np.linspace(0, 1, 100)
x2 = np.linspace(0, 1, 100)

xvals = np.meshgrid(x1, x2)

xvals = np.column_stack((xvals[0].ravel(), xvals[1].ravel()))

z1 = np.dot(xvals, w1) + b1
a1 = np.maximum(0, z1) # ReLU
z2 = np.dot(a1, w2) + b2
a2 = sigmoid(z2)  # the prediction from the net

print(f"graph preds: {a2}")

"""
    loss = np.mean((a2 - y) ** 2)  # mse
    print(f"loss: {loss}")

    if i % 1000 == 0:
        print(f"loss: {loss}")
"""


output = a2.reshape(100, 100)
# print(output.shape)
import matplotlib.pyplot as plt
plt.contourf(x1, x2, output, levels=50, cmap='RdPu')
plt.colorbar()
plt.scatter([0,0,1,1], [0,1,0,1], c=[0,1,1,0], cmap='RdPu', edgecolors='black', s=100)
plt.show()