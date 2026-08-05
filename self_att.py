import numpy as np

x = np.array([
    [1,0],
    [0,1],
    [1,1]
],dtype=float)

print(x)

Q = x
k = x
v = x

scores = Q @ k.T

print(scores)


def softmax(x):

    exp = np.exp(x-np.max(x, axis=1, keepdims=True))

    return exp / exp.sum(axis=1, keepdims=True)

attention = softmax(scores)
print(attention)

output = attention @ v
print(output)