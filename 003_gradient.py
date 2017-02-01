import numpy as np


def sigmoid(x):
    """
    Calculate sigmoid
    """
    return 1 / (1 + np.exp(-x))


learnrate = 0.5
x = np.array([1, 2])
y = np.array(0.5)

# Initial weights
w = np.array([0.5, -0.5])

# Calculate one gradient descent step for each weight

nn_output = sigmoid(np.dot(x, w))

error = y - nn_output

del_w = [learnrate * error * nn_output * (1 - nn_output) * 1,
         learnrate * error * nn_output * (1 - nn_output) * 2]

print('Neural Network output:')
print(nn_output)
print('Amount of Error:')
print(error)
print('Change in Weights:')
print(del_w)
