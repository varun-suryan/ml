import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score
from sklearn.datasets import make_blobs
from sklearn.inspection import DecisionBoundaryDisplay

matplotlib.use('TkAgg')

epochs = 10000
X, y = make_blobs(n_samples=500, centers=[[-1, -1], [1, 1]], cluster_std=0.5, random_state=0)

plt.scatter(X[:, 0], X[:, 1], c=y)
plt.title("Two normally-distributed clusters")
plt.show(block=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
y_train, y_test = y_train[:, np.newaxis], y_test[:, np.newaxis]

n_train, learning_rate = X_train.shape[0], 1e-1

[w1, w2] = map(float, np.random.rand(2, 1))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


losses_train, losses_val, accs_train, accs_val = [], [], [], []
for epoch in range(epochs):
    # Forward pass
    a = np.matmul(X_train, np.array([[w1], [w2]])) + 1  # n*2 * 2*1 = n*1
    y_hat = sigmoid(a)
    derivative_a = np.multiply(y_hat, 1 - y_hat)

    # Backward pass
    diff = np.subtract(y_train, y_hat)  # n*1
    del_w2 = -np.dot(np.multiply(diff, derivative_a).T, X_train[:, [1]]) / n_train  # divide by n
    del_w1 = -np.dot(np.multiply(diff, derivative_a).T, X_train[:, [0]]) / n_train

    w1 -= learning_rate * del_w1[0][0]
    w2 -= learning_rate * del_w2[0][0]

    if epoch % 500 == 0:
        print("Epoch {}".format(epoch))
        train_loss = log_loss(y_train, y_hat)
        accuracy_train = accuracy_score(y_train, np.round(y_hat))
        print("Training loss: {}.".format(train_loss) + " Accuracy {}.".format(accuracy_train))
        a = np.matmul(X_test, [w1, w2]) + 1  # n*2 * 2*1
        y_test_hat = sigmoid(a)
        val_loss = log_loss(y_test, y_test_hat)
        accuracy_val = accuracy_score(y_test, np.round(y_test_hat))
        print("Validation loss: {}".format(val_loss) + " Accuracy {}.".format(accuracy_val))
        print("--------")
        losses_train.append(train_loss)
        losses_val.append(val_loss)
        accs_train.append(accuracy_train)
        accs_val.append(accuracy_val)

# Plot the decision surface
min_x, min_y, max_x, max_y = np.min(X[:, 0]), np.min(X[:, 1]), np.max(X[:, 0]), np.max(X[:, 1])
feature_1, feature_2 = np.meshgrid(np.linspace(min_x, max_x), np.linspace(min_y, max_y))
grid = np.vstack([feature_1.ravel(), feature_2.ravel()]).T

a = np.matmul(grid, [w1, w2]) + 1  # n*2 * 2*1
y_test_hat = np.reshape(sigmoid(a), feature_1.shape)
display = DecisionBoundaryDisplay(xx0=feature_1, xx1=feature_2, response=y_test_hat)
display.plot()

display.ax_.scatter(X[:, 0], X[:, 1], c=y, edgecolor="black")
plt.show()
