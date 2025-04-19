import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.inspection import DecisionBoundaryDisplay
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import TextVectorization, Embedding, Dense, Dropout
from tensorflow.keras import losses, optimizers, metrics
matplotlib.use('TkAgg')


def plot_graphs(_history, string):
    plt.plot(_history.history[string])
    plt.plot(_history.history['val_' + string])
    plt.grid()
    plt.xlabel('Epochs')
    plt.ylabel(string)
    plt.legend([string, 'val_' + string])
    plt.show()


epochs = 250
X, y = make_blobs(n_samples=500, centers=[[-1, -1], [1, 1], [-1, 1], [1, -1]], cluster_std=0.25, random_state=0)
y = ((y == 0) | (y == 1))
# X, y = make_circles(n_samples=500, noise=0.1, factor=0.3, random_state=0)
# X, y = make_moons(n_samples=500, noise=0.1, random_state=0)

plt.scatter(X[:, 0], X[:, 1], c=y)
plt.grid()
plt.show(block=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# y_train, y_test = y_train[:, np.newaxis], y_test[:, np.newaxis]

# X_train, y_train = np.array([[-1, -1], [1, 1]]), np.array([[0], [1]])
learning_rate = 1e-2
model = keras.Sequential([Dense(10, activation='sigmoid'),
                          Dropout(0.05),
                          Dense(10, activation='sigmoid'),
                          Dropout(0.05),
                          Dense(1, activation='sigmoid')])

model.compile(loss=losses.BinaryCrossentropy(),
              optimizer=optimizers.Adam(learning_rate),
              metrics=metrics.BinaryAccuracy())
# print(model.summary())
history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test))

plot_graphs(history, "loss")
plot_graphs(history, "binary_accuracy")

min_x, min_y, max_x, max_y = np.min(X[:, 0]), np.min(X[:, 1]), np.max(X[:, 0]), np.max(X[:, 1])
feature_1, feature_2 = np.meshgrid(np.linspace(min_x, max_x), np.linspace(min_y, max_y))
grid = np.vstack([feature_1.ravel(), feature_2.ravel()]).T

y_test_hat = np.reshape(model.predict(grid), feature_1.shape)
display = DecisionBoundaryDisplay(xx0=feature_1, xx1=feature_2, response=y_test_hat)
display.plot()

display.ax_.scatter(X[:, 0], X[:, 1], c=y, edgecolor="black")
plt.show()
