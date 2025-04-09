import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import random
from sklearn import model_selection
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import TextVectorization, Embedding, GlobalAveragePooling1D, Dropout, Dense
from tensorflow.keras import losses, optimizers

df_data = pd.read_csv('./FinalBalancedDataset.csv')

def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_' + string])
    plt.xlabel('Epochs')
    plt.ylabel(string)
    plt.legend([string, 'val_' + string])
    plt.show()

X, y, val_frac = df_data["tweet"], df_data["Toxicity"], 0.00005
X_train, X_test, y_train, y_test = model_selection.train_test_split(np.array(X), np.array(y), test_size=val_frac, random_state=42)

max_vocab_length, avg_tokens = 10000, 15
layer_vectorizer = TextVectorization(max_tokens=max_vocab_length, standardize="lower_and_strip_punctuation", output_sequence_length = avg_tokens)

layer_vectorizer.adapt(X_train)


embedding = Embedding(input_dim=max_vocab_length, # set input shape
                             output_dim=3, # set size of embedding vector
                             embeddings_initializer="uniform", # default, intialize randomly
                             name="embedding_1") 

model = keras.Sequential([
    layer_vectorizer,
    embedding, 
    Dropout(0.2),  
    GlobalAveragePooling1D(), 
    Dropout(0.2), 
    Dense(1, activation='sigmoid')
    ])


learning_rate = 0.001
model.compile(loss=losses.BinaryCrossentropy(), optimizer=optimizers.Adam(learning_rate), metrics=['accuracy'])
print(model.summary())
history_1 = model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))
# plot_graphs(history=history_1, string="accuracy")
# plot_graphs(history=history_1, string="loss")

print(model.predict(X_test))
print(y_test)