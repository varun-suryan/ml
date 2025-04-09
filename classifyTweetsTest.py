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
from tensorflow.keras.layers import TextVectorization, Embedding

embedding = Embedding(input_dim=3, output_dim=2)
print(embedding([[1, 0, 2]]))

model = keras.Sequential([Embedding(input_dim=5, output_dim=2, name="my_embedding")])
model.compile('rmsprop')
print(model.get_layer("my_embedding").get_weights().shape)
print(model.get_layer("my_embedding").get_weights())
