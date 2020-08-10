import keras
from keras.models import load_model
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import tensorflow as tf2
import tensorflow.compat.v1 as tf1
import numpy as np

import foolbox as fb

# Load dataset
dataset = pd.read_pickle("./saved_dataframe.pkl")

X = dataset.drop(['class'], axis=1)
y = dataset['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
X_train = X_train.to_numpy()
X_test = X_test.to_numpy()
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

print(X_test.shape)
print(y_test.shape)
# y_train = y_train.reshape((-1, 1))
# y_test = y_test.reshape((-1, 1))

# pre-processing
scaler1 = preprocessing.StandardScaler().fit(X_train)
X_train = scaler1.transform(X_train)

scaler2 = preprocessing.StandardScaler().fit(X_test)
X_test = scaler2.transform(X_test)

# input_shape = X_train.shape[1]

print(type(X_test))
print(type(y_test))

# model_name = "bayesOpt_nn_model_1"
#
# # Load the model
# loaded_model = load_model("./pickles/" + model_name + ".h5")

