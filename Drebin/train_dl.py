from Drebin import read
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import keras
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
import tensorflow as tf
from tensorflow.python.client import device_lib
from keras import backend as K

import sklearn.metrics as metrics
import matplotlib.pyplot as plt

print(device_lib.list_local_devices())

config = tf.compat.v1.ConfigProto(device_count={'GPU': 1, 'CPU': 8})
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)

print("Reading data ...")
x_all, y_all = read.read(load_data=False)
x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.3, random_state=42)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)


# model = keras.Sequential([
#     # keras.layers.Flatten(input_shape=(8,)),
#     keras.layers.Dense(8, activation=tf.nn.relu, input_shape=(8,)),
#     keras.layers.Dense(4, activation=tf.nn.relu),
#     keras.layers.Dense(1, activation=tf.nn.sigmoid),
# ])


def build_classifer(optimizer):
    classifier = keras.Sequential([
        keras.layers.Dense(8, activation=tf.nn.relu, input_shape=(8,)),
        keras.layers.Dense(4, activation=tf.nn.relu),
        keras.layers.Dense(1, activation=tf.nn.sigmoid),
    ])
    classifier.compile(optimizer=optimizer,
                       loss='binary_crossentropy',
                       metrics=['accuracy'])
    return classifier


classifier = KerasClassifier(build_fn=build_classifer)
parameters = {'batch_size': [32, 64],
              'epochs': [500, 100],
              'optimizer': ['adam', 'rmsprop']}

grid_search = GridSearchCV(estimator=classifier,
                           param_grid=parameters,
                           scoring='accuracy',
                           cv=10)
grid_search.fit(x_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_

print(best_accuracy)
# print("----------model----------")
# print(model)
#
# model.compile(optimizer='adam',
#               loss='binary_crossentropy',
#               metrics=['accuracy'])

# model.fit(x_train, y_train, epochs=50, batch_size=20)
# test_loss, test_acc = model.evaluate(x_test, y_test)
#
# print('Test accuracy:', test_acc)
# #
# # y_pred = model.predict(x_test)
# # print(y_pred[:10])
