import pandas as pd
# import tensorflow as tf

# AttributeError: 'Tensor' object has no attribute 'numpy'
# input_shape = input_shape.numpy()
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()
# tf.executing_eagerly()
import keras
from os import path
import pickle, sys
from keras.layers import Input, Dense, Dropout, Activation
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import numpy as np
import pprint as pprint
from keras import backend as K
import random

# tf.enable_eager_execution(
#     config=None,
#     device_policy=None,
#     execution_mode=None
# )

file_path = "../data/ContagioPDF/ConsolidateData.csv"


def load_dataset(file_path):
    df = pd.read_csv(file_path, encoding='utf8', low_memory=False)
    df = df.drop(['filename'], axis=1)
    df.columns = df.columns.str.lstrip()

    print(df.shape)  # (27205, 136)
    print(df.dtypes)
    print(df.head())
    print(df.describe())
    # print(list(df.Label.unique()))
    print(df['class'].value_counts())
    print(df['class'].isnull().sum())

    df = df.dropna()
    df = df.reset_index(drop=True)
    print(df.shape)  # (27205, 136)

    print(df.select_dtypes(exclude=['int', 'float']))

    df['class'] = df['class'].astype(str).map({'False': 0, 'True': 1})
    df['box_other_only'] = df['box_other_only'].astype(str).map({'False': 0, 'True': 1})
    df['pdfid_mismatch'] = df['pdfid_mismatch'].astype(str).map({'False': 0, 'True': 1})

    # print(df.select_dtypes(exclude=['int', 'float']))
    print(df['class'].value_counts())
    print(df['class'].dtypes)
    print(df['box_other_only'].dtypes)
    print(df['pdfid_mismatch'].dtypes)

    df = df.drop_duplicates()
    print(df.shape)  # (22525, 136)

    df.to_pickle("./saved_dataframe.pkl")
    return df


if path.exists("saved_dataframe.pkl"):
    df = pd.read_pickle("./saved_dataframe.pkl")
else:
    df = load_dataset(file_path)

# print(df)

X = df.drop(['Label'], axis=1)
print(X)
y = df['Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

print("Data shapes", X_train.shape, X_test.shape, y_train.shape, y_test.shape)
print("Data types", type(X_train), type(X_test), type(y_train), type(y_test))

X_train = X_train.to_numpy()
X_test = X_test.to_numpy()
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

print("Data shapes", X_train.shape, X_test.shape, y_train.shape, y_test.shape)
print("Data types", type(X_train), type(X_test), type(y_train), type(y_test))

# Pass -1 as the value, and NumPy will calculate this number for you.
y_train = y_train.reshape((-1, 1))
y_test = y_test.reshape((-1, 1))

# pre-processing
scaler1 = preprocessing.StandardScaler().fit(X_train)
X_train = scaler1.transform(X_train)

scaler2 = preprocessing.StandardScaler().fit(X_test)
X_test = scaler2.transform(X_test)

print("Data shapes", X_train.shape, X_test.shape, y_train.shape, y_test.shape)


def create_model(X_train):
    config = tf.compat.v1.ConfigProto(device_count={'GPU': 1, 'CPU': 8})
    sess = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(sess)

    classifier = keras.Sequential()
    # dropout to avoid overfitting
    layers = [
        Dense(X_train.shape[1], input_shape=(X_train.shape[1],)),
        Activation('relu'),
        # Dropout(0.5),
        Dense(64),
        Activation('relu'),
        # Dropout(0.5),
        Dense(32),
        Activation('relu'),
        # Dropout(0.5),
        Dense(16),
        Activation('relu'),
        # Dropout(0.5),
        Dense(1),
        Activation('sigmoid')
    ]

    for layer in layers:
        classifier.add(layer)

    classifier.compile(optimizer='adam',
                       loss='binary_crossentropy',
                       metrics=['accuracy'])

    return classifier


model = create_model(X_train)
print(model.summary())

model.fit(X_train, y_train, batch_size=32, epochs=2)
print("Base accuracy on regular images:", model.evaluate(x=X_test, y=y_test, verbose=0))

X_train_first = X_train[0]
y_train_first = y_train[0]


def adversarial_pattern(X_train_first, y_train_first):
    X_train_first = tf.cast(X_train_first, tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(X_train_first)
        prediction = model(X_train_first)
        loss = tf.keras.losses.binary_crossentropy(y_train_first, prediction)
    gradient = tape.gradient(loss, X_train_first)
    signed_grad = tf.sign(gradient)
    return signed_grad

# # test on one row
# print(type(X_train_first))
# print(X_train_first.shape)
#
# X_train_first = X_train_first.reshape((1, 135))
# y_train_first = y_train_first.reshape((1, 1))
#
# # perturbations = adversarial_pattern(X_train_first.reshape((1, 135)), y_train_first.reshape((1, 1)))
# perturbations = adversarial_pattern(X_train_first, y_train_first)
# sess = tf.compat.v1.keras.backend.get_session()
# perturbations_toarray = sess.run(perturbations)
# print(X_train_first.shape)
# print(perturbations_toarray.shape)
# adversarial = X_train_first + perturbations_toarray * 0.1
# print(adversarial)
# print(adversarial.shape)
#
# # # convert tensor to ndarray
# # sess = tf.compat.v1.keras.backend.get_session()
# # array = sess.run(adversarial)
# # print(array)
# # print(type(array))
# # print(array.shape)
# sys.exit(-1)


def generate_adversarials(batch_size):
    while True:
        X = []
        y = []
        for batch in range(batch_size):
            if batch_size > 10000 and batch % 1000 == 0:
                print(batch / batch_size)

            N = random.randint(0, 100)

            y_train_each = y_train[N]
            X_train_each = X_train[N]

            X_train_each = X_train_each.reshape((1, 71))
            y_train_each = y_train_each.reshape((1, 1))

            perturbations = adversarial_pattern(X_train_each, y_train_each)
            sess = tf.compat.v1.keras.backend.get_session()
            perturbations_toarray = sess.run(perturbations)

            epsilon = 0.4
            adversarial = X_train_each + perturbations_toarray * epsilon

            X.append(adversarial)
            y.append(y_train[N])

        X = np.asarray(X).reshape((batch_size, 71))
        y = np.asarray(y)

        yield X, y


# Generate adversarial data
# x_adversarial_train, y_adversarial_train = next(generate_adversarials(200))
x_adversarial_test, y_adversarial_test = next(generate_adversarials(500))

# Assess base model on adversarial data
print("Base accuracy on adversarial images:", model.evaluate(x=x_adversarial_test, y=y_adversarial_test, verbose=0))
