import pandas as pd
import tensorflow as tf

# # AttributeError: 'Tensor' object has no attribute 'numpy'
# # input_shape = input_shape.numpy()
# import tensorflow.compat.v1 as tf
#
# tf.compat.v1.disable_eager_execution()
# # tf.executing_eagerly()
# from tensorflow.python.framework import ops

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
from art.attacks import FastGradientMethod
from art.classifiers import KerasClassifier

import gc

# tf.enable_eager_execution(
#     config=None,
#     device_policy=None,
#     execution_mode=None
# )

file_path = "../../data/ContagioPDF/ConsolidateData.csv"


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


def create_model(input_shape):
    # config = tf.compat.v1.ConfigProto(device_count={'GPU': 1, 'CPU': 8})
    # # sess = tf.compat.v1.Session(config=config)
    # # tf.compat.v1.keras.backend.set_session(sess)

    classifier = keras.Sequential()
    # dropout to avoid overfitting
    layers = [
        Dense(X_train.shape[1], input_shape=(input_shape,)),
        Activation('relu'),
        Dropout(0.2),
        Dense(64),
        Activation('relu'),
        Dropout(0.2),
        Dense(32),
        Activation('relu'),
        Dropout(0.2),
        Dense(16),
        Activation('relu'),
        Dropout(0.2),
        Dense(1),
        Activation('sigmoid')
    ]

    for layer in layers:
        classifier.add(layer)

    classifier.compile(optimizer='adam',
                       loss='binary_crossentropy',
                       metrics=['accuracy'])

    return classifier

@tf.function
def adversarial_pattern(X_train_each, y_train_each, model):
    X_train_each = tf.cast(X_train_each, tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(X_train_each)
        prediction = model(X_train_each)
        loss = tf.keras.losses.binary_crossentropy(y_train_each, prediction)

    gradient = tape.gradient(loss, X_train_each)
    signed_grad = tf.sign(gradient)
    return signed_grad

def fgsm_attack(model, X_train, y_train, percentage=0.1, epsilon=0.2, clip_min=0.0, clip_max=1.0):
    """Create FGSM attack points for each row of x"""

    # # determine how many dataset to be perturbed
    # batch_size = int(X_train.shape[0] * percentage)
    #
    # for batch in range(batch_size):
    #     print("batch: ", batch)
    #     # if batch_size > 100 and batch % 100 == 0:
    #     #     print("batch: ", batch, " ", batch / batch_size)
    #
    #     N = random.randint(0, X_train.shape[0])
    #     X_train_each = X_train[N].reshape((1, X_train.shape[1]))
    #     y_train_each = y_train[N].reshape((1, 1))
    #     # print(X_train_each.shape)
    #
    #     # # reshape
    #     # X_train_each = X_train_each.reshape((1, X_train.shape[1]))
    #     # y_train_each = y_train_each.reshape((1, 1))
    #
    #     signed_grad = adversarial_pattern(X_train_each, y_train_each, model)
    #     sess = tf.compat.v1.keras.backend.get_session()
    #     signed_grad_toarray = sess.run(signed_grad)
    #     X_train_each_adv = X_train_each + signed_grad_toarray * epsilon
    #     # X_train_each_adv = X_train_each_adv.clip(clip_min, clip_max)
    #
    #     # need optimization here TODO
    #     X_train[N] = X_train_each_adv
    #
    # yield X_train, y_train

    # determine how many dataset to be perturbed
    batch_size = int(X_train.shape[0] * percentage)
    # sess = tf.compat.v1.keras.backend.get_session()

    while True:
        X = []
        y = []
        for batch in range(batch_size):
            print("batch: ", batch)
            # if batch_size > 100 and batch % 100 == 0:
            #     print("batch: ", batch, " ", batch / batch_size)

            N = random.randint(0, X_train.shape[0]-1)

            X_train_each = X_train[N].reshape((1, 135))
            y_train_each = y_train[N].reshape((1, 1))

            # X_train_each = X_train_each.reshape((1, 135))
            # y_train_each = y_train_each.reshape((1, 1))

            # add more operation to sesstion that make code run slower
            perturbations = adversarial_pattern(X_train_each, y_train_each, model)
            # sess = tf.compat.v1.keras.backend.get_session()
            # # # tf.initialize_all_variables().run()
            # perturbations_toarray = sess.run(perturbations)
            # sess.close()# too slow
            # # tf.reset_default_graph()
            # # sess = tf.InteractiveSession()

            # perturbations_toarray = perturbations.eval(session=tf.compat.v1.keras.backend.get_session())

            adversarial = X_train_each + perturbations * epsilon
            # tf.compat.v1.keras.backend.clear_session()

            X.append(adversarial)
            y.append(y_train[N])

        X = np.asarray(X).reshape((batch_size, 135))
        y = np.asarray(y)

        return X, y


def bim_a_attack(model, X_train, y_train, epsilon=0.2, clip_min=0.0, clip_max=1.0, iterations=10):
    for i in X_train.shape[0]:
        X_train_each = X_train[i]
        y_train_each = y_train[i]
        X_train_each = X_train_each.reshape((1, X_train.shape[1]))
        y_train_each = y_train_each.reshape((1, 1))

        for k in range(iterations):
            perturbations = adversarial_pattern(X_train_each, y_train_each, model)
            sess = tf.compat.v1.keras.backend.get_session()
            perturbations_toarray = sess.run(perturbations)
            X_train_each_adv = X_train_each + perturbations_toarray * epsilon
            if model.predict(X_train_each_adv) != y_train_each:
                break

        X_train = np.delete(X_train, i, axis=0)
        X_train = np.vstack((X_train, X_train_each_adv))

    return X_train, y_train


def bim_b_attack(model, X_train, y_train, epsilon=0.2, clip_min=0.0, clip_max=1.0, iterations=10):
    for i in X_train.shape[0]:
        X_train_each = X_train[i]
        y_train_each = y_train[i]
        X_train_each = X_train_each.reshape((1, X_train.shape[1]))
        y_train_each = y_train_each.reshape((1, 1))

        for k in range(iterations):
            perturbations = adversarial_pattern(X_train_each, y_train_each, model)
            sess = tf.compat.v1.keras.backend.get_session()
            perturbations_toarray = sess.run(perturbations)
            X_train_each_adv = X_train_each + perturbations_toarray * epsilon

        X_train = np.delete(X_train, i, axis=0)
        X_train = np.vstack((X_train, X_train_each_adv))

    return X_train, y_train


def jsma_attack():
    pass


def deepfool_attack():
    pass


def c_and_w_attack():
    pass


if __name__ == "__main__":

    if path.exists("../../CICAndMal2017/saved_dataframe.pkl"):
        df = pd.read_pickle("../../CSE-CIC-IDS2018/saved_dataframe.pkl")
    else:
        df = load_dataset(file_path)

    # print(df)

    X = df.drop(['class'], axis=1)
    y = df['class']

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

    # creating model with original dataset
    input_shape = X_train.shape[1]
    model = create_model(input_shape)
    print(model.summary())

    # classifier = KerasClassifier(model=model, clip_values=(0.0, 2.0))
    # classifier.fit(X_train, y_train, nb_epochs=5, batch_size=32)

    model.fit(X_train, y_train, batch_size=32, epochs=1)
    print("Base accuracy on original dataset:", model.evaluate(x=X_test, y=y_test, verbose=0))

    X_train_adv, y_train_adv = fgsm_attack(model, X_train, y_train, percentage=1.0, epsilon=0.5, clip_min=0.0, clip_max=1.0)

    model2 = create_model(input_shape)
    model2.fit(X_train_adv, y_train_adv, batch_size=32, epochs=1)

    # Assess base model on adversarial data
    print("Base accuracy on adversarial dataset:", model2.evaluate(x=X_test, y=y_test, verbose=0))
    sys.exit(-1)

    # indicate how many of training data are perturbed
    X_adversarial_train, y_adversarial_train = next(fgsm_attack(18020))

# # test on one row
# X_train_first = X_train[0]
# y_train_first = y_train[0]
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


# epsilon = 0.1
# adv_crafter = FastGradientMethod(model, epsilon)
# X_train_adv = adv_crafter.generate(x=X_train)
#
# model2 = create_model(X_train_adv)
# model2.fit(X_train_adv, y_train)
#
# # Assess base model on adversarial data
# print("Base accuracy on adversarial dataset:", model.evaluate(x=X_test, y=y_test, verbose=0))
#
# sys.exit(-1)
# Generate adversarial data
X_adversarial_train, y_adversarial_train = next(generate_adversarials(18020))
# X_adversarial_test, y_adversarial_test = next(generate_adversarials(200))


model2 = create_model(X_adversarial_train)
model2.fit(X_adversarial_train, y_adversarial_train, batch_size=32, epochs=1)

# Assess base model on adversarial data
print("Base accuracy on adversarial dataset:", model.evaluate(x=X_test, y=y_test, verbose=0))

# # Assess base model on adversarial data
# print("Base accuracy on adversarial images:", model.evaluate(x=x_adversarial_test, y=y_adversarial_test, verbose=0))
