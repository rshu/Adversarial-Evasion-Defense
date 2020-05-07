import tensorflow as tf
import keras
from os import path
import pickle, sys
from keras.layers import Dense, Dropout, Activation
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import numpy as np
import random
import pandas as pd
import csv
from find_two_models import dnn_model

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

    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return classifier


def create_model2(input_shape):
    # config = tf.compat.v1.ConfigProto(device_count={'GPU': 1, 'CPU': 8})
    # # sess = tf.compat.v1.Session(config=config)
    # # tf.compat.v1.keras.backend.set_session(sess)

    classifier = keras.Sequential()
    # dropout to avoid overfitting
    layers = [
        Dense(X_train.shape[1], input_shape=(input_shape,)),
        Activation('elu'),
        Dropout(0.1),
        Dense(127),
        Activation('relu'),
        Dropout(0.1),
        Dense(35),
        Activation('relu'),
        Dropout(0.1),
        Dense(12),
        Activation('relu'),
        Dropout(0.1),
        Dense(1),
        Activation('sigmoid')
    ]

    for layer in layers:
        classifier.add(layer)

    classifier.compile(optimizer='Adadelta', loss='binary_crossentropy', metrics=['accuracy'])
    return classifier


@tf.function
def adversarial_pattern(X_train_each, y_train_each, input_model):
    X_train_each = tf.cast(X_train_each, tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(X_train_each)
        prediction = input_model(X_train_each)
        loss = tf.keras.losses.binary_crossentropy(y_train_each, prediction)

    gradient = tape.gradient(loss, X_train_each)
    signed_grad = tf.sign(gradient)
    return signed_grad


def fgsm_attack(model, X_input, y_input, epsilon=0.2, clip_min=0.0, clip_max=1.0):
    """Create FGSM attack points for each row of x"""

    # determine how many dataset to be perturbed
    batch_size = X_input.shape[0]

    for i in range(batch_size):
        # print("batch: ", i)
        if batch_size > 100 and i % 100 == 0:
            print("batch: ", i, " ", i / batch_size)

        # N = random.randint(0, X_train.shape[0]-1)
        X_input_each = X_input[i].reshape((1, X_input.shape[1]))
        y_input_each = y_input[i].reshape((1, 1))

        signed_grad = adversarial_pattern(X_input_each, y_input_each, model)
        X_input_each_adv = X_input_each + signed_grad * epsilon

        # clip function
        X_input_each_adv = tf.clip_by_value(X_input_each_adv, clip_min, clip_max)

        X_input[i] = X_input_each_adv

    return X_input, y_input


def bim_a_attack(model, X_input, y_input, epsilon=0.2, clip_min=0.0, clip_max=1.0, iterations=10):
    # iterate each line
    for i in range(X_input.shape[0]):
        X_input_each = X_input[i].reshape((1, X_input.shape[1]))
        y_input_each = y_input[i].reshape((1, 1))

        for k in range(iterations):
            # print("row :", i, " iteration: ", k)
            signed_grad = adversarial_pattern(X_input_each, y_input_each, model)

            # the problem is that X_train_each_adv might be the same every time
            X_input_each_adv = X_input_each + signed_grad * epsilon

            # clip function
            X_input_each_adv = tf.clip_by_value(X_input_each_adv, clip_min, clip_max)

            X_input_each_adv = X_input_each_adv.numpy()

            prediction = model.predict_classes(X_input_each_adv)
            # print("prediction: ", prediction, " actual: ", y_train_each)
            if not np.equal(prediction, y_input_each):
                break

        X_input[i] = X_input_each_adv

    return X_input, y_input


def bim_b_attack(model, X_input, y_input, epsilon=0.2, clip_min=0.0, clip_max=1.0, iterations=10):
    for i in range(X_input.shape[0]):
        X_input_each = X_input[i].reshape((1, X_input.shape[1]))
        y_input_each = y_input[i].reshape((1, 1))

        for k in range(iterations):
            # print("row :", i, " iteration: ", k)
            signed_grad = adversarial_pattern(X_input_each, y_input_each, model)

            # the problem is that X_train_each_adv might be the same every time
            X_input_each_adv = X_input_each + signed_grad * epsilon
            # clip function
            X_input_each_adv = tf.clip_by_value(X_input_each_adv, clip_min, clip_max)

            X_input_each_adv = X_input_each_adv.numpy()

            # prediction = model.predict_classes(X_train_each_adv)
            # print("prediction: ", prediction, " actual: ", y_train_each)

        X_input[i] = X_input_each_adv

    return X_input, y_input


def jsma_attack():
    pass


def deepfool_attack():
    pass


def c_and_w_attack():
    pass


if __name__ == "__main__":

    if path.exists("saved_dataframe.pkl"):
        dataset = pd.read_pickle("./saved_dataframe.pkl")
    else:
        dataset = load_dataset(file_path)

    # print(df)

    X = dataset.drop(['class'], axis=1)
    y = dataset['class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    # print("Data shapes", X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    # print("Data types", type(X_train), type(X_test), type(y_train), type(y_test))

    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()

    # print("Data shapes", X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    # print("Data types", type(X_train), type(X_test), type(y_train), type(y_test))

    # Pass -1 as the value, and NumPy will calculate this number for you.
    y_train = y_train.reshape((-1, 1))
    y_test = y_test.reshape((-1, 1))

    # pre-processing
    scaler1 = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler1.transform(X_train)

    scaler2 = preprocessing.StandardScaler().fit(X_test)
    X_test = scaler2.transform(X_test)

    print("Data shapes", X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    df = pd.read_csv('dl_trials_2.csv')
    # print(df.shape)

    # Sort with best scores on top and reset index for slicing
    df.sort_values('loss', ascending=True, inplace=True)
    df.reset_index(inplace=True, drop=True)
    # print(df.head())

    best_result = df.iloc[0]
    # print(best_result['params'])

    best_model = pd.read_pickle("./pickles/bayesOpt_nn_model_" + str(best_result['iteration']) + ".pkl")
    print("Base accuracy on best model:", best_model.evaluate(x=X_test, y=y_test, verbose=0))

    candidate_model = pd.read_pickle("./pickles/bayesOpt_nn_model_23.pkl")
    print("Base accuracy on candidate model:", candidate_model.evaluate(x=X_test, y=y_test, verbose=0))

    # make a deep copy of existing X_test and y_test
    X_test_fgsm = X_test.copy()
    y_test_fgsm = y_test.copy()

    X_test_bim_a = X_test.copy()
    y_test_bim_a = y_test.copy()

    X_test_bim_b = X_test.copy()
    y_test_bim_b = y_test.copy()

    # fgsm attack on testing dataset
    X_test_adv_fgsm, y_test_adv_fgsm = fgsm_attack(best_model, X_test_fgsm, y_test_fgsm, epsilon=0.2, clip_min=0.0,
                                                   clip_max=1.0)

    print("Base accuracy of fgsm on best model:", best_model.evaluate(x=X_test_adv_fgsm, y=y_test_adv_fgsm, verbose=0))
    print("Base accuracy of fgsm on candidate model:",
          candidate_model.evaluate(x=X_test_adv_fgsm, y=y_test_adv_fgsm, verbose=0))
    print("")

    # BIM A attack
    X_test_adv_bim_a, y_test_adv_bim_a = bim_a_attack(best_model, X_test_bim_a, y_test_bim_a, epsilon=0.2, clip_min=0.0,
                                                      clip_max=1.0, iterations=10)

    print("Base accuracy of BIM A on best model:",
          best_model.evaluate(x=X_test_adv_bim_a, y=y_test_adv_bim_a, verbose=0))
    print("Base accuracy of BIM A on candidate model:",
          candidate_model.evaluate(x=X_test_adv_bim_a, y=y_test_adv_bim_a, verbose=0))
    print("")

    # BIM B attack
    X_test_adv_bim_b, y_test_adv_bim_b = bim_b_attack(best_model, X_test_bim_b, y_test_bim_b, epsilon=0.2, clip_min=0.0,
                                                      clip_max=1.0, iterations=10)

    print("Base accuracy of BIM B on best model:",
          best_model.evaluate(x=X_test_adv_bim_b, y=y_test_adv_bim_b, verbose=0))
    print("Base accuracy of BIM B on candidate model:",
          candidate_model.evaluate(x=X_test_adv_bim_b, y=y_test_adv_bim_b, verbose=0))

    print(np.array_equal(X_test, X_test_adv_fgsm))
    print(np.array_equal(X_test_adv_fgsm, X_test_adv_bim_a))
    print(np.array_equal(X_test_adv_fgsm, X_test_adv_bim_b))
    print(np.array_equal(X_test_adv_bim_a, X_test_adv_bim_b))
