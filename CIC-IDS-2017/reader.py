import pandas as pd
import sys
import tensorflow as tf
import keras
from keras.layers import Input, Dense, Dropout, Activation
import pickle
from os import path
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import numpy as np

desired_width = 320
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 50)

label_map = {'BENIGN': 0, 'PortScan': 1, 'FTP-Patator': 1, 'SSH-Patator': 1, 'Bot': 1, 'Infiltration': 1,
             'Web Attack � Brute Force': 1, 'Web Attack � XSS': 1, 'Web Attack � Sql Injection': 1, 'DDoS': 1,
             'DoS slowloris': 1, 'DoS Slowhttptest': 1, 'DoS Hulk': 1, 'DoS GoldenEye': 1, 'Heartbleed': 1}


def load_dataset(file_path):
    df = pd.read_csv(file_path, encoding='utf8', low_memory=False)
    df.columns = df.columns.str.lstrip()

    print(df.shape)  # (2830743, 79)
    print(df.head())
    print(df.describe())
    print(df['Label'].value_counts())

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.dropna()
    df = df.reset_index(drop=True)
    print(df.shape)  # (2827876, 79)

    print(df.select_dtypes(exclude=['int', 'float']))
    # print(list(df.Label.unique()))
    df['Label'] = df['Label'].map(label_map)
    print(df['Label'].value_counts())

    # Drop the features which have only 1 unique value:
    nunique = df.apply(pd.Series.nunique)
    cols_to_drop = nunique[nunique == 1].index
    df.drop(cols_to_drop, axis=1, inplace=True)
    print(df.shape)  # (2827876, 71)

    # sample 20%
    df = df.sample(frac=.20, random_state=20)
    print(df.shape)

    df.to_pickle("./saved_dataframe.pkl")
    return df


def build_models(X_train, y_train, X_test, y_test):
    config = tf.compat.v1.ConfigProto(device_count={'GPU': 1, 'CPU': 8})
    sess = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(sess)

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

    classifier = keras.Sequential()
    for layer in layers:
        classifier.add(layer)

    classifier.compile(optimizer='adam',
                       loss='binary_crossentropy',
                       metrics=['accuracy'])
    classifier.fit(X_train, y_train, batch_size=32, epochs=10)

    # save the model
    nn_model_pickel_file = 'saved_nn_model.pkl'
    pickle.dump(classifier, open(nn_model_pickel_file, 'wb'))

    # load the model from disk
    # loaded_model = pickle.load(open(nn_model_pickel_file, 'rb'))

    print("evaluation...")
    print("")
    print(classifier.evaluate(X_test, y_test, verbose=2))


if __name__ == "__main__":
    file_path = "../data/CIC-IDS-2017/ConsolidateData.csv"

    labels = ['BENIGN', 'PortScan', 'FTP-Patator', 'SSH-Patator', 'Bot', 'Infiltration', 'Web Attack � Brute Force',
              'Web Attack � XSS', 'Web Attack � Sql Injection', 'DDoS', 'DoS slowloris', 'DoS Slowhttptest', 'DoS Hulk',
              'DoS GoldenEye', 'Heartbleed']
    label_map = {}

    for label in labels:
        if label == 'BENIGN':
            label_map[label] = 0
        else:
            label_map[label] = 1

    # print(label_map)

    if path.exists("saved_dataframe.pkl"):
        df = pd.read_pickle("./saved_dataframe.pkl")
    else:
        df = load_dataset(file_path)

    print(df.shape)
    print(df['Label'].value_counts())

    X = df.drop(['Label'], axis=1)
    y = df['Label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    print(y_train.value_counts())
    print(y_test.value_counts())

    scaler1 = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler1.transform(X_train)

    scaler5 = preprocessing.StandardScaler().fit(X_test)
    X_test = scaler5.transform(X_test)

    build_models(X_train, y_train, X_test, y_test)
