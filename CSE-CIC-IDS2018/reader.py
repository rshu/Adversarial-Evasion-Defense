import pandas as pd
from os import path
import random
import numpy as np
import sys, collections
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import pickle
import scipy.stats
import seaborn as sns
from correlation import plot_correlation, drop_lin_correlated
import tensorflow as tf
import keras
from keras.layers import Input, Dense, Dropout, Activation
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

desired_width = 320
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 50)


def load_dataset(file_path):
    # too big to read, even using chunks
    # (16233001, 80)
    # chunk = 1000000
    # chunks = pd.read_csv(file_path, chunksize=chunk, encoding='utf8', low_memory=False)
    # df = pd.concat(chunks)

    # sampling the csv reading
    # Count the lines
    # num_lines = sum(1 for l in open(file_path))
    num_lines = 16233001  # because I know

    # Sample size - in this case ~5%
    size = int(num_lines / 20)

    # The row indices to skip - make sure 0 is not included to keep the header!
    skip_idx = random.sample(range(1, num_lines), num_lines - size)

    df = pd.read_csv(file_path, header=0, skiprows=skip_idx, encoding='utf8', low_memory=False)
    df.columns = df.columns.str.lstrip()
    df.drop_duplicates(keep=False, inplace=True)

    print(df.shape)
    print(df.head())
    print(df['Label'].value_counts())
    print(df['Label'].isnull().sum())

    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    print(df.select_dtypes(exclude=['int', 'float', 'datetime']))

    header_list = list(df.columns.values)
    for col in header_list:
        if col != 'Timestamp' and col != 'Label':
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # second time check, only label left
    print(df.select_dtypes(exclude=['int', 'float', 'datetime']))

    # Drop the features which have only 1 unique value:
    nunique = df.apply(pd.Series.nunique)
    cols_to_drop = nunique[nunique == 1].index
    df.drop(cols_to_drop, axis=1, inplace=True)
    print(df.shape)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.dropna()
    df = df.reset_index(drop=True)
    print(df.shape)

    label_list = list(df.Label.unique())
    label_map = {}
    for label in label_list:
        if label == 'Benign':
            label_map[label] = 0
        else:
            label_map[label] = 1

    print(label_map)
    df['Label'] = df['Label'].map(label_map)

    # third time check, empty now
    print(df.select_dtypes(exclude=['int', 'float', 'datetime']))

    print(df.head())
    print(df['Label'].value_counts())

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
    # print(len(col_names))
    # print(len(set(col_names)))
    # print([item for item, count in collections.Counter(col_names).items() if count > 1])

    file_path = "../data/CSE-CIC-IDS2018/ConsolidateData.csv"

    if path.exists("saved_dataframe.pkl"):
        df = pd.read_pickle("./saved_dataframe.pkl")
    else:
        df = load_dataset(file_path)

    X = df.drop(['Label', 'Timestamp'], axis=1)
    y = df['Label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    scaler1 = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler1.transform(X_train)

    scaler5 = preprocessing.StandardScaler().fit(X_test)
    X_test = scaler5.transform(X_test)

    build_models(X_train, y_train, X_test, y_test)
