import pandas as pd
from keras.layers import Input, Dense, Dropout, Activation
import keras
from os import path
from sklearn.model_selection import train_test_split
from attack_backup_2 import load_dataset
import random
from sklearn import preprocessing
import numpy as np
from statistics import mode

desired_width = 320
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 50)

file_path = "../../data/ContagioPDF/ConsolidateData.csv"
random.seed(42)


def dnn_model(params, input_shape):
    # config = tf.compat.v1.ConfigProto(device_count={'GPU': 1, 'CPU': 8})
    # sess = tf.compat.v1.Session(config=config)
    # tf.compat.v1.keras.backend.set_session(sess)

    classifier = keras.Sequential()
    # dropout to avoid overfitting
    layers = [
        Dense(input_shape, input_shape=(input_shape,)),
        Activation(params['hidden_layer_activation']),
        Dropout(params['drop_out']),
        Dense(params['first_layer_dense']),
        Activation(params['hidden_layer_activation']),
        Dropout(params['drop_out']),
        Dense(params['second_layer_dense']),
        Activation(params['hidden_layer_activation']),
        Dropout(params['drop_out']),
        Dense(params['third_layer_dense']),
        Activation(params['hidden_layer_activation']),
        Dropout(params['drop_out']),
        Dense(1),
        Activation(params['output_layer_activation'])
    ]

    for layer in layers:
        classifier.add(layer)

    if params['optimizer'] == "Adadelta":
        optimizer = keras.optimizers.Adadelta(learning_rate=params['learning_rate'])
    elif params['optimizer'] == "Adagrad":
        optimizer = keras.optimizers.Adagrad(learning_rate=params['learning_rate'])
    elif params['optimizer'] == "Adam":
        optimizer = keras.optimizers.Adam(learning_rate=params['learning_rate'])
    elif params['optimizer'] == "Adamax":
        optimizer = keras.optimizers.Adamax(learning_rate=params['learning_rate'])
    elif params['optimizer'] == "NAdam":
        optimizer = keras.optimizers.Nadam(learning_rate=params['learning_rate'])
    elif params['optimizer'] == "RMSprop":
        optimizer = keras.optimizers.RMSprop(learning_rate=params['learning_rate'])
    elif params['optimizer'] == "SGD":
        optimizer = keras.optimizers.SGD(learning_rate=params['learning_rate'])

    classifier.compile(optimizer=optimizer,
                       loss='binary_crossentropy',
                       metrics=['accuracy'])

    return classifier


if __name__ == "__main__":
    N = 20

    if path.exists("../../CICAndMal2017/saved_dataframe.pkl"):
        dataset = pd.read_pickle("../../CSE-CIC-IDS2018/saved_dataframe.pkl")
    else:
        dataset = load_dataset(file_path)

    X = dataset.drop(['class'], axis=1)
    y = dataset['class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()

    y_train = y_train.reshape((-1, 1))
    y_test = y_test.reshape((-1, 1))

    # pre-processing
    scaler1 = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler1.transform(X_train)

    scaler2 = preprocessing.StandardScaler().fit(X_test)
    X_test = scaler2.transform(X_test)

    input_shape = X_train.shape[1]

    df = pd.read_csv('../../CIC-IDS-2017/distance_result.csv')
    df.drop(df[df['distance'] < 0.5].index, inplace=True)
    df_sample = df.sample(n=N, replace=True, random_state=1)
    print(df_sample)

    prediction_result = []

    for i in range(df_sample.shape[0]):
        params_dict = eval(df_sample.iloc[i]['params'])
        print(params_dict)
        classifier = dnn_model(params_dict, input_shape)
        classifier.fit(X_train, y_train, batch_size=int(params_dict['batch_size']),
                       epochs=int(params_dict['num_epochs']))
        print(classifier.evaluate(X_test, y_test, verbose=0))
        prediction = classifier.predict(X_test).ravel().tolist()
        prediction = [1 if x > 0.5 else 0 for x in prediction]
        # print(prediction)
        prediction_result.append(prediction)

    print(prediction_result)

    final_list = list(zip(*prediction_result))
    print(final_list)

    agg_precition = []
    for i in range(len(final_list)):
        t = final_list[i]
        m = mode(t)
        agg_precition.append(m)

    print("Ensemble prediction:")
    print(agg_precition)

    print("")
    print("Test:")
    print(y_test.ravel().tolist())
    y_test_list = y_test.ravel().tolist()

    count = 0
    for i in range(len(agg_precition)):
        if agg_precition[i] == y_test_list[i]:
            count += 1

    print("Ensemble Accuracy: ", count / len(agg_precition))
