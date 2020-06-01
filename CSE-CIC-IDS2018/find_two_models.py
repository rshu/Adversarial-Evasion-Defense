import pandas as pd
import tensorflow as tf

# AttributeError: 'Tensor' object has no attribute 'numpy'
# input_shape = input_shape.numpy()
# import tensorflow.compat.v1 as tf

# tf.compat.v1.disable_eager_execution()
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
from art.attacks import FastGradientMethod
from art.classifiers import KerasClassifier

# tf.enable_eager_execution(
#     config=None,
#     device_policy=None,
#     execution_mode=None
# )

from hyperopt import hp
from hyperopt.pyll.stochastic import sample
from hyperopt import Trials
from hyperopt import tpe
import csv
from hyperopt import fmin
from timeit import default_timer as timer
from hyperopt import STATUS_OK
from attack import load_dataset, create_model
import pickle

file_path = "../data/ContagioPDF/ConsolidateData.csv"
out_file = 'dl_trials_1.csv'
MAX_EVALS = 500


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

    METRICS = [
        keras.metrics.TruePositives(name='tp'),
        keras.metrics.FalsePositives(name='fp'),
        keras.metrics.TrueNegatives(name='tn'),
        keras.metrics.FalseNegatives(name='fn'),
        keras.metrics.BinaryAccuracy(name='accuracy'),
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall'),
        keras.metrics.AUC(name='auc'),
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


def dnn_model2(params, input_shape):
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

    METRICS = [
        keras.metrics.TruePositives(name='tp'),
        keras.metrics.FalsePositives(name='fp'),
        keras.metrics.TrueNegatives(name='tn'),
        keras.metrics.FalseNegatives(name='fn'),
        keras.metrics.BinaryAccuracy(name='accuracy'),
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall'),
        keras.metrics.AUC(name='auc'),
    ]

    for layer in layers:
        classifier.add(layer)

    classifier.compile(optimizer=params['optimizer'],
                       loss='binary_crossentropy',
                       metrics=METRICS)

    return classifier


# deep learning optimization using bayesian optimization

def objective(params, ):
    # Keep track of evals
    global ITERATION

    ITERATION += 1

    # Make sure parameters that need to be integers are integers
    for parameter_name in ['first_layer_dense', 'second_layer_dense', 'third_layer_dense']:
        params[parameter_name] = int(params[parameter_name])

    params['drop_out'] = float(params['drop_out'])

    start = timer()
    classifier = dnn_model(params, input_shape)
    classifier.fit(X_train, y_train, batch_size=int(params['batch_size']), epochs=int(params['num_epochs']))
    acc_results = classifier.evaluate(x=X_test, y=y_test, verbose=0)[1]

    # save to pickles
    pickle_model_index = ITERATION
    pickle_model_file = "./pickles/bayesOpt_nn_model_" + str(pickle_model_index) + ".pkl"
    pickle.dump(classifier, open(pickle_model_file, 'wb'))

    run_time = timer() - start

    # Hyperopt works to minimize a function
    # Loss must be minimized
    loss = 1 - acc_results

    # Write to the csv file ('a' means append)
    of_connection = open(out_file, 'a')
    writer = csv.writer(of_connection)
    writer.writerow([loss, params, ITERATION, run_time])

    # Dictionary with information for evaluation
    return {'loss': loss, 'params': params, 'iteration': ITERATION,
            'run_time': run_time, 'status': STATUS_OK}


space = {
    # 'hidden_layer_activation': hp.choice('hidden_layer_activation',
    #                                      ['deserialize', 'elu', 'exponential', 'get', 'hard_sigmoid', 'linear', 'relu',
    #                                       'selu', 'serialize', 'sigmoid', 'softmax', 'softplus', 'softsign', 'tanh']),
    # 'output_layer_activation': hp.choice('output_layer_activation',
    #                                      ['deserialize', 'elu', 'exponential', 'get', 'hard_sigmoid', 'linear', 'relu',
    #                                       'selu', 'serialize', 'sigmoid', 'softmax', 'softplus', 'softsign', 'tanh']),
    'hidden_layer_activation': hp.choice('hidden_layer_activation',
                                         ['elu', 'relu', 'selu', 'sigmoid', 'softmax', 'tanh', 'hard_sigmoid',
                                          'softplus', 'softsign', 'linear', 'exponential']),
    'output_layer_activation': hp.choice('output_layer_activation',
                                         ['elu', 'relu', 'selu', 'sigmoid', 'softmax', 'tanh', 'hard_sigmoid',
                                          'softplus', 'softsign', 'linear', 'exponential']),
    'first_layer_dense': hp.quniform('first_layer_dense', 30, 150, 1),
    'second_layer_dense': hp.quniform('second_layer_dense', 30, 50, 1),
    'third_layer_dense': hp.quniform('third_layer_dense', 10, 32, 1),
    'drop_out': hp.uniform('drop_out', 0.0, 0.5),
    'optimizer': hp.choice('optimizer', ['Adadelta', 'Adagrad', 'Adam', 'Adamax', 'NAdam', 'RMSprop', 'SGD']),
    'batch_size': hp.choice('batch_size', [16, 32, 64, 128]),
    'num_epochs': hp.quniform('num_epochs', 5, 20, 1),
    'learning_rate': hp.choice('learning_rate', [0.001, 0.01, 0.1])
}

if __name__ == "__main__":

    if path.exists("saved_dataframe.pkl"):
        df = pd.read_pickle("./saved_dataframe.pkl")
    else:
        df = load_dataset(file_path)

    # print(df)

    X = df.drop(['class'], axis=1)
    y = df['class']

    # train: 0.6, val: 0.2, test: 0.2
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    # split into training and validation dataset
    # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, stratify=y, test_size=0.25, random_state=42)

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

    # using bayesian optimization
    sample_space = sample(space)
    print(sample_space)

    # optimization algorithm: Tree Parzen Estimator
    # the method for constructing the surrogate function and
    # choosing the next hyperparameters to evaluate.
    tpe_algorithm = tpe.suggest

    # Keep track of results
    bayes_trials = Trials()

    # File to save first results
    # Every time the objective function is called, it will write one line to this file.
    of_connection = open(out_file, 'w')
    writer = csv.writer(of_connection)

    # Write the headers to the file
    writer.writerow(['loss', 'params', 'iteration', 'run_time'])
    of_connection.close()

    # Global variable
    global ITERATION

    ITERATION = 0

    # Run optimization
    best = fmin(fn=objective, space=space, algo=tpe.suggest,
                max_evals=MAX_EVALS, trials=bayes_trials, rstate=np.random.RandomState(50))

    # Sort the trials with lowest loss (highest AUC) first
    bayes_trials_results = sorted(bayes_trials.results, key=lambda x: x['loss'])
    print(bayes_trials_results[:2])

    results = pd.read_csv('dl_trials_1.csv')

    # Sort with best scores on top and reset index for slicing
    results.sort_values('loss', ascending=True, inplace=True)
    results.reset_index(inplace=True, drop=True)
    print(results.head())

    # without optimization
    # model = create_model(input_shape)
    # print(model.summary())
    #
    # model.fit(X_train, y_train, batch_size=32, epochs=1)
    # print("Base accuracy on original dataset:", model.evaluate(x=X_test, y=y_test, verbose=0))
