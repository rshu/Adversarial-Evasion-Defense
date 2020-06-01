import pandas as pd
import keras
from os import path
from keras.layers import Dense, Dropout, Activation
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import numpy as np
from hyperopt import hp
from hyperopt.pyll.stochastic import sample
from hyperopt import Trials
from hyperopt import tpe
import csv
from hyperopt import fmin
from timeit import default_timer as timer
from hyperopt import STATUS_OK
from misc.attack import load_dataset
import pickle

file_path = "../../data/ContagioPDF/ConsolidateData.csv"
out_file = 'dl_trials_sample.csv'
MAX_EVALS = 500


def dnn_model(params, input_shape):
    classifier = keras.Sequential()

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
    pickle_model_file = "./sample_pickles/bayesOpt_nn_model_" + str(pickle_model_index) + ".pkl"
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

    if path.exists("../../CICAndMal2017/saved_dataframe.pkl"):
        df = pd.read_pickle("../../CSE-CIC-IDS2018/saved_dataframe.pkl")
    else:
        df = load_dataset(file_path)

    df_sample = df.sample(frac=0.2)
    df_sample.to_pickle("./saved_dataframe_sample.pkl")

    X = df_sample.drop(['class'], axis=1)
    y = df_sample['class']

    # train: 0.6, val: 0.2, test: 0.2
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    # split into training and validation dataset
    # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, stratify=y, test_size=0.25, random_state=42)

    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()

    # Pass -1 as the value, and NumPy will calculate this number for you.
    y_train = y_train.reshape((-1, 1))
    y_test = y_test.reshape((-1, 1))

    # pre-processing
    scaler1 = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler1.transform(X_train)
    scaler2 = preprocessing.StandardScaler().fit(X_test)
    X_test = scaler2.transform(X_test)

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

    results = pd.read_csv('dl_trials_sample.csv')

    # Sort with best scores on top and reset index for slicing
    results.sort_values('loss', ascending=True, inplace=True)
    results.reset_index(inplace=True, drop=True)
    print(results.head())
