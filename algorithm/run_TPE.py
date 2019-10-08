import csv
import numpy as np
import pandas as pd
from hyperopt import STATUS_OK
from timeit import default_timer as timer
import lightgbm as lgb
from hyperopt import tpe
from hyperopt import Trials
from hyperopt import fmin
from pathlib import Path
import ast
from sklearn.metrics import roc_auc_score

import parameterGrid
from util import readData
from parameterGrid import get_tpe_parameter

MAX_EVALS = 10
N_FOLDS = 10
base_path = Path(__file__).parent
out_file = (base_path / "../results/gbm_trials.csv").resolve()


def tpe_objective(params, n_folds=N_FOLDS):
    """Objective function for Gradient Boosting Machine Hyperparameter Optimization"""

    # Keep track of evals
    global ITERATION
    ITERATION += 1

    # Retrieve the subsample if present otherwise set to 1.0
    subsample = params['boosting_type'].get('subsample', 1.0)

    # Extract the boosting type
    params['boosting_type'] = params['boosting_type']['boosting_type']
    params['subsample'] = subsample

    # Make sure parameters that need to be integers are integers
    for parameter_name in ['num_leaves', 'subsample_for_bin', 'min_child_samples']:
        params[parameter_name] = int(params[parameter_name])

    start = timer()

    # Perform n_folds cross validation
    # Use early stopping and evalute based on ROC AUC
    cv_results = lgb.cv(params, train_set, num_boost_round=10000, nfold=n_folds,
                        early_stopping_rounds=100, metrics='auc', seed=50)

    run_time = timer() - start

    best_score = np.max(cv_results['auc-mean'])
    loss = 1 - best_score

    # Boosting rounds that returned the highest cv score
    n_estimators = int(np.argmax(cv_results['auc-mean']) + 1)

    # Write to the csv file ('a' means append)
    of_connection = open(out_file, 'a')
    writer = csv.writer(of_connection)
    writer.writerow([loss, params, ITERATION, n_estimators, run_time])

    # Dictionary with information for evaluation
    return {'loss': loss, 'params': params, 'iteration': ITERATION,
            'estimators': n_estimators,
            'train_time': run_time, 'status': STATUS_OK}


# Tree Parzen Estimator (TPE) is a Bayesian Optimization
def run_TPE(dataset):
    # params = get_tpe_parameter()
    train_features, test_features, train_label, test_label = readData.prepareData(dataset)

    global train_set
    # Create a lgb dataset
    train_set = lgb.Dataset(train_features, label=train_label)

    # # optimization algorithm
    # tpe_algorithm = tpe.suggest

    # Keep track of results
    bayes_trials = Trials()

    of_connection = open(out_file, 'w')
    writer = csv.writer(of_connection)

    # Write the headers to the file
    writer.writerow(['loss', 'params', 'iteration', 'estimators', 'train_time'])
    of_connection.close()

    # Global variable
    global ITERATION
    ITERATION = 0

    # Run optimization
    # Each iteration, the algorithm chooses new hyperparameter values from the
    # surrogate function which is constructed based on the previous results
    # and evaluates these values in the objective function.
    # This continues for MAX_EVALS evaluations of the objective function
    # with the surrogate function continually updated with each new result.
    best = fmin(fn=tpe_objective, space=parameterGrid.tpe_grid, algo=tpe.suggest,
                max_evals=MAX_EVALS, trials=bayes_trials, rstate=np.random.RandomState(50))

    # # Sort the trials with lowest loss (highest AUC) first
    # bayes_trials_results = sorted(bayes_trials.results, key=lambda x: x['loss'])
    # print(bayes_trials_results)

    results = pd.read_csv(out_file)

    # Sort with best scores on top and reset index for slicing
    results.sort_values('loss', ascending=True, inplace=True)
    results.reset_index(inplace=True, drop=True)

    # Extract the ideal number of estimators and hyperparameters
    # use the number of estimators that returned the lowest loss
    # in cross validation with early stopping
    best_bayes_estimators = int(results.loc[0, 'estimators'])

    # Convert from a string to a dictionary
    best_bayes_params = ast.literal_eval(results.loc[0, 'params']).copy()

    # Re-create the best model and train on the training data
    best_bayes_model = lgb.LGBMClassifier(n_estimators=best_bayes_estimators, n_jobs=-1,
                                          objective='binary', random_state=50, **best_bayes_params)
    best_bayes_model.fit(train_features, train_label)

    # Evaluate on the testing data
    preds = best_bayes_model.predict_proba(test_features)[:, 1]
    print('The best model from Bayes optimization scores {:.5f} AUC ROC on the test set.'.format(
        roc_auc_score(test_label, preds)))
    print('This was achieved after {} search iterations'.format(results.loc[0, 'iteration']))


# Note:
# The optimal hyperparameters are those that do best in cross validation and not necessarily
# those that do best on the testing data. When we use cross validation, we hope that these
# results generalize to the testing data.

# Even using 10-fold cross-validation, the hyperparameter tuning overfits to the training data.
# The best score from cross-validation is significantly higher than that on the testing data.

# Random search may return better hyperparameters just by sheer luck (re-running the notebook
# can change the results). Bayesian optimization is not guaranteed to find better hyperparameters
# and can get stuck in a local minimum of the objective function.
