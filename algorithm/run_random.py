import random
import numpy as np
import pandas as pd
import lightgbm as lgb
from timeit import default_timer as timer
from sklearn.metrics import roc_auc_score

from util import parameterGrid
from util import readData

MAX_EVALS = 10
N_FOLDS = 10
random.seed(50)


# Random search objective function. Takes in hyperparameters
# and returns a list of results to be saved.
# Perform n_folds cross validation
# Loss must be minimized
# Boosting rounds that returned the highest cv score
# Return list of results
def random_objective(params, train_set, iteration, n_folds=N_FOLDS):
    start = timer()
    cv_results = lgb.cv(params, train_set, num_boost_round=10000, nfold=n_folds,
                        early_stopping_rounds=100, metrics='auc', seed=50, verbose_eval=True)
    end = timer()
    best_score = np.max(cv_results['auc-mean'])
    loss = 1 - best_score
    n_estimators = int(np.argmax(cv_results['auc-mean']) + 1)
    return [loss, params, iteration, n_estimators, end - start]


def run_random(dataset):
    train_features, test_features, train_label, test_label = readData.prepareData(dataset)

    # Create a lgb dataset
    train_set = lgb.Dataset(train_features, label=train_label)

    # params = parameterGrid.get_parameters()
    # r = lgb.cv(params, train_set, num_boost_round=10000, nfold=10, metrics='auc',
    #            early_stopping_rounds=100, verbose_eval=False, seed=50)
    #
    # # Highest score
    # r_best = np.max(r['auc-mean'])
    #
    # # Standard deviation of best score
    # r_best_std = r['auc-stdv'][np.argmax(r['auc-mean'])]
    #
    # print('The maximium ROC AUC on the validation set was {:.5f} with std of {:.5f}.'.format(r_best, r_best_std))
    # print('The ideal number of iterations was {}.'.format(np.argmax(r['auc-mean']) + 1))
    #
    # Dataframe to hold cv results
    random_results = pd.DataFrame(columns=['loss', 'params', 'iteration', 'estimators', 'time'],
                                  index=list(range(MAX_EVALS)))

    # Iterate through the specified number of evaluations
    # Randomly sample parameters for gbm
    for i in range(MAX_EVALS):
        params = parameterGrid.get_parameters()

        # if params['boosting_type'] == 'goss':
        #     # Cannot subsample with goss
        #     params['subsample'] = 1.0
        # else:
        #     # Subsample supported for gdbt and dart
        #     params['subsample'] = random.sample(parameterGrid.subsample_dist, 1)[0]

        results_list = random_objective(params, train_set, i)

        # Add results to next row in dataframe
        random_results.loc[i, :] = results_list

    random_results.sort_values('loss', ascending=True, inplace=True)
    random_results.reset_index(inplace=True, drop=True)

    # Find the best parameters and number of estimators
    best_random_params = random_results.loc[0, 'params'].copy()
    best_random_estimators = int(random_results.loc[0, 'estimators'])
    best_random_model = lgb.LGBMClassifier(n_estimators=best_random_estimators, n_jobs=-1,
                                           objective='binary', **best_random_params, random_state=50)

    # Fit on the training data
    # Make test predictions
    best_random_model.fit(train_features, train_label)
    predictions = best_random_model.predict_proba(test_features)[:, 1]
    print('The best model from random search scores {:.4f} on the test data.'.format(
        roc_auc_score(test_label, predictions)))
    print('This was achieved using {} search iterations.'.format(random_results.loc[0, 'iteration']))
