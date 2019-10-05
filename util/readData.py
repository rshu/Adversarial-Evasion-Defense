import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def prepareData(dataset):
    if dataset == 'moodle':
        data = pd.read_csv('./data/moodle-2_0_0-metrics.csv')
        y = data.IsVulnerable
        X = data.drop('IsVulnerable', axis=1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=30)
        train_features = np.array(X_train)
        test_features = np.array(X_test)
        train_label = y_train[:]
        test_label = y_test[:]
    return train_features, test_features, train_label, test_label
