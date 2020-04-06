import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def read(dataset):
    if dataset is "moodle":
        data = pd.read_csv('../data/SVP/moodle-2_0_0-metrics.csv')
    elif dataset is "drupal":
        data = pd.read_csv('../data/SVP/drupal-6_0-metrics.csv')
    elif dataset is "phpmyadmin":
        data = pd.read_csv('../data/SVP/phpmyadmin-3_3_0-metrics.csv')

    y = data.IsVulnerable
    X = data.drop('IsVulnerable', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)
    train_features = np.array(X_train)
    test_features = np.array(X_test)
    train_label = y_train[:]
    test_label = y_test[:]
    return train_features, test_features, train_label, test_label
