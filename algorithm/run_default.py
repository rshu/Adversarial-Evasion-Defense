import matplotlib.pyplot as plt
import seaborn as sns
from timeit import default_timer as timer
from sklearn.metrics import roc_auc_score

from util import models
from util import readData


def run_default(dataset):
    train_features, test_features, train_label, test_label = readData.prepareData(dataset)

    model = models.LGBMClassifier()

    start = timer()
    model.fit(train_features, train_label)
    train_time = timer() - start

    predictions = model.predict_proba(test_features)[:, 1]
    auc = roc_auc_score(test_label, predictions)

    print('The baseline score on the test set is {:.4f}.'.format(auc))
    print('The baseline training time is {:.4f} seconds'.format(train_time))
