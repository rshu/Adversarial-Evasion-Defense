import logging
from datetime import timedelta
import os, sys, sklearn
import numpy as np
import pandas as pd
from read_data import read_NSL_KDD_data, read_CIC_IDS_2017_data, read_contagio_data, read_CSE_CIC_IDS_2018_data, \
    read_CICAndMal_data


def run(choose_dataset):
    if choose_dataset == "ContagioPDF":
        X_train_scaled, X_test_scaled, y_train, y_test = read_contagio_data()
    elif choose_dataset == "CIC-IDS-2017":
        X_train_scaled, X_test_scaled, y_train, y_test = read_CIC_IDS_2017_data()
    elif choose_dataset == "CICAndMal2017":
        X_train_scaled, X_test_scaled, y_train, y_test = read_CICAndMal_data()
    elif choose_dataset == "CSE-CIC-IDS2018":
        X_train_scaled, X_test_scaled, y_train, y_test = read_CSE_CIC_IDS_2018_data()
    elif choose_dataset == "NSL-KDD":
        X_train_scaled, X_test_scaled, y_train, y_test = read_NSL_KDD_data()
    else:
        pass

    return X_train_scaled, X_test_scaled, y_train, y_test


if __name__ == "__main__":
    print(pd.__version__)
    print(np.__version__)
    print(sys.version)
    print(sklearn.__version__)
    # ["ContagioPDF", "CIC-IDS-2017", "CICAndMal2017", "CSE-CIC-IDS2018", "NSL-KDD"]
    choose_dataset = "NSL-KDD"
    X_train_scaled, X_test_scaled, y_train, y_test = run(choose_dataset)
