import logging
from datetime import timedelta
import os, sys, sklearn
import numpy as np
import pandas as pd
from read_data import read_NSL_KDD_data, read_CIC_IDS_2017_data, read_contagio_data, read_CSE_CIC_IDS_2018_data, \
    read_CICAndMal_data
import tensorflow as tf
from model import baseline_model
from tensorflow.python.client import device_lib


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
    # find out which devices your operations and tensors are assigned to
    tf.debugging.set_log_device_placement(True)

    print(sys.version)
    print(pd.__version__)
    print(np.__version__)
    print(tf.__version__)
    print(sklearn.__version__)
    print(tf.config.list_physical_devices('GPU'))
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print(device_lib.list_local_devices())

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    # exit(-1)
    # ["ContagioPDF", "CIC-IDS-2017", "CICAndMal2017", "CSE-CIC-IDS2018", "NSL-KDD"]
    choose_dataset = "ContagioPDF"
    X_train_scaled, X_test_scaled, y_train, y_test = run(choose_dataset)
    baseline_model(X_train_scaled, X_test_scaled, y_train, y_test)
